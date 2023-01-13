import os
import sys
import re
import sqlite3
import json
import argparse
import logging
import functools
import operator
import pandas as pd
import numpy as np
from pyserini.search.lucene import LuceneSearcher
from rich.progress import Progress, TimeElapsedColumn
from afc.utils.io import save_to_jsonl
from pprint import pformat
from dataclasses import dataclass
from typing import Dict, List


# TODO: The part of the code handling similar documents sampling is ugly

@dataclass
class DatasetPart:
    """Class containing info about part of the dataset that should be generated."""

    n_examples: int = 1000  # Number of examples in the dataset part
    n_sentences: int = 10  # Number of sentences in each example
    n_random_sentences: int = 8  # Number of sentences mixed into the original document


@dataclass
class SimilarDocumentSamplingConfig:
    """Class containing configuration for the sampling of similar documents."""

    use_similar_sampling: bool = True  # Whether similar sampling should be used
    k_hits: int = 100  # Number of documents to be retrived
    drop_top_m: int = 0  # Number of top documents to be dropped
    use_top_n: int = 20  # Number of top documents to sample from


DEFAULT_CONFIG = {
    "seed": 123,
    "splits": {
        "train": [
            DatasetPart(5000, 10, 8),
            DatasetPart(3000, 10, 4),
            DatasetPart(2000, 10, 2),
            DatasetPart(10000, 10, 0),
        ],
        "validation": [
            DatasetPart(500, 10, 8),
            DatasetPart(250, 10, 4),
            DatasetPart(250, 10, 2),
            DatasetPart(1000, 10, 0),
        ],
        "test": [
            DatasetPart(500, 10, 8),
            DatasetPart(250, 10, 4),
            DatasetPart(250, 10, 2),
            DatasetPart(1000, 10, 0),
        ],
    },
}


DEFAULT_SIMILAR_SAMPLING_CONFIG = SimilarDocumentSamplingConfig(
    k_hits=100, drop_top_m=0, use_top_n=20
)

DEFAULT_SPLIT_ORDER = ["test", "validation", "train"]


class MultiSourceDatasetGenerator:
    def __init__(
        self,
        db_path: str,
        index_path: str = None,
        no_overlap: bool = True,
        similar_sampling_config: SimilarDocumentSamplingConfig = None,
    ):
        self.no_overlap = no_overlap
        self.similar_sampling_config = similar_sampling_config if similar_sampling_config is not None else DEFAULT_SIMILAR_SAMPLING_CONFIG

        if self.no_overlap:
            logging.info("Enforcing no overlap between documents in individual splits")

        self._prepare_index_searcher(index_path)
        self._load_and_prepare_db(db_path)

    def _load_and_prepare_db(self, db_path):
        logging.info("Reading data from the database...")
        with sqlite3.connect(db_path) as con:
            self.db = pd.read_sql_query("SELECT id, lines FROM documents", con)
        logging.info("Done!")

        self.db.set_index("id", drop=False, inplace=True)

        logging.info("Preprocessing records...")
        logging.info("\tmapping lines to sentences...")
        self.db["sentences"] = self.db["lines"].map(self._process_lines)
        self.db.drop(labels="lines", axis=1, inplace=True)
        logging.info("\tcounting sentences...")
        self.db["n_sentences"] = self.db["sentences"].map(len)
        logging.info("Done!")

        # NOTE: A lookup table is being generated every time a new split
        # generation starts, because the documents allowed to be sampled
        # have changed

    def _prepare_index_searcher(self, index_path):
        if index_path:
            self.searcher = LuceneSearcher(index_path)
            self.using_similarity_index = True
            logging.info("[SAMPLING] Sampling similar documents using similarity index")
        else:
            self.searcher = None
            self.using_similarity_index = False
            logging.info("[SAMPLING] Sampling random documents without regards to similarity")

    @staticmethod
    def _detokenize(txt):
        """Reverts the PTB tokenization. Taken from Honza's repository, *slightly* modified."""
        txt = txt.replace(" .", ".").replace(" ,", ",")
        txt = txt.replace(" :", ":").replace(" ;", ";").replace(" ?", "?")
        txt = txt.replace("`` ", '"').replace(" ''", '"').replace(" '", "'")
        txt = txt.replace("-LRB- ", "(").replace("-LRB-", "(")
        txt = txt.replace("-RRB- ", ")").replace("-RRB-", ")")
        txt = txt.replace("( ", "(").replace(" )", ")")
        txt = txt.replace("\t", " ")
        txt = " ".join(txt.split())
        txt = txt.strip()
        return txt

    @staticmethod
    def _process_lines(lines: str):
        res = re.split("\\n*\d+\\t", lines)
        res = list(filter(len, res))

        return res

    @staticmethod
    def _create_lookup_table(db):
        """Creates a lookup table for faster sampling of documents with required minimal number of sentences"""
        lookup_table = {0: np.array(range(len(db)))} | {
            val: np.where((db["n_sentences"] >= val).to_numpy())[0]
            for val in db["n_sentences"].unique()
        }

        return lookup_table

    def _sample_random_similar_documents(
        self,
        orig_docid: str,
        query: str,
        available_docs: pd.DataFrame,
        n_docs: int,
        random_state: np.random.RandomState,
        k_hits: int = 100,
        use_top_n: int = 10,
        drop_top_m: int = 0,
    ):
        assert (
            k_hits >= use_top_n
        ), "The number of queried documents must be greater than or equal the number of documents from which the sampling is performed"
        # To be used if similar docs should be sampled without replacement
        # assert use_top_n >= n_docs

        # Get viable hits
        # NOTE: It might be unnecessary to sort the hits since they should already be sorted.
        #       This is just a precaution which might end up slowing the generation process down
        #       even though it is not required.
        #       Filtering based on docid is probably also not necessary.
        hits = self.searcher.search(query, k=k_hits)
        hits = list(
            filter(
                lambda hit: hit.docid != orig_docid
                and hit.docid in available_docs.index,
                hits,
            )
        )

        hits = hits[drop_top_m:]
        if len(hits) <= use_top_n:
            logging.warning(
                "Requested sampling from top %i documents, but only %i document(s) are available!",
                use_top_n,
                len(hits),
            )

        hits_ids = [hit.docid for hit in hits[:use_top_n]]
        n_available = len(hits_ids)
        n_similar = min(n_available, n_docs)

        # TODO: Decide whether similar docs should be sampled with replacement
        similar_docs = available_docs.loc[available_docs.index.isin(hits_ids)].sample(
            n=n_similar, replace=True, random_state=random_state
        )

        # If there are not enough available similar documents in the top 100 query results,
        # return random available documents instead.
        n_random = max(0, n_docs - n_available)
        random_docs = available_docs.sample(
            n=n_random, replace=True, random_state=random_state
        )

        if n_random != 0:
            logging.warning(
                "Sampling random documents (%i/%i) because a similar document is not available!",
                n_random,
                n_docs,
            )

        # TODO: Take note of the similarity of individual documents

        return pd.concat((similar_docs, random_docs))

    def _sample_single_example(
        self,
        n_sentences: int,
        n_random_sentences: int,
        db: pd.DataFrame,
        lookup_table: dict,
        random_state: np.random.RandomState,
        keep_orig_order: bool = False,
    ):
        assert n_sentences >= n_random_sentences

        # How many sentences from a single source are necessary?
        n_orig_sentences = n_sentences - n_random_sentences

        # Using the lookup table for documents with at least the desired number of sentences
        orig_document = db.iloc[
            random_state.choice(lookup_table[n_orig_sentences])
        ].copy()

        if n_random_sentences != 0 and self.using_similarity_index:
            random_documents = self._sample_random_similar_documents(
                orig_docid=orig_document["id"],
                query=" ".join(orig_document["sentences"][:2]),
                available_docs=db,
                n_docs=n_random_sentences,
                k_hits=self.similar_sampling_config.k_hits,
                drop_top_m=self.similar_sampling_config.drop_top_m,
                use_top_n=self.similar_sampling_config.use_top_n,
                random_state=random_state,
            )
        else:
            random_documents = db.sample(
                n=n_random_sentences, replace=True, random_state=random_state
            )

        # Sample the sentences from the selected documents
        sentences = [random_state.choice(x) for x in random_documents["sentences"]]

        if keep_orig_order:
            sentences += list(orig_document["sentences"])[:n_orig_sentences]
        else:
            sentences += list(
                random_state.choice(
                    orig_document["sentences"], size=n_orig_sentences, replace=False
                )
            )

        ids = random_documents["id"].tolist() + [orig_document["id"]] * n_orig_sentences

        permutation = random_state.permutation(n_sentences)
        sentences = [sentences[idx] for idx in permutation]
        ids = [ids[idx] for idx in permutation]

        example = {
            "label": len(set(ids)) != 1,
            "ids": ids,
            "sentences": sentences,
        }

        return example

    def _generate_split(
        self,
        split_name: str,
        split_composition: List[DatasetPart],
        db: pd.DataFrame,
        random_state: np.random.RandomState,
    ):
        lookup_table = self._create_lookup_table(db)
        split_size = sum([part.n_examples for part in split_composition])
        split = []

        logging.info("Generating the %s split", split_name)

        with Progress(*Progress.get_default_columns(), TimeElapsedColumn()) as progress:
            task = progress.add_task(
                f"Generating the {split_name} split...", total=split_size
            )

            for part in split_composition:
                for _ in range(part.n_examples):
                    split.append(
                        self._sample_single_example(
                            n_sentences=part.n_sentences,
                            n_random_sentences=part.n_random_sentences,
                            db=db,
                            lookup_table=lookup_table,
                            random_state=random_state,
                        )
                    )

                    progress.update(task, advance=1)

        return split

    @staticmethod
    def _create_database_subsets(db: pd.DataFrame, parts: List[str]):
        belongs_to = np.random.choice(range(len(parts)), size=len(db))
        subsets = {
            part_name: db[belongs_to == idx] for idx, part_name in enumerate(parts)
        }

        return subsets

    def generate_dataset(self, composition: dict, split_order=None, seed=123):
        random_state = np.random.RandomState(seed)

        dataset = dict()
        db_to_use = self.db.copy()
        # split_order = split_order if split_order is not None else list(composition.keys())
        split_order = split_order if split_order is not None else DEFAULT_SPLIT_ORDER

        # for split_name, split_config in composition.items():
        for split_name in split_order:
            # Generate the split and store it in the dataset
            split_config = composition[split_name]
            split_data = self._generate_split(
                split_name, split_config, db_to_use, random_state
            )
            dataset[split_name] = split_data

            # Filter out used documents
            if self.no_overlap:
                used_docs_ids = set(
                    functools.reduce(
                        operator.iconcat, [x["ids"] for x in split_data], []
                    )
                )
                documents_mask = db_to_use["id"].isin(used_docs_ids)
                db_to_use = db_to_use.loc[~documents_mask]

        return dataset


def load_config(config_path: str):
    filetype = config_path.split(".")[-1]
    if filetype == "json":
        with open(config_path, "r") as f:
            config = json.load(f)
    else:
        return None, None

    # Determine what kind of document sampling should be used
    # and possibly how it should be configured
    if "similar_sampling_config" in config.keys():
        similar_sampling_config = SimilarDocumentSamplingConfig(
            use_similar_sampling=config.get("use_similar_sampling", True),
            k_hits=config["similar_sampling_config"]["k_hits"],
            drop_top_m=config["similar_sampling_config"]["drop_top_m"],
            use_top_n=config["similar_sampling_config"]["use_top_n"],
        )
    elif "use_similar_sampling" in config.keys():
        similar_sampling_config = SimilarDocumentSamplingConfig(
            use_similar_sampling=config["use_similar_sampling"],
        )
    else:
        similar_sampling_config = DEFAULT_SIMILAR_SAMPLING_CONFIG

    config = {"seed": config["seed"]} | {
        "splits": {
            split_name: [
                DatasetPart(
                    part["n_examples"], part["n_sentences"], part["n_random_sentences"]
                )
                for part in split_parts
            ]
            for split_name, split_parts in config["splits"].items()
        }
    }

    return config, similar_sampling_config


def setup_logging(args):
    logging.basicConfig(
        filename=os.path.join(args.output_dir, "generator.log"),
        encoding="utf-8",
        filemode="w",
        level=logging.INFO,
    )
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))


def main(args):

    os.makedirs(args.output_dir, exist_ok=True)
    setup_logging(args)

    logging.info("Current working directory: %s", os.getcwd())

    if args.config and os.path.exists(args.config):
        logging.info("Using the provided config: %s", args.config)
        config, similar_sampling_config = load_config(args.config)
    else:
        logging.warning(
            "Config file could not have been found or was not provided, using the default configuration instead."
        )
        config, similar_sampling_config = (
            DEFAULT_CONFIG,
            DEFAULT_SIMILAR_SAMPLING_CONFIG,
        )

    logging.info("Used configuration:")
    logging.info(pformat(config))

    if args.index_path:
        logging.info("Used similar sampling config:")
        logging.info(pformat(similar_sampling_config))

    seed = args.seed if args.seed is not None else config["seed"]

    dataset = MultiSourceDatasetGenerator(
        db_path=args.db_path,
        index_path=args.index_path
        if similar_sampling_config.use_similar_sampling
        else None,
        similar_sampling_config=similar_sampling_config,
    ).generate_dataset(config["splits"], seed=seed)

    for split_name, split_data in dataset.items():
        save_to_jsonl(split_data, os.path.join(args.output_dir, f"{split_name}.jsonl"))
        logging.info("The %s split has been saved!", split_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        help="Path to the configuration file containing the sampling settings",
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Seed to be used for the dataset generation",
        required=False,
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Path to where the generated dataset should be saved",
    )
    parser.add_argument(
        "--db-path",
        type=str,
        default="/mnt/data/factcheck/fever/data_sentence-en-latest/fever/fever.db",
        help="Path to the FEVER Wikipedia articles database",
    )
    parser.add_argument(
        "--index-path",
        type=str,
        default=None,
        required=False,
        help="Path to the folder containing the Anserini similarity index",
    )
    parser.add_argument(
        "--no-overlap",
        type=bool,
        default=True,
        required=False,
        help="Whether it should be enforced that there are no overlapping documents among the dataset splits",
    )

    args = parser.parse_args()

    main(args)
