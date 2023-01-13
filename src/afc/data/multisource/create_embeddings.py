import torch
import datasets
from transformers import AutoModel, AutoTokenizer
from sentence_transformers import SentenceTransformer
import argparse
import afc

dev = "cuda" if torch.cuda.is_available() else "cpu"


def sentence_embedding_hf(sentence, model, tokenizer, n_layers):
    tokenized_sentence = tokenizer(
        sentence,
        max_length=512,
        padding="max_length",
        return_tensors="pt",
        truncation=True,
    )
    tokenized_sentence = {k: v.to(dev) for k, v in tokenized_sentence.items()}

    with torch.no_grad():
        embedding = model(**tokenized_sentence)

    embedding = torch.stack(
        [embedding.hidden_states[-i] for i in range(1, n_layers + 1)]
    )
    embedding = embedding.mean(dim=0)
    embedding = embedding[:, 0]

    return embedding


def create_example_embedding_hf(example, model, tokenizer, n_layers):
    example["sentences_embedding"] = [
        sentence_embedding_hf(sentence, model, tokenizer, n_layers)
        for sentence in example["sentences"]
    ]

    return example


def create_embeddings_hf(dataset, model_name):
    model = AutoModel.from_pretrained(args.model_name, output_hidden_states=True)
    model = model.to(dev)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    return dataset.map(
        lambda x: create_example_embedding_hf(x, model, tokenizer, args.n_layers)
    )


def create_example_embedding_sbert(example, model):
    # NOTE: By default, SBERT models allow 256 (128) input tokens to be processed, when
    #       using the "all-MiniLM-L6-v2" ("all-MiniLM-L12-v2") model
    # TODO: The unqsueeze operation is here only to allow compatibility with HF
    #       embedded data. The HF embeddings need to be rewritten so that they
    #       are created more efficiently and stored correctly.
    example["sentences_embedding"] = model.encode(example["sentences"])[:, None, :]

    return example


def create_embeddings_sbert(dataset, model_name):
    model = SentenceTransformer(model_name)
    return dataset.map(lambda x: create_example_embedding_sbert(x, model))


def main(args):
    dataset = afc.data.multisource.load_multisource(args.dataset_path)

    if not args.use_sbert:
        embedded_dataset = create_embeddings_hf(dataset, args.model_name)
    else:
        embedded_dataset = create_embeddings_sbert(dataset, args.model_name)

    print("Saving the embedded dataset...", end=" ")
    embedded_dataset.save_to_disk(args.output_dir)
    print("done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-path", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--model-name", default="bert-base-cased")
    parser.add_argument("--use-sbert", action="store_true")
    parser.add_argument("--n-layers", type=int, default=2)

    args = parser.parse_args()

    main(args)
