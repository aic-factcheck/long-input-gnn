import torch
import torch.nn as nn
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer,
)
from datasets import load_metric
from argparse import ArgumentParser
import afc


def experiment_main(args):
    hotpot = afc.data.hotpotqa.load_hotpotqa()
    # hotpot["train"] = hotpot["train"].select(range(100))
    # hotpot["validation"] = hotpot["validation"].select(range(100))

    if args.load_checkpoint:
        model = AutoModelForSequenceClassification.from_pretrained(
            args.load_checkpoint, num_labels=2
        )
    else:
        model = AutoModelForSequenceClassification.from_pretrained(
            "bert-base-uncased", num_labels=2
        )

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    def preprocess_example(example, max_length=512):
        preprocessed_example = {
            "label": torch.tensor([example["answer"]]),
        }

        if args.oracle:
            documents_to_use = [
                example["context"]["sentences"][
                    example["context"]["title"].index(title)
                ][sentence_idx]
                for title, sentence_idx in zip(*example["supporting_facts"].values())
            ]
        elif args.supporting_first:
            # Provide the articles known to support the statement as first
            supporting_idxs = [
                example["context"]["title"].index(title)
                for title in set(example["supporting_facts"]["title"])
            ]
            other_idxs = filter(
                lambda x: x not in supporting_idxs,
                range(len(example["context"]["title"])),
            )

            documents_to_use = [
                example["context"]["sentences"][idx] for idx in supporting_idxs
            ]
            documents_to_use += [
                example["context"]["sentences"][idx] for idx in other_idxs
            ]
        else:
            # Otherwise just use the order in which they were provided
            documents_to_use = example["context"]["sentences"]

        documents = " ".join(["".join(document) for document in documents_to_use])
        # Compute the average length of the documents over all examples to see whether
        # the oracle setting works correctly

        if args.use_sep:
            preprocessed_example = preprocessed_example | tokenizer(
                example["question"],
                documents,
                max_length=max_length,
                truncation=True,
                padding="max_length",
            )
        else:
            preprocessed_example = preprocessed_example | tokenizer(
                example["question"] + " " + documents,
                max_length=max_length,
                truncation=True,
                padding="max_length",
            )

        return preprocessed_example

    tokenized_hotpot = hotpot.map(preprocess_example)
    tokenized_hotpot.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "token_type_ids", "label"],
    )

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    metric = load_metric("accuracy")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = logits.argmax(axis=1)
        return metric.compute(predictions=predictions, references=labels)

    training_args = TrainingArguments(
        output_dir="./results",
        learning_rate=2e-5,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        num_train_epochs=10,
        # weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        report_to="none",
        load_best_model_at_end=True,
        save_total_limit=1,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_hotpot["train"],
        eval_dataset=tokenized_hotpot["test"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    if not args.eval_only:
        trainer.train()

    ce = nn.CrossEntropyLoss()
    for split in ["train", "validation", "test"]:
        predictions = (
            torch.tensor(trainer.predict(tokenized_hotpot[split]).predictions)
            .argmax(dim=1)
            .view(-1, 1)
            .to(float)
        )
        labels = torch.tensor(tokenized_hotpot[split]["label"]).view(-1, 1).to(float)

        accuracy = (predictions == labels).float().mean()
        loss = ce(predictions, labels).item()
        print(f"{split} accuracy:", accuracy)
        print(f"{split} loss:", loss)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--load-checkpoint", default=None, type=str)
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--supporting-first", action="store_true")
    parser.add_argument("--oracle", action="store_true")
    parser.add_argument("--use-sep", action="store_true")

    args = parser.parse_args()

    args.supporting_first = False if args.oracle else args.supporting_first

    experiment_main(args)
