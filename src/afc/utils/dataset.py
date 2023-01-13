import torch
import torch_geometric as geom
from transformers import AutoTokenizer, PreTrainedTokenizer, PreTrainedModel
from typing import List


def sliding_window_words(text: str, max_seq_length: int = 32, stride: int = 24):
    """Simple sliding window over words in the text."""

    words = text.split()  # naive approach?
    start_idx = list(range(0, len(words), stride))
    end_idx = [min(len(words), x + max_seq_length) + 1 for x in start_idx]
    sequences = [" ".join(words[start:end]) for start, end in zip(start_idx, end_idx)]

    return sequences


def sliding_window_sentences(sentences: str, window_size: int = 2):
    """Simple sliding window over sentences. Fixed unit stride."""

    sequences = [
        " ".join(sentences[idx : (idx + window_size)]).strip()
        for idx in range(len(sentences) - window_size + 1)
    ]

    return sequences


def split_and_tokenize(
    text: str, max_length: int = 256, tokenizer: PreTrainedTokenizer = None
):
    """Splits the example into chunks and tokenizes them."""

    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    tokenized = tokenizer(
        text,
        max_length=max_length,
        truncation=True,
        return_tensors="pt",
        padding="max_length",
        return_overflowing_tokens=True,
    )

    del tokenized["overflow_to_sample_mapping"]

    return tokenized


def tokenize_split_encode(
    text: str,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    max_length: int = 256,
    n_last_layers: int = 4,
    return_chunks: bool = True,
    dev: str = "cuda",
):
    """Splits the document, possibly into chunks, tokenizes them, and encodes each chunk using a given model."""

    tokenized = tokenizer(
        text,
        max_length=max_length,
        truncation=True,
        return_tensors="pt",
        padding="max_length",
        return_overflowing_tokens=return_chunks,
    )

    if return_chunks:
        del tokenized["overflow_to_sample_mapping"]

    tokenized = {k: v.to(dev) for k, v in tokenized.items()}

    with torch.no_grad():
        encoding = model(**tokenized)

    # Stack the last n hidden states
    output = torch.stack(
        [encoding.hidden_states[-i] for i in range(1, n_last_layers + 1)]
    )

    # Take the mean of the hidden states
    output = output.mean(0)

    # The encoding of the chunk is given by the CLS token (might be BERT specific)
    output = output[:, 0]

    return output
