"""IMDb preprocessing for downstream representation analysis."""

import json
from pathlib import Path
from src.data.preprocess import tokenize
import html
import re


MAX_REVIEW_TOKENS = 768
CHUNK_SIZE = 64


def tokenize_imdb(text):
    """Clean IMDb HTML and reuse Part I tokenization."""
    text = re.sub(
        r"<br\s*/?>", " ", text, flags=re.IGNORECASE
    )
    text = re.sub(r"<[^>]+>", " ", text)
    text = html.unescape(text)

    return tokenize(text)

def split_into_chunks(token_ids, chunk_size=CHUNK_SIZE,
                      max_tokens=MAX_REVIEW_TOKENS):
    """Split a review into nonempty chunks without padding."""
    if chunk_size <= 0 or max_tokens <= 0:
        raise ValueError("chunk_size and max_tokens must be positive.")

    truncated = token_ids[:max_tokens]

    return [
        truncated[i:i + chunk_size]
        for i in range(0, len(truncated), chunk_size)
    ]


def load_part1_vocab():
    """Load the original WikiText-2 vocabulary without modification."""
    project_root = Path(__file__).resolve().parents[2]
    vocab_path = project_root / "data" / "processed" / "vocab.json"

    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab = json.load(f)

    assert vocab["<pad>"] == 0
    assert vocab["<unk>"] == 1
    assert vocab["<bos>"] == 2
    assert vocab["<eos>"] == 3

    return vocab


def encode_imdb_review(text, vocab):
    """Encode one IMDb review using the frozen Part I vocabulary."""
    tokens = tokenize_imdb(text)
    unk_id = vocab["<unk>"]

    return [vocab.get(token, unk_id) for token in tokens]


def split_into_overlapping_chunks(
    token_ids,
    chunk_size=64,
    stride=32,
    max_tokens=768,
):
    """Split a review into overlapping windows without padding."""
    if chunk_size <= 0 or max_tokens <= 0:
        raise ValueError("chunk_size and max_tokens must be positive.")

    if stride <= 0 or stride > chunk_size:
        raise ValueError("stride must be between 1 and chunk_size.")

    tokens = token_ids[:max_tokens]

    if not tokens:
        return []

    chunks = []

    for start in range(0, len(tokens), stride):
        chunk = tokens[start:start + chunk_size]

        if chunk:
            chunks.append(chunk)

        if start + chunk_size >= len(tokens):
            break

    return chunks
