"""Utilities for constructing embedding matrices for Part II."""

import numpy as np
import torch

from gensim.models import Word2Vec
from src.data.lm_dataset import load_vocabulary


SPECIAL_TOKENS = {"<pad>", "<unk>", "<bos>", "<eos>"}


def create_fallback_matrix(vocab_size, embedding_dim=100, seed=42):
    """Create reproducible fallback vectors shared by frozen models."""
    rng = np.random.default_rng(seed)

    matrix = rng.normal(
        loc=0.0,
        scale=0.01,
        size=(vocab_size, embedding_dim),
    ).astype(np.float32)

    matrix[0] = 0.0  # <pad>
    return matrix


def build_embedding_matrix(
    source,
    vocabulary,
    embedding_dim=100,
    seed=42,
    lowercase_fallback=False,
):
    """
    Map pretrained embeddings onto the existing project vocabulary.

    source: Gensim KeyedVectors or equivalent lookup object.
    """
    matrix = create_fallback_matrix(
        len(vocabulary), embedding_dim, seed
    )

    exact_count = 0
    lowercase_count = 0
    missing_count = 0

    for word, token_id in vocabulary.items():
        if word in SPECIAL_TOKENS:
            continue

        if word in source:
            vector = source[word]
            exact_count += 1
        elif lowercase_fallback and word.lower() in source:
            vector = source[word.lower()]
            lowercase_count += 1
        else:
            missing_count += 1
            continue

        vector = np.asarray(vector, dtype=np.float32)

        if vector.shape != (embedding_dim,):
            raise ValueError(
                f"Unexpected embedding dimension for {word}"
            )

        if not np.isfinite(vector).all():
            raise ValueError(
                f"Non-finite embedding values for {word}"
            )

        matrix[token_id] = vector

    ordinary_count = (
        exact_count + lowercase_count + missing_count
    )

    stats = {
        "ordinary_vocabulary_size": ordinary_count,
        "exact_matches": exact_count,
        "lowercase_matches": lowercase_count,
        "missing": missing_count,
        "coverage": (
            (exact_count + lowercase_count) / ordinary_count
        ),
    }

    return torch.from_numpy(matrix), stats


def load_skipgram_matrix(
    checkpoint_path="checkpoints/skipgram_word2vec.model",
):
    vocabulary = load_vocabulary()

    model = Word2Vec.load(checkpoint_path)

    return build_embedding_matrix(
        model.wv,
        vocabulary,
        embedding_dim=100,
        seed=42,
        lowercase_fallback=False,
    )


def load_glove_matrix():
    import gensim.downloader as api

    vocabulary = load_vocabulary()
    glove = api.load("glove-wiki-gigaword-100")

    return build_embedding_matrix(
        glove,
        vocabulary,
        embedding_dim=100,
        seed=42,
        lowercase_fallback=True,
    )
