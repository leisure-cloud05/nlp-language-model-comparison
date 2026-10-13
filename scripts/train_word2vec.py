
"""Train and save self-trained Skip-gram embeddings."""

import json
import time
from pathlib import Path

from src.embeddings.word2vec import train_skipgram
from src.data.lm_dataset import load_vocabulary


ROOT = Path(__file__).resolve().parents[1]


def main():
    checkpoint_dir = ROOT / "checkpoints"
    result_dir = ROOT / "results"

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)

    print("Training Skip-gram Word2Vec...")
    start = time.perf_counter()

    model = train_skipgram(
        vector_size=100,
        window=5,
        epochs=5,
        seed=42,
    )

    elapsed = time.perf_counter() - start

    # Save the complete Word2Vec model for reproducibility.
    checkpoint_path = checkpoint_dir / "skipgram_word2vec.model"
    model.save(str(checkpoint_path))

    vocabulary = load_vocabulary()

    special_tokens = {"<pad>", "<unk>", "<bos>", "<eos>"}
    ordinary_words = [
        word for word in vocabulary
        if word not in special_tokens
    ]

    covered_words = [
        word for word in ordinary_words
        if word in model.wv
    ]

    coverage = len(covered_words) / len(ordinary_words)

    results = {
        "model": "Skip-gram Word2Vec",
        "implementation": "gensim",
        "corpus": "WikiText-2 training split only",
        "vector_size": model.vector_size,
        "window": model.window,
        "epochs": model.epochs,
        "negative": model.negative,
        "min_count": model.min_count,
        "workers": model.workers,
        "seed": 42,
        "training_seconds": elapsed,
        "learned_vocabulary_size": len(model.wv),
        "project_ordinary_vocabulary_size": len(ordinary_words),
        "covered_words": len(covered_words),
        "coverage": coverage,
        "checkpoint": str(checkpoint_path.relative_to(ROOT)),
    }

    result_path = result_dir / "skipgram_training.json"
    result_path.write_text(
        json.dumps(results, indent=2),
        encoding="utf-8",
    )

    print(f"Training time: {elapsed:.2f} s")
    print(f"Learned vocabulary: {len(model.wv)}")
    print(f"Vocabulary coverage: {coverage:.2%}")
    print(f"Model saved: {checkpoint_path}")
    print(f"Results saved: {result_path}")


if __name__ == "__main__":
    main()
