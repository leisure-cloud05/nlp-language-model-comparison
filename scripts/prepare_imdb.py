"""Prepare reproducible IMDb train/validation indices."""

from pathlib import Path

import numpy as np
from datasets import load_dataset
from sklearn.model_selection import train_test_split


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "data" / "processed"

SEED = 42


def main():
    dataset = load_dataset("stanfordnlp/imdb")

    labels = np.asarray(dataset["train"]["label"])

    train_idx, val_idx = train_test_split(
        np.arange(len(labels)),
        test_size=0.2,
        random_state=SEED,
        stratify=labels,
    )

    assert len(train_idx) == 20_000
    assert len(val_idx) == 5_000
    assert len(set(train_idx) & set(val_idx)) == 0

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    output_path = OUTPUT_DIR / "imdb_split_indices.npz"

    np.savez_compressed(
        output_path,
        train_idx=train_idx,
        val_idx=val_idx,
    )

    print("Saved:", output_path)
    print("Train:", len(train_idx))
    print("Validation:", len(val_idx))
    print("Test: 25,000 (official IMDb test split)")


if __name__ == "__main__":
    main()
