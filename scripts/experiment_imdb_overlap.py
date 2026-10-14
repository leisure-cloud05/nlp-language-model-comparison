"""Compare non-overlapping and overlapping IMDb representations."""

import argparse
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.data.imdb_preprocess import (
    load_part1_vocab,
    encode_imdb_review,
)
from src.models.checkpoint_loader import (
    load_frozen_lstm,
    load_frozen_transformer,
)
from src.models.representation_extractor import (
    extract_overlapping_review_batched,
)


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "processed"

TRAIN_LIMIT = 2000
VAL_LIMIT = 500

CONFIGS = {
    "lstm": {
        "feature_name": "lstm_mean",
        "dimension": 256,
        "C": 0.01,
    },
    "transformer": {
        "feature_name": "transformer",
        "dimension": 128,
        "C": 10.0,
    },
}


def load_existing_features(feature_name, split, limit):
    """Load saved non-overlapping features as the baseline."""
    path = DATA_DIR / (
        f"imdb_{feature_name}_{split}_features.npz"
    )

    with np.load(path) as data:
        X = data["X"][:limit].copy()
        y = data["y"][:limit].copy()
        indices = data["indices"][:limit].copy()

    assert len(X) == len(y) == len(indices) == limit
    assert np.isfinite(X).all()

    return X, y, indices


def extract_overlap_features(
    model,
    model_type,
    device,
    imdb_train,
    vocab,
    indices,
    expected_dim,
):
    """Extract overlapping features in exactly the baseline order."""
    features = []
    labels = []

    for i, idx in enumerate(indices, start=1):
        example = imdb_train[int(idx)]
        token_ids = encode_imdb_review(
            example["text"], vocab
        )

        if not token_ids:
            raise ValueError(f"Empty review: {idx}")

        representation = extract_overlapping_review_batched(
            model=model,
            token_ids=token_ids,
            model_type=model_type,
            device=device,
            stride=32,
        )

        feature = representation.squeeze(0).cpu().numpy()

        assert feature.shape == (expected_dim,)
        assert np.isfinite(feature).all()

        features.append(feature)
        labels.append(int(example["label"]))

        if i % 500 == 0:
            print(f"Extracted {i:,}/{len(indices):,}")

    X = np.stack(features).astype(np.float32)
    y = np.asarray(labels, dtype=np.int64)

    return X, y


def evaluate(X_train, y_train, X_val, y_val, C):
    """Fit a fresh classifier; fit scaler only on training data."""
    classifier = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=C,
            max_iter=2000,
            random_state=42,
        ),
    )

    classifier.fit(X_train, y_train)
    predictions = classifier.predict(X_val)

    return (
        accuracy_score(y_val, predictions),
        f1_score(y_val, predictions, average="macro"),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        choices=list(CONFIGS),
        required=True,
    )
    args = parser.parse_args()

    config = CONFIGS[args.model]
    feature_name = config["feature_name"]
    expected_dim = config["dimension"]
    C = config["C"]

    # Reuse the original feature matrices.
    X_train_base, y_train, train_indices = (
        load_existing_features(
            feature_name, "train", TRAIN_LIMIT
        )
    )

    X_val_base, y_val, val_indices = (
        load_existing_features(
            feature_name, "validation", VAL_LIMIT
        )
    )

    # Check that the indices match the saved official split.
    with np.load(DATA_DIR / "imdb_split_indices.npz") as saved:
        assert np.array_equal(
            train_indices,
            saved["train_idx"][:TRAIN_LIMIT],
        )
        assert np.array_equal(
            val_indices,
            saved["val_idx"][:VAL_LIMIT],
        )

    device = torch.device(
        "mps" if torch.backends.mps.is_available() else "cpu"
    )

    print("Model:", args.model)
    print("Device:", device)
    print("Train:", TRAIN_LIMIT)
    print("Validation:", VAL_LIMIT)
    print("Classifier C:", C)

    imdb_train = load_dataset("stanfordnlp/imdb")["train"]
    vocab = load_part1_vocab()

    if args.model == "lstm":
        model = load_frozen_lstm(device)
    else:
        model = load_frozen_transformer(device)

    print("\nExtracting overlapping TRAIN features...")

    X_train_overlap, y_train_overlap = extract_overlap_features(
        model,
        args.model,
        device,
        imdb_train,
        vocab,
        train_indices,
        expected_dim,
    )

    print("\nExtracting overlapping VALIDATION features...")

    X_val_overlap, y_val_overlap = extract_overlap_features(
        model,
        args.model,
        device,
        imdb_train,
        vocab,
        val_indices,
        expected_dim,
    )

    # Both feature extraction methods must have identical labels.
    assert np.array_equal(y_train, y_train_overlap)
    assert np.array_equal(y_val, y_val_overlap)

    assert X_train_base.shape == X_train_overlap.shape
    assert X_val_base.shape == X_val_overlap.shape

    print("\nFeature and label alignment: PASS")

    base_acc, base_f1 = evaluate(
        X_train_base,
        y_train,
        X_val_base,
        y_val,
        C,
    )

    overlap_acc, overlap_f1 = evaluate(
        X_train_overlap,
        y_train_overlap,
        X_val_overlap,
        y_val_overlap,
        C,
    )

    print("\n=== Small-sample comparison ===")
    print(f"Model: {args.model}")
    print(f"Train: {TRAIN_LIMIT}, Validation: {VAL_LIMIT}")

    print(
        f"Non-overlap: Accuracy={base_acc:.4f} "
        f"Macro F1={base_f1:.4f}"
    )

    print(
        f"Overlap:     Accuracy={overlap_acc:.4f} "
        f"Macro F1={overlap_f1:.4f}"
    )

    print(
        f"Difference (Overlap - Non-overlap): "
        f"{(overlap_acc - base_acc) * 100:+.2f} percentage points"
    )

    print("\nExperiment completed. No files saved.")


if __name__ == "__main__":
    main()
