"""Extract overlapping-window IMDb features using frozen Part I models."""

import argparse
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset

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


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        required=True,
        choices=["lstm", "transformer"],
    )

    parser.add_argument(
        "--split",
        required=True,
        choices=["train", "validation"],
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--stride",
        type=int,
        choices=[16, 32],
        default=32,
    )
    args = parser.parse_args()

    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be positive.")

    device = torch.device(
        "mps" if torch.backends.mps.is_available() else "cpu"
    )

    print("Device:", device)
    print("Model:", args.model)
    print("Split:", args.split)
    print("Window Size: 64")
    print("Stride:", args.stride)

    with np.load(DATA_DIR / "imdb_split_indices.npz") as saved:
        key = "train_idx" if args.split == "train" else "val_idx"
        selected_indices = saved[key].copy()

    if args.limit is not None:
        selected_indices = selected_indices[:args.limit]

    if len(selected_indices) == 0:
        raise ValueError("No reviews selected.")

    dataset = load_dataset("stanfordnlp/imdb")["train"]
    vocab = load_part1_vocab()

    if args.model == "lstm":
        model = load_frozen_lstm(device)
        expected_dim = 256
        baseline_name = "lstm_mean"
    else:
        model = load_frozen_transformer(device)
        expected_dim = 128
        baseline_name = "transformer"

    features = []
    labels = []

    for i, idx in enumerate(selected_indices, start=1):
        example = dataset[int(idx)]

        token_ids = encode_imdb_review(
            example["text"],
            vocab,
        )

        if not token_ids:
            raise ValueError(f"Empty review at index {idx}.")

        representation = extract_overlapping_review_batched(
            model=model,
            token_ids=token_ids,
            model_type=args.model,
            device=device,
            stride=args.stride,
        )

        feature = representation.squeeze(0).cpu().numpy()

        if feature.shape != (expected_dim,):
            raise ValueError(
                f"Unexpected feature shape: {feature.shape}"
            )

        if not np.isfinite(feature).all():
            raise ValueError(
                f"Non-finite feature at index {idx}."
            )

        features.append(feature)
        labels.append(int(example["label"]))

        if i % 1000 == 0:
            print(f"Processed {i:,}/{len(selected_indices):,}")

    X = np.stack(features).astype(np.float32)
    y = np.asarray(labels, dtype=np.int64)

    # Compare sample order and labels against original features.
    
    # Compare sample order and labels against original features.
    baseline_path = DATA_DIR / (
        f"imdb_{baseline_name}_{args.split}_features.npz"
    )

    with np.load(baseline_path) as baseline:
        assert np.array_equal(
            selected_indices,
            baseline["indices"][:len(selected_indices)],
        )
        assert np.array_equal(
            y,
            baseline["y"][:len(selected_indices)],
        )


    assert X.shape == (len(selected_indices), expected_dim)
    assert np.isfinite(X).all()

    print("\nFeatures:", X.shape)
    print("Labels:", y.shape)
    print("Negative:", int((y == 0).sum()))
    print("Positive:", int((y == 1).sum()))
    print("Baseline alignment: PASS")

    # A smoke test must not save features.
    if args.limit is not None:
        print("\nSmoke test completed. No file saved.")
        return

    output_path = DATA_DIR / (
    f"imdb_{args.model}_stride{args.stride}_{args.split}_features.npz"
)

    if output_path.exists():
        raise FileExistsError(
            f"Refusing to overwrite: {output_path}"
        )

    np.savez_compressed(
        output_path,
        X=X,
        y=y,
        indices=selected_indices,
    )

    print("\nSaved:", output_path)


if __name__ == "__main__":
    main()
