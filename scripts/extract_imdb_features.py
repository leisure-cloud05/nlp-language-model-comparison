"""Extract frozen language-model features for IMDb sentiment analysis."""

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
    extract_review_representation_batched,
    extract_lstm_mean_review_batched,
    extract_lstm_mean_weighted_review_batched,
    extract_transformer_mean_weighted_review_batched,
    extract_mean_max_review_batched,
)

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "processed"


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        required=True,
        choices=[
            "lstm",
            "lstm_mean",
            "lstm_mean_weighted",
            "lstm_mean_max",
            "transformer",
            "transformer_mean_weighted",
            "transformer_mean_max",
        ],
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
        help="Number of reviews for a smoke test.",
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

    dataset = load_dataset("stanfordnlp/imdb")

    with np.load(DATA_DIR / "imdb_split_indices.npz") as saved:
        key = "train_idx" if args.split == "train" else "val_idx"
        selected_indices = saved[key].copy()

    if args.limit is not None:
        selected_indices = selected_indices[:args.limit]

    if len(selected_indices) == 0:
        raise ValueError("No reviews selected.")

    train_data = dataset["train"]
    vocab = load_part1_vocab()

    if args.model.startswith("lstm"):
        model = load_frozen_lstm(device)
        expected_dim = 512 if args.model == "lstm_mean_max" else 256
    else:
        model = load_frozen_transformer(device)
        expected_dim = 256 if args.model == "transformer_mean_max" else 128

    features = []
    labels = []

    for i, idx in enumerate(selected_indices, start=1):
        example = train_data[int(idx)]

        token_ids = encode_imdb_review(example["text"], vocab)

        if not token_ids:
            raise ValueError(f"Empty review at IMDb index {idx}.")

        
        
        if args.model in ("lstm_mean_max", "transformer_mean_max"):
            model_type = (
                "lstm" if args.model == "lstm_mean_max"
                else "transformer"
            )

            representation = extract_mean_max_review_batched(
                model=model,
                token_ids=token_ids,
                model_type=model_type,
                device=device,
            )

        elif args.model == "lstm_mean_weighted":
            representation = extract_lstm_mean_weighted_review_batched(
                model=model,
                token_ids=token_ids,
                device=device,
            )

        elif args.model == "transformer_mean_weighted":
            representation = extract_transformer_mean_weighted_review_batched(
                model=model,
                token_ids=token_ids,
                device=device,
            )

        elif args.model == "lstm_mean":
            representation = extract_lstm_mean_review_batched(
                model=model,
                token_ids=token_ids,
                device=device,
            )

        else:
            representation = extract_review_representation_batched(
                model=model,
                token_ids=token_ids,
                model_type=args.model,
                device=device,
            )


        feature = representation.squeeze(0).cpu().numpy()

        if feature.shape != (expected_dim,):
            raise ValueError(
                f"Unexpected feature shape: {feature.shape}"
            )

        if not np.isfinite(feature).all():
            raise ValueError(
                f"Non-finite feature at IMDb index {idx}."
            )

        features.append(feature)
        labels.append(int(example["label"]))

        if i % 1000 == 0:
            print(
                f"Processed {i:,}/{len(selected_indices):,}"
            )

    X = np.stack(features).astype(np.float32)
    y = np.asarray(labels, dtype=np.int64)

    assert X.shape == (len(selected_indices), expected_dim)
    assert y.shape == (len(selected_indices),)
    assert np.isfinite(X).all()

    print("\nFeatures:", X.shape)
    print("Labels:", y.shape)
    print("Negative:", int((y == 0).sum()))
    print("Positive:", int((y == 1).sum()))
    print("Finite features:", bool(np.isfinite(X).all()))

    # Smoke tests do not save generated files.
    if args.limit is not None:
        print("\nSmoke test completed. No file saved.")
        return

    output_path = DATA_DIR / (
        f"imdb_{args.model}_{args.split}_features.npz"
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
