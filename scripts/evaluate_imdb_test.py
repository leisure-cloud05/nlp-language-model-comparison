"""Evaluate frozen language-model representations on the IMDb test set."""

import argparse
from pathlib import Path

import numpy as np
from datasets import load_dataset
import joblib
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from src.data.imdb_preprocess import (
    load_part1_vocab,
    encode_imdb_review,
)

import torch
import json
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
    extract_overlapping_review_batched,
)


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "processed"

# All saved Logistic Regression probe configurations.
MODEL_CONFIGS = {
    "lstm": {"dimension": 256, "model_type": "lstm"},
    "lstm_mean": {"dimension": 256, "model_type": "lstm"},
    "lstm_mean_weighted": {"dimension": 256, "model_type": "lstm"},
    "lstm_mean_max": {"dimension": 512, "model_type": "lstm"},
    "lstm_overlap": {"dimension": 256, "model_type": "lstm"},
    "lstm_stride16": {"dimension": 256, "model_type": "lstm"},

    "transformer": {"dimension": 128, "model_type": "transformer"},
    "transformer_mean_weighted": {
        "dimension": 128,
        "model_type": "transformer",
    },
    "transformer_mean_max": {
        "dimension": 256,
        "model_type": "transformer",
    },
    "transformer_overlap": {
        "dimension": 128,
        "model_type": "transformer",
    },
    "transformer_stride16": {
        "dimension": 128,
        "model_type": "transformer",
    },
}


def extract_test_representation(model, token_ids, model_name, device):
    """Use exactly the representation used to train each classifier."""

    if model_name in ("lstm_overlap", "transformer_overlap"):
        return extract_overlapping_review_batched(
            model=model,
            token_ids=token_ids,
            model_type="lstm" if model_name.startswith("lstm") else "transformer",
            device=device,
            stride=32,
        )

    if model_name in ("lstm_stride16", "transformer_stride16"):
        return extract_overlapping_review_batched(
            model=model,
            token_ids=token_ids,
            model_type="lstm" if model_name.startswith("lstm") else "transformer",
            device=device,
            stride=16,
        )

    if model_name in ("lstm_mean_max", "transformer_mean_max"):
        return extract_mean_max_review_batched(
            model=model,
            token_ids=token_ids,
            model_type="lstm" if model_name.startswith("lstm") else "transformer",
            device=device,
        )

    if model_name == "lstm_mean_weighted":
        return extract_lstm_mean_weighted_review_batched(
            model=model,
            token_ids=token_ids,
            device=device,
        )

    if model_name == "transformer_mean_weighted":
        return extract_transformer_mean_weighted_review_batched(
            model=model,
            token_ids=token_ids,
            device=device,
        )

    if model_name == "lstm_mean":
        return extract_lstm_mean_review_batched(
            model=model,
            token_ids=token_ids,
            device=device,
        )

    if model_name in ("lstm", "transformer"):
        return extract_review_representation_batched(
            model=model,
            token_ids=token_ids,
            model_type=model_name,
            device=device,
        )

    raise ValueError(f"Unknown model: {model_name}")


def evaluate_saved_probe(model_name, X, y):
    """Evaluate an already-trained probe without refitting."""

    checkpoint_path = (
        ROOT
        / "checkpoints"
        / "imdb_probes"
        / f"{model_name}_probe.joblib"
    )

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Missing saved classifier: {checkpoint_path}"
        )

    # Load only locally generated, trusted joblib files.
    classifier = joblib.load(checkpoint_path)

    predictions = classifier.predict(X)

    accuracy = accuracy_score(y, predictions)
    macro_f1 = f1_score(
        y,
        predictions,
        average="macro",
    )
    matrix = confusion_matrix(
        y,
        predictions,
        labels=[0, 1],
    )

    return accuracy, macro_f1, matrix, predictions


def load_test_feature_cache(cache_path, expected_dim, test_size):
    """Load and validate cached IMDb test features."""

    with np.load(cache_path, allow_pickle=False) as data:
        X = data["X"].copy()
        y = data["y"].copy()
        indices = data["indices"].copy()

    expected_indices = np.arange(test_size, dtype=np.int64)

    if X.shape != (test_size, expected_dim):
        raise ValueError(f"Incorrect cached feature shape: {X.shape}")

    if y.shape != (test_size,):
        raise ValueError(f"Incorrect cached label shape: {y.shape}")

    if not np.array_equal(indices, expected_indices):
        raise ValueError("Cached IMDb test indices are incorrect.")

    if not np.isfinite(X).all():
        raise ValueError("Cached features contain NaN or Inf.")

    if not np.isin(y, [0, 1]).all():
        raise ValueError("Cached labels are invalid.")

    return X, y, indices

def save_test_results(
    model_name,
    accuracy,
    macro_f1,
    matrix,
    predictions,
    labels,
    indices,
):
    """Save final IMDb test metrics and predictions."""
    results_dir = ROOT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = results_dir / f"imdb_{model_name}_test.json"
    predictions_path = (
        results_dir / f"imdb_{model_name}_test_predictions.npz"
    )

    # Never silently overwrite final test results.
    for path in (metrics_path, predictions_path):
        if path.exists():
            raise FileExistsError(
                f"Refusing to overwrite: {path}"
            )

    metrics = {
        "model": model_name,
        "dataset": "stanfordnlp/imdb",
        "split": "test",
        "num_samples": int(len(labels)),
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
        "confusion_matrix": matrix.tolist(),
        "label_order": [0, 1],
    }

    with metrics_path.open("x", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)

    np.savez_compressed(
        predictions_path,
        indices=np.asarray(indices, dtype=np.int64),
        y_true=np.asarray(labels, dtype=np.int64),
        y_pred=np.asarray(predictions, dtype=np.int64),
    )

    print("Saved metrics:", metrics_path)
    print("Saved predictions:", predictions_path)

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        required=True,
        choices=sorted(MODEL_CONFIGS),
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Number of IMDb test reviews for a smoke test.",
    )

    args = parser.parse_args()

    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be positive.")

    cache_path = DATA_DIR / (
        f"imdb_{args.model}_test_features.npz"
    )

    print("Test feature cache:", cache_path)


    # Reuse cached full-test features when available.
    if args.limit is None and cache_path.exists():
        print("Loading existing test feature cache:", cache_path)

        X, labels, selected_indices = load_test_feature_cache(
            cache_path=cache_path,
            expected_dim=config["dimension"],
            test_size=25000,
        )

        accuracy, macro_f1, matrix, predictions = evaluate_saved_probe(
            model_name=args.model,
            X=X,
            y=labels,
        )

        print("\n=== IMDb Test Evaluation (cached) ===")
        print("Model:", args.model)
        print(f"Accuracy: {accuracy:.6f}")
        print(f"Macro F1: {macro_f1:.6f}")
        print("Confusion Matrix:")
        print(matrix)
        print("Predictions:", predictions.shape)
        save_test_results(
            model_name=args.model,
            accuracy=accuracy,
            macro_f1=macro_f1,
            matrix=matrix,
            predictions=predictions,
            labels=labels,
            indices=selected_indices,
        )
        print("Full test evaluation completed.")
        return

    config = MODEL_CONFIGS[args.model]

    # IMDb official test set, not our train/validation split.
    test_data = load_dataset("stanfordnlp/imdb")["test"]

    selected_indices = np.arange(len(test_data), dtype=np.int64)

    if args.limit is not None:
        selected_indices = selected_indices[:args.limit]

    vocab = load_part1_vocab()

    print("Model:", args.model)
    print("LM type:", config["model_type"])
    print("Expected feature dimension:", config["dimension"])
    print("Official IMDb test size:", len(test_data))
    print("Selected samples:", len(selected_indices))

    # Load the frozen Part I checkpoint.
    device = torch.device(
        "mps" if torch.backends.mps.is_available() else "cpu"
    )

    if config["model_type"] == "lstm":
        model = load_frozen_lstm(device)
    else:
        model = load_frozen_transformer(device)

    model.eval()

    print("Device:", device)

    # Extract features and labels in the same IMDb test order.
    features = []
    labels = []

    for idx in selected_indices:
        example = test_data[int(idx)]

        token_ids = encode_imdb_review(
            example["text"],
            vocab,
        )

        if not token_ids:
            raise ValueError(
                f"Empty tokenized test review at index {idx}."
            )

        
        representation = extract_test_representation(
            model=model,
            token_ids=token_ids,
            model_name=args.model,
            device=device,
        )

        feature = representation.squeeze(0).cpu().numpy()

        if feature.shape != (config["dimension"],):
            raise ValueError(
                f"Unexpected feature shape: {feature.shape}"
            )

        if not np.isfinite(feature).all():
            raise ValueError(
                f"Non-finite feature at test index {idx}."
            )

        features.append(feature)
        labels.append(int(example["label"]))

    X = np.stack(features).astype(np.float32)

    assert X.shape == (
        len(selected_indices),
        config["dimension"],
    )

    print("Test features:", X.shape)
    print("Finite features:", bool(np.isfinite(X).all()))

    labels = np.asarray(labels, dtype=np.int64)

    if not np.isin(labels, [0, 1]).all():
        raise ValueError("Unexpected IMDb test labels.")

    print("Labels:", labels.shape)
    print("Negative:", int((labels == 0).sum()))
    print("Positive:", int((labels == 1).sum()))
    print("Test dataset and tokenizer: PASS")
    
    accuracy, macro_f1, matrix, predictions = evaluate_saved_probe(
        model_name=args.model,
        X=X,
        y=labels,
    )

    print("\n=== IMDb Test Evaluation ===")
    print("Model:", args.model)
    print(f"Accuracy: {accuracy:.6f}")
    print(f"Macro F1: {macro_f1:.6f}")
    print("Confusion Matrix:")
    print(matrix)
    print("Predictions:", predictions.shape)

    if args.limit is not None:
        print("\nSmoke test completed. No files saved.")
        return

        cache_path.parent.mkdir(parents=True, exist_ok=True)

    if cache_path.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing cache: {cache_path}"
        )

    np.savez_compressed(
        cache_path,
        X=X,
        y=labels,
        indices=selected_indices,
    )

    print("\nSaved test features:", cache_path)
    save_test_results(
        model_name=args.model,
        accuracy=accuracy,
        macro_f1=macro_f1,
        matrix=matrix,
        predictions=predictions,
        labels=labels,
        indices=selected_indices,
    )
    print("Full test evaluation completed.")

if __name__ == "__main__":
    main()