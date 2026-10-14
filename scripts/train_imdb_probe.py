"""Train reproducible linear probes on frozen IMDb representations."""

import argparse
import json
from pathlib import Path

import joblib
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "processed"
RESULTS_DIR = ROOT / "results"
PROBE_DIR = ROOT / "checkpoints" / "imdb_probes"

MODEL_NAMES = [
    "lstm",
    "lstm_mean",
    "lstm_mean_weighted",
    "lstm_mean_max",
    "transformer",
    "transformer_mean_weighted",
    "transformer_mean_max",
    "lstm_overlap",
    "transformer_overlap",
    "lstm_stride16",
    "transformer_stride16",
    "lstm_stride32",
    "transformer_stride32",
]

C_VALUES = [0.01, 0.1, 1.0, 10.0, 100.0]
SEED = 42


def load_features(model_name, split):
    path = DATA_DIR / f"imdb_{model_name}_{split}_features.npz"

    with np.load(path) as data:
        X = data["X"].copy()
        y = data["y"].copy()
        indices = data["indices"].copy()

    return X, y, indices


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        required=True,
        choices=MODEL_NAMES,
    )
    args = parser.parse_args()

    X_train, y_train, train_idx = load_features(
        args.model, "train"
    )
    X_val, y_val, val_idx = load_features(
        args.model, "validation"
    )

    # Validate dataset integrity.
    assert X_train.shape[0] == 20_000
    assert X_val.shape[0] == 5_000
    assert X_train.shape[1] == X_val.shape[1]

    assert np.intersect1d(train_idx, val_idx).size == 0
    assert np.isfinite(X_train).all()
    assert np.isfinite(X_val).all()

    trials = []
    best_f1 = -1.0
    best_c = None
    best_pipeline = None
    best_accuracy = None

    for c in C_VALUES:
        pipeline = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=c,
                max_iter=2000,
                random_state=SEED,
            ),
        )

        # Scaler and classifier fit on Train only.
        pipeline.fit(X_train, y_train)

        predictions = pipeline.predict(X_val)

        accuracy = float(accuracy_score(y_val, predictions))
        macro_f1 = float(
            f1_score(y_val, predictions, average="macro")
        )

        trials.append({
            "C": c,
            "validation_accuracy": accuracy,
            "validation_macro_f1": macro_f1,
        })

        print(
            f"C={c:<7g} "
            f"Accuracy={accuracy:.6f} "
            f"Macro F1={macro_f1:.6f}"
        )

        # Exact F1 ties favor the smaller C.
        if macro_f1 > best_f1:
            best_f1 = macro_f1
            best_c = c
            best_accuracy = accuracy
            best_pipeline = pipeline

    result = {
        "model": args.model,
        "task": "imdb_binary_sentiment",
        "seed": SEED,
        "train_size": len(y_train),
        "validation_size": len(y_val),
        "feature_dim": int(X_train.shape[1]),
        "classifier": "StandardScaler + LogisticRegression",
        "selection_metric": "validation_macro_f1",
        "best_C": best_c,
        "best_validation_accuracy": best_accuracy,
        "best_validation_macro_f1": best_f1,
        "trials": trials,
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    PROBE_DIR.mkdir(parents=True, exist_ok=True)

    result_path = RESULTS_DIR / f"imdb_{args.model}_probe.json"
    probe_path = PROBE_DIR / f"{args.model}_probe.joblib"

    # Protect previous experiments from accidental overwriting.
    if result_path.exists() or probe_path.exists():
        raise FileExistsError(
            "Output already exists. Refusing to overwrite."
        )

    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    joblib.dump(best_pipeline, probe_path)

    print(f"\nBest C: {best_c:g}")
    print(f"Best Validation Accuracy: {best_accuracy:.6f}")
    print(f"Best Validation Macro F1: {best_f1:.6f}")
    print("Saved results:", result_path)
    print("Saved classifier:", probe_path)


if __name__ == "__main__":
    main()