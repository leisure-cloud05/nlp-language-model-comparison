"""Train and validate an interpolated trigram language model."""

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.lm_dataset import load_documents, load_vocabulary
from src.models.ngram import NGramLanguageModel


def main():
    print("Loading processed WikiText-2...")
    train_docs = load_documents("train")
    val_docs = load_documents("validation")
    vocab = load_vocabulary()

    print(f"Vocabulary size: {len(vocab)}")
    print(f"Training documents: {len(train_docs)}")
    print(f"Validation documents: {len(val_docs)}")

    # Start with a small, predefined search.
    # All hyperparameters are selected using validation data.
    candidates = [
    {"alpha": a, "tau2": 320.0, "tau3": 320.0}
    for a in [5.0, 10.0, 20.0, 50.0, 100.0]
    ]

    model = NGramLanguageModel(
        vocab_size=len(vocab),
        pad_id=vocab["<pad>"],
        bos_id=vocab["<bos>"],
    )

    print("\nCounting training n-grams...")
    start = time.perf_counter()
    model.fit(train_docs)
    training_seconds = time.perf_counter() - start

    statistics = model.model_statistics()

    print(f"Counting time: {training_seconds:.2f} s")
    print("Model statistics:", statistics)

    results = []

    print("\nValidation hyperparameter search:")

    for params in candidates:
        # Counts do not depend on smoothing parameters.
        # No need to fit the model again.
        model.alpha = params["alpha"]
        model.tau2 = params["tau2"]
        model.tau3 = params["tau3"]

        start = time.perf_counter()
        metrics = model.evaluate(val_docs)
        evaluation_seconds = time.perf_counter() - start

        record = {
            **params,
            **metrics,
            "evaluation_seconds": evaluation_seconds,
        }

        results.append(record)

        print(
            f"alpha={params['alpha']}, "
            f"tau2={params['tau2']}, "
            f"tau3={params['tau3']} | "
            f"CE={metrics['cross_entropy']:.4f}, "
            f"PPL={metrics['perplexity']:.2f}, "
            f"time={evaluation_seconds:.2f}s"
        )

    best = min(results, key=lambda x: x["cross_entropy"])

    print("\nBest validation configuration:")
    print(json.dumps(best, indent=2))

    output = {
        "model": "interpolated_trigram",
        "dataset": "wikitext-2-raw-v1",
        "vocab_size": len(vocab),
        "context_size": 2,
        "training_seconds": training_seconds,
        "model_statistics": statistics,
        "validation_search": results,
        "best_validation": best,
    }

    output_path = ROOT / "results" / "ngram_validation.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
