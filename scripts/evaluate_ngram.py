"""Final evaluation of the tuned interpolated trigram model."""

import json
import os
import pickle
import sys
import tempfile
import time
import pickle
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.lm_dataset import load_documents, load_vocabulary
from src.models.ngram import NGramLanguageModel


def measure_serialized_size(model):
    """
    Measure the model's serialized storage size.

    This creates a temporary file and removes it automatically.
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        path = Path(temp_dir) / "ngram_model.pkl"

        with path.open("wb") as file:
            pickle.dump(
                model,
                file,
                protocol=pickle.HIGHEST_PROTOCOL,
            )

        return os.path.getsize(path)


def main():
    print("Loading processed WikiText-2...")

    train_docs = load_documents("train")
    val_docs = load_documents("validation")
    test_docs = load_documents("test")
    vocab = load_vocabulary()

    # Hyperparameters selected on the validation set.
    best_params = {
        "alpha": 10.0,
        "tau2": 320.0,
        "tau3": 320.0,
    }

    model = NGramLanguageModel(
        vocab_size=len(vocab),
        pad_id=vocab["<pad>"],
        bos_id=vocab["<bos>"],
        **best_params,
    )

    
    checkpoint_path = ROOT / "checkpoints" / "ngram.pkl"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    if checkpoint_path.exists():
        print("\nLoading saved trigram model...")

        with checkpoint_path.open("rb") as file:
            model = pickle.load(file)

        # Ensure the loaded model uses our selected configuration.
        assert model.vocab_size == len(vocab)
        assert model.pad_id == vocab["<pad>"]
        assert model.bos_id == vocab["<bos>"]
        assert model.alpha == best_params["alpha"]
        assert model.tau2 == best_params["tau2"]
        assert model.tau3 == best_params["tau3"]

        previous_result_path = ROOT / "results" / "ngram_final.json"

        if previous_result_path.exists():
            with previous_result_path.open("r", encoding="utf-8") as file:
                previous_results = json.load(file)
            training_seconds = previous_results.get("training_seconds")
        else:
            training_seconds = None
        print("Checkpoint loaded successfully.")

    else:
        print("\nTraining final trigram model...")

        start = time.perf_counter()
        model.fit(train_docs)
        training_seconds = time.perf_counter() - start

        with checkpoint_path.open("wb") as file:
            pickle.dump(
                model,
                file,
                protocol=pickle.HIGHEST_PROTOCOL,
            )

        print(f"Counting time: {training_seconds:.2f} s")
        print(f"Model saved to: {checkpoint_path}")


    print("\nEvaluating validation set...")
    validation_metrics = model.evaluate(val_docs)

    print(
        f"Validation CE: "
        f"{validation_metrics['cross_entropy']:.4f}"
    )
    print(
        f"Validation PPL: "
        f"{validation_metrics['perplexity']:.2f}"
    )

    print("\nEvaluating test set...")

    start = time.perf_counter()
    test_metrics = model.evaluate(test_docs)
    test_seconds = time.perf_counter() - start

    throughput = (
        test_metrics["num_targets"] / test_seconds
    )

    print(
        f"Test CE: {test_metrics['cross_entropy']:.4f}"
    )
    print(
        f"Test PPL: {test_metrics['perplexity']:.2f}"
    )
    print(
        f"Test targets: {test_metrics['num_targets']}"
    )
    print(
        f"Test evaluation time: {test_seconds:.2f} s"
    )
    print(
        f"Evaluation throughput: {throughput:,.0f} tokens/s"
    )

    print("\nMeasuring serialized model size...")
    size_bytes = measure_serialized_size(model)
    size_mib = size_bytes / (1024 ** 2)

    print(f"Serialized model size: {size_mib:.2f} MiB")

    output = {
        "model": "interpolated_trigram",
        "dataset": "wikitext-2-raw-v1",
        "vocab_size": len(vocab),
        "context_size": 2,
        "hyperparameters": best_params,
        "training_seconds": training_seconds,
        "model_statistics": model.model_statistics(),
        "validation": validation_metrics,
        "test": test_metrics,
        "test_evaluation_seconds": test_seconds,
        "evaluation_tokens_per_second": throughput,
        "serialized_model_size_bytes": size_bytes,
        "serialized_model_size_mib": size_mib,
        "size_measurement": "pickle serialized size",
        "throughput_measurement": (
            "CPU evaluation of the test corpus; "
            "includes Python probability and NLL computation"
        ),
    }

    output_path = ROOT / "results" / "ngram_final.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(output, file, indent=2)

    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
