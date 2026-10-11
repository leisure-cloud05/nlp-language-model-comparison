"""Evaluate the best Vanilla RNN checkpoint on WikiText-2."""

import json
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.data.lm_dataset import (
    LanguageModelDataset,
    load_documents,
    load_vocabulary,
    make_collate_fn,
)
from src.models.rnn import RNNLanguageModel


def synchronize(device):
    if device.type == "mps":
        torch.mps.synchronize()


def main():
    device = torch.device(
        "mps" if torch.backends.mps.is_available() else "cpu"
    )

    print(f"Device: {device}")

    vocab = load_vocabulary()

    checkpoint_path = ROOT / "checkpoints" / "rnn_best.pt"

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}"
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )

    config = checkpoint["config"]

    model = RNNLanguageModel(
        vocab_size=checkpoint["vocab_size"],
        embedding_dim=config["embedding_dim"],
        hidden_dim=config["hidden_dim"],
        num_layers=config["num_layers"],
        pad_id=checkpoint["pad_id"],
    )

    assert checkpoint["vocab_size"] == len(vocab)
    assert checkpoint["pad_id"] == vocab["<pad>"]

    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    print(f"Best epoch: {checkpoint['best_epoch']}")
    print(f"Parameters: {model.num_parameters():,}")

    test_docs = load_documents("test")

    test_dataset = LanguageModelDataset(
        test_docs,
        seq_len=config["seq_len"],
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        collate_fn=make_collate_fn(vocab["<pad>"]),
        num_workers=0,
    )

    total_nll = 0.0
    total_targets = 0

    synchronize(device)
    start = time.perf_counter()

    with torch.inference_mode():
        for batch in test_loader:
            inputs = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            logits, _ = model(inputs)

            nll = F.cross_entropy(
                logits.reshape(-1, model.vocab_size),
                labels.reshape(-1),
                ignore_index=-100,
                reduction="sum",
            )

            total_nll += nll.item()
            total_targets += (labels != -100).sum().item()

    synchronize(device)
    evaluation_seconds = time.perf_counter() - start

    test_ce = total_nll / total_targets
    test_ppl = math.exp(test_ce)
    throughput = total_targets / evaluation_seconds

    expected_targets = sum(
        len(document) - 1 for document in test_docs
    )

    assert total_targets == expected_targets

    size_bytes = checkpoint_path.stat().st_size

    results = {
        "model": "vanilla_rnn",
        "best_epoch": checkpoint["best_epoch"],
        "config": config,
        "parameters": model.num_parameters(),
        "test": {
            "cross_entropy": test_ce,
            "perplexity": test_ppl,
            "num_targets": total_targets,
        },
        "evaluation_seconds": evaluation_seconds,
        "evaluation_tokens_per_second": throughput,
        "checkpoint_size_bytes": size_bytes,
        "checkpoint_size_mib": size_bytes / (1024 ** 2),
        "device": str(device),
    }

    output_path = ROOT / "results" / "rnn_final.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(results, file, indent=2)

    print("\nFinal RNN evaluation:")
    print(f"Test targets: {total_targets:,}")
    print(f"Test CE: {test_ce:.4f}")
    print(f"Test PPL: {test_ppl:.2f}")
    print(f"Evaluation time: {evaluation_seconds:.2f} s")
    print(f"Throughput: {throughput:,.0f} tokens/s")
    print(
        "Checkpoint size: "
        f"{size_bytes / (1024 ** 2):.2f} MiB"
    )
    print(f"Results saved to: {output_path}")


if __name__ == "__main__":
    main()
