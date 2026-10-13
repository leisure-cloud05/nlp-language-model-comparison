"""Evaluate Part II embedding ablation models on WikiText-2 test set."""

import argparse
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
from src.models.embedding_lstm import EmbeddingAblationLSTM


STRATEGIES = ("random", "skipgram", "glove")


def synchronize(device):
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()


def evaluate_one(strategy, device, vocab):
    checkpoint_path = (
        ROOT / "checkpoints"
        / f"embedding_ablation_{strategy}_best.pt"
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )

    assert checkpoint["embedding_strategy"] == strategy
    assert checkpoint["vocab_size"] == len(vocab)
    assert checkpoint["pad_id"] == vocab["<pad>"]

    config = checkpoint["config"]
    frozen = checkpoint["freeze_embeddings"]

    assert frozen == (strategy != "random")

    # Frozen embeddings are restored entirely from the checkpoint.
    # A zero-initialized placeholder avoids loading Gensim/GloVe again.
    pretrained = None

    if frozen:
        pretrained = torch.zeros(
            len(vocab),
            config["embedding_dim"],
        )

    model = EmbeddingAblationLSTM(
        vocab_size=len(vocab),
        embedding_dim=config["embedding_dim"],
        hidden_dim=config["hidden_dim"],
        num_layers=config["num_layers"],
        pad_id=vocab["<pad>"],
        pretrained_embeddings=pretrained,
        freeze_embeddings=frozen,
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    assert model.embedding.weight.requires_grad == (not frozen)

    test_docs = load_documents("test")

    dataset = LanguageModelDataset(
        test_docs,
        seq_len=config["seq_len"],
    )

    loader = DataLoader(
        dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        collate_fn=make_collate_fn(vocab["<pad>"]),
        num_workers=0,
    )

    total_nll = 0.0
    total_tokens = 0

    # Warm up without including warmup time in the measurement.
    with torch.inference_mode():
        first_batch = next(iter(loader))
        warmup_inputs = first_batch["input_ids"].to(device)
        model(warmup_inputs)

    synchronize(device)
    start = time.perf_counter()

    with torch.inference_mode():
        for batch in loader:
            inputs = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            logits, _ = model(inputs)

            nll = F.cross_entropy(
                logits.reshape(-1, len(vocab)),
                labels.reshape(-1),
                ignore_index=-100,
                reduction="sum",
            )

            total_nll += nll.item()
            total_tokens += (labels != -100).sum().item()

    synchronize(device)
    elapsed = time.perf_counter() - start

    assert total_tokens == 238352, (
        f"Unexpected number of test targets: {total_tokens}"
    )

    test_ce = total_nll / total_tokens
    test_ppl = math.exp(test_ce)

    result = {
        "model": "embedding_ablation_lstm",
        "embedding_strategy": strategy,
        "freeze_embeddings": frozen,
        "dataset": "wikitext-2-raw-v1",
        "split": "test",
        "device": str(device),
        "seed": checkpoint["seed"],
        "config": config,
        "best_epoch": checkpoint["best_epoch"],
        "best_val_ce": checkpoint["best_val_ce"],
        "embedding_coverage": checkpoint["embedding_coverage"],
        "trainable_parameters": model.num_parameters(),
        "test_ce": test_ce,
        "test_ppl": test_ppl,
        "test_targets": total_tokens,
        "evaluation_seconds": elapsed,
        "evaluation_tokens_per_second": total_tokens / elapsed,
        "checkpoint": str(checkpoint_path.relative_to(ROOT)),
    }

    output_path = (
        ROOT / "results"
        / f"embedding_ablation_{strategy}_final.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    output_path.write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )

    print(f"\n{strategy.upper()}")
    print(f"Best epoch: {checkpoint['best_epoch']}")
    print(f"Test CE: {test_ce:.4f}")
    print(f"Test PPL: {test_ppl:.2f}")
    print(f"Test targets: {total_tokens:,}")
    print(f"Evaluation time: {elapsed:.2f}s")
    print(
        f"Throughput: {total_tokens / elapsed:,.0f} tokens/s"
    )
    print(f"Results saved: {output_path}")

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--embedding",
        choices=(*STRATEGIES, "all"),
        default="all",
    )
    args = parser.parse_args()

    device = torch.device(
        "mps" if torch.backends.mps.is_available() else "cpu"
    )

    vocab = load_vocabulary()
    print(f"Device: {device}")

    strategies = (
        STRATEGIES
        if args.embedding == "all"
        else (args.embedding,)
    )

    for strategy in strategies:
        evaluate_one(strategy, device, vocab)

    print("\nEvaluation completed.")


if __name__ == "__main__":
    main()
