"""Train Part II LSTM language models with three embedding strategies.

Run from project root, e.g.:
    python scripts/train_embedding_ablation.py --embedding random
    python scripts/train_embedding_ablation.py --embedding skipgram
    python scripts/train_embedding_ablation.py --embedding glove
"""

import argparse
import json
import math
import random
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

SEED = 42
CONFIG = {
    "embedding_dim": 100,
    "hidden_dim": 256,
    "num_layers": 1,
    "batch_size": 16,
    "seq_len": 64,
    "learning_rate": 0.001,
    "epochs": 15,
    "early_stopping_patience": 3,
    "min_delta": 0.001,
    "grad_clip": 1.0,
}


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)


def synchronize(device):
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()


def build_loader(documents, vocab, shuffle):
    dataset = LanguageModelDataset(documents, seq_len=CONFIG["seq_len"])
    return DataLoader(
        dataset,
        batch_size=CONFIG["batch_size"],
        shuffle=shuffle,
        collate_fn=make_collate_fn(vocab["<pad>"]),
        num_workers=0,
    )


def run_epoch(model, loader, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total_nll = 0.0
    total_tokens = 0

    for step, batch in enumerate(loader, start=1):
        inputs = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            # Consistent with Part I: independent windows, zero initial states.
            logits, _ = model(inputs)
            nll = F.cross_entropy(
                logits.reshape(-1, model.vocab_size),
                labels.reshape(-1),
                ignore_index=-100,
                reduction="sum",
            )
            num_tokens = (labels != -100).sum()
            loss = nll / num_tokens

            if training:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    (p for p in model.parameters() if p.requires_grad),
                    max_norm=CONFIG["grad_clip"],
                )
                optimizer.step()

        total_nll += nll.detach().item()
        total_tokens += num_tokens.item()

        if training and step % 200 == 0:
            print(
                f"  Step {step}/{len(loader)} | Batch loss: {loss.item():.4f}",
                flush=True,
            )

    return total_nll / total_tokens, total_tokens


def build_model(embedding_name, vocab):
    pretrained_embeddings = None
    coverage_stats = None
    freeze_embeddings = embedding_name != "random"

    if embedding_name == "skipgram":
        from src.embeddings.embedding_utils import load_skipgram_matrix
        pretrained_embeddings, coverage_stats = load_skipgram_matrix(
            str(ROOT / "checkpoints" / "skipgram_word2vec.model")
        )
    elif embedding_name == "glove":
        from src.embeddings.embedding_utils import load_glove_matrix
        pretrained_embeddings, coverage_stats = load_glove_matrix()

    model = EmbeddingAblationLSTM(
        vocab_size=len(vocab),
        embedding_dim=CONFIG["embedding_dim"],
        hidden_dim=CONFIG["hidden_dim"],
        num_layers=CONFIG["num_layers"],
        pad_id=vocab["<pad>"],
        pretrained_embeddings=pretrained_embeddings,
        freeze_embeddings=freeze_embeddings,
    )
    assert model.embedding.weight.requires_grad == (not freeze_embeddings)
    return model, coverage_stats


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--embedding",
        required=True,
        choices=("random", "skipgram", "glove"),
    )
    args = parser.parse_args()
    embedding_name = args.embedding

    set_seed(SEED)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Embedding strategy: {embedding_name}", flush=True)
    print(f"Device: {device}", flush=True)

    vocab = load_vocabulary()
    train_docs = load_documents("train")
    val_docs = load_documents("validation")
    train_loader = build_loader(train_docs, vocab, shuffle=True)
    val_loader = build_loader(val_docs, vocab, shuffle=False)

    # The same seed is used for neural model initialization and shuffle for
    # every strategy, regardless of any work done by an embedding loader.
    set_seed(SEED)
    model, coverage_stats = build_model(embedding_name, vocab)
    model = model.to(device)
    optimizer = torch.optim.Adam(
        (p for p in model.parameters() if p.requires_grad),
        lr=CONFIG["learning_rate"],
    )

    checkpoint_dir = ROOT / "checkpoints"
    result_dir = ROOT / "results"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"embedding_ablation_{embedding_name}"
    best_path = checkpoint_dir / f"{prefix}_best.pt"
    history_path = result_dir / f"{prefix}_training.json"

    print(f"Trainable parameters: {model.num_parameters():,}")
    print(f"Frozen embeddings: {embedding_name != 'random'}")
    print(f"Coverage: {coverage_stats}")
    print(f"Train batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")
    print(f"Config: {CONFIG}", flush=True)

    history = []
    best_val_ce = float("inf")
    best_epoch = None
    epochs_without_improvement = 0

    for epoch in range(1, CONFIG["epochs"] + 1):
        print(f"\nEpoch {epoch}/{CONFIG['epochs']}", flush=True)
        synchronize(device)
        start = time.perf_counter()
        train_ce, train_targets = run_epoch(model, train_loader, device, optimizer)
        synchronize(device)
        train_seconds = time.perf_counter() - start

        start = time.perf_counter()
        val_ce, val_targets = run_epoch(model, val_loader, device)
        synchronize(device)
        val_seconds = time.perf_counter() - start
        val_ppl = math.exp(val_ce)
        history.append({
            "epoch": epoch,
            "train_ce": train_ce,
            "val_ce": val_ce,
            "val_ppl": val_ppl,
            "train_targets": train_targets,
            "val_targets": val_targets,
            "train_seconds": train_seconds,
            "val_seconds": val_seconds,
        })
        print(
            f"Train CE: {train_ce:.4f} | Val CE: {val_ce:.4f} | "
            f"Val PPL: {val_ppl:.2f}",
            flush=True,
        )
        print(
            f"Train time: {train_seconds:.1f}s | Val time: {val_seconds:.1f}s",
            flush=True,
        )

        previous_best_ce = best_val_ce
        if val_ce < best_val_ce:
            best_val_ce = val_ce
            best_epoch = epoch
            torch.save({
                "model_state_dict": model.state_dict(),
                "config": CONFIG.copy(),
                "vocab_size": len(vocab),
                "pad_id": vocab["<pad>"],
                "seed": SEED,
                "best_epoch": best_epoch,
                "best_val_ce": best_val_ce,
                "embedding_strategy": embedding_name,
                "freeze_embeddings": embedding_name != "random",
                "embedding_coverage": coverage_stats,
            }, best_path)
            print("Best checkpoint updated.", flush=True)

        if val_ce < previous_best_ce - CONFIG["min_delta"]:
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            print(
                "No significant validation improvement: "
                f"{epochs_without_improvement}/"
                f"{CONFIG['early_stopping_patience']}",
                flush=True,
            )

        with history_path.open("w", encoding="utf-8") as f:
            json.dump({
                "model": "embedding_ablation_lstm",
                "embedding_strategy": embedding_name,
                "freeze_embeddings": embedding_name != "random",
                "embedding_coverage": coverage_stats,
                "dataset": "wikitext-2-raw-v1",
                "seed": SEED,
                "config": CONFIG.copy(),
                "parameters": model.num_parameters(),
                "history": history,
                "best_epoch": best_epoch,
                "best_val_ce": best_val_ce,
                "epochs_without_improvement": epochs_without_improvement,
            }, f, indent=2)

        if epochs_without_improvement >= CONFIG["early_stopping_patience"]:
            print(f"Early stopping triggered at epoch {epoch}.", flush=True)
            break

    print("\nTraining completed.")
    print(f"Epochs completed: {len(history)}")
    print(f"Best epoch: {best_epoch}")
    print(f"Best validation CE: {best_val_ce:.4f}")
    print(f"Checkpoint: {best_path}")
    print(f"Training history: {history_path}")


if __name__ == "__main__":
    main()
