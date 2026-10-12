
"""Train an LSTM language model on WikiText-2."""

import json
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
from src.models.lstm import LSTMLanguageModel


SEED = 42

CONFIG = {
    "embedding_dim": 128,
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
    dataset = LanguageModelDataset(
        documents,
        seq_len=CONFIG["seq_len"],
    )

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
            # Each batch starts with zero LSTM hidden/cell states.
            # No states are carried across independent windows.
            logits, _ = model(inputs)

            # Sum NLL over valid targets only.
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
                    model.parameters(),
                    max_norm=CONFIG["grad_clip"],
                )

                optimizer.step()

        total_nll += nll.detach().item()
        total_tokens += num_tokens.item()

        if training and step % 200 == 0:
            print(
                f"  Step {step}/{len(loader)} | "
                f"Batch loss: {loss.item():.4f}",
                flush=True,
            )

    return total_nll / total_tokens, total_tokens


def main():
    set_seed(SEED)

    device = torch.device(
        "mps" if torch.backends.mps.is_available() else "cpu"
    )

    print(f"Device: {device}", flush=True)
    print("Loading processed WikiText-2...", flush=True)

    vocab = load_vocabulary()
    train_docs = load_documents("train")
    val_docs = load_documents("validation")

    train_loader = build_loader(train_docs, vocab, shuffle=True)
    val_loader = build_loader(val_docs, vocab, shuffle=False)

    model = LSTMLanguageModel(
        vocab_size=len(vocab),
        embedding_dim=CONFIG["embedding_dim"],
        hidden_dim=CONFIG["hidden_dim"],
        num_layers=CONFIG["num_layers"],
        pad_id=vocab["<pad>"],
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=CONFIG["learning_rate"],
    )

    checkpoint_dir = ROOT / "checkpoints"
    result_dir = ROOT / "results"

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)

    best_path = checkpoint_dir / "lstm_best.pt"
    history_path = result_dir / "lstm_training.json"

    print(f"Parameters: {model.num_parameters():,}")
    print(f"Train batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")
    print(f"Config: {CONFIG}", flush=True)

    history = []
    best_val_ce = float("inf")
    best_epoch = None
    epochs_without_improvement = 0

    for epoch in range(1, CONFIG["epochs"] + 1):
        print(
            f"\nEpoch {epoch}/{CONFIG['epochs']}",
            flush=True,
        )

        synchronize(device)
        start = time.perf_counter()

        train_ce, train_targets = run_epoch(
            model, train_loader, device, optimizer
        )

        synchronize(device)
        train_seconds = time.perf_counter() - start

        start = time.perf_counter()

        val_ce, val_targets = run_epoch(
            model, val_loader, device
        )

        synchronize(device)
        val_seconds = time.perf_counter() - start

        val_ppl = float(torch.exp(torch.tensor(val_ce)))

        record = {
            "epoch": epoch,
            "train_ce": train_ce,
            "val_ce": val_ce,
            "val_ppl": val_ppl,
            "train_targets": train_targets,
            "val_targets": val_targets,
            "train_seconds": train_seconds,
            "val_seconds": val_seconds,
        }

        history.append(record)

        print(
            f"Train CE: {train_ce:.4f} | "
            f"Val CE: {val_ce:.4f} | "
            f"Val PPL: {val_ppl:.2f}",
            flush=True,
        )

        print(
            f"Train time: {train_seconds:.1f}s | "
            f"Val time: {val_seconds:.1f}s",
            flush=True,
        )

        
        # Always save the checkpoint with the lowest validation CE.
        previous_best_ce = best_val_ce

        if val_ce < best_val_ce:
            best_val_ce = val_ce
            best_epoch = epoch

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": CONFIG.copy(),
                    "vocab_size": len(vocab),
                    "pad_id": vocab["<pad>"],
                    "seed": SEED,
                    "best_epoch": best_epoch,
                    "best_val_ce": best_val_ce,
                },
                best_path,
            )

            print("Best checkpoint updated.", flush=True)

        # Apply the same early stopping rule as the RNN.
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

        # Save the history of every completed epoch.
        with history_path.open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "model": "lstm",
                    "dataset": "wikitext-2-raw-v1",
                    "seed": SEED,
                    "config": CONFIG,
                    "parameters": model.num_parameters(),
                    "history": history,
                    "best_epoch": best_epoch,
                    "best_val_ce": best_val_ce,
                    "epochs_without_improvement": (
                        epochs_without_improvement
                    ),
                },
                f,
                indent=2,
            )

        if (
            epochs_without_improvement
            >= CONFIG["early_stopping_patience"]
        ):
            print(
                f"Early stopping triggered at epoch {epoch}.",
                flush=True,
            )
            break


    print("\nTraining completed.")
    print(f"Epochs completed: {len(history)}")
    print(f"Best epoch: {best_epoch}")
    print(f"Best validation CE: {best_val_ce:.4f}")
    print(f"Checkpoint: {best_path}")
    print(f"Training history: {history_path}")

if __name__ == "__main__":
    main()
