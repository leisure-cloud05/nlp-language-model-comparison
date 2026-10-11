"""Small-scale causal Transformer training test on real WikiText-2 data."""

import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

from src.data.lm_dataset import (
    LanguageModelDataset,
    load_documents,
    load_vocabulary,
    make_collate_fn,
)
from src.models.transformer import CausalTransformerLanguageModel


SEED = 42
BATCH_SIZE = 8
SEQ_LEN = 64
NUM_STEPS = 20
LEARNING_RATE = 0.001


def synchronize(device):
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()


def main():
    random.seed(SEED)
    torch.manual_seed(SEED)

    device = torch.device(
        "mps" if torch.backends.mps.is_available() else "cpu"
    )

    print(f"Device: {device}")
    print("Loading training data...")

    vocab = load_vocabulary()
    documents = load_documents("train")

    dataset = LanguageModelDataset(
        documents,
        seq_len=SEQ_LEN,
    )

    # Reproducible subset, consistent with RNN/LSTM smoke tests.
    generator = torch.Generator().manual_seed(SEED)
    chosen = torch.randperm(
        len(dataset), generator=generator
    )[:BATCH_SIZE * NUM_STEPS].tolist()

    loader = DataLoader(
        Subset(dataset, chosen),
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=make_collate_fn(vocab["<pad>"]),
        num_workers=0,
    )

    model = CausalTransformerLanguageModel(
        vocab_size=len(vocab),
        embedding_dim=128,
        num_heads=4,
        num_layers=2,
        feed_forward_dim=512,
        max_seq_len=64,
        dropout=0.1,
        pad_id=vocab["<pad>"],
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    criterion = nn.CrossEntropyLoss(ignore_index=-100)

    print(f"Model parameters: {model.num_parameters():,}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Sequence length: {SEQ_LEN}")
    print(f"Training steps: {NUM_STEPS}")

    model.train()

    losses = []
    step_times = []

    for step, batch in enumerate(loader, start=1):
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)

        synchronize(device)
        start = time.perf_counter()

        optimizer.zero_grad(set_to_none=True)

        logits = model(input_ids)

        loss = criterion(
            logits.reshape(-1, model.vocab_size),
            labels.reshape(-1),
        )

        if not torch.isfinite(loss):
            raise RuntimeError(
                f"Non-finite loss at step {step}: {loss.item()}"
            )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0,
        )

        optimizer.step()

        synchronize(device)
        elapsed = time.perf_counter() - start

        losses.append(loss.item())
        step_times.append(elapsed)

        print(
            f"Step {step:02d}/{NUM_STEPS} | "
            f"Loss: {loss.item():.4f} | "
            f"Time: {elapsed:.2f} s",
            flush=True,
        )

    steady_times = step_times[3:]

    print("\nSmoke test summary:")
    print(f"Initial loss: {losses[0]:.4f}")
    print(f"Final loss: {losses[-1]:.4f}")
    print(
        "Average step time (after warmup): "
        f"{sum(steady_times) / len(steady_times):.2f} s"
    )
    print("Smoke test completed successfully.")


if __name__ == "__main__":
    main()
