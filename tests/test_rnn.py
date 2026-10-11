"""Correctness tests for the Vanilla RNN language model."""

import sys
from pathlib import Path

sys.path.insert(
    0, str(Path(__file__).resolve().parents[1])
)

import torch
from torch import nn

from src.models.rnn import RNNLanguageModel


def main():
    torch.manual_seed(42)

    model = RNNLanguageModel(
        vocab_size=20,
        embedding_dim=8,
        hidden_dim=16,
        num_layers=1,
        pad_id=0,
    )

    # Two small sequences, with padding in the second.
    inputs = torch.tensor(
        [
            [2, 4, 5, 6],
            [2, 7, 8, 0],
        ],
        dtype=torch.long,
    )

    targets = torch.tensor(
        [
            [4, 5, 6, 3],
            [7, 8, 3, -100],
        ],
        dtype=torch.long,
    )

    logits, hidden = model(inputs)

    assert logits.shape == (2, 4, 20)
    assert hidden.shape == (1, 2, 16)

    print("Forward shapes: PASS")

    criterion = nn.CrossEntropyLoss(
        ignore_index=-100
    )

    loss = criterion(
        logits.reshape(-1, 20),
        targets.reshape(-1),
    )

    assert torch.isfinite(loss)
    assert loss.item() > 0

    print("Cross-entropy with padding: PASS")

    loss.backward()

    assert model.embedding.weight.grad is not None
    assert model.rnn.weight_ih_l0.grad is not None
    assert model.output_layer.weight.grad is not None

    assert torch.isfinite(
        model.output_layer.weight.grad
    ).all()

    print("Backward gradients: PASS")

    # Padding embedding should not be updated by gradients.
    assert torch.all(
        model.embedding.weight.grad[0] == 0
    )

    print("Padding embedding gradients: PASS")

    print("Trainable parameters:", model.num_parameters())
    print("\nAll RNN tests passed!")


if __name__ == "__main__":
    main()
