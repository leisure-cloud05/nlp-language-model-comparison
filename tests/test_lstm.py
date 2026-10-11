"""Tests for the LSTM language model."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from torch import nn

from src.models.lstm import LSTMLanguageModel


def main():
    torch.manual_seed(42)

    model = LSTMLanguageModel(
        vocab_size=20,
        embedding_dim=8,
        hidden_dim=16,
        num_layers=1,
        pad_id=0,
    )

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

    logits, (h_n, c_n) = model(inputs)

    assert logits.shape == (2, 4, 20)
    assert h_n.shape == (1, 2, 16)
    assert c_n.shape == (1, 2, 16)

    print("Forward shapes and LSTM states: PASS")

    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    loss = criterion(
        logits.reshape(-1, model.vocab_size),
        targets.reshape(-1),
    )

    assert torch.isfinite(loss)
    loss.backward()

    for param in model.parameters():
        assert param.grad is not None
        assert torch.isfinite(param.grad).all()

    print("Cross-entropy and backward gradients: PASS")

    assert torch.all(model.embedding.weight.grad[0] == 0)

    print("Padding embedding gradient: PASS")

    # A future token must not affect earlier token predictions.
    model.eval()

    original = torch.tensor([[2, 4, 5, 6]])
    changed = torch.tensor([[2, 4, 9, 10]])

    with torch.no_grad():
        original_logits, _ = model(original)
        changed_logits, _ = model(changed)

    torch.testing.assert_close(
        original_logits[:, :2, :],
        changed_logits[:, :2, :],
        rtol=1e-5,
        atol=1e-6,
    )

    print("Causal next-token behavior: PASS")

    # Check configuration and parameter count.
    full_model = LSTMLanguageModel(
        vocab_size=20000,
        embedding_dim=128,
        hidden_dim=256,
        num_layers=1,
        pad_id=0,
    )

    assert full_model.num_parameters() == 8_095_264

    print("Full-model parameter count: PASS")
    print("Parameters:", full_model.num_parameters())

    print("\nAll LSTM tests passed!")


if __name__ == "__main__":
    main()
