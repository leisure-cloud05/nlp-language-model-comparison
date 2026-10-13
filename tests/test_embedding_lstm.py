"""Correctness tests for the Part II embedding ablation LSTM."""

import torch
from torch.nn import functional as F

from src.models.embedding_lstm import EmbeddingAblationLSTM


def test_forward_shapes():
    model = EmbeddingAblationLSTM(
        vocab_size=20,
        embedding_dim=5,
        hidden_dim=8,
    )

    inputs = torch.tensor([
        [1, 2, 3, 0],
        [4, 5, 0, 0],
    ])

    logits, (h_n, c_n) = model(inputs)

    assert logits.shape == (2, 4, 20)
    assert h_n.shape == (1, 2, 8)
    assert c_n.shape == (1, 2, 8)
    assert torch.isfinite(logits).all()

    print("Forward outputs and LSTM states: PASS")


def test_random_trainable():
    torch.manual_seed(42)

    model = EmbeddingAblationLSTM(
        vocab_size=20,
        embedding_dim=5,
        hidden_dim=8,
    )

    assert model.embedding.weight.requires_grad

    before = model.embedding.weight.detach().clone()

    optimizer = torch.optim.Adam(
        model.parameters(), lr=0.01
    )

    inputs = torch.tensor([[1, 2, 3]])
    labels = torch.tensor([[2, 3, 4]])

    optimizer.zero_grad()

    logits, _ = model(inputs)
    loss = F.cross_entropy(
        logits.reshape(-1, 20),
        labels.reshape(-1),
    )

    loss.backward()
    optimizer.step()

    assert not torch.equal(
        before[1],
        model.embedding.weight.detach()[1],
    )

    assert torch.count_nonzero(
        model.embedding.weight.detach()[0]
    ) == 0

    print("Random trainable embedding updates: PASS")


def test_frozen_pretrained():
    torch.manual_seed(42)

    matrix = torch.randn(20, 5)
    matrix[0] = 0.0

    model = EmbeddingAblationLSTM(
        vocab_size=20,
        embedding_dim=5,
        hidden_dim=8,
        pretrained_embeddings=matrix,
        freeze_embeddings=True,
    )

    assert not model.embedding.weight.requires_grad
    assert torch.equal(
        model.embedding.weight.detach(), matrix
    )

    before_embedding = (
        model.embedding.weight.detach().clone()
    )
    before_lstm = (
        model.lstm.weight_ih_l0.detach().clone()
    )
    before_output = (
        model.output_layer.weight.detach().clone()
    )

    optimizer = torch.optim.Adam(
        (p for p in model.parameters() if p.requires_grad),
        lr=0.01,
    )

    inputs = torch.tensor([[1, 2, 3]])
    labels = torch.tensor([[2, 3, 4]])

    optimizer.zero_grad()

    logits, _ = model(inputs)
    loss = F.cross_entropy(
        logits.reshape(-1, 20),
        labels.reshape(-1),
    )

    loss.backward()

    assert model.embedding.weight.grad is None
    assert model.lstm.weight_ih_l0.grad is not None

    optimizer.step()

    assert torch.equal(
        before_embedding,
        model.embedding.weight.detach(),
    )
    assert not torch.equal(
        before_lstm,
        model.lstm.weight_ih_l0.detach(),
    )
    assert not torch.equal(
        before_output,
        model.output_layer.weight.detach(),
    )

    print("Frozen embedding and trainable LSTM: PASS")


def test_invalid_inputs():
    wrong_shape = torch.randn(10, 5)

    try:
        EmbeddingAblationLSTM(
            vocab_size=20,
            embedding_dim=5,
            pretrained_embeddings=wrong_shape,
            freeze_embeddings=True,
        )
    except ValueError:
        pass
    else:
        raise AssertionError(
            "Expected error for incorrect embedding shape"
        )

    try:
        EmbeddingAblationLSTM(
            vocab_size=20,
            embedding_dim=5,
            freeze_embeddings=True,
        )
    except ValueError:
        pass
    else:
        raise AssertionError(
            "Expected error when freezing without pretrained weights"
        )

    matrix = torch.randn(20, 5)
    matrix[0] = 0.0
    matrix[2, 1] = float("nan")

    try:
        EmbeddingAblationLSTM(
            vocab_size=20,
            embedding_dim=5,
            pretrained_embeddings=matrix,
            freeze_embeddings=True,
        )
    except ValueError:
        pass
    else:
        raise AssertionError(
            "Expected error for non-finite embeddings"
        )

    print("Invalid embedding configurations rejected: PASS")


if __name__ == "__main__":
    test_forward_shapes()
    test_random_trainable()
    test_frozen_pretrained()
    test_invalid_inputs()

    print("\nAll embedding LSTM tests passed!")
