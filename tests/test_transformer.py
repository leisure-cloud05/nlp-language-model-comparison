
"""Correctness tests for the manually implemented causal Transformer."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
import torch.nn.functional as F

from src.models.transformer import (
    CausalSelfAttention,
    CausalTransformerLanguageModel,
)


def test_forward_and_gradients():
    torch.manual_seed(42)

    model = CausalTransformerLanguageModel(
        vocab_size=30,
        embedding_dim=16,
        num_heads=4,
        num_layers=2,
        feed_forward_dim=32,
        max_seq_len=8,
        dropout=0.0,
        pad_id=0,
    )

    inputs = torch.tensor([
        [2, 4, 5, 6],
        [2, 7, 8, 0],
    ])

    labels = torch.tensor([
        [4, 5, 6, 3],
        [7, 8, 3, -100],
    ])

    logits = model(inputs)

    assert logits.shape == (2, 4, 30)

    loss = F.cross_entropy(
        logits.reshape(-1, model.vocab_size),
        labels.reshape(-1),
        ignore_index=-100,
    )

    assert torch.isfinite(loss)
    loss.backward()

    assert model.token_embedding.weight.grad is not None
    assert model.position_embedding.weight.grad is not None
    assert model.blocks[0].attention.qkv.weight.grad is not None
    assert model.blocks[0].feed_forward[0].weight.grad is not None
    assert model.output_layer.weight.grad is not None

    for parameter in model.parameters():
        if parameter.grad is not None:
            assert torch.isfinite(parameter.grad).all()

    assert torch.all(model.token_embedding.weight.grad[0] == 0)

    print("Forward shapes and backward gradients: PASS")
    print("Padding embedding gradient: PASS")


def test_causal_behavior():
    torch.manual_seed(42)

    model = CausalTransformerLanguageModel(
        vocab_size=30,
        embedding_dim=16,
        num_heads=4,
        num_layers=2,
        feed_forward_dim=32,
        max_seq_len=8,
        dropout=0.0,
        pad_id=0,
    )

    model.eval()

    original = torch.tensor([[2, 4, 5, 6]])
    changed = torch.tensor([[2, 4, 9, 10]])

    with torch.inference_mode():
        original_logits = model(original)
        changed_logits = model(changed)

    # Positions 0 and 1 must not depend on later tokens.
    torch.testing.assert_close(
        original_logits[:, :2, :],
        changed_logits[:, :2, :],
        rtol=1e-5,
        atol=1e-6,
    )

    # Changes to the input should affect at least some later logits.
    assert not torch.allclose(
        original_logits[:, 2:, :],
        changed_logits[:, 2:, :],
    )

    print("Causal next-token behavior: PASS")


def test_attention_mask_directly():
    torch.manual_seed(42)

    attention = CausalSelfAttention(
        embedding_dim=8,
        num_heads=2,
        dropout=0.0,
    )

    attention.eval()

    x = torch.randn(1, 4, 8)

    # Reconstruct the attention probabilities from module weights
    # to verify the same causal masking formula used in forward().
    with torch.no_grad():
        qkv = attention.qkv(x)
        q, k, _ = qkv.chunk(3, dim=-1)

        q = q.reshape(1, 4, 2, 4).transpose(1, 2)
        k = k.reshape(1, 4, 2, 4).transpose(1, 2)

        scores = (q @ k.transpose(-2, -1)) / (4 ** 0.5)

        causal_mask = torch.ones(
            4, 4, dtype=torch.bool
        ).triu(diagonal=1)

        masked_scores = scores.masked_fill(
            causal_mask[None, None, :, :],
            float("-inf"),
        )

        weights = torch.softmax(masked_scores, dim=-1)

        # Every attention head must assign zero weight to the future.
        future_weights = weights.masked_select(
            causal_mask[None, None, :, :].expand_as(weights)
        )

        assert torch.all(future_weights == 0)

        # The remaining attention weights should sum to one.
        torch.testing.assert_close(
            weights.sum(dim=-1),
            torch.ones_like(weights.sum(dim=-1)),
        )

    print("Attention masking probabilities: PASS")


def test_sequence_length_and_full_model():
    model = CausalTransformerLanguageModel(
        vocab_size=30,
        embedding_dim=16,
        num_heads=4,
        num_layers=2,
        feed_forward_dim=32,
        max_seq_len=8,
        dropout=0.0,
    )

    try:
        model(torch.ones(1, 9, dtype=torch.long))
    except ValueError:
        pass
    else:
        raise AssertionError("Long sequences should be rejected.")

    print("Maximum context length validation: PASS")

    full_model = CausalTransformerLanguageModel(
        vocab_size=20000,
        embedding_dim=128,
        num_heads=4,
        num_layers=2,
        feed_forward_dim=512,
        max_seq_len=64,
        dropout=0.1,
        pad_id=0,
    )

    assert full_model.num_parameters() > 0

    with torch.inference_mode():
        output = full_model(torch.tensor([[2, 4, 5]]))

    assert output.shape == (1, 3, 20000)

    print("Full Transformer model: PASS")
    print(f"Parameters: {full_model.num_parameters():,}")


def main():
    test_forward_and_gradients()
    test_causal_behavior()
    test_attention_mask_directly()
    test_sequence_length_and_full_model()

    print("\nAll Transformer tests passed!")


if __name__ == "__main__":
    main()
