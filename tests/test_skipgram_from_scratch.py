"""Correctness tests for the from-scratch SGNS implementation."""

import torch
from torch.nn import functional as F

from src.embeddings.skipgram_from_scratch import (
    SkipGramNegativeSampling,
    build_skipgram_pairs,
    sample_negative_ids,
)


def test_pairs():
    documents = [
        [1, 2, 3],
        [4, 5],
    ]

    pairs = build_skipgram_pairs(
        documents,
        window_size=1,
    )

    expected = [
        (1, 2),
        (2, 1),
        (2, 3),
        (3, 2),
        (4, 5),
        (5, 4),
    ]

    assert pairs == expected
    print("Skip-gram pair construction: PASS")


def test_loss_formula():
    torch.manual_seed(42)

    model = SkipGramNegativeSampling(
        vocab_size=6,
        embedding_dim=3,
    )

    center = torch.tensor([1, 2])
    positive = torch.tensor([3, 4])
    negative = torch.tensor([
        [0, 5],
        [1, 3],
    ])

    actual = model(center, positive, negative)

    u = model.input_embeddings(center)
    v_pos = model.output_embeddings(positive)
    v_neg = model.output_embeddings(negative)

    positive_scores = (u * v_pos).sum(dim=-1)
    negative_scores = (
        v_neg * u.unsqueeze(1)
    ).sum(dim=-1)

    expected = -(
        F.logsigmoid(positive_scores)
        + F.logsigmoid(-negative_scores).sum(dim=1)
    ).mean()

    assert torch.allclose(actual, expected, atol=1e-7)
    print("SGNS loss formula: PASS")


def test_gradients():
    torch.manual_seed(42)

    model = SkipGramNegativeSampling(
        vocab_size=8,
        embedding_dim=4,
    )

    center = torch.tensor([1, 2, 3])
    positive = torch.tensor([4, 5, 6])
    negative = torch.tensor([
        [0, 7],
        [0, 7],
        [0, 7],
    ])

    loss = model(center, positive, negative)
    loss.backward()

    assert model.input_embeddings.weight.grad is not None
    assert model.output_embeddings.weight.grad is not None

    assert torch.isfinite(
        model.input_embeddings.weight.grad
    ).all()

    assert torch.isfinite(
        model.output_embeddings.weight.grad
    ).all()

    assert (
        model.output_embeddings.weight.grad.abs().sum() > 0
    )

    print("SGNS backward gradients: PASS")


def test_negative_sampling():
    torch.manual_seed(42)

    samples = sample_negative_ids(
        batch_size=4,
        num_negatives=5,
        vocab_size=20,
    )

    assert samples.shape == (4, 5)
    assert samples.min() >= 0
    assert samples.max() < 20

    print("Negative sampling shape and range: PASS")


def test_training():
    torch.manual_seed(42)

    model = SkipGramNegativeSampling(
        vocab_size=5,
        embedding_dim=8,
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.05,
    )

    center = torch.tensor([0, 1, 2, 3])
    positive = torch.tensor([1, 2, 3, 4])
    negative = torch.tensor([
        [3, 4],
        [4, 0],
        [0, 1],
        [1, 2],
    ])

    with torch.no_grad():
        initial_loss = model(
            center, positive, negative
        ).item()

    for _ in range(100):
        optimizer.zero_grad()
        loss = model(center, positive, negative)
        loss.backward()
        optimizer.step()

    final_loss = model(
        center, positive, negative
    ).item()

    assert final_loss < initial_loss, (
        f"Loss did not decrease: "
        f"{initial_loss:.4f} -> {final_loss:.4f}"
    )

    print(
        f"Toy training loss: "
        f"{initial_loss:.4f} -> {final_loss:.4f}"
    )
    print("SGNS toy training: PASS")


def test_known_loss_and_gradients():
    """Verify SGNS loss and gradients using hand-derived values."""
    import math

    model = SkipGramNegativeSampling(
        vocab_size=3, embedding_dim=2
    )

    with torch.no_grad():
        model.input_embeddings.weight.zero_()
        model.output_embeddings.weight.zero_()

        # Center u = (1, 0)
        model.input_embeddings.weight[0] = torch.tensor([1., 0.])

        # Positive context v+ = (1, 0)
        model.output_embeddings.weight[1] = torch.tensor([1., 0.])

        # Negative context v- = (-1, 0)
        model.output_embeddings.weight[2] = torch.tensor([-1., 0.])

    loss = model(
        torch.tensor([0]),
        torch.tensor([1]),
        torch.tensor([[2]]),
    )

    # Independently derived:
    # L = 2 * log(1 + exp(-1))
    expected_loss = 2 * math.log1p(math.exp(-1))

    assert abs(loss.item() - expected_loss) < 1e-6

    loss.backward()

    p = 1 / (1 + math.exp(1))

    # Analytical SGNS gradients:
    expected_center = torch.tensor([-2 * p, 0.])
    expected_positive = torch.tensor([-p, 0.])
    expected_negative = torch.tensor([p, 0.])

    input_grad = model.input_embeddings.weight.grad
    output_grad = model.output_embeddings.weight.grad

    assert torch.allclose(
        input_grad[0], expected_center, atol=1e-6
    )
    assert torch.allclose(
        output_grad[1], expected_positive, atol=1e-6
    )
    assert torch.allclose(
        output_grad[2], expected_negative, atol=1e-6
    )

    # Embedding rows unused in this batch have zero gradients.
    assert torch.count_nonzero(input_grad[1:]) == 0
    assert torch.count_nonzero(output_grad[0]) == 0

    print("Known SGNS loss and analytical gradients: PASS")


def test_finite_difference_gradients():
    """Numerically check the actual model's parameter gradients."""
    torch.manual_seed(42)

    model = SkipGramNegativeSampling(
        vocab_size=5, embedding_dim=3
    ).double()

    with torch.no_grad():
        model.input_embeddings.weight.uniform_(-0.4, 0.4)
        model.output_embeddings.weight.uniform_(-0.4, 0.4)

    center = torch.tensor([0, 1])
    positive = torch.tensor([2, 3])
    negative = torch.tensor([[3, 4], [4, 2]])

    model.zero_grad()
    loss = model(center, positive, negative)
    loss.backward()

    epsilon = 1e-6

    # Compare finite differences with autograd directly
    # on the parameters used by the actual SGNS module.
    for parameter in (
        model.input_embeddings.weight,
        model.output_embeddings.weight,
    ):
        analytical = parameter.grad.detach().clone()

        for row in range(parameter.shape[0]):
            for col in range(parameter.shape[1]):
                with torch.no_grad():
                    original = parameter[row, col].item()

                    parameter[row, col] = original + epsilon
                    loss_plus = model(
                        center, positive, negative
                    ).item()

                    parameter[row, col] = original - epsilon
                    loss_minus = model(
                        center, positive, negative
                    ).item()

                    parameter[row, col] = original

                numerical = (
                    loss_plus - loss_minus
                ) / (2 * epsilon)

                assert abs(
                    numerical - analytical[row, col].item()
                ) < 1e-5, (
                    f"Gradient mismatch at ({row}, {col}): "
                    f"numerical={numerical}, "
                    f"autograd={analytical[row, col].item()}"
                )

    print("SGNS finite-difference gradients: PASS")


def test_frequency_based_negative_sampling():
    """Check exclusions, reproducibility and sampling probabilities."""
    import math

    frequencies = torch.tensor(
        [0., 0., 0., 0., 1., 4., 16.],
        dtype=torch.float64,
    )

    generator1 = torch.Generator().manual_seed(42)
    generator2 = torch.Generator().manual_seed(42)

    kwargs = dict(
        batch_size=2000,
        num_negatives=50,
        vocab_size=7,
        frequencies=frequencies,
        excluded_ids={0, 1, 2, 3},
    )

    samples1 = sample_negative_ids(
        **kwargs, generator=generator1
    )
    samples2 = sample_negative_ids(
        **kwargs, generator=generator2
    )

    assert torch.equal(samples1, samples2)
    assert samples1.shape == (2000, 50)
    assert samples1.min() >= 4
    assert samples1.max() <= 6

    # Expected probabilities proportional to f ** 0.75.
    weights = frequencies[4:].pow(0.75)
    expected_probs = weights / weights.sum()

    observed_probs = torch.bincount(
        samples1.flatten() - 4,
        minlength=3,
    ).double() / samples1.numel()

    assert torch.allclose(
        observed_probs,
        expected_probs,
        atol=0.01,
        rtol=0.0,
    ), (
        f"Expected {expected_probs.tolist()}, "
        f"observed {observed_probs.tolist()}"
    )

    print("Frequency-based negative sampling: PASS")


if __name__ == "__main__":
    test_pairs()
    test_loss_formula()
    test_gradients()
    test_negative_sampling()
    test_training()

    test_known_loss_and_gradients()
    test_finite_difference_gradients()
    test_frequency_based_negative_sampling()

    print("\nAll from-scratch SGNS tests passed!")