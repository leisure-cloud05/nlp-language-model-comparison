"""Tests for interpolated trigram language model."""

import math
import sys
from pathlib import Path

sys.path.insert(
    0, str(Path(__file__).resolve().parents[1])
)

from src.models.ngram import NGramLanguageModel


def main():
    # 0 PAD, 1 UNK, 2 BOS, 3 EOS
    # 4 The, 5 cat, 6 sleeps, 7 dog
    documents = [
        [2, 4, 5, 6, 3],
        [2, 4, 7, 6, 3],
    ]

    model = NGramLanguageModel(
        vocab_size=8,
        alpha=0.1,
        tau2=10.0,
        tau3=10.0,
    )

    model.fit(documents)
    stats = model.model_statistics()

    assert stats["training_targets"] == 8
    assert stats["unigrams"] == 5
    assert sum(model.trigram.values()) == 8

    # Both documents begin with BOS BOS -> The.
    assert model.trigram[(2, 2, 4)] == 2

    print("Trigram counting and BOS padding: PASS")

    valid_targets = [1, 3, 4, 5, 6, 7]

    for context in [
        (2, 2),
        (2, 4),
        (4, 5),
        (5, 7),  # Unseen history
    ]:
        probabilities = [
            model.probability(w, context)
            for w in valid_targets
        ]

        assert all(p > 0 for p in probabilities)
        assert math.isclose(
            sum(probabilities), 1.0, abs_tol=1e-10
        )

    assert model.probability(0, (2, 2)) == 0
    assert model.probability(2, (2, 2)) == 0

    print("Probability normalization and smoothing: PASS")

    assert (
        model.probability(4, (2, 2))
        > model.probability(1, (2, 2))
    )

    print("Learned context preference: PASS")

    results = model.evaluate(documents)

    assert results["num_targets"] == 8
    assert results["cross_entropy"] > 0
    assert math.isclose(
        results["perplexity"],
        math.exp(results["cross_entropy"]),
    )

    print("Cross-entropy and perplexity: PASS")
    print("Example results:", results)
    print("\nAll n-gram tests passed!")


if __name__ == "__main__":
    main()
