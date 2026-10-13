"""From-scratch Skip-gram with Negative Sampling (SGNS)."""

import torch
from torch import nn
from torch.nn import functional as F


class SkipGramNegativeSampling(nn.Module):
    """Minimal SGNS implemented using PyTorch tensor operations."""

    def __init__(self, vocab_size, embedding_dim):
        super().__init__()

        self.input_embeddings = nn.Embedding(
            vocab_size, embedding_dim
        )
        self.output_embeddings = nn.Embedding(
            vocab_size, embedding_dim
        )

        # Standard small random initialization for center embeddings.
        nn.init.uniform_(
            self.input_embeddings.weight,
            -0.5 / embedding_dim,
            0.5 / embedding_dim,
        )

        # Context embeddings initially start at zero.
        nn.init.zeros_(self.output_embeddings.weight)

    def forward(self, center_ids, positive_ids, negative_ids):
        """
        Args:
            center_ids:   [batch]
            positive_ids: [batch]
            negative_ids: [batch, num_negatives]

        Returns:
            Scalar mean SGNS loss over positive word pairs.
        """
        centers = self.input_embeddings(center_ids)
        positives = self.output_embeddings(positive_ids)
        negatives = self.output_embeddings(negative_ids)

        # Positive dot products: [batch]
        positive_scores = torch.sum(
            centers * positives, dim=-1
        )

        # Negative dot products: [batch, num_negatives]
        negative_scores = torch.bmm(
            negatives,
            centers.unsqueeze(-1),
        ).squeeze(-1)

        positive_log_prob = F.logsigmoid(positive_scores)

        negative_log_prob = F.logsigmoid(
            -negative_scores
        ).sum(dim=1)

        loss = -(
            positive_log_prob + negative_log_prob
        ).mean()

        return loss


def build_skipgram_pairs(documents, window_size=2):
    """Build positive center-context ID pairs within each document."""

    if window_size < 1:
        raise ValueError("window_size must be positive")

    pairs = []

    for document in documents:
        for center_position, center_id in enumerate(document):
            left = max(0, center_position - window_size)
            right = min(
                len(document),
                center_position + window_size + 1,
            )

            for context_position in range(left, right):
                if context_position == center_position:
                    continue

                pairs.append((
                    center_id,
                    document[context_position],
                ))

    return pairs



def sample_negative_ids(
    batch_size,
    num_negatives,
    vocab_size,
    device="cpu",
    frequencies=None,
    excluded_ids=None,
    generator=None,
):
    """
    Sample negative word IDs.

    If frequencies are provided, use the standard
    Word2Vec distribution proportional to count ** 0.75.

    Otherwise, sample uniformly over eligible IDs.

    Excluded IDs and zero-frequency words cannot be sampled.
    """
    if batch_size < 1 or num_negatives < 1:
        raise ValueError("batch_size and num_negatives must be positive")

    if vocab_size < 1:
        raise ValueError("vocab_size must be positive")

    if frequencies is None:
        weights = torch.ones(vocab_size, dtype=torch.float64)
    else:
        weights = torch.as_tensor(
            frequencies, dtype=torch.float64
        ).clone()

        if weights.shape != (vocab_size,):
            raise ValueError("frequencies must have shape [vocab_size]")

        if not torch.isfinite(weights).all():
            raise ValueError("frequencies must be finite")

        if (weights < 0).any():
            raise ValueError("frequencies cannot be negative")

        weights = weights.pow(0.75)

    if excluded_ids is not None:
        for token_id in excluded_ids:
            if not 0 <= token_id < vocab_size:
                raise ValueError("Invalid excluded token ID")
            weights[token_id] = 0.0

    if weights.sum() <= 0:
        raise ValueError("No eligible tokens for negative sampling")

    probabilities = weights / weights.sum()

    samples = torch.multinomial(
        probabilities,
        num_samples=batch_size * num_negatives,
        replacement=True,
        generator=generator,
    )

    return samples.reshape(
        batch_size, num_negatives
    ).to(device)
