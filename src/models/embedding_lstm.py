"""LSTM language model for Part II embedding ablation."""

import torch
from torch import nn

from src.models.lstm import LSTMLanguageModel


class EmbeddingAblationLSTM(LSTMLanguageModel):
    """Reuse the Part I LSTM with configurable word embeddings."""

    def __init__(
        self,
        vocab_size: int = 20000,
        embedding_dim: int = 100,
        hidden_dim: int = 256,
        num_layers: int = 1,
        pad_id: int = 0,
        pretrained_embeddings: torch.Tensor | None = None,
        freeze_embeddings: bool = False,
    ):
        super().__init__(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            pad_id=pad_id,
        )

        if pretrained_embeddings is None:
            if freeze_embeddings:
                raise ValueError(
                    "Cannot freeze embeddings without a "
                    "pretrained embedding matrix."
                )
            # Keep the randomly initialized trainable embedding.
            return

        if pretrained_embeddings.shape != (
            vocab_size,
            embedding_dim,
        ):
            raise ValueError(
                "Pretrained embedding matrix has incorrect shape."
            )

        if not torch.isfinite(pretrained_embeddings).all():
            raise ValueError(
                "Pretrained embeddings contain NaN or Inf."
            )

        if torch.count_nonzero(
            pretrained_embeddings[pad_id]
        ).item() != 0:
            raise ValueError(
                "The PAD embedding must be a zero vector."
            )

        self.embedding = nn.Embedding.from_pretrained(
            pretrained_embeddings.detach().clone(),
            freeze=freeze_embeddings,
            padding_idx=pad_id,
        )
