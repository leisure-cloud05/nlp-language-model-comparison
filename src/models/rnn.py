"""Vanilla RNN language model for next-token prediction."""

import torch
from torch import nn


class RNNLanguageModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 128,
        hidden_dim: int = 256,
        num_layers: int = 1,
        pad_id: int = 0,
    ):
        super().__init__()

        if vocab_size <= 0:
            raise ValueError("vocab_size must be positive.")
        if embedding_dim <= 0 or hidden_dim <= 0:
            raise ValueError("Model dimensions must be positive.")
        if num_layers <= 0:
            raise ValueError("num_layers must be positive.")

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.pad_id = pad_id

        # Randomly initialized and trainable.
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=pad_id,
        )

        # Vanilla RNN with tanh activation.
        self.rnn = nn.RNN(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            nonlinearity="tanh",
            batch_first=True,
        )

        # Predict a vocabulary distribution at every time step.
        self.output_layer = nn.Linear(
            hidden_dim,
            vocab_size,
        )

    def forward(self, input_ids, hidden=None):
        """
        Args:
            input_ids: LongTensor [batch_size, sequence_length]
            hidden: Optional RNN hidden state.

        Returns:
            logits: [batch_size, sequence_length, vocab_size]
            final_hidden: [num_layers, batch_size, hidden_dim]
        """
        embeddings = self.embedding(input_ids)

        outputs, final_hidden = self.rnn(
            embeddings,
            hidden,
        )

        # Raw logits, not softmax probabilities.
        logits = self.output_layer(outputs)

        return logits, final_hidden

    def num_parameters(self):
        """Number of trainable model parameters."""
        return sum(
            parameter.numel()
            for parameter in self.parameters()
            if parameter.requires_grad
        )
