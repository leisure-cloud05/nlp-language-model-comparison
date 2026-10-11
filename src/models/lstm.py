"""LSTM language model for next-token prediction."""

from torch import nn


class LSTMLanguageModel(nn.Module):
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
        if not 0 <= pad_id < vocab_size:
            raise ValueError("pad_id is outside the vocabulary.")

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.pad_id = pad_id

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=pad_id,
        )

        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )

        self.output_layer = nn.Linear(
            hidden_dim,
            vocab_size,
        )

    def forward(self, input_ids, hidden=None):
        """
        Args:
            input_ids: [batch, seq_len]
            hidden: Optional (h0, c0) tuple.

        Returns:
            logits: [batch, seq_len, vocab_size]
            state: (h_n, c_n), each
                   [num_layers, batch, hidden_dim]
        """
        embeddings = self.embedding(input_ids)
        outputs, state = self.lstm(embeddings, hidden)
        logits = self.output_layer(outputs)

        return logits, state

    def num_parameters(self):
        return sum(
            p.numel()
            for p in self.parameters()
            if p.requires_grad
        )
