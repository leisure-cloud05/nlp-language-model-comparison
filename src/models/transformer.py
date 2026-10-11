"""Causal Transformer language model with manual multi-head attention."""

import math

import torch
from torch import nn
from torch.nn import functional as F


class CausalSelfAttention(nn.Module):
    def __init__(self, embedding_dim, num_heads, dropout=0.1):
        super().__init__()

        if embedding_dim % num_heads != 0:
            raise ValueError(
                "embedding_dim must be divisible by num_heads."
            )

        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads

        self.qkv = nn.Linear(embedding_dim, 3 * embedding_dim)
        self.output_projection = nn.Linear(
            embedding_dim, embedding_dim
        )

        self.attention_dropout = nn.Dropout(dropout)
        self.residual_dropout = nn.Dropout(dropout)

    def forward(self, x):
        batch_size, seq_len, embedding_dim = x.shape

        # [B, T, 3C] -> three tensors of [B, H, T, D].
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.reshape(
            batch_size, seq_len, self.num_heads, self.head_dim
        ).transpose(1, 2)

        k = k.reshape(
            batch_size, seq_len, self.num_heads, self.head_dim
        ).transpose(1, 2)

        v = v.reshape(
            batch_size, seq_len, self.num_heads, self.head_dim
        ).transpose(1, 2)

        # Attention scores: [B, H, T, T].
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # True indicates a future position that must be masked.
        causal_mask = torch.ones(
            seq_len, seq_len,
            device=x.device,
            dtype=torch.bool,
        ).triu(diagonal=1)

        scores = scores.masked_fill(
            causal_mask[None, None, :, :],
            float("-inf"),
        )

        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.attention_dropout(
            attention_weights
        )

        context = attention_weights @ v

        # [B, H, T, D] -> [B, T, C].
        context = context.transpose(1, 2).contiguous().reshape(
            batch_size, seq_len, embedding_dim
        )

        output = self.output_projection(context)
        return self.residual_dropout(output)


class TransformerBlock(nn.Module):
    def __init__(
        self,
        embedding_dim,
        num_heads,
        feed_forward_dim,
        dropout=0.1,
    ):
        super().__init__()

        self.ln1 = nn.LayerNorm(embedding_dim)
        self.attention = CausalSelfAttention(
            embedding_dim, num_heads, dropout
        )

        self.ln2 = nn.LayerNorm(embedding_dim)

        self.feed_forward = nn.Sequential(
            nn.Linear(embedding_dim, feed_forward_dim),
            nn.GELU(),
            nn.Linear(feed_forward_dim, embedding_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x = x + self.attention(self.ln1(x))
        x = x + self.feed_forward(self.ln2(x))
        return x


class CausalTransformerLanguageModel(nn.Module):
    def __init__(
        self,
        vocab_size,
        embedding_dim=128,
        num_heads=4,
        num_layers=2,
        feed_forward_dim=512,
        max_seq_len=64,
        dropout=0.1,
        pad_id=0,
    ):
        super().__init__()

        if vocab_size <= 0:
            raise ValueError("vocab_size must be positive.")
        if max_seq_len <= 0:
            raise ValueError("max_seq_len must be positive.")
        if num_layers <= 0:
            raise ValueError("num_layers must be positive.")
        if not 0 <= pad_id < vocab_size:
            raise ValueError("Invalid pad_id.")

        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len

        self.token_embedding = nn.Embedding(
            vocab_size, embedding_dim, padding_idx=pad_id
        )

        self.position_embedding = nn.Embedding(
            max_seq_len, embedding_dim
        )

        self.embedding_dropout = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            TransformerBlock(
                embedding_dim,
                num_heads,
                feed_forward_dim,
                dropout,
            )
            for _ in range(num_layers)
        ])

        self.final_norm = nn.LayerNorm(embedding_dim)
        self.output_layer = nn.Linear(embedding_dim, vocab_size)

    def forward(self, input_ids):
        if input_ids.ndim != 2:
            raise ValueError("Expected [batch, seq_len] token IDs.")

        batch_size, seq_len = input_ids.shape

        if seq_len == 0 or seq_len > self.max_seq_len:
            raise ValueError("Invalid sequence length.")

        positions = torch.arange(
            seq_len, device=input_ids.device
        )

        x = (
            self.token_embedding(input_ids)
            + self.position_embedding(positions)[None, :, :]
        )

        x = self.embedding_dropout(x)

        for block in self.blocks:
            x = block(x)

        x = self.final_norm(x)
        return self.output_layer(x)

    def num_parameters(self):
        return sum(
            p.numel() for p in self.parameters()
            if p.requires_grad
        )
