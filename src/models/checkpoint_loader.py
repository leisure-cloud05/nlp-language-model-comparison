"""Load and freeze trained Part I language models."""

from pathlib import Path

import torch

from src.models.lstm import LSTMLanguageModel
from src.models.transformer import CausalTransformerLanguageModel


ROOT = Path(__file__).resolve().parents[2]


def load_frozen_lstm(device):
    checkpoint_path = ROOT / "checkpoints" / "lstm_best.pt"

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )

    config = checkpoint["config"]

    model = LSTMLanguageModel(
        vocab_size=checkpoint["vocab_size"],
        embedding_dim=config["embedding_dim"],
        hidden_dim=config["hidden_dim"],
        num_layers=config["num_layers"],
        pad_id=checkpoint["pad_id"],
    )

    model.load_state_dict(checkpoint["model_state_dict"])

    # Freeze all pretrained language-model parameters.
    model.requires_grad_(False)

    model = model.to(device)
    model.eval()

    return model


def load_frozen_transformer(device):
    checkpoint_path = ROOT / "checkpoints" / "transformer_best.pt"

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )

    config = checkpoint["config"]

    model = CausalTransformerLanguageModel(
        vocab_size=checkpoint["vocab_size"],
        embedding_dim=config["embedding_dim"],
        num_heads=config["num_heads"],
        num_layers=config["num_layers"],
        feed_forward_dim=config["feed_forward_dim"],
        max_seq_len=config["max_seq_len"],
        dropout=config["dropout"],
        pad_id=checkpoint["pad_id"],
    )

    model.load_state_dict(checkpoint["model_state_dict"])

    model.requires_grad_(False)

    model = model.to(device)
    model.eval()

    return model
