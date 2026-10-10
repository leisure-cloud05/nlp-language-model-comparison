import json
from pathlib import Path

import torch
from torch.utils.data import Dataset


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "processed"


def load_documents(split):
    """Load preprocessed token-ID documents."""
    if split not in {"train", "validation", "test"}:
        raise ValueError(f"Invalid split: {split}")

    path = DATA_DIR / f"{split}.json"

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_vocabulary():
    """Load the shared vocabulary."""
    path = DATA_DIR / "vocab.json"

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


class LanguageModelDataset(Dataset):
    """
    Create fixed-window next-token prediction samples.

    - Never cross document boundaries.
    - Predict each token after the first exactly once.
    - Keep the final short window.
    """

    def __init__(self, documents, seq_len=64):
        if seq_len < 1:
            raise ValueError("seq_len must be >= 1")

        self.seq_len = seq_len
        self.samples = []

        for doc in documents:
            if len(doc) < 2:
                continue

            # Each window contains at most seq_len targets.
            for start in range(0, len(doc) - 1, seq_len):
                end = min(start + seq_len, len(doc) - 1)

                inputs = doc[start:end]
                targets = doc[start + 1:end + 1]

                self.samples.append((inputs, targets))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        inputs, targets = self.samples[index]

        return (
            torch.tensor(inputs, dtype=torch.long),
            torch.tensor(targets, dtype=torch.long),
        )


def make_collate_fn(pad_id):
    """Pad variable-length samples for batching."""

    def collate_fn(batch):
        inputs, targets = zip(*batch)

        input_ids = torch.nn.utils.rnn.pad_sequence(
            inputs,
            batch_first=True,
            padding_value=pad_id,
        )

        labels = torch.nn.utils.rnn.pad_sequence(
            targets,
            batch_first=True,
            padding_value=-100,
        )

        attention_mask = input_ids.ne(pad_id)

        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
        }

    return collate_fn
