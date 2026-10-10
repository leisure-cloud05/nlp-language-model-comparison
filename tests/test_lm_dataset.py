
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader


# Allow imports from the project root
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.lm_dataset import (
    LanguageModelDataset,
    load_documents,
    load_vocabulary,
    make_collate_fn,
)


def test_synthetic_samples():
    """Check next-token pairs and document boundaries."""
    documents = [
        [10, 11, 12, 13, 14],
        [20, 21, 22],
    ]

    dataset = LanguageModelDataset(documents, seq_len=3)

    assert len(dataset) == 3

    expected = [
        ([10, 11, 12], [11, 12, 13]),
        ([13], [14]),
        ([20, 21], [21, 22]),
    ]

    for index, (expected_x, expected_y) in enumerate(expected):
        x, y = dataset[index]

        assert x.tolist() == expected_x
        assert y.tolist() == expected_y

    expected_targets = (
        documents[0][1:] + documents[1][1:]
    )

    actual_targets = []

    for _, targets in dataset:
        actual_targets.extend(targets.tolist())

    assert actual_targets == expected_targets

    print("Synthetic next-token samples: PASS")
    print("Document boundaries: PASS")
    print("Target coverage: PASS")


def test_padding():
    """Check batch padding and ignored labels."""
    documents = [
        [10, 11, 12, 13, 14],
        [20, 21, 22],
    ]

    pad_id = 0

    dataset = LanguageModelDataset(documents, seq_len=3)

    loader = DataLoader(
        dataset,
        batch_size=3,
        shuffle=False,
        collate_fn=make_collate_fn(pad_id),
    )

    batch = next(iter(loader))

    input_ids = batch["input_ids"]
    labels = batch["labels"]
    attention_mask = batch["attention_mask"]

    assert input_ids.shape == labels.shape
    assert input_ids.shape == attention_mask.shape

    assert torch.all(labels[~attention_mask] == -100)
    assert torch.all(input_ids[~attention_mask] == pad_id)

    assert torch.all(labels[attention_mask] != -100)

    print("Padding and attention mask: PASS")


def test_real_data():
    """Check vocabulary and real WikiText-2 samples."""
    vocab = load_vocabulary()

    assert len(vocab) == 20000
    assert vocab["<pad>"] == 0
    assert vocab["<unk>"] == 1
    assert vocab["<eos>"] == 2

    print(f"Vocabulary size: {len(vocab)}")

    for split in ["train", "validation", "test"]:
        documents = load_documents(split)

        assert len(documents) > 0

        assert all(
            0 <= token_id < len(vocab)
            for doc in documents
            for token_id in doc
        )

        assert all(
            vocab["<pad>"] not in doc
            for doc in documents
        )

        dataset = LanguageModelDataset(
            documents,
            seq_len=64,
        )

        expected_targets = sum(
            max(0, len(doc) - 1)
            for doc in documents
        )

        actual_targets = sum(
            len(targets)
            for _, targets in dataset
        )

        assert actual_targets == expected_targets

        print(
            f"{split}: "
            f"{len(documents)} documents, "
            f"{len(dataset)} samples, "
            f"{actual_targets} targets - PASS"
        )


def main():
    test_synthetic_samples()
    test_padding()
    test_real_data()

    print("\nAll dataset tests passed!")


if __name__ == "__main__":
    main()
