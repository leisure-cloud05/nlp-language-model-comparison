"""Train an MLP classifier on frozen IMDb language-model features."""

import argparse
import copy
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.models.imdb_mlp import IMDbMLP


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "processed"

SEED = 42


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        choices=["lstm_mean", "transformer"],
        required=True,
    )
    args = parser.parse_args()

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    device = torch.device(
        "mps" if torch.backends.mps.is_available() else "cpu"
    )

    with np.load(
        DATA_DIR / f"imdb_{args.model}_train_features.npz"
    ) as data:
        X_train = data["X"].copy()
        y_train = data["y"].copy()

    with np.load(
        DATA_DIR / f"imdb_{args.model}_validation_features.npz"
    ) as data:
        X_val = data["X"].copy()
        y_val = data["y"].copy()

    # Fit scaler ONLY on training data.
    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train).astype(np.float32)
    X_val = scaler.transform(X_val).astype(np.float32)

    train_dataset = TensorDataset(
        torch.from_numpy(X_train),
        torch.from_numpy(y_train).long(),
    )

    loader = DataLoader(
        train_dataset,
        batch_size=128,
        shuffle=True,
        num_workers=0,
    )

    X_val_tensor = torch.from_numpy(X_val).to(device)

    model = IMDbMLP(
        input_dim=X_train.shape[1],
        hidden_dim=128,
        dropout=0.2,
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-3,
        weight_decay=1e-4,
    )

    criterion = nn.CrossEntropyLoss()

    best_f1 = -1.0
    best_epoch = None
    best_state = None

    patience = 5
    no_improvement = 0

    for epoch in range(1, 51):
        model.train()

        total_loss = 0.0

        for features, labels in loader:
            features = features.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            logits = model(features)
            loss = criterion(logits, labels)

            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(labels)

        train_loss = total_loss / len(train_dataset)

        model.eval()

        with torch.inference_mode():
            logits = model(X_val_tensor)
            predictions = logits.argmax(dim=1).cpu().numpy()

        val_accuracy = accuracy_score(y_val, predictions)
        val_f1 = f1_score(
            y_val,
            predictions,
            average="macro",
        )

        print(
            f"Epoch {epoch:2d} "
            f"Train Loss={train_loss:.4f} "
            f"Val Accuracy={val_accuracy:.4f} "
            f"Val Macro F1={val_f1:.4f}"
        )

        if val_f1 > best_f1:
            best_f1 = val_f1
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            no_improvement = 0
        else:
            no_improvement += 1

        if no_improvement >= patience:
            print("Early stopping triggered.")
            break

    model.load_state_dict(best_state)

    print("\nModel:", args.model)
    print("Best epoch:", best_epoch)
    print("Best Validation Macro F1:", round(best_f1, 6))
    print("Best model restored.")

    # First experiment: do not save or overwrite any files.
    print("Experimental run completed. No files saved.")


if __name__ == "__main__":
    main()
