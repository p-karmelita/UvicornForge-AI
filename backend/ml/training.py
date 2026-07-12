from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .dataset import CAT_COLUMNS, DEFAULT_DATASET_PATH, NUM_COLUMNS, TARGET_COLUMN
from .evaluation import evaluate_success_model
from .model import SuccessScoreMLP


class StartupsDataset(Dataset):
    def __init__(self, features: np.ndarray, targets: np.ndarray):
        self.features = torch.from_numpy(features).float()
        self.targets = torch.from_numpy(targets).float()

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, index: int):
        return self.features[index], self.targets[index]


def prepare_training_frame(
    dataset_path: Path | None = None,
) -> tuple[pd.DataFrame, pd.Series, list[str], StandardScaler]:
    path = dataset_path or DEFAULT_DATASET_PATH
    df = pd.read_csv(path)
    used_cols = CAT_COLUMNS + NUM_COLUMNS + [TARGET_COLUMN]
    df = df[used_cols].dropna().reset_index(drop=True)

    encoded = pd.get_dummies(df, columns=CAT_COLUMNS, drop_first=True)
    feature_columns = [col for col in encoded.columns if col != TARGET_COLUMN]

    features = encoded[feature_columns].astype("float32")
    targets = encoded[TARGET_COLUMN].astype("float32")

    scaler = StandardScaler()
    features[NUM_COLUMNS] = scaler.fit_transform(features[NUM_COLUMNS])
    return features, targets, feature_columns, scaler


def train_success_model(
    dataset_path: Path | None = None,
    output_dir: Path | None = None,
    epochs: int = 20,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    random_state: int = 42,
) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    features, targets, feature_columns, scaler = prepare_training_frame(dataset_path)

    x_train, x_val, y_train, y_val = train_test_split(
        features.values,
        targets.values,
        test_size=0.2,
        random_state=random_state,
    )

    train_loader = DataLoader(StartupsDataset(x_train, y_train), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(StartupsDataset(x_val, y_val), batch_size=256)

    model = SuccessScoreMLP(x_train.shape[1], dropout=0.05).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=4
    )

    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        train_count = 0
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad()
            preds = model(batch_x)
            loss = criterion(preds, batch_y)
            loss.backward()
            optimizer.step()
            batch_size_actual = batch_x.size(0)
            train_loss += loss.item() * batch_size_actual
            train_count += batch_size_actual

        val_loss = _evaluate(model, val_loader, criterion, device)
        scheduler.step(val_loss)

        epoch_stats = {
            "epoch": epoch,
            "train_mse": train_loss / train_count,
            "val_mse": val_loss,
        }
        history.append(epoch_stats)
        print(
            f"Epoch {epoch:02d} | train MSE: {epoch_stats['train_mse']:.4f} "
            f"| val MSE: {epoch_stats['val_mse']:.4f}"
        )

    _, _, val_metrics = evaluate_success_model(model, val_loader, device)

    out_dir = output_dir or Path(__file__).resolve().parent.parent / "trained_models" / "startup_success_mlp"
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.save(model.state_dict(), out_dir / "model.pt")
    metadata = {
        "feature_columns": feature_columns,
        "num_columns": NUM_COLUMNS,
        "cat_columns": CAT_COLUMNS,
        "target_column": TARGET_COLUMN,
        "scaler": scaler,
        "training_samples": int(len(features)),
        "validation_samples": int(len(y_val)),
        "device": str(device),
        "final_val_mse": history[-1]["val_mse"],
        "val_mae": val_metrics["mae"],
        "val_rmse": val_metrics["rmse"],
        "val_r2": val_metrics["r2"],
        "history": history,
    }
    with open(out_dir / "metadata.pkl", "wb") as handle:
        pickle.dump(metadata, handle)

    return {
        "output_dir": str(out_dir),
        "feature_columns": len(feature_columns),
        "training_samples": len(features),
        "final_val_mse": history[-1]["val_mse"],
        "val_metrics": val_metrics,
        "device": str(device),
        "history": history,
    }


def _evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device) -> float:
    model.eval()
    total_loss = 0.0
    total_count = 0
    with torch.no_grad():
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            preds = model(batch_x)
            loss = criterion(preds, batch_y)
            batch_size_actual = batch_x.size(0)
            total_loss += loss.item() * batch_size_actual
            total_count += batch_size_actual
    return total_loss / total_count