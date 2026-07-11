from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch.utils.data import DataLoader

from .model import SuccessScoreMLP


def compute_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mse = float(mean_squared_error(y_true, y_pred))
    return {
        "mae": round(float(mean_absolute_error(y_true, y_pred)), 4),
        "rmse": round(float(np.sqrt(mse)), 4),
        "mse": round(mse, 4),
        "r2": round(float(r2_score(y_true, y_pred)), 4),
    }


def evaluate_success_model(
    model: SuccessScoreMLP,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    model.eval()
    predictions: list[float] = []
    actuals: list[float] = []

    with torch.no_grad():
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            preds = model(batch_x).cpu().numpy()
            predictions.extend(preds.tolist())
            actuals.extend(batch_y.numpy().tolist())

    y_true = np.asarray(actuals, dtype=float)
    y_pred = np.asarray(predictions, dtype=float)
    return y_true, y_pred, compute_regression_metrics(y_true, y_pred)


def evaluate_saved_model(
    models_dir: Optional[Path] = None,
    dataset_path: Optional[Path] = None,
) -> dict:
    import pickle

    from sklearn.model_selection import train_test_split

    from .training import StartupsDataset, prepare_training_frame

    models_dir = models_dir or Path(__file__).resolve().parent.parent / "trained_models" / "startup_success_mlp"
    metadata_path = models_dir / "metadata.pkl"
    weights_path = models_dir / "model.pt"

    if not metadata_path.exists() or not weights_path.exists():
        raise FileNotFoundError(f"Missing model artifacts in {models_dir}")

    with open(metadata_path, "rb") as handle:
        metadata = pickle.load(handle)

    feature_columns = metadata["feature_columns"]
    features, targets, _, _ = prepare_training_frame(dataset_path)
    x_values = features[feature_columns].values
    y_values = targets.values

    _, x_val, _, y_val = train_test_split(
        x_values,
        y_values,
        test_size=0.2,
        random_state=42,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SuccessScoreMLP(len(feature_columns))
    model.load_state_dict(torch.load(weights_path, map_location="cpu", weights_only=True))
    model = model.to(device)

    val_loader = DataLoader(StartupsDataset(x_val, y_val), batch_size=256)
    y_true, y_pred, metrics = evaluate_success_model(model, val_loader, device)

    return {
        "metrics": metrics,
        "samples": int(len(y_true)),
        "predictions": [
            {"actual": round(float(actual), 2), "predicted": round(float(pred), 2)}
            for actual, pred in zip(y_true[:5], y_pred[:5])
        ],
    }