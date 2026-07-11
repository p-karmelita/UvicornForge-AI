from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from .dataset import CAT_COLUMNS, NUM_COLUMNS, load_dataset


def fit_scaler() -> Optional[StandardScaler]:
    """Fit scaler on the new AMD-aware dataset columns (Team Size, AMD Platform, Fireworks credits, etc.)."""
    df = load_dataset()
    if df is None:
        return None

    scaler = StandardScaler()
    scaler.fit(df[NUM_COLUMNS].astype("float32"))
    return scaler


def row_to_vector(
    row: dict[str, object],
    feature_columns: list[str],
    scaler: Optional[StandardScaler],
) -> np.ndarray:
    """
    Convert a (partial) user input row into the vector expected by the SuccessScoreMLP.
    Designed for the new dataset that emphasizes AMD sponsor technologies:
    - Compute Platform / AMD Platform Used
    - Tech stacks (Backend + Frontend)
    - Fireworks credits, realistic Team Size, MRR, etc.
    """
    frame = pd.DataFrame([row])
    encoded = pd.get_dummies(frame, columns=CAT_COLUMNS, drop_first=True)

    # Ensure every column the model was trained on is present
    for col in feature_columns:
        if col not in encoded.columns:
            encoded[col] = 0.0

    encoded = encoded[feature_columns].astype("float32")

    if scaler is not None:
        numeric_cols = [c for c in NUM_COLUMNS if c in encoded.columns]
        if numeric_cols:
            encoded[numeric_cols] = scaler.transform(encoded[numeric_cols])

    return encoded.values.astype("float32").reshape(-1)