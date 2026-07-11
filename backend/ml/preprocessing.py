from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from .dataset import CAT_COLUMNS, NUM_COLUMNS, load_dataset


def fit_scaler() -> Optional[StandardScaler]:
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
    """Replicate notebook preprocessing: one-hot encoding + numeric scaling."""
    frame = pd.DataFrame([row])
    encoded = pd.get_dummies(frame, columns=CAT_COLUMNS, drop_first=True)

    for col in feature_columns:
        if col not in encoded.columns:
            encoded[col] = 0.0

    encoded = encoded[feature_columns].astype("float32")

    if scaler is not None:
        encoded[NUM_COLUMNS] = scaler.transform(encoded[NUM_COLUMNS])

    return encoded.values.astype("float32").reshape(-1)