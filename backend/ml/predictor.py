from __future__ import annotations

import os
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from .dataset import TARGET_COLUMN, find_similar_startups
from .feature_mapper import MappedFeatures, map_request_to_features
from .model import SuccessScoreMLP
from .preprocessing import fit_scaler, row_to_vector

try:
    import torch
except Exception:  # pragma: no cover
    torch = None


@dataclass
class PredictionResult:
    score: float
    score_normalized: float
    label: str
    factors: dict[str, str]
    similar_startups: list[dict[str, str]]
    model_source: str


def _score_label(score: float) -> str:
    # Adjusted for new dataset (Success Score 1-10 scale)
    if score >= 8.0:
        return "Very strong potential"
    if score >= 6.5:
        return "Strong potential"
    if score >= 5.0:
        return "Moderate potential"
    if score >= 3.5:
        return "Early-stage risk"
    return "High risk"


class SuccessPredictor:
    def __init__(self, models_dir: Optional[Path] = None):
        self.models_dir = models_dir or Path(__file__).resolve().parent.parent / "trained_models" / "startup_success_mlp"
        self.model = None
        self.feature_columns: list[str] = []
        self.scaler = None
        self.training_metadata: dict = {}
        self.ready = False
        self._load()

    def _load(self) -> None:
        if torch is None:
            return

        metadata_path = self.models_dir / "metadata.pkl"
        weights_path = self.models_dir / "model.pt"
        if not metadata_path.exists() or not weights_path.exists():
            return

        try:
            with open(metadata_path, "rb") as handle:
                metadata = pickle.load(handle)

            self.training_metadata = metadata if isinstance(metadata, dict) else {}
            self.feature_columns = list(metadata.get("feature_columns", []))
            if not self.feature_columns:
                return

            saved_scaler = metadata.get("scaler")
            self.scaler = saved_scaler if saved_scaler is not None else fit_scaler()

            input_dim = len(self.feature_columns)
            model = SuccessScoreMLP(input_dim)
            state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
            model.load_state_dict(state_dict)
            model.eval()

            device = self._resolve_device()
            self.model = model.to(device)
            self.device = device
            self.ready = True
        except Exception as exc:
            print(f"[unicornforge] Failed to load success model: {exc}")
            self.ready = False

    def _resolve_device(self) -> str:
        if torch is None:
            return "cpu"
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"

    def predict_from_payload(
        self,
        project_idea: str,
        target_users: Optional[str] = None,
        industry: Optional[str] = None,
        available_time: Optional[str] = None,
        available_technologies: Optional[str] = None,
        compute_platform: Optional[str] = None,
        amd_platform: Optional[str] = None,
    ) -> Optional[PredictionResult]:
        mapped = map_request_to_features(
            project_idea=project_idea,
            target_users=target_users,
            industry=industry,
            available_time=available_time,
            available_technologies=available_technologies,
            compute_platform=compute_platform,
            amd_platform=amd_platform,
        )
        return self.predict_from_mapped(mapped)

    def predict_from_mapped(self, mapped: MappedFeatures) -> Optional[PredictionResult]:
        if not self.ready or self.model is None or torch is None:
            return None

        try:
            vector = row_to_vector(mapped.row, self.feature_columns, self.scaler)
            tensor = torch.from_numpy(vector).unsqueeze(0).to(self.device)

            with torch.no_grad():
                raw = float(self.model(tensor).squeeze().cpu().numpy())

            score = float(np.clip(raw, 1.0, 10.0))
            normalized = (score - 1.0) / 9.0  # adjusted for 1-10 range
            similar = find_similar_startups(mapped.industry, mapped.tech_stack)

            return PredictionResult(
                score=round(score, 2),
                score_normalized=round(normalized, 3),
                label=_score_label(score),
                factors=mapped.factors,
                similar_startups=similar,
                model_source="pytorch_mlp",
            )
        except Exception as exc:
            print(f"[unicornforge] Success prediction failed: {exc}")
            return None