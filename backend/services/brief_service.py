from __future__ import annotations

from typing import Optional

from pydantic import BaseModel

from ml.brief_generator import generate_dataset_brief
from ml.dataset import find_similar_startup_rows, get_dataset_info
from ml.feature_mapper import map_request_to_features
from ml.predictor import PredictionResult, SuccessPredictor
from ml.prompts import build_hackathon_prompt
from services.fireworks_client import FireworksClient


class GenerateBriefRequest(BaseModel):
    project_idea: str
    target_users: Optional[str] = None
    industry: Optional[str] = None
    available_time: Optional[str] = None
    available_technologies: Optional[str] = None


class SimilarStartup(BaseModel):
    name: str
    industry: str
    score: str


class GenerateBriefResponse(BaseModel):
    project_name: str
    one_sentence_pitch: str
    problem: str
    solution: str
    target_market: str
    mvp_scope: str
    key_features: str
    demo_scenario: str
    business_model: str
    why_it_can_win: str
    # High business value additions
    risks_and_mitigations: Optional[str] = None
    go_to_market: Optional[str] = None
    key_metrics_to_track: Optional[str] = None
    recommended_next_steps: Optional[str] = None
    hackathon_tips: Optional[str] = None
    improvement_suggestions: Optional[str] = None   # High-value: actionable advice from the model + factors

    success_score: Optional[float] = None
    success_score_normalized: Optional[float] = None
    success_label: Optional[str] = None
    similar_startups: Optional[list[SimilarStartup]] = None
    score_factors: Optional[dict[str, str]] = None
    model_source: Optional[str] = None
    llm_source: Optional[str] = None


class BriefService:
    def __init__(
        self,
        predictor: Optional[SuccessPredictor] = None,
        fireworks_client: Optional[FireworksClient] = None,
    ):
        self.predictor = predictor or SuccessPredictor()
        self.fireworks = fireworks_client or FireworksClient()

    def get_model_info(self) -> dict:
        dataset = get_dataset_info()
        try:
            import torch

            torch_available = True
            cuda_available = torch.cuda.is_available()
            device_name = torch.cuda.get_device_name(0) if cuda_available else None
        except Exception:
            torch_available = False
            cuda_available = False
            device_name = None

        training_meta = getattr(self.predictor, "training_metadata", {})
        return {
            "success_model_ready": self.predictor.ready,
            "feature_columns": len(self.predictor.feature_columns),
            "dataset_loaded": dataset["loaded"],
            "dataset_rows": dataset["rows"],
            "dataset_path": dataset["path"],
            "torch_available": torch_available,
            "cuda_available": cuda_available,
            "device_name": device_name,
            "fireworks_configured": self.fireworks.configured,
            "fireworks_key_format_valid": self.fireworks.key_format_valid,
            "fireworks_model": self.fireworks.model,
            "fireworks_last_error": self.fireworks.last_error,
            "fireworks_help": self.fireworks.status().get("help"),
            "training_samples": training_meta.get("training_samples"),
            "validation_samples": training_meta.get("validation_samples"),
            "final_val_mse": training_meta.get("final_val_mse"),
            "val_mae": training_meta.get("val_mae"),
            "val_rmse": training_meta.get("val_rmse"),
            "val_r2": training_meta.get("val_r2"),
            "scaler_saved": self.predictor.scaler is not None,
        }

    def generate(self, payload: GenerateBriefRequest) -> GenerateBriefResponse:
        # Honest defaults: only apply strong AMD/Fireworks when user explicitly mentions them.
        # This keeps the success score credible and high-business-value (reward real choices).
        techs = (payload.available_technologies or "").lower()
        mentions_amd = any(k in techs for k in ["amd", "mi300", "mi250", "rocm", "instinct"])
        mentions_fw = "fireworks" in techs

        if mentions_amd:
            comp = "Own AMD GPU cluster"
            amd = "AMD Instinct MI300X" if "mi300" in techs or "300" in techs else "AMD Instinct MI250"
        elif mentions_fw:
            comp = "Fireworks AI API"
            amd = "—"
        else:
            # Neutral realistic default for a hackathon team
            comp = "Cloud GPU (generic)"
            amd = "—"

        prediction = self.predictor.predict_from_payload(
            project_idea=payload.project_idea,
            target_users=payload.target_users,
            industry=payload.industry,
            available_time=payload.available_time,
            available_technologies=payload.available_technologies,
            compute_platform=comp,
            amd_platform=amd,
        )
        prompt = self._build_prompt(payload)

        fireworks_brief = self.fireworks.generate_brief(prompt)
        if fireworks_brief is not None:
            brief = GenerateBriefResponse(**fireworks_brief.model_dump())
            return self._attach_prediction(brief, prediction, "fireworks")

        brief = self._generate_dataset_brief(payload)
        return self._attach_prediction(brief, prediction, "dataset")

    def _norm_field(self, value: Optional[str], default: str = "not specified") -> str:
        return value.strip() if value and value.strip() else default

    def _build_prompt(self, payload: GenerateBriefRequest) -> str:
        mapped = map_request_to_features(
            project_idea=payload.project_idea,
            target_users=payload.target_users,
            industry=payload.industry,
            available_time=payload.available_time,
            available_technologies=payload.available_technologies,
        )
        # Use short, non-leaking pattern summaries for the LLM prompt
        # (raw row_to_description contains unrealistic scale and synthetic names)
        raw_refs = find_similar_startup_rows(mapped.industry, mapped.tech_stack, limit=3)
        references = []
        for r in raw_refs:
            tech = str(r.get("Backend Tech Stack", "") or r.get("Tech Stack", "")).strip()
            stage = str(r.get("Funding Stage", "")).strip()
            score = float(r.get("Success Score", 0))
            summary = f"High-scoring {mapped.industry} example"
            if tech:
                summary += f" using {tech.split(',')[0]}"
            if stage:
                summary += f" at {stage} stage"
            if score:
                summary += f" (score {score:.1f}/10)"
            references.append(summary)

        return build_hackathon_prompt(
            project_idea=payload.project_idea.strip(),
            target_users=self._norm_field(payload.target_users),
            industry=self._norm_field(payload.industry, mapped.industry),
            available_time=self._norm_field(payload.available_time),
            available_technologies=self._norm_field(payload.available_technologies, mapped.tech_stack),
            reference_profiles=references,
        )

    def _generate_dataset_brief(self, payload: GenerateBriefRequest) -> GenerateBriefResponse:
        return GenerateBriefResponse(
            **generate_dataset_brief(
                project_idea=payload.project_idea,
                target_users=payload.target_users,
                industry=payload.industry,
                available_time=payload.available_time,
                available_technologies=payload.available_technologies,
            )
        )

    def _attach_prediction(
        self,
        response: GenerateBriefResponse,
        prediction: Optional[PredictionResult],
        llm_source: str,
    ) -> GenerateBriefResponse:
        updates: dict = {"llm_source": llm_source}
        if prediction is not None:
            updates.update(
                {
                    "success_score": prediction.score,
                    "success_score_normalized": prediction.score_normalized,
                    "success_label": prediction.label,
                    "score_factors": prediction.factors,
                    "model_source": prediction.model_source,
                    "similar_startups": [
                        SimilarStartup(**item) for item in prediction.similar_startups
                    ],
                }
            )

            # High-business-value: synthesize concrete improvement suggestions
            suggestions = self._build_improvement_suggestions(prediction, response)
            if suggestions:
                updates["improvement_suggestions"] = suggestions

        return response.model_copy(update=updates)

    def _build_improvement_suggestions(self, prediction: "PredictionResult", brief: GenerateBriefResponse) -> str:
        tips = []
        factors = prediction.factors or {}
        score = prediction.score or 6.0

        if "AMD" not in str(factors.get("compute_platform", "")) and "AMD" not in str(factors.get("amd_platform", "")):
            tips.append("Explicitly use AMD GPUs / ROCm in your demo and tech description — it gives a measurable lift in our model.")
        if factors.get("team_size") and float(str(factors.get("team_size", "3")).split("-")[0] or 3) < 4:
            tips.append("Consider highlighting (or increasing) team size / complementary skills — small teams score lower on average.")
        if "Fireworks" not in str(brief.key_features or "") and "Fireworks" not in str(factors.get("tech_stack", "")):
            tips.append("Mention Fireworks AI in your architecture or LLM usage for better sponsor alignment and score signal.")
        if score < 7.0:
            tips.append("Strengthen the business model and 'why it wins' sections — these correlate with higher outcomes in the dataset.")
        if not tips:
            tips.append("Focus on a killer, time-boxed demo and crisp 60-second pitch. Your current signals are already solid.")

        return " • ".join(tips)