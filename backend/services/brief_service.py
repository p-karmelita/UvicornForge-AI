from __future__ import annotations

import os
import re
from typing import Optional

import requests
from pydantic import BaseModel

from ml.brief_generator import generate_dataset_brief
from ml.dataset import find_similar_startup_rows, get_dataset_info
from ml.feature_mapper import map_request_to_features
from ml.predictor import PredictionResult, SuccessPredictor
from ml.prompts import build_hackathon_prompt, row_to_description


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
    success_score: Optional[float] = None
    success_score_normalized: Optional[float] = None
    success_label: Optional[str] = None
    similar_startups: Optional[list[SimilarStartup]] = None
    score_factors: Optional[dict[str, str]] = None
    model_source: Optional[str] = None
    llm_source: Optional[str] = None


SECTION_LABELS = [
    ("project_name", "Project name"),
    ("one_sentence_pitch", "One-sentence pitch"),
    ("problem", "Problem"),
    ("solution", "Solution"),
    ("target_market", "Target market"),
    ("mvp_scope", "MVP scope"),
    ("key_features", "Key features"),
    ("demo_scenario", "Demo scenario"),
    ("business_model", "Business model"),
    ("why_it_can_win", "Why this project can win a hackathon"),
]


class BriefService:
    def __init__(self, predictor: Optional[SuccessPredictor] = None):
        self.predictor = predictor or SuccessPredictor()
        self.xai_api_key = os.getenv("XAI_API_KEY")
        self.xai_api_base = "https://api.x.ai/v1"
        self.xai_model = os.getenv("XAI_MODEL", "grok-4.5")

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
            "xai_configured": bool(self.xai_api_key),
            "xai_model": self.xai_model,
            "training_samples": training_meta.get("training_samples"),
            "final_val_mse": training_meta.get("final_val_mse"),
            "scaler_saved": self.predictor.scaler is not None,
        }

    def generate(self, payload: GenerateBriefRequest) -> GenerateBriefResponse:
        prediction = self.predictor.predict_from_payload(
            project_idea=payload.project_idea,
            target_users=payload.target_users,
            industry=payload.industry,
            available_time=payload.available_time,
            available_technologies=payload.available_technologies,
        )
        prompt = self._build_prompt(payload)

        grok_text = self._call_grok(prompt)
        if grok_text:
            brief = self._parse_brief_from_text(grok_text, payload)
            return self._attach_prediction(brief, prediction, "grok")

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
        references = [
            row_to_description(row)
            for row in find_similar_startup_rows(mapped.industry, mapped.tech_stack, limit=3)
        ]
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

    def _parse_brief_from_text(self, text: str, payload: GenerateBriefRequest) -> GenerateBriefResponse:
        sections = self._extract_labeled_sections(text)
        if len(sections) < 4:
            sections.update(self._extract_numbered_sections(text))

        idea = payload.project_idea.strip()
        return GenerateBriefResponse(
            project_name=sections.get("project_name", f"{idea} — UnicornForge AI"),
            one_sentence_pitch=sections.get("one_sentence_pitch", f"{idea} — AI-powered startup brief."),
            problem=sections.get("problem", "Teams need to pitch quickly."),
            solution=sections.get("solution", "Generate a structured startup brief instantly."),
            target_market=sections.get("target_market", self._norm_field(payload.target_users, "Hackathon teams")),
            mvp_scope=sections.get("mvp_scope", "Single-page web app for brief generation."),
            key_features=sections.get(
                "key_features",
                "- Idea-to-brief generation\n- Clear problem & solution framing",
            ),
            demo_scenario=sections.get("demo_scenario", "1) Enter idea. 2) Generate. 3) Copy."),
            business_model=sections.get("business_model", "Freemium SaaS."),
            why_it_can_win=sections.get("why_it_can_win", "Helps teams pitch quickly using AMD GPUs."),
        )

    def _extract_labeled_sections(self, text: str) -> dict[str, str]:
        sections: dict[str, str] = {}
        label_patterns = "|".join(re.escape(label) for _, label in SECTION_LABELS)
        chunks = re.split(rf"(?=(?:{label_patterns})\s*:)", text, flags=re.IGNORECASE)

        for chunk in chunks:
            chunk = chunk.strip()
            if not chunk:
                continue
            for key, label in SECTION_LABELS:
                prefix = f"{label}:"
                if chunk.lower().startswith(prefix.lower()):
                    sections[key] = chunk[len(label) + 1 :].strip()
                    break
        return sections

    def _extract_numbered_sections(self, text: str) -> dict[str, str]:
        mapping = {
            1: "project_name",
            2: "one_sentence_pitch",
            3: "problem",
            4: "solution",
            5: "target_market",
            6: "mvp_scope",
            7: "key_features",
            8: "demo_scenario",
            9: "business_model",
            10: "why_it_can_win",
        }
        sections: dict[str, str] = {}
        matches = list(re.finditer(r"(?m)^\s*(\d{1,2})\.\s*(.+)$", text))
        for index, match in enumerate(matches):
            number = int(match.group(1))
            start = match.end()
            end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
            content = text[start:end].strip()
            key = mapping.get(number)
            if key and content:
                sections[key] = content
        return sections

    def _call_grok(self, prompt: str) -> Optional[str]:
        if not self.xai_api_key:
            return None

        for endpoint, payload in (
            ("responses", {"model": self.xai_model, "input": prompt}),
            (
                "chat/completions",
                {
                    "model": self.xai_model,
                    "messages": [{"role": "user", "content": prompt}],
                },
            ),
        ):
            text = self._post_xai(endpoint, payload)
            if text:
                return text
        return None

    def _post_xai(self, endpoint: str, payload: dict) -> Optional[str]:
        try:
            response = requests.post(
                f"{self.xai_api_base}/{endpoint}",
                headers={
                    "Authorization": f"Bearer {self.xai_api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
                timeout=120,
            )
            response.raise_for_status()
            data = response.json()

            if endpoint == "responses":
                output_text = data.get("output_text")
                if isinstance(output_text, str) and output_text.strip():
                    return output_text
                for item in data.get("output", []):
                    if item.get("type") != "message":
                        continue
                    for content in item.get("content", []):
                        if content.get("type") == "output_text" and content.get("text"):
                            return content["text"]

            choices = data.get("choices", [])
            if choices:
                message = choices[0].get("message", {})
                content = message.get("content")
                if isinstance(content, str) and content.strip():
                    return content
        except Exception as exc:
            print(f"[unicornforge] xAI {endpoint} failed: {exc}")
        return None

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
        return response.model_copy(update=updates)