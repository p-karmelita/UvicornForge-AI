from __future__ import annotations

import json
import os
from typing import Optional

import requests
from pydantic import BaseModel, Field


class FireworksBriefPayload(BaseModel):
    """Extended schema for high-business-value briefs (core + actionable sections)."""
    project_name: str = Field(description="Short product-like project name")
    one_sentence_pitch: str = Field(description="One concise sentence pitch")
    problem: str = Field(description="Problem statement")
    solution: str = Field(description="Proposed solution")
    target_market: str = Field(description="Target market description")
    mvp_scope: str = Field(description="MVP scope for hackathon timeline")
    key_features: str = Field(description="Key features, bullet-style allowed")
    demo_scenario: str = Field(description="Demo walkthrough for judges")
    business_model: str = Field(description="Business model sketch")
    why_it_can_win: str = Field(description="Why this can win a hackathon")

    # New high-value business sections (optional for backward + fallback)
    risks_and_mitigations: Optional[str] = Field(default=None, description="Key risks + mitigations")
    go_to_market: Optional[str] = Field(default=None, description="Go-to-market / launch strategy")
    key_metrics_to_track: Optional[str] = Field(default=None, description="Important metrics for first months")
    recommended_next_steps: Optional[str] = Field(default=None, description="30/60/90 day plan")
    hackathon_tips: Optional[str] = Field(default=None, description="Specific tips for winning the hackathon")

    # Rich starter documentation for immediate project kickoff
    recommended_tech_stack: Optional[str] = Field(default=None, description="Concrete tech recommendations with AMD/Fireworks")
    architecture_overview: Optional[str] = Field(default=None, description="High level architecture")
    project_structure: Optional[str] = Field(default=None, description="Suggested folder/file structure")
    mvp_checklist: Optional[str] = Field(default=None, description="Prioritized tasks and MVP checklist")
    starter_readme: Optional[str] = Field(default=None, description="Ready-to-use initial README.md content")
    pitch_outline: Optional[str] = Field(default=None, description="Pitch deck outline")
    demo_script: Optional[str] = Field(default=None, description="3-5 min demo script outline")


BRIEF_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "project_name": {"type": "string"},
        "one_sentence_pitch": {"type": "string"},
        "problem": {"type": "string"},
        "solution": {"type": "string"},
        "target_market": {"type": "string"},
        "mvp_scope": {"type": "string"},
        "key_features": {"type": "string"},
        "demo_scenario": {"type": "string"},
        "business_model": {"type": "string"},
        "why_it_can_win": {"type": "string"},
        "risks_and_mitigations": {"type": "string"},
        "go_to_market": {"type": "string"},
        "key_metrics_to_track": {"type": "string"},
        "recommended_next_steps": {"type": "string"},
        "hackathon_tips": {"type": "string"},
        "recommended_tech_stack": {"type": "string"},
        "architecture_overview": {"type": "string"},
        "project_structure": {"type": "string"},
        "mvp_checklist": {"type": "string"},
        "starter_readme": {"type": "string"},
        "pitch_outline": {"type": "string"},
        "demo_script": {"type": "string"},
    },
    "required": [
        "project_name",
        "one_sentence_pitch",
        "problem",
        "solution",
        "target_market",
        "mvp_scope",
        "key_features",
        "demo_scenario",
        "business_model",
        "why_it_can_win",
    ],
    "additionalProperties": False,
}


class FireworksClient:
    def __init__(self) -> None:
        # Support both common env var names for the Fireworks key
        self.api_key = (
            os.getenv("FIREWORKS_API_KEY", "").strip()
            or os.getenv("fireworks_key", "").strip()
        )
        self.api_base = os.getenv(
            "FIREWORKS_API_BASE", "https://api.fireworks.ai/inference/v1"
        ).rstrip("/")
        # Default to a capable model available on Fireworks; override via env if needed
        self.model = os.getenv(
            "FIREWORKS_MODEL", "accounts/fireworks/models/llama-v3p1-8b-instruct"
        )  # change to a model you have access to on Fireworks (e.g. qwen or llama)
        self.last_error: Optional[str] = None

    @property
    def configured(self) -> bool:
        return bool(self.api_key)

    @property
    def key_format_valid(self) -> bool:
        if not self.api_key:
            return False
        # Fireworks keys typically start with "fw_"
        return self.api_key.startswith("fw_")

    def status(self) -> dict:
        return {
            "configured": self.configured,
            "key_format_valid": self.key_format_valid,
            "model": self.model,
            "last_error": self.last_error,
            "help": (
                None
                if self.key_format_valid
                else "Set FIREWORKS_API_KEY (or fireworks_key) in .env. Keys start with 'fw_'."
            ),
        }

    def generate_brief(self, prompt: str) -> Optional[FireworksBriefPayload]:
        if not self.configured:
            self.last_error = "FIREWORKS_API_KEY (or fireworks_key) is not set"
            return None

        if not self.key_format_valid:
            self.last_error = "Invalid key format: Fireworks keys start with 'fw_'"
            print(f"[unicornforge] {self.last_error}")
            return None

        structured = self._generate_structured(prompt)
        if structured is not None:
            return structured

        text = self._generate_text(prompt)
        if not text:
            return None

        # If primary model failed (e.g. 404), try common fallbacks once
        for fallback in [
            "accounts/fireworks/models/llama-v3p1-8b-instruct",
            "accounts/fireworks/models/qwen2-72b-instruct",
            "accounts/fireworks/models/mixtral-8x7b-instruct",
        ]:
            if self.model == fallback:
                continue
            old_model = self.model
            self.model = fallback
            try:
                structured = self._generate_structured(prompt)
                if structured:
                    print(f"[unicornforge] Fell back to Fireworks model {fallback}")
                    return structured
            finally:
                self.model = old_model
            text = self._generate_text(prompt)
            if text:
                print(f"[unicornforge] Fell back to Fireworks model {fallback}")
                return self._parse_text_fallback(text) if text else None
        return None

        try:
            data = json.loads(text)
            return FireworksBriefPayload.model_validate(data)
        except Exception:
            return self._parse_text_fallback(text)

    def _generate_structured(self, prompt: str) -> Optional[FireworksBriefPayload]:
        try:
            response = requests.post(
                f"{self.api_base}/chat/completions",
                headers=self._headers(),
                json={
                    "model": self.model,
                    "messages": [
                        {
                            "role": "system",
                            "content": (
                                "You are an expert startup advisor for hackathon teams. "
                                "Return only JSON matching the schema."
                            ),
                        },
                        {"role": "user", "content": prompt},
                    ],
                    "response_format": {
                        "type": "json_object",
                    },
                    "max_tokens": 2048,
                },
                timeout=120,
            )
            if not response.ok:
                self.last_error = f"structured chat/completions {response.status_code}: {response.text[:300]}"
                print(f"[unicornforge] Fireworks structured failed: {self.last_error}")
                return None

            content = self._extract_chat_content(response.json())
            if not content:
                return None
            return FireworksBriefPayload.model_validate(json.loads(content))
        except Exception as exc:
            self.last_error = f"structured parse error: {exc}"
            print(f"[unicornforge] Fireworks structured error: {exc}")
            return None

    def _generate_text(self, prompt: str) -> Optional[str]:
        for endpoint, payload in (
            (
                "chat/completions",
                {
                    "model": self.model,
                    "messages": [
                        {
                            "role": "system",
                            "content": "You are an expert startup advisor for hackathon teams.",
                        },
                        {"role": "user", "content": prompt},
                    ],
                    "max_tokens": 2048,
                },
            ),
        ):
            text = self._post(endpoint, payload)
            if text:
                return text
        return None

    def _post(self, endpoint: str, payload: dict) -> Optional[str]:
        try:
            response = requests.post(
                f"{self.api_base}/{endpoint}",
                headers=self._headers(),
                json=payload,
                timeout=120,
            )
            if not response.ok:
                self.last_error = f"{endpoint} {response.status_code}: {response.text[:300]}"
                print(f"[unicornforge] Fireworks {endpoint} failed: {self.last_error}")
                return None
            return self._extract_response_text(endpoint, response.json())
        except Exception as exc:
            self.last_error = str(exc)
            print(f"[unicornforge] Fireworks {endpoint} error: {exc}")
            return None

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

    def _extract_chat_content(self, data: dict) -> Optional[str]:
        choices = data.get("choices", [])
        if not choices:
            return None
        message = choices[0].get("message", {})
        content = message.get("content")
        return content.strip() if isinstance(content, str) and content.strip() else None

    def _extract_response_text(self, endpoint: str, data: dict) -> Optional[str]:
        # Fireworks primarily uses chat/completions
        return self._extract_chat_content(data)

    def _parse_text_fallback(self, text: str) -> Optional[FireworksBriefPayload]:
        # Simple label-based fallback (same as before)
        sections: dict[str, str] = {}
        labels = {
            "project_name": "project name",
            "one_sentence_pitch": "one-sentence pitch",
            "problem": "problem",
            "solution": "solution",
            "target_market": "target market",
            "mvp_scope": "mvp scope",
            "key_features": "key features",
            "demo_scenario": "demo scenario",
            "business_model": "business model",
            "why_it_can_win": "why this project can win a hackathon",
        }
        lowered = text.lower()
        for key, label in labels.items():
            marker = f"{label}:"
            start = lowered.find(marker)
            if start == -1:
                continue
            start += len(marker)
            end = len(text)
            for other in labels.values():
                if other == label:
                    continue
                pos = lowered.find(f"{other}:", start)
                if pos != -1:
                    end = min(end, pos)
            sections[key] = text[start:end].strip()

        if len(sections) < 4:
            return None
        return FireworksBriefPayload.model_validate(sections)
