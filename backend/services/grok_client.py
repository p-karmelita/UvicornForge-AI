from __future__ import annotations

import json
import os
from typing import Optional

import requests
from pydantic import BaseModel, Field


class GrokBriefPayload(BaseModel):
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


class GrokClient:
    def __init__(self) -> None:
        self.api_key = os.getenv("XAI_API_KEY", "").strip()
        self.api_base = os.getenv("XAI_API_BASE", "https://api.x.ai/v1").rstrip("/")
        self.model = os.getenv("XAI_MODEL", "grok-4.5")
        self.last_error: Optional[str] = None

    @property
    def configured(self) -> bool:
        return bool(self.api_key)

    @property
    def key_format_valid(self) -> bool:
        if not self.api_key:
            return False
        return not self.api_key.startswith("xai-token-")

    def status(self) -> dict:
        return {
            "configured": self.configured,
            "key_format_valid": self.key_format_valid,
            "model": self.model,
            "last_error": self.last_error,
            "help": (
                None
                if self.key_format_valid
                else "Replace XAI_API_KEY with a key from https://console.x.ai (xai-token-* keys do not work with api.x.ai)."
            ),
        }

    def generate_brief(self, prompt: str) -> Optional[GrokBriefPayload]:
        if not self.configured:
            self.last_error = "XAI_API_KEY is not set"
            return None

        if not self.key_format_valid:
            self.last_error = "Invalid key format: use a console.x.ai API key, not xai-token-*"
            print(f"[unicornforge] {self.last_error}")
            return None

        structured = self._generate_structured(prompt)
        if structured is not None:
            return structured

        text = self._generate_text(prompt)
        if not text:
            return None

        try:
            data = json.loads(text)
            return GrokBriefPayload.model_validate(data)
        except Exception:
            return self._parse_text_fallback(text)

    def test_connection(self) -> dict:
        if not self.configured:
            return {"ok": False, "error": "XAI_API_KEY is not set"}

        if not self.key_format_valid:
            return {
                "ok": False,
                "error": "Invalid key format. Obtain an API key from https://console.x.ai",
            }

        try:
            response = requests.post(
                f"{self.api_base}/chat/completions",
                headers=self._headers(),
                json={
                    "model": self.model,
                    "messages": [{"role": "user", "content": "Reply with the single word: pong"}],
                    "max_tokens": 8,
                },
                timeout=30,
            )
            if response.ok:
                return {"ok": True, "model": self.model}
            self.last_error = response.text[:300]
            return {"ok": False, "status_code": response.status_code, "error": self.last_error}
        except Exception as exc:
            self.last_error = str(exc)
            return {"ok": False, "error": self.last_error}

    def _generate_structured(self, prompt: str) -> Optional[GrokBriefPayload]:
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
                        "type": "json_schema",
                        "json_schema": {
                            "name": "startup_brief",
                            "schema": BRIEF_JSON_SCHEMA,
                            "strict": True,
                        },
                    },
                },
                timeout=120,
            )
            if not response.ok:
                self.last_error = f"structured chat/completions {response.status_code}: {response.text[:300]}"
                print(f"[unicornforge] Grok structured failed: {self.last_error}")
                return None

            content = self._extract_chat_content(response.json())
            if not content:
                return None
            return GrokBriefPayload.model_validate(json.loads(content))
        except Exception as exc:
            self.last_error = f"structured parse error: {exc}"
            print(f"[unicornforge] Grok structured error: {exc}")
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
                },
            ),
            ("responses", {"model": self.model, "input": prompt}),
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
                print(f"[unicornforge] Grok {endpoint} failed: {self.last_error}")
                return None
            return self._extract_response_text(endpoint, response.json())
        except Exception as exc:
            self.last_error = str(exc)
            print(f"[unicornforge] Grok {endpoint} error: {exc}")
            return None

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def _extract_chat_content(self, data: dict) -> Optional[str]:
        choices = data.get("choices", [])
        if not choices:
            return None
        message = choices[0].get("message", {})
        content = message.get("content")
        return content.strip() if isinstance(content, str) and content.strip() else None

    def _extract_response_text(self, endpoint: str, data: dict) -> Optional[str]:
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
            return None
        return self._extract_chat_content(data)

    def _parse_text_fallback(self, text: str) -> Optional[GrokBriefPayload]:
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
        return GrokBriefPayload.model_validate(sections)