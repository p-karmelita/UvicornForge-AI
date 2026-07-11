from __future__ import annotations

from typing import Optional

import os
import re
import requests
from dotenv import load_dotenv

from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from ml.brief_generator import generate_dataset_brief
from ml.dataset import build_dataset_context, get_dataset_info
from ml.feature_mapper import map_request_to_features
from ml.predictor import PredictionResult, SuccessPredictor

load_dotenv()

try:
    import torch
except Exception:  # pragma: no cover
    torch = None


# ==== Data models =============================================================

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


PROMPT_TEMPLATE = """You are an experienced startup advisor and hackathon mentor.

Your task is to transform a rough idea into a clear, structured startup brief that a hackathon team or early-stage founder can immediately use for planning, demos, and pitching.

Use the information below:

Project idea:
{project_idea}

Target users:
{target_users}

Industry:
{industry}

Available time:
{available_time}

Available technologies:
{available_technologies}

Reference data from global_startup_success_dataset.csv:
{dataset_context}

Now generate a complete startup brief in English.

Return the content in 10 clearly separated sections, in this exact order:

1. Project name
2. One-sentence pitch
3. Problem
4. Solution
5. Target market
6. MVP scope
7. Key features
8. Demo scenario
9. Business model
10. Why this project can win a hackathon

Guidelines:
- Be specific and practical.
- Assume the team is building this as a hackathon or early MVP project.
- Keep each section concise but informative.
- Use the dataset references to ground market and success patterns.
- Highlight where the available technologies (especially AMD GPUs and Fireworks AI API, if mentioned) can play a role.
"""


# ==== FastAPI app ============================================================

app = FastAPI(
    title="UnicornForge AI Backend",
    description="MVP API for generating structured startup briefs.",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["null"],
    allow_origin_regex=r"https?://(localhost|127\.0\.0\.1)(:\d+)?",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
    max_age=3600,
)

FRONTEND_DIR = os.path.join(os.path.dirname(__file__), "..", "frontend")
SUCCESS_PREDICTOR = SuccessPredictor()

XAI_API_KEY = os.getenv("XAI_API_KEY")
XAI_API_BASE = "https://api.x.ai/v1"
XAI_MODEL = os.getenv("XAI_MODEL", "grok-4.5")


def _log_startup_status() -> None:
    dataset = get_dataset_info()
    print("[unicornforge] torch available:", bool(torch))
    if torch is not None:
        try:
            print("[unicornforge] torch.__version__:", torch.__version__)
            print("[unicornforge] cuda available:", torch.cuda.is_available())
            print("[unicornforge] device_count:", torch.cuda.device_count())
            if torch.cuda.device_count() > 0:
                for i in range(torch.cuda.device_count()):
                    try:
                        print(f"[unicornforge] device {i}:", torch.cuda.get_device_name(i))
                    except Exception:
                        pass
        except Exception:
            pass
    print(f"[unicornforge] success model ready: {SUCCESS_PREDICTOR.ready}")
    print("[unicornforge] xAI configured:", bool(XAI_API_KEY))
    print(
        "[unicornforge] dataset loaded:",
        dataset["loaded"],
        f"({dataset['rows']} rows)" if dataset["loaded"] else "",
        f"from {dataset['path']}" if dataset["path"] else "",
    )


_log_startup_status()


# ==== Helpers =================================================================

def _norm_field(value: Optional[str], default: str = "not specified") -> str:
    return value.strip() if value and value.strip() else default


def _predict_success(payload: GenerateBriefRequest) -> Optional[PredictionResult]:
    return SUCCESS_PREDICTOR.predict_from_payload(
        project_idea=payload.project_idea,
        target_users=payload.target_users,
        industry=payload.industry,
        available_time=payload.available_time,
        available_technologies=payload.available_technologies,
    )


def _attach_prediction(
    response: GenerateBriefResponse,
    prediction: Optional[PredictionResult],
    llm_source: Optional[str] = None,
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


def _build_prompt(payload: GenerateBriefRequest) -> str:
    mapped = map_request_to_features(
        project_idea=payload.project_idea,
        target_users=payload.target_users,
        industry=payload.industry,
        available_time=payload.available_time,
        available_technologies=payload.available_technologies,
    )
    dataset_context = build_dataset_context(mapped.industry, mapped.tech_stack)

    return PROMPT_TEMPLATE.format(
        project_idea=payload.project_idea.strip(),
        target_users=_norm_field(payload.target_users),
        industry=_norm_field(payload.industry, mapped.industry),
        available_time=_norm_field(payload.available_time),
        available_technologies=_norm_field(payload.available_technologies, mapped.tech_stack),
        dataset_context=dataset_context,
    )


def _generate_dataset_brief_response(payload: GenerateBriefRequest) -> GenerateBriefResponse:
    sections = generate_dataset_brief(
        project_idea=payload.project_idea,
        target_users=payload.target_users,
        industry=payload.industry,
        available_time=payload.available_time,
        available_technologies=payload.available_technologies,
    )
    return GenerateBriefResponse(**sections)


def _parse_brief_from_text(text: str, payload: GenerateBriefRequest) -> GenerateBriefResponse:
    sections: dict[str, str] = {}
    patterns = [
        ("project_name", r"(?:project\s+)?name:?\s*(.+?)(?=(?:one[- ]sentence|pitch|problem|$))", re.IGNORECASE),
        ("one_sentence_pitch", r"(?:one[- ]sentence)?(?:\s*)?pitch:?\s*(.+?)(?=problem|$)", re.IGNORECASE),
        ("problem", r"problem:?\s*(.+?)(?=solution|$)", re.IGNORECASE),
        ("solution", r"solution:?\s*(.+?)(?=target|market|$)", re.IGNORECASE),
        ("target_market", r"target\s+market:?\s*(.+?)(?=mvp|scope|$)", re.IGNORECASE),
        ("mvp_scope", r"(?:mvp|minimum viable product)\s+scope:?\s*(.+?)(?=(?:key\s+)?features|$)", re.IGNORECASE),
        ("key_features", r"(?:key\s+)?features:?\s*(.+?)(?=demo|scenario|$)", re.IGNORECASE),
        ("demo_scenario", r"demo\s+scenario:?\s*(.+?)(?=business\s+model|$)", re.IGNORECASE),
        ("business_model", r"business\s+model:?\s*(.+?)(?=why|$)", re.IGNORECASE),
        ("why_it_can_win", r"(?:why|why\s+this|can\s+win).*?:?\s*(.+?)$", re.IGNORECASE | re.MULTILINE),
    ]

    for key, pattern, flags in patterns:
        match = re.search(pattern, text, flags)
        if match:
            content = match.group(1).strip()
            sections[key] = content.split("\n", 1)[0].strip()

    idea = payload.project_idea.strip()
    return GenerateBriefResponse(
        project_name=sections.get("project_name", f"{idea} — UnicornForge AI"),
        one_sentence_pitch=sections.get("one_sentence_pitch", f"{idea} — AI-powered startup brief."),
        problem=sections.get("problem", "Teams need to pitch quickly."),
        solution=sections.get("solution", "Generate a structured startup brief instantly."),
        target_market=sections.get("target_market", _norm_field(payload.target_users, "Hackathon teams")),
        mvp_scope=sections.get("mvp_scope", "Single-page web app for brief generation."),
        key_features=sections.get(
            "key_features",
            "- Idea-to-brief generation\n- Clear problem & solution framing",
        ),
        demo_scenario=sections.get("demo_scenario", "1) Enter idea. 2) Generate. 3) Copy."),
        business_model=sections.get("business_model", "Freemium SaaS."),
        why_it_can_win=sections.get("why_it_can_win", "Helps teams pitch quickly using AMD GPUs."),
    )


def _call_grok_api(prompt: str) -> Optional[str]:
    if not XAI_API_KEY:
        return None

    try:
        response = requests.post(
            f"{XAI_API_BASE}/responses",
            headers={
                "Authorization": f"Bearer {XAI_API_KEY}",
                "Content-Type": "application/json",
            },
            json={"model": XAI_MODEL, "input": prompt},
            timeout=120,
        )
        response.raise_for_status()
        data = response.json()

        output_text = data.get("output_text")
        if isinstance(output_text, str) and output_text.strip():
            return output_text

        for item in data.get("output", []):
            if item.get("type") != "message":
                continue
            for content in item.get("content", []):
                if content.get("type") == "output_text" and content.get("text"):
                    return content["text"]
    except Exception as exc:
        print(f"[unicornforge] Error calling xAI Grok API: {exc}")

    return None


def _call_ai_model(
    prompt: str,
    payload: GenerateBriefRequest,
    prediction: Optional[PredictionResult],
) -> GenerateBriefResponse:
    grok_text = _call_grok_api(prompt)
    if grok_text:
        return _attach_prediction(_parse_brief_from_text(grok_text, payload), prediction, "grok")

    return _attach_prediction(_generate_dataset_brief_response(payload), prediction, "dataset")


# ==== Routes =================================================================

@app.get("/")
async def serve_frontend():
    return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))


@app.post("/generate-brief", response_model=GenerateBriefResponse)
async def generate_brief(payload: GenerateBriefRequest) -> GenerateBriefResponse:
    if not payload.project_idea or not payload.project_idea.strip():
        raise HTTPException(status_code=400, detail="project_idea is required")

    dataset = get_dataset_info()
    if not dataset["loaded"]:
        raise HTTPException(
            status_code=503,
            detail="global_startup_success_dataset.csv not found in backend/",
        )

    prediction = _predict_success(payload)
    return _call_ai_model(_build_prompt(payload), payload, prediction)


@app.get("/health")
async def health():
    dataset = get_dataset_info()
    return {
        "ok": True,
        "success_model_ready": SUCCESS_PREDICTOR.ready,
        "feature_columns": len(SUCCESS_PREDICTOR.feature_columns),
        "dataset_loaded": dataset["loaded"],
        "dataset_rows": dataset["rows"],
        "dataset_path": dataset["path"],
        "torch_available": torch is not None,
        "cuda_available": (
            getattr(torch, "cuda", None) is not None and torch.cuda.is_available()
            if torch is not None
            else False
        ),
        "xai_configured": bool(XAI_API_KEY),
    }