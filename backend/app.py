from __future__ import annotations

from typing import Optional

import os
import re
import requests
import google.generativeai as genai
from dotenv import load_dotenv

from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from ml.predictor import PredictionResult, SuccessPredictor

# Load environment variables from .env file
load_dotenv()

# Optional ML dependencies; guarded imports for environments without torch
try:
    import torch
except Exception:  # pragma: no cover - best-effort import
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


# ==== Prompt template (from docs, simplified) =================================

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
- Highlight where the available technologies (especially AMD GPUs and Fireworks AI API, if mentioned) can play a role.
"""


# ==== FastAPI app ============================================================


app = FastAPI(
    title="UnicornForge AI Backend",
    description="MVP API for generating structured startup briefs.",
    version="0.1.0",
)

# Allow frontend from local dev servers (PyCharm, Live Server, etc.)
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


# ==== Local success model ====================================================

SUCCESS_PREDICTOR = SuccessPredictor()

try:
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
except Exception:
    pass


# ==== AI provider initialization (xAI Grok + Google Gemini fallback) ==========

XAI_API_KEY = os.getenv("XAI_API_KEY")
XAI_API_BASE = "https://api.x.ai/v1"
XAI_MODEL = os.getenv("XAI_MODEL", "grok-4.5")

if XAI_API_KEY:
    print("[unicornforge] xAI Grok API configured successfully")
else:
    print("[unicornforge] WARNING: XAI_API_KEY not found in environment")

GEMINI_API_KEY = os.getenv("GOOGLE_GEMINI_API_KEY")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    print("[unicornforge] Google Gemini API configured successfully")
else:
    print("[unicornforge] WARNING: GOOGLE_GEMINI_API_KEY not found in environment")


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
    updates = {"llm_source": llm_source}
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


# ==== AI integration =========================================================

def _build_prompt(payload: GenerateBriefRequest) -> str:
    """Fill the prompt template with request data."""
    def norm(value: Optional[str]) -> str:
        return value.strip() if value and value.strip() else "not specified"

    return PROMPT_TEMPLATE.format(
        project_idea=payload.project_idea.strip(),
        target_users=norm(payload.target_users),
        industry=norm(payload.industry),
        available_time=norm(payload.available_time),
        available_technologies=norm(payload.available_technologies),
    )


def _call_grok_api(prompt: str) -> Optional[str]:
    """Call xAI Grok via the OpenAI-compatible responses API."""
    if not XAI_API_KEY:
        return None

    try:
        response = requests.post(
            f"{XAI_API_BASE}/responses",
            headers={
                "Authorization": f"Bearer {XAI_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": XAI_MODEL,
                "input": prompt,
            },
            timeout=120,
        )
        response.raise_for_status()
        data = response.json()

        if isinstance(data.get("output_text"), str) and data["output_text"].strip():
            return data["output_text"]

        for item in data.get("output", []):
            if item.get("type") == "message":
                for content in item.get("content", []):
                    if content.get("type") == "output_text" and content.get("text"):
                        return content["text"]
    except Exception as e:
        print(f"[unicornforge] Error calling xAI Grok API: {e}")

    return None


def _call_ai_model(
    prompt: str,
    payload: Optional[GenerateBriefRequest] = None,
    prediction: Optional[PredictionResult] = None,
) -> GenerateBriefResponse:
    """
    Generate intelligent brief using xAI Grok (primary) or Google Gemini (fallback).
    Falls back to keyword-based generation if APIs are unavailable.
    """
    grok_text = _call_grok_api(prompt)
    if grok_text:
        return _attach_prediction(_parse_brief_from_text(grok_text, payload), prediction, "grok")

    if GEMINI_API_KEY:
        try:
            model = genai.GenerativeModel("gemini-pro")
            response = model.generate_content(prompt)

            if response.text:
                return _attach_prediction(_parse_brief_from_text(response.text, payload), prediction, "gemini")
        except Exception as e:
            print(f"[unicornforge] Error calling Gemini API: {e}")
            print("[unicornforge] Falling back to keyword-based generation")
    
    # Fallback to intelligent keyword-based generation
    if payload is None:
        return _attach_prediction(_generate_stub_brief(payload), prediction, "template")
    
    try:
        idea = payload.project_idea.strip() if payload.project_idea else "Your Project"
        target = payload.target_users.strip() if payload.target_users else "End users"
        industry = payload.industry.strip() if payload.industry else "Technology"
        time_available = payload.available_time.strip() if payload.available_time else "2 weeks"
        tech = payload.available_technologies.strip() if payload.available_technologies else "Standard tech stack"
        
        # Extract keywords from idea for personalization
        idea_words = idea.lower().split()
        keywords = [w for w in idea_words if len(w) > 3][:3]
        keyword_str = " + ".join(keywords) if keywords else idea
        
        # Generate unique brief sections based on inputs
        project_name = f"{idea} Platform"
        one_sentence = f"A {industry.lower()}-focused solution that helps {target.lower()} with {idea.lower()}."
        
        problem = f"{target.capitalize()} struggle with inefficient solutions for {idea.lower()}. Current alternatives are complex, expensive, and don't integrate well. This creates friction and limits adoption in the {industry.lower()} space."
        
        solution = f"We're building a streamlined platform that addresses {keyword_str} with modern tech ({tech.lower()}). The MVP focuses on core {idea.lower()} capabilities, leveraging {tech.lower()} for optimal performance."
        
        target_market = f"Primary: {target}. Secondary: organizations in {industry} seeking digital transformation. TAM: $500M+ in the {industry.lower()} sector."
        
        mvp_scope = f"Core features: {idea.lower()} automation, user dashboard, {tech.split(',')[0] if ',' in tech else 'integration'} APIs. Timeline: {time_available}. Tech: {tech}. No: advanced analytics, multi-team support (Phase 2)."
        
        key_features = f"- Smart {idea.lower()} processing\n- Real-time dashboard\n- {tech.split()[0]}-powered optimization\n- One-click integrations\n- Role-based access control\n- Automated reporting"
        
        demo_scenario = f"1) Log in with demo account.\n2) Show {idea.lower()} workflow in action.\n3) Demonstrate time/cost savings vs alternatives.\n4) Show {tech.split()[0]} performance metrics.\n5) Q&A on {industry} use cases."
        
        business_model = f"Freemium tier (5 free {idea.lower()} operations/month). Pro ($29/mo): unlimited {idea.lower()} + priority support. Enterprise: custom pricing + dedicated support. Target: 1000 paying users in Year 1."
        
        why_win = f"We solve a real pain point for {target.lower()} in {industry}. Built with {tech}, we can scale efficiently. The market is hungry for solutions like this. Our demo showcases immediate ROI. Hackathon win = proof of concept for enterprise pitch."
        
        return _attach_prediction(
            GenerateBriefResponse(
                project_name=project_name,
                one_sentence_pitch=one_sentence,
                problem=problem,
                solution=solution,
                target_market=target_market,
                mvp_scope=mvp_scope,
                key_features=key_features,
                demo_scenario=demo_scenario,
                business_model=business_model,
                why_it_can_win=why_win,
            ),
            prediction,
            "template",
        )
    except Exception as e:
        print(f"[unicornforge] Error in keyword-based generation: {e}")
        return _attach_prediction(_generate_stub_brief(payload), prediction, "template")


def _parse_brief_from_text(text: str, payload: Optional[GenerateBriefRequest]) -> GenerateBriefResponse:
    """
    Parse the 10-section brief from LLM response using keyword matching.
    More flexible than expecting strict numbering.
    """
    sections = {}
    
    # Split by keywords and extract content
    keywords = [
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
    
    for key, pattern, flags in keywords:
        match = re.search(pattern, text, flags)
        if match:
            content = match.group(1).strip()
            # Clean up bullet points and formatting
            lines = content.split('\n')
            sections[key] = lines[0].strip() if lines else content
    
    # Use defaults for missing sections
    idea = payload.project_idea if payload else "Your Idea"
    
    return GenerateBriefResponse(
        project_name=sections.get("project_name", f"{idea} — UnicornForge AI").strip(),
        one_sentence_pitch=sections.get("one_sentence_pitch", f"{idea} — AI-powered startup brief.").strip(),
        problem=sections.get("problem", "Teams need to pitch quickly.").strip(),
        solution=sections.get("solution", "Generate a structured startup brief instantly.").strip(),
        target_market=sections.get("target_market", payload.target_users or "Hackathon teams").strip(),
        mvp_scope=sections.get("mvp_scope", "Single-page web app for brief generation.").strip(),
        key_features=sections.get("key_features", "- Idea-to-brief generation\n- Clear problem & solution framing").strip(),
        demo_scenario=sections.get("demo_scenario", "1) Enter idea. 2) Generate. 3) Copy.").strip(),
        business_model=sections.get("business_model", "Freemium SaaS.").strip(),
        why_it_can_win=sections.get("why_it_can_win", "Helps teams pitch quickly using AMD GPUs.").strip(),
    )


def _generate_stub_brief(payload: Optional[GenerateBriefRequest]) -> GenerateBriefResponse:
    """
    Fallback stub that uses request data so the response reflects the frontend input.
    Also runs the local success model (if available) and attaches a success_score to the response.
    """
    def safe(val: Optional[str], default: str = "not specified") -> str:
        return val.strip() if val and val.strip() else default

    idea = safe(payload.project_idea) if payload else "not specified"
    target_users = safe(payload.target_users) if payload else "Hackathon participants, students, founders"
    industry = safe(payload.industry) if payload else "not specified"
    available_time = safe(payload.available_time) if payload else "not specified"
    available_technologies = safe(payload.available_technologies) if payload else "not specified"

    project_name = f"{idea[:60]} — UnicornForge AI" if idea != "not specified" else "UnicornForge AI (Stubbed)"
    one_sentence_pitch = f"{idea} — AI-generated startup brief for hackathons and early founders."
    problem = f"Teams need to turn the idea '{idea}' into a clear pitch and MVP quickly."
    solution = (
        f"Generate a concise 10-section startup brief from the idea '{idea}', "
        f"tailored to {target_users} in {industry}."
    )
    target_market = target_users
    mvp_scope = (
        f"Single-page web app: enter the idea ('{idea}') and optional context, "
        "click Generate, and receive a 10-section brief suitable for a hackathon demo."
    )
    key_features = (
        "- Idea-to-brief generation\n"
        "- Clear problem & solution framing\n"
        "- MVP scope and prioritized features\n"
        "- Demo scenario and pitch copy\n"
        "- Business model sketch\n"
        "- AMD-ready deployment hints"
    )
    demo_scenario = (
        "1) Enter idea and context.\n"
        "2) Click 'Generate Startup Brief'.\n"
        "3) Review the 10 sections and adapt for the demo.\n"
        "4) Copy as Markdown into the pitch deck."
    )
    business_model = "Freemium SaaS with team subscriptions and institutional licenses."
    why_it_can_win = (
        f"Helps teams quickly convert '{idea}' into a pitch-ready plan; "
        f"designed to showcase AMD GPUs and ROCm-powered inference for open models."
    )

    return GenerateBriefResponse(
        project_name=project_name,
        one_sentence_pitch=one_sentence_pitch,
        problem=problem,
        solution=solution,
        target_market=target_market,
        mvp_scope=mvp_scope,
        key_features=key_features,
        demo_scenario=demo_scenario,
        business_model=business_model,
        why_it_can_win=why_it_can_win,
    )


# ==== Routes =================================================================

@app.get("/")
async def serve_frontend():
    """Serve the single-page UI from the same origin as the API."""
    return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))


@app.post("/generate-brief", response_model=GenerateBriefResponse)
async def generate_brief(payload: GenerateBriefRequest) -> GenerateBriefResponse:
    """
    Generate a structured startup brief from a rough idea.

    - Validates input.
    - Builds the AI prompt.
    - Calls the (currently stubbed) AI integration.
    """
    if not payload.project_idea or not payload.project_idea.strip():
        raise HTTPException(status_code=400, detail="project_idea is required")

    prompt = _build_prompt(payload)
    prediction = _predict_success(payload)
    response = _call_ai_model(prompt, payload, prediction)
    return response


@app.get("/health")
async def health():
    """Simple health endpoint reporting model and environment status."""
    return {
        "ok": True,
        "success_model_ready": SUCCESS_PREDICTOR.ready,
        "feature_columns": len(SUCCESS_PREDICTOR.feature_columns),
        "torch_available": torch is not None,
        "cuda_available": getattr(torch, "cuda", None) is not None and torch.cuda.is_available() if torch is not None else False,
        "xai_configured": bool(XAI_API_KEY),
        "gemini_configured": bool(GEMINI_API_KEY),
    }