from __future__ import annotations

from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel


# ==== Data models =============================================================

class GenerateBriefRequest(BaseModel):
    project_idea: str
    target_users: Optional[str] = None
    industry: Optional[str] = None
    available_time: Optional[str] = None
    available_technologies: Optional[str] = None


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


# ==== AI integration stub ====================================================

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


def _call_ai_model(prompt: str) -> GenerateBriefResponse:
    """
    Placeholder for the actual AI call.

    For now, returns a static but structured response that follows
    the output contract, so the frontend can integrate against this API.

    Later, replace this with:
    - call to Fireworks AI API, or
    - call to an open-source model running on AMD GPU.
    """
    # TODO: integrate real model using the `prompt` argument.
    # Keep this stub simple and deterministic for now.
    return GenerateBriefResponse(
        project_name="UnicornForge AI (Stubbed)",
        one_sentence_pitch=(
            "UnicornForge AI turns rough hackathon ideas into "
            "startup-ready briefs, MVP scopes, and demo scenarios in minutes."
        ),
        problem=(
            "Hackathon teams and early founders struggle to translate vague "
            "ideas into clear startup concepts under tight time constraints."
        ),
        solution=(
            "Provide an AI-powered mentor that uses structured prompts to "
            "generate a complete startup brief from a short idea description."
        ),
        target_market=(
            "Hackathon participants, students, accelerators, incubators, and "
            "innovation teams in companies and universities."
        ),
        mvp_scope=(
            "Single-page web app where the user enters an idea and optional "
            "context, clicks Generate, and receives a 10-section startup brief."
        ),
        key_features=(
            "- Idea-to-brief generation\n"
            "- Clear problem and solution framing\n"
            "- MVP scope and key features suggestion\n"
            "- Demo scenario tailored for hackathons\n"
            "- Business model and 'why this can win' summary"
        ),
        demo_scenario=(
            "1) User enters a rough idea and context.\n"
            "2) Clicks 'Generate Startup Brief'.\n"
            "3) App shows all 10 sections in a scrollable panel.\n"
            "4) User copies the brief as Markdown into their pitch deck."
        ),
        business_model=(
            "Freemium SaaS with limits on free generations, team "
            "subscriptions for collaboration, and B2B licenses for "
            "accelerators and universities."
        ),
        why_it_can_win=(
            "Directly supports the Unicorn Track by showcasing a creative, "
            "AMD-ready AI product that helps more teams ship better pitches, "
            "not just better benchmarks."
        ),
    )


# ==== Routes =================================================================

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
    # In the future, `prompt` will be sent to an AI model.
    response = _call_ai_model(prompt)
    return response