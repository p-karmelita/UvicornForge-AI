from __future__ import annotations

from typing import Optional

from .dataset import find_similar_startup_rows
from .feature_mapper import map_request_to_features


def _norm(value: Optional[str], default: str = "not specified") -> str:
    return value.strip() if value and value.strip() else default


def generate_dataset_brief(
    project_idea: str,
    target_users: Optional[str] = None,
    industry: Optional[str] = None,
    available_time: Optional[str] = None,
    available_technologies: Optional[str] = None,
) -> dict[str, str]:
    """Build a startup brief grounded in the local CSV dataset."""
    mapped = map_request_to_features(
        project_idea=project_idea,
        target_users=target_users,
        industry=industry,
        available_time=available_time,
        available_technologies=available_technologies,
    )

    idea = project_idea.strip()
    target = _norm(target_users, "hackathon teams and early founders")
    time_available = _norm(available_time, "one weekend")
    tech = _norm(available_technologies, mapped.tech_stack)
    references = find_similar_startup_rows(mapped.industry, mapped.tech_stack, limit=2)

    # Use realistic small-team numbers for hackathon context instead of raw (skewed) dataset medians
    realistic_benchmarks = {
        "employees": "4-12",
        "funding": "< $500k (bootstrapped or small pre-seed)",
        "revenue": "pre-revenue or <$50k ARR in early pilots",
    }

    # Avoid leaking synthetic "Startup_XXX" names to users
    reference_names = "leading performers in the dataset"
    if references:
        # Extract useful signals without names (new schema)
        techs = {str(r.get("Backend Tech Stack", "") or r.get("Tech Stack", "")) for r in references}
        techs = {t for t in techs if t}
        reference_names = ", ".join(sorted(techs)) or reference_names

    # Keep short, non-leaking insights for internal use only (not dumped raw)
    top_score = float(references[0].get("Success Score", 5)) if references else 5.0

    # Simple, reliable product name for the dataset fallback
    # (when the LLM via Fireworks is available it usually generates much better names)
    idea_lower = idea.lower()
    if "agent" in idea_lower:
        project_name = "Autonomous Task Agent"
    elif "model" in idea_lower and ("select" in idea_lower or "route" in idea_lower or "decid" in idea_lower):
        project_name = "Smart Model Router"
    elif any(k in idea_lower for k in ["customer", "lead", "find", "acquir"]):
        project_name = "Early Customer Finder"
    else:
        tech = mapped.tech_stack.split(",")[0].strip()
        project_name = f"{tech} {mapped.industry} Assistant"[:34]

    return {
        "project_name": project_name,
        "one_sentence_pitch": (
            f"{idea[:110].strip()} — an {mapped.industry} tool that helps {target} ship faster, "
            f"using patterns from high-performing teams in the dataset."
        ),
        "problem": (
            f"{target.capitalize()} building in {mapped.industry} often waste precious hackathon hours "
            f"on boilerplate, unclear scoping, and generic advice. They need focused, data-informed "
            f"guidance that respects tight time and resource constraints."
        ),
        "solution": (
            f"A lightweight {mapped.industry} assistant built with {tech} that turns a rough idea into "
            f"a structured, judge-ready brief + success estimate in minutes. Designed for {time_available} "
            f"timelines and small teams. Architecture draws from what actually correlates with higher "
            f"outcomes in similar past projects (model choice, scoping, demo focus)."
        ),
        "target_market": (
            f"Primary: {target} at hackathons and early-stage builders. "
            f"Realistic early validation: small teams (3-12 people), minimal funding, focus on "
            f"working demo + clear narrative rather than scale."
        ),
        "mvp_scope": (
            f"Deliver a complete, self-contained flow for \"{idea[:55]}\" within {time_available}. "
            f"Must-haves: idea input, brief generation, score + factors, markdown export. "
            f"Use {mapped.tech_stack}. Skip multi-user, billing, and heavy infra."
        ),
        "key_features": (
            f"- Success scoring grounded in dataset patterns ({mapped.industry})\n"
            f"- Clear 10-section brief optimized for hackathon judges\n"
            f"- Relevant reference patterns (without leaking raw company data)\n"
            f"- AMD GPU / {tech} friendly inference path\n"
            f"- One-click Markdown export for pitch decks and READMEs"
        ),
        "demo_scenario": (
            f"1) Paste your idea: \"{idea[:45]}...\".\n"
            f"2) (Optional) Add target users, industry, time, and tech stack.\n"
            f"3) Generate → see structured brief + predicted success factors.\n"
            f"4) Review similar patterns and export clean Markdown.\n"
            f"5) Use it directly in your hackathon submission / pitch."
        ),
        "business_model": (
            f"Freemium for individual hackathon participants and student teams. "
            f"Paid tier for bootcamps, accelerators, and university programs that want bulk usage, "
            f"custom scoring models, or private deployment. Focus on volume at events first."
        ),
        "why_it_can_win": (
            f"Hackathon projects win when they are focused, well-scoped, and demonstrate real "
            f"understanding of the problem. This tool helps teams achieve exactly that quickly. "
            f"It combines a trained outcome model, practical dataset signals, and AMD-ready "
            f"execution — directly relevant to the Unicorn Track goals."
        ),
        # High-value additions (dataset fallback - simpler but useful)
        "risks_and_mitigations": "Risk: scope creep in short time. Mitigation: strict MVP cut. Risk: weak demo. Mitigation: focus on one killer flow + clear narrative.",
        "go_to_market": "Launch at the hackathon itself. Collect feedback from 5-10 teams on-site. Use as lead magnet for future events.",
        "key_metrics_to_track": "Time-to-first-brief (< 90s), user satisfaction (NPS after use), % of users who export to pitch deck.",
        "recommended_next_steps": "30d: polish UI + add 3 more presets. 60d: integrate real AMD inference demo. 90d: pilot with 2 university hackathons.",
        "hackathon_tips": "Emphasize the AMD + Fireworks angle in the demo. Show the success score live. Export the brief as your own project README.",
        # Rich starter kit defaults (dataset fallback)
        "recommended_tech_stack": f"Backend: {tech}. Frontend: simple React or plain HTML+JS for speed. LLM: Fireworks AI for brief generation. Compute: AMD GPUs / ROCm for any ML parts or demo acceleration.",
        "architecture_overview": "Simple client-server: static or minimal frontend -> FastAPI backend that calls Fireworks for text + local PyTorch MLP (or the hosted model) for success scoring. Data flows from form -> mapped features -> prediction + LLM prompt.",
        "project_structure": "project/\n  frontend/\n    index.html (or React app)\n  backend/\n    app.py\n    ml/\n      predictor.py\n      prompts.py\n    requirements.txt\n  README.md\n  demo/\n    script.md",
        "mvp_checklist": "Hour 0-1: Set up repo + basic FastAPI + static HTML form.\nHour 1-3: Implement /generate endpoint using Fireworks + basic scoring stub.\nHour 3-6: Wire the trained success model, polish prompt, add score display.\nHour 6-8: Nice UI polish, copy/download buttons, AMD mention in demo.\nHour 8+: Record 3-min demo video, prepare pitch, test on real idea.",
        "starter_readme": f"# {project_name}\n\n{idea}\n\n## Quickstart\n\n1. Install backend deps\n2. Run uvicorn\n3. Open frontend\n4. Enter idea and generate\n\n## Tech\n- Python + FastAPI\n- Fireworks AI\n- AMD GPU ready\n\n## Next\nSee the generated brief for full plan.",
        "pitch_outline": "Slide 1: Problem + One sentence pitch\nSlide 2: Solution overview\nSlide 3: Target users & market\nSlide 4: Demo (live or video)\nSlide 5: Tech (AMD + Fireworks)\nSlide 6: Traction / why win (success score + plan)\nSlide 7: Ask / next steps",
        "demo_script": "0:00-0:30 Problem story\n0:30-1:00 Enter idea in the tool\n1:00-2:00 Show generated brief + rich docs\n2:00-3:00 Show live score + AMD mention\n3:00-4:00 Explain architecture & how you built it fast\n4:00-5:00 Close with why it wins + call to action",
    }