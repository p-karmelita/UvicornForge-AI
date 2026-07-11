from __future__ import annotations

from typing import Optional

from .dataset import build_dataset_context, find_similar_startup_rows, get_industry_medians
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
    medians = get_industry_medians(mapped.industry)
    references = find_similar_startup_rows(mapped.industry, mapped.tech_stack, limit=2)
    reference_names = ", ".join(str(row.get("Startup Name", "a comparable startup")) for row in references)
    top_score = float(references[0].get("Success Score", medians.get("Annual Revenue ($M)", 5))) if references else 5.0

    return {
        "project_name": f"{idea[:48].strip().title()}",
        "one_sentence_pitch": (
            f"{idea[:120].strip()} — a {mapped.industry} product for {target}, "
            f"inspired by high-performing startups such as {reference_names or 'industry leaders'}."
        ),
        "problem": (
            f"{target.capitalize()} in {mapped.industry} struggle to move from a rough concept like "
            f"\"{idea[:80]}\" to a convincing MVP and pitch under tight timelines. "
            f"Existing tools are generic and ignore market patterns visible in real startup outcomes."
        ),
        "solution": (
            f"Build a focused {mapped.industry} MVP using {tech} that solves the core workflow behind "
            f"\"{idea[:80]}\". Position it for a {mapped.funding_stage} stage team in {mapped.country}, "
            f"with architecture choices informed by successful peers in the dataset."
        ),
        "target_market": (
            f"Primary users: {target}. Industry: {mapped.industry}. "
            f"Dataset benchmark for the segment: ~{medians['Number of Employees']:.0f} employees, "
            f"~{medians['Total Funding ($M)']:.1f}M USD funding, "
            f"~{medians['Annual Revenue ($M)']:.1f}M USD revenue."
        ),
        "mvp_scope": (
            f"Deliver one end-to-end flow for \"{idea[:60]}\" within {time_available}. "
            f"Must-have: input form, AI brief generation, success score, and export. "
            f"Use {mapped.tech_stack}. Defer multi-tenant analytics and enterprise integrations."
        ),
        "key_features": (
            f"- Dataset-informed success scoring ({mapped.industry}, {mapped.funding_stage})\n"
            f"- Structured 10-section startup brief for \"{idea[:40]}\"\n"
            f"- Similar startup references from global_startup_success_dataset.csv\n"
            f"- AMD GPU / {tech} ready inference path\n"
            f"- Markdown export for pitch decks"
        ),
        "demo_scenario": (
            f"1) Enter the idea: \"{idea[:50]}\".\n"
            f"2) Show mapped industry ({mapped.industry}) and funding stage ({mapped.funding_stage}).\n"
            f"3) Generate the brief and highlight the predicted success score.\n"
            f"4) Compare with dataset references: {reference_names or 'top peers'}.\n"
            f"5) Copy Markdown into the hackathon pitch."
        ),
        "business_model": (
            f"Freemium for hackathon teams, Pro for founders and accelerators. "
            f"Dataset benchmarks suggest strong {mapped.industry} monetization potential "
            f"around subscription + export upsell, with median revenue near "
            f"{medians['Annual Revenue ($M)']:.1f}M USD in comparable companies."
        ),
        "why_it_can_win": (
            f"The concept aligns with proven {mapped.industry} patterns in "
            f"global_startup_success_dataset.csv. Comparable startups reach scores up to "
            f"{top_score:.1f}/9. The demo combines a real trained model, dataset-backed insights, "
            f"and AMD-ready AI infrastructure — strong fit for the Unicorn Track."
        ),
    }