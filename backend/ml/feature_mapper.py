from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime
from difflib import get_close_matches
from typing import Optional

from .dataset import (
    FUNDING_STAGE_MULTIPLIERS,
    get_category_values,
    get_industry_medians,
)


@dataclass
class MappedFeatures:
    """Features aligned with the new AMD-rich dataset."""
    row: dict[str, object]
    industry: str
    tech_stack: str
    funding_stage: str
    country: str = "USA"
    compute_platform: str = "Own AMD GPU cluster"
    amd_platform: str = "AMD Instinct MI300X"
    factors: dict[str, str] = field(default_factory=dict)


INDUSTRY_KEYWORDS = {
    "AI": ["ai", "machine learning", "ml", "llm", "neural", "deep learning", "gpt"],
    "EdTech": ["education", "edtech", "student", "learning", "course", "school", "university"],
    "FinTech": ["fintech", "finance", "bank", "payment", "crypto", "trading", "invest"],
    "Healthcare": ["health", "medical", "clinic", "patient", "hospital", "wellness"],
    "Gaming": ["game", "gaming", "esport", "player"],
    "E-commerce": ["ecommerce", "e-commerce", "shop", "retail", "marketplace", "store"],
    "Logistics": ["logistics", "delivery", "shipping", "supply chain", "warehouse"],
    "FoodTech": ["food", "restaurant", "meal", "kitchen"],
    "Energy": ["energy", "solar", "battery", "green", "climate", "renewable"],
    "Tech": ["software", "saas", "platform", "tool", "productivity", "developer"],
}

TECH_KEYWORDS = {
    "Python, AI": ["python", "pytorch", "tensorflow", "ai", "ml", "llm", "fireworks", "amd gpu", "rocm"],
    "Node.js, React": ["node", "react", "javascript", "typescript", "next.js", "frontend"],
    "Java, Spring": ["java", "spring", "kotlin"],
    "PHP, Laravel": ["php", "laravel"],
    "C++, ML": ["c++", "cpp", "cuda", "rocm"],
}


def _normalize(text: Optional[str]) -> str:
    return (text or "").strip().lower()


def _match_industry(industry: Optional[str], project_idea: str) -> str:
    categories = get_category_values()["Industry"]
    if industry:
        close = get_close_matches(industry.strip(), categories, n=1, cutoff=0.5)
        if close:
            return close[0]

    haystack = _normalize(f"{industry or ''} {project_idea}")
    # Score actual categories from data using keyword groups
    scores = {name: 0 for name in categories}
    for coarse, keywords in INDUSTRY_KEYWORDS.items():
        for cat in categories:
            if any(kw in cat.lower() for kw in [coarse.lower()] + keywords):
                for keyword in keywords:
                    if keyword in haystack:
                        scores[cat] += 1
                        break

    best = max(scores, key=scores.get)
    if scores[best] > 0:
        return best

    # fallback to first available category
    return categories[0] if categories else "Enterprise SaaS"


def _match_tech_stack(available_technologies: Optional[str]) -> str:
    # Use Backend Tech Stack from new schema (more relevant for AMD/Fireworks)
    cat_values = get_category_values()
    cat_key = "Backend Tech Stack"
    if cat_key in cat_values:
        categories = cat_values[cat_key]
    else:
        # fallback for old or missing
        categories = cat_values.get("Tech Stack", ["Python, FastAPI", "Node.js, Express", "Java, Spring Boot"])
    haystack = _normalize(available_technologies)
    if not haystack:
        return categories[0] if categories else "Python, FastAPI"

    scores = {name: 0 for name in categories}
    for name, keywords in TECH_KEYWORDS.items():
        for keyword in keywords:
            if keyword in haystack:
                # score categories that contain the coarse name
                for cat in categories:
                    if name.split(",")[0].lower() in cat.lower():
                        scores[cat] += 1

    best = max(scores, key=scores.get)
    if scores[best] > 0:
        return best

    close = get_close_matches((available_technologies or "").strip(), categories, n=1, cutoff=0.45)
    return close[0] if close else (categories[0] if categories else "Python, FastAPI")


def _infer_funding_stage(available_time: Optional[str]) -> str:
    text = _normalize(available_time)
    if not text:
        return "Seed"

    if any(token in text for token in ["24", "48", "hour", "weekend", "day", "hackathon"]):
        return "Seed"
    if any(token in text for token in ["week", "2 week", "month", "4 week", "pilot"]):
        return "Series A"
    if any(token in text for token in ["semester", "quarter", "6 month"]):
        return "Series B"
    if any(token in text for token in ["year", "annual"]):
        return "Series C"
    return "Seed"


def _infer_country(target_users: Optional[str], industry: str) -> str:
    text = _normalize(target_users)
    country_aliases = {
        "usa": "USA",
        "united states": "USA",
        "uk": "UK",
        "britain": "UK",
        "germany": "Germany",
        "france": "France",
        "india": "India",
        "china": "China",
        "japan": "Japan",
        "canada": "Canada",
        "brazil": "Brazil",
        "australia": "Australia",
    }
    for alias, country in country_aliases.items():
        if alias in text:
            return country
    return "USA"


def _estimate_ambition_factor(project_idea: str) -> float:
    """Make numeric features vary based on how ambitious the idea sounds.
    This is what allows different project ideas to get meaningfully different success scores.
    """
    text = _normalize(project_idea)
    factor = 1.0

    ambitious_keywords = [
        "global", "worldwide", "scale", "millions", "billion", "enterprise",
        "revolutionary", "advanced", "platform", "ai", "large", "massive",
        "disrupt", "transform", "industry", "next-gen", "cutting-edge"
    ]
    modest_keywords = [
        "simple", "basic", "small", "local", "personal", "todo", "for me",
        "prototype", "weekend", "quick", "minimal", "hobby"
    ]

    for kw in ambitious_keywords:
        if kw in text:
            factor += 0.25
    for kw in modest_keywords:
        if kw in text:
            factor -= 0.35

    # Longer more detailed ideas get a small boost
    if len(text) > 120:
        factor += 0.15
    if len(text) < 40:
        factor -= 0.2

    return max(0.4, min(2.8, factor))


def map_request_to_features(
    project_idea: str,
    target_users: Optional[str] = None,
    industry: Optional[str] = None,
    available_time: Optional[str] = None,
    available_technologies: Optional[str] = None,
    compute_platform: Optional[str] = None,
    amd_platform: Optional[str] = None,
    team_size: Optional[float] = None,
    total_funding: Optional[float] = None,  # in thousands USD, as in form
) -> MappedFeatures:
    """Translate free-form user input into dataset-aligned model features.
    Now supports explicit AMD/Fireworks choices to show their positive impact
    on predicted success score.
    Direct team_size and total_funding from form are used preferentially
    for full 1-10 score range.
    """
    matched_industry = _match_industry(industry, project_idea)
    matched_tech = _match_tech_stack(available_technologies)
    matched_stage = _infer_funding_stage(available_time)
    matched_country = _infer_country(target_users, matched_industry)

    medians = get_industry_medians(matched_industry)
    stage_multiplier = FUNDING_STAGE_MULTIPLIERS.get(matched_stage, 0.2)
    ambition = _estimate_ambition_factor(project_idea)
    effective_mult = stage_multiplier * ambition

    # New schema numeric columns (AMD-aware)
    # Use direct user inputs for team/funding when provided (allows full 1-10 range).
    # Otherwise fall back to medians scaled by stage+ambition.
    if team_size is not None and team_size > 0:
        ts = float(team_size)
    else:
        ts = max(2.0, medians.get("Team Size", 5.0) * effective_mult)

    if total_funding is not None and total_funding > 0:
        tf = float(total_funding) * 1000.0  # assume kUSD as in form
    else:
        tf = max(5.0, medians.get("Total Funding ($)", 150.0) * effective_mult)

    numeric = {
        "Founded Year": float(datetime.now().year),
        "Total Funding ($)": tf,
        "Team Size": ts,
        "Monthly Recurring Revenue ($)": max(0.0, medians.get("Monthly Recurring Revenue ($)", 50.0) * effective_mult),
        "Valuation ($)": max(500.0, medians.get("Valuation ($)", 2000.0) * effective_mult),
        "Customer Base": max(5.0, medians.get("Customer Base", 50.0) * effective_mult),
        "Fireworks AI Credits Used ($, cumulative)": max(0.0, medians.get("Fireworks AI Credits Used ($, cumulative)", 2.0) * effective_mult),
        "Social Media Followers": max(50.0, medians.get("Social Media Followers", 500.0) * effective_mult),
    }

    if ambition > 1.2:
        boost = 0.8 + (ambition - 1) * 0.6
        for k in ["Monthly Recurring Revenue ($)", "Valuation ($)", "Customer Base", "Social Media Followers"]:
            numeric[k] = numeric[k] * boost

    # Compute an overall "level" from provided team, funding, ambition and stage.
    # This ensures that high team/funding/ambitious ideas get high values across the board,
    # allowing the predicted score to span the full 1-10 range even for short available time.
    team_level = (ts / max(1, medians.get("Team Size", 5))) if ts else 0
    fund_level = (tf / 1000 / max(1, medians.get("Total Funding ($)", 150))) if tf else 0
    overall_level = max(stage_multiplier, ambition, team_level, fund_level, 0.1)
    overall_level = min(overall_level, 5.0)  # cap to avoid crazy values

    for k in ["Monthly Recurring Revenue ($)", "Valuation ($)", "Customer Base", "Social Media Followers", "Fireworks AI Credits Used ($, cumulative)"]:
        med = medians.get(k, 50)
        # use at least the level-scaled value, stronger for high ambition
        numeric[k] = max(numeric[k], med * overall_level)

    # Honest defaults — only strong AMD if user or caller explicitly provides it.
    # Neutral default keeps scores believable and the product high-value.
    comp = compute_platform or "Cloud GPU (generic)"
    amd = amd_platform or "—"
    primary_model = "Qwen2.5" if "AMD" in str(amd) or "ROCm" in str(amd) else "Mixtral 8x7B"

    # Build row using new column names + AMD tech choices
    row = {
        "Industry": matched_industry,
        "Funding Stage": matched_stage,
        "Product Stage": "MVP",
        "Backend Tech Stack": matched_tech,
        "Frontend Tech": "React",
        "Compute Platform": comp,
        "AMD Platform Used": amd,
        "Primary Model Used": primary_model,
        **numeric,
    }

    factors = {
        "industry": matched_industry,
        "tech_stack": matched_tech,
        "funding_stage": matched_stage,
        "compute_platform": comp,
        "amd_platform": amd,
        "stage_multiplier": f"{stage_multiplier:.2f}",
        "ambition_factor": f"{ambition:.2f}",
        "team_size": ts,
        "total_funding_k": tf / 1000 if tf else None,
    }

    return MappedFeatures(
        row=row,
        industry=matched_industry,
        tech_stack=matched_tech,
        funding_stage=matched_stage,
        country=matched_country,
        compute_platform=comp,
        amd_platform=amd,
        factors=factors,
    )