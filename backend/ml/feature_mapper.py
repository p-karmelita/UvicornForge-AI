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

    # fallback: pick first or "Enterprise SaaS"
    return categories[0] if categories else "Enterprise SaaS"


def _match_tech_stack(available_technologies: Optional[str]) -> str:
    # Use Backend Tech Stack from new schema (more relevant for AMD/Fireworks)
    cat_key = "Backend Tech Stack"
    try:
        categories = get_category_values()[cat_key]
    except KeyError:
        categories = ["Python, FastAPI", "Node.js, Express", "Java, Spring Boot"]
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


def map_request_to_features(
    project_idea: str,
    target_users: Optional[str] = None,
    industry: Optional[str] = None,
    available_time: Optional[str] = None,
    available_technologies: Optional[str] = None,
) -> MappedFeatures:
    """Translate free-form user input into dataset-aligned model features."""
    matched_industry = _match_industry(industry, project_idea)
    matched_tech = _match_tech_stack(available_technologies)
    matched_stage = _infer_funding_stage(available_time)
    matched_country = _infer_country(target_users, matched_industry)

    medians = get_industry_medians(matched_industry)
    multiplier = FUNDING_STAGE_MULTIPLIERS.get(matched_stage, 0.2)

    # New schema numeric columns (AMD-aware)
    numeric = {
        "Founded Year": float(datetime.now().year),
        "Total Funding ($)": max(5.0, medians.get("Total Funding ($)", 150.0) * multiplier),
        "Team Size": max(2.0, medians.get("Team Size", 5.0) * multiplier),
        "Monthly Recurring Revenue ($)": max(0.0, medians.get("Monthly Recurring Revenue ($)", 50.0) * multiplier),
        "Valuation ($)": max(500.0, medians.get("Valuation ($)", 2000.0) * multiplier),
        "Customer Base": max(5.0, medians.get("Customer Base", 50.0) * multiplier),
        "Fireworks AI Credits Used ($, cumulative)": max(0.0, medians.get("Fireworks AI Credits Used ($, cumulative)", 2.0) * multiplier),
        "Social Media Followers": max(50.0, medians.get("Social Media Followers", 500.0) * multiplier),
    }

    # Build row using new column names + AMD tech choices
    row = {
        "Industry": matched_industry,
        "Funding Stage": matched_stage,
        "Product Stage": "MVP",                    # sensible default for hackathon
        "Backend Tech Stack": matched_tech,
        "Frontend Tech": "React",                  # default
        "Compute Platform": "Own AMD GPU cluster", # default — this is powerful for scoring
        "AMD Platform Used": "AMD Instinct MI300X",
        "Primary Model Used": "Qwen2.5",
        **numeric,
    }

    factors = {
        "industry": matched_industry,
        "tech_stack": matched_tech,
        "funding_stage": matched_stage,
        "compute_platform": row["Compute Platform"],
        "amd_platform": row["AMD Platform Used"],
        "stage_multiplier": f"{multiplier:.2f}",
    }

    return MappedFeatures(
        row=row,
        industry=matched_industry,
        tech_stack=matched_tech,
        funding_stage=matched_stage,
        country=matched_country,
        compute_platform=row.get("Compute Platform", "Own AMD GPU cluster"),
        amd_platform=row.get("AMD Platform Used", "AMD Instinct MI300X"),
        factors=factors,
    )