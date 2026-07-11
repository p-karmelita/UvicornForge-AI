from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Optional

import pandas as pd

from .prompts import row_to_description

BACKEND_DIR = Path(__file__).resolve().parent.parent
DEFAULT_DATASET_PATH = BACKEND_DIR / "global_startup_success_dataset.csv"

# New dataset schema (tailored for AMD hackathon project)
# Focus on realistic early-stage metrics + AMD sponsor tech choices
CAT_COLUMNS = [
    "Industry",
    "Funding Stage",
    "Product Stage",
    "Backend Tech Stack",
    "Frontend Tech",
    "Compute Platform",      # "Own AMD GPU cluster" vs "Fireworks AI API" — key for AMD story
    "AMD Platform Used",     # MI300X, MI250, Radeon, etc.
    "Primary Model Used",
]

NUM_COLUMNS = [
    "Founded Year",
    "Total Funding ($)",
    "Team Size",                          # much more realistic than old "employees"
    "Monthly Recurring Revenue ($)",
    "Valuation ($)",
    "Customer Base",
    "Fireworks AI Credits Used ($, cumulative)",  # sponsor signal
    "Social Media Followers",
]

TARGET_COLUMN = "Success Score"
MODEL_COLUMNS = CAT_COLUMNS + NUM_COLUMNS + [TARGET_COLUMN]

FUNDING_STAGE_MULTIPLIERS = {
    "Pre-Seed": 0.10,
    "Bootstrapped": 0.12,
    "Angel": 0.20,
    "Seed": 0.30,
    "Series A": 0.50,
    "Series B": 0.70,
}


def get_dataset_path() -> Optional[Path]:
    env_path = os.getenv("STARTUP_DATASET_PATH")
    if env_path:
        path = Path(env_path)
        if path.exists():
            return path

    if DEFAULT_DATASET_PATH.exists():
        return DEFAULT_DATASET_PATH

    return None


@lru_cache(maxsize=1)
def load_raw_dataset() -> Optional[pd.DataFrame]:
    path = get_dataset_path()
    if path is None:
        return None
    return pd.read_csv(path)


@lru_cache(maxsize=1)
def load_dataset() -> Optional[pd.DataFrame]:
    df = load_raw_dataset()
    if df is None:
        return None

    missing = [col for col in MODEL_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Dataset at {get_dataset_path()} is missing columns: {missing}")

    return df[MODEL_COLUMNS].dropna().reset_index(drop=True)


def get_dataset_info() -> dict:
    path = get_dataset_path()
    df = load_raw_dataset()
    return {
        "loaded": df is not None,
        "path": str(path) if path else None,
        "rows": int(len(df)) if df is not None else 0,
        "columns": list(df.columns) if df is not None else [],
    }


def get_category_values() -> dict[str, list[str]]:
    df = load_dataset()
    if df is None:
        # Realistic defaults based on new AMD-tailored dataset
        defaults = {
            "Industry": ["FinTech AI", "Climate & Energy AI", "Gaming AI", "Enterprise SaaS", "EdTech AI", "Logistics & Supply Chain AI"],
            "Funding Stage": ["Pre-Seed", "Bootstrapped", "Angel", "Seed", "Series A"],
            "Product Stage": ["Idea / Concept", "Prototype", "MVP", "Private Beta", "Public Beta"],
            "Backend Tech Stack": ["Node.js, Express", "FastAPI", "Java, Spring Boot", "Python, FastAPI", "TypeScript, NestJS"],
            "Frontend Tech": ["React", "Next.js", "Svelte", "Vue", "React Native", "Flutter (Dart)"],
            "Compute Platform": ["Own AMD GPU cluster", "Fireworks AI API"],
            "AMD Platform Used": ["AMD Instinct MI300X", "ROCm on MI300X cluster", "AMD Instinct MI250", "AMD Radeon PRO W7900"],
            "Primary Model Used": ["Qwen2.5", "Llama 3.1", "DeepSeek", "Mixtral"],
        }
        # alias for backward compatibility in old code
        defaults["Tech Stack"] = defaults["Backend Tech Stack"]
        return defaults

    cats = {col: sorted(df[col].dropna().unique().tolist()) for col in CAT_COLUMNS}
    # alias for old code that still looks for "Tech Stack"
    if "Backend Tech Stack" in cats:
        cats["Tech Stack"] = cats["Backend Tech Stack"]
    return cats


def get_industry_medians(industry: str) -> dict[str, float]:
    """Return median values for numeric columns. Uses realistic early-stage numbers."""
    df = load_dataset()
    defaults = {
        "Founded Year": 2024.0,
        "Total Funding ($)": 150.0,
        "Team Size": 5.0,
        "Monthly Recurring Revenue ($)": 80.0,
        "Valuation ($)": 2500.0,
        "Customer Base": 120.0,
        "Fireworks AI Credits Used ($, cumulative)": 5.0,
        "Social Media Followers": 800.0,
    }
    if df is None:
        return defaults

    subset = df[df["Industry"] == industry] if "Industry" in df.columns else df
    if subset.empty:
        subset = df

    return {col: float(subset[col].median()) for col in NUM_COLUMNS if col in subset.columns}


def find_similar_startup_rows(industry: str, tech_stack: str, limit: int = 3) -> list[pd.Series]:
    """Find top similar startups. Prefers matching on Backend Tech Stack from new dataset."""
    df = load_raw_dataset()
    if df is None or "Startup Name" not in df.columns:
        return []

    filtered = df[df["Industry"] == industry] if "Industry" in df.columns and industry in df["Industry"].values else df

    # Use new column name
    tech_col = "Backend Tech Stack"
    if tech_stack and tech_col in filtered.columns:
        tech_filtered = filtered[filtered[tech_col] == tech_stack]
        if not tech_filtered.empty:
            filtered = tech_filtered

    return [row for _, row in filtered.nlargest(limit, TARGET_COLUMN).iterrows()]


def find_similar_startups(industry: str, tech_stack: str, limit: int = 3) -> list[dict[str, str]]:
    """Return anonymized similar startups for user display.

    The raw dataset uses synthetic names (Startup_N) and unrealistic scale,
    so we show generic labels + only the success signal.
    """
    rows = find_similar_startup_rows(industry, tech_stack, limit)
    results = []
    for i, row in enumerate(rows, 1):
        score = float(row.get(TARGET_COLUMN, 0))
        tech = str(row.get("Backend Tech Stack", "") or row.get("Tech Stack", "")).strip()
        label = f"Top {industry} performer #{i}"
        if tech:
            label += f" ({tech.split(',')[0].strip()})"
        results.append({
            "name": label,
            "industry": str(row.get("Industry", industry)),
            "score": f"{score:.1f}",
        })
    return results


def build_dataset_context(industry: str, tech_stack: str, limit: int = 3) -> str:
    rows = find_similar_startup_rows(industry, tech_stack, limit)
    if not rows:
        return "No comparable startup profiles found in global_startup_success_dataset.csv."

    profiles = [f"- {row_to_description(row)}" for row in rows]
    return "Comparable startups from global_startup_success_dataset.csv:\n" + "\n".join(profiles)