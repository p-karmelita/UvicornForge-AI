from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Optional

import pandas as pd

BACKEND_DIR = Path(__file__).resolve().parent.parent
DEFAULT_DATASET_PATH = BACKEND_DIR / "global_startup_success_dataset.csv"

CAT_COLUMNS = ["Country", "Industry", "Funding Stage", "Tech Stack"]
NUM_COLUMNS = [
    "Founded Year",
    "Total Funding ($M)",
    "Number of Employees",
    "Annual Revenue ($M)",
    "Customer Base (Millions)",
]
TARGET_COLUMN = "Success Score"
MODEL_COLUMNS = CAT_COLUMNS + NUM_COLUMNS + [TARGET_COLUMN]

FUNDING_STAGE_MULTIPLIERS = {
    "Seed": 0.15,
    "Series A": 0.35,
    "Series B": 0.6,
    "Series C": 0.85,
    "IPO": 1.0,
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
        return {
            "Country": ["Australia", "Brazil", "Canada", "China", "France", "Germany", "India", "Japan", "UK", "USA"],
            "Industry": ["AI", "E-commerce", "EdTech", "Energy", "FinTech", "FoodTech", "Gaming", "Healthcare", "Logistics", "Tech"],
            "Funding Stage": ["IPO", "Seed", "Series A", "Series B", "Series C"],
            "Tech Stack": ["C++, ML", "Java, Spring", "Node.js, React", "PHP, Laravel", "Python, AI"],
        }

    return {col: sorted(df[col].dropna().unique().tolist()) for col in CAT_COLUMNS}


def get_industry_medians(industry: str) -> dict[str, float]:
    df = load_dataset()
    defaults = {
        "Founded Year": 2024.0,
        "Total Funding ($M)": 1.0,
        "Number of Employees": 12.0,
        "Annual Revenue ($M)": 0.5,
        "Customer Base (Millions)": 0.05,
    }
    if df is None:
        return defaults

    subset = df[df["Industry"] == industry]
    if subset.empty:
        subset = df

    return {col: float(subset[col].median()) for col in NUM_COLUMNS}


def find_similar_startup_rows(industry: str, tech_stack: str, limit: int = 3) -> list[pd.Series]:
    df = load_raw_dataset()
    if df is None or "Startup Name" not in df.columns:
        return []

    filtered = df[df["Industry"] == industry] if industry in df["Industry"].values else df
    if tech_stack and "Tech Stack" in filtered.columns:
        tech_filtered = filtered[filtered["Tech Stack"] == tech_stack]
        if not tech_filtered.empty:
            filtered = tech_filtered

    return [row for _, row in filtered.nlargest(limit, TARGET_COLUMN).iterrows()]


def find_similar_startups(industry: str, tech_stack: str, limit: int = 3) -> list[dict[str, str]]:
    return [
        {
            "name": str(row.get("Startup Name", "Unknown")),
            "industry": str(row.get("Industry", industry)),
            "score": f"{float(row.get(TARGET_COLUMN, 0)):.1f}",
        }
        for row in find_similar_startup_rows(industry, tech_stack, limit)
    ]


def row_to_profile(row: pd.Series) -> str:
    name = row.get("Startup Name", "This startup")
    industry = row.get("Industry", "")
    country = row.get("Country", "")
    funding_stage = row.get("Funding Stage", "")
    founded_year = row.get("Founded Year", "")
    total_funding = row.get("Total Funding ($M)", "")
    employees = row.get("Number of Employees", "")
    revenue = row.get("Annual Revenue ($M)", "")
    customers = row.get("Customer Base (Millions)", "")
    tech_stack = row.get("Tech Stack", "")
    success_score = row.get(TARGET_COLUMN, "")

    parts = [f"{name} operates in the {industry} industry based in {country}."]
    if pd.notna(founded_year) and founded_year != "":
        parts.append(f"Founded in {int(float(founded_year))}.")
    if pd.notna(total_funding) and total_funding != "":
        parts.append(f"Raised around {total_funding}M USD ({funding_stage}).")
    if pd.notna(employees) and employees != "":
        parts.append(f"Employs about {employees} people.")
    if pd.notna(revenue) and revenue != "":
        parts.append(f"Annual revenue near {revenue}M USD.")
    if pd.notna(customers) and customers != "":
        parts.append(f"Serves roughly {customers}M customers.")
    if isinstance(tech_stack, str) and tech_stack:
        parts.append(f"Tech stack: {tech_stack}.")
    if pd.notna(success_score) and success_score != "":
        parts.append(f"Success score: {float(success_score):.1f}/9.")

    return " ".join(parts)


def build_dataset_context(industry: str, tech_stack: str, limit: int = 3) -> str:
    rows = find_similar_startup_rows(industry, tech_stack, limit)
    if not rows:
        return "No comparable startup profiles found in global_startup_success_dataset.csv."

    profiles = [f"- {row_to_profile(row)}" for row in rows]
    return "Comparable startups from global_startup_success_dataset.csv:\n" + "\n".join(profiles)