from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Optional

import pandas as pd

CAT_COLUMNS = ["Country", "Industry", "Funding Stage", "Tech Stack"]
NUM_COLUMNS = [
    "Founded Year",
    "Total Funding ($M)",
    "Number of Employees",
    "Annual Revenue ($M)",
    "Customer Base (Millions)",
]
TARGET_COLUMN = "Success Score"
USED_COLUMNS = CAT_COLUMNS + NUM_COLUMNS + [TARGET_COLUMN]

FUNDING_STAGE_MULTIPLIERS = {
    "Seed": 0.15,
    "Series A": 0.35,
    "Series B": 0.6,
    "Series C": 0.85,
    "IPO": 1.0,
}


def _resolve_dataset_path() -> Optional[Path]:
    env_path = os.getenv("STARTUP_DATASET_PATH")
    if env_path and Path(env_path).exists():
        return Path(env_path)

    local_candidates = [
        Path(__file__).resolve().parent.parent / "global_startup_success_dataset.csv",
        Path(__file__).resolve().parent.parent / "global-startup-success-dataset" / "global_startup_success_dataset.csv",
    ]
    for candidate in local_candidates:
        if candidate.exists():
            return candidate

    cache_glob = Path.home() / ".cache" / "kagglehub" / "datasets" / "hamnakaleemds" / "global-startup-success-dataset"
    if cache_glob.exists():
        for version_dir in sorted(cache_glob.glob("versions/*"), reverse=True):
            csv_path = version_dir / "global_startup_success_dataset.csv"
            if csv_path.exists():
                return csv_path

    return None


@lru_cache(maxsize=1)
def load_dataset() -> Optional[pd.DataFrame]:
    path = _resolve_dataset_path()
    if path is None:
        return None

    df = pd.read_csv(path)
    missing = [col for col in USED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Dataset at {path} is missing columns: {missing}")

    return df[USED_COLUMNS].dropna().reset_index(drop=True)


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

    medians = {}
    for col in NUM_COLUMNS:
        medians[col] = float(subset[col].median())
    return medians


def find_similar_startups(industry: str, tech_stack: str, limit: int = 3) -> list[dict[str, str]]:
    df = load_dataset()
    if df is None:
        return []

    full_path = _resolve_dataset_path()
    if full_path is None:
        return []

    raw = pd.read_csv(full_path)
    if "Startup Name" not in raw.columns:
        return []

    filtered = raw[raw["Industry"] == industry] if industry in raw["Industry"].values else raw
    if tech_stack and "Tech Stack" in filtered.columns:
        tech_filtered = filtered[filtered["Tech Stack"] == tech_stack]
        if not tech_filtered.empty:
            filtered = tech_filtered

    top = filtered.nlargest(limit, TARGET_COLUMN)
    results = []
    for _, row in top.iterrows():
        results.append(
            {
                "name": str(row.get("Startup Name", "Unknown")),
                "industry": str(row.get("Industry", industry)),
                "score": f"{float(row.get(TARGET_COLUMN, 0)):.1f}",
            }
        )
    return results