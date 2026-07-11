from __future__ import annotations

import pandas as pd


def row_to_description(row: pd.Series) -> str:
    """Canonical startup profile text adapted for the new AMD-tailored dataset.
    Handles both old and new column names for backward compatibility.
    """
    name = row.get("Startup Name", "")
    industry = row.get("Industry", "")
    country = row.get("Country", "")
    status = row.get("Funding Stage", "")
    founded_year = row.get("Founded Year", "")
    total_funding = row.get("Total Funding ($)", "") or row.get("Total Funding ($M)", "")
    team_size = row.get("Team Size", "") or row.get("Number of Employees", "")
    revenue = row.get("Monthly Recurring Revenue ($)", "") or row.get("Annual Revenue ($M)", "")
    valuation = row.get("Valuation ($)", "") or row.get("Valuation ($B)", "")
    customers = row.get("Customer Base", "") or row.get("Customer Base (Millions)", "")
    tech_stack = row.get("Backend Tech Stack", "") or row.get("Tech Stack", "")

    parts = []
    if isinstance(name, str) and name:
        parts.append(f"{name} is a startup")
    else:
        parts.append("This startup")

    if isinstance(industry, str) and industry:
        parts.append(f"operating in the {industry} industry")

    if isinstance(country, str) and country:
        parts.append(f"based in {country}")

    sentence = " ".join(parts) + "."
    extra = []

    if pd.notna(founded_year) and founded_year != "":
        extra.append(f"It was founded in {int(float(founded_year))}.")

    if pd.notna(total_funding) and total_funding != "":
        extra.append(f"It has raised around {total_funding} million USD in funding.")

    if pd.notna(team_size) and team_size != "":
        extra.append(f"It employs approximately {team_size} people.")

    if pd.notna(revenue) and revenue != "":
        extra.append(f"It generates about {revenue} million USD in monthly recurring revenue.")

    if pd.notna(valuation) and valuation != "":
        extra.append(f"Its valuation is estimated at {valuation} billion USD.")

    if pd.notna(customers) and customers != "":
        extra.append(f"It serves roughly {customers} million customers.")

    if isinstance(tech_stack, str) and tech_stack:
        extra.append(f"It uses the following tech stack: {tech_stack}.")

    if isinstance(status, str) and status:
        extra.append(f"Its current funding stage is: {status}.")

    success_score = row.get("Success Score")
    if pd.notna(success_score) and success_score != "":
        extra.append(f"Its success score is {float(success_score):.1f}/10.")

    return " ".join([sentence] + extra)


# Legacy function kept only for the old training notebook.
# The current system uses build_hackathon_prompt + sanitized references.
def build_unicornforge_prompt(startup_profile: str) -> str:
    """Legacy prompt template (only for unicornforge_amd_training.ipynb)."""
    return f"""You are an experienced startup advisor and hackathon mentor.

Based on the following real-world startup profile, generate a structured startup brief with 10 sections:
1. Project name
2. One-sentence pitch
3. Problem
4. Solution
5. Target market
6. MVP scope
7. Key features
8. Demo scenario
9. Business model
10. Why this startup could succeed

Startup profile:
{startup_profile}
"""


def build_hackathon_prompt(
    project_idea: str,
    target_users: str,
    industry: str,
    available_time: str,
    available_technologies: str,
    reference_profiles: list[str],
) -> str:
    references = "\n".join(f"- {profile}" for profile in reference_profiles) or "- No references available."
    return f"""You are an experienced startup advisor and hackathon mentor.

Transform the user's hackathon idea into a startup brief grounded in real startup patterns from global_startup_success_dataset.csv.

User idea:
{project_idea}

Target users:
{target_users}

Industry:
{industry}

Available time:
{available_time}

Available technologies:
{available_technologies}

Reference startups from global_startup_success_dataset.csv:
{references}

Return exactly these 15 sections (use clear, professional, investor/hackathon-judge ready language):

Project name:
One-sentence pitch:
Problem:
Solution:
Target market:
MVP scope:
Key features:
Demo scenario:
Business model:
Why this project can win a hackathon:
Risks and mitigations:
Go-to-market strategy:
Key metrics to track in first 3 months:
Recommended next steps (30/60/90 days):
Hackathon winning tips (demo, pitch, sponsor alignment):
"""