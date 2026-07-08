```markdown
# UnicornForge AI

UnicornForge AI is an AI-powered startup co‑founder that helps hackathon teams, founders, students, and innovation teams turn rough ideas into complete startup briefs, MVP plans, and demo strategies within minutes.

## What It Does

1. You enter a rough project idea plus optional context:
   - target users
   - industry
   - available time
   - available technologies

2. UnicornForge AI generates a structured startup brief:
   - project name
   - one‑sentence pitch
   - problem & solution
   - target market
   - MVP scope
   - key features
   - demo scenario
   - business model
   - “why this can win a hackathon”

The goal is to help teams think like startup founders, not just builders.

## Why It Fits AMD Hackathon — Track 3 (Unicorn Track)

- **Creativity & originality** – an AI “startup mentor” instead of a generic chatbot.  
- **Completeness** – clear idea → generated brief → ready for pitch / demo.  
- **AMD platforms** – designed to run open‑source models on AMD GPUs and/or use Fireworks AI API.  
- **Product/market potential** – useful for hackathons, accelerators, universities, and innovation teams.

## High-Level Architecture

- **Frontend** – single‑page web UI for entering the idea and viewing the generated brief.
- **Backend (Python)** – API endpoint that:
  - accepts user input,
  - builds a structured prompt,
  - calls the AI model (AMD GPU / Fireworks AI),
  - returns a JSON startup brief.

## Quick Start (planned)

- `backend/app.py` — Python API (FastAPI/Flask)  
- `frontend/index.html` — simple UI calling the backend

> This repository is being built during the AMD Hackathon. Initial focus: a working MVP that turns a rough idea into a high‑quality startup brief in one click.
```
