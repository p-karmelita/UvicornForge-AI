from __future__ import annotations

import os

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from ml.dataset import get_dataset_info
from services.brief_service import (
    BriefService,
    GenerateBriefRequest,
    GenerateBriefResponse,
)

load_dotenv()

try:
    import torch
except Exception:  # pragma: no cover
    torch = None

app = FastAPI(
    title="UnicornForge AI Backend",
    description="MVP API for generating structured startup briefs.",
    version="0.2.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["null"],
    allow_origin_regex=r"https?://(localhost|127\.0\.0\.1)(:\d+)?",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
    max_age=3600,
)

FRONTEND_DIR = os.path.join(os.path.dirname(__file__), "..", "frontend")
BRIEF_SERVICE = BriefService()


def _log_startup_status() -> None:
    info = BRIEF_SERVICE.get_model_info()
    print("[unicornforge] torch available:", info["torch_available"])
    if info["torch_available"] and torch is not None:
        print("[unicornforge] torch.__version__:", torch.__version__)
    print("[unicornforge] cuda available:", info["cuda_available"])
    if info["device_name"]:
        print("[unicornforge] device:", info["device_name"])
    print("[unicornforge] success model ready:", info["success_model_ready"])
    print("[unicornforge] xAI configured:", info["xai_configured"])
    print(
        "[unicornforge] dataset loaded:",
        info["dataset_loaded"],
        f"({info['dataset_rows']} rows)" if info["dataset_loaded"] else "",
        f"from {info['dataset_path']}" if info["dataset_path"] else "",
    )


_log_startup_status()


@app.get("/")
async def serve_frontend():
    return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))


@app.post("/generate-brief", response_model=GenerateBriefResponse)
async def generate_brief(payload: GenerateBriefRequest) -> GenerateBriefResponse:
    if not payload.project_idea or not payload.project_idea.strip():
        raise HTTPException(status_code=400, detail="project_idea is required")

    if not get_dataset_info()["loaded"]:
        raise HTTPException(
            status_code=503,
            detail="global_startup_success_dataset.csv not found in backend/",
        )

    return BRIEF_SERVICE.generate(payload)


@app.get("/health")
async def health():
    info = BRIEF_SERVICE.get_model_info()
    return {"ok": True, **info}


@app.get("/model-info")
async def model_info():
    return BRIEF_SERVICE.get_model_info()