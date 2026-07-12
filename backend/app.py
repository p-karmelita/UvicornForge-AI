from __future__ import annotations

import asyncio
import os
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from typing import Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from ml.dataset import get_dataset_info
from ml.evaluation import evaluate_saved_model
from services.brief_service import (
    BriefService,
    GenerateBriefRequest,
    GenerateBriefResponse,
)

load_dotenv()

FRONTEND_DIR = os.path.join(os.path.dirname(__file__), "..", "frontend")
_MODEL_DIR = os.path.join(os.path.dirname(__file__), "trained_models", "startup_success_mlp")

# Populated asynchronously after the event loop starts — never blocks startup.
_brief_service: Optional[BriefService] = None
_init_error: Optional[str] = None
_executor = ThreadPoolExecutor(max_workers=1)


# ---------------------------------------------------------------------------
# Startup helpers
# ---------------------------------------------------------------------------

def _auto_retrain_if_needed() -> None:
    """If model files are absent but the dataset is present, retrain automatically."""
    metadata_missing = not os.path.exists(os.path.join(_MODEL_DIR, "metadata.pkl"))
    weights_missing = not os.path.exists(os.path.join(_MODEL_DIR, "model.pt"))
    if not (metadata_missing or weights_missing):
        return

    dataset_info = get_dataset_info()
    if not dataset_info["loaded"]:
        print(
            "[unicornforge] WARNING: Model files are missing and the dataset is also "
            "unavailable — cannot auto-retrain. Place the dataset CSV in backend/ and "
            "run `cd backend && python train_model.py` to train the model."
        )
        return

    print("[unicornforge] Model files missing — auto-retraining now. This may take a minute...")
    try:
        from ml.training import train_success_model
        result = train_success_model()
        print(
            f"[unicornforge] Auto-retrain complete. "
            f"Model saved to: {result['output_dir']}. "
            f"Final val MSE: {result['final_val_mse']:.4f}"
        )
    except Exception as exc:
        print(
            f"[unicornforge] ERROR: Auto-retrain failed: {exc}. "
            "Run `cd backend && python train_model.py` manually to recover."
        )


def _build_service() -> BriefService:
    """Runs in a thread-pool executor — safe to do blocking I/O here."""
    _auto_retrain_if_needed()
    return BriefService()


def _log_startup_status(service: BriefService) -> None:
    info = service.get_model_info()
    print("[unicornforge] torch available:", info["torch_available"])
    if info["torch_available"]:
        try:
            import torch as _torch
            print("[unicornforge] torch.__version__:", _torch.__version__)
        except Exception:
            pass
    print("[unicornforge] cuda available:", info["cuda_available"])
    if info["device_name"]:
        print("[unicornforge] device:", info["device_name"])
    if info["success_model_ready"]:
        print("[unicornforge] success model ready: True")
    else:
        print(
            "[unicornforge] WARNING: success model is NOT ready. "
            "Success scoring will be skipped. "
            "To fix: run `cd backend && python train_model.py` to retrain the model."
        )
    print("[unicornforge] fireworks configured:", info.get("fireworks_configured"))
    print(
        "[unicornforge] dataset loaded:",
        info["dataset_loaded"],
        f"({info['dataset_rows']} rows)" if info["dataset_loaded"] else "",
        f"from {info['dataset_path']}" if info["dataset_path"] else "",
    )


async def _init_service_task() -> None:
    """Background task: loads the model without blocking the event loop."""
    global _brief_service, _init_error
    try:
        loop = asyncio.get_running_loop()
        service = await loop.run_in_executor(_executor, _build_service)
        _brief_service = service
        _log_startup_status(service)
    except Exception as exc:
        _init_error = str(exc)
        print(f"[unicornforge] Failed to initialize BriefService: {exc}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Schedule model loading as a background task so the server starts
    # accepting requests (and passes health checks) immediately.
    asyncio.create_task(_init_service_task())
    yield
    _executor.shutdown(wait=False)


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="UnicornForge AI Backend",
    description="MVP API for generating structured startup briefs.",
    version="0.2.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
    max_age=3600,
)


def _get_service() -> BriefService:
    """Return the initialized service, or raise an appropriate HTTP error."""
    if _brief_service is None:
        if _init_error:
            raise HTTPException(status_code=500, detail=f"Service failed to initialize: {_init_error}")
        raise HTTPException(status_code=503, detail="Service is initializing, please retry in a moment.")
    return _brief_service


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

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
    return _get_service().generate(payload)


@app.get("/health")
async def health():
    if _brief_service is None:
        return {"ok": True, "status": "initializing"}
    return {"ok": True, **_brief_service.get_model_info()}


@app.get("/model-info")
async def model_info():
    if _brief_service is None:
        return {"status": "initializing", "success_model_ready": False}
    return _brief_service.get_model_info()


@app.get("/model-metrics")
async def model_metrics():
    service = _get_service()
    if not service.get_model_info()["success_model_ready"]:
        raise HTTPException(status_code=503, detail="Success model is not loaded")
    try:
        return evaluate_saved_model()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
