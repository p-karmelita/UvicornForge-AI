# UnicornForge AI

AI-powered startup co-founder that turns rough hackathon ideas into complete, pitch-ready startup briefs with realistic success predictions — in under a minute.

## Stack

- **Backend**: Python / FastAPI + uvicorn, served from `backend/`
- **ML**: PyTorch MLP trained on 10,000 startup rows (`backend/trained_models/startup_success_mlp/`)
- **Frontend**: Static HTML (`frontend/index.html`), served by FastAPI at `/`
- **Optional LLM**: xAI (Grok) via `XAI_API_KEY`; falls back to dataset-based generation if not set

## Running the app

The workflow `Start application` runs:
```
cd backend && python -m uvicorn app:app --host 0.0.0.0 --port 5000
```

The app is available at port 5000.

## Environment variables

| Variable | Purpose | Required |
|---|---|---|
| `XAI_API_KEY` | xAI/Grok API key for richer LLM-generated briefs | No (falls back) |
| `XAI_MODEL` | Model identifier (default: `grok-4.5`) | No |
| `STARTUP_DATASET_PATH` | Override dataset CSV path | No |

## Key endpoints

- `GET /` — serves the frontend
- `POST /generate-brief` — generate a startup brief
- `GET /health` — health + model status
- `GET /model-info` — detailed model/config info
- `GET /model-metrics` — ML model evaluation metrics

## Retraining the model

```bash
cd backend && python train_model.py
```

## User preferences

(none yet)
