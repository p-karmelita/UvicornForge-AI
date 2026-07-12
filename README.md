# UnicornForge AI

**AI-powered startup co-founder that turns rough hackathon ideas into complete, pitch-ready startup briefs with realistic success predictions — in under a minute.**

Built specifically for the **AMD Hackathon – Unicorn Track**.

---

## ✨ What It Does

1. Enter a rough project idea + optional context:
   - Target users
   - Industry
   - Available time
   - Available technologies / stack
   - Team size
   - Total funding (k$)

2. UnicornForge AI instantly returns:
   - A **structured, professional startup brief** (problem, solution, market, MVP scope, features, demo scenario, business model, why it can win, etc.)
   - A **predicted startup success score** (1–10) with human-readable label
   - Key contributing **factors**
   - **Similar high-performing startups** from the dataset
   - Actionable **improvement suggestions** tied to the score
   - Rich technical kickoff materials (recommended tech stack, architecture, project structure, MVP checklist, starter README, pitch outline, demo script)

3. One-click **Copy as Markdown** or export the entire output.

The goal is to help teams think and communicate like founders, not just builders.

---

## 🚀 Key Features

| Feature                        | Description                                                                 |
|--------------------------------|-----------------------------------------------------------------------------|
| **Rich Brief Generation**      | Full structured output with 15+ high-value sections                         |
| **Predictive Success Model**   | Custom PyTorch MLP trained on 10,000 realistic early-stage startup rows    |
| **AMD-Aware Scoring**          | Explicit bonuses when you mention AMD GPUs, ROCm, Instinct MI300X/MI250     |
| **Fireworks AI Integration**   | High-quality LLM generation (graceful fallback to dataset-based generator) |
| **Honest Scoring**             | Only rewards real signals (team size, funding, ambition, tech choices)      |
| **Live System Status**         | Shows model readiness, GPU/AMD availability, Fireworks config, dataset info |
| **Demo Presets**               | One-click examples for AMD + Fireworks + classic hackathon ideas            |
| **Animated Score Display**     | Beautiful count-up animation + progress bar + label                         |
| **Improvement Suggestions**    | Context-aware tips generated from the prediction + factors                  |
| **Fully Local-first**          | Works without Fireworks key (falls back to dataset generator)               |

---

## 🏗️ Architecture

```
Frontend (index.html)
        ↓
FastAPI Backend (app.py)
        ↓
BriefService
   ├── FireworksClient (LLM)  → structured brief
   └── SuccessPredictor (PyTorch MLP)
           └── feature_mapper → vector → model → score + factors
```

- **Frontend**: Single-file, modern dark UI with AMD red accents.
- **Backend**: FastAPI serving the SPA + `/generate-brief` endpoint.
- **ML Layer**:
  - `SuccessScoreMLP` (512→256→128→64→1 with BatchNorm + Dropout)
  - Trained with AdamW + MSE on engineered 10k-row dataset
  - Realistic AMD / Fireworks signals built into features
- **LLM**: Fireworks AI (OpenAI-compatible) with strong fallback.

---

## 🛠 Tech Stack

**Backend**
- Python 3.12 + FastAPI + Uvicorn
- PyTorch (CPU / CUDA / ROCm)
- scikit-learn, pandas, numpy
- Pydantic, python-dotenv, requests

**Frontend**
- Vanilla HTML + CSS + JavaScript (single file)
- Responsive, dark theme with smooth animations

**ML / Data**
- Custom 10,000-row dataset (`global_startup_success_dataset.csv`)
- Feature engineering aware of AMD platforms and Fireworks usage
- Trained SuccessScoreMLP model (`trained_models/startup_success_mlp/`)

**Optional**
- Fireworks AI API (for premium brief generation)
- AMD ROCm + Instinct GPUs (for local inference acceleration)

---

## ⚡ Quick Start

### 1. Backend

```bash
cd UvicornForge-AI/backend

# Recommended
./run_local.sh

# Or manually
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

### 2. Open the App

The backend serves the frontend at `http://localhost:8000`.

### 3. Generate Your First Brief

1. Click a **demo preset** (AI Hackathon, AMD ML, or Fireworks Studio)
2. Adjust **Team Size** and **Total Funding** (optional but powerful)
3. Click **Generate Startup Brief**
4. Watch the animated score + rich brief appear

### 4. (Optional) Add Fireworks API Key

Create `backend/.env`:

```env
FIREWORKS_API_KEY=fw_your_key_here
FIREWORKS_MODEL=accounts/fireworks/models/llama-v3p1-8b-instruct
```

Without a key the system uses the high-quality dataset-based generator.

---

## 📡 API Endpoints

| Method | Endpoint              | Description                              |
|--------|-----------------------|------------------------------------------|
| POST   | `/generate-brief`     | Main endpoint — returns full brief + score |
| GET    | `/model-info`         | Live system status (model, GPU, Fireworks, metrics) |
| GET    | `/model-metrics`      | Freshly evaluated model performance      |
| GET    | `/health`             | Simple health + model info               |
| GET    | `/`                   | Serves the frontend SPA                  |

---

## 🧠 The Success Prediction Model

- **Model**: `SuccessScoreMLP` — 5-layer feed-forward network
- **Training**: 80/20 split, AdamW, ReduceLROnPlateau, MSE loss
- **Performance** (typical): R² ≈ 0.85, low MAE on 1–10 scale
- **Inputs that matter**:
  - Team size & total funding (direct from form)
  - Industry, tech stack, funding stage
  - Ambition signals from idea text
  - **Explicit AMD / Fireworks mentions** → measurable score lift
- **Retraining**:

```bash
cd backend
python train_model.py --epochs 20
```

Model artifacts are saved to `trained_models/startup_success_mlp/`.

---

## 📊 Dataset

- **File**: `backend/global_startup_success_dataset.csv` (10,000 rows)
- Tailored for the hackathon with realistic early-stage metrics
- Includes sponsor-aligned signals:
  - Compute Platform (`Own AMD GPU cluster` vs `Fireworks AI API`)
  - AMD Platform Used
  - Fireworks AI Credits Used ($, cumulative)

---

## 🔥 AMD + Fireworks Alignment

UnicornForge AI was built for the **Unicorn Track**:

- Rewards **real** use of AMD technology in the scoring model
- Supports local AMD GPU inference (CUDA/ROCm)
- Integrates Fireworks AI for high-quality generation
- Produces immediately usable pitch & demo artifacts
- Full end-to-end engineering showcase (data → model → API → beautiful UI)

---

## 📁 Project Structure

```
UvicornForge-AI/
├── backend/
│   ├── app.py                    # FastAPI application
│   ├── services/
│   │   └── fireworks_client.py   # LLM client + fallback
│   ├── ml/
│   │   ├── predictor.py          # SuccessPredictor (PyTorch)
│   │   ├── brief_service.py      # Orchestrates brief + prediction
│   │   ├── feature_mapper.py     # Input → model features
│   │   ├── training.py           # Training pipeline
│   │   ├── model.py              # SuccessScoreMLP definition
│   │   └── prompts.py            # Prompt engineering
│   ├── trained_models/
│   ├── global_startup_success_dataset.csv
│   ├── train_model.py            # CLI training entrypoint
│   └── run_local.sh
├── frontend/
│   └── index.html                # Complete single-file UI
├── docs/                         # Product specs, wireframes, scripts
├── notebooks/                    # AMD + AI workshop notebooks
└── README.md
```

---

## 🛠 Development

- Run with hot reload: `python -m uvicorn app:app --reload`
- Check live model status: `GET /model-info`
- Retrain model: `python train_model.py`
- Frontend is pure static HTML — edit `frontend/index.html` directly

### Docker

```bash
# CPU
docker build -f backend/Dockerfile -t unicornforge .

# ROCm (AMD)
docker build -f backend/Dockerfile.rocm -t unicornforge-rocm .
```

---

## 🔐 Environment Variables

| Variable               | Purpose                              | Required |
|------------------------|--------------------------------------|----------|
| `FIREWORKS_API_KEY`    | Fireworks AI API key (starts with `fw_`) | No (falls back) |
| `FIREWORKS_MODEL`      | Model identifier                     | No       |
| `FIREWORKS_API_BASE`   | API base URL                         | No       |

---

## 🎯 Why This Project Fits the Unicorn Track

- **Creativity**: An actual AI startup mentor, not another generic chatbot
- **Completeness**: From vague idea → full brief + score + pitch artifacts
- **Technical depth**: Real trained PyTorch model, feature engineering, production API
- **Sponsor alignment**: Deep, honest integration of AMD + Fireworks signals
- **Real utility**: Something hackathon teams and founders can actually use

---

## 📝 License

This project was created during the AMD Hackathon.

Feel free to use the code and ideas for your own hackathons, demos, or products.

---

**Built with ❤️ + PyTorch + AMD GPUs + Fireworks AI**

*Give it a rough idea. Get a startup-ready brief + honest score. Ship faster.*

