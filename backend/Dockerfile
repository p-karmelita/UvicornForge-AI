# CPU-friendly Dockerfile for UnicornForge AI backend
# For ROCm/AMD GPU support, use the provided Dockerfile.rocm or replace base image with a ROCm-enabled PyTorch image.

FROM python:3.12-slim
WORKDIR /app

# Install runtime deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY backend/requirements.txt ./backend/requirements.txt 2>/dev/null || true
RUN pip install --no-cache-dir -r backend/requirements.txt 2>/dev/null || \
    pip install --no-cache-dir fastapi uvicorn pydantic

COPY backend /app/backend

EXPOSE 8000
CMD ["python", "-m", "uvicorn", "backend.app:app", "--host", "0.0.0.0", "--port", "8000"]
