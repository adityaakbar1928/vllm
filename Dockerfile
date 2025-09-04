# Backend Dockerfile
FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/workspace/hf_cache
ENV TRANSFORMERS_CACHE=/workspace/hf_cache
ENV HUGGINGFACE_HUB_CACHE=/workspace/hf_cache
ENV TORCH_CUDA_ALLOC_CONF=max_split_size_mb:256

RUN apt-get update && apt-get install -y \
    python3.10 python3.10-venv python3-pip git curl && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN python3 -m pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# kode
COPY app/ ./app/

# workspace untuk cache model & vectorstore
RUN mkdir -p /workspace/data /workspace/vectorstore && \
    chown -R root:root /workspace

# default env (override di Helm/Deployment)
ENV APP_ENV=production \
    MODEL_NAME=meta-llama/Llama-3.1-8B-Instruct \
    GPU_MEM_UTIL=0.30 \
    MAX_MODEL_LEN=8192 \
    TEMP=0.7 TOP_P=0.95 MAX_TOKENS=1024 \
    SHOW_TPS=true \
    RAG_MODE=off \
    API_KEY=jakarta321

EXPOSE 8000
CMD ["gunicorn","-w","1","-k","uvicorn.workers.UvicornWorker","-b","0.0.0.0:8000","app.main:app"]
