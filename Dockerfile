# Dockerfile untuk Development & Production Hybrid

# --- Tahap 1: Base Image ---
# Kita tetap mulai dari base image NVIDIA yang lengkap dengan CUDA tools.
FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

# --- Metadata ---
LABEL maintainer="Aditya Akbar"
LABEL description="Image Dev-Prod untuk aplikasi LLM dengan console, vLLM, RAG, FastAPI."

# --- Konfigurasi Lingkungan & Instalasi Tools Dasar ---
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
# Set cache directory agar berada di volume yang nanti kita mount
ENV TRANSFORMERS_CACHE="/workspace/hf_cache"
ENV HUGGINGFACE_HUB_CACHE="/workspace/hf_cache"

# Install Python, Git, SSH, dan tools dasar lainnya
RUN apt-get update && \
    apt-get install -y \
    python3.10 python3-pip git openssh-server sudo vim curl \
    && apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# --- Buat User untuk Development ---
# Bekerja sebagai root terus menerus bukanlah praktik yang baik.
RUN useradd -m -s /bin/bash -G sudo devuser
# Set password untuk user 'devuser'. Ganti 'yourpassword' dengan password yang aman.
RUN echo "devuser:jakarta321" | chpasswd
# Konfigurasi SSH server
RUN mkdir /var/run/sshd
EXPOSE 22

# --- Setup Lingkungan Kerja ---
WORKDIR /app
# Salin semua file proyek ke dalam direktori kerja
COPY . .

# Berikan kepemilikan folder ke user baru kita
RUN chown -R devuser:devuser /app

# --- Install Dependensi Python ---
# Install dari requirements.txt yang sudah kita perbaiki
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# --- Jalankan Proses Indexing RAG ---
# Ini akan membuat vectorstore langsung di dalam image
RUN python3 rag_processor.py
RUN chown -R devuser:devuser /app/vectorstore

# --- Pindah ke User 'devuser' ---
USER devuser
WORKDIR /app

# --- Port Exposure ---
# Port untuk FastAPI, Streamlit, dan SSH
EXPOSE 8000 8501 22

# --- Perintah Default ---
# Perintah ini akan dijalankan jika Anda 'docker run' tanpa argumen.
# Ini menjalankan FastAPI. Anda bisa override ini untuk mendapatkan shell.
ENV APP_ENV="production"
CMD ["gunicorn", "-w", "1", "-k", "uvicorn.workers.UvicornWorker", "-b", "0.0.0.0:8000", "main:app"]