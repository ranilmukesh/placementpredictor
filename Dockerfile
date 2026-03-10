# Use official Python 3.11 as base image
FROM python:3.11-slim

# Install system dependencies for scientific compute
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set up non-root user (HF Standard)
RUN useradd -m -u 1000 user
ENV HOME=/home/user \
	PATH=/home/user/.local/bin:$PATH

WORKDIR $HOME/app

# Copy and install basic deps
COPY --chown=user requirements.txt requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir --upgrade -r requirements.txt

# Copy application files
COPY --chown=user . .

# HF Spaces only allows writing to /tmp
RUN mkdir -p /tmp/tmp && chown -R user:user /tmp/tmp
RUN ln -s /tmp/tmp tmp

# Hugging Face PORT is 7860
ENV PORT=7860 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

USER user
EXPOSE 7860

# Run with uvicorn directly (1 worker with multiple threads/concurrency handled by async)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]
