# Use official Python 3.11 as base image
FROM python:3.11-slim

# Install system dependencies including timezone data
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    tzdata \
    && rm -rf /var/lib/apt/lists/*

# Set timezone to Asia/Kolkata (IST)
ENV TZ=Asia/Kolkata
RUN ln -snf /usr/share/zoneinfo/Asia/Kolkata /etc/localtime && echo Asia/Kolkata > /etc/timezone

# Set up non-root user (HF Standard)
RUN useradd -m -u 1000 user
ENV HOME=/home/user \
	PATH=/home/user/.local/bin:$PATH \
	MPLCONFIGDIR=/tmp/.matplotlib \
	PYTHONUNBUFFERED=1 \
	PYTHONDONTWRITEBYTECODE=1 \
	TZ=Asia/Kolkata

WORKDIR $HOME/app

# Create matplotlib config directory
RUN mkdir -p /tmp/.matplotlib && chown -R user:user /tmp/.matplotlib

# Copy and install basic deps
COPY --chown=user requirements.txt requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir --upgrade -r requirements.txt

# Copy application files
COPY --chown=user . .

# Run training to generate placement_artifacts.pkl before starting
RUN python train_model.py

# HF Spaces only allows writing to /tmp
RUN mkdir -p /tmp/tmp && chown -R user:user /tmp/tmp
RUN ln -s /tmp/tmp tmp

# Hugging Face PORT is 7860
ENV PORT=7860

USER user
EXPOSE 7860

# Run with uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "4"]
