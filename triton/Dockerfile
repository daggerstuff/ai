# Multi-stage build for Pixel Triton deployment
# Stage 1: Builder - Prepare model and dependencies
FROM nvidia/cuda:12.2.0-cudnn8-runtime-ubuntu22.04 AS builder

LABEL maintainer="Pixelated Empathy Team"
LABEL description="Triton Inference Server with Pixel therapeutic model"

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CUDNN_VERSION=8.9.7.29
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3-pip \
    python3-dev \
    wget \
    curl \
    git \
    libssl-dev \
    libffi-dev \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Create Python symlink
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1

# Upgrade pip
RUN python -m pip install --upgrade pip setuptools wheel

# Install Python dependencies for model processing
RUN pip install --no-cache-dir \
    torch==2.2.0 \
    torchvision==0.17.0 \
    torchaudio==2.2.0 \
    transformers==4.38.0 \
    tokenizers==0.15.0 \
    sentencepiece==0.2.0 \
    peft==0.7.0 \
    bitsandbytes==0.41.0 \
    accelerate==0.26.0 \
    safetensors==0.4.1 \
    numpy==1.24.3 \
    pydantic==2.5.0

# Stage 2: Runtime - Triton with Pixel model
FROM nvcr.io/nvidia/tritonserver:24.02-py3

LABEL stage=runtime

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV HOME=/root

# Install additional runtime dependencies
RUN apt-get update && apt-get install -y \
    python3-pip \
    libssl3 \
    ca-certificates \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN python -m pip install --upgrade pip

# Install Python client libraries
RUN pip install --no-cache-dir \
    tritonclient[all]==2.46.0 \
    pydantic==2.5.0 \
    pydantic-settings==2.1.0 \
    numpy==1.24.3 \
    torch==2.2.0 \
    transformers==4.38.0 \
    safetensors==0.4.1

# Create model repository structure
RUN mkdir -p /models/pixel/1 /models/pixel/2
RUN mkdir -p /workspace/scripts /workspace/config

# Copy model configuration
COPY ai/triton/model_repository/pixel/config.pbtxt /models/pixel/config.pbtxt

# Copy model files (will be mounted or built)
# COPY checkpoints/pixel_base_model /models/pixel/1/model

# Copy Python client libraries
COPY ai/triton/pixel_client.py /workspace/pixel_client.py
COPY ai/triton/export_pixel_model.py /workspace/export_pixel_model.py

# Copy startup scripts
COPY ai/triton/scripts/start_triton.sh /workspace/scripts/start_triton.sh
COPY ai/triton/scripts/health_check.sh /workspace/scripts/health_check.sh
COPY ai/triton/scripts/monitor_triton.sh /workspace/scripts/monitor_triton.sh

# Make scripts executable
RUN chmod +x /workspace/scripts/*.sh

# Create Prometheus metrics configuration
RUN mkdir -p /etc/triton && \
    echo "enabled_metrics=[\"gpu_utilization\",\"inference_request_duration\",\"inference_queue_duration\"]" > /etc/triton/metrics.txt

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD /workspace/scripts/health_check.sh

# Set working directory
WORKDIR /workspace

# Expose ports
# 8000 - HTTP
# 8001 - gRPC
# 8002 - Metrics
EXPOSE 8000 8001 8002

# Default entrypoint - start Triton server
ENTRYPOINT ["/workspace/scripts/start_triton.sh"]
CMD []

# Labels for metadata
LABEL version="1.0"
LABEL model="pixel"
LABEL framework="triton"
LABEL cuda="12.2"
LABEL triton="24.02"
