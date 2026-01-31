FROM nvidia/cuda:11.8-devel-ubuntu20.04

# Install Python and dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip3 install -r requirements.txt

# Copy model and scripts
COPY wayfarer-finetuned/ ./wayfarer-finetuned/
COPY inference.py .
COPY api_server.py .

# Expose port
EXPOSE 8000

# Default command
CMD ["python3", "api_server.py"]
