# ============================================================
# Factory AI — Docker Deployment
# ============================================================
# Build:  docker build -t factory-ai .
# Run:    docker run --gpus all -p 5000:5000 factory-ai
# ============================================================

FROM nvcr.io/nvidia/pytorch:23.12-py3

# Avoid interactive prompts
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libasound2-dev \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Create required directories
RUN mkdir -p logs/snapshots models datasets

# Expose dashboard port
EXPOSE 5000

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:5000/api/system || exit 1

# Default command: run with no-display (headless for Docker)
CMD ["python", "main.py", "--no-display"]
