# ============================================================================
# Mycelial Agent Engine (MAE) - Docker Image
# ============================================================================
#
# Multi-stage build for optimized production image
#
# Build: docker build -t mae:latest .
# Run:   docker run -it mae:latest
#
# ============================================================================

# ==========================================================================
# Stage 1: Builder - Install dependencies
# ==========================================================================
FROM python:3.11-slim as builder

# Set working directory
WORKDIR /build

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    make \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ==========================================================================
# Stage 2: Runtime - Create lean production image
# ==========================================================================
FROM python:3.11-slim

# Metadata
LABEL maintainer="your-email@example.com"
LABEL description="Mycelial Agent Engine - Decentralized Multi-Agent Learning System"
LABEL version="1.0.0"

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    MAE_CONFIG_PATH=/app/config/config.yaml

# Create app user (don't run as root)
RUN useradd -m -u 1000 -s /bin/bash mae && \
    mkdir -p /app /app/data /app/logs /app/simulation_results && \
    chown -R mae:mae /app

# Set working directory
WORKDIR /app

# Install runtime system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy Python packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code
COPY --chown=mae:mae src/ /app/src/
COPY --chown=mae:mae config/ /app/config/
COPY --chown=mae:mae run_simulation.py /app/
COPY --chown=mae:mae run_live.py /app/

# Switch to app user
USER mae

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import sys; sys.exit(0)"

# Default command (can be overridden)
CMD ["python", "run_simulation.py"]
