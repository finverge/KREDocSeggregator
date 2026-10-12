# ==============================================================================
# Dockerfile for KYC Document Sorter (VLM-Only)
# Multi-stage/Slim production build with non-root runtime security
# ==============================================================================

FROM python:3.11-slim-bookworm AS runtime

# Prevent Python from writing .pyc files and enable unbuffered logging output
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive

# Install minimal OS dependencies for security certificates and TLS
RUN apt-get update && \
    apt-get install -y --no-install-recommends ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# Create a dedicated non-root user and group
RUN groupadd -g 10001 appgroup && \
    useradd -u 10001 -g appgroup -s /bin/bash -m appuser

# Set working directory
WORKDIR /app

# Pre-create application and volume mount points with correct ownership
RUN mkdir -p /app /data/input /data/output /data/journal /data/logs && \
    chown -R appuser:appgroup /app /data && \
    chmod -R 775 /data

# Copy dependency definition first to maximize Docker layer caching
COPY --chown=appuser:appgroup requirements.txt /app/requirements.txt

# Install dependencies in the system environment
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application source code and default configuration
COPY --chown=appuser:appgroup main.py /app/main.py
COPY --chown=appuser:appgroup config.yaml /app/config.yaml
COPY --chown=appuser:appgroup kyc_sorter/ /app/kyc_sorter/

# Switch to non-root runtime user
USER appuser

# Default runtime configuration environment variables
ENV KYC_INPUT_DIR=/data/input \
    KYC_OUTPUT_DIR=/data/output \
    KYC_JOURNAL_PATH=/data/journal/kyc_journal.db \
    KYC_LOG_DIR=/data/logs \
    KYC_OPERATION=copy \
    KYC_MAX_WORKERS=2 \
    KYC_BATCH_SIZE=100 \
    KYC_CONFIDENCE_THRESHOLD=0.80 \
    KYC_DRY_RUN=false \
    KYC_VLM_ENDPOINT=http://host.docker.internal:11434/api/chat \
    KYC_VLM_MODEL=qwen3-vl:4b-instruct \
    KYC_VLM_TIMEOUT=300 \
    KYC_VLM_MAX_RETRIES=3 \
    KYC_VLM_KEEP_ALIVE=24h \
    KYC_VLM_NUM_CTX=8192 \
    KYC_SINGLE_PASS_MODE=true \
    KYC_WATCHER_MAX_QUEUE=1000

# Declare persistent data volumes
VOLUME ["/data/input", "/data/output", "/data/journal", "/data/logs"]

# Entrypoint delegates to main.py, allowing any CLI arguments to be passed at runtime
ENTRYPOINT ["python", "main.py"]

# Default execution: run batch mode using environment variables
CMD []
