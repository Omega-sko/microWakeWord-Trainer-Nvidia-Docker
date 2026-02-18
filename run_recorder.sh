#!/usr/bin/env bash
set -euo pipefail

ROOTDIR="$(dirname "$(realpath "$0")")"

# Training convention
DATA_DIR="${DATA_DIR:-/data}"
HOST="${REC_HOST:-0.0.0.0}"
PORT="${REC_PORT:-8888}"

echo "microWakeWord Recorder (Docker)"
echo "-> ROOTDIR:  ${ROOTDIR}"
echo "-> DATA_DIR: ${DATA_DIR}"
echo "-> URL:      http://localhost:${PORT}/"

mkdir -p "${DATA_DIR}"

# -----------------------------
# Check for required packages
# -----------------------------
echo "Checking for required recorder packages (fastapi, uvicorn, python-multipart)..."
if ! python3 -c "import fastapi, uvicorn, multipart" 2>/dev/null; then
  echo "ERROR: Required recorder packages not found in system Python."
  echo "Please ensure the Docker image includes fastapi, uvicorn, and python-multipart."
  echo "Add to Dockerfile: RUN pip install fastapi uvicorn[standard] python-multipart"
  exit 1
fi
echo "✅ Recorder dependencies found in system Python"

# -----------------------------
# Recorder server env
# -----------------------------
export DATA_DIR="${DATA_DIR}"
export STATIC_DIR="${ROOTDIR}/static"
export PERSONAL_DIR="${DATA_DIR}/personal_samples"

# IMPORTANT: Training uses system Python (from NGC container)
# The NGC container has TensorFlow pre-installed with proper CUDA/PTX support
export TRAIN_CMD="train_wake_word --data-dir='${DATA_DIR}'"

echo "Launching uvicorn on ${HOST}:${PORT}"
cd "${ROOTDIR}"
exec python3 -m uvicorn recorder_server:app --host "${HOST}" --port "${PORT}"