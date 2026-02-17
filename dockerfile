# Base: NVIDIA TensorFlow NGC container with CUDA 12.6 and TensorFlow 2.18
# This image includes TensorFlow built with PTX 8.7+ which supports sm_120 (Compute Capability 12.0)
FROM nvcr.io/nvidia/tensorflow:25.02-tf2-py3

ENV DEBIAN_FRONTEND=noninteractive

# System deps (NGC container already has python3, git, etc. but we add any missing utilities)
# Install python3-venv to ensure venv/ensurepip support on Ubuntu 24.04
RUN apt-get update && apt-get install -y --no-install-recommends \
    nano less python3-venv \
 && rm -rf /var/lib/apt/lists/* \
 && mkdir -p /data

# Build-time verification: ensure venv creation and ensurepip work
RUN python3 -c "import ensurepip" && \
    python3 -m venv /tmp/test-venv && \
    rm -rf /tmp/test-venv && \
    echo "[OK] venv and ensurepip verified"

# Recorder port
EXPOSE 8789

# Script root
WORKDIR /root/mww-scripts

# Bash environment
COPY --chown=root:root --chmod=0755 .bashrc /root/

# Root-level entrypoints
COPY --chown=root:root --chmod=0755 \
    train_wake_word \
    run_recorder.sh \
    recorder_server.py \
    requirements.txt \
    /root/mww-scripts/

# CLI folder
COPY --chown=root:root cli/ /root/mww-scripts/cli/

# Make all CLI scripts executable (avoids "Permission denied")
RUN chmod -R a+x /root/mww-scripts/cli

# Install Python modules
RUN pip install --no-cache-dir -r /root/mww-scripts/requirements.txt

# Static UI for recorder
COPY --chown=root:root --chmod=0644 static/index.html /root/mww-scripts/static/index.html

# Install training dependencies in the system Python
# NGC container already has TensorFlow, but we need additional packages
RUN pip install --no-cache-dir \
    numpy==1.26.4 \
    scipy==1.12.0 \
    librosa==0.10.2.post1 \
    soundfile==0.12.1 \
    tqdm==4.67.1 \
    scikit-learn==1.6.0 \
    numba==0.63.1 \
    PyYAML==6.0.3 \
    ai_edge_litert \
    tensorboard \
    tensorboard-data-server \
    "keras==3.12.0" && \
    pip install --no-cache-dir \
    "torch==2.9.1" \
    "torchaudio==2.9.1" --index-url https://download.pytorch.org/whl/cu126 && \
    pip install --no-cache-dir "onnxruntime-gpu>=1.16.0"

# Clone and install microwakeword and piper-sample-generator
RUN mkdir -p /root/mww-tools && \
    cd /root/mww-tools && \
    git clone https://github.com/Omega-sko/micro-wake-word microwakeword && \
    cd microwakeword && \
    # Patch setup.py to not reinstall TensorFlow (NGC container already has it)
    sed -i 's/"tensorflow>=2.16"/"tensorflow"/g' setup.py && \
    pip install --no-cache-dir -e . && \
    cd /root/mww-tools && \
    git clone https://github.com/rhasspy/piper-sample-generator && \
    cd piper-sample-generator && \
    pip install --no-cache-dir -e . && \
    mkdir -p models && \
    cd models && \
    curl -sfL https://github.com/rhasspy/piper-sample-generator/releases/download/v2.0.0/en_US-libritts_r-medium.pt -o en_US-libritts_r-medium.pt && \
    # JSON file is not available in releases, fetch from repo pinned to commit ded9350 (corresponds to v2.0.0)
    curl -sfL https://raw.githubusercontent.com/rhasspy/piper-sample-generator/ded9350eaff558af07f312464ac71baf7de834df/models/en_US-libritts_r-medium.pt.json -o en_US-libritts_r-medium.pt.json

# Set environment variables for training
ENV PIPER_SAMPLE_GENERATOR_DIR=/root/mww-tools/piper-sample-generator
ENV MICROWAKEWORD_DIR=/root/mww-tools/microwakeword

# recorder server
CMD ["/bin/bash", "-lc", "/root/mww-scripts/run_recorder.sh"]
