# Base - NVIDIA CUDA 12.5.1 + cuDNN 9 Runtime on Ubuntu 22.04
# Matches tf-nightly[and-cuda] build expectations (CUDA 12.5.1 / cuDNN 9)
FROM nvidia/cuda:12.5.1-cudnn9-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# System deps (Python 3.12 via deadsnakes PPA; Ubuntu 22.04 default is 3.10)
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
 && add-apt-repository -y ppa:deadsnakes/ppa \
 && apt-get update && apt-get install -y --no-install-recommends \
    python3.12 python3.12-venv python3.12-dev python3-pip python-is-python3 \
    git wget curl unzip ca-certificates nano less \
 && rm -rf /var/lib/apt/lists/* \
 && mkdir -p /data

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

# Copy Patch folder
COPY --chown=root:root patches/ /root/mww-scripts/patches/

# CLI folder
COPY --chown=root:root cli/ /root/mww-scripts/cli/

# Make all CLI scripts executable (avoids "Permission denied")
RUN chmod -R a+x /root/mww-scripts/cli

# Static UI for recorder
COPY --chown=root:root --chmod=0644 static/index.html /root/mww-scripts/static/index.html

# recorder server
CMD ["/bin/bash", "-lc", "/root/mww-scripts/run_recorder.sh"]
