# NGC TensorFlow variant — CUDA 12.8 / Blackwell (sm_120) capable
# Requires a free NGC account: https://ngc.nvidia.com
# Login: docker login nvcr.io  (username: $oauthtoken, password: <NGC API key>)
ARG NGC_TF_TAG=25.02-tf2-py3
FROM nvcr.io/nvidia/tensorflow:${NGC_TF_TAG}

ENV DEBIAN_FRONTEND=noninteractive

# System deps needed by scripts and venv creation
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-venv python-is-python3 \
    git curl wget unzip ca-certificates nano less perl libtinfo6 \
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

# Signal to setup_python_venv that the NGC container TF should be used as-is
ENV TF_VARIANT=ngc

# recorder server
CMD ["/bin/bash", "-lc", "/root/mww-scripts/run_recorder.sh"]
