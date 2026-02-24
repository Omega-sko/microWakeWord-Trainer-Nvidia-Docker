<div align="center">
  <h1>🎙️ microWakeWord Nvidia Trainer & Recorder</h1>
  <img width="1002" height="593" alt="Screenshot 2026-01-18 at 8 13 35 AM" src="https://github.com/user-attachments/assets/e1411d8a-8638-4df8-992b-09a46c6e5ddc" />
</div>

Train **microWakeWord** detection models using a simple **web-based recorder + trainer UI**, packaged in a Docker container.

No Jupyter notebooks required. No manual cell execution. Just record your voice (optional) and train.

---

<img width="100" height="44" alt="unraid_logo_black-339076895" src="https://github.com/user-attachments/assets/87351bed-3321-4a43-924f-fecf2e4e700f" />

**microWakeWord_Trainer-Nvidia** is available in the **Unraid Community Apps** store.
Install directly from the Unraid App Store with a one-click template.

---

<img width="100" height="56" alt="unraid_logo_black-339076895" src="https://github.com/user-attachments/assets/bf959585-ae13-4b4d-ae62-4202a850d35a" />


### Pull the Docker Image

```bash
docker pull ghcr.io/tatertotterson/microwakeword:latest
```

---

### Run the Container

```bash
docker run -d \
  --gpus all \
  -p 8888:8888 \
  -v $(pwd):/data \
  ghcr.io/tatertotterson/microwakeword:latest
```

**What these flags do:**
- `--gpus all` → Enables GPU acceleration  
- `-p 8888:8888` → Exposes the Recorder + Trainer WebUI  
- `-v $(pwd):/data` → Persists all models, datasets, and cache  

---

### Open the Recorder WebUI

Open your browser and go to:

👉 **http://localhost:8888**

You’ll see the **microWakeWord Recorder & Trainer UI**.

---

## 🎤 Recording Voice Samples (Optional)

Personal voice recordings are **optional**.

- You may **record your own voice** for better accuracy  
- Or simply **click “Train” without recording anything**

If no recordings are present, training will proceed using **synthetic TTS samples only**.

### Remote systems (important)
If you are running this on a **remote PC / server**, browser-based recording will not work unless:
- You use a **reverse proxy** (HTTPS + mic permissions), **or**
- You access the UI via **localhost** on the same machine

Training itself works fine remotely — only recording requires local microphone access.

---

### 🎙️ Recording Flow

1. Enter your wake word
2. Test pronunciation with **Test TTS**
3. Choose:
   - Number of speakers (e.g. family members)
   - Takes per speaker (default: 10)
4. Click **Begin recording**
5. Speak naturally — recording:
   - Starts when you talk
   - Stops automatically after silence
6. Repeat for each speaker

Files are saved automatically to:

```
personal_samples/
  speaker01_take01.wav
  speaker01_take02.wav
  speaker02_take01.wav
  ...
```

---

## 🧠 Training Behavior (Important Notes)

### ⏬ First training run
The **first time you click Train**, the system will download **large training datasets** (background noise, speech corpora, etc.).

- This can take **several minutes**
- This happens **only once**
- Data is cached inside `/data`

You **will NOT need to download these again** unless you delete `/data`.

---

### 🔁 Re-training is safe and incremental

- You can train **multiple wake words** back-to-back
- You do **NOT** need to clear any folders between runs
- Old models are preserved in timestamped output directories
- All required cleanup and reuse logic is handled automatically

---

## 📦 Output Files

When training completes, you’ll get:
- `<wake_word>.tflite` – quantized streaming model  
- `<wake_word>.json` – ESPHome-compatible metadata  

Both are saved under:

```text
/data/output/
```

Each run is placed in its own timestamped folder.

---

## 🎤 Optional: Personal Voice Samples (Advanced)

If you record personal samples, the trainer automatically detects and uses them:

- Place personal WAV recordings in **`/data/personal_samples/`** (If you recorded the samples yourself, copy the .wav files into this folder. Samples recorded with the recorder will be automatically saved in this folder.)
- Before training starts, the system checks for WAV files in `/data/personal_samples/`
- If WAV files are present, **personal features** are automatically generated and saved to `/data/work/personal_augmented_features/`
- These personal features are then **up-weighted during training**, significantly improving real-world accuracy for your voice

Personal feature generation is triggered automatically when:
- WAV files are found in `/data/personal_samples/` and features have not yet been generated, **or**
- Existing WAV files are newer than the previously generated features

No configuration required — the entire process is automatic.

---

## 🔄 Resetting Everything (Optional)

If you want a **completely clean slate**:

Delete the /data folder

Then restart the container.

⚠️ This will:
- Remove cached datasets
- Require re-downloading training data
- Delete trained models

---

## 🙌 Credits

Built on top of the excellent  
**https://github.com/kahrendt/microWakeWord**

Huge thanks to the original authors ❤️

---

## 🔧 Local Patches & Customizations

This section documents all local modifications applied automatically to upstream dependencies during setup.
These patches live in the [`patches/`](patches/) directory and are applied by `cli/setup_python_venv`
right after cloning — so they are re-applied automatically whenever `/data/tools` is recreated.

---

### `Dockerfile.source-build` — TensorFlow Built from Source for sm_120 (Blackwell)

For the guaranteed, native `sm_120` (Compute Capability 12.0 — Blackwell / RTX 5000-series)
experience, use the **source-build image** (`ghcr.io/omega-sko/microwakeword:dev-beta`).
This image compiles TensorFlow from source during the Docker build with CUDA 12.8 + cuDNN 9
and embeds `sm_80`, `sm_89`, `sm_90`, and `sm_120` SASS kernels directly — no PTX JIT fall-back.

#### Build the image yourself

```bash
# Clone the repo and check out the dev-beta branch (or use the published image below)
git clone https://github.com/Omega-sko/microWakeWord-Trainer-Nvidia-Docker
cd microWakeWord-Trainer-Nvidia-Docker

docker build -f Dockerfile.source-build \
  --build-arg TF_GIT_TAG=master \
  --build-arg CUDA_COMPUTE_CAPS=8.0,8.9,9.0,12.0 \
  -t ghcr.io/omega-sko/microwakeword:dev-beta \
  .
```

**Build-host requirements:** 16+ GB RAM · 80+ GB free disk · 8+ CPU cores  
**Build time:** ~2–6 hours depending on hardware (Bazel artifact cache is reused on re-builds).

#### Build arguments

| Argument | Default | Description |
|---|---|---|
| `TF_GIT_TAG` | `master` | TensorFlow git tag or branch to build (e.g. `v2.19.0`, `v2.20.0`) |
| `CUDA_COMPUTE_CAPS` | `8.0,8.9,9.0,12.0` | CUDA Compute Capabilities to embed in the wheel |

#### Pull the pre-built image (GHCR)

```bash
docker pull ghcr.io/omega-sko/microwakeword:dev-beta
```

Published automatically by the `publish-ghcr-source-build.yml` workflow on every push to
the `dev-beta` branch (or via manual workflow dispatch).

#### Run the source-build container

```bash
docker run -d \
  --gpus all \
  -p 8789:8789 \
  -v $(pwd):/data \
  ghcr.io/omega-sko/microwakeword:dev-beta
```

#### Verify sm_120 support inside the container

```python
python3 -c "
import tensorflow as tf, pprint
pprint.pprint(tf.sysconfig.get_build_info())
# cuda_compute_capabilities must include sm_120.
# Depending on the TF version the value appears as 'sm_120', '12.0', or plain '120'.
"
```

#### How sm_120 works through the whole stack

1. **Docker build (`Dockerfile.source-build`)** — TF is compiled from source.
   `configure.py --defaults` reads `TF_CUDA_COMPUTE_CAPABILITIES=8.0,8.9,9.0,12.0` and
   bakes the list into `.tf_configure.bazelrc`.  Bazel then compiles SASS kernels for all
   listed architectures including `sm_120`.

2. **System Python** — The finished wheel is installed into system Python and also persisted
   at `/opt/tensorflow-wheel/tensorflow-*.whl`.

3. **User venv (`cli/setup_python_venv`)** — When this script detects
   `/opt/tensorflow-wheel/tensorflow-*.whl`, it installs that wheel into `/data/.venv`
   instead of downloading `tf-nightly` from PyPI.  The venv therefore gets the same
   sm_120-capable build.

4. **Training** — `train_wake_word` → `wake_word_sample_trainer` → TF uses native sm_120
   SASS kernels; no PTX JIT, no `LLVM ERROR: PTX version X.Y does not support sm_120`.

---

### CUDA 12.8 Runtime Base Image

**Why `nvidia/cuda:12.8.0-cudnn-runtime-ubuntu24.04` instead of conda CUDA:**

The `dockerfile` now uses a public NVIDIA CUDA 12.8 + cuDNN runtime image as its base instead of
installing Miniconda and pulling `cuda-12.5.0` via conda. This removes the conda/CUDA mixed-install
fragility and ensures the container ships a self-consistent CUDA 12.8 stack without requiring an
NGC (NVIDIA GPU Cloud) account — the `nvidia/cuda` images on Docker Hub are freely accessible.

Benefits:
- No more `LD_LIBRARY_PATH=/opt/conda/lib` override fighting system libraries
- CUDA 12.8 ships PTX 8.7+ which covers `sm_120` (Blackwell / RTX 5000-series) via JIT
- cuDNN is bundled at the correct version for CUDA 12.8

---

### `TF_VARIANT=blackwell` — Clean TensorFlow Reinstall for Blackwell GPUs

**Problem:**  
The current nightly TensorFlow wheel may be compiled with `cuda_version: 12.5` and only list
`sm_60`, `sm_70`, `sm_80`, `sm_89`, `compute_90` as embedded compute capabilities.  On
`sm_120` (RTX 5070 Ti Laptop GPU and other Blackwell cards) TensorFlow falls back to PTX JIT
compilation, which can crash with:

```
LLVM ERROR: PTX version 8.5 does not support target 'sm_120'
```

**Fix — `TF_VARIANT=blackwell`:**  
Set the environment variable before running `setup_python_venv` (GPU must be available):

```bash
TF_VARIANT=blackwell cli/setup_python_venv --data-dir /data --gpu
```

When `TF_VARIANT=blackwell` is set and `--gpu` is active, the script will:

1. Uninstall any existing `tf-nightly`, `tensorflow`, `tensorflow-gpu`, `ai_edge_litert`,
   `tensorboard`, and `tensorboard-data-server` packages.
2. Reinstall them fresh with `--no-cache-dir --upgrade --pre` to pull the latest nightly wheel
   (which is most likely to include CUDA 12.8 + sm_120 support).
3. Print TensorFlow build info for verification:

```python
import tensorflow as tf
print("tf.__version__:", tf.__version__)
import pprint; pprint.pprint(tf.sysconfig.get_build_info())
print("Physical GPUs:", tf.config.list_physical_devices('GPU'))
```

**Diagnostic commands** (run inside the container or activated venv):

```bash
# Check TF version and build info
python -c "import tensorflow as tf; import pprint; print(tf.__version__); pprint.pprint(tf.sysconfig.get_build_info())"

# List detected GPUs
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"

# Check visible CUDA devices
nvidia-smi
```

**Note on sm_120 and PTX JIT:**  
Even after a clean nightly reinstall, TensorFlow may still rely on PTX JIT for `sm_120` if the
wheel does not yet embed `sm_120` SASS kernels. If training crashes with an `sm_120`-related LLVM
or PTX error, the next step is to use the official **NGC TensorFlow container**
(`nvcr.io/nvidia/tensorflow:xx.xx-tf2-py3`) which is built and tested against the latest CUDA
and includes Blackwell-native kernels. Note that NGC images require a (free) NGC account.

---

### `micro-wake-word` — `microwakeword/train.py`: robust NumPy conversion (`_to_numpy`)

**Patch file:** [`patches/micro-wake-word-train-to_numpy.patch`](patches/micro-wake-word-train-to_numpy.patch)

**Problem:**  
`model.evaluate(..., return_dict=True)` can return either a TensorFlow `EagerTensor` or a plain NumPy
array, depending on the `tf-nightly` build in use. Calling `.numpy()` directly on a plain NumPy array
raises an `AttributeError` and breaks training.

**Fix:**  
A small helper function `_to_numpy(x)` is injected into `train.py`:

```python
def _to_numpy(x):
    """Convert TF tensors/variables OR numpy-like results to a NumPy array."""
    return x.numpy() if hasattr(x, "numpy") else np.asarray(x)
```

All four bare `.numpy()` calls on metric results (`fp`, `tp`, `fn`) inside
`validate_nonstreaming()` are replaced with `_to_numpy(...)`.

**Affected lines in upstream `train.py`:**
- `test_set_fp = result["fp"].numpy()` → `_to_numpy(result["fp"])`
- `all_true_positives = ambient_predictions["tp"].numpy()` → `_to_numpy(ambient_predictions["tp"])`
- `ambient_false_positives = ambient_predictions["fp"].numpy() - test_set_fp` → `_to_numpy(ambient_predictions["fp"]) - test_set_fp`
- `all_false_negatives = ambient_predictions["fn"].numpy()` → `_to_numpy(ambient_predictions["fn"])`

---

### Personal voice samples → automatic feature generation (Recorder → Training)

**Problem:**  
The Recorder stores personal takes as WAV files in:

- `/data/personal_samples/*.wav`

However, the training pipeline does **not** consume these WAVs directly.  
`wake_word_sample_trainer` only enables the “personal up-weighting” path when it finds **precomputed personal features** at:

- `/data/work/personal_augmented_features/training`

So even if personal WAVs exist, training will ignore them unless personal features are generated first.

**Customization / Fix:**  
Before starting training, `train_wake_word` now checks for personal WAV files. If any are present, it runs:

```bash
wake_word_sample_augmenter \
  --data-dir /data \
  --personal-dir /data/personal_samples \
  --personal-output-dir /data/work/personal_augmented_features
```

This generates/updates the personal feature datasets under:

- `/data/work/personal_augmented_features/{training,validation,test}`

Once those directories exist, `wake_word_sample_trainer` automatically detects them and injects the personal feature block into `training_parameters.yaml` (with `sampling_weight: 3.0`), so personal samples are **up-weighted** during training.

**Notes:**
- If `/data/personal_samples` is empty, this step is skipped and training proceeds normally.
- If personal feature generation fails, training fails fast (to avoid silently running without personal data).
- To fully reset this behavior, delete both:
  - `/data/personal_samples` (personal WAVs)
  - `/data/work/personal_augmented_features` (generated personal features)
---
