<div align="center">
  <h1>🎙️ microWakeWord Nvidia Trainer & Recorder</h1>
  <img width="1002" height="593" alt="Screenshot 2026-01-18 at 8 13 35 AM" src="https://github.com/user-attachments/assets/e1411d8a-8638-4df8-992b-09a46c6e5ddc" />
</div>

Train **microWakeWord** detection models using a simple **web-based recorder + trainer UI**, packaged in a Docker container.

No Jupyter notebooks required. No manual cell execution. Just record your voice (optional) and train.

**Built on Official TensorFlow 2.18.0 GPU Image** – Full GPU support for modern NVIDIA GPUs with CUDA and cuDNN libraries.

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

If you record personal samples:
- They are automatically augmented
- They are **up-weighted during training**
- This significantly improves real-world accuracy

No configuration required — detection is automatic.

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

## 🖥️ Technical Details

### Base Image

This container is built on the **Official TensorFlow 2.18.0 GPU Image** (`tensorflow/tensorflow:2.18.0-gpu`), which includes:
- **TensorFlow 2.18.0** with GPU acceleration
- **CUDA and cuDNN** libraries pre-configured for GPU training
- **Python 3.11** with pip
- Optimized for modern NVIDIA GPUs

### GPU Support

Training automatically uses GPU if available. The container supports:
- **Modern NVIDIA GPUs**: Compute Capability 6.0+ (Pascal, Volta, Turing, Ampere, Ada Lovelace, Hopper)
  - TensorFlow 2.18.0 officially supports CUDA Compute Capability 6.0 and higher
  - **RTX 50xx series** (5060, 5070, 5080, 5090) fully supported with automatic XLA PTX workarounds
- **Automatic CPU Fallback**: If GPU training fails (OOM, driver issues), the system automatically retries on CPU

**GPU Training is the Default:**
- First attempt: GPU training with optimized settings
- Second attempt: CPU fallback only if GPU fails
- Your RTX 5070 Ti will be used by default for all training

### Force CPU Training

To force CPU-only training (disable GPU), run the container with:

```bash
docker run -d \
  -e CUDA_VISIBLE_DEVICES="" \
  -p 8888:8888 \
  -v $(pwd):/data \
  ghcr.io/tatertotterson/microwakeword:latest
```

The `CUDA_VISIBLE_DEVICES=""` environment variable disables GPU access.

---

## ✅ Compatibility Fixes

### CPU Training - TensorFlow/Keras Compatibility

**Issue**: Training on CPU crashes with `AttributeError: 'numpy.ndarray' object has no attribute 'numpy'`

**Status**: ✅ **FIXED** - This repository includes automatic patches for microwakeword training code.

**What we fixed:**
- Modern TensorFlow/Keras (v2.18.0+) returns numpy arrays directly from `model.evaluate(..., return_dict=True)` on CPU
- The upstream microwakeword code calls `.numpy()` on these already-materialized arrays, causing AttributeError
- We apply build-time patches to handle both Tensor objects (with `.numpy()` method) and plain numpy arrays

**Technical Details:**
- Patched lines in `/root/mww-tools/microwakeword/microwakeword/train.py`: 73, 104, 105, 106
- Uses `hasattr()` check: `fp = result["fp"]; fp_val = fp.numpy() if hasattr(fp, "numpy") else fp`
- Works with both GPU (Tensor) and CPU (numpy array) execution modes
- A sanity check script is available: `test_training_metrics` (installed in container)

**Validation:**
Run the sanity check inside the container:
```bash
test_training_metrics
```

---

## ⚠️ Known Issues

### RTX 50xx Series (Blackwell Architecture) - TensorFlow CUDA Compatibility

**Issue**: RTX 50xx series GPUs (Compute Capability 12.0) may show warnings about missing CUDA kernel binaries:

```
W0000 00:00:1771428664.557890 159 gpu_device.cc:2431] TensorFlow was not built 
with CUDA kernel binaries compatible with compute capability 12.0. CUDA kernels 
will be jit-compiled from PTX, which could take 30 minutes or longer.
```

**Status**: ✅ **FIXED** - This repository includes automatic workarounds for RTX 50xx GPUs.

**GPU Training is Enabled:**
RTX 50xx GPUs (including your RTX 5070 Ti) will train on GPU by default. The fixes enable PTX JIT compilation, so your GPU remains the primary training device.

**What we do automatically:**
1. **XLA PTX Fallback**: Set `XLA_FLAGS=--xla_gpu_unsafe_fallback_to_driver_on_ptxas_not_found` to enable driver-side PTX compilation
2. **Disable Auto JIT**: Set `TF_XLA_FLAGS=--tf_xla_auto_jit=0` to avoid XLA JIT compilation issues
3. **CPU Fallback**: If GPU training still fails, automatic detection triggers CPU-only training
4. **Extended Error Detection**: GPU failure markers now detect compute capability errors, PTX issues, and traceback failures

**For advanced users:**
- Force CPU training: Set `CUDA_VISIBLE_DEVICES=""` environment variable
- Override XLA flags: Set `XLA_FLAGS` before running the container (will skip auto-detection)

**Technical Details:**
- TensorFlow 2.18.0 does not ship with pre-compiled CUDA kernels for sm_120 (Blackwell architecture)
- PTX JIT compilation can fail or be very slow without proper XLA flags
- Our workarounds enable runtime PTX compilation and provide graceful CPU fallback
- This affects: RTX 5060, 5070, 5080, 5090 (laptop and desktop variants)

**References:**
- [TensorFlow Issue #89272: RTX 5090 Support](https://github.com/tensorflow/tensorflow/issues/89272)
- [TensorFlow Issue #101746: PTX Error Workarounds](https://github.com/tensorflow/tensorflow/issues/101746)
- [NVIDIA Blackwell Compatibility Guide](https://docs.nvidia.com/cuda/blackwell-compatibility-guide/)

---

## 🙌 Credits

Built on top of the excellent  
**https://github.com/kahrendt/microWakeWord**

Huge thanks to the original authors ❤️
