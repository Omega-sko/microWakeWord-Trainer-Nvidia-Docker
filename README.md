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
- **Automatic CPU Fallback**: If GPU training fails (OOM, driver issues), the system automatically retries on CPU

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

## ⚠️ Known Issues

### RTX 50xx Series (Blackwell Architecture) - TensorFlow CUDA Error

**Issue**: Training may fail on NVIDIA RTX 50xx series GPUs (Compute Capability 12.0) with the following error:

```
tensorflow.python.framework.errors_impl.InternalError: ...
'cuLaunchKernel(function ...) failed with 'CUDA_ERROR_INVALID_HANDLE'
[Op:Cast] name: ...
```

**Cause**: 
- This error occurs due to missing or defective TensorFlow/JIT kernels for Compute Capability 12.0 (Ada/Blackwell RTX 50xx series GPUs)
- Upstream TensorFlow does not yet provide maintained CUDA kernels for `sm_120` architecture
- PTX JIT compilation fails, resulting in `InternalError` with `CUDA_ERROR_INVALID_HANDLE`
- This issue persists even with the latest TensorFlow nightly builds

**Workarounds**:
1. **Use CPU Fallback**: The training scripts automatically detect GPU failures and fall back to CPU training
2. **Force CPU Training**: Set `CUDA_VISIBLE_DEVICES=""` environment variable to skip GPU entirely
3. **Wait for Upstream Fix**: Monitor TensorFlow release notes and GitHub issues for Blackwell architecture support

**References**:
- [TensorFlow GPU Support Documentation](https://www.tensorflow.org/install/gpu)
- Known issue tracking: Search TensorFlow GitHub for "CUDA_ERROR_INVALID_HANDLE" and "Compute Capability 12.0"

---

## 🙌 Credits

Built on top of the excellent  
**https://github.com/kahrendt/microWakeWord**

Huge thanks to the original authors ❤️
