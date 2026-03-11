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

## ⚡ Blackwell / RTX 50-series GPU Support

NVIDIA Blackwell GPUs (RTX 5070, 5080, 5090, etc.) with **compute capability 12.x** are supported with automatic compatibility handling.

### What happens automatically

- **TensorFlow package**: When a Blackwell GPU is detected during environment setup, `tf-nightly[and-cuda]` is installed instead of the stable `tensorflow==2.20.0`. This is because TF 2.20.0 does not ship pre-compiled CUDA kernels for compute capability 12.x, causing 30+ minute JIT compile times and potential runtime failures.
- **Allocator**: The `cuda_malloc_async` allocator is disabled on Blackwell (uses BFC allocator instead) to avoid known instability with TF 2.20 + Blackwell.
- **XLA**: Auto-JIT XLA compilation is disabled via `TF_XLA_FLAGS=--tf_xla_auto_jit=0`, which is safer for this architecture.
- **PTX JIT fallback**: `--xla_gpu_unsafe_fallback_to_driver_on_ptxas_not_found` is set automatically, allowing the driver to JIT-compile PTX when `ptxas` is unavailable.
- **Retry logic**: Training will automatically retry with progressively more conservative GPU profiles before falling back to CPU.

### Recommended Docker flags

Always pass `--gpus all` when running the container:

```bash
docker run -d \
  --gpus all \
  -p 8888:8888 \
  -v $(pwd):/data \
  ghcr.io/tatertotterson/microwakeword:latest
```

### Environment variable overrides

| Variable | Description | Default |
|---|---|---|
| `MWW_TF_SPEC` | Override TensorFlow package spec, e.g. `tf-nightly[and-cuda]` | Auto-detected |
| `MWW_KERAS_SPEC` | Override Keras package spec, e.g. `keras==3.12.0` | Auto-detected |
| `MWW_ALLOW_CPU_FALLBACK` | Allow CPU fallback if GPU training fails (`true`/`false`) | `true` |
| `MWW_TENSORBOARD_SPEC` | Comma-separated TensorBoard specs, e.g. `tensorboard==2.20.0,tensorboard-data-server==0.7.2` | Auto-detected |

### First-run note

On Blackwell, the first training run **will be slower** than subsequent runs because TensorFlow must JIT-compile GPU kernels via PTX. This is a one-time cost per session. You may see a message like:

> `TensorFlow was not built with CUDA kernel binaries compatible with compute capability 12.0. CUDA kernels will be jit-compiled from PTX, which could take 30 minutes or longer.`

This is expected. Subsequent runs in the same session reuse the compiled kernels.

### Diagnosing GPU detection

The setup and training logs now explicitly report:
- Whether `/dev/nvidiactl` and `/dev/nvidia0` are present
- Detected compute capability from `nvidia-smi`
- Which TensorFlow spec was chosen and why
- Whether `MWW_TF_SPEC` override is active
- Which allocator and XLA flags are in use
- Whether GPU or CPU training is being attempted

---



Built on top of the excellent  
**https://github.com/kahrendt/microWakeWord**

Huge thanks to the original authors ❤️
