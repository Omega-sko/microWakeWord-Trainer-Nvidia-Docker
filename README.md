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
