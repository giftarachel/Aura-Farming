# 🌿 Aura Farming

**An AI-Powered, Reliability-Aware Crop Disease Detection System**

> *Smarter Farming. Healthier Crops.*

Aura Farming is a college-level Computer Science and Engineering research project that analyses photographs of crop leaves, identifies potential diseases using a deep learning model, assesses prediction reliability, and provides understandable visual explanations.

---

## Current Status: Phase 4 — Model Training

The dataset is prepared (18,160 tomato images across 10 classes, split 70/15/15 with duplicate-group leakage prevention). The MobileNetV3 Small training pipeline is implemented and ready. Training runs on Google Colab with a free T4 GPU.

---

## Quick Start (Streamlit UI)

### 1. Prerequisites

- Python 3.10 or 3.11 recommended (3.12 also works)
- `pip` and `venv` (included with Python 3.3+)
- macOS, Linux, or Windows

> **Note:** The default `.venv` uses Python 3.14, which is fine for the
> Streamlit UI and tests. TensorFlow training requires Python 3.10–3.12.
> See "Model Training" below for details.

### 2. Clone the repository

```bash
git clone <your-repo-url>
cd "Aura Farming"
```

### 3. Create a virtual environment

```bash
python -m venv .venv
```

Activate it:

```bash
# macOS / Linux
source .venv/bin/activate

# Windows
.venv\Scripts\activate
```

### 4. Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 5. Run the application

```bash
streamlit run app.py
```

The application opens at **http://localhost:8501** in your browser.

---

## Project Structure

```
aura-farming/
├── app.py                          # Homepage (Streamlit entry point)
├── pages/
│   ├── 1_Detect_Disease.py         # Upload and detection page
│   └── 2_About_Aura_Farming.py     # About and workflow explanation
├── src/                            # Core application modules
│   ├── ui_components.py            # Branding, CSS, reusable UI
│   ├── image_validation.py         # Format, size, decode checks
│   ├── image_quality.py            # Blur/brightness gate (Phase 5)
│   ├── preprocessing.py            # Tensor preparation for inference
│   ├── predictor.py                # Model loading and inference
│   ├── calibration.py              # Calibration & abstention (Phase 5)
│   ├── gradcam.py                  # Grad-CAM visualization (Phase 5)
│   ├── disease_information.py      # Disease descriptions (Phase 5)
│   └── guidance.py                 # Prevention guidance (Phase 5)
├── training/                       # Offline training scripts
│   ├── inspect_dataset.py          # Dataset inspection (Phase 3)
│   ├── prepare_dataset.py          # Splitting with leakage prevention
│   ├── duplicate_detection.py     # MD5 + pHash duplicate detection
│   ├── train_baseline.py           # MobileNetV3 training (Phase 4)
│   ├── colab_utils.py              # Path remapping for Colab (Phase 4)
│   ├── Aura_Farming_Colab_Training.ipynb  # Colab notebook
│   ├── calibrate_model.py          # (Phase 5)
│   └── evaluate_model.py           # (Phase 7)
├── configs/
│   ├── config.yaml                 # Central configuration
│   └── class_mapping.json          # Label index map
├── models/                         # Trained model files (not in git)
│   └── README.md
├── reports/                        # Experiment outputs (not in git)
│   └── README.md
├── tests/                          # Unit tests
├── assets/                         # Static UI assets
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Running Tests

```bash
pytest tests/ -v
```

To include coverage:

```bash
pytest tests/ -v --cov=src --cov-report=term-missing
```

---

## Dataset Setup (Phase 3)

**Do not commit datasets to Git.** Datasets are stored locally in `data/` (excluded by `.gitignore`).

1. Download PlantVillage from: https://github.com/spMohanty/PlantVillage-Dataset
2. Place the color variant at: `data/plantvillage/color/`
3. Run inspection: `python training/inspect_dataset.py`
4. Detect duplicates: `python training/duplicate_detection.py`
5. Create splits: `python training/prepare_dataset.py`

See `reports/DATASET_PREPARATION.md` for full details.

---

## Model Training (Phase 4)

### ⚠ Python version requirement

**TensorFlow officially supports Python 3.9–3.12 only.** This project's
default `.venv` uses Python 3.14, which is fine for the Streamlit UI and
tests but **cannot run TensorFlow training**.

### Option A — Google Colab (recommended, free T4 GPU)

1. Upload the entire `Aura Farming/` project folder to Google Drive.
2. Upload the PlantVillage `color/` folder to `data/plantvillage/color/`
   inside your project folder on Drive.
3. Open `training/Aura_Farming_Colab_Training.ipynb` in Colab.
4. Set the runtime type to **GPU** (Runtime → Change runtime type → T4 GPU).
5. Run each section in order:
   - Section 1: Mount Drive and set paths
   - Section 2: Install dependencies (skips TF — already in Colab)
   - Section 3: Environment check (verifies GPU)
   - Section 4: Remap manifest paths (Mac paths → Colab paths)
   - Section 5: Smoke test (one forward pass, one training step)
   - Section 6: Full Phase A training (~15–25 min on T4)
   - Section 7: Phase B fine-tuning (~15–20 min, optional but recommended)
   - Section 8: Update config.yaml and download model files
6. Download the generated `.keras` and report files back to your Mac.

The Colab notebook includes a **smoke test** that loads 8 real training
images, builds MobileNetV3 Small, and runs one forward + backward pass
before committing to full training. If the smoke test fails, fix the
issue before running the full pipeline.

### Option B — Local Python 3.12 venv

```bash
python3.12 -m venv .venv312
source .venv312/bin/activate
pip install -r requirements.txt

# Apple Silicon (M1/M2/M3):
pip install tensorflow-macos tensorflow-metal

# Standard (Linux/Windows):
pip install tensorflow

python training/train_baseline.py --fine-tune
```

### Training commands

```bash
# Phase A only (feature extraction with frozen base):
python training/train_baseline.py

# Phase A + Phase B (fine-tuning):
python training/train_baseline.py --fine-tune

# Custom batch size and epoch limit:
python training/train_baseline.py --batch-size 16 --epochs 20

# Disable augmentation:
python training/train_baseline.py --no-augment
```

### What training produces

```
models/mobilenetv3_phase_a.keras       — best Phase A weights
models/mobilenetv3_baseline.keras      — best Phase B weights (if --fine-tune)
reports/training_log_phase_a.csv       — per-epoch metrics
reports/training_history_phase_a.json  — full history
reports/val_metrics_phase_a.json       — validation accuracy + F1
reports/confusion_matrix_phase_a.png   — confusion matrix
```

After training, `configs/config.yaml` is automatically updated with the
model path so the Streamlit app can load it.

**The test manifest is never used during training.** It is reserved for
Phase 7 final evaluation only.

---

## Configuration

All thresholds, paths, and model parameters are in `configs/config.yaml`. Edit this file to:
- Update model file paths after training.
- Adjust image quality thresholds (Phase 5).
- Set the confidence threshold (Phase 5).

---

## Disclaimer

**Aura Farming is a research prototype.** AI predictions are not a substitute for diagnosis by a qualified agricultural expert or plant pathologist. Always consult a professional before making treatment decisions.

---

## Development Phases

| Phase | Description | Status |
|---|---|---|
| Phase 1 | Specification and planning | ✅ Complete |
| Phase 2 | Project foundation and initial UI | ✅ Complete |
| Phase 3 | Dataset inspection and preparation | ✅ Complete |
| Phase 4 | Baseline MobileNetV3 training | ✅ Code ready — training pending |
| Phase 5 | Quality gate, calibration, Grad-CAM | Upcoming |
| Phase 6 | Full application integration | Upcoming |
| Phase 7 | Evaluation and experiments | Upcoming |
| Phase 8 | Final polish and documentation | Upcoming |
