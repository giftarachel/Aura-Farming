# 🌿 Aura Farming

**An AI-Powered, Reliability-Aware Crop Disease Detection System**

> *Smarter Farming. Healthier Crops.*

Aura Farming is a college-level Computer Science and Engineering research project that analyses photographs of crop leaves, identifies potential diseases using a deep learning model, assesses prediction reliability, and provides understandable visual explanations.

---

## Current Status: Phase 7 — Scientific Evaluation Complete

Scientific evaluation on the sealed test set (2,724 images across all 10 classes) is complete. MobileNetV3 Small achieved **97.25% test accuracy**, **96.42% macro F1-score**, and **0.9995 macro OvR ROC-AUC** with average inference latency of **3.48 ms/image**. Post-hoc calibration maintains <1% calibration error. All 263 unit and integration tests pass cleanly. Out-of-domain PlantDoc evaluation protocols and class mismatch handling are documented.

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

## Phase 5 — Reliability Components (Quality Gate, Calibration, Grad-CAM)

Phase 5 implements the core trustworthiness and explainability layer of Aura Farming.

### 1. Image Quality Assessment (`src/image_quality.py`)
- **Decode & Dimension Validation:** Verifies image decodability and handles RGB, RGBA, and Grayscale modes. Enforces minimum (64 px) and maximum (8000 px) dimensions.
- **Blur Detection:** Uses the variance of the Laplacian operator (`cv2.Laplacian(gray, cv2.CV_64F).var()`). Blurry photographs produce low edge variance. Configured via `quality.blur_laplacian_threshold: 100.0`.
- **Exposure & Brightness:** Computes mean pixel intensity in HSV Value (V) space (`0–255` scale). Flags underexposure (`< 30.0`) and overexposure/glare (`> 225.0`).
- **Structured Report:** Produces a `QualityReport` distinguishing between hard rejections (`status: FAILED`) and cautionary notices (`status: WARNING`).

### 2. Post-Hoc Probability Calibration (`src/calibration.py`, `training/calibrate_model.py`)
- **Mathematical Approach:** Multiclass Temperature Scaling ($T > 0$). Divides raw logits by scalar $T$ before softmax:
  $$\hat{q}_k(T) = \frac{\exp(z_k / T)}{\sum_j \exp(z_j / T)}$$
- **Optimization:** Scalar $T$ is fitted strictly by minimizing Negative Log-Likelihood (NLL) on the **Validation Split only**. The **Test Set is sealed** until Phase 7.
- **Metrics Tracked:**
  - **Expected Calibration Error (ECE):** Evaluated using $M=10$ equal-width bins across $[0, 1]$.
  - **Negative Log-Likelihood (NLL):** Multi-class cross-entropy loss.
  - **Brier Score:** Mean squared error between predicted probabilities and one-hot ground truth.
- **Artifact Persistence:** Parameters and metrics are stored in `models/mobilenetv3_calibration.json`. When missing, the application falls back safely to uncalibrated inference ($T = 1.0$).
- **Calibration Command:**
  ```bash
  # Run calibration on validation data once model is trained:
  python training/calibrate_model.py

  # Verify mathematical calibration pipeline on synthetic validation fixtures:
  python training/calibrate_model.py --self-test
  ```

### 3. Confidence & Abstention Policy
Predictions are categorized into three reliability tiers:
- **CONFIDENT:** Top calibrated probability $\ge 0.70$, distinct margin ($\ge 0.05$) over alternative classes, and image quality gate passed.
- **UNCERTAIN:** Top calibrated probability between $0.40$ and $0.70$, narrow class margin, or photograph quality warnings. Displayed with a clear advisory note.
- **ABSTAINED:** Top calibrated probability $< 0.40$ or image quality gate failed. Diagnosis is withheld with corrective guidelines.

### 4. Explainable AI: Grad-CAM (`src/gradcam.py`)
- **Architecture Integration:** Targets the final convolutional feature layer of MobileNetV3 Small (`mobilenetv3small`, output shape $7 \times 7 \times 576$).
- **Heatmap Generation:** Computes gradients via `tf.GradientTape`, applies global average pooling across spatial dimensions, computes positive weighted activations via ReLU, normalizes to $[0, 1]$, and blends with the original image using JET colormap ($\alpha = 0.4$).
- **Scientific Caveat:** Visualizations illustrate regions that influenced the model's score; they do not establish causal diagnosis or prove pathogen presence.
- **Failure Resilience:** Grad-CAM errors never abort the disease diagnosis.

### 5. Disease Information & Prevention Guidance (`src/disease_information.py`, `src/guidance.py`)
- Full profiles for all 10 tomato classes covering pathogen taxonomy, observable symptoms, immediate physical/cultural field actions, long-term IPM prevention, and field scouting guidelines.
- Sourced and cited from CABI PlantwisePlus, FAO, UC IPM, Cornell Extension, and Penn State Extension.
- Educational disclaimers accompany all guidance.

---

## Phase 6 — Full Application Integration

Phase 6 connects all core components into an end-to-end user-facing diagnosis and explainability workflow:

```
[Leaf Image Upload]
         ↓
[Image Validation] (Format, size <= 20MB, decodability, min/max dimensions)
         ↓
[Quality Gate] (Laplacian blur edge variance >= 100.0, HSV brightness 30–225)
         ↓
[Model Preprocessing] (Rescaling to [-1.0, 1.0], 224x224x3 float32 tensor)
         ↓
[Trained Model Inference] (MobileNetV3 Small top-layer inference on CPU/GPU)
         ↓
[Post-Hoc Probability Calibration] (Temperature Scaling T=0.8036 fitted on validation split)
         ↓
[Reliability Decision & Abstention] (Confident >= 0.70 / Uncertain 0.40–0.70 / Abstained < 0.40)
         ↓
[Agronomic Guidance & Visual Explanation]
  ├── Structured Biology & Symptoms (CABI/FAO literature)
  ├── Immediate Field Actions & Cultural Measures
  ├── Long-Term Integrated Pest Management (IPM) & Scouting
  └── Grad-CAM Visual Attention Overlay (Heatmap on MobileNetV3 Conv features)
```

### End-to-End Validation
The full application pipeline is verified via automated integration checks:
```bash
# Verify end-to-end model inference, calibration, and Grad-CAM:
python training/validate_end_to_end.py

# Run all 263 unit and integration tests:
pytest tests/ -v
```

---

## Phase 7 — Scientific Model Evaluation

Phase 7 evaluates the fine-tuned MobileNetV3 Small model against the sealed test set (`data/splits/test_manifest.csv`) and out-of-domain field protocols:

### Sealed Test Set Benchmark (2,724 Images)
- **Test Set Accuracy:** **97.25%** (vs 97.54% validation accuracy — negligible generalization gap of 0.29%).
- **Macro-Averaged F1-Score:** **96.42%**
- **Macro-Averaged OvR ROC-AUC:** **0.9995**
- **Inference Latency:** **3.48 ms/image** (287 images/second on CPU).
- **Probabilistic Calibration:** Test ECE is 0.0089 (calibrated) and 0.0045 (uncalibrated), well below the 1.0% reliability ceiling.
- **Selective Classification (Abstention):** **97.28% selective accuracy** on retained predictions. Confident tier accuracy reaches **98.50%** (2,666 images), while uncertain tier drops to **40.35%** (57 images), proving that abstention successfully concentrates model errors.

### Out-of-Domain Generalization (PlantDoc)
- Evaluates domain transfer from studio photography to unconstrained agricultural field photography.
- Class mapping schema connects 9 of 10 tomato disease classes.
- Explicitly documents domain mismatch: **Tomato — Target Spot (Class 6)** is missing in PlantDoc annotations.
- Full protocol documented in [`reports/PLANTDOC_EVALUATION.md`](reports/PLANTDOC_EVALUATION.md).

```bash
# Run scientific evaluation on sealed test set:
python training/evaluate_model.py
```

---

## Configuration

All thresholds, paths, and model parameters are in `configs/config.yaml`:
- `quality.blur_laplacian_threshold`: Laplacian variance threshold (default 100.0).
- `quality.brightness_min` & `brightness_max`: HSV brightness thresholds (30 and 225).
- `prediction.confidence_threshold`: Confident threshold (0.70).
- `prediction.abstention_threshold`: Abstention threshold (0.40).
- `prediction.margin_threshold`: Minimum differential margin (0.05).
- `gradcam.alpha`: Overlay opacity (0.4).

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
| Phase 4 | Baseline MobileNetV3 Small training (97.54% val acc) | ✅ Complete |
| Phase 5 | Reliability components (Quality gate, calibration, Grad-CAM) | ✅ Complete |
| Phase 6 | Full application integration (263/263 tests passing) | ✅ Complete |
| Phase 7 | Scientific evaluation — sealed test set (97.25% acc, 0.9995 AUC), calibration & coverage | ✅ Complete |
| Phase 8 | Final polish, documentation, viva preparation | 🔲 Next |
