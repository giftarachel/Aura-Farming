# 🌿 Aura Farming — Project Handoff Document

**An AI-Powered, Reliability-Aware Crop Disease Detection System**

> *Smarter Farming. Healthier Crops.*

---

## Project status at handoff

| Phase | Description | Status |
|---|---|---|
| Phase 1 | Specification and architecture planning | ✅ Complete |
| Phase 2 | Project foundation — Streamlit UI, branding, image validation | ✅ Complete |
| Phase 3 | Dataset preparation — splits, duplicate detection, leakage prevention | ✅ Complete |
| Phase 4 | MobileNetV3 Small baseline training pipeline | ✅ Code complete — Phase A trained |
| Phase 5 | Image quality gate, probability calibration, Grad-CAM | 🔲 Not started |
| Phase 6 | Full application integration | 🔲 Not started |
| Phase 7 | Scientific evaluation — test set, calibration error, coverage | 🔲 Not started |
| Phase 8 | Final polish, documentation, viva preparation | 🔲 Not started |

---

## What has been built

### Streamlit application (Phases 1–2)
- **Homepage** (`app.py`) — branded landing page with Aura Farming identity, feature cards, workflow explanation, disclaimer
- **Detect Disease page** (`pages/1_Detect_Disease.py`) — image uploader, format/size/corruption validation, preview
- **About page** (`pages/2_About_Aura_Farming.py`) — project purpose, AI workflow, phase status, limitations
- Centralized design system in `src/ui_components.py` — all 7 oak-green-and-beige hex values in one place
- Upload validation: JPG/JPEG/PNG/WEBP, 20 MB limit, corrupt-file detection

### Dataset (Phase 3)
- PlantVillage color variant downloaded and placed at `data/plantvillage/color/`
- 18,160 tomato images across 10 disease classes
- Stratified 70/15/15 train/val/test split with duplicate-group leakage prevention
- Near-duplicate detection (pHash, Hamming ≤ 4) — 6 cross-class groups handled globally
- Split manifests written to `data/splits/` — absolute paths, not committed to git
- Split report: `data/splits/split_report.json`

### Training pipeline (Phase 4)
- `training/train_baseline.py` — full MobileNetV3 Small pipeline
  - Phase A: feature extraction (frozen base, head-only training)
  - Phase B: fine-tuning (upper layers unfrozen, BN layers always frozen, reloads Phase A checkpoint)
  - Class weights computed from training split only
  - Manual MobileNetV3 rescaling (`x / 127.5 - 1.0`) — works correctly in Keras 3 / TF 2.17+
  - Explicit `name="mobilenetv3small"` on base model — version-independent `get_layer()` lookup
- `training/Aura_Farming_Colab_Training.ipynb` — beginner-friendly Colab notebook with path remapping, environment check, smoke test, Phase A/B training, and download instructions
- `training/colab_utils.py` — path remapping, GPU check, install helpers
- Phase A checkpoint saved at `models/mobilenetv3_phase_a.keras`

### Source modules
| Module | Status | Purpose |
|---|---|---|
| `src/image_validation.py` | ✅ Complete | Format, size, decode, dimension checks |
| `src/preprocessing.py` | ✅ Complete | PIL → normalized NumPy tensor for inference |
| `src/predictor.py` | ✅ Complete | Model loading, caching, softmax inference |
| `src/image_quality.py` | 🔲 Placeholder | Blur/brightness gate (Phase 5) |
| `src/calibration.py` | 🔲 Placeholder | Temperature scaling, abstention (Phase 5) |
| `src/gradcam.py` | 🔲 Placeholder | Grad-CAM heatmap (Phase 5) |
| `src/disease_information.py` | 🔲 Placeholder | Disease descriptions (Phase 5) |
| `src/guidance.py` | 🔲 Placeholder | Prevention guidance (Phase 5) |

### Tests
- **172 tests pass** in Python 3.14 (non-TF tests only)
- **190 tests pass** in Python 3.12 + TensorFlow 2.17.1 (confirmed by student run)
- 20 tests skip when TF is absent (correct behavior)
- Test files: `tests/test_image_validation.py`, `test_image_quality.py`, `test_predictor.py`, `test_preprocessing.py`, `test_prepare_dataset.py`, `test_dataset_utils.py`, `test_train_baseline.py`, `test_colab_utils.py`

---

## Critical file locations

```
Aura Farming/
├── app.py                          # Streamlit entry point
├── configs/
│   ├── config.yaml                 # Central config (weights_path set to Phase A model)
│   └── class_mapping.json          # 10-class tomato label map with exact folder names
├── data/                           # ⚠ NOT in git — dataset is large
│   ├── plantvillage/color/         # 18,160 tomato images (918 MB)
│   └── splits/                     # train/val/test manifests + split_report.json
├── models/
│   ├── README.md                   # Instructions for model storage
│   └── mobilenetv3_phase_a.keras   # ⚠ NOT in git — 4.4 MB Phase A checkpoint
├── reports/
│   └── DATASET_PREPARATION.md      # Dataset source, license, class table
├── src/                            # Application modules
├── training/
│   ├── train_baseline.py           # Main training script
│   ├── Aura_Farming_Colab_Training.ipynb  # Colab notebook
│   ├── colab_utils.py              # Path remapping for Colab
│   ├── prepare_dataset.py          # Split generation with leakage prevention
│   └── duplicate_detection.py      # MD5 + pHash duplicate detection
└── tests/                          # Unit tests
```

---

## What to do next (Phase 5)

### Immediate next step — Phase B fine-tuning
Phase A training is complete. Run Phase B from Google Colab:
```bash
python training/train_baseline.py --fine-tune
```
Or use Section 7 of `training/Aura_Farming_Colab_Training.ipynb`.

The Phase B model saves to `models/mobilenetv3_baseline.keras`.

### Phase 5 — Reliability components
1. **Image quality gate** (`src/image_quality.py`) — implement blur detection (Laplacian variance via OpenCV) and brightness check; select thresholds on validation data
2. **Probability calibration** (`src/calibration.py`) — implement temperature scaling; fit temperature parameter on validation set only; never use test set
3. **Confidence threshold** — select the threshold τ on the validation set targeting ~85% selective accuracy (research goal, not guaranteed); record actual coverage and selective accuracy
4. **Grad-CAM** (`src/gradcam.py`) — identify last conv layer of MobileNetV3Small; implement class-weighted activation maps; blend heatmap over original image
5. **Disease information** (`src/disease_information.py`) and **guidance** (`src/guidance.py`) — source content from CABI/FAO/extension services; cite every claim

### Phase 6 — Application integration
Connect `src/predictor.py`, `src/calibration.py`, `src/gradcam.py`, and the disease/guidance modules to the Streamlit detect page.

### Phase 7 — Evaluation
Evaluate the **baseline** (no quality gate, no calibration) and **full pipeline** on:
- PlantVillage validation set (already split, untouched)
- PlantVillage **test set** (`data/splits/test_manifest.csv`) — **never opened until Phase 7**
- PlantDoc field images (download separately; check license)

Report accuracy, macro-F1, per-class recall, confusion matrix, ECE, coverage, and selective accuracy honestly.

---

## Environment and setup

### Running the Streamlit app (macOS, any Python ≥ 3.10)
```bash
source .venv/bin/activate
streamlit run app.py
# Opens at http://localhost:8501
```

### Running tests (no TensorFlow required)
```bash
.venv/bin/pytest tests/ -v
```

### Running training (Python 3.12 + TensorFlow required)
```bash
# Option A — Google Colab (recommended)
# Open training/Aura_Farming_Colab_Training.ipynb in Colab
# Set runtime to T4 GPU and follow the notebook sections

# Option B — local Python 3.12 environment
python3.12 -m venv .venv312
source .venv312/bin/activate
pip install -r requirements.txt
pip install tensorflow-macos tensorflow-metal   # Apple Silicon
# OR: pip install tensorflow                   # Standard
python training/train_baseline.py --fine-tune
```

---

## Dataset and model files — NOT in git

These files are excluded from the repository and must be stored separately:

| File/Directory | Size | Storage suggestion |
|---|---|---|
| `data/plantvillage/` | ~918 MB | Google Drive or re-clone from GitHub |
| `data/splits/*.csv` | Small (~4 MB total) | Re-generate with `prepare_dataset.py`, or keep on Drive |
| `models/mobilenetv3_phase_a.keras` | 4.4 MB | Google Drive |
| `models/mobilenetv3_baseline.keras` | ~4.4 MB (after Phase B) | Google Drive |

To re-generate the splits after re-downloading the dataset:
```bash
python training/inspect_dataset.py
python training/duplicate_detection.py
python training/prepare_dataset.py
```

---

## Key design decisions (from approved Phase 1 spec)

- **Model:** MobileNetV3 Small, 224 × 224 input
- **Preprocessing:** manual `x / 127.5 - 1.0` (identical in training and inference)
- **Class imbalance:** sklearn `balanced` class weights, computed from training set only
- **Confidence threshold:** will be set on validation set in Phase 5; not yet set; do not use 0.85 as a hard-coded value
- **Grad-CAM failure:** Option A — show result without heatmap; do not treat visualization failure as inference failure
- **Upload limit:** 20 MB
- **Test set:** `data/splits/test_manifest.csv` is sealed — never load it during training, calibration, or threshold selection

---

## Dataset attribution (required)

> Mohanty, S. P., Hughes, D. P., & Salathé, M. (2016). *Using deep learning for image-based plant disease detection.* Frontiers in Plant Science, 7, 1419. https://doi.org/10.3389/fpls.2016.01419

License: **CC BY-SA 3.0** — attribution required in any published work.

---

## Disclaimer

Aura Farming is a research prototype. AI predictions are not a substitute for diagnosis by a qualified agricultural expert or plant pathologist. Always consult a professional before making treatment decisions.
