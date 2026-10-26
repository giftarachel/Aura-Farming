# Aura Farming — Dataset Preparation Document

**Phase 3 — Dataset Source, Licensing, and Preparation Methodology**

---

## 1. Dataset Source

**Dataset:** PlantVillage  
**Original paper:** Mohanty, S. P., Hughes, D. P., & Salathé, M. (2016). *Using deep learning for image-based plant disease detection.* Frontiers in Plant Science, 7, 1419. https://doi.org/10.3389/fpls.2016.01419  
**GitHub repository:** https://github.com/spMohanty/PlantVillage-Dataset  
**Hugging Face Hub:** https://huggingface.co/datasets/mohanty/PlantVillage  
**License:** CC BY-SA 3.0 (Creative Commons Attribution-ShareAlike 3.0)

### License obligations

- **Attribution required:** Cite Mohanty, Hughes & Salathé (2016) in any published work using this dataset.
- **ShareAlike:** Derivative datasets must be shared under the same or a compatible license.
- **Free to use** for research and educational purposes.

---

## 2. Dataset Characteristics

| Property | Value |
|---|---|
| Variant used | Color (RGB) |
| Image format | JPEG (.JPG) |
| Typical dimensions | 256 × 256 pixels (original paper) |
| Model input size | 224 × 224 (resized during preprocessing, Phase 4) |
| Controlled conditions | Yes — lab background, uniform lighting |
| Total tomato images | ~18,160 across 10 classes (approx.) |
| Total dataset (all crops) | ~54,306 images across 38 classes |

---

## 3. Tomato Class Scope

The initial version of Aura Farming uses the 10 tomato classes available in PlantVillage.
All 9 disease conditions plus the healthy class are included.

| Index | Folder Name | Display Name | Approx. Images |
|---|---|---|---|
| 0 | `Tomato___Bacterial_spot` | Bacterial Spot | ~2,127 |
| 1 | `Tomato___Early_blight` | Early Blight | ~1,000+ |
| 2 | `Tomato___Late_blight` | Late Blight | ~1,909 |
| 3 | `Tomato___Leaf_Mold` | Leaf Mold | ~952 |
| 4 | `Tomato___Septoria_leaf_spot` | Septoria Leaf Spot | ~1,771 |
| 5 | `Tomato___Spider_mites Two-spotted_spider_mite` | Spider Mites | ~1,676 |
| 6 | `Tomato___Target_Spot` | Target Spot | ~1,404 |
| 7 | `Tomato___Tomato_Yellow_Leaf_Curl_Virus` | Yellow Leaf Curl Virus | ~5,357 |
| 8 | `Tomato___Tomato_mosaic_virus` | Tomato Mosaic Virus | ~373 |
| 9 | `Tomato___healthy` | Healthy | ~1,591 |

⚠ **Note on counts:** GitHub API returns a maximum of 1,000 items per directory listing.
Classes showing exactly 1,000 may have more. Confirmed counts come only from `inspect_dataset.py`
run against the downloaded dataset.

⚠ **Class imbalance:** The largest class (Yellow Leaf Curl Virus, ~5,357 images) is approximately
14× larger than the smallest (Tomato Mosaic Virus, ~373 images). Class weighting or oversampling
should be considered during Phase 4 training.

---

## 4. Index Assignment Procedure

Integer class indices (0–9) are assigned **alphabetically by PlantVillage folder name**.
This is defined explicitly in `configs/class_mapping.json` under `folder_to_index`.

**The training pipeline must load this mapping file explicitly.** It must never infer indices
from `os.listdir()` sort order, which could vary across operating systems or Python versions.

---

## 5. Download Instructions

### Option A — Hugging Face (recommended)

Provides pre-built 80/20 train/test splits with leaf-grouping leakage prevention.
```bash
pip install datasets
python -c "
from datasets import load_dataset
ds = load_dataset('mohanty/PlantVillage', 'color')
"
```

Then export the images to the local directory structure expected by this project.

### Option B — Git clone (~2 GB)

```bash
git clone https://github.com/spMohanty/PlantVillage-Dataset
mkdir -p data/plantvillage
mv PlantVillage-Dataset/raw/color data/plantvillage/color
```

---

## 6. Split Strategy

| Split | Ratio | Purpose |
|---|---|---|
| Training | 70% | Model training only |
| Validation | 15% | Hyperparameter selection, early stopping, threshold selection |
| Test | 15% | Final evaluation — **never touched during training or calibration** |

Random seed: 42 (configurable in `configs/config.yaml`)

### Leakage prevention

1. **Exact duplicates** are detected by MD5 hash. Duplicate groups are kept in a single split.
2. **Near-duplicates** are detected by perceptual hash (pHash) with Hamming distance ≤ 4.
   Near-duplicate groups are flagged for review and kept in a single split.
3. **Post-split verification** confirms no file path appears in more than one manifest.
4. The split manifests (`train_manifest.csv`, `val_manifest.csv`, `test_manifest.csv`) are
   saved to `data/splits/` and excluded from Git.

---

## 7. Actual Results

> **Status: PENDING** — Run the following scripts against the downloaded dataset to populate
> this section with real results. Do not fill in fabricated counts.

```bash
python training/inspect_dataset.py --output data/reports/inspection.json
python training/duplicate_detection.py --output data/reports/duplicates.json
python training/prepare_dataset.py
```

After running, record here:
- Actual confirmed image counts per class
- Number of exact duplicate groups found
- Number of near-duplicate groups flagged
- Final train/val/test counts (total and per class)
- Leakage check result

---

## 8. Limitations

- PlantVillage images are collected under controlled laboratory conditions with uniform lighting
  and plain backgrounds. Performance on real farm photographs (which have variable lighting,
  backgrounds, and imaging conditions) may differ significantly.
- `Tomato___Tomato_mosaic_virus` has only ~373 images, which is substantially less than other
  classes. This class will have small validation and test sets (~56 and ~56 images respectively
  at a 70/15/15 split), limiting the precision of per-class evaluation.
- The class imbalance between Yellow Leaf Curl Virus (~5,357) and Mosaic Virus (~373) is ~14×.
  This will be addressed with class weighting during Phase 4 training.

---

## 9. Citation

```bibtex
@article{Mohanty_Hughes_Salathe_2016,
  title   = {Using deep learning for image-based plant disease detection},
  volume  = {7},
  DOI     = {10.3389/fpls.2016.01419},
  journal = {Frontiers in Plant Science},
  author  = {Mohanty, Sharada P. and Hughes, David P. and Salath\'{e}, Marcel},
  year    = {2016},
  month   = {Sep}
}
```
