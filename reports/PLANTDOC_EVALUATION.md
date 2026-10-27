# 🍃 Aura Farming — PlantDoc Out-of-Domain Evaluation Protocol

## 1. Domain Adaptation & Dataset Context
**PlantDoc** is a benchmark dataset of crop photographs captured in authentic agricultural field conditions (uncontrolled backgrounds, variable natural sunlight, overlapping foliage, and camera angles), unlike the controlled, uniform-background studio setting of **PlantVillage**.

Evaluating on PlantDoc represents a stringent test of **out-of-domain generalization** and robustness to distribution shift.

---

## 2. Class Mapping Specification
Aura Farming's model classifies across 10 tomato categories. PlantDoc provides label strings that map to 9 of these 10 categories:

| PlantDoc Label Name | Aura Farming Target Class | Class Index | Status |
|---|---|:---:|---|
| `Tomato leaf bacterial spot` | Tomato — Bacterial Spot | 0 | Mapped |
| `Tomato Early blight leaf` | Tomato — Early Blight | 1 | Mapped |
| `Tomato Late blight leaf` | Tomato — Late Blight | 2 | Mapped |
| `Tomato mold leaf` | Tomato — Leaf Mold | 3 | Mapped |
| `Tomato Septoria leaf spot` | Tomato — Septoria Leaf Spot | 4 | Mapped |
| `Tomato two spotted spider mites leaf` | Tomato — Two-spotted Spider Mites | 5 | Mapped |
| `Tomato leaf yellow virus` | Tomato — Yellow Leaf Curl Virus | 7 | Mapped |
| `Tomato leaf mosaic virus` | Tomato — Mosaic Virus | 8 | Mapped |
| `Tomato leaf` | Tomato — Healthy | 9 | Mapped |
| *(Missing in PlantDoc)* | **Tomato — Target Spot** | **6** | ❌ **Incompatible (Missing)** |

### Key Domain Mismatch Note:
- **Target Spot (Class 6):** PlantDoc does not contain annotated Target Spot samples. When evaluating on PlantDoc, Target Spot must be excluded from ground-truth support, or the evaluation must use a 9-class constrained mapping.
- **Background Clutter:** Field images contain non-leaf background, soil, and stems. The Image Quality Gate (`src/image_quality.py`) and abstention thresholds play a vital role in preventing overconfident hallucinations on cluttered out-of-domain field photographs.

---

## 3. Dataset Setup & Execution Instructions
To evaluate on PlantDoc when downloaded:
1. Clone or extract the PlantDoc dataset to:
   ```bash
   data/plantdoc/
   ```
2. Verify folder structure matches standard PlantDoc format:
   ```
   data/plantdoc/
   ├── test/
   │   ├── Tomato leaf bacterial spot/
   │   └── ...
   ```
3. Run the evaluation script with the PlantDoc flag:
   ```bash
   python training/evaluate_model.py --plantdoc-dir data/plantdoc
   ```
