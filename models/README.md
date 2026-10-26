# Aura Farming — models/

This directory stores trained model artifacts. These files are **not committed to Git** because they can be hundreds of megabytes in size.

## Files that belong here

| Filename | Created in | Description |
|---|---|---|
| `mobilenetv3_baseline.keras` | Phase 4 | Trained MobileNetV3 Small weights |
| `mobilenetv3_calibration.json` | Phase 5 | Temperature scaling parameter and confidence threshold |
| `efficientnetb0_baseline.keras` | Phase 4 (optional) | EfficientNetB0 comparison model |

## How to obtain the model files

1. **Training on Google Colab (recommended):**
   - Open the Colab notebook provided in the project documentation.
   - Mount your Google Drive.
   - Run `training/train_baseline.py` and `training/calibrate_model.py`.
   - Download the resulting `.keras` and `.json` files to this directory.

2. **Shared model files:**
   - If model files are shared via Google Drive or another platform,
     download them and place them here.
   - The application reads the path from `configs/config.yaml`
     under `model.weights_path` and `model.calibration_path`.

## Updating config.yaml after placing a model

Edit `configs/config.yaml` and set:

```yaml
model:
  weights_path: "models/mobilenetv3_baseline.keras"
  calibration_path: "models/mobilenetv3_calibration.json"
```

The application will then detect the model and enable AI inference.
