"""
Aura Farming — src/preprocessing.py
======================================
Image preprocessing pipeline for MobileNetV3Small inference.

Phase 4 status: COMPLETE
─────────────────────────
This module converts a PIL Image (already validated and quality-checked)
into the NumPy tensor expected by the trained MobileNetV3Small model.

Critical consistency requirement:
    The preprocessing here MUST be identical to what training/train_baseline.py
    applies in build_tf_dataset() during training:

        Step 1: Resize to (224, 224)
        Step 2: Convert to float32
        Step 3: Manual MobileNetV3 rescaling: x / 127.5 - 1.0 → [-1, 1]

    Using a different normalization at inference time is a common source
    of silently degraded accuracy. The integration test in
    tests/test_preprocessing.py verifies the output range is [-1, 1].

Why manual rescaling instead of keras.applications.mobilenet_v3.preprocess_input:
    In Keras 3 (TF >= 2.16), preprocess_input became a no-op (identity
    function) because the rescaling was moved into the model's internal
    Rescaling layer. Since we build the model with
    include_preprocessing=False, we must rescale manually. The formula
    x / 127.5 - 1.0 is identical to what the internal Rescaling layer does.

TensorFlow import:
    TensorFlow is imported lazily (inside the function) so the Streamlit UI
    can import this module without crashing when TF is not installed.
    The app shows a development-state notice in that case.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import yaml
from PIL import Image


def _load_config() -> dict:
    config_path = Path(__file__).parent.parent / "configs" / "config.yaml"
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def preprocess_for_inference(image: Image.Image) -> Optional[np.ndarray]:
    """
    Preprocess a PIL image for MobileNetV3Small inference.

    This function applies exactly the same steps as the training pipeline
    in training/train_baseline.py → build_tf_dataset → _load_and_preprocess.
    Both use mobilenet_v3.preprocess_input which maps [0, 255] → [-1, 1].

    Args:
        image: A validated, quality-checked PIL Image in RGB mode.
               Must have been produced by src/image_validation.validate_upload.

    Returns:
        NumPy float32 array of shape (1, 224, 224, 3) with values in [-1, 1],
        ready to pass to model.predict().
        Returns None if TensorFlow is not available (graceful degradation).

    Raises:
        ValueError: If the image is not in RGB mode.
    """
    if image.mode != "RGB":
        raise ValueError(
            f"Expected an RGB image, got mode '{image.mode}'. "
            "Ensure the image was decoded via image_validation.validate_upload."
        )

    cfg = _load_config()
    h, w = cfg["model"]["input_size"]  # [224, 224]

    try:
        import tensorflow as tf  # noqa: F401 — availability check only
    except ImportError:
        # TensorFlow not installed — return None so the UI can show
        # a development-state notice rather than crashing.
        return None

    # Resize using Pillow (LANCZOS = high quality downsampling)
    resized = image.resize((w, h), resample=Image.Resampling.LANCZOS)

    # Convert to NumPy float32 array, shape (H, W, 3)
    arr = np.array(resized, dtype=np.float32)

    # Manual MobileNetV3 rescaling: [0, 255] → [-1, 1]
    #
    # We apply this manually instead of using
    # keras.applications.mobilenet_v3.preprocess_input, which became
    # a no-op in Keras 3 (TF >= 2.16). The model is built with
    # include_preprocessing=False, so rescaling must happen here.
    #
    # The formula x / 127.5 - 1.0 is identical to the Rescaling layer
    # that Keras uses internally when include_preprocessing=True.
    arr = arr / 127.5 - 1.0

    # Add batch dimension: (H, W, 3) → (1, H, W, 3)
    arr = np.expand_dims(arr, axis=0)

    return arr
