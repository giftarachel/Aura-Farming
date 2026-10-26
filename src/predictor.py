"""
Aura Farming — src/predictor.py
==================================
Model loading and disease prediction for the Streamlit application.

Phase 4 status: COMPLETE
─────────────────────────
This module loads the trained MobileNetV3Small model on first use and
runs inference on preprocessed image tensors.

Usage pattern in the UI:
    from src.predictor import is_model_available, predict, ModelNotAvailableError

    if not is_model_available():
        render_dev_notice("Model not yet connected.")
    else:
        probs = predict(tensor)   # shape (10,)
        # pass probs to src/calibration.py in Phase 5

Model caching:
    The model is expensive to load (~10 MB, several hundred milliseconds).
    @st.cache_resource ensures it is loaded once per Streamlit session and
    reused for every subsequent prediction.

What predict() returns:
    A 1-D NumPy float32 array of shape (num_classes,) containing the raw
    softmax probabilities from the model output. The argmax is NOT taken
    here — that decision is deferred to src/calibration.py in Phase 5,
    which may apply temperature scaling and a confidence threshold.

Test set protection:
    This module never reads from data/splits/test_manifest.csv.
    Inference is on individual uploaded images, not batch evaluation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import yaml


def _load_config() -> dict:
    config_path = Path(__file__).parent.parent / "configs" / "config.yaml"
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


# ─────────────────────────────────────────────────────────────────
# Custom exceptions
# ─────────────────────────────────────────────────────────────────

class ModelNotAvailableError(Exception):
    """
    Raised when the trained model weights file is not found.

    This is an expected state before Phase 4 training is complete.
    The Streamlit UI catches this and shows a development-state notice.
    It must never be silently swallowed into a prediction.
    """


class InferenceError(Exception):
    """
    Raised when the model forward pass fails for any reason.

    This must never be silently converted into a disease prediction.
    The UI must handle it explicitly (show an error message).
    """


# ─────────────────────────────────────────────────────────────────
# Model availability check (no TF required)
# ─────────────────────────────────────────────────────────────────

def is_model_available() -> bool:
    """
    Return True if a trained model file exists at the configured path.

    Reads model.weights_path from config.yaml. Does NOT import
    TensorFlow, so this function is safe to call from any context.

    Returns:
        True if the weights file exists on disk; False otherwise.
    """
    cfg = _load_config()
    weights_path = cfg["model"].get("weights_path")
    if not weights_path:
        return False
    resolved = Path(__file__).parent.parent / weights_path
    return resolved.is_file()


# ─────────────────────────────────────────────────────────────────
# Cached model loader
# ─────────────────────────────────────────────────────────────────

def _load_model_cached():
    """
    Load the trained Keras model, cached for the Streamlit session.

    This function is decorated with @st.cache_resource when called
    from the Streamlit app. It is defined separately so it can also
    be called from non-Streamlit contexts (e.g. tests) without
    needing the Streamlit runtime.

    Returns:
        The loaded tf.keras.Model.

    Raises:
        ModelNotAvailableError: weights_path not set or file not found.
        InferenceError: model file found but fails to load.
    """
    try:
        import tensorflow as tf
    except ImportError:
        raise ModelNotAvailableError(
            "TensorFlow is not installed in this environment. "
            "See README.md → 'Training Setup' for installation instructions."
        )

    cfg = _load_config()
    weights_path_str = cfg["model"].get("weights_path")

    if not weights_path_str:
        raise ModelNotAvailableError(
            "model.weights_path is not set in configs/config.yaml. "
            "Run training/train_baseline.py to train a model first."
        )

    weights_path = Path(__file__).parent.parent / weights_path_str
    if not weights_path.is_file():
        raise ModelNotAvailableError(
            f"Model file not found: {weights_path}\n"
            "Run training/train_baseline.py to train a model first."
        )

    try:
        model = tf.keras.models.load_model(str(weights_path))
        return model
    except Exception as exc:
        raise InferenceError(
            f"Failed to load model from {weights_path}: {exc}"
        ) from exc


# ─────────────────────────────────────────────────────────────────
# Public inference function
# ─────────────────────────────────────────────────────────────────

def predict(tensor: np.ndarray) -> np.ndarray:
    """
    Run inference on a preprocessed image tensor.

    The tensor must have been produced by
    src.preprocessing.preprocess_for_inference, which applies
    exactly the same normalization as training.

    Args:
        tensor: float32 NumPy array of shape (1, 224, 224, 3),
                values in [-1, 1].

    Returns:
        float32 NumPy array of shape (num_classes,) containing the
        raw softmax probability distribution.
        The argmax is NOT taken — pass the full distribution to
        src.calibration.apply_calibration_and_threshold (Phase 5).

    Raises:
        ModelNotAvailableError: No trained model configured or found.
        InferenceError: The forward pass raised an unexpected error.
        ValueError: Tensor has wrong shape or dtype.
    """
    if tensor is None:
        raise ModelNotAvailableError(
            "Received None tensor. TensorFlow may not be installed; "
            "src.preprocessing.preprocess_for_inference returned None."
        )

    if tensor.ndim != 4 or tensor.shape[1:] != (224, 224, 3):
        raise ValueError(
            f"Expected tensor shape (1, 224, 224, 3), got {tensor.shape}. "
            "Ensure src.preprocessing.preprocess_for_inference was used."
        )

    model = _load_model_cached()

    try:
        probs_batch = model.predict(tensor, verbose=0)  # shape (1, num_classes)
        probs = probs_batch[0].astype(np.float32)       # shape (num_classes,)
        return probs
    except Exception as exc:
        raise InferenceError(
            f"Model inference failed: {exc}\n"
            "Check that the model file is not corrupted and that "
            "the input tensor has the correct shape and dtype."
        ) from exc
