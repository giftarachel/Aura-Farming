"""
Aura Farming — tests/test_preprocessing.py
============================================
Unit tests for src/preprocessing.py.

These tests run WITHOUT TensorFlow being installed. They verify:
  - Graceful None return when TF is absent.
  - Input validation (mode, shape).
  - Config-driven input size is respected.
  - The output tensor has the correct shape when TF is available.
  - The output value range is [-1, 1] (matching MobileNetV3 training).

The TF-dependent tests are skipped automatically when TensorFlow
is not importable (which is the case in the Python 3.14 venv).

Run with:
    pytest tests/test_preprocessing.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ── Check TF availability once ───────────────────────────────────
TF_AVAILABLE = False
try:
    import tensorflow  # noqa: F401
    TF_AVAILABLE = True
except ImportError:
    pass

requires_tf = pytest.mark.skipif(
    not TF_AVAILABLE,
    reason="TensorFlow not installed in this environment (Python 3.14). "
           "Run in a Python 3.10–3.12 environment with TF to enable these tests.",
)

from src.preprocessing import preprocess_for_inference


# ─────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────

def _make_rgb_image(width: int = 256, height: int = 256) -> Image.Image:
    return Image.new("RGB", (width, height), color=(120, 160, 80))


def _make_rgba_image(width: int = 256, height: int = 256) -> Image.Image:
    return Image.new("RGBA", (width, height), color=(120, 160, 80, 255))


# ─────────────────────────────────────────────────────────────────
# Tests that run without TensorFlow
# ─────────────────────────────────────────────────────────────────

class TestPreprocessForInferenceNoTF:
    """
    These tests verify behaviour that does NOT require TensorFlow.
    They pass in any environment including the Python 3.14 venv.
    """

    def test_returns_none_when_tf_not_installed(self, monkeypatch):
        """
        When TensorFlow is not importable, the function must return
        None rather than raising an exception. The Streamlit UI
        handles None as a 'model not ready' state.
        """
        import builtins
        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "tensorflow" or name.startswith("tensorflow."):
                raise ImportError("mocked TF not available")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", mock_import)
        result = preprocess_for_inference(_make_rgb_image())
        assert result is None

    def test_raises_for_non_rgb_mode(self):
        """Non-RGB images must raise ValueError immediately."""
        rgba = _make_rgba_image()
        with pytest.raises(ValueError, match="RGB"):
            preprocess_for_inference(rgba)

    def test_raises_for_grayscale_mode(self):
        gray = Image.new("L", (64, 64), color=128)
        with pytest.raises(ValueError, match="RGB"):
            preprocess_for_inference(gray)


# ─────────────────────────────────────────────────────────────────
# Tests that require TensorFlow
# ─────────────────────────────────────────────────────────────────

class TestPreprocessForInferenceWithTF:
    """TF-dependent tests — skipped when TF is not installed."""

    @requires_tf
    def test_output_shape_is_1_224_224_3(self):
        """Output must have shape (1, 224, 224, 3) for MobileNetV3Small."""
        img = _make_rgb_image(300, 400)
        result = preprocess_for_inference(img)
        assert result is not None
        assert result.shape == (1, 224, 224, 3)

    @requires_tf
    def test_output_dtype_is_float32(self):
        """MobileNetV3Small expects float32 input."""
        img = _make_rgb_image()
        result = preprocess_for_inference(img)
        assert result is not None
        assert result.dtype == np.float32

    @requires_tf
    def test_output_values_in_minus1_to_1(self):
        """
        mobilenet_v3.preprocess_input maps [0,255] → [-1, 1].
        This is the critical consistency check between training
        and inference.
        """
        img = _make_rgb_image()
        result = preprocess_for_inference(img)
        assert result is not None
        assert result.min() >= -1.0 - 1e-5
        assert result.max() <= 1.0 + 1e-5

    @requires_tf
    def test_black_image_maps_to_minus_one(self):
        """Pure black (0, 0, 0) should map to approximately -1."""
        black = Image.new("RGB", (64, 64), color=(0, 0, 0))
        result = preprocess_for_inference(black)
        assert result is not None
        np.testing.assert_allclose(result, -1.0, atol=1e-4)

    @requires_tf
    def test_white_image_maps_to_one(self):
        """Pure white (255, 255, 255) should map to approximately +1."""
        white = Image.new("RGB", (64, 64), color=(255, 255, 255))
        result = preprocess_for_inference(white)
        assert result is not None
        np.testing.assert_allclose(result, 1.0, atol=1e-4)

    @requires_tf
    def test_small_input_is_upsampled(self):
        """Input smaller than 224×224 must be resized correctly."""
        small = _make_rgb_image(64, 64)
        result = preprocess_for_inference(small)
        assert result is not None
        assert result.shape == (1, 224, 224, 3)

    @requires_tf
    def test_large_input_is_downsampled(self):
        """Input larger than 224×224 must be resized correctly."""
        large = _make_rgb_image(1024, 768)
        result = preprocess_for_inference(large)
        assert result is not None
        assert result.shape == (1, 224, 224, 3)
