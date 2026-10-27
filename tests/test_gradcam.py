"""
Aura Farming — tests/test_gradcam.py
======================================
Unit tests for Grad-CAM explainability (src/gradcam.py).

Covers:
  - Heatmap computation from features and gradients
  - ReLU activation and normalization to [0.0, 1.0]
  - Handling of zero and degenerate gradients
  - Heatmap resizing and alpha blending onto original PIL images
  - Error handling: missing model, None tensor, wrong shapes, NaN/Inf
  - Lightweight test model evaluation (when TensorFlow is present)

Run with:
    pytest tests/test_gradcam.py -v
"""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from src.gradcam import (
    generate_heatmap_from_gradients,
    blend_heatmap_with_image,
    generate_gradcam_overlay,
    GradCAMError,
)


class TestHeatmapMath:
    def test_heatmap_output_shape_and_range(self):
        # 1 sample, 7x7 spatial, 16 channels
        features = np.random.randn(1, 7, 7, 16).astype(np.float32)
        grads = np.random.randn(1, 7, 7, 16).astype(np.float32)

        cam = generate_heatmap_from_gradients(features, grads)
        assert cam.shape == (7, 7)
        assert cam.dtype == np.float32
        assert np.all(cam >= 0.0)
        assert np.all(cam <= 1.0)
        # Max should be 1.0 when positive activations exist
        assert np.isclose(np.max(cam), 1.0)

    def test_relu_zeros_out_negative_contributions(self):
        # If all features and gradients are negative, ReLU yields all zeros
        features = np.ones((1, 5, 5, 4), dtype=np.float32)
        grads = -1.0 * np.ones((1, 5, 5, 4), dtype=np.float32)

        cam = generate_heatmap_from_gradients(features, grads)
        assert cam.shape == (5, 5)
        assert np.all(cam == 0.0)

    def test_zero_gradients_handled_safely(self):
        features = np.ones((1, 7, 7, 8), dtype=np.float32)
        grads = np.zeros((1, 7, 7, 8), dtype=np.float32)

        cam = generate_heatmap_from_gradients(features, grads)
        assert np.all(cam == 0.0)

    def test_nan_gradients_raise_gradcam_error(self):
        features = np.ones((1, 4, 4, 2), dtype=np.float32)
        grads = np.ones((1, 4, 4, 2), dtype=np.float32)
        grads[0, 1, 1, 0] = np.nan

        with pytest.raises(GradCAMError, match="NaN"):
            generate_heatmap_from_gradients(features, grads)

    def test_mismatched_dimensions_raise_value_error(self):
        features = np.ones((1, 7, 7, 8), dtype=np.float32)
        grads = np.ones((1, 7, 7, 16), dtype=np.float32)

        with pytest.raises(ValueError, match="must match"):
            generate_heatmap_from_gradients(features, grads)


class TestOverlayBlending:
    def test_blended_image_dimensions_match_original(self):
        orig = Image.new("RGB", (320, 240), color=(100, 150, 80))
        heatmap = np.random.uniform(0.0, 1.0, size=(7, 7)).astype(np.float32)

        blended = blend_heatmap_with_image(orig, heatmap, alpha=0.4)
        assert isinstance(blended, Image.Image)
        assert blended.size == (320, 240)
        assert blended.mode == "RGB"

    def test_alpha_zero_preserves_original(self):
        orig = Image.new("RGB", (100, 100), color=(50, 100, 150))
        heatmap = np.ones((7, 7), dtype=np.float32)

        blended = blend_heatmap_with_image(orig, heatmap, alpha=0.0)
        np.testing.assert_array_equal(np.array(orig), np.array(blended))

    def test_rgba_input_converted_to_rgb(self):
        orig_rgba = Image.new("RGBA", (128, 128), color=(40, 80, 120, 255))
        heatmap = np.zeros((4, 4), dtype=np.float32)

        blended = blend_heatmap_with_image(orig_rgba, heatmap, alpha=0.5)
        assert blended.mode == "RGB"


class TestGradCAMInterfaceErrors:
    def test_none_tensor_raises_error(self):
        img = Image.new("RGB", (224, 224))
        with pytest.raises(GradCAMError, match="Input tensor is None"):
            generate_gradcam_overlay(image=img, tensor=None, class_index=0)

    def test_invalid_class_index_raises_error(self):
        img = Image.new("RGB", (224, 224))
        tensor = np.zeros((1, 224, 224, 3), dtype=np.float32)
        with pytest.raises(GradCAMError, match="Invalid class index"):
            generate_gradcam_overlay(image=img, tensor=tensor, class_index=None)

    def test_missing_model_raises_error(self, monkeypatch):
        img = Image.new("RGB", (224, 224))
        tensor = np.zeros((1, 224, 224, 3), dtype=np.float32)
        # Without providing model and with weights missing, raises GradCAMError
        def _mock_load():
            raise ModelNotAvailableError("Weights missing")
        monkeypatch.setattr("src.gradcam._load_model_cached", _mock_load)
        with pytest.raises(GradCAMError):
            generate_gradcam_overlay(image=img, tensor=tensor, class_index=0, model=None)


class TestLightweightTFModelGradCAM:
    """Tests Grad-CAM end-to-end on a lightweight mock Keras model if TF is available."""

    def test_mock_functional_model(self):
        try:
            import tensorflow as tf
        except ImportError:
            pytest.skip("TensorFlow not installed in current environment.")

        # Build minimal mock model: Input -> Conv2D -> GAP -> Dense
        inputs = tf.keras.Input(shape=(224, 224, 3))
        x = tf.keras.layers.Conv2D(8, (3, 3), padding="same", name="mock_conv")(inputs)
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        outputs = tf.keras.layers.Dense(10, activation="softmax")(x)
        mock_model = tf.keras.Model(inputs, outputs)

        img = Image.new("RGB", (224, 224), color=(100, 150, 80))
        tensor = np.zeros((1, 224, 224, 3), dtype=np.float32)

        overlay = generate_gradcam_overlay(
            image=img,
            tensor=tensor,
            class_index=2,
            model=mock_model,
            layer_name="mock_conv",
        )
        assert isinstance(overlay, Image.Image)
        assert overlay.size == (224, 224)
