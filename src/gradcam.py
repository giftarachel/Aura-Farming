"""
Aura Farming — src/gradcam.py
================================
Grad-CAM (Gradient-weighted Class Activation Mapping) visualization.

Phase 5 Implementation: COMPLETE
─────────────────────────────────
What Grad-CAM does:
  Grad-CAM computes the visual explanation for a deep learning model's prediction
  by measuring the gradient of the predicted class score with respect to the
  activations of the final convolutional layer. High-gradient regions indicate
  features that contributed most positively to the classification decision.

MobileNetV3 Small Architecture:
  The Aura Farming baseline wraps MobileNetV3Small (named 'mobilenetv3small')
  with a custom classification head (GAP -> BatchNorm -> Dropout -> Dense(10, softmax)).
  The output of 'mobilenetv3small' provides the 4D convolutional feature map
  (1, 7, 7, 576), from which class gradients are computed using tf.GradientTape.

Scientific Limitations:
  - Grad-CAM highlights image regions that influenced the model's mathematical score.
  - Heatmaps do NOT establish causal disease etiology or prove disease presence.
  - Saliency must not be mistaken for definitive ground truth biological markers.
"""

from __future__ import annotations

from typing import Optional, Tuple, Union

import cv2
import numpy as np
from PIL import Image

from src.predictor import _load_model_cached, ModelNotAvailableError


# ─────────────────────────────────────────────────────────────────
# Custom Exception
# ─────────────────────────────────────────────────────────────────

class GradCAMError(Exception):
    """
    Raised when Grad-CAM heatmap generation fails.

    Per project architecture (Option A):
        Grad-CAM failure does NOT constitute an inference failure.
        The UI layer catches GradCAMError and displays the prediction
        along with an explanation that the visualization is unavailable.
    """


# ─────────────────────────────────────────────────────────────────
# Standalone NumPy Core for Heatmap Computation & Blending
# (Usable with or without active TensorFlow runtime)
# ─────────────────────────────────────────────────────────────────

def generate_heatmap_from_gradients(
    conv_features: np.ndarray,
    gradients: np.ndarray,
) -> np.ndarray:
    """
    Compute normalized 2D Grad-CAM heatmap from feature maps and gradients.

    L_{Grad-CAM}^c = ReLU( sum_k alpha_k^c * A^k )
    Where alpha_k^c = (1 / (H * W)) * sum_i sum_j (d y^c / d A_{i, j}^k)

    Args:
        conv_features: NumPy array of shape (1, H, W, C) or (H, W, C).
        gradients: NumPy array of shape (1, H, W, C) or (H, W, C).

    Returns:
        2D NumPy float32 array of shape (H, W) with values normalized to [0.0, 1.0].
    """
    feat = np.asarray(conv_features, dtype=np.float32)
    grads = np.asarray(gradients, dtype=np.float32)

    if feat.ndim == 4:
        feat = feat[0]
    if grads.ndim == 4:
        grads = grads[0]

    if feat.ndim != 3 or grads.ndim != 3:
        raise ValueError(f"Expected 3D feature arrays (H, W, C), got {feat.shape} and {grads.shape}.")

    if feat.shape != grads.shape:
        raise ValueError(f"Feature shape {feat.shape} and gradient shape {grads.shape} must match.")

    # Check for NaN / Inf
    if np.any(np.isnan(feat)) or np.any(np.isnan(grads)):
        raise GradCAMError("Convolutional features or gradients contain NaN values.")

    # Global average pooling of gradients along spatial dimensions (H, W)
    weights = np.mean(grads, axis=(0, 1))  # shape: (C,)

    # Linear combination of feature maps weighted by channel importance
    cam = np.sum(feat * weights, axis=-1)  # shape: (H, W)

    # Apply ReLU: keep only features with positive influence on the target class
    cam = np.maximum(cam, 0.0)

    # Normalize to [0.0, 1.0]
    max_val = float(np.max(cam))
    if max_val > 1e-8:
        cam = cam / max_val
    else:
        # Near-zero or uniform zero heatmap
        cam = np.zeros_like(cam, dtype=np.float32)

    return cam.astype(np.float32)


def blend_heatmap_with_image(
    original_image: Image.Image,
    heatmap: np.ndarray,
    alpha: float = 0.4,
    colormap: int = cv2.COLORMAP_JET,
) -> Image.Image:
    """
    Resize heatmap, apply colormap, and blend with the original PIL image.

    Args:
        original_image: Original input PIL Image (RGB).
        heatmap: 2D float32 NumPy array in [0.0, 1.0].
        alpha: Blending weight for the heatmap (0.0 = original only, 1.0 = heatmap only).
        colormap: OpenCV colormap constant (default: cv2.COLORMAP_JET).

    Returns:
        PIL.Image in RGB mode showing the heatmap overlay.
    """
    if original_image.mode != "RGB":
        orig_rgb = original_image.convert("RGB")
    else:
        orig_rgb = original_image

    w, h = orig_rgb.size
    orig_arr = np.array(orig_rgb, dtype=np.uint8)

    # 1. Resize heatmap to match original image dimensions (W, H)
    resized_cam = cv2.resize(heatmap, (w, h), interpolation=cv2.INTER_LINEAR)
    resized_cam = np.clip(resized_cam, 0.0, 1.0)

    # 2. Convert to uint8 (0-255) and apply OpenCV colormap
    cam_uint8 = np.uint8(255 * resized_cam)
    colored_bgr = cv2.applyColorMap(cam_uint8, colormap)
    colored_rgb = cv2.cvtColor(colored_bgr, cv2.COLOR_BGR2RGB)

    # 3. Alpha blending: overlay = alpha * heatmap + (1 - alpha) * original
    alpha = float(np.clip(alpha, 0.0, 1.0))
    blended = (alpha * colored_rgb.astype(np.float32) + (1.0 - alpha) * orig_arr.astype(np.float32))
    blended = np.clip(blended, 0.0, 255.0).astype(np.uint8)

    return Image.fromarray(blended, mode="RGB")


# ─────────────────────────────────────────────────────────────────
# Model Layer Inspector
# ─────────────────────────────────────────────────────────────────

def find_target_convolutional_layer(model, layer_name: Optional[str] = None):
    """
    Locate the target convolutional layer for Grad-CAM.

    For Aura Farming's MobileNetV3 Small architecture:
      - Checks for the base layer named 'mobilenetv3small'.
      - If found, searches inside the base model for the final Conv2D/activation layer,
        or returns the base layer itself.
      - If model is flat or custom, searches backwards for the last layer with 4D output.

    Args:
        model: tf.keras.Model
        layer_name: Optional explicit layer name.

    Returns:
        Target layer object.

    Raises:
        GradCAMError: If no suitable convolutional layer can be identified.
    """
    if layer_name:
        try:
            return model.get_layer(layer_name)
        except Exception:
            pass

    # 1. Check for nested MobileNetV3Small base
    try:
        base_layer = model.get_layer("mobilenetv3small")
        # Base layer itself produces the 4D feature map (1, 7, 7, 576)
        return base_layer
    except (ValueError, KeyError, AttributeError):
        pass

    # 2. Search backwards in model layers for last 4D output
    for layer in reversed(model.layers):
        try:
            shape = getattr(layer, "output_shape", None)
            if shape and len(shape) == 4:
                return layer
        except Exception:
            continue

    raise GradCAMError("Could not identify a convolutional feature layer for Grad-CAM.")


# ─────────────────────────────────────────────────────────────────
# Gradient Computation with TensorFlow
# ─────────────────────────────────────────────────────────────────

def compute_gradcam_tensors(
    model,
    tensor: np.ndarray,
    class_index: int,
    layer_name: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute forward feature activations and backward gradients for class_index.

    Supports both nested MobileNetV3 architecture and flat models.

    Args:
        model: tf.keras.Model.
        tensor: NumPy array of shape (1, 224, 224, 3) in [-1.0, 1.0].
        class_index: Target class integer index.
        layer_name: Optional layer name.

    Returns:
        Tuple of (conv_features, gradients) as NumPy arrays.

    Raises:
        GradCAMError: On model failure or gradient computation failure.
    """
    try:
        import tensorflow as tf
    except ImportError as exc:
        raise GradCAMError(
            "TensorFlow is not installed in the active environment. "
            "Grad-CAM requires TensorFlow for automatic differentiation."
        ) from exc

    if tensor.ndim != 4 or tensor.shape[1:] != (224, 224, 3):
        raise GradCAMError(f"Invalid input tensor shape {tensor.shape}; expected (1, 224, 224, 3).")

    tf_tensor = tf.convert_to_tensor(tensor, dtype=tf.float32)

    # Strategy A: Nested MobileNetV3Small (Aura Farming architecture)
    try:
        base_layer = model.get_layer("mobilenetv3small")
        gap_layer = model.get_layer("gap")
        bn_layer = model.get_layer("bn_head")
        drop_layer = model.get_layer("dropout_head")
        pred_layer = model.get_layer("predictions")

        with tf.GradientTape() as tape:
            conv_features = base_layer(tf_tensor, training=False)
            tape.watch(conv_features)
            # Forward pass through classification head
            x = gap_layer(conv_features)
            x = bn_layer(x, training=False)
            x = drop_layer(x, training=False)
            predictions = pred_layer(x)
            target_score = predictions[:, class_index]

        grads = tape.gradient(target_score, conv_features)
        if grads is None:
            raise GradCAMError("Computed gradient is None.")

        return conv_features.numpy(), grads.numpy()

    except (ValueError, KeyError, AttributeError):
        pass

    # Strategy B: Standard or Flat Model
    target_layer = find_target_convolutional_layer(model, layer_name)
    try:
        grad_model = tf.keras.models.Model(
            inputs=model.inputs,
            outputs=[target_layer.output, model.output],
        )

        with tf.GradientTape() as tape:
            conv_features, predictions = grad_model(tf_tensor, training=False)
            target_score = predictions[:, class_index]

        grads = tape.gradient(target_score, conv_features)
        if grads is None:
            raise GradCAMError("Computed gradient is None.")

        return conv_features.numpy(), grads.numpy()

    except Exception as exc:
        raise GradCAMError(f"Gradient computation failed: {exc}") from exc


# ─────────────────────────────────────────────────────────────────
# Public Interface
# ─────────────────────────────────────────────────────────────────

def generate_gradcam_overlay(
    image: Image.Image,
    tensor: np.ndarray,
    class_index: int,
    model=None,
    alpha: float = 0.4,
    layer_name: Optional[str] = None,
) -> Optional[Image.Image]:
    """
    Generate a Grad-CAM heatmap overlay for a predicted class on a crop leaf photograph.

    Args:
        image: Original uploaded PIL Image (RGB mode).
        tensor: Preprocessed tensor of shape (1, 224, 224, 3) in [-1, 1].
        class_index: Predicted disease class index.
        model: Optional tf.keras.Model. If None, attempts to load cached model.
        alpha: Blending weight for heatmap (0.0 to 1.0).
        layer_name: Optional convolutional layer name.

    Returns:
        PIL.Image of original photograph overlaid with the Grad-CAM heatmap.

    Raises:
        GradCAMError: If model is missing, inputs are invalid, or gradients fail.
    """
    if tensor is None:
        raise GradCAMError("Input tensor is None.")

    if class_index is None or not isinstance(class_index, (int, np.integer)):
        raise GradCAMError(f"Invalid class index: {class_index}.")

    # Load model if not provided
    if model is None:
        try:
            model = _load_model_cached()
        except ModelNotAvailableError as exc:
            raise GradCAMError(f"Trained model checkpoint is not available: {exc}") from exc
        except Exception as exc:
            raise GradCAMError(f"Could not load model for Grad-CAM: {exc}") from exc

    # Compute features and gradients
    conv_features, grads = compute_gradcam_tensors(
        model=model,
        tensor=tensor,
        class_index=int(class_index),
        layer_name=layer_name,
    )

    # Compute heatmap
    heatmap = generate_heatmap_from_gradients(conv_features, grads)

    # Blend with original image
    overlay = blend_heatmap_with_image(
        original_image=image,
        heatmap=heatmap,
        alpha=alpha,
    )

    return overlay
