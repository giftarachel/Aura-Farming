"""
Aura Farming — src/gradcam.py
================================
Grad-CAM (Gradient-weighted Class Activation Mapping) visualization.

Phase 2 status: PLACEHOLDER
─────────────────────────────
This module will be fully implemented in Phase 5.

What Grad-CAM does (explained simply):
  After the model makes a prediction, Grad-CAM computes which
  parts of the input image had the most influence on that
  prediction. It does this by measuring the gradient of the
  predicted class score with respect to the final convolutional
  layer's output. Areas with strong gradients are shown in warm
  colours (red/orange); areas with little influence are shown in
  cool colours (blue/green).

  The resulting heatmap is overlaid on the original image so
  users can see WHICH leaf regions the model focused on.

Important caveats:
  - Grad-CAM shows model attention, NOT confirmed disease markers.
  - The highlighted regions are not ground truth for diagnosis.
  - If the heatmap fails to generate, the prediction result is
    still shown (per the Phase 2 decision: Grad-CAM failure ≠
    prediction failure). A clear notice replaces the heatmap.

Planned functionality (Phase 5):
  - Identify the last convolutional layer of MobileNetV3Small.
  - Build a Keras gradient tape model for the target layer.
  - Compute class-weighted activation maps.
  - Resize the heatmap to input image dimensions.
  - Blend with the original image using a colormap.
  - Return a PIL Image for display in the Streamlit results page.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from PIL import Image


# ─────────────────────────────────────────────────────────────────
# Custom exceptions
# ─────────────────────────────────────────────────────────────────

class GradCAMError(Exception):
    """
    Raised when Grad-CAM generation fails.

    Per the approved design (Option A), this exception should be
    caught by the UI layer, which then displays the prediction
    result alongside a notice that the heatmap is unavailable.
    It must NEVER silently become a disease prediction.
    """


# ─────────────────────────────────────────────────────────────────
# Public interface
# ─────────────────────────────────────────────────────────────────

def generate_gradcam_overlay(
    image: Image.Image,
    tensor: np.ndarray,
    class_index: int,
) -> Optional[Image.Image]:
    """
    Generate a Grad-CAM heatmap overlay for a predicted class.

    Phase 2: Returns None (placeholder). No TensorFlow dependency
             required at this stage.
    Phase 5: Will return a PIL Image with the heatmap blended
             onto the original photograph.

    Args:
        image: The original uploaded PIL Image (RGB), used for overlay.
        tensor: Preprocessed NumPy tensor of shape (1, 224, 224, 3).
        class_index: The predicted class index for which to generate
                     the explanation.

    Returns:
        None (Phase 2 placeholder).
        PIL Image with Grad-CAM overlay from Phase 5 onward.

    Raises:
        GradCAMError: If heatmap generation fails in Phase 5.
    """
    # Phase 2 placeholder — implementation deferred to Phase 5.
    return None
