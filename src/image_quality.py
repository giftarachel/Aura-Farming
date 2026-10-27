"""
Aura Farming — src/image_quality.py
======================================
Image quality assessment gate.

Phase 5 Implementation: COMPLETE
─────────────────────────────────
This module evaluates the photographic quality of crop leaf images before
they are submitted to the neural network for disease classification.

Objectives:
  1. Image validity:
     - Decode verification (PIL & raw bytes)
     - Support for RGB, RGBA, and Grayscale modes
     - Dimension sanity checks (min and max pixel bounds)
  2. Blur detection:
     - Variance of the Laplacian operator (OpenCV)
     - Configurable threshold (quality.blur_laplacian_threshold in config.yaml)
     - Empirical justification: High frequency edge response drops under defocus/motion blur.
  3. Brightness and exposure:
     - Mean pixel intensity in HSV value (V) channel
     - Configurable minimum (underexposed) and maximum (overexposed) thresholds
  4. Structured QualityReport:
     - Clear distinction between hard rejections and soft warnings
     - Actionable feedback and corrective guidance for farmers/users
"""

from __future__ import annotations

import io
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional, Union

import cv2
import numpy as np
import yaml
from PIL import Image, UnidentifiedImageError

from src.image_validation import (
    MIN_DIMENSION_PX,
    MAX_DIMENSION_PX,
    decode_image,
    CorruptedImageError,
    InvalidDimensionsError,
)


# ─────────────────────────────────────────────────────────────────
# Quality Status Enum
# ─────────────────────────────────────────────────────────────────

class QualityStatus(str, Enum):
    """Overall status of the image quality assessment."""
    PASSED = "passed"
    WARNING = "warning"
    FAILED = "failed"


# ─────────────────────────────────────────────────────────────────
# Result dataclass
# ─────────────────────────────────────────────────────────────────

@dataclass
class QualityReport:
    """
    Holds the structured result of an image quality assessment.

    Attributes:
        passed: True if no hard rejection criteria were met; False otherwise.
        blur_score: Laplacian variance score (higher = sharper).
        brightness_mean: Mean pixel brightness (0–255 scale in HSV V-channel).
        width: Image width in pixels.
        height: Image height in pixels.
        failure_reasons: List of critical issues preventing reliable classification.
        warnings: List of non-fatal warnings requiring user caution.
        status: QualityStatus enum (PASSED, WARNING, or FAILED).
        feedback: Human-readable guidance summarizing recommendations.
        can_proceed: Helper boolean indicating if classification can proceed.
        is_placeholder: False for Phase 5 real assessments.
    """
    passed: bool = True
    blur_score: Optional[float] = None
    brightness_mean: Optional[float] = None
    width: int = 0
    height: int = 0
    failure_reasons: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    status: QualityStatus = QualityStatus.PASSED
    feedback: str = ""
    can_proceed: bool = True
    is_placeholder: bool = False


# ─────────────────────────────────────────────────────────────────
# Configuration loader
# ─────────────────────────────────────────────────────────────────

def _load_quality_config() -> dict:
    """Load quality thresholds from configs/config.yaml."""
    config_path = Path(__file__).parent.parent / "configs" / "config.yaml"
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
            return cfg.get("quality", {})
    except Exception:
        return {
            "blur_laplacian_threshold": 100.0,
            "brightness_min": 30,
            "brightness_max": 225,
        }


# ─────────────────────────────────────────────────────────────────
# Quality Assessment Metrics
# ─────────────────────────────────────────────────────────────────

def compute_blur_score(rgb_array: np.ndarray) -> float:
    """
    Compute sharpness score via the variance of the Laplacian.
    """
    if rgb_array.size == 0:
        return 0.0

    gray = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    variance = float(laplacian.var())
    return variance


def compute_brightness_mean(rgb_array: np.ndarray) -> float:
    """
    Compute the mean brightness of an image in HSV color space.
    """
    if rgb_array.size == 0:
        return 0.0

    hsv = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2HSV)
    v_channel = hsv[:, :, 2]
    return float(np.mean(v_channel))


# ─────────────────────────────────────────────────────────────────
# Public Interface
# ─────────────────────────────────────────────────────────────────

def assess_quality(
    image: Union[Image.Image, bytes, np.ndarray],
    blur_threshold: Optional[float] = None,
    brightness_min: Optional[float] = None,
    brightness_max: Optional[float] = None,
) -> QualityReport:
    """
    Assess the photographic quality of a crop leaf photograph.
    """
    cfg = _load_quality_config()
    blur_thresh = float(blur_threshold if blur_threshold is not None else cfg.get("blur_laplacian_threshold", 100.0))
    b_min = float(brightness_min if brightness_min is not None else cfg.get("brightness_min", 30))
    b_max = float(brightness_max if brightness_max is not None else cfg.get("brightness_max", 225))

    failure_reasons: list[str] = []
    warnings: list[str] = []

    # 1. Normalize and validate input to PIL.Image in RGB mode
    pil_img: Optional[Image.Image] = None
    if isinstance(image, bytes):
        try:
            pil_img, _, _ = decode_image(image)
        except (CorruptedImageError, InvalidDimensionsError, Exception) as exc:
            return QualityReport(
                passed=False,
                failure_reasons=[str(exc)],
                status=QualityStatus.FAILED,
                feedback=f"Image decoding failed: {exc}",
                can_proceed=False,
                is_placeholder=False,
            )
    elif isinstance(image, np.ndarray):
        if image.ndim == 2:
            pil_img = Image.fromarray(image).convert("RGB")
        elif image.ndim == 3 and image.shape[2] in (1, 3, 4):
            if image.shape[2] == 1:
                pil_img = Image.fromarray(image.squeeze(-1)).convert("RGB")
            elif image.shape[2] == 4:
                pil_img = Image.fromarray(image, mode="RGBA").convert("RGB")
            else:
                pil_img = Image.fromarray(image, mode="RGB")
        else:
            return QualityReport(
                passed=False,
                failure_reasons=["Invalid NumPy array dimensions for an image."],
                status=QualityStatus.FAILED,
                feedback="Uploaded image array has unsupported shape.",
                can_proceed=False,
                is_placeholder=False,
            )
    elif isinstance(image, Image.Image):
        try:
            pil_img = image.convert("RGB") if image.mode != "RGB" else image
        except Exception as exc:
            return QualityReport(
                passed=False,
                failure_reasons=[f"Could not convert image mode '{image.mode}' to RGB: {exc}"],
                status=QualityStatus.FAILED,
                feedback="Image color mode is unsupported or damaged.",
                can_proceed=False,
                is_placeholder=False,
            )
    else:
        return QualityReport(
            passed=False,
            failure_reasons=["Unsupported input type. Expected PIL Image, bytes, or NumPy array."],
            status=QualityStatus.FAILED,
            feedback="Invalid input provided to quality gate.",
            can_proceed=False,
            is_placeholder=False,
        )

    width, height = pil_img.size

    # 2. Check Dimensions
    dimension_failure = False
    if width < MIN_DIMENSION_PX or height < MIN_DIMENSION_PX:
        failure_reasons.append(
            f"Image dimensions ({width}×{height} px) are smaller than the minimum "
            f"required ({MIN_DIMENSION_PX}×{MIN_DIMENSION_PX} px)."
        )
        dimension_failure = True
    elif width < 128 or height < 128:
        warnings.append(
            f"Image resolution ({width}×{height} px) is low. For higher diagnostic accuracy, "
            f"capture images at 224×224 px or higher."
        )

    if width > MAX_DIMENSION_PX or height > MAX_DIMENSION_PX:
        failure_reasons.append(
            f"Image dimensions ({width}×{height} px) exceed maximum allowed "
            f"({MAX_DIMENSION_PX}×{MAX_DIMENSION_PX} px)."
        )
        dimension_failure = True

    if dimension_failure:
        feedback_lines = ["Photograph dimensions are outside acceptable bounds:"]
        feedback_lines.extend([f"• {r}" for r in failure_reasons])
        return QualityReport(
            passed=False,
            blur_score=None,
            brightness_mean=None,
            width=width,
            height=height,
            failure_reasons=failure_reasons,
            warnings=warnings,
            status=QualityStatus.FAILED,
            feedback="\n".join(feedback_lines),
            can_proceed=False,
            is_placeholder=False,
        )

    # Convert to NumPy uint8 RGB array for OpenCV analysis
    rgb_arr = np.array(pil_img, dtype=np.uint8)

    # 3. Blur Detection
    blur_score = compute_blur_score(rgb_arr)
    severe_blur_threshold = max(20.0, blur_thresh * 0.4)
    if blur_score < severe_blur_threshold:
        failure_reasons.append(
            f"Image is significantly blurry (sharpness score: {blur_score:.1f}, "
            f"threshold: {severe_blur_threshold:.1f}). Please steady the camera and refocus on the leaf."
        )
    elif blur_score < blur_thresh:
        warnings.append(
            f"Image has moderate blur (sharpness score: {blur_score:.1f}, "
            f"recommended: ≥ {blur_thresh:.1f}). Fine disease lesions might be indistinct."
        )

    # 4. Brightness and Exposure Detection
    brightness_mean = compute_brightness_mean(rgb_arr)
    if brightness_mean < b_min:
        failure_reasons.append(
            f"Image is underexposed/too dark (mean brightness: {brightness_mean:.1f}/255, "
            f"minimum required: {b_min:.1f}). Please photograph with adequate lighting."
        )
    elif brightness_mean < b_min + 20:
        warnings.append(
            f"Image lighting is dim (mean brightness: {brightness_mean:.1f}/255). "
            f"Better illumination is recommended."
        )

    if brightness_mean > b_max:
        failure_reasons.append(
            f"Image is overexposed/too bright (mean brightness: {brightness_mean:.1f}/255, "
            f"maximum allowed: {b_max:.1f}). Avoid harsh direct flash or intense glare."
        )
    elif brightness_mean > b_max - 20:
        warnings.append(
            f"Image has bright exposure/glare (mean brightness: {brightness_mean:.1f}/255). "
            f"Leaf color contrast may be washed out."
        )

    # Determine overall status and synthesize feedback
    passed = len(failure_reasons) == 0
    can_proceed = passed

    if not passed:
        status = QualityStatus.FAILED
        feedback_lines = ["Photograph quality is insufficient for reliable AI analysis:"]
        feedback_lines.extend([f"• {reason}" for reason in failure_reasons])
        feedback_lines.append("Recommendation: Please retake the photograph following the guidelines.")
        feedback = "\n".join(feedback_lines)
    elif len(warnings) > 0:
        status = QualityStatus.WARNING
        feedback_lines = ["Image quality is acceptable to proceed, but note:"]
        feedback_lines.extend([f"• {warn}" for warn in warnings])
        feedback = "\n".join(feedback_lines)
    else:
        status = QualityStatus.PASSED
        feedback = (
            f"Image quality check passed. Sharpness: {blur_score:.1f} (good), "
            f"Exposure: {brightness_mean:.1f}/255 (balanced)."
        )

    return QualityReport(
        passed=passed,
        blur_score=round(blur_score, 2),
        brightness_mean=round(brightness_mean, 2),
        width=width,
        height=height,
        failure_reasons=failure_reasons,
        warnings=warnings,
        status=status,
        feedback=feedback,
        can_proceed=can_proceed,
        is_placeholder=False,
    )
