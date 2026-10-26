"""
Aura Farming — src/image_quality.py
======================================
Image quality assessment gate.

Phase 2 status: PLACEHOLDER
─────────────────────────────
This module will be fully implemented in Phase 5.

The public function `assess_quality` is defined here as a stub
that always returns a passing result, so the application can be
run and tested without blocking on Phase 5 features.

Planned functionality (Phase 5):
- Blur detection via Laplacian variance (OpenCV)
- Brightness check (mean pixel intensity in HSV space)
- Dimension verification (duplicated from image_validation for
  belt-and-suspenders safety)
- A composite quality report with pass/fail per dimension
- Threshold values loaded from configs/config.yaml
- Threshold selection methodology documented in the project report

IMPORTANT: This module deliberately does NOT claim to detect
whether a leaf is present. It performs only basic photographic
quality checks. Leaf detection would require a separate model.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from PIL import Image


# ─────────────────────────────────────────────────────────────────
# Result dataclass
# ─────────────────────────────────────────────────────────────────

@dataclass
class QualityReport:
    """
    Holds the result of an image quality assessment.

    Attributes:
        passed: True if all checks passed; False if any check failed.
        blur_score: Laplacian variance score (higher = sharper).
                    None until Phase 5 is implemented.
        brightness_mean: Mean pixel brightness (0–255).
                         None until Phase 5 is implemented.
        width: Image width in pixels.
        height: Image height in pixels.
        failure_reasons: List of human-readable failure messages.
                         Empty if all checks passed.
        is_placeholder: True while Phase 5 is not yet implemented.
    """
    passed: bool = True
    blur_score: Optional[float] = None
    brightness_mean: Optional[float] = None
    width: int = 0
    height: int = 0
    failure_reasons: list[str] = field(default_factory=list)
    is_placeholder: bool = True


# ─────────────────────────────────────────────────────────────────
# Public interface
# ─────────────────────────────────────────────────────────────────

def assess_quality(image: Image.Image) -> QualityReport:
    """
    Assess the photographic quality of a validated PIL image.

    Phase 2: Returns a placeholder QualityReport that always passes.
    Phase 5: Will compute blur score, brightness, and return a
             genuine assessment against configured thresholds.

    Args:
        image: A PIL Image in RGB mode, already decoded and
               validated by src/image_validation.py.

    Returns:
        A QualityReport describing the assessment outcome.
    """
    width, height = image.size
    # Placeholder: always pass. Phase 5 will replace this body.
    return QualityReport(
        passed=True,
        blur_score=None,
        brightness_mean=None,
        width=width,
        height=height,
        failure_reasons=[],
        is_placeholder=True,
    )
