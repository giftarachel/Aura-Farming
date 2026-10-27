"""
Aura Farming — tests/test_image_quality.py
============================================
Unit tests for src/image_quality.py (Phase 5).

Covers:
  - Valid RGB image assessment
  - Grayscale ('L') and RGBA image modes
  - Corrupted and invalid inputs
  - Dimension bounds (too small < 64px, too large > 8000px)
  - Blurry images vs. sharp structured images
  - Underexposed (dark) and overexposed (bright) images
  - Warnings vs. hard rejections
  - Threshold parameter overrides
  - QualityReport structure and QualityStatus enum

Run with:
    pytest tests/test_image_quality.py -v
"""

from __future__ import annotations

import io
import numpy as np
import pytest
from PIL import Image, ImageDraw, ImageFilter

from src.image_quality import (
    assess_quality,
    compute_blur_score,
    compute_brightness_mean,
    QualityReport,
    QualityStatus,
)


def make_sharp_rgb_image(width: int = 224, height: int = 224) -> Image.Image:
    """Create a synthetic high-contrast textured image with distinct edges."""
    img = Image.new("RGB", (width, height), color=(128, 128, 128))
    draw = ImageDraw.Draw(img)
    for i in range(0, width, 16):
        draw.line([(i, 0), (i, height)], fill=(255, 255, 255), width=2)
    for j in range(0, height, 16):
        draw.line([(0, j), (width, j)], fill=(0, 0, 0), width=2)
    return img


def make_uniform_rgb_image(width: int = 224, height: int = 224,
                           color=(120, 160, 80)) -> Image.Image:
    """Create a uniform solid-color image (low edge variance)."""
    return Image.new("RGB", (width, height), color=color)


class TestImageQualityCore:
    def test_returns_quality_report(self):
        img = make_sharp_rgb_image()
        result = assess_quality(img)
        assert isinstance(result, QualityReport)
        assert result.is_placeholder is False

    def test_dimensions_recorded(self):
        img = make_sharp_rgb_image(320, 240)
        result = assess_quality(img)
        assert result.width == 320
        assert result.height == 240

    def test_sharp_image_passes(self):
        img = make_sharp_rgb_image(224, 224)
        result = assess_quality(img)
        assert result.passed is True
        assert result.can_proceed is True
        assert result.status in (QualityStatus.PASSED, QualityStatus.WARNING)
        assert result.blur_score is not None
        assert result.blur_score > 100.0

    def test_grayscale_image_handled(self):
        img_gray = Image.new("L", (224, 224), color=128)
        result = assess_quality(img_gray)
        assert isinstance(result, QualityReport)
        assert result.width == 224
        assert result.height == 224

    def test_rgba_image_handled(self):
        img_rgba = Image.new("RGBA", (224, 224), color=(100, 150, 80, 255))
        result = assess_quality(img_rgba)
        assert isinstance(result, QualityReport)
        assert result.width == 224

    def test_raw_bytes_input_handled(self):
        img = make_sharp_rgb_image()
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        raw_bytes = buf.getvalue()
        result = assess_quality(raw_bytes)
        assert isinstance(result, QualityReport)
        assert result.passed is True
        assert result.width == 224

    def test_corrupted_bytes_returns_failed_report(self):
        corrupted = b"NOT_AN_IMAGE_RANDOM_GARBAGE_BYTES_12345"
        result = assess_quality(corrupted)
        assert result.passed is False
        assert result.can_proceed is False
        assert result.status == QualityStatus.FAILED
        assert len(result.failure_reasons) > 0


class TestImageDimensionsGate:
    def test_too_small_image_rejected(self):
        img_tiny = Image.new("RGB", (48, 48), color=(100, 150, 80))
        result = assess_quality(img_tiny)
        assert result.passed is False
        assert result.can_proceed is False
        assert any("smaller than the minimum" in r for r in result.failure_reasons)

    def test_too_large_image_rejected(self):
        # Create image with width > MAX_DIMENSION_PX (8000px)
        img = Image.new("RGB", (64, 64), color=(100, 100, 100))
        img._size = (8001, 200)
        result = assess_quality(img)
        assert result.passed is False
        assert any("exceed maximum allowed" in r for r in result.failure_reasons)

    def test_low_resolution_warning(self):
        img = Image.new("RGB", (96, 96), color=(120, 150, 80))
        result = assess_quality(img)
        assert any("resolution" in w.lower() for w in result.warnings)


class TestBlurDetection:
    def test_heavy_blur_fails(self):
        sharp = make_sharp_rgb_image(224, 224)
        blurry = sharp.filter(ImageFilter.GaussianBlur(radius=15))
        result = assess_quality(blurry)
        assert result.blur_score is not None
        assert result.blur_score < 40.0
        assert result.passed is False
        assert any("blurry" in r.lower() for r in result.failure_reasons)

    def test_compute_blur_score_empty_array(self):
        arr = np.zeros((0, 0, 3), dtype=np.uint8)
        assert compute_blur_score(arr) == 0.0

    def test_blur_threshold_override(self):
        img = make_sharp_rgb_image()
        result = assess_quality(img, blur_threshold=999999.0)
        assert any("blur" in w.lower() or "blur" in r.lower()
                   for item in (result.warnings + result.failure_reasons)
                   for w in [item])


class TestBrightnessAndExposure:
    def test_excessively_dark_image_fails(self):
        dark_img = Image.new("RGB", (224, 224), color=(5, 5, 5))
        result = assess_quality(dark_img)
        assert result.passed is False
        assert result.brightness_mean < 30
        assert any("dark" in r.lower() or "underexposed" in r.lower()
                   for r in result.failure_reasons)

    def test_excessively_bright_image_fails(self):
        bright_img = Image.new("RGB", (224, 224), color=(250, 250, 250))
        result = assess_quality(bright_img)
        assert result.passed is False
        assert result.brightness_mean > 225
        assert any("bright" in r.lower() or "overexposed" in r.lower()
                   for r in result.failure_reasons)

    def test_balanced_brightness_passes(self):
        img = make_sharp_rgb_image()
        result = assess_quality(img)
        assert 30 <= result.brightness_mean <= 225

    def test_compute_brightness_mean_empty_array(self):
        arr = np.zeros((0, 0, 3), dtype=np.uint8)
        assert compute_brightness_mean(arr) == 0.0

    def test_brightness_threshold_overrides(self):
        img = Image.new("RGB", (224, 224), color=(100, 100, 100))
        result = assess_quality(img, brightness_min=150.0)
        assert result.passed is False
        assert any("dark" in r.lower() for r in result.failure_reasons)
