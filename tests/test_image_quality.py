"""
Aura Farming — tests/test_image_quality.py
============================================
Unit tests for src/image_quality.py.

Phase 2: Tests confirm the placeholder always returns a passing
result and that the result carries the correct dimensions.

Phase 5: These tests will be extended to cover real blur and
brightness detection with synthetic test images.

Run with:
    pytest tests/test_image_quality.py -v
"""

import io

import pytest
from PIL import Image

from src.image_quality import assess_quality, QualityReport


def make_rgb_image(width: int = 224, height: int = 224,
                   color=(120, 160, 80)) -> Image.Image:
    return Image.new("RGB", (width, height), color=color)


class TestAssessQuality:
    def test_returns_quality_report(self):
        img = make_rgb_image()
        result = assess_quality(img)
        assert isinstance(result, QualityReport)

    def test_placeholder_always_passes(self):
        img = make_rgb_image()
        result = assess_quality(img)
        assert result.passed is True

    def test_placeholder_flag_is_set(self):
        img = make_rgb_image()
        result = assess_quality(img)
        assert result.is_placeholder is True

    def test_dimensions_recorded(self):
        img = make_rgb_image(320, 240)
        result = assess_quality(img)
        assert result.width == 320
        assert result.height == 240

    def test_no_failure_reasons_in_placeholder(self):
        img = make_rgb_image()
        result = assess_quality(img)
        assert result.failure_reasons == []

    def test_blur_score_is_none_in_placeholder(self):
        img = make_rgb_image()
        result = assess_quality(img)
        assert result.blur_score is None

    def test_brightness_is_none_in_placeholder(self):
        img = make_rgb_image()
        result = assess_quality(img)
        assert result.brightness_mean is None
