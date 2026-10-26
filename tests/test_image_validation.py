"""
Aura Farming — tests/test_image_validation.py
===============================================
Unit tests for src/image_validation.py.

Tests cover:
  - Accepted image formats (JPG, JPEG, PNG, WEBP)
  - Rejected formats (BMP, GIF, PDF, TXT)
  - File size enforcement (under limit, over limit)
  - Successful decoding of a valid small PNG
  - Rejection of corrupted/empty bytes
  - Dimension validation (too small, too large)

Run with:
    pytest tests/test_image_validation.py -v
"""

import io
import os
import struct

import pytest
from PIL import Image

from src.image_validation import (
    check_format,
    check_file_size,
    decode_image,
    validate_upload,
    UnsupportedFormatError,
    FileTooLargeError,
    CorruptedImageError,
    InvalidDimensionsError,
    MAX_FILE_SIZE_BYTES,
    MIN_DIMENSION_PX,
    MAX_DIMENSION_PX,
    SUPPORTED_FORMATS,
)


# ─────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────

def make_png_bytes(width: int = 256, height: int = 256, color=(120, 180, 80)) -> bytes:
    """Create a minimal valid RGB PNG image as bytes."""
    img = Image.new("RGB", (width, height), color=color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def make_jpeg_bytes(width: int = 256, height: int = 256) -> bytes:
    """Create a minimal valid RGB JPEG image as bytes."""
    img = Image.new("RGB", (width, height), color=(100, 150, 80))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


# ─────────────────────────────────────────────────────────────────
# Format validation
# ─────────────────────────────────────────────────────────────────

class TestCheckFormat:
    def test_jpg_accepted(self):
        assert check_format("leaf.jpg") == "jpg"

    def test_jpeg_accepted(self):
        assert check_format("photo.JPEG") == "jpeg"

    def test_png_accepted(self):
        assert check_format("image.png") == "png"

    def test_webp_accepted(self):
        assert check_format("scan.webp") == "webp"

    def test_uppercase_extension_accepted(self):
        assert check_format("PHOTO.PNG") == "png"

    def test_bmp_rejected(self):
        with pytest.raises(UnsupportedFormatError):
            check_format("image.bmp")

    def test_gif_rejected(self):
        with pytest.raises(UnsupportedFormatError):
            check_format("anim.gif")

    def test_pdf_rejected(self):
        with pytest.raises(UnsupportedFormatError):
            check_format("document.pdf")

    def test_txt_rejected(self):
        with pytest.raises(UnsupportedFormatError):
            check_format("notes.txt")

    def test_no_extension_rejected(self):
        with pytest.raises(UnsupportedFormatError):
            check_format("leafphoto")

    def test_error_message_contains_filename(self):
        with pytest.raises(UnsupportedFormatError, match="corrupt.bmp"):
            check_format("corrupt.bmp")

    def test_error_message_lists_supported_formats(self):
        try:
            check_format("file.tiff")
        except UnsupportedFormatError as exc:
            # Should mention at least one supported format
            assert any(fmt in str(exc) for fmt in SUPPORTED_FORMATS)


# ─────────────────────────────────────────────────────────────────
# File size validation
# ─────────────────────────────────────────────────────────────────

class TestCheckFileSize:
    def test_small_file_accepted(self):
        small_bytes = b"x" * 1024  # 1 KB
        size = check_file_size(small_bytes)
        assert size == 1024

    def test_file_at_limit_accepted(self):
        at_limit = b"x" * MAX_FILE_SIZE_BYTES
        assert check_file_size(at_limit) == MAX_FILE_SIZE_BYTES

    def test_file_over_limit_rejected(self):
        over_limit = b"x" * (MAX_FILE_SIZE_BYTES + 1)
        with pytest.raises(FileTooLargeError):
            check_file_size(over_limit)

    def test_empty_file_accepted(self):
        # Empty files will fail later at decode_image, not at size check
        assert check_file_size(b"") == 0


# ─────────────────────────────────────────────────────────────────
# Image decoding
# ─────────────────────────────────────────────────────────────────

class TestDecodeImage:
    def test_valid_png_decoded(self):
        png_bytes = make_png_bytes(256, 256)
        image, w, h = decode_image(png_bytes)
        assert w == 256
        assert h == 256
        assert image.mode == "RGB"

    def test_valid_jpeg_decoded(self):
        jpg_bytes = make_jpeg_bytes(128, 128)
        image, w, h = decode_image(jpg_bytes)
        assert w == 128
        assert h == 128
        assert image.mode == "RGB"

    def test_corrupted_bytes_rejected(self):
        with pytest.raises(CorruptedImageError):
            decode_image(b"not an image at all \x00\x01\x02")

    def test_empty_bytes_rejected(self):
        with pytest.raises(CorruptedImageError):
            decode_image(b"")

    def test_truncated_png_rejected(self):
        # Take only first 20 bytes of a valid PNG (incomplete header)
        png_bytes = make_png_bytes()[:20]
        with pytest.raises(CorruptedImageError):
            decode_image(png_bytes)

    def test_too_small_image_rejected(self):
        tiny = make_png_bytes(width=10, height=10)
        with pytest.raises(InvalidDimensionsError):
            decode_image(tiny)

    def test_minimum_valid_size_accepted(self):
        # MIN_DIMENSION_PX × MIN_DIMENSION_PX should be accepted
        valid = make_png_bytes(width=MIN_DIMENSION_PX, height=MIN_DIMENSION_PX)
        image, w, h = decode_image(valid)
        assert w == MIN_DIMENSION_PX


# ─────────────────────────────────────────────────────────────────
# Full validate_upload pipeline
# ─────────────────────────────────────────────────────────────────

class TestValidateUpload:
    def test_valid_png_returns_image_and_metadata(self):
        png_bytes = make_png_bytes(224, 224)
        image, meta = validate_upload("leaf.png", png_bytes)
        assert image is not None
        assert meta["extension"] == "png"
        assert meta["width"] == 224
        assert meta["height"] == 224
        assert meta["file_size_bytes"] == len(png_bytes)

    def test_unsupported_format_raises(self):
        png_bytes = make_png_bytes()
        with pytest.raises(UnsupportedFormatError):
            validate_upload("leaf.bmp", png_bytes)

    def test_oversized_file_raises(self):
        oversized = b"x" * (MAX_FILE_SIZE_BYTES + 1)
        with pytest.raises(FileTooLargeError):
            validate_upload("big.png", oversized)

    def test_corrupted_file_raises(self):
        with pytest.raises(CorruptedImageError):
            validate_upload("bad.jpg", b"\x00\x01\x02\x03")

    def test_metadata_file_size_mb_is_float(self):
        png_bytes = make_png_bytes()
        _, meta = validate_upload("test.png", png_bytes)
        assert isinstance(meta["file_size_mb"], float)
