"""
Aura Farming — src/image_validation.py
=========================================
First-pass validation of uploaded image files.

Responsibilities:
- Check that the file extension is in the supported list.
- Check that the file size is within the configured maximum.
- Attempt to decode the file as an image (detect corruption).
- Return image dimensions on success.
- Raise descriptive exceptions on failure so the UI layer can
  display a helpful message without exposing a raw Python traceback.

This module does NOT assess image quality (blur, brightness, etc.).
That is handled separately in src/image_quality.py (Phase 5).

Phase 2 status: COMPLETE — full file-level validation implemented.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Tuple

import yaml
from PIL import Image, UnidentifiedImageError

# ─────────────────────────────────────────────────────────────────
# Load configuration
# ─────────────────────────────────────────────────────────────────

def _load_config() -> dict:
    config_path = Path(__file__).parent.parent / "configs" / "config.yaml"
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


_CONFIG = _load_config()
_UPLOAD_CFG = _CONFIG["upload"]

SUPPORTED_FORMATS: list[str] = [fmt.lower() for fmt in _UPLOAD_CFG["supported_formats"]]
MAX_FILE_SIZE_BYTES: int = int(_UPLOAD_CFG["max_file_size_mb"] * 1024 * 1024)
MIN_DIMENSION_PX: int = _UPLOAD_CFG["min_dimension_px"]
MAX_DIMENSION_PX: int = _UPLOAD_CFG["max_dimension_px"]


# ─────────────────────────────────────────────────────────────────
# Custom exceptions
# ─────────────────────────────────────────────────────────────────

class ValidationError(Exception):
    """Raised when an uploaded file fails any validation check."""


class UnsupportedFormatError(ValidationError):
    """The file extension is not in the supported list."""


class FileTooLargeError(ValidationError):
    """The file exceeds the configured size limit."""


class CorruptedImageError(ValidationError):
    """The file cannot be decoded as a valid image."""


class InvalidDimensionsError(ValidationError):
    """The image dimensions are outside acceptable bounds."""


# ─────────────────────────────────────────────────────────────────
# Validation functions
# ─────────────────────────────────────────────────────────────────

def check_format(filename: str) -> str:
    """
    Validate the file extension against the supported formats list.

    Args:
        filename: The original filename provided by the uploader.

    Returns:
        The normalised lowercase extension (without the leading dot).

    Raises:
        UnsupportedFormatError: If the extension is not supported.
    """
    suffix = Path(filename).suffix.lstrip(".").lower()
    if suffix not in SUPPORTED_FORMATS:
        supported = ", ".join(f".{f}" for f in SUPPORTED_FORMATS)
        raise UnsupportedFormatError(
            f"The file '{filename}' has an unsupported format (.{suffix}). "
            f"Please upload a {supported} image."
        )
    return suffix


def check_file_size(file_bytes: bytes) -> int:
    """
    Validate that the file does not exceed the configured size limit.

    Args:
        file_bytes: Raw bytes of the uploaded file.

    Returns:
        File size in bytes.

    Raises:
        FileTooLargeError: If the file is larger than the limit.
    """
    size = len(file_bytes)
    max_mb = _UPLOAD_CFG["max_file_size_mb"]
    if size > MAX_FILE_SIZE_BYTES:
        size_mb = size / (1024 * 1024)
        raise FileTooLargeError(
            f"This file is {size_mb:.1f} MB, which exceeds the {max_mb} MB limit. "
            f"Please use a smaller image."
        )
    return size


def decode_image(file_bytes: bytes) -> Tuple[Image.Image, int, int]:
    """
    Attempt to decode the raw bytes as a PIL image.

    This catches corrupted files before any processing is attempted.
    The image is converted to RGB to ensure consistent colour mode
    regardless of the source format (RGBA, grayscale, palette, etc.).

    Args:
        file_bytes: Raw bytes of the uploaded file.

    Returns:
        Tuple of (PIL.Image in RGB mode, width in px, height in px).

    Raises:
        CorruptedImageError: If the bytes cannot be decoded as an image.
        InvalidDimensionsError: If decoded dimensions are out of range.
    """
    try:
        image = Image.open(io.BytesIO(file_bytes))
        image.verify()  # Verify structural integrity (seeks to end of file)
        # Re-open after verify() because verify() closes the internal stream
        image = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    except (UnidentifiedImageError, Exception) as exc:
        raise CorruptedImageError(
            "This file could not be read as an image. It may be damaged or "
            "corrupted. Please try a different photograph."
        ) from exc

    width, height = image.size

    if width < MIN_DIMENSION_PX or height < MIN_DIMENSION_PX:
        raise InvalidDimensionsError(
            f"This image is too small ({width}×{height} px). "
            f"Please upload an image that is at least "
            f"{MIN_DIMENSION_PX}×{MIN_DIMENSION_PX} pixels."
        )

    if width > MAX_DIMENSION_PX or height > MAX_DIMENSION_PX:
        raise InvalidDimensionsError(
            f"This image is very large ({width}×{height} px). "
            f"Please resize it to a maximum of "
            f"{MAX_DIMENSION_PX}×{MAX_DIMENSION_PX} pixels before uploading."
        )

    return image, width, height


def validate_upload(filename: str, file_bytes: bytes) -> Tuple[Image.Image, dict]:
    """
    Run all first-pass validation checks on an uploaded file.

    This is the main entry point for the UI layer. It runs format,
    size, and decode checks in sequence. The first failure raises
    an exception; all checks are not run after a failure.

    Args:
        filename: Original filename from the file uploader.
        file_bytes: Raw bytes of the uploaded file.

    Returns:
        Tuple of (decoded PIL.Image in RGB mode, metadata dict).
        Metadata keys: filename, extension, file_size_bytes,
                       file_size_mb, width, height.

    Raises:
        UnsupportedFormatError, FileTooLargeError,
        CorruptedImageError, InvalidDimensionsError.
    """
    ext = check_format(filename)
    size_bytes = check_file_size(file_bytes)
    image, width, height = decode_image(file_bytes)

    metadata = {
        "filename": filename,
        "extension": ext,
        "file_size_bytes": size_bytes,
        "file_size_mb": round(size_bytes / (1024 * 1024), 2),
        "width": width,
        "height": height,
    }
    return image, metadata
