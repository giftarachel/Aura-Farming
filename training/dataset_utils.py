"""
Aura Farming — training/dataset_utils.py
==========================================
Shared dataset utilities used by inspect_dataset.py,
duplicate_detection.py, prepare_dataset.py, and tests.

Phase 3 status: COMPLETE
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Optional

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def is_image_file(path: Path) -> bool:
    """Return True if the file has a recognised image extension."""
    return path.suffix.lower() in IMAGE_EXTENSIONS


def collect_all_images(directory: Path) -> list[Path]:
    """
    Recursively collect all image files under a directory.

    Args:
        directory: Root path to search.

    Returns:
        Sorted list of image Paths.
    """
    paths = []
    for root, _, files in __import__("os").walk(directory):
        for fname in files:
            p = Path(root) / fname
            if is_image_file(p):
                paths.append(p)
    return sorted(paths)


def load_manifest(manifest_path: Path) -> list[dict]:
    """
    Load a split manifest CSV into a list of record dicts.

    Args:
        manifest_path: Path to a CSV manifest file.

    Returns:
        List of dicts with keys: path, class_index, class_folder, group_id.
    """
    records = []
    with open(manifest_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            records.append({
                "path": row["path"],
                "class_index": int(row["class_index"]),
                "class_folder": row["class_folder"],
                "group_id": row["group_id"] or None,
            })
    return records


def verify_manifests_no_overlap(
    *manifest_paths: Path,
) -> tuple[bool, list[str]]:
    """
    Verify that no file path appears in more than one manifest.

    Args:
        manifest_paths: Any number of manifest CSV paths.

    Returns:
        Tuple (passed: bool, violations: list of overlapping paths).
    """
    seen: dict[str, str] = {}  # path → manifest name
    violations: list[str] = []

    for mp in manifest_paths:
        records = load_manifest(mp)
        for r in records:
            p = r["path"]
            if p in seen:
                violations.append(f"{p}  [in {seen[p]} AND {mp.name}]")
            else:
                seen[p] = mp.name

    return len(violations) == 0, violations


def summarise_manifest(manifest_path: Path) -> dict:
    """
    Compute per-class counts and total from a manifest CSV.

    Returns:
        dict with keys: total, per_class (class_index → count).
    """
    records = load_manifest(manifest_path)
    per_class: dict[int, int] = {}
    for r in records:
        ci = r["class_index"]
        per_class[ci] = per_class.get(ci, 0) + 1
    return {
        "total": len(records),
        "per_class": per_class,
    }
