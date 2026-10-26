"""
Aura Farming — training/inspect_dataset.py
============================================
Dataset inspection script for PlantVillage (tomato classes).

Run this script BEFORE prepare_dataset.py to verify the downloaded
dataset matches the expected class structure.

Usage (from project root, with venv active):
    python training/inspect_dataset.py
    python training/inspect_dataset.py --dataset-dir data/plantvillage/color
    python training/inspect_dataset.py --variant grayscale
    python training/inspect_dataset.py --output data/reports/inspection_report.json

What this script does:
  1. Locates the dataset directory via config.yaml or CLI argument.
  2. Discovers all class subdirectories.
  3. For each class directory: counts images, records formats and
     dimensions (sampled), and flags unreadable files.
  4. Identifies which tomato classes from class_mapping.json are
     present, missing, or have unexpected names.
  5. Reports class imbalance statistics.
  6. Writes a machine-readable JSON report and a human-readable
     summary to stdout.

This script reads actual files. It never fabricates counts.
If the dataset is not present, it exits with a clear error message.

Phase 3 status: COMPLETE
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional

# ── Ensure project root is on the Python path ────────────────────
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import yaml
from PIL import Image, UnidentifiedImageError


# ─────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}

# Maximum number of images to open per class for dimension sampling
# (opening every image is slow; a sample is sufficient for inspection)
DIMENSION_SAMPLE_SIZE = 50


# ─────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────

def load_config() -> dict:
    config_path = PROJECT_ROOT / "configs" / "config.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)


def load_class_mapping() -> dict:
    mapping_path = PROJECT_ROOT / "configs" / "class_mapping.json"
    with open(mapping_path) as f:
        return json.load(f)


def _is_image_file(path: Path) -> bool:
    return path.suffix.lower() in SUPPORTED_EXTENSIONS


def _try_open_image(path: Path) -> Optional[tuple[int, int]]:
    """
    Try to open and verify an image file.
    Returns (width, height) on success, None on failure.
    """
    try:
        with Image.open(path) as img:
            img.verify()
        with Image.open(path) as img:
            return img.size  # (width, height)
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────
# Core inspection logic
# ─────────────────────────────────────────────────────────────────

def inspect_class_directory(class_dir: Path) -> dict:
    """
    Inspect a single class directory and return statistics.

    Args:
        class_dir: Path to the class folder.

    Returns:
        dict with keys:
          class_name, total_files, image_files, unreadable_files,
          non_image_files, formats, sampled_dimensions,
          unreadable_paths
    """
    all_files = list(class_dir.iterdir())
    image_files = [f for f in all_files if f.is_file() and _is_image_file(f)]
    non_image = [f.name for f in all_files if f.is_file() and not _is_image_file(f)]

    # Count formats
    formats: dict[str, int] = {}
    for f in image_files:
        ext = f.suffix.lower()
        formats[ext] = formats.get(ext, 0) + 1

    # Sample dimensions and detect unreadable files
    unreadable: list[str] = []
    dimensions: list[tuple[int, int]] = []
    sample = image_files[:DIMENSION_SAMPLE_SIZE]

    for f in sample:
        result = _try_open_image(f)
        if result is None:
            unreadable.append(f.name)
        else:
            dimensions.append(result)

    # Check remaining files for readability (no dimension collection)
    for f in image_files[DIMENSION_SAMPLE_SIZE:]:
        result = _try_open_image(f)
        if result is None:
            unreadable.append(f.name)

    # Summarise dimensions
    dim_summary = {}
    if dimensions:
        widths = [d[0] for d in dimensions]
        heights = [d[1] for d in dimensions]
        unique_dims = set(dimensions)
        dim_summary = {
            "sample_size": len(dimensions),
            "unique_sizes": len(unique_dims),
            "min_width": min(widths),
            "max_width": max(widths),
            "min_height": min(heights),
            "max_height": max(heights),
            "most_common": max(set(dimensions), key=dimensions.count),
        }

    return {
        "class_name": class_dir.name,
        "total_files": len(all_files),
        "image_files": len(image_files),
        "unreadable_files": len(unreadable),
        "non_image_files": len(non_image),
        "formats": formats,
        "dimension_summary": dim_summary,
        "unreadable_paths": unreadable[:20],  # cap to 20 for report
    }


def inspect_dataset(dataset_dir: Path, variant: str = "color") -> dict:
    """
    Inspect the full PlantVillage dataset directory.

    Expected structure:
        dataset_dir/
            color/           (or grayscale/ or segmented/)
                Tomato___Bacterial_spot/
                    img1.JPG
                    ...
                Tomato___Early_blight/
                ...

    Args:
        dataset_dir: Root directory of the PlantVillage download.
        variant: "color", "grayscale", or "segmented".

    Returns:
        Full inspection report as a dict.
    """
    # Resolve the class root — support both flat and variant-subdirectory layouts
    variant_dir = dataset_dir / variant
    if variant_dir.is_dir():
        class_root = variant_dir
    elif dataset_dir.is_dir():
        # Flat layout: classes are directly inside dataset_dir
        class_root = dataset_dir
    else:
        return {
            "error": f"Dataset directory not found: {dataset_dir}",
            "suggestion": (
                "Download PlantVillage and place the tomato class folders at "
                f"'{dataset_dir}' or '{variant_dir}'. "
                "See models/README.md for download instructions."
            ),
        }

    class_dirs = sorted([d for d in class_root.iterdir() if d.is_dir()])
    if not class_dirs:
        return {
            "error": f"No class subdirectories found in: {class_root}",
            "suggestion": "Verify the dataset was extracted correctly.",
        }

    # Load expected tomato classes from mapping
    mapping = load_class_mapping()
    expected_folders = set(mapping["folder_to_index"].keys())
    discovered_folders = {d.name for d in class_dirs}

    tomato_dirs = [d for d in class_dirs if d.name.startswith("Tomato")]
    non_tomato_dirs = [d for d in class_dirs if not d.name.startswith("Tomato")]

    # Cross-check discovered vs expected
    found_expected = expected_folders & discovered_folders
    missing_expected = expected_folders - discovered_folders
    unexpected_tomato = {n for n in discovered_folders
                         if n.startswith("Tomato") and n not in expected_folders}

    print(f"\n{'='*60}")
    print(f"  Aura Farming — Dataset Inspection")
    print(f"  Path   : {class_root}")
    print(f"  Variant: {variant}")
    print(f"{'='*60}")
    print(f"\n  Total class directories found : {len(class_dirs)}")
    print(f"  Tomato class directories      : {len(tomato_dirs)}")
    print(f"  Non-tomato class directories  : {len(non_tomato_dirs)}")
    print(f"  Expected tomato classes found : {len(found_expected)}/10")

    if missing_expected:
        print(f"\n  ⚠  MISSING expected classes:")
        for m in sorted(missing_expected):
            print(f"     - {m}")
    if unexpected_tomato:
        print(f"\n  ⚠  UNEXPECTED tomato folders (not in class_mapping.json):")
        for u in sorted(unexpected_tomato):
            print(f"     - {u}")

    # Inspect each tomato class
    print(f"\n{'─'*60}")
    print(f"  {'Class':<50} {'Images':>7}  {'Unreadable':>10}")
    print(f"{'─'*60}")

    class_reports = []
    total_images = 0
    total_unreadable = 0

    for d in tomato_dirs:
        report = inspect_class_directory(d)
        class_reports.append(report)
        total_images += report["image_files"]
        total_unreadable += report["unreadable_files"]
        status = "⚠" if report["unreadable_files"] > 0 else " "
        print(
            f"  {status} {report['class_name']:<48} "
            f"{report['image_files']:>7}  "
            f"{report['unreadable_files']:>10}"
        )

    print(f"{'─'*60}")
    print(f"  {'TOTAL (tomato classes)':<50} {total_images:>7}  {total_unreadable:>10}")
    print(f"{'='*60}\n")

    # Class imbalance report
    if class_reports:
        counts = {r["class_name"]: r["image_files"] for r in class_reports}
        min_class = min(counts, key=counts.get)
        max_class = max(counts, key=counts.get)
        imbalance_ratio = counts[max_class] / max(counts[min_class], 1)
        print(f"  Class imbalance:")
        print(f"    Smallest class : {min_class} ({counts[min_class]} images)")
        print(f"    Largest class  : {max_class} ({counts[max_class]} images)")
        print(f"    Imbalance ratio: {imbalance_ratio:.1f}x\n")

        if imbalance_ratio > 5:
            print(
                "  ⚠  WARNING: Imbalance ratio > 5x. Class weighting or "
                "oversampling should be considered during Phase 4 training.\n"
            )

    full_report = {
        "dataset_dir": str(dataset_dir),
        "variant": variant,
        "class_root": str(class_root),
        "total_class_dirs": len(class_dirs),
        "tomato_class_dirs": len(tomato_dirs),
        "non_tomato_class_dirs": len(non_tomato_dirs),
        "expected_classes_found": sorted(found_expected),
        "missing_expected_classes": sorted(missing_expected),
        "unexpected_tomato_folders": sorted(unexpected_tomato),
        "total_tomato_images": total_images,
        "total_unreadable": total_unreadable,
        "class_reports": class_reports,
    }

    return full_report


# ─────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Aura Farming — Dataset Inspection Script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use default path from config.yaml
  python training/inspect_dataset.py

  # Specify dataset directory explicitly
  python training/inspect_dataset.py --dataset-dir data/plantvillage

  # Use a different variant
  python training/inspect_dataset.py --variant segmented

  # Save JSON report
  python training/inspect_dataset.py --output data/reports/inspection.json
        """,
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=None,
        help="Path to the PlantVillage dataset directory (overrides config.yaml)",
    )
    parser.add_argument(
        "--variant",
        choices=["color", "grayscale", "segmented"],
        default=None,
        help="Dataset variant to inspect (default from config.yaml: color)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path to save the JSON report (e.g. data/reports/inspection.json)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config()

    # Resolve dataset directory
    dataset_dir = args.dataset_dir
    if dataset_dir is None:
        dataset_dir = PROJECT_ROOT / cfg["dataset"]["plantvillage_dir"]

    # Resolve variant
    variant = args.variant or cfg["dataset"].get("plantvillage_variant", "color")

    print(f"\n[Aura Farming] inspect_dataset.py — Phase 3")
    print(f"[Aura Farming] Dataset path : {dataset_dir}")
    print(f"[Aura Farming] Variant      : {variant}")

    if not dataset_dir.exists():
        print(
            f"\n❌  ERROR: Dataset directory does not exist: {dataset_dir}\n"
            f"\n   To download PlantVillage:\n"
            f"\n   Option 1 — Hugging Face (recommended):\n"
            f"     pip install datasets\n"
            f"     python -c \"\n"
            f"       from datasets import load_dataset\n"
            f"       ds = load_dataset('mohanty/PlantVillage', 'color')\n"
            f"     \"\n"
            f"\n   Option 2 — Git clone (~2 GB):\n"
            f"     git clone https://github.com/spMohanty/PlantVillage-Dataset\n"
            f"     mv PlantVillage-Dataset/raw/color data/plantvillage/color\n"
            f"\n   Then re-run: python training/inspect_dataset.py\n"
        )
        sys.exit(1)

    report = inspect_dataset(dataset_dir, variant)

    if "error" in report:
        print(f"\n❌  ERROR: {report['error']}")
        print(f"   Suggestion: {report.get('suggestion', '')}\n")
        sys.exit(1)

    # Save JSON report if requested
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        # Convert Path objects and tuples for JSON serialisation
        def _serialisable(obj):
            if isinstance(obj, Path):
                return str(obj)
            if isinstance(obj, tuple):
                return list(obj)
            raise TypeError(f"Not serialisable: {type(obj)}")

        with open(args.output, "w") as f:
            json.dump(report, f, indent=2, default=_serialisable)
        print(f"[Aura Farming] JSON report saved to: {args.output}\n")

    print("[Aura Farming] Inspection complete.\n")
    return report


if __name__ == "__main__":
    main()
