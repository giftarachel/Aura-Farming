"""
Aura Farming — training/colab_utils.py
========================================
Utilities for running Aura Farming training on Google Colab.

The split manifests (train_manifest.csv, val_manifest.csv,
test_manifest.csv) were generated on a Mac and contain absolute
paths like:
    /Users/giftarachel/Aura Farming/data/plantvillage/color/Tomato___...

On Colab, the project will be at a different root, e.g.:
    /content/drive/MyDrive/AuraFarming

This module provides `remap_manifest_paths`, which reads a manifest
CSV and rewrites all absolute image paths to point at the Colab
dataset directory, while preserving class labels, group IDs,
and split assignments exactly.

IMPORTANT: These functions NEVER write new manifest files or change
the original CSVs on your Mac. They return Python lists/dicts in
memory. The original manifests are read-only.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Optional


# ─────────────────────────────────────────────────────────────────
# Path remapping
# ─────────────────────────────────────────────────────────────────

def remap_manifest_paths(
    manifest_path: Path,
    colab_dataset_root: Path,
) -> tuple[list[str], list[int]]:
    """
    Load a manifest CSV and rewrite image paths to the Colab dataset root.

    The manifest contains paths like:
        /Users/giftarachel/Aura Farming/data/plantvillage/color/Tomato___X/img.JPG

    This function extracts the portion of each path starting from the
    class folder name (e.g. "Tomato___Bacterial_spot/img.JPG") and
    joins it to colab_dataset_root.

    Example:
        colab_dataset_root = Path("/content/drive/MyDrive/AuraFarming/data/plantvillage/color")
        Input path : /Users/giftarachel/Aura Farming/data/plantvillage/color/Tomato___X/img.JPG
        Output path: /content/drive/MyDrive/AuraFarming/data/plantvillage/color/Tomato___X/img.JPG

    Class labels, split assignments, and group IDs are preserved exactly.
    No files are written; the original CSVs are not modified.

    Args:
        manifest_path: Path to a manifest CSV (train, val, or test).
        colab_dataset_root: Absolute path to the dataset color/ directory
                            on Colab (where the Tomato___ folders live).

    Returns:
        Tuple of (remapped_paths, labels).
        remapped_paths: List of absolute path strings valid on Colab.
        labels: List of integer class indices (unchanged from manifest).

    Raises:
        FileNotFoundError: If manifest_path does not exist.
        ValueError: If a manifest row cannot be parsed or its class_folder
                    is empty (which would cause ambiguous path reconstruction).
    """
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Manifest not found: {manifest_path}\n"
            "Upload the data/splits/ directory to Google Drive and update the path."
        )

    remapped: list[str] = []
    labels: list[int] = []

    with open(manifest_path, newline="") as f:
        reader = csv.DictReader(f)
        for row_num, row in enumerate(reader, start=2):  # 1-indexed, row 1 is header
            class_folder = row.get("class_folder", "").strip()
            if not class_folder:
                raise ValueError(
                    f"Row {row_num} of {manifest_path.name} has an empty "
                    "class_folder — cannot reconstruct Colab path. "
                    "Re-run training/prepare_dataset.py to regenerate manifests."
                )

            # Robust filename extraction: handles Windows (\) and Mac/Linux (/)
            raw_path = str(row["path"]).replace("\\", "/")
            filename = raw_path.split("/")[-1]

            # Colab is always Linux; ensure forward slashes regardless of host OS
            colab_root_str = str(colab_dataset_root).replace("\\", "/")
            colab_path = f"{colab_root_str.rstrip('/')}/{class_folder}/{filename}"
            remapped.append(colab_path)

            try:
                labels.append(int(row["class_index"]))
            except (KeyError, ValueError) as exc:
                raise ValueError(
                    f"Row {row_num} of {manifest_path.name} has an invalid "
                    f"class_index: {row.get('class_index')!r}"
                ) from exc

    return remapped, labels


def verify_remapped_paths(
    paths: list[str],
    sample_size: int = 10,
    verbose: bool = True,
) -> tuple[bool, list[str]]:
    """
    Spot-check that remapped paths exist on disk.

    Checks the first `sample_size` paths from the list. In Colab this
    gives quick feedback before the full training pipeline starts.

    Args:
        paths: List of remapped absolute path strings.
        sample_size: How many paths to check (default 10).
        verbose: Whether to print a summary.

    Returns:
        Tuple (all_found: bool, missing: list of missing path strings).
    """
    to_check = paths[:sample_size]
    missing = [p for p in to_check if not Path(p).exists()]

    if verbose:
        checked = len(to_check)
        found = checked - len(missing)
        print(f"  Path verification: {found}/{checked} paths exist on disk.")
        if missing:
            print("  ❌ Missing paths (first 5):")
            for p in missing[:5]:
                print(f"     {p}")
        else:
            print("  ✓  Sample paths verified.")

    return len(missing) == 0, missing


# ─────────────────────────────────────────────────────────────────
# GPU / environment checks
# ─────────────────────────────────────────────────────────────────

def check_colab_environment(verbose: bool = True) -> dict:
    """
    Report the Colab runtime environment: GPU, Python, TF version.

    Safe to call before importing TensorFlow (handles ImportError).

    Args:
        verbose: Whether to print a formatted summary.

    Returns:
        Dict with keys: python_version, tf_version, gpu_available,
        gpu_name, numpy_version.
    """
    import sys
    import platform

    info: dict = {
        "python_version": sys.version,
        "platform": platform.platform(),
        "tf_version": None,
        "gpu_available": False,
        "gpu_name": None,
        "numpy_version": None,
    }

    try:
        import numpy as np
        info["numpy_version"] = np.__version__
    except ImportError:
        pass

    try:
        import tensorflow as tf
        info["tf_version"] = tf.__version__
        gpus = tf.config.list_physical_devices("GPU")
        info["gpu_available"] = len(gpus) > 0
        info["gpu_name"] = str(gpus[0]) if gpus else None
    except ImportError:
        pass

    if verbose:
        print("=" * 55)
        print("  Aura Farming — Colab Environment Check")
        print("=" * 55)
        print(f"  Python      : {sys.version.split()[0]}")
        print(f"  NumPy       : {info['numpy_version'] or 'not installed'}")
        print(f"  TensorFlow  : {info['tf_version'] or '❌ not installed'}")
        if info["gpu_available"]:
            print(f"  GPU         : ✅ {info['gpu_name']}")
        else:
            print("  GPU         : ⚠  No GPU detected (training will be slow)")
        print("=" * 55)

    return info


# ─────────────────────────────────────────────────────────────────
# Colab requirements installer
# ─────────────────────────────────────────────────────────────────

COLAB_REQUIREMENTS = [
    # PyYAML — config parsing
    "PyYAML>=6.0.1,<7.0.0",
    # Pillow — image I/O for preprocessing
    "Pillow>=10.2.0,<11.0.0",
    # scikit-learn — class weight computation + metrics
    "scikit-learn>=1.4.0,<2.0.0",
    # seaborn — confusion matrix heatmap
    "seaborn>=0.13.0,<1.0.0",
    # NOTE: TensorFlow, numpy, pandas, and matplotlib are pre-installed
    # in Google Colab and must NOT be reinstalled (doing so can break
    # the runtime or cause version conflicts).
]

# Packages that are already in Colab and must NOT be reinstalled
# (reinstalling them can break the runtime or cause version conflicts)
COLAB_PREINSTALLED = {
    "tensorflow",
    "numpy",
    "pandas",
    "matplotlib",
}


def print_colab_install_commands() -> None:
    """
    Print the pip install commands needed in Colab.

    Only installs packages that are NOT pre-installed in Colab.
    Does NOT reinstall TensorFlow or numpy.
    """
    to_install = [
        r for r in COLAB_REQUIREMENTS
        if r.split(">=")[0].lower().replace("-", "_")
        not in COLAB_PREINSTALLED
    ]
    print("# Run this cell first in your Colab notebook:")
    print("!pip install --quiet " + " ".join(f'"{r}"' for r in to_install))
