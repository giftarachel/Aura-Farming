"""
Aura Farming — training/duplicate_detection.py
================================================
Exact and near-duplicate image detection for the PlantVillage dataset.

Purpose:
  Duplicate images distributed across train, validation, and test
  splits cause data leakage — the model sees test data during
  training, inflating apparent accuracy.

  This script identifies:
    1. Exact duplicates (identical file content, via MD5 hash).
    2. Near-duplicates (visually similar images, via perceptual hash
       with configurable Hamming distance threshold).

  It NEVER silently deletes original dataset files. It only produces
  a report of duplicate groups. The split script (prepare_dataset.py)
  uses this report to keep duplicate groups in the same split.

Usage (from project root):
    python training/duplicate_detection.py
    python training/duplicate_detection.py --dataset-dir data/plantvillage/color
    python training/duplicate_detection.py --output data/reports/duplicates.json
    python training/duplicate_detection.py --exact-only

What "near-duplicate" means here:
  Two images are considered near-duplicates if their perceptual
  hash (pHash, DCT-based) has a Hamming distance ≤ threshold T.
  T=0: only exact hash matches (very strict).
  T=4: visually similar images (recommended starting point).
  T=6: more lenient — may group images that are merely similar crops.
  The threshold is read from config.yaml and can be overridden by CLI.

  Near-duplicate groups are FLAGGED for review, not automatically
  removed. Humans must decide whether flagged pairs are genuine
  duplicates or merely visually similar photographs.

Requires: imagehash, Pillow, numpy, tqdm (all in requirements.txt)

Phase 3 status: COMPLETE
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Optional

# ── Ensure project root is on the Python path ────────────────────
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import yaml

try:
    import imagehash
    from PIL import Image
    IMAGEHASH_AVAILABLE = True
except ImportError:
    IMAGEHASH_AVAILABLE = False


# ─────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────

def load_config() -> dict:
    with open(PROJECT_ROOT / "configs" / "config.yaml") as f:
        return yaml.safe_load(f)


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


# ─────────────────────────────────────────────────────────────────
# Exact duplicate detection (MD5)
# ─────────────────────────────────────────────────────────────────

def md5_hash(path: Path, chunk_size: int = 65536) -> str:
    """Compute MD5 hash of a file."""
    h = hashlib.md5()
    with open(path, "rb") as f:
        while chunk := f.read(chunk_size):
            h.update(chunk)
    return h.hexdigest()


def find_exact_duplicates(image_paths: list[Path]) -> dict[str, list[Path]]:
    """
    Find exact file duplicates by MD5 hash.

    Args:
        image_paths: List of image file paths.

    Returns:
        Dict mapping MD5 hash → list of paths with that hash.
        Only groups with 2+ paths are included (i.e., actual duplicates).
    """
    print(f"\n  [exact] Computing MD5 hashes for {len(image_paths)} images...")
    hash_to_paths: dict[str, list[Path]] = defaultdict(list)

    for i, path in enumerate(image_paths):
        if i % 1000 == 0 and i > 0:
            print(f"    ... {i}/{len(image_paths)}")
        try:
            h = md5_hash(path)
            hash_to_paths[h].append(path)
        except OSError as exc:
            print(f"    ⚠  Could not hash {path.name}: {exc}")

    duplicates = {h: paths for h, paths in hash_to_paths.items() if len(paths) > 1}
    return duplicates


# ─────────────────────────────────────────────────────────────────
# Near-duplicate detection (perceptual hash)
# ─────────────────────────────────────────────────────────────────

def phash_image(path: Path, hash_size: int = 8) -> Optional[object]:
    """
    Compute perceptual hash (DCT-based pHash) for an image.

    Returns None if the image cannot be opened.
    """
    try:
        with Image.open(path) as img:
            return imagehash.phash(img, hash_size=hash_size)
    except Exception:
        return None


def find_near_duplicates(
    image_paths: list[Path],
    hash_size: int = 8,
    threshold: int = 4,
) -> list[list[Path]]:
    """
    Find near-duplicate images using perceptual hash comparison.

    This is an O(n²) comparison in the worst case, which is slow for
    large datasets. For PlantVillage's ~18,000 tomato images it
    completes in a few minutes on a modern machine.

    Args:
        image_paths: List of image file paths.
        hash_size: pHash DCT size (8 = standard 64-bit hash).
        threshold: Maximum Hamming distance to consider near-duplicate.

    Returns:
        List of groups, where each group is a list of Paths that are
        mutually near-duplicate according to the threshold.
    """
    if not IMAGEHASH_AVAILABLE:
        print("  ⚠  imagehash not installed — skipping near-duplicate detection.")
        print("     Install with: pip install imagehash")
        return []

    print(
        f"\n  [near-dup] Computing pHashes for {len(image_paths)} images "
        f"(hash_size={hash_size})..."
    )

    hashes: list[tuple[Path, object]] = []
    failed = 0
    for i, path in enumerate(image_paths):
        if i % 1000 == 0 and i > 0:
            print(f"    ... {i}/{len(image_paths)}")
        h = phash_image(path, hash_size=hash_size)
        if h is not None:
            hashes.append((path, h))
        else:
            failed += 1

    if failed:
        print(f"  ⚠  {failed} images could not be hashed (skipped).")

    print(
        f"  [near-dup] Comparing {len(hashes)} hashes "
        f"(threshold={threshold})..."
    )

    # Union-find grouping: build adjacency from Hamming distances
    n = len(hashes)
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        parent[find(x)] = find(y)

    comparisons = 0
    for i in range(n):
        for j in range(i + 1, n):
            comparisons += 1
            dist = hashes[i][1] - hashes[j][1]  # imagehash Hamming distance
            if dist <= threshold:
                union(i, j)

    # Collect groups
    groups: dict[int, list[Path]] = defaultdict(list)
    for i, (path, _) in enumerate(hashes):
        groups[find(i)].append(path)

    near_dup_groups = [
        sorted(paths, key=lambda p: str(p))
        for paths in groups.values()
        if len(paths) > 1
    ]

    print(f"  [near-dup] Comparisons: {comparisons:,}")

    return near_dup_groups


# ─────────────────────────────────────────────────────────────────
# Group ID assignment
# ─────────────────────────────────────────────────────────────────

def assign_group_ids(
    exact_duplicates: dict[str, list[Path]],
    near_dup_groups: list[list[Path]],
    all_paths: list[Path],
) -> dict[str, str]:
    """
    Assign a stable duplicate-group ID to each image path.

    Images with a group ID must be kept in the same split.
    Images not in any duplicate group have group_id = None.

    Args:
        exact_duplicates: MD5 → list of paths.
        near_dup_groups: List of near-duplicate path groups.
        all_paths: All image paths.

    Returns:
        Dict mapping str(path) → group_id string.
        group_id is None for images with no detected duplicate.
    """
    group_map: dict[str, str] = {}

    # Exact duplicates: group_id = "exact_<md5[:8]>"
    for md5, paths in exact_duplicates.items():
        gid = f"exact_{md5[:8]}"
        for p in paths:
            group_map[str(p)] = gid

    # Near-duplicates (that aren't already exact duplicates):
    # group_id = "near_<index>"
    for idx, group in enumerate(near_dup_groups):
        gid = f"near_{idx:04d}"
        for p in group:
            key = str(p)
            if key not in group_map:
                group_map[key] = gid

    return group_map


# ─────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────

def collect_image_paths(dataset_dir: Path) -> list[Path]:
    """Recursively collect all image file paths under dataset_dir."""
    paths = []
    for root, _, files in os.walk(dataset_dir):
        for fname in files:
            p = Path(root) / fname
            if p.suffix.lower() in IMAGE_EXTENSIONS:
                paths.append(p)
    return sorted(paths)


def run_duplicate_detection(
    dataset_dir: Path,
    exact_only: bool = False,
    hash_size: int = 8,
    threshold: int = 4,
    output_path: Optional[Path] = None,
) -> dict:
    """
    Run the full duplicate detection pipeline.

    Returns a report dict suitable for JSON serialisation.
    """
    import os  # local import to avoid shadowing at module level

    if not dataset_dir.exists():
        return {
            "error": f"Dataset directory not found: {dataset_dir}",
            "suggestion": "Run inspect_dataset.py first to verify the dataset.",
        }

    print(f"\n{'='*60}")
    print(f"  Aura Farming — Duplicate Detection")
    print(f"  Dataset : {dataset_dir}")
    print(f"{'='*60}")

    all_paths = collect_image_paths(dataset_dir)
    print(f"\n  Total images discovered: {len(all_paths)}")

    if not all_paths:
        return {"error": "No images found in the dataset directory."}

    # Step 1: Exact duplicates
    exact_dups = find_exact_duplicates(all_paths)
    exact_dup_count = sum(len(v) - 1 for v in exact_dups.values())
    print(f"\n  Exact duplicate groups : {len(exact_dups)}")
    print(f"  Redundant exact copies : {exact_dup_count}")

    # Step 2: Near-duplicates
    near_dup_groups: list[list[Path]] = []
    if not exact_only:
        near_dup_groups = find_near_duplicates(all_paths, hash_size, threshold)
        print(f"\n  Near-duplicate groups  : {len(near_dup_groups)}")
        total_near = sum(len(g) for g in near_dup_groups)
        print(f"  Images in near-dup groups: {total_near}")
    else:
        print("\n  [Near-duplicate detection skipped (--exact-only)]")

    # Step 3: Group ID assignment
    group_map = assign_group_ids(exact_dups, near_dup_groups, all_paths)
    images_with_group = sum(1 for v in group_map.values() if v is not None)
    print(f"\n  Images assigned to a duplicate group: {images_with_group}")

    # Build report
    def paths_to_str(paths):
        return [str(p) for p in paths]

    report = {
        "dataset_dir": str(dataset_dir),
        "total_images": len(all_paths),
        "exact_duplicate_groups": len(exact_dups),
        "exact_redundant_copies": exact_dup_count,
        "near_duplicate_groups": len(near_dup_groups),
        "images_in_near_dup_groups": sum(len(g) for g in near_dup_groups),
        "images_with_group_id": images_with_group,
        "phash_threshold": threshold if not exact_only else None,
        "exact_groups": {
            md5: paths_to_str(paths) for md5, paths in exact_dups.items()
        },
        "near_dup_groups": [paths_to_str(g) for g in near_dup_groups],
        "group_id_map": {k: v for k, v in group_map.items() if v is not None},
    }

    print(f"\n{'='*60}")
    print(f"  Summary:")
    print(f"    Exact duplicate groups   : {len(exact_dups)}")
    print(f"    Near-duplicate groups    : {len(near_dup_groups)}")
    print(f"    Images needing grouping  : {images_with_group}")
    print(f"{'='*60}\n")

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(report, f, indent=2)
        print(f"  Report saved to: {output_path}\n")

    return report


def parse_args():
    parser = argparse.ArgumentParser(
        description="Aura Farming — Duplicate Detection Script"
    )
    parser.add_argument("--dataset-dir", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--exact-only",
        action="store_true",
        help="Only run exact (MD5) duplicate detection, skip perceptual hashing.",
    )
    parser.add_argument(
        "--threshold",
        type=int,
        default=None,
        help="Hamming distance threshold for near-duplicate detection (default from config.yaml)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config()
    ds_cfg = cfg["dataset"]

    dataset_dir = args.dataset_dir
    if dataset_dir is None:
        variant = ds_cfg.get("plantvillage_variant", "color")
        dataset_dir = PROJECT_ROOT / ds_cfg["plantvillage_dir"] / variant

    threshold = args.threshold or ds_cfg.get("phash_threshold", 4)
    hash_size = ds_cfg.get("phash_size", 8)
    output = args.output or (PROJECT_ROOT / ds_cfg.get("reports_dir", "data/reports") / "duplicates.json")

    if not dataset_dir.exists():
        print(f"\n❌  Dataset not found: {dataset_dir}")
        print("    Run inspect_dataset.py and download PlantVillage first.\n")
        sys.exit(1)

    run_duplicate_detection(
        dataset_dir=dataset_dir,
        exact_only=args.exact_only,
        hash_size=hash_size,
        threshold=threshold,
        output_path=output,
    )


import os  # used in collect_image_paths

if __name__ == "__main__":
    main()
