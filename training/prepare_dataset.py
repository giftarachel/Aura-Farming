"""
Aura Farming — training/prepare_dataset.py
============================================
Reproducible stratified dataset splitting with leakage prevention.

Prerequisites:
  - Run inspect_dataset.py to confirm the dataset is correctly placed.
  - Optionally run duplicate_detection.py first to generate
    data/reports/duplicates.json (used to keep duplicate groups
    in the same split). If not present, the script proceeds with
    exact-duplicate detection only (no near-duplicate grouping).

Usage (from project root):
    python training/prepare_dataset.py
    python training/prepare_dataset.py --dataset-dir data/plantvillage/color
    python training/prepare_dataset.py --dry-run

Outputs (all under data/splits/, excluded from Git):
    train_manifest.csv      — training split
    val_manifest.csv        — validation split
    test_manifest.csv       — test split
    split_report.json       — counts, per-class breakdown, leakage check

Split guarantees:
  1. Stratified: class proportions are preserved across splits.
  2. Duplicate-safe: duplicate groups are kept in a single split,
     preventing leakage of visually identical images.
  3. Reproducible: given the same seed, class mapping, and dataset,
     the splits are always identical.
  4. Leakage-verified: a post-split check confirms no file appears
     in more than one manifest.
  5. Test set is NEVER used for model selection, calibration, or
     threshold tuning. It is reserved for final evaluation only.

Phase 3 status: COMPLETE
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import yaml

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


# ─────────────────────────────────────────────────────────────────
# Configuration loading
# ─────────────────────────────────────────────────────────────────

def load_config() -> dict:
    with open(PROJECT_ROOT / "configs" / "config.yaml") as f:
        return yaml.safe_load(f)


def load_class_mapping() -> dict:
    with open(PROJECT_ROOT / "configs" / "class_mapping.json") as f:
        return json.load(f)


# ─────────────────────────────────────────────────────────────────
# Image collection
# ─────────────────────────────────────────────────────────────────

def collect_images_by_class(
    dataset_dir: Path,
    folder_to_index: dict[str, int],
) -> dict[int, list[Path]]:
    """
    Collect all image paths grouped by class index.

    The returned paths are always absolute (resolved) so they can be
    compared directly against the normalized keys in group_id_map
    regardless of whether dataset_dir was supplied as a relative or
    absolute path on the command line.

    Args:
        dataset_dir: Directory containing class subdirectories.
        folder_to_index: Mapping of folder name → integer class index.

    Returns:
        Dict mapping class_index → list of absolute image Paths.
    """
    # Resolve once so all child paths are absolute.
    dataset_dir = dataset_dir.resolve()

    images_by_class: dict[int, list[Path]] = defaultdict(list)
    missing_folders = []

    for folder_name, class_idx in sorted(folder_to_index.items()):
        class_dir = dataset_dir / folder_name
        if not class_dir.is_dir():
            missing_folders.append(folder_name)
            continue

        images = [
            f for f in class_dir.iterdir()
            if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS
        ]
        images_by_class[class_idx] = sorted(images)

    if missing_folders:
        print(f"  ⚠  Missing class folders (will be skipped):")
        for m in missing_folders:
            print(f"     - {m}")

    return dict(images_by_class)


# ─────────────────────────────────────────────────────────────────
# Duplicate group loading
# ─────────────────────────────────────────────────────────────────

def _normalize_path(raw: str, report_dataset_dir: Optional[str]) -> str:
    """
    Normalise a path key from the duplicate report to an absolute string.

    The duplicate report stores paths relative to the project root
    (e.g. 'data/plantvillage/color/Tomato___healthy/img.JPG').
    collect_images_by_class produces absolute paths.
    This function resolves the relative key against PROJECT_ROOT so
    lookups succeed regardless of which form was stored.

    If the key is already absolute, it is returned unchanged.

    Args:
        raw: Path string as stored in the duplicate report.
        report_dataset_dir: The 'dataset_dir' field from the report
            (used only for diagnostic messages, not for resolution).

    Returns:
        Absolute path string suitable for comparison with str(Path).
    """
    p = Path(raw)
    if p.is_absolute():
        return str(p)
    if raw.startswith(("/", "\\")):
        return raw
    return str(PROJECT_ROOT / p)


def load_group_id_map(duplicates_report_path: Path) -> dict[str, str]:
    """
    Load the group_id_map from a duplicate detection report.

    Returns an empty dict if the report is not available.

    Path normalisation:
        Keys in the JSON are stored as relative paths from the project
        root.  This function converts them to absolute paths so they
        match the absolute paths produced by collect_images_by_class.

    Returns:
        Dict mapping absolute_path_string → group_id.
    """
    if not duplicates_report_path.exists():
        print(
            f"  ℹ  No duplicate report found at {duplicates_report_path}.\n"
            "     Proceeding without near-duplicate grouping.\n"
            "     Run duplicate_detection.py to generate this report."
        )
        return {}
    with open(duplicates_report_path) as f:
        report = json.load(f)

    raw_map: dict[str, str] = report.get("group_id_map", {})
    report_dataset_dir: Optional[str] = report.get("dataset_dir")

    # Normalise all keys to absolute paths
    group_map: dict[str, str] = {
        _normalize_path(k, report_dataset_dir): v
        for k, v in raw_map.items()
    }
    print(
        f"  ✓  Loaded {len(group_map)} grouped-image entries from duplicate "
        f"report (paths normalised to absolute)."
    )
    return group_map


# ─────────────────────────────────────────────────────────────────
# Duplicate-report coverage audit
# ─────────────────────────────────────────────────────────────────

class GroupIdCoverageError(RuntimeError):
    """
    Raised when a non-empty duplicate report matches zero selected images.

    This almost always means the absolute paths stored in the duplicate
    report were generated on a different machine (e.g. Google Colab)
    and do not match the local paths in images_by_class.  If we
    continued silently every image would be treated as a singleton,
    completely defeating duplicate-group leakage prevention.
    """


def audit_group_id_coverage(
    images_by_class: dict[int, list[Path]],
    group_id_map: dict[str, str],
) -> tuple[int, int]:
    """
    Compare selected image paths against the keys in group_id_map.

    Call this BEFORE stratified_split_with_groups so path-mismatch
    problems are caught at a meaningful point.

    Rules enforced:
      - If group_id_map is empty, nothing to check — returns (0, total).
      - If group_id_map is non-empty but ZERO selected paths match,
        raises GroupIdCoverageError with a diagnostic message.
      - If some paths match and some do not, prints a warning and
        returns the counts.  Unmatched images become singletons later.

    Args:
        images_by_class: class_index → list of image Paths (already
                         collected from the local dataset directory).
        group_id_map:    str(path) → group_id, loaded from the
                         duplicate-detection report.

    Returns:
        Tuple (matched, unmatched) — counts of selected image paths
        that were / were not found as keys in group_id_map.

    Raises:
        GroupIdCoverageError: Non-empty report, zero paths matched.
    """
    all_selected = [
        str(p)
        for paths in images_by_class.values()
        for p in paths
    ]
    total = len(all_selected)

    if not group_id_map:
        print(
            f"  ℹ  group_id_map is empty — all {total} images "
            "will be treated as singletons."
        )
        return 0, total

    matched = sum(1 for p in all_selected if p in group_id_map)
    unmatched = total - matched

    if matched == 0:
        example_report_path = next(iter(group_id_map))
        example_dataset_path = all_selected[0] if all_selected else "<no images collected>"
        raise GroupIdCoverageError(
            f"\n❌  DUPLICATE REPORT MISMATCH\n"
            f"   The duplicate report contains {len(group_id_map)} entries "
            f"but NONE of the {total} selected image paths matched.\n"
            f"\n   Most likely cause: the report was generated on a different "
            f"machine (e.g. Google Colab) where absolute paths differ.\n"
            f"\n   Fixes:\n"
            f"     1. Re-run duplicate_detection.py on this machine to "
            f"regenerate the report with matching paths.\n"
            f"     2. Or delete the report and re-run this script; all images "
            f"will be treated as singletons (no near-duplicate grouping).\n"
            f"\n   Example path in report  : {example_report_path!r}\n"
            f"   Example path in dataset : {example_dataset_path!r}\n"
        )

    if unmatched > 0:
        pct = 100.0 * unmatched / total
        print(
            f"  ⚠  {matched}/{total} selected images matched the duplicate "
            f"report ({unmatched} unmatched, {pct:.1f}%). "
            "Unmatched images will be treated as singletons."
        )
    else:
        print(
            f"  ✓  All {total} selected images matched the duplicate report."
        )

    return matched, unmatched


# ─────────────────────────────────────────────────────────────────
# Post-split group-level leakage assertion
# ─────────────────────────────────────────────────────────────────

def verify_no_group_leakage(
    train: list[dict],
    val: list[dict],
    test: list[dict],
) -> tuple[bool, list[str]]:
    """
    Assert that no duplicate group ID appears in more than one split.

    This is a defence-in-depth check that catches a regression in
    the grouping logic that verify_no_leakage (path-level) would miss.
    It only examines records that carry a non-None, non-singleton
    group_id (i.e. records that were part of a named duplicate group).

    Args:
        train, val, test: Lists of record dicts, each with a
                          "group_id" key (str or None).

    Returns:
        Tuple (passed: bool, violations: list of descriptive strings).
        violations is empty when passed is True.
    """
    def _named_groups(records: list[dict]) -> set[str]:
        return {
            r["group_id"]
            for r in records
            if r.get("group_id") and not r["group_id"].startswith("_singleton_")
        }

    train_groups = _named_groups(train)
    val_groups   = _named_groups(val)
    test_groups  = _named_groups(test)

    violations: list[str] = []
    for gid in train_groups & val_groups:
        violations.append(f"group '{gid}' appears in both train and val")
    for gid in train_groups & test_groups:
        violations.append(f"group '{gid}' appears in both train and test")
    for gid in val_groups & test_groups:
        violations.append(f"group '{gid}' appears in both val and test")

    return len(violations) == 0, violations


# ─────────────────────────────────────────────────────────────────
# Stratified split with group constraints
# ─────────────────────────────────────────────────────────────────

def stratified_split_with_groups(
    images_by_class: dict[int, list[Path]],
    group_id_map: dict[str, str],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
    min_images_per_class: int,
) -> tuple[list[dict], list[dict], list[dict], dict]:
    """
    Perform a stratified train/val/test split with duplicate-group safety.

    Algorithm:
      Pre-pass (cross-class group resolution):
        Build a global map of group_id → assigned split.  Any group ID
        that appears across multiple classes is assigned exactly once
        here, so both the Bacterial_spot and Septoria_leaf_spot members
        of the same near-duplicate group go to the same split.

      Per-class pass:
        1. Look up each image's group_id.
        2. If the group was already assigned globally, honour that
           assignment for this image.
        3. Images whose group appears only within this class are treated
           as a local unit and assigned by the per-class ratio.
        4. Images with no group_id each form their own singleton unit.

    Args:
        images_by_class: class_index → list of image Paths.
        group_id_map: absolute_path_string → group_id for grouped images.
        train_ratio, val_ratio, test_ratio: Split fractions (must sum to 1.0).
        seed: Random seed for reproducibility.
        min_images_per_class: Minimum images required; smaller classes warned.

    Returns:
        Tuple of (train_records, val_records, test_records, split_report).
        Each record is a dict: {path, class_index, class_folder, group_id}.
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "Split ratios must sum to 1.0"

    rng = random.Random(seed)

    # ── Pre-pass: find groups that span multiple classes ──────────
    # For each named group_id, collect which class indices it touches.
    group_to_classes: dict[str, set[int]] = defaultdict(set)
    for class_idx, paths in images_by_class.items():
        for p in paths:
            gid = group_id_map.get(str(p))
            if gid:
                group_to_classes[gid].add(class_idx)

    cross_class_groups = {
        gid for gid, classes in group_to_classes.items()
        if len(classes) > 1
    }
    if cross_class_groups:
        print(
            f"  ℹ  {len(cross_class_groups)} duplicate group(s) span multiple "
            f"classes and will be assigned globally before per-class splitting."
        )

    # Globally assign cross-class groups to a split now, using the same
    # RNG so the result is deterministic given the seed.
    cross_class_list = sorted(cross_class_groups)
    rng.shuffle(cross_class_list)
    n_cc = len(cross_class_list)
    n_cc_train = max(1, round(n_cc * train_ratio)) if n_cc else 0
    n_cc_val   = max(1, round(n_cc * val_ratio))   if n_cc else 0
    n_cc_test  = n_cc - n_cc_train - n_cc_val
    if n_cc_test < 0:
        n_cc_val  = max(0, n_cc_val - 1)
        n_cc_test = n_cc - n_cc_train - n_cc_val

    global_assignment: dict[str, str] = {}  # group_id → "train" | "val" | "test"
    for gid in cross_class_list[:n_cc_train]:
        global_assignment[gid] = "train"
    for gid in cross_class_list[n_cc_train: n_cc_train + n_cc_val]:
        global_assignment[gid] = "val"
    for gid in cross_class_list[n_cc_train + n_cc_val:]:
        global_assignment[gid] = "test"

    # ── Per-class splitting ───────────────────────────────────────
    train_records: list[dict] = []
    val_records: list[dict] = []
    test_records: list[dict] = []
    split_report: dict[str, dict] = {}

    for class_idx in sorted(images_by_class.keys()):
        paths = images_by_class[class_idx]
        n = len(paths)

        if n < min_images_per_class:
            print(
                f"  ⚠  Class {class_idx} has only {n} images "
                f"(minimum recommended: {min_images_per_class}). "
                "It will be included but proportions may be imprecise."
            )

        # Separate images into:
        #   globally_fixed   — cross-class group, assignment already decided
        #   local_units      — group only seen within this class → unit-based split
        #   singletons       — no group_id → own unit
        globally_fixed: dict[str, list[Path]] = defaultdict(list)
        local_units: dict[str, list[Path]] = defaultdict(list)

        for p in paths:
            gid = group_id_map.get(str(p))
            if gid and gid in global_assignment:
                globally_fixed[gid].append(p)
            elif gid:
                local_units[gid].append(p)
            else:
                singleton_key = f"_singleton_{str(p)}"
                local_units[singleton_key].append(p)

        # Expand globally-fixed images into their predetermined splits
        def _make_record(p: Path, gid_override: Optional[str] = None) -> dict:
            return {
                "path": str(p),
                "class_index": class_idx,
                "class_folder": p.parent.name,
                "group_id": gid_override if gid_override is not None
                             else group_id_map.get(str(p)),
            }

        class_train: list[dict] = []
        class_val:   list[dict] = []
        class_test:  list[dict] = []

        for gid, gp_paths in globally_fixed.items():
            target = global_assignment[gid]
            records = [_make_record(p) for p in gp_paths]
            if target == "train":
                class_train.extend(records)
            elif target == "val":
                class_val.extend(records)
            else:
                class_test.extend(records)

        # Unit-based split for local and singleton groups
        unit_keys = sorted(local_units.keys())
        rng.shuffle(unit_keys)

        n_units = len(unit_keys)
        n_train = max(1, round(n_units * train_ratio)) if n_units else 0
        n_val   = max(1, round(n_units * val_ratio))   if n_units else 0
        n_test  = n_units - n_train - n_val
        if n_test < 0:
            n_val  = max(0, n_val - 1)
            n_test = n_units - n_train - n_val

        train_keys = unit_keys[:n_train]
        val_keys   = unit_keys[n_train: n_train + n_val]
        test_keys  = unit_keys[n_train + n_val:]

        for uid in train_keys:
            class_train.extend(_make_record(p) for p in local_units[uid])
        for uid in val_keys:
            class_val.extend(_make_record(p) for p in local_units[uid])
        for uid in test_keys:
            class_test.extend(_make_record(p) for p in local_units[uid])

        train_records.extend(class_train)
        val_records.extend(class_val)
        test_records.extend(class_test)

        split_report[str(class_idx)] = {
            "class_index": class_idx,
            "total": n,
            "train": len(class_train),
            "val":   len(class_val),
            "test":  len(class_test),
            "cross_class_groups": len(
                {group_id_map.get(str(p)) for p in paths
                 if group_id_map.get(str(p)) in global_assignment}
            ),
            "local_duplicate_units": len(
                [k for k in unit_keys
                 if not k.startswith("_singleton_")]
            ),
        }

    return train_records, val_records, test_records, split_report


# ─────────────────────────────────────────────────────────────────
# Leakage verification
# ─────────────────────────────────────────────────────────────────

def verify_no_leakage(
    train: list[dict],
    val: list[dict],
    test: list[dict],
) -> tuple[bool, list[str]]:
    """
    Verify that no file path appears in more than one split.

    Args:
        train, val, test: Lists of record dicts.

    Returns:
        Tuple (passed: bool, violations: list of path strings).
        violations is empty when passed is True.
    """
    train_paths = {r["path"] for r in train}
    val_paths = {r["path"] for r in val}
    test_paths = {r["path"] for r in test}

    violations = list(
        (train_paths & val_paths)
        | (train_paths & test_paths)
        | (val_paths & test_paths)
    )
    return len(violations) == 0, violations


# ─────────────────────────────────────────────────────────────────
# Manifest writing
# ─────────────────────────────────────────────────────────────────

def write_manifest(records: list[dict], output_path: Path) -> None:
    """Write a split manifest to a CSV file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["path", "class_index", "class_folder", "group_id"]
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            # Write group_id as empty string if None
            row = record.copy()
            row["group_id"] = record["group_id"] or ""
            writer.writerow(row)


# ─────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Aura Farming — Dataset Splitting Script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python training/prepare_dataset.py
  python training/prepare_dataset.py --dataset-dir data/plantvillage/color
  python training/prepare_dataset.py --dry-run
  python training/prepare_dataset.py --seed 123
        """,
    )
    parser.add_argument("--dataset-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--duplicates-report", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print split counts without writing any files.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config()
    ds_cfg = cfg["dataset"]
    mapping = load_class_mapping()
    folder_to_index: dict[str, int] = mapping["folder_to_index"]

    # Resolve paths
    variant = ds_cfg.get("plantvillage_variant", "color")
    dataset_dir = args.dataset_dir or (
        PROJECT_ROOT / ds_cfg["plantvillage_dir"] / variant
    )
    output_dir = args.output_dir or (PROJECT_ROOT / ds_cfg["splits_dir"])
    dup_report_path = args.duplicates_report or (
        PROJECT_ROOT / ds_cfg.get("reports_dir", "data/reports") / "duplicates.json"
    )
    seed = args.seed if args.seed is not None else ds_cfg["random_seed"]
    train_ratio = ds_cfg["train_ratio"]
    val_ratio = ds_cfg["val_ratio"]
    test_ratio = ds_cfg["test_ratio"]
    min_images = ds_cfg.get("min_images_per_class", 50)

    print(f"\n{'='*60}")
    print(f"  Aura Farming — Dataset Splitting")
    print(f"  Dataset   : {dataset_dir}")
    print(f"  Output    : {output_dir}")
    print(f"  Split     : {train_ratio:.0%} / {val_ratio:.0%} / {test_ratio:.0%}")
    print(f"  Seed      : {seed}")
    print(f"{'='*60}")

    if not dataset_dir.exists():
        print(f"\n❌  Dataset directory not found: {dataset_dir}")
        print("    Run inspect_dataset.py first.\n")
        sys.exit(1)

    # Collect images
    print("\n  Collecting images by class...")
    images_by_class = collect_images_by_class(dataset_dir, folder_to_index)
    total = sum(len(v) for v in images_by_class.values())
    print(f"  Total images collected: {total}")
    print(f"  Classes found: {len(images_by_class)}")

    if total == 0:
        print("\n❌  No images found. Check dataset path and class_mapping.json.\n")
        sys.exit(1)

    # Load duplicate groups
    group_id_map = load_group_id_map(dup_report_path)

    # Audit coverage: catch path-mismatch before splitting
    print("\n  Auditing duplicate-report coverage...")
    audit_group_id_coverage(images_by_class, group_id_map)

    # Split
    print("\n  Splitting...")
    train, val, test, split_report = stratified_split_with_groups(
        images_by_class=images_by_class,
        group_id_map=group_id_map,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed,
        min_images_per_class=min_images,
    )

    # Verify no path-level leakage
    print("\n  Verifying path-level leakage...")
    passed, violations = verify_no_leakage(train, val, test)
    if not passed:
        print(f"\n❌  PATH LEAKAGE — {len(violations)} files appear in multiple splits!")
        for v in violations[:10]:
            print(f"    {v}")
        sys.exit(1)
    print(f"  ✓  Path-level leakage check passed.")

    # Verify no group-level leakage
    print("\n  Verifying group-level leakage...")
    grp_passed, grp_violations = verify_no_group_leakage(train, val, test)
    if not grp_passed:
        print(f"\n❌  GROUP LEAKAGE — {len(grp_violations)} duplicate group(s) "
              f"appear in more than one split!")
        for v in grp_violations[:10]:
            print(f"    {v}")
        sys.exit(1)
    print(f"  ✓  Group-level leakage check passed.")

    # Print summary
    print(f"\n{'─'*60}")
    print(f"  {'Split':<10} {'Images':>8}  {'Pct':>6}")
    print(f"{'─'*60}")
    for name, records in [("train", train), ("val", val), ("test", test)]:
        pct = 100 * len(records) / max(total, 1)
        print(f"  {name:<10} {len(records):>8}  {pct:>5.1f}%")
    print(f"  {'TOTAL':<10} {total:>8}")
    print(f"{'─'*60}")

    print(f"\n  Per-class split counts:")
    print(f"  {'Class':>5}  {'Total':>6}  {'Train':>6}  {'Val':>6}  {'Test':>6}")
    for cidx, sr in sorted(split_report.items(), key=lambda x: int(x[0])):
        print(
            f"  {cidx:>5}  {sr['total']:>6}  "
            f"{sr['train']:>6}  {sr['val']:>6}  {sr['test']:>6}"
        )

    if args.dry_run:
        print("\n  [Dry run] No files written.\n")
        return

    # Write manifests
    output_dir.mkdir(parents=True, exist_ok=True)
    write_manifest(train, output_dir / "train_manifest.csv")
    write_manifest(val, output_dir / "val_manifest.csv")
    write_manifest(test, output_dir / "test_manifest.csv")

    # Write split report
    report_out = {
        "seed": seed,
        "train_ratio": train_ratio,
        "val_ratio": val_ratio,
        "test_ratio": test_ratio,
        "total_images": total,
        "train_count": len(train),
        "val_count": len(val),
        "test_count": len(test),
        "leakage_check_passed": True,
        "per_class": split_report,
    }
    report_path = output_dir / "split_report.json"
    with open(report_path, "w") as f:
        json.dump(report_out, f, indent=2)

    print(f"\n  ✓  Manifests written to {output_dir}/")
    print(f"     train_manifest.csv, val_manifest.csv, test_manifest.csv")
    print(f"     split_report.json")
    print(f"\n  ⚠  REMINDER: Do not use test_manifest.csv during training,")
    print(f"     calibration, or threshold selection. It is reserved for")
    print(f"     final evaluation in Phase 7 only.\n")

    print(f"{'='*60}")
    print(f"  Dataset splitting complete.\n")


if __name__ == "__main__":
    main()
