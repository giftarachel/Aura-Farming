"""
Aura Farming — tests/test_prepare_dataset.py
==============================================
Unit tests for the splitting logic in training/prepare_dataset.py.

These tests construct a tiny synthetic dataset in a temp directory
so they run WITHOUT the PlantVillage dataset being present.

Tests verify:
  - Stratified splitting produces correct approximate ratios.
  - No file appears in more than one split (leakage).
  - Duplicate groups are never split across train/val/test.
  - Small classes (< min_images) are handled without crashing.
  - The same seed always produces the same split.
  - The leakage verifier correctly detects overlap.
  - Manifests are written as valid CSVs.
  - The split report JSON is correct.

Run with:
    pytest tests/test_prepare_dataset.py -v
"""

from __future__ import annotations

import csv
import io
import json
import sys
from pathlib import Path

import pytest
from PIL import Image

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from training.prepare_dataset import (
    stratified_split_with_groups,
    verify_no_leakage,
    verify_no_group_leakage,
    audit_group_id_coverage,
    GroupIdCoverageError,
    write_manifest,
    collect_images_by_class,
    load_group_id_map,
    _normalize_path,
)


# ─────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────

def _make_fake_image(path: Path):
    """Write a minimal 64×64 RGB JPEG to the given path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", (64, 64), color=(100, 150, 80))
    img.save(path, format="JPEG")


def _make_synthetic_dataset(
    base_dir: Path,
    class_counts: dict[int, int],
    folder_names: dict[int, str] | None = None,
) -> tuple[dict[int, list[Path]], dict[str, int]]:
    """
    Create a temporary directory tree with fake JPEG images.

    Args:
        base_dir: Temp directory to create images in.
        class_counts: class_index → number of images.
        folder_names: Optional class_index → folder name mapping.

    Returns:
        Tuple of (images_by_class, folder_to_index).
    """
    if folder_names is None:
        folder_names = {i: f"Class_{i:02d}" for i in class_counts}

    folder_to_index = {v: k for k, v in folder_names.items()}
    images_by_class: dict[int, list[Path]] = {}

    for class_idx, count in class_counts.items():
        folder = folder_names[class_idx]
        class_dir = base_dir / folder
        paths = []
        for j in range(count):
            p = class_dir / f"img_{j:04d}.jpg"
            _make_fake_image(p)
            paths.append(p)
        images_by_class[class_idx] = paths

    return images_by_class, folder_to_index


# ─────────────────────────────────────────────────────────────────
# Core split tests
# ─────────────────────────────────────────────────────────────────

class TestStratifiedSplitWithGroups:
    def test_no_leakage_basic(self, tmp_path):
        images_by_class, _ = _make_synthetic_dataset(
            tmp_path,
            {0: 100, 1: 80, 2: 60},
        )
        train, val, test, _ = stratified_split_with_groups(
            images_by_class=images_by_class,
            group_id_map={},
            train_ratio=0.70,
            val_ratio=0.15,
            test_ratio=0.15,
            seed=42,
            min_images_per_class=10,
        )
        passed, violations = verify_no_leakage(train, val, test)
        assert passed is True, f"Leakage detected: {violations[:3]}"

    def test_all_images_assigned(self, tmp_path):
        counts = {0: 100, 1: 80, 2: 60}
        total = sum(counts.values())
        images_by_class, _ = _make_synthetic_dataset(tmp_path, counts)
        train, val, test, _ = stratified_split_with_groups(
            images_by_class=images_by_class,
            group_id_map={},
            train_ratio=0.70, val_ratio=0.15, test_ratio=0.15,
            seed=42, min_images_per_class=10,
        )
        assert len(train) + len(val) + len(test) == total

    def test_train_is_largest_split(self, tmp_path):
        images_by_class, _ = _make_synthetic_dataset(
            tmp_path, {0: 200, 1: 150}
        )
        train, val, test, _ = stratified_split_with_groups(
            images_by_class=images_by_class,
            group_id_map={},
            train_ratio=0.70, val_ratio=0.15, test_ratio=0.15,
            seed=42, min_images_per_class=10,
        )
        assert len(train) > len(val)
        assert len(train) > len(test)

    def test_reproducibility_same_seed(self, tmp_path):
        images_by_class, _ = _make_synthetic_dataset(
            tmp_path, {0: 100, 1: 80}
        )
        result_a = stratified_split_with_groups(
            images_by_class=images_by_class,
            group_id_map={},
            train_ratio=0.70, val_ratio=0.15, test_ratio=0.15,
            seed=42, min_images_per_class=5,
        )
        result_b = stratified_split_with_groups(
            images_by_class=images_by_class,
            group_id_map={},
            train_ratio=0.70, val_ratio=0.15, test_ratio=0.15,
            seed=42, min_images_per_class=5,
        )
        paths_a = sorted(r["path"] for r in result_a[0])
        paths_b = sorted(r["path"] for r in result_b[0])
        assert paths_a == paths_b

    def test_different_seeds_produce_different_splits(self, tmp_path):
        images_by_class, _ = _make_synthetic_dataset(
            tmp_path, {0: 100, 1: 80}
        )
        result_42 = stratified_split_with_groups(
            images_by_class=images_by_class,
            group_id_map={},
            train_ratio=0.70, val_ratio=0.15, test_ratio=0.15,
            seed=42, min_images_per_class=5,
        )
        result_99 = stratified_split_with_groups(
            images_by_class=images_by_class,
            group_id_map={},
            train_ratio=0.70, val_ratio=0.15, test_ratio=0.15,
            seed=99, min_images_per_class=5,
        )
        paths_42 = sorted(r["path"] for r in result_42[0])
        paths_99 = sorted(r["path"] for r in result_99[0])
        assert paths_42 != paths_99

    def test_duplicate_group_stays_together(self, tmp_path):
        """Images in the same duplicate group must all end up in the same split."""
        images_by_class, _ = _make_synthetic_dataset(
            tmp_path, {0: 30}
        )
        paths = images_by_class[0]

        # Mark the first 3 images as an exact duplicate group
        group_id_map = {
            str(paths[0]): "exact_aaaa",
            str(paths[1]): "exact_aaaa",
            str(paths[2]): "exact_aaaa",
        }
        train, val, test, _ = stratified_split_with_groups(
            images_by_class=images_by_class,
            group_id_map=group_id_map,
            train_ratio=0.70, val_ratio=0.15, test_ratio=0.15,
            seed=42, min_images_per_class=5,
        )
        # Find which split contains paths[0]
        grouped = {str(paths[0]), str(paths[1]), str(paths[2])}
        for split_records, split_name in [(train, "train"), (val, "val"), (test, "test")]:
            split_paths = {r["path"] for r in split_records}
            if str(paths[0]) in split_paths:
                # All grouped images must be in this same split
                for p in grouped:
                    assert p in split_paths, (
                        f"Duplicate group member {p} not in {split_name}"
                    )
                break

    def test_small_class_does_not_crash(self, tmp_path):
        """Classes with very few images should be handled without an exception."""
        images_by_class, _ = _make_synthetic_dataset(
            tmp_path, {0: 5, 1: 100}
        )
        train, val, test, _ = stratified_split_with_groups(
            images_by_class=images_by_class,
            group_id_map={},
            train_ratio=0.70, val_ratio=0.15, test_ratio=0.15,
            seed=42, min_images_per_class=50,  # 5 < 50 → warning, not crash
        )
        passed, _ = verify_no_leakage(train, val, test)
        assert passed is True

    def test_each_class_represented_in_train(self, tmp_path):
        """Every class should have at least 1 image in the training split."""
        images_by_class, _ = _make_synthetic_dataset(
            tmp_path, {0: 20, 1: 20, 2: 20}
        )
        train, val, test, _ = stratified_split_with_groups(
            images_by_class=images_by_class,
            group_id_map={},
            train_ratio=0.70, val_ratio=0.15, test_ratio=0.15,
            seed=42, min_images_per_class=5,
        )
        train_classes = {r["class_index"] for r in train}
        assert 0 in train_classes
        assert 1 in train_classes
        assert 2 in train_classes


# ─────────────────────────────────────────────────────────────────
# Leakage verifier tests
# ─────────────────────────────────────────────────────────────────

class TestVerifyNoLeakage:
    def _make_records(self, paths: list[str]) -> list[dict]:
        return [{"path": p, "class_index": 0,
                 "class_folder": "x", "group_id": None}
                for p in paths]

    def test_clean_split_passes(self):
        train = self._make_records(["/a.jpg", "/b.jpg"])
        val = self._make_records(["/c.jpg"])
        test = self._make_records(["/d.jpg"])
        passed, violations = verify_no_leakage(train, val, test)
        assert passed is True
        assert violations == []

    def test_train_test_overlap_detected(self):
        train = self._make_records(["/shared.jpg", "/a.jpg"])
        val = self._make_records(["/b.jpg"])
        test = self._make_records(["/shared.jpg", "/c.jpg"])
        passed, violations = verify_no_leakage(train, val, test)
        assert passed is False
        assert "/shared.jpg" in violations

    def test_train_val_overlap_detected(self):
        train = self._make_records(["/shared.jpg"])
        val = self._make_records(["/shared.jpg"])
        test = self._make_records(["/c.jpg"])
        passed, violations = verify_no_leakage(train, val, test)
        assert passed is False

    def test_val_test_overlap_detected(self):
        train = self._make_records(["/a.jpg"])
        val = self._make_records(["/shared.jpg"])
        test = self._make_records(["/shared.jpg"])
        passed, violations = verify_no_leakage(train, val, test)
        assert passed is False


# ─────────────────────────────────────────────────────────────────
# Manifest writing tests
# ─────────────────────────────────────────────────────────────────

class TestWriteManifest:
    def test_manifest_is_valid_csv(self, tmp_path):
        records = [
            {"path": "/img1.jpg", "class_index": 0,
             "class_folder": "Tomato___healthy", "group_id": None},
            {"path": "/img2.jpg", "class_index": 1,
             "class_folder": "Tomato___Early_blight", "group_id": "exact_ab12"},
        ]
        output = tmp_path / "train.csv"
        write_manifest(records, output)
        assert output.exists()

        with open(output) as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        assert len(rows) == 2
        assert rows[0]["path"] == "/img1.jpg"
        assert rows[0]["class_index"] == "0"
        assert rows[1]["group_id"] == "exact_ab12"

    def test_none_group_id_written_as_empty(self, tmp_path):
        records = [{"path": "/img.jpg", "class_index": 0,
                    "class_folder": "x", "group_id": None}]
        output = tmp_path / "manifest.csv"
        write_manifest(records, output)
        with open(output) as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        assert rows[0]["group_id"] == ""

    def test_parent_directory_created(self, tmp_path):
        nested = tmp_path / "deep" / "nested" / "manifest.csv"
        write_manifest([], nested)
        assert nested.exists()


# ─────────────────────────────────────────────────────────────────
# audit_group_id_coverage tests
# ─────────────────────────────────────────────────────────────────

class TestAuditGroupIdCoverage:
    """Tests for path-mismatch detection before splitting."""

    def _make_images_by_class(
        self, tmp_path: Path, counts: dict[int, int]
    ) -> dict[int, list[Path]]:
        """Create fake image files and return images_by_class dict."""
        images_by_class: dict[int, list[Path]] = {}
        for class_idx, count in counts.items():
            class_dir = tmp_path / f"Class_{class_idx:02d}"
            class_dir.mkdir(parents=True, exist_ok=True)
            paths = []
            for j in range(count):
                p = class_dir / f"img_{j:04d}.jpg"
                _make_fake_image(p)
                paths.append(p)
            images_by_class[class_idx] = paths
        return images_by_class

    def test_empty_map_returns_zero_matched(self, tmp_path):
        ibc = self._make_images_by_class(tmp_path, {0: 10})
        matched, unmatched = audit_group_id_coverage(ibc, {})
        assert matched == 0
        assert unmatched == 10

    def test_all_paths_matched(self, tmp_path):
        ibc = self._make_images_by_class(tmp_path, {0: 5, 1: 5})
        all_paths = [p for paths in ibc.values() for p in paths]
        group_map = {str(p): "exact_abc" for p in all_paths}
        matched, unmatched = audit_group_id_coverage(ibc, group_map)
        assert matched == 10
        assert unmatched == 0

    def test_partial_match_returns_correct_counts(self, tmp_path):
        ibc = self._make_images_by_class(tmp_path, {0: 6})
        paths = ibc[0]
        # Only match first 4 of 6
        group_map = {str(p): "exact_abc" for p in paths[:4]}
        matched, unmatched = audit_group_id_coverage(ibc, group_map)
        assert matched == 4
        assert unmatched == 2

    def test_zero_match_with_nonempty_map_raises(self, tmp_path):
        """Non-empty report but no local paths match → must raise."""
        ibc = self._make_images_by_class(tmp_path, {0: 5})
        # Group map with completely different (fake remote) paths
        group_map = {
            "/remote/colab/data/img_0000.jpg": "exact_xyz",
            "/remote/colab/data/img_0001.jpg": "exact_xyz",
        }
        with pytest.raises(GroupIdCoverageError) as exc_info:
            audit_group_id_coverage(ibc, group_map)
        # Error message must include actionable guidance
        msg = str(exc_info.value)
        assert "MISMATCH" in msg or "mismatch" in msg.lower()
        assert "Re-run" in msg or "re-run" in msg.lower() or "regenerate" in msg.lower()

    def test_zero_match_error_shows_example_paths(self, tmp_path):
        """Error message should show one report path and one dataset path."""
        ibc = self._make_images_by_class(tmp_path, {0: 3})
        group_map = {"/different/machine/img.jpg": "exact_aaa"}
        with pytest.raises(GroupIdCoverageError) as exc_info:
            audit_group_id_coverage(ibc, group_map)
        msg = str(exc_info.value)
        assert "/different/machine/img.jpg" in msg

    def test_empty_images_by_class_with_nonempty_map(self, tmp_path):
        """Edge case: no images collected at all, non-empty map → raises."""
        with pytest.raises(GroupIdCoverageError):
            audit_group_id_coverage({}, {"/some/path.jpg": "grp_01"})

    def test_partial_match_does_not_raise(self, tmp_path):
        """Partial match is allowed — should not raise."""
        ibc = self._make_images_by_class(tmp_path, {0: 10})
        paths = ibc[0]
        group_map = {str(paths[0]): "exact_001"}  # only 1 of 10 matched
        # Must not raise — should return (1, 9)
        matched, unmatched = audit_group_id_coverage(ibc, group_map)
        assert matched == 1
        assert unmatched == 9


# ─────────────────────────────────────────────────────────────────
# verify_no_group_leakage tests
# ─────────────────────────────────────────────────────────────────

class TestVerifyNoGroupLeakage:
    """Tests for the post-split group-level leakage assertion."""

    def _rec(self, path: str, group_id: str | None) -> dict:
        return {
            "path": path,
            "class_index": 0,
            "class_folder": "x",
            "group_id": group_id,
        }

    def test_clean_no_named_groups_passes(self):
        train = [self._rec("/a.jpg", None), self._rec("/b.jpg", None)]
        val   = [self._rec("/c.jpg", None)]
        test  = [self._rec("/d.jpg", None)]
        passed, violations = verify_no_group_leakage(train, val, test)
        assert passed is True
        assert violations == []

    def test_same_group_in_train_and_test_detected(self):
        train = [self._rec("/a.jpg", "exact_001"), self._rec("/b.jpg", "exact_001")]
        val   = [self._rec("/c.jpg", None)]
        test  = [self._rec("/d.jpg", "exact_001")]   # same group as train
        passed, violations = verify_no_group_leakage(train, val, test)
        assert passed is False
        assert any("exact_001" in v for v in violations)
        assert any("train" in v and "test" in v for v in violations)

    def test_same_group_in_train_and_val_detected(self):
        train = [self._rec("/a.jpg", "near_0002")]
        val   = [self._rec("/b.jpg", "near_0002")]   # same group as train
        test  = [self._rec("/c.jpg", None)]
        passed, violations = verify_no_group_leakage(train, val, test)
        assert passed is False
        assert any("near_0002" in v for v in violations)

    def test_same_group_in_val_and_test_detected(self):
        train = [self._rec("/a.jpg", None)]
        val   = [self._rec("/b.jpg", "exact_007")]
        test  = [self._rec("/c.jpg", "exact_007")]
        passed, violations = verify_no_group_leakage(train, val, test)
        assert passed is False
        assert any("exact_007" in v for v in violations)

    def test_different_groups_in_each_split_passes(self):
        train = [self._rec("/a.jpg", "exact_001")]
        val   = [self._rec("/b.jpg", "exact_002")]
        test  = [self._rec("/c.jpg", "exact_003")]
        passed, violations = verify_no_group_leakage(train, val, test)
        assert passed is True
        assert violations == []

    def test_singleton_keys_ignored(self):
        """Keys starting with _singleton_ must not trigger group-leakage."""
        sk = "_singleton_/some/path.jpg"
        train = [self._rec("/a.jpg", sk)]
        val   = [self._rec("/b.jpg", sk)]   # same singleton key
        test  = [self._rec("/c.jpg", None)]
        passed, violations = verify_no_group_leakage(train, val, test)
        assert passed is True

    def test_multiple_group_violations_all_reported(self):
        train = [
            self._rec("/a.jpg", "exact_001"),
            self._rec("/b.jpg", "exact_002"),
        ]
        val = [self._rec("/c.jpg", None)]
        test = [
            self._rec("/d.jpg", "exact_001"),
            self._rec("/e.jpg", "exact_002"),
        ]
        passed, violations = verify_no_group_leakage(train, val, test)
        assert passed is False
        assert len(violations) == 2

    def test_split_produced_by_stratified_has_no_group_leakage(self, tmp_path):
        """
        End-to-end: splits produced by stratified_split_with_groups
        must pass the group-level check.
        """
        images_by_class, _ = _make_synthetic_dataset(tmp_path, {0: 30, 1: 20})
        paths_0 = images_by_class[0]
        # Two named duplicate groups in class 0
        group_map = {
            str(paths_0[0]): "exact_aaa",
            str(paths_0[1]): "exact_aaa",
            str(paths_0[2]): "exact_bbb",
            str(paths_0[3]): "exact_bbb",
            str(paths_0[4]): "exact_bbb",
        }
        train, val, test, _ = stratified_split_with_groups(
            images_by_class=images_by_class,
            group_id_map=group_map,
            train_ratio=0.70, val_ratio=0.15, test_ratio=0.15,
            seed=42, min_images_per_class=5,
        )
        passed, violations = verify_no_group_leakage(train, val, test)
        assert passed is True, f"Group leakage found: {violations}"


# ─────────────────────────────────────────────────────────────────
# _normalize_path tests
# ─────────────────────────────────────────────────────────────────

class TestNormalizePath:
    """Unit tests for the path-normalisation helper."""

    def test_absolute_path_unchanged(self):
        result = _normalize_path("/absolute/path/img.jpg", None)
        assert result == "/absolute/path/img.jpg"

    def test_relative_path_resolved_against_project_root(self):
        from training.prepare_dataset import PROJECT_ROOT
        rel = "data/plantvillage/color/Tomato___healthy/img.jpg"
        result = _normalize_path(rel, None)
        expected = str(PROJECT_ROOT / rel)
        assert result == expected

    def test_relative_path_with_no_report_dir(self):
        """report_dataset_dir is only used for diagnostics; None is fine."""
        from training.prepare_dataset import PROJECT_ROOT
        rel = "data/plantvillage/color/Tomato___Leaf_Mold/img.JPG"
        result = _normalize_path(rel, None)
        assert result == str(PROJECT_ROOT / rel)

    def test_already_absolute_not_double_prefixed(self):
        """Absolute paths must never have PROJECT_ROOT prepended."""
        from training.prepare_dataset import PROJECT_ROOT
        abs_path = str(PROJECT_ROOT / "data/plantvillage/color/Tomato___healthy/img.jpg")
        result = _normalize_path(abs_path, None)
        assert result == abs_path
        assert result.count(str(PROJECT_ROOT)) == 1


# ─────────────────────────────────────────────────────────────────
# load_group_id_map path-normalisation integration test
# ─────────────────────────────────────────────────────────────────

class TestLoadGroupIdMapNormalization:
    """Verify load_group_id_map converts relative keys to absolute paths."""

    def test_relative_keys_become_absolute(self, tmp_path):
        """
        A report stored with relative keys (as the real duplicates.json uses)
        must produce a group_id_map whose keys are absolute paths matching
        what collect_images_by_class returns.
        """
        from training.prepare_dataset import PROJECT_ROOT

        # Build a fake report with a relative key
        rel_key = "data/plantvillage/color/Tomato___healthy/some_img.JPG"
        report = {
            "dataset_dir": "data/plantvillage/color",
            "group_id_map": {rel_key: "exact_abc123"},
        }
        report_path = tmp_path / "duplicates.json"
        report_path.write_text(json.dumps(report))

        loaded = load_group_id_map(report_path)

        expected_abs = str(PROJECT_ROOT / rel_key)
        assert expected_abs in loaded, (
            f"Expected absolute key {expected_abs!r} not found. "
            f"Keys present: {list(loaded.keys())[:3]}"
        )
        assert loaded[expected_abs] == "exact_abc123"

    def test_absolute_keys_preserved(self, tmp_path):
        """If the report already stores absolute paths, they pass through unchanged."""
        abs_key = "/some/absolute/path/img.jpg"
        report = {
            "dataset_dir": "/some/absolute/path",
            "group_id_map": {abs_key: "near_0001"},
        }
        report_path = tmp_path / "duplicates.json"
        report_path.write_text(json.dumps(report))

        loaded = load_group_id_map(report_path)
        assert abs_key in loaded
        assert loaded[abs_key] == "near_0001"

    def test_missing_report_returns_empty(self, tmp_path):
        result = load_group_id_map(tmp_path / "nonexistent.json")
        assert result == {}


# ─────────────────────────────────────────────────────────────────
# Cross-class duplicate group tests
# ─────────────────────────────────────────────────────────────────

class TestCrossClassGroups:
    """
    Tests for groups whose members span two or more classes.
    The critical invariant: every member of a group must land in
    the same split regardless of which class it belongs to.
    """

    def test_cross_class_group_stays_in_one_split(self, tmp_path):
        """
        A group with one image in class 0 and one in class 1 must not
        be split across train/val/test.
        """
        ibc, _ = _make_synthetic_dataset(tmp_path, {0: 20, 1: 20})
        p0 = ibc[0][0]   # class 0
        p1 = ibc[1][0]   # class 1
        group_map = {str(p0): "cross_group_001", str(p1): "cross_group_001"}

        train, val, test, _ = stratified_split_with_groups(
            images_by_class=ibc,
            group_id_map=group_map,
            train_ratio=0.70, val_ratio=0.15, test_ratio=0.15,
            seed=42, min_images_per_class=5,
        )
        passed, violations = verify_no_group_leakage(train, val, test)
        assert passed is True, f"Cross-class group leaked: {violations}"

    def test_cross_class_group_all_members_present(self, tmp_path):
        """Every image in a cross-class group must appear exactly once."""
        ibc, _ = _make_synthetic_dataset(tmp_path, {0: 20, 1: 20})
        p0 = ibc[0][0]
        p1 = ibc[1][0]
        group_map = {str(p0): "cross_grp", str(p1): "cross_grp"}

        train, val, test, _ = stratified_split_with_groups(
            images_by_class=ibc,
            group_id_map=group_map,
            train_ratio=0.70, val_ratio=0.15, test_ratio=0.15,
            seed=42, min_images_per_class=5,
        )
        all_records = train + val + test
        all_paths = {r["path"] for r in all_records}
        assert str(p0) in all_paths, "p0 missing from all splits"
        assert str(p1) in all_paths, "p1 missing from all splits"

    def test_cross_class_group_total_images_preserved(self, tmp_path):
        """Total image count must equal sum of class sizes."""
        counts = {0: 30, 1: 25, 2: 20}
        total = sum(counts.values())
        ibc, _ = _make_synthetic_dataset(tmp_path, counts)

        # Make two cross-class groups
        group_map = {
            str(ibc[0][0]): "xgrp_A",
            str(ibc[1][0]): "xgrp_A",   # class 0 + class 1
            str(ibc[1][1]): "xgrp_B",
            str(ibc[2][0]): "xgrp_B",   # class 1 + class 2
        }
        train, val, test, _ = stratified_split_with_groups(
            images_by_class=ibc,
            group_id_map=group_map,
            train_ratio=0.70, val_ratio=0.15, test_ratio=0.15,
            seed=42, min_images_per_class=5,
        )
        assert len(train) + len(val) + len(test) == total

    def test_multiple_cross_class_groups_no_leakage(self, tmp_path):
        """Many cross-class groups must all pass both leakage checks."""
        ibc, _ = _make_synthetic_dataset(tmp_path, {0: 50, 1: 50})
        # Create 10 cross-class pairs
        group_map = {}
        for i in range(10):
            gid = f"xgrp_{i:03d}"
            group_map[str(ibc[0][i])] = gid
            group_map[str(ibc[1][i])] = gid

        train, val, test, _ = stratified_split_with_groups(
            images_by_class=ibc,
            group_id_map=group_map,
            train_ratio=0.70, val_ratio=0.15, test_ratio=0.15,
            seed=42, min_images_per_class=5,
        )
        path_ok, path_violations = verify_no_leakage(train, val, test)
        grp_ok, grp_violations = verify_no_group_leakage(train, val, test)
        assert path_ok, f"Path leakage: {path_violations[:3]}"
        assert grp_ok, f"Group leakage: {grp_violations[:3]}"

    def test_cross_class_group_reproducible(self, tmp_path):
        """Cross-class group assignment must be deterministic with fixed seed."""
        ibc, _ = _make_synthetic_dataset(tmp_path, {0: 30, 1: 30})
        group_map = {
            str(ibc[0][0]): "xgrp_rep",
            str(ibc[1][0]): "xgrp_rep",
        }
        result_a = stratified_split_with_groups(
            images_by_class=ibc, group_id_map=group_map,
            train_ratio=0.70, val_ratio=0.15, test_ratio=0.15,
            seed=42, min_images_per_class=5,
        )
        result_b = stratified_split_with_groups(
            images_by_class=ibc, group_id_map=group_map,
            train_ratio=0.70, val_ratio=0.15, test_ratio=0.15,
            seed=42, min_images_per_class=5,
        )
        paths_a = sorted(r["path"] for r in result_a[0])
        paths_b = sorted(r["path"] for r in result_b[0])
        assert paths_a == paths_b


# ─────────────────────────────────────────────────────────────────
# Unmatched images remain singletons
# ─────────────────────────────────────────────────────────────────

class TestUnmatchedImagesAsSingletons:
    """Images not in the group_id_map must become independent singleton units."""

    def test_unmatched_images_all_assigned(self, tmp_path):
        """Even with a partial group_id_map every image must end up in a split."""
        counts = {0: 20, 1: 20}
        total = sum(counts.values())
        ibc, _ = _make_synthetic_dataset(tmp_path, counts)
        # Only mark 3 images as grouped
        group_map = {
            str(ibc[0][0]): "exact_001",
            str(ibc[0][1]): "exact_001",
            str(ibc[1][0]): "exact_002",
        }
        train, val, test, _ = stratified_split_with_groups(
            images_by_class=ibc,
            group_id_map=group_map,
            train_ratio=0.70, val_ratio=0.15, test_ratio=0.15,
            seed=42, min_images_per_class=5,
        )
        assert len(train) + len(val) + len(test) == total

    def test_unmatched_images_have_none_group_id(self, tmp_path):
        """Records for unmatched images must carry group_id=None."""
        ibc, _ = _make_synthetic_dataset(tmp_path, {0: 10})
        paths = ibc[0]
        # Only first image is in the group
        group_map = {str(paths[0]): "exact_aaa"}
        train, val, test, _ = stratified_split_with_groups(
            images_by_class=ibc,
            group_id_map=group_map,
            train_ratio=0.70, val_ratio=0.15, test_ratio=0.15,
            seed=42, min_images_per_class=5,
        )
        all_records = train + val + test
        unmatched_records = [
            r for r in all_records if r["path"] != str(paths[0])
        ]
        for r in unmatched_records:
            assert r["group_id"] is None, (
                f"Expected None group_id for unmatched image, got {r['group_id']!r}"
            )

    def test_partial_match_passes_leakage_checks(self, tmp_path):
        """Partial group_id_map must still pass both leakage checks."""
        ibc, _ = _make_synthetic_dataset(tmp_path, {0: 30, 1: 30})
        group_map = {
            str(ibc[0][0]): "exact_x",
            str(ibc[0][1]): "exact_x",
        }
        train, val, test, _ = stratified_split_with_groups(
            images_by_class=ibc,
            group_id_map=group_map,
            train_ratio=0.70, val_ratio=0.15, test_ratio=0.15,
            seed=42, min_images_per_class=5,
        )
        path_ok, _ = verify_no_leakage(train, val, test)
        grp_ok, _ = verify_no_group_leakage(train, val, test)
        assert path_ok
        assert grp_ok
