"""
Aura Farming — tests/test_dataset_utils.py
============================================
Unit tests for training/dataset_utils.py and the
class_mapping.json schema.

These tests run without the PlantVillage dataset present.
They verify:
  - class_mapping.json schema and consistency
  - manifest loading and summarising
  - leakage verification logic (no-overlap check)
  - is_image_file extension filtering

Run with:
    pytest tests/test_dataset_utils.py -v
"""

from __future__ import annotations

import csv
import json
import os
import sys
import tempfile
from pathlib import Path

import pytest

# Ensure project root is on the path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from training.dataset_utils import (
    is_image_file,
    load_manifest,
    verify_manifests_no_overlap,
    summarise_manifest,
)


# ─────────────────────────────────────────────────────────────────
# class_mapping.json schema tests
# ─────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def class_mapping():
    mapping_path = PROJECT_ROOT / "configs" / "class_mapping.json"
    with open(mapping_path) as f:
        return json.load(f)


class TestClassMapping:
    """Verify the class_mapping.json is structurally correct."""

    def test_mapping_file_exists(self):
        assert (PROJECT_ROOT / "configs" / "class_mapping.json").exists()

    def test_has_classes_key(self, class_mapping):
        assert "classes" in class_mapping

    def test_has_ten_classes(self, class_mapping):
        classes = class_mapping["classes"]
        assert len(classes) == 10, f"Expected 10 classes, found {len(classes)}"

    def test_indices_are_zero_to_nine(self, class_mapping):
        indices = [int(k) for k in class_mapping["classes"].keys()]
        assert sorted(indices) == list(range(10))

    def test_every_class_has_folder_name(self, class_mapping):
        for idx, info in class_mapping["classes"].items():
            assert "folder_name" in info, f"Class {idx} missing folder_name"
            assert info["folder_name"], f"Class {idx} has empty folder_name"

    def test_every_class_has_display_name(self, class_mapping):
        for idx, info in class_mapping["classes"].items():
            assert "display_name" in info, f"Class {idx} missing display_name"

    def test_all_folder_names_start_with_tomato(self, class_mapping):
        for idx, info in class_mapping["classes"].items():
            assert info["folder_name"].startswith("Tomato___"), (
                f"Class {idx} folder_name '{info['folder_name']}' "
                "does not start with 'Tomato___'"
            )

    def test_folder_to_index_matches_classes(self, class_mapping):
        """folder_to_index must be the inverse of classes."""
        folder_to_index = class_mapping.get("folder_to_index", {})
        for idx, info in class_mapping["classes"].items():
            folder = info["folder_name"]
            assert folder in folder_to_index, (
                f"folder_to_index missing entry for '{folder}'"
            )
            assert folder_to_index[folder] == int(idx), (
                f"Index mismatch for '{folder}': "
                f"classes says {idx}, folder_to_index says {folder_to_index[folder]}"
            )

    def test_folder_to_index_has_ten_entries(self, class_mapping):
        assert len(class_mapping.get("folder_to_index", {})) == 10

    def test_spider_mites_folder_name_has_space(self, class_mapping):
        """The Spider Mites folder name contains a literal space — easy to get wrong."""
        spider_class = class_mapping["classes"]["5"]
        assert " " in spider_class["folder_name"], (
            "Spider Mites folder name should contain a space: "
            "'Tomato___Spider_mites Two-spotted_spider_mite'"
        )

    def test_healthy_folder_name_is_lowercase(self, class_mapping):
        healthy_class = class_mapping["classes"]["9"]
        assert healthy_class["folder_name"] == "Tomato___healthy"

    def test_no_duplicate_folder_names(self, class_mapping):
        folders = [info["folder_name"] for info in class_mapping["classes"].values()]
        assert len(folders) == len(set(folders)), "Duplicate folder names found"

    def test_no_duplicate_display_names(self, class_mapping):
        names = [info["display_name"] for info in class_mapping["classes"].values()]
        assert len(names) == len(set(names)), "Duplicate display names found"


# ─────────────────────────────────────────────────────────────────
# is_image_file tests
# ─────────────────────────────────────────────────────────────────

class TestIsImageFile:
    def test_jpg_is_image(self):
        assert is_image_file(Path("leaf.jpg")) is True

    def test_JPG_uppercase_is_image(self):
        assert is_image_file(Path("leaf.JPG")) is True

    def test_jpeg_is_image(self):
        assert is_image_file(Path("photo.jpeg")) is True

    def test_png_is_image(self):
        assert is_image_file(Path("image.png")) is True

    def test_webp_is_image(self):
        assert is_image_file(Path("scan.webp")) is True

    def test_bmp_is_image(self):
        assert is_image_file(Path("bitmap.bmp")) is True

    def test_txt_is_not_image(self):
        assert is_image_file(Path("notes.txt")) is False

    def test_pdf_is_not_image(self):
        assert is_image_file(Path("doc.pdf")) is False

    def test_no_extension_is_not_image(self):
        assert is_image_file(Path("noextension")) is False


# ─────────────────────────────────────────────────────────────────
# Manifest loading and summarisation
# ─────────────────────────────────────────────────────────────────

def _write_temp_manifest(records: list[dict], tmp_path: Path) -> Path:
    """Helper: write records to a temporary CSV manifest."""
    p = tmp_path / "manifest.csv"
    fieldnames = ["path", "class_index", "class_folder", "group_id"]
    with open(p, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in records:
            writer.writerow({
                "path": r["path"],
                "class_index": r["class_index"],
                "class_folder": r.get("class_folder", ""),
                "group_id": r.get("group_id") or "",
            })
    return p


class TestLoadManifest:
    def test_loads_records(self, tmp_path):
        records = [
            {"path": "/data/img1.jpg", "class_index": 0,
             "class_folder": "Tomato___healthy", "group_id": None},
            {"path": "/data/img2.jpg", "class_index": 1,
             "class_folder": "Tomato___Early_blight", "group_id": "exact_abc"},
        ]
        p = _write_temp_manifest(records, tmp_path)
        loaded = load_manifest(p)
        assert len(loaded) == 2
        assert loaded[0]["path"] == "/data/img1.jpg"
        assert loaded[0]["class_index"] == 0
        assert loaded[0]["group_id"] is None
        assert loaded[1]["group_id"] == "exact_abc"

    def test_empty_group_id_becomes_none(self, tmp_path):
        records = [{"path": "/img.jpg", "class_index": 0,
                    "class_folder": "x", "group_id": None}]
        p = _write_temp_manifest(records, tmp_path)
        loaded = load_manifest(p)
        assert loaded[0]["group_id"] is None


class TestSummariseManifest:
    def test_correct_total(self, tmp_path):
        records = [
            {"path": f"/data/img{i}.jpg", "class_index": i % 3,
             "class_folder": "x", "group_id": None}
            for i in range(9)
        ]
        p = _write_temp_manifest(records, tmp_path)
        summary = summarise_manifest(p)
        assert summary["total"] == 9

    def test_per_class_counts(self, tmp_path):
        records = [
            {"path": "/a.jpg", "class_index": 0, "class_folder": "x", "group_id": None},
            {"path": "/b.jpg", "class_index": 0, "class_folder": "x", "group_id": None},
            {"path": "/c.jpg", "class_index": 1, "class_folder": "y", "group_id": None},
        ]
        p = _write_temp_manifest(records, tmp_path)
        summary = summarise_manifest(p)
        assert summary["per_class"][0] == 2
        assert summary["per_class"][1] == 1


# ─────────────────────────────────────────────────────────────────
# Leakage verification
# ─────────────────────────────────────────────────────────────────

class TestVerifyManifestsNoOverlap:
    def _make_manifest(self, paths: list[str], tmp_path: Path, name: str) -> Path:
        p = tmp_path / name
        records = [{"path": path, "class_index": 0,
                    "class_folder": "x", "group_id": None}
                   for path in paths]
        _write_temp_manifest(records, p.parent)
        # rename the default manifest.csv to the requested name
        (tmp_path / "manifest.csv").rename(p)
        return p

    def test_no_overlap_passes(self, tmp_path):
        a_dir = tmp_path / "a"
        b_dir = tmp_path / "b"
        a_dir.mkdir()
        b_dir.mkdir()

        a = a_dir / "train.csv"
        b = b_dir / "test.csv"

        train_records = [{"path": f"/img{i}.jpg", "class_index": 0,
                          "class_folder": "x", "group_id": None}
                         for i in range(5)]
        test_records = [{"path": f"/img{i}.jpg", "class_index": 0,
                         "class_folder": "x", "group_id": None}
                        for i in range(5, 10)]

        _write_temp_manifest(train_records, a_dir)
        (a_dir / "manifest.csv").rename(a)
        _write_temp_manifest(test_records, b_dir)
        (b_dir / "manifest.csv").rename(b)

        passed, violations = verify_manifests_no_overlap(a, b)
        assert passed is True
        assert violations == []

    def test_overlap_detected(self, tmp_path):
        a_dir = tmp_path / "a"
        b_dir = tmp_path / "b"
        a_dir.mkdir()
        b_dir.mkdir()

        shared = "/shared_img.jpg"
        train_records = [{"path": shared, "class_index": 0,
                          "class_folder": "x", "group_id": None}]
        test_records = [{"path": shared, "class_index": 0,
                         "class_folder": "x", "group_id": None}]

        a = a_dir / "train.csv"
        b = b_dir / "test.csv"
        _write_temp_manifest(train_records, a_dir)
        (a_dir / "manifest.csv").rename(a)
        _write_temp_manifest(test_records, b_dir)
        (b_dir / "manifest.csv").rename(b)

        passed, violations = verify_manifests_no_overlap(a, b)
        assert passed is False
        assert len(violations) == 1
        assert shared in violations[0]

    def test_three_way_no_overlap(self, tmp_path):
        dirs = [tmp_path / f"s{i}" for i in range(3)]
        for d in dirs:
            d.mkdir()

        manifests = []
        for i, d in enumerate(dirs):
            records = [{"path": f"/img{i*5+j}.jpg", "class_index": 0,
                        "class_folder": "x", "group_id": None}
                       for j in range(5)]
            _write_temp_manifest(records, d)
            mp = d / f"split{i}.csv"
            (d / "manifest.csv").rename(mp)
            manifests.append(mp)

        passed, violations = verify_manifests_no_overlap(*manifests)
        assert passed is True
