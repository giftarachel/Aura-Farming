"""
Aura Farming — tests/test_colab_utils.py
==========================================
Unit tests for training/colab_utils.py.

These tests verify the path-remapping logic that allows training
split manifests generated on a Mac to work on Google Colab. They
run without TensorFlow and without the actual PlantVillage dataset.

Tests cover:
  - Path remapping preserves class labels exactly.
  - Path remapping preserves the order of records.
  - Path remapping produces valid absolute paths on Colab.
  - Path remapping handles spaces in filenames.
  - Path remapping raises on missing manifest.
  - Path remapping raises on missing class_folder.
  - verify_remapped_paths reports missing files correctly.
  - check_colab_environment returns expected keys.
  - print_colab_install_commands does not crash.

Run with:
    pytest tests/test_colab_utils.py -v
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from training.colab_utils import (
    remap_manifest_paths,
    verify_remapped_paths,
    check_colab_environment,
    print_colab_install_commands,
    COLAB_REQUIREMENTS,
    COLAB_PREINSTALLED,
)


# ─────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────

def _write_manifest(
    path: Path,
    rows: list[dict],
) -> None:
    """Write a manifest CSV with the standard schema."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["path", "class_index", "class_folder", "group_id"]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                "path": row["path"],
                "class_index": row["class_index"],
                "class_folder": row.get("class_folder", ""),
                "group_id": row.get("group_id", ""),
            })


def _make_mac_path(mac_root: str, class_folder: str, filename: str) -> str:
    """Build an absolute macOS-style path matching the manifest format."""
    return f"{mac_root}/data/plantvillage/color/{class_folder}/{filename}"


# ─────────────────────────────────────────────────────────────────
# remap_manifest_paths tests
# ─────────────────────────────────────────────────────────────────

class TestRemapManifestPaths:
    """Tests for the path-remapping function."""

    MAC_ROOT = "/Users/giftarachel/Aura Farming"
    COLAB_ROOT = "/content/drive/MyDrive/AuraFarming/data/plantvillage/color"

    def _make_manifest(self, tmp_path: Path, rows: list[dict]) -> Path:
        manifest = tmp_path / "train_manifest.csv"
        _write_manifest(manifest, rows)
        return manifest

    def test_returns_paths_and_labels(self, tmp_path):
        rows = [
            {"path": _make_mac_path(self.MAC_ROOT, "Tomato___healthy", "img1.JPG"),
             "class_index": 9, "class_folder": "Tomato___healthy"},
            {"path": _make_mac_path(self.MAC_ROOT, "Tomato___healthy", "img2.JPG"),
             "class_index": 9, "class_folder": "Tomato___healthy"},
        ]
        manifest = self._make_manifest(tmp_path, rows)
        paths, labels = remap_manifest_paths(manifest, Path(self.COLAB_ROOT))
        assert len(paths) == 2
        assert len(labels) == 2

    def test_labels_preserved_exactly(self, tmp_path):
        rows = [
            {"path": _make_mac_path(self.MAC_ROOT, "Tomato___Bacterial_spot", "a.JPG"),
             "class_index": 0, "class_folder": "Tomato___Bacterial_spot"},
            {"path": _make_mac_path(self.MAC_ROOT, "Tomato___healthy", "b.JPG"),
             "class_index": 9, "class_folder": "Tomato___healthy"},
            {"path": _make_mac_path(self.MAC_ROOT, "Tomato___Late_blight", "c.JPG"),
             "class_index": 2, "class_folder": "Tomato___Late_blight"},
        ]
        manifest = self._make_manifest(tmp_path, rows)
        _, labels = remap_manifest_paths(manifest, Path(self.COLAB_ROOT))
        assert labels == [0, 9, 2]

    def test_order_preserved(self, tmp_path):
        rows = [
            {"path": _make_mac_path(self.MAC_ROOT, "Tomato___healthy", f"img{i:03d}.JPG"),
             "class_index": 9, "class_folder": "Tomato___healthy"}
            for i in range(10)
        ]
        manifest = self._make_manifest(tmp_path, rows)
        paths, _ = remap_manifest_paths(manifest, Path(self.COLAB_ROOT))
        for i, p in enumerate(paths):
            assert f"img{i:03d}.JPG" in p

    def test_paths_use_colab_root(self, tmp_path):
        rows = [
            {"path": _make_mac_path(self.MAC_ROOT, "Tomato___healthy", "test.JPG"),
             "class_index": 9, "class_folder": "Tomato___healthy"},
        ]
        manifest = self._make_manifest(tmp_path, rows)
        paths, _ = remap_manifest_paths(manifest, Path(self.COLAB_ROOT))
        assert paths[0].startswith(self.COLAB_ROOT)
        assert self.MAC_ROOT not in paths[0]

    def test_spaces_in_filenames_handled(self, tmp_path):
        """PlantVillage filenames contain spaces (e.g. 'FREC_Scab 3335.JPG')."""
        rows = [
            {"path": _make_mac_path(self.MAC_ROOT, "Tomato___healthy",
                                     "GH_HL Leaf 259.1.JPG"),
             "class_index": 9, "class_folder": "Tomato___healthy"},
        ]
        manifest = self._make_manifest(tmp_path, rows)
        paths, _ = remap_manifest_paths(manifest, Path(self.COLAB_ROOT))
        assert "Leaf 259.1.JPG" in paths[0]
        assert paths[0].startswith(self.COLAB_ROOT)

    def test_windows_paths_handled(self, tmp_path):
        """Windows manifest paths with drive letter and backslashes must remap correctly to POSIX."""
        win_path = r"D:\College Files\Year 2\Aura Farming\data\plantvillage\color\Tomato___healthy\leaf 123.JPG"
        rows = [
            {"path": win_path, "class_index": 9, "class_folder": "Tomato___healthy"},
        ]
        manifest = self._make_manifest(tmp_path, rows)
        paths, _ = remap_manifest_paths(manifest, Path(self.COLAB_ROOT))
        assert "leaf 123.JPG" in paths[0]
        assert paths[0].startswith(self.COLAB_ROOT)
        assert "\\" not in paths[0]
        assert paths[0] == f"{self.COLAB_ROOT}/Tomato___healthy/leaf 123.JPG"

    def test_special_class_folder_with_space(self, tmp_path):
        """Spider mites folder name contains a space."""
        folder = "Tomato___Spider_mites Two-spotted_spider_mite"
        rows = [
            {"path": _make_mac_path(self.MAC_ROOT, folder, "spider.JPG"),
             "class_index": 5, "class_folder": folder},
        ]
        manifest = self._make_manifest(tmp_path, rows)
        paths, _ = remap_manifest_paths(manifest, Path(self.COLAB_ROOT))
        assert "Spider_mites Two-spotted_spider_mite" in paths[0]

    def test_missing_manifest_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="Manifest not found"):
            remap_manifest_paths(tmp_path / "nonexistent.csv", Path(self.COLAB_ROOT))

    def test_empty_class_folder_raises(self, tmp_path):
        """A row with an empty class_folder must raise ValueError."""
        manifest = tmp_path / "train_manifest.csv"
        _write_manifest(manifest, [
            {"path": "/Users/x/data/color/Tomato___healthy/img.JPG",
             "class_index": 9, "class_folder": "", "group_id": ""},
        ])
        with pytest.raises(ValueError, match="empty class_folder"):
            remap_manifest_paths(manifest, Path(self.COLAB_ROOT))

    def test_invalid_class_index_raises(self, tmp_path):
        manifest = tmp_path / "train_manifest.csv"
        _write_manifest(manifest, [
            {"path": _make_mac_path(self.MAC_ROOT, "Tomato___healthy", "x.JPG"),
             "class_index": "not_an_int", "class_folder": "Tomato___healthy",
             "group_id": ""},
        ])
        with pytest.raises(ValueError, match="invalid class_index"):
            remap_manifest_paths(manifest, Path(self.COLAB_ROOT))

    def test_all_ten_classes_remappable(self, tmp_path):
        """Every one of the 10 tomato classes should remap correctly."""
        folders = [
            "Tomato___Bacterial_spot",
            "Tomato___Early_blight",
            "Tomato___Late_blight",
            "Tomato___Leaf_Mold",
            "Tomato___Septoria_leaf_spot",
            "Tomato___Spider_mites Two-spotted_spider_mite",
            "Tomato___Target_Spot",
            "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
            "Tomato___Tomato_mosaic_virus",
            "Tomato___healthy",
        ]
        rows = [
            {"path": _make_mac_path(self.MAC_ROOT, folder, f"img{i}.JPG"),
             "class_index": i, "class_folder": folder, "group_id": ""}
            for i, folder in enumerate(folders)
        ]
        manifest = self._make_manifest(tmp_path, rows)
        paths, labels = remap_manifest_paths(manifest, Path(self.COLAB_ROOT))
        assert len(paths) == 10
        assert labels == list(range(10))
        # Each path must contain its class folder
        for path, folder in zip(paths, folders):
            assert folder in path

    def test_original_manifest_not_modified(self, tmp_path):
        """The function must not write to or modify the input CSV."""
        rows = [
            {"path": _make_mac_path(self.MAC_ROOT, "Tomato___healthy", "x.JPG"),
             "class_index": 9, "class_folder": "Tomato___healthy",
             "group_id": ""},
        ]
        manifest = self._make_manifest(tmp_path, rows)
        original_content = manifest.read_text()

        remap_manifest_paths(manifest, Path(self.COLAB_ROOT))

        assert manifest.read_text() == original_content, \
            "Original manifest CSV was modified"


# ─────────────────────────────────────────────────────────────────
# verify_remapped_paths tests
# ─────────────────────────────────────────────────────────────────

class TestVerifyRemappedPaths:
    """Tests for the path-existence spot-check."""

    def test_all_paths_exist_returns_true(self, tmp_path):
        # Create real files
        paths = []
        for i in range(5):
            p = tmp_path / f"img{i}.jpg"
            p.write_bytes(b"fake")
            paths.append(str(p))
        ok, missing = verify_remapped_paths(paths, sample_size=5, verbose=False)
        assert ok is True
        assert missing == []

    def test_missing_paths_reported(self, tmp_path):
        # Create 3 real, 2 missing
        paths = []
        for i in range(3):
            p = tmp_path / f"real{i}.jpg"
            p.write_bytes(b"fake")
            paths.append(str(p))
        paths.append(str(tmp_path / "missing1.jpg"))
        paths.append(str(tmp_path / "missing2.jpg"))
        ok, missing = verify_remapped_paths(paths, sample_size=5, verbose=False)
        assert ok is False
        assert len(missing) == 2

    def test_sample_size_respected(self, tmp_path):
        """Only the first sample_size paths should be checked."""
        paths = [str(tmp_path / f"img{i}.jpg") for i in range(20)]
        # Make first 10 exist, last 10 missing
        for i in range(10):
            Path(paths[i]).write_bytes(b"x")
        ok, missing = verify_remapped_paths(paths, sample_size=5, verbose=False)
        assert ok is True  # only checked first 5, which exist

    def test_empty_paths_list(self):
        ok, missing = verify_remapped_paths([], sample_size=10, verbose=False)
        assert ok is True
        assert missing == []


# ─────────────────────────────────────────────────────────────────
# check_colab_environment tests
# ─────────────────────────────────────────────────────────────────

class TestCheckColabEnvironment:
    """Tests for the environment reporting function."""

    def test_returns_dict_with_expected_keys(self):
        info = check_colab_environment(verbose=False)
        for key in ["python_version", "tf_version", "gpu_available",
                    "gpu_name", "numpy_version"]:
            assert key in info

    def test_gpu_available_is_bool(self):
        info = check_colab_environment(verbose=False)
        assert isinstance(info["gpu_available"], bool)

    def test_tf_version_is_str_or_none(self):
        info = check_colab_environment(verbose=False)
        assert info["tf_version"] is None or isinstance(info["tf_version"], str)

    def test_does_not_raise_when_tf_missing(self, monkeypatch):
        """Must not crash if TensorFlow cannot be imported."""
        import builtins
        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "tensorflow" or name.startswith("tensorflow."):
                raise ImportError("mocked")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", mock_import)
        info = check_colab_environment(verbose=False)
        assert info["tf_version"] is None
        assert info["gpu_available"] is False


# ─────────────────────────────────────────────────────────────────
# print_colab_install_commands tests
# ─────────────────────────────────────────────────────────────────

class TestPrintColabInstallCommands:
    """Tests for the install-command printer."""

    def test_does_not_crash(self, capsys):
        print_colab_install_commands()
        captured = capsys.readouterr()
        assert "pip install" in captured.out

    def test_excludes_preinstalled_packages(self, capsys):
        """TensorFlow and numpy should NOT appear in install commands."""
        print_colab_install_commands()
        captured = capsys.readouterr()
        # The command line should not contain 'tensorflow' as a package to install
        # (it may appear as part of a comment, but the actual pip command should not)
        assert "tensorflow" not in captured.out.lower().split("pip install")[-1]

    def test_includes_yaml(self, capsys):
        print_colab_install_commands()
        captured = capsys.readouterr()
        assert "PyYAML" in captured.out or "pyyaml" in captured.out.lower()

    def test_includes_sklearn(self, capsys):
        print_colab_install_commands()
        captured = capsys.readouterr()
        assert "scikit-learn" in captured.out


# ─────────────────────────────────────────────────────────────────
# Module constants tests
# ─────────────────────────────────────────────────────────────────

class TestColabConstants:
    """Sanity checks on the package lists."""

    def test_preinstalled_includes_tensorflow(self):
        assert "tensorflow" in COLAB_PREINSTALLED

    def test_preinstalled_includes_numpy(self):
        assert "numpy" in COLAB_PREINSTALLED

    def test_requirements_excludes_preinstalled(self):
        """No package in COLAB_REQUIREMENTS should be in COLAB_PREINSTALLED."""
        for req in COLAB_REQUIREMENTS:
            pkg_name = req.split(">=")[0].split("<")[0].lower().replace("-", "_")
            assert pkg_name not in COLAB_PREINSTALLED, (
                f"{pkg_name} is in COLAB_PREINSTALLED but also in COLAB_REQUIREMENTS"
            )

    def test_requirements_includes_yaml(self):
        assert any("PyYAML" in r for r in COLAB_REQUIREMENTS)

    def test_requirements_includes_sklearn(self):
        assert any("scikit-learn" in r for r in COLAB_REQUIREMENTS)
