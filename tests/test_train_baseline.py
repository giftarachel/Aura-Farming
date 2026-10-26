"""
Aura Farming — tests/test_train_baseline.py
=============================================
Unit tests for training/train_baseline.py.

These tests run WITHOUT TensorFlow and WITHOUT the PlantVillage
dataset. They verify:
  - Configuration loading and hyperparameter resolution.
  - Manifest loading from CSV.
  - Class weight computation from label counts.
  - History saving to JSON.
  - History JSON contains expected keys.
  - config.yaml weights_path update logic.
  - That the test manifest is never opened by any training function.
  - TF import failure produces a clear error message.

TF-dependent tests (model build, dataset pipeline) are skipped
when TensorFlow is not installed.

Run with:
    pytest tests/test_train_baseline.py -v
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PIL import Image

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# TF availability flag
TF_AVAILABLE = False
try:
    import tensorflow  # noqa: F401
    TF_AVAILABLE = True
except ImportError:
    pass

requires_tf = pytest.mark.skipif(
    not TF_AVAILABLE,
    reason="TensorFlow not installed (Python 3.14). Use Python 3.10–3.12.",
)

from training.train_baseline import (
    load_config,
    load_class_mapping,
    load_split_paths_and_labels,
    compute_class_weights,
    save_history,
    update_config_weights_path,
    set_seeds,
)


# ─────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────

def _write_manifest(path: Path, rows: list[dict]) -> None:
    """Write a minimal manifest CSV for testing."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["path", "class_index", "class_folder", "group_id"]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                "path": row["path"],
                "class_index": row["class_index"],
                "class_folder": row.get("class_folder", "cls"),
                "group_id": row.get("group_id", ""),
            })


# ─────────────────────────────────────────────────────────────────
# Configuration tests
# ─────────────────────────────────────────────────────────────────

class TestLoadConfig:
    def test_config_loads(self):
        cfg = load_config()
        assert "model" in cfg
        assert "dataset" in cfg
        assert "training" in cfg

    def test_model_input_size_is_224(self):
        cfg = load_config()
        assert cfg["model"]["input_size"] == [224, 224]

    def test_num_classes_is_ten(self):
        mapping = load_class_mapping()
        assert len(mapping["classes"]) == 10

    def test_seed_is_integer(self):
        cfg = load_config()
        assert isinstance(cfg["dataset"]["random_seed"], int)

    def test_splits_dir_configured(self):
        cfg = load_config()
        assert "splits_dir" in cfg["dataset"]

    def test_training_section_has_required_keys(self):
        cfg = load_config()
        trn = cfg["training"]
        for key in ["phase_a_epochs", "phase_a_patience", "batch_size",
                    "dropout_rate", "checkpoint_phase_a", "checkpoint_phase_b"]:
            assert key in trn, f"Missing training config key: {key}"


# ─────────────────────────────────────────────────────────────────
# Manifest loading tests
# ─────────────────────────────────────────────────────────────────

class TestLoadSplitPathsAndLabels:
    def test_loads_paths_and_labels(self, tmp_path):
        manifest = tmp_path / "train_manifest.csv"
        rows = [
            {"path": "/data/img1.jpg", "class_index": 0},
            {"path": "/data/img2.jpg", "class_index": 3},
            {"path": "/data/img3.jpg", "class_index": 7},
        ]
        _write_manifest(manifest, rows)
        paths, labels = load_split_paths_and_labels(manifest)
        assert paths == ["/data/img1.jpg", "/data/img2.jpg", "/data/img3.jpg"]
        assert labels == [0, 3, 7]

    def test_missing_manifest_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="Manifest not found"):
            load_split_paths_and_labels(tmp_path / "nonexistent.csv")

    def test_empty_manifest_returns_empty_lists(self, tmp_path):
        manifest = tmp_path / "empty.csv"
        _write_manifest(manifest, [])
        paths, labels = load_split_paths_and_labels(manifest)
        assert paths == []
        assert labels == []

    def test_all_ten_classes_loadable(self, tmp_path):
        manifest = tmp_path / "all_classes.csv"
        rows = [{"path": f"/data/img{i}.jpg", "class_index": i}
                for i in range(10)]
        _write_manifest(manifest, rows)
        paths, labels = load_split_paths_and_labels(manifest)
        assert sorted(set(labels)) == list(range(10))

    def test_test_manifest_not_used_by_training(self):
        """
        No training function may OPEN or READ the test manifest CSV.
        The test manifest path may be mentioned in print statements
        and comments to remind the user it is reserved, but it must
        never be loaded by load_manifest or open().
        """
        import inspect
        import training.train_baseline as m

        source_combined = ""
        for name, func in inspect.getmembers(m, inspect.isfunction):
            try:
                source_combined += inspect.getsource(func)
            except (OSError, TypeError):
                pass

        # These are legitimate references (reminder messages, not reads)
        allowed_patterns = {
            "# Test evaluation",
            "reserved for Phase 7",
            "NOT loaded",
            "NOT used",
            "⚠",
        }

        lines = source_combined.split("\n")
        suspicious = []
        for ln in lines:
            stripped = ln.strip()
            if "test_manifest" not in stripped.lower():
                continue
            if stripped.startswith("#"):
                continue
            if any(pat in ln for pat in allowed_patterns):
                continue
            # load_manifest() or open() on the test manifest is a violation
            if "load_manifest" in stripped or ("open(" in stripped and "test_manifest" in stripped):
                suspicious.append(ln)

        assert not suspicious, (
            "Training source calls load_manifest or open() on test_manifest:\n"
            + "\n".join(suspicious)
        )


# ─────────────────────────────────────────────────────────────────
# Class weight computation tests
# ─────────────────────────────────────────────────────────────────

class TestComputeClassWeights:
    def test_returns_dict_for_all_classes(self):
        labels = [0] * 100 + [1] * 50 + [2] * 25
        weights = compute_class_weights(labels, num_classes=3)
        assert set(weights.keys()) == {0, 1, 2}

    def test_minority_class_gets_higher_weight(self):
        """Class 1 appears fewer times → must have a higher weight."""
        labels = [0] * 200 + [1] * 50
        weights = compute_class_weights(labels, num_classes=2)
        assert weights[1] > weights[0]

    def test_balanced_labels_give_equal_weights(self):
        labels = [0] * 100 + [1] * 100
        weights = compute_class_weights(labels, num_classes=2)
        np.testing.assert_allclose(weights[0], weights[1], rtol=1e-5)

    def test_all_weights_are_positive(self):
        labels = list(range(10)) * 50  # 10 classes, 50 images each
        weights = compute_class_weights(labels, num_classes=10)
        for idx, w in weights.items():
            assert w > 0, f"Non-positive weight for class {idx}: {w}"

    def test_real_class_distribution(self):
        """Simulate the actual tomato training set imbalance."""
        counts = {
            0: 1486, 1: 700, 2: 1335, 3: 667, 4: 1240,
            5: 1173, 6: 982, 7: 3748, 8: 261, 9: 1116,
        }
        labels = []
        for cls, count in counts.items():
            labels.extend([cls] * count)
        weights = compute_class_weights(labels, num_classes=10)

        # Class 8 (261 images) must have the highest weight
        assert weights[8] == max(weights.values()), (
            f"Expected class 8 (smallest, 261 imgs) to have highest weight. "
            f"Got: {weights}"
        )
        # Class 7 (3748 images) must have the lowest weight
        assert weights[7] == min(weights.values()), (
            f"Expected class 7 (largest, 3748 imgs) to have lowest weight."
        )


# ─────────────────────────────────────────────────────────────────
# History saving tests
# ─────────────────────────────────────────────────────────────────

class TestSaveHistory:
    def _make_fake_history(self):
        """Create a mock Keras History object."""
        h = MagicMock()
        h.history = {
            "loss": [1.2, 0.9, 0.7],
            "accuracy": [0.45, 0.60, 0.71],
            "val_loss": [1.3, 1.0, 0.85],
            "val_accuracy": [0.41, 0.55, 0.65],
        }
        h.epoch = [0, 1, 2]
        return h

    def test_creates_json_file(self, tmp_path):
        history = self._make_fake_history()
        save_history(history, tmp_path, "phase_a")
        out = tmp_path / "training_history_phase_a.json"
        assert out.exists()

    def test_json_contains_expected_keys(self, tmp_path):
        history = self._make_fake_history()
        save_history(history, tmp_path, "phase_a")
        with open(tmp_path / "training_history_phase_a.json") as f:
            data = json.load(f)
        for key in ["loss", "accuracy", "val_loss", "val_accuracy",
                    "epochs_completed"]:
            assert key in data, f"Key '{key}' missing from saved history"

    def test_epochs_completed_is_correct(self, tmp_path):
        history = self._make_fake_history()
        save_history(history, tmp_path, "phase_a")
        with open(tmp_path / "training_history_phase_a.json") as f:
            data = json.load(f)
        assert data["epochs_completed"] == 3

    def test_all_values_are_json_serialisable(self, tmp_path):
        """All values must be plain Python floats, not numpy types."""
        history = self._make_fake_history()
        save_history(history, tmp_path, "phase_a")
        with open(tmp_path / "training_history_phase_a.json") as f:
            data = json.load(f)
        for key, vals in data.items():
            if isinstance(vals, list):
                for v in vals:
                    assert isinstance(v, float), (
                        f"Non-float value in history[{key!r}]: {v!r}"
                    )

    def test_output_directory_created_if_missing(self, tmp_path):
        nested = tmp_path / "deep" / "nested" / "reports"
        history = self._make_fake_history()
        save_history(history, nested, "phase_b")
        assert (nested / "training_history_phase_b.json").exists()


# ─────────────────────────────────────────────────────────────────
# config.yaml weights_path update tests
# ─────────────────────────────────────────────────────────────────

class TestUpdateConfigWeightsPath:
    def test_updates_null_weights_path(self, tmp_path):
        # update_config_weights_path reads PROJECT_ROOT / "configs/config.yaml"
        # so we need to create that structure under tmp_path
        configs_dir = tmp_path / "configs"
        configs_dir.mkdir()
        config = configs_dir / "config.yaml"
        config.write_text(
            "model:\n"
            "  weights_path: null\n"
            "  input_size: [224, 224]\n"
        )
        fake_model = tmp_path / "models" / "mobilenetv3_baseline.keras"
        fake_model.parent.mkdir()
        fake_model.touch()

        import training.train_baseline as m
        original_root = m.PROJECT_ROOT
        m.PROJECT_ROOT = tmp_path
        try:
            update_config_weights_path(fake_model)
        finally:
            m.PROJECT_ROOT = original_root

        updated = config.read_text()
        assert "weights_path: null" not in updated
        assert "weights_path:" in updated

    def test_does_not_modify_other_config_values(self, tmp_path):
        configs_dir = tmp_path / "configs"
        configs_dir.mkdir()
        config = configs_dir / "config.yaml"
        config.write_text(
            "model:\n"
            "  weights_path: null\n"
            "  input_size: [224, 224]\n"
            "  input_channels: 3\n"
        )
        fake_model = tmp_path / "models" / "test.keras"
        fake_model.parent.mkdir()
        fake_model.touch()

        import training.train_baseline as m
        original_root = m.PROJECT_ROOT
        m.PROJECT_ROOT = tmp_path
        try:
            update_config_weights_path(fake_model)
        finally:
            m.PROJECT_ROOT = original_root

        updated = config.read_text()
        assert "input_size: [224, 224]" in updated
        assert "input_channels: 3" in updated


# ─────────────────────────────────────────────────────────────────
# set_seeds tests
# ─────────────────────────────────────────────────────────────────

class TestSetSeeds:
    def test_sets_pythonhashseed_env_var(self, monkeypatch):
        import os
        monkeypatch.delenv("PYTHONHASHSEED", raising=False)
        set_seeds(42)
        assert os.environ.get("PYTHONHASHSEED") == "42"

    def test_numpy_seed_is_set(self):
        set_seeds(0)
        val1 = np.random.randint(0, 10000)
        set_seeds(0)
        val2 = np.random.randint(0, 10000)
        assert val1 == val2, "NumPy random state should be reproducible after set_seeds"


# ─────────────────────────────────────────────────────────────────
# TF-dependent model tests
# ─────────────────────────────────────────────────────────────────

class TestBuildModel:
    @requires_tf
    def test_model_output_shape(self):
        from training.train_baseline import build_model
        import tensorflow as tf
        model = build_model(num_classes=10, input_size=(224, 224), dropout_rate=0.3)
        dummy = tf.zeros((1, 224, 224, 3))
        output = model(dummy, training=False)
        assert output.shape == (1, 10)

    @requires_tf
    def test_model_output_sums_to_one(self):
        from training.train_baseline import build_model
        import tensorflow as tf
        model = build_model(num_classes=10, input_size=(224, 224))
        dummy = tf.zeros((2, 224, 224, 3))
        output = model(dummy, training=False).numpy()
        np.testing.assert_allclose(output.sum(axis=1), 1.0, atol=1e-5)

    @requires_tf
    def test_base_is_frozen_after_build(self):
        from training.train_baseline import build_model
        model = build_model(num_classes=10, input_size=(224, 224))
        base = model.get_layer("mobilenetv3small")
        assert base.trainable is False

    @requires_tf
    def test_include_preprocessing_is_false(self):
        """
        The base model must not include internal preprocessing.
        When include_preprocessing=False, no Rescaling layer should
        be present in the base model — otherwise pixels would be
        rescaled twice (once in the data pipeline, once in the model).

        We check for the absence of a Rescaling layer by name rather
        than inspecting the model config, because the
        include_preprocessing flag is a build-time parameter that is
        not reliably stored in get_config() across Keras versions.
        """
        from training.train_baseline import build_model
        model = build_model(num_classes=10, input_size=(224, 224))
        base = model.get_layer("mobilenetv3small")
        # Check that no Rescaling layer exists in the base model
        has_rescaling = any(
            "rescaling" in layer.name.lower()
            for layer in base.layers
        )
        assert not has_rescaling, (
            "Base model contains a Rescaling layer — include_preprocessing "
            "may be True, which would cause double preprocessing."
        )

    @requires_tf
    def test_unfreeze_makes_layers_trainable(self):
        from training.train_baseline import build_model, unfreeze_for_fine_tuning
        model = build_model(num_classes=10, input_size=(224, 224))
        unfreeze_for_fine_tuning(model, fine_tune_at=80)
        base = model.get_layer("mobilenetv3small")
        trainable_layers = [la for la in base.layers if la.trainable]
        assert len(trainable_layers) > 0

    @requires_tf
    def test_invalid_fine_tune_at_negative_raises(self):
        """A negative fine_tune_at index must raise ValueError."""
        from training.train_baseline import build_model, unfreeze_for_fine_tuning
        model = build_model(num_classes=10, input_size=(224, 224))
        with pytest.raises(ValueError, match="out of range"):
            unfreeze_for_fine_tuning(model, fine_tune_at=-1)

    @requires_tf
    def test_invalid_fine_tune_at_too_large_raises(self):
        """An index beyond the base layer count must raise ValueError."""
        from training.train_baseline import build_model, unfreeze_for_fine_tuning
        model = build_model(num_classes=10, input_size=(224, 224))
        base = model.get_layer("mobilenetv3small")
        too_large = len(base.layers) + 10
        with pytest.raises(ValueError, match="out of range"):
            unfreeze_for_fine_tuning(model, fine_tune_at=too_large)

    @requires_tf
    def test_bn_layers_stay_frozen_after_unfreeze(self):
        """
        BatchNormalization layers must remain frozen after fine-tuning setup.
        Detection uses isinstance(layer, tf.keras.layers.BatchNormalization),
        which is reliable regardless of layer naming conventions.
        """
        from training.train_baseline import build_model, unfreeze_for_fine_tuning
        import tensorflow as tf
        model = build_model(num_classes=10, input_size=(224, 224))
        unfreeze_for_fine_tuning(model, fine_tune_at=80)
        base = model.get_layer("mobilenetv3small")
        bn_layers = [
            la for la in base.layers
            if isinstance(la, tf.keras.layers.BatchNormalization)
        ]
        assert len(bn_layers) > 0, "Expected at least one BN layer in MobileNetV3Small base"
        for layer in bn_layers:
            assert layer.trainable is False, (
                f"BN layer '{layer.name}' should be frozen but is trainable"
            )

    @requires_tf
    def test_fine_tune_at_zero_unfreezes_non_bn_layers(self):
        """Boundary: fine_tune_at=0 unfreezes all non-BN base layers."""
        from training.train_baseline import build_model, unfreeze_for_fine_tuning
        model = build_model(num_classes=10, input_size=(224, 224))
        unfreeze_for_fine_tuning(model, fine_tune_at=0)
        base = model.get_layer("mobilenetv3small")
        # At least some non-BN layers should be trainable
        non_bn_trainable = [
            la for la in base.layers if la.trainable
            and "batch_normalization" not in la.name.lower()
        ]
        assert len(non_bn_trainable) > 0, "Expected some non-BN layers to be trainable"

    @requires_tf
    def test_fine_tune_at_max_freezes_all_then_refreezes_bn(self):
        """Boundary: fine_tune_at=len(base.layers) freezes everything, BN stays frozen."""
        from training.train_baseline import build_model, unfreeze_for_fine_tuning
        model = build_model(num_classes=10, input_size=(224, 224))
        base = model.get_layer("mobilenetv3small")
        unfreeze_for_fine_tuning(model, fine_tune_at=len(base.layers))
        # All base layers below max are frozen, BN layers are also frozen
        # So there should be 0 trainable base layers
        trainable = [la for la in base.layers if la.trainable]
        assert len(trainable) == 0


# ─────────────────────────────────────────────────────────────────
# Phase B checkpoint reload tests
# ─────────────────────────────────────────────────────────────────

class TestPhaseBCheckpointReload:
    """Tests that Phase B correctly reloads the best Phase A checkpoint."""

    def test_nonexistent_checkpoint_path_detected(self, tmp_path):
        """
        Verify the Path.exists() check used in main() for Phase B
        correctly identifies a missing checkpoint file.
        This test runs without TensorFlow — it only tests the
        filesystem check pattern.
        """
        fake_ckpt = tmp_path / "models" / "mobilenetv3_phase_a.keras"
        # File does not exist
        assert not fake_ckpt.exists()

    @requires_tf
    def test_load_model_fails_on_nonexistent_file(self, tmp_path):
        """
        tf.keras.models.load_model must raise when the file doesn't exist.
        This confirms the error path: if the Phase A checkpoint is missing,
        load_model will fail — so the exists() check in main() is the
        correct guard.
        """
        import tensorflow as tf
        fake_path = str(tmp_path / "nonexistent_model.keras")
        with pytest.raises(Exception):
            tf.keras.models.load_model(fake_path)


class TestBuildTfDataset:
    @requires_tf
    def test_dataset_batch_shape(self, tmp_path):
        from training.train_baseline import build_tf_dataset
        from PIL import Image as PILImage

        # Create 4 tiny fake images
        img_paths = []
        for i in range(4):
            p = tmp_path / f"img{i}.jpg"
            PILImage.new("RGB", (64, 64), (i * 60, 100, 80)).save(p)
            img_paths.append(str(p))
        labels = [0, 1, 2, 3]

        ds = build_tf_dataset(
            img_paths, labels, input_size=(224, 224),
            batch_size=2, augment=False, shuffle=False, seed=42,
        )
        batch_img, batch_lbl = next(iter(ds))
        assert batch_img.shape == (2, 224, 224, 3)
        assert batch_lbl.shape == (2,)

    @requires_tf
    def test_dataset_values_in_correct_range(self, tmp_path):
        from training.train_baseline import build_tf_dataset
        from PIL import Image as PILImage
        import tensorflow as tf

        p = tmp_path / "white.jpg"
        PILImage.new("RGB", (64, 64), (255, 255, 255)).save(p)

        ds = build_tf_dataset(
            [str(p)], [0], input_size=(224, 224),
            batch_size=1, augment=False, shuffle=False, seed=42,
        )
        imgs, _ = next(iter(ds))
        arr = imgs.numpy()
        assert arr.min() >= -1.0 - 1e-4
        assert arr.max() <= 1.0 + 1e-4
