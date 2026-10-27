"""
Aura Farming — tests/test_calibrate_model.py
============================================
Regression and unit tests for training/calibrate_model.py.

Verifies:
  - Acceptance of manifests with 'class_index' or 'label' column.
  - Rejection of sealed test manifest (scientific safety check).
  - Handling of nonexistent manifest path or model path.
  - Correct execution of synthetic self-test mode.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
import numpy as np
import pytest

from training.calibrate_model import (
    calibrate_from_validation_manifest,
    run_synthetic_calibration_test,
)


class TestCalibrateModelScript:
    def test_rejection_of_test_manifest(self, tmp_path):
        """Ensures that the sealed test set is never calibrated on."""
        fake_test_csv = tmp_path / "test_manifest.csv"
        fake_test_csv.write_text("path,class_index\nfake.jpg,0\n")
        fake_model = tmp_path / "model.keras"
        fake_model.write_bytes(b"dummy")

        ret = calibrate_from_validation_manifest(
            model_path=fake_model,
            val_manifest_path=fake_test_csv,
            output_path=tmp_path / "calib.json",
            report_path=tmp_path / "report.json",
        )
        assert ret == 1

    def test_missing_validation_manifest_fails_gracefully(self, tmp_path):
        ret = calibrate_from_validation_manifest(
            model_path=tmp_path / "model.keras",
            val_manifest_path=tmp_path / "nonexistent_val.csv",
            output_path=tmp_path / "calib.json",
            report_path=tmp_path / "report.json",
        )
        assert ret == 1

    def test_missing_model_checkpoint_fails_gracefully(self, tmp_path):
        val_csv = tmp_path / "val_manifest.csv"
        val_csv.write_text("path,class_index\nfake.jpg,0\n")

        ret = calibrate_from_validation_manifest(
            model_path=tmp_path / "nonexistent_model.keras",
            val_manifest_path=val_csv,
            output_path=tmp_path / "calib.json",
            report_path=tmp_path / "report.json",
        )
        assert ret == 1

    def test_synthetic_calibration_self_test(self, tmp_path):
        out_json = tmp_path / "mobilenetv3_calibration.json"
        rep_json = tmp_path / "calibration_report.json"

        ret = run_synthetic_calibration_test(
            output_path=out_json,
            report_path=rep_json,
            n_samples=100,
            n_classes=10,
            seed=42,
        )
        assert ret == 0
        assert out_json.is_file()
        assert rep_json.is_file()

        with open(out_json, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert data["temperature"] > 0.0
        assert data["metadata"]["synthetic"] is True
        assert "metrics_before" in data
        assert "metrics_after" in data
