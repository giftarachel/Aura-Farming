"""
Aura Farming — tests/test_validate_end_to_end.py
==================================================
Unit tests for the end-to-end validation script (training/validate_end_to_end.py).

Verifies:
  - Fixture mode runs successfully and exercises all pipeline contracts.
  - Test set protection guard halts execution if test manifest is specified.
  - Handling of missing checkpoint paths.
"""

from __future__ import annotations

import json
from pathlib import Path
import pytest

from training.validate_end_to_end import (
    validate_fixture_mode,
    validate_trained_pipeline,
    ValidationReport,
)


class TestValidateEndToEndWorkflow:
    def test_fixture_mode_all_checks_pass(self):
        report = ValidationReport()
        exit_code = validate_fixture_mode(report)
        assert exit_code == 0
        assert report.all_passed is True
        assert len(report.checks) >= 8

    def test_rejection_of_test_manifest_for_calibration(self, tmp_path):
        """Validates that test set is strictly protected and never touched."""
        report = ValidationReport()
        test_manifest = tmp_path / "test_manifest.csv"
        test_manifest.write_text("path,class_index\nfake.jpg,0\n")
        dummy_model = tmp_path / "model.keras"
        dummy_model.write_bytes(b"dummy")

        exit_code = validate_trained_pipeline(
            model_path=dummy_model,
            val_manifest_path=test_manifest,
        )
        assert exit_code == 1

    def test_missing_model_checkpoint_fails_cleanly(self, tmp_path):
        report = ValidationReport()
        val_manifest = tmp_path / "val_manifest.csv"
        val_manifest.write_text("path,class_index\nfake.jpg,0\n")

        exit_code = validate_trained_pipeline(
            model_path=tmp_path / "nonexistent.keras",
            val_manifest_path=val_manifest,
        )
        assert exit_code == 1
