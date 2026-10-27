"""
Aura Farming — tests/test_calibration.py
=========================================
Unit tests for probability calibration and abstention (src/calibration.py).

Covers:
  - Logit-to-probability and probability-to-logit conversions
  - Temperature scaling: T > 1 softens, T < 1 sharpens, T = 1 identity
  - Probability normalization
  - Numerical stability (NaN, Inf, negative temperature)
  - Missing calibration artifact handling and graceful fallback
  - Calibration artifact persistence (save & load JSON)
  - Reliability & Abstention policy:
    * High confidence -> CONFIDENT
    * Borderline confidence -> UNCERTAIN
    * Very low confidence -> ABSTAINED
    * Top-2 margin tie -> UNCERTAIN
    * Image quality failure integration -> ABSTAINED
  - Calibration metrics: ECE, NLL, Brier score

Run with:
    pytest tests/test_calibration.py -v
"""

from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pytest

from src.calibration import (
    apply_temperature_scaling,
    apply_calibration_and_threshold,
    fit_temperature_scaling,
    compute_ece,
    compute_nll,
    compute_brier_score,
    is_probability_distribution,
    probabilities_to_logits,
    stable_softmax,
    save_calibration_artifact,
    load_calibration_artifact,
    PredictionResult,
    ReliabilityStatus,
)
from src.image_quality import QualityReport, QualityStatus


@pytest.fixture
def mock_class_names() -> dict[int, str]:
    return {
        0: "Tomato — Bacterial Spot",
        1: "Tomato — Early Blight",
        2: "Tomato — Late Blight",
        3: "Tomato — Leaf Mold",
        4: "Tomato — Septoria Leaf Spot",
        5: "Tomato — Spider Mites",
        6: "Tomato — Target Spot",
        7: "Tomato — Yellow Leaf Curl Virus",
        8: "Tomato — Mosaic Virus",
        9: "Tomato — Healthy",
    }


class TestProbabilityMath:
    def test_is_probability_distribution_valid(self):
        p = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
        assert is_probability_distribution(p) is True

    def test_is_probability_distribution_invalid_sum(self):
        p = np.array([0.1, 0.2, 0.3, 0.9], dtype=np.float32)
        assert is_probability_distribution(p) is False

    def test_is_probability_distribution_negative(self):
        p = np.array([-0.1, 0.5, 0.6], dtype=np.float32)
        assert is_probability_distribution(p) is False

    def test_stable_softmax_sums_to_one(self):
        logits = np.array([2.0, 1.0, 0.1, -1.0])
        probs = stable_softmax(logits, temperature=1.0)
        assert np.isclose(np.sum(probs), 1.0, atol=1e-5)
        assert np.all(probs >= 0.0)

    def test_stable_softmax_negative_temperature_raises(self):
        logits = np.array([1.0, 2.0])
        with pytest.raises(ValueError, match="strictly positive"):
            stable_softmax(logits, temperature=-0.5)

    def test_stable_softmax_nan_input_raises(self):
        logits = np.array([1.0, np.nan, 2.0])
        with pytest.raises(ValueError, match="NaN or Infinite"):
            stable_softmax(logits, temperature=1.0)

    def test_probabilities_to_logits_inversion(self):
        orig_p = np.array([0.05, 0.70, 0.15, 0.10], dtype=np.float64)
        logits = probabilities_to_logits(orig_p)
        recovered_p = stable_softmax(logits, temperature=1.0)
        np.testing.assert_allclose(orig_p, recovered_p, atol=1e-5)


class TestTemperatureScaling:
    def test_temperature_one_is_identity_for_probs(self):
        p = np.array([0.1, 0.6, 0.2, 0.1], dtype=np.float32)
        scaled = apply_temperature_scaling(p, temperature=1.0)
        np.testing.assert_allclose(p, scaled, atol=1e-5)

    def test_temperature_greater_than_one_softens(self):
        p = np.array([0.05, 0.85, 0.05, 0.05], dtype=np.float32)
        scaled = apply_temperature_scaling(p, temperature=2.0)
        # Top probability should decrease (less overconfident)
        assert scaled[1] < p[1]
        # Argmax should be preserved
        assert np.argmax(scaled) == np.argmax(p)
        assert np.isclose(np.sum(scaled), 1.0, atol=1e-5)

    def test_temperature_less_than_one_sharpens(self):
        p = np.array([0.2, 0.5, 0.2, 0.1], dtype=np.float32)
        scaled = apply_temperature_scaling(p, temperature=0.5)
        # Top probability should increase
        assert scaled[1] > p[1]
        assert np.argmax(scaled) == np.argmax(p)
        assert np.isclose(np.sum(scaled), 1.0, atol=1e-5)

    def test_2d_batch_temperature_scaling(self):
        batch = np.array([
            [0.1, 0.8, 0.1],
            [0.3, 0.4, 0.3],
        ], dtype=np.float32)
        scaled = apply_temperature_scaling(batch, temperature=1.5)
        assert scaled.shape == (2, 3)
        np.testing.assert_allclose(np.sum(scaled, axis=1), [1.0, 1.0], atol=1e-5)


class TestMetrics:
    def test_perfect_calibration_ece_is_zero(self):
        # 10 samples all predicted with 1.0 confidence and all correct
        probs = np.zeros((10, 2), dtype=np.float32)
        probs[:, 0] = 1.0
        labels = np.zeros(10, dtype=np.int64)
        ece = compute_ece(probs, labels, num_bins=10)
        assert np.isclose(ece, 0.0, atol=1e-5)

    def test_overconfident_ece_is_positive(self):
        # 10 samples predicted with 0.9 confidence for class 0, but only 50% accurate
        probs = np.zeros((10, 2), dtype=np.float32)
        probs[:, 0] = 0.9
        probs[:, 1] = 0.1
        labels = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1], dtype=np.int64)
        ece = compute_ece(probs, labels, num_bins=10)
        # Accuracy is 0.5, confidence is 0.9 -> difference ~ 0.4
        assert np.isclose(ece, 0.4, atol=1e-4)

    def test_compute_nll(self):
        probs = np.array([[0.8, 0.2], [0.1, 0.9]], dtype=np.float32)
        labels = np.array([0, 1], dtype=np.int64)
        nll = compute_nll(probs, labels)
        expected = - (np.log(0.8) + np.log(0.9)) / 2.0
        assert np.isclose(nll, expected, atol=1e-5)

    def test_compute_brier_score(self):
        probs = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        labels = np.array([0, 1], dtype=np.int64)
        brier = compute_brier_score(probs, labels)
        assert np.isclose(brier, 0.0, atol=1e-5)


class TestArtifactPersistence:
    def test_save_and_load_roundtrip(self, tmp_path: Path):
        calib_file = tmp_path / "test_calibration.json"
        save_calibration_artifact(
            filepath=calib_file,
            temperature=1.45,
            metrics_before={"ece": 0.15, "nll": 1.2, "brier": 0.3},
            metrics_after={"ece": 0.04, "nll": 0.9, "brier": 0.2},
            metadata={"model": "MobileNetV3Small"},
        )
        assert calib_file.is_file()

        loaded = load_calibration_artifact(calib_file)
        assert loaded["is_calibrated"] is True
        assert np.isclose(loaded["temperature"], 1.45, atol=1e-4)
        assert loaded["metrics_after"]["ece"] == 0.04

    def test_missing_artifact_graceful_fallback(self, tmp_path: Path):
        missing = tmp_path / "does_not_exist.json"
        loaded = load_calibration_artifact(missing)
        assert loaded["is_calibrated"] is False
        assert loaded["temperature"] == 1.0


class TestAbstentionPolicy:
    def test_confident_prediction(self, mock_class_names):
        # 10 classes, index 1 has 90% confidence, tested at T=1.0
        probs = np.array([0.01, 0.90, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.02])
        res = apply_calibration_and_threshold(
            raw_probabilities=probs,
            class_names=mock_class_names,
            temperature=1.0,
            confidence_threshold=0.70,
            abstention_threshold=0.40,
        )
        assert res.reliability == ReliabilityStatus.CONFIDENT
        assert res.class_index == 1
        assert res.class_name == "Tomato — Early Blight"
        assert res.top_probability >= 0.70
        assert res.is_placeholder is False

    def test_uncertain_prediction_due_to_moderate_confidence(self, mock_class_names):
        # Index 2 has 55% confidence (between 0.40 and 0.70) at T=1.0
        probs = np.array([0.05, 0.05, 0.55, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05])
        res = apply_calibration_and_threshold(
            raw_probabilities=probs,
            class_names=mock_class_names,
            temperature=1.0,
            confidence_threshold=0.70,
            abstention_threshold=0.40,
        )
        assert res.reliability == ReliabilityStatus.UNCERTAIN
        assert res.class_index == 2
        assert "moderate confidence" in res.explanation.lower()

    def test_abstained_due_to_low_confidence(self, mock_class_names):
        # Top confidence is only 25% (< 0.40)
        probs = np.ones(10) / 10.0
        probs[0] = 0.25
        probs = probs / np.sum(probs)
        res = apply_calibration_and_threshold(
            raw_probabilities=probs,
            class_names=mock_class_names,
            confidence_threshold=0.70,
            abstention_threshold=0.40,
        )
        assert res.reliability == ReliabilityStatus.ABSTAINED
        assert res.class_index is None
        assert res.class_name is None
        assert "below the minimum reliability threshold" in res.explanation.lower()

    def test_uncertain_due_to_narrow_margin(self, mock_class_names):
        # Top 1 = 0.72, Top 2 = 0.70 (margin = 0.02 < margin_threshold 0.05)
        probs = np.zeros(10)
        probs[3] = 0.45
        probs[4] = 0.43
        probs = probs / np.sum(probs)
        res = apply_calibration_and_threshold(
            raw_probabilities=probs,
            class_names=mock_class_names,
            confidence_threshold=0.40,
            margin_threshold=0.05,
        )
        assert res.reliability == ReliabilityStatus.UNCERTAIN
        assert "narrow" in res.explanation.lower()

    def test_abstained_due_to_failed_quality_report(self, mock_class_names):
        probs = np.zeros(10)
        probs[9] = 0.99
        probs = probs / np.sum(probs)
        qr = QualityReport(
            passed=False,
            failure_reasons=["Photograph is completely blurry."],
            status=QualityStatus.FAILED,
            feedback="Please retake photo.",
            can_proceed=False,
        )
        res = apply_calibration_and_threshold(
            raw_probabilities=probs,
            class_names=mock_class_names,
            quality_report=qr,
        )
        assert res.reliability == ReliabilityStatus.ABSTAINED
        assert res.class_index is None
        assert "quality failure" in res.explanation.lower()
