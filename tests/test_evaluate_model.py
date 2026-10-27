"""
Aura Farming — tests/test_evaluate_model.py
============================================
Unit tests for the Phase 7 scientific evaluation script (training/evaluate_model.py).

Verifies:
  - Classification metrics calculation (accuracy, macro/weighted F1, ROC-AUC).
  - Calibration and reliability analysis (ECE, NLL, Brier, selective accuracy).
  - 3-tier abstention policy logic.
  - Graceful handling of PlantDoc dataset presence vs. absence.
  - Scientific Markdown report synthesis.
"""

from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pytest

from training.evaluate_model import (
    compute_classification_metrics,
    compute_reliability_analysis,
    evaluate_plantdoc_generalization,
    write_scientific_evaluation_report,
)


@pytest.fixture
def sample_metrics_data():
    """Generates synthetic 10-class evaluation vectors for deterministic testing."""
    np.random.seed(42)
    n_samples = 100
    n_classes = 10
    class_names = {i: f"Class {i}" for i in range(n_classes)}

    y_true = np.random.randint(0, n_classes, size=n_samples)
    # Generate realistic uncalibrated probabilities with slight overconfidence
    raw_logits = np.random.randn(n_samples, n_classes)
    # Bias logits towards true class for ~80% accuracy
    for idx, true_c in enumerate(y_true):
        if idx % 5 != 0:
            raw_logits[idx, true_c] += 3.0

    exp_l = np.exp(raw_logits - np.max(raw_logits, axis=1, keepdims=True))
    probs_uncal = exp_l / np.sum(exp_l, axis=1, keepdims=True)

    temperature = 0.85
    exp_cal = np.exp(raw_logits / temperature - np.max(raw_logits / temperature, axis=1, keepdims=True))
    probs_cal = exp_cal / np.sum(exp_cal, axis=1, keepdims=True)

    y_pred = np.argmax(probs_cal, axis=1)

    return {
        "y_true": y_true,
        "y_pred": y_pred,
        "probs_uncal": probs_uncal,
        "probs_cal": probs_cal,
        "class_names": class_names,
        "temperature": temperature,
        "n_samples": n_samples,
    }


class TestComputeClassificationMetrics:
    def test_metrics_keys_and_ranges(self, sample_metrics_data):
        d = sample_metrics_data
        metrics = compute_classification_metrics(
            y_true=d["y_true"],
            y_pred=d["y_pred"],
            y_prob=d["probs_uncal"],
            class_names=d["class_names"],
        )

        assert "accuracy" in metrics
        assert "macro_f1" in metrics
        assert "macro_precision" in metrics
        assert "macro_recall" in metrics
        assert "roc_auc_macro" in metrics
        assert "per_class" in metrics
        assert "confusion_matrix" in metrics

        assert 0.0 <= metrics["accuracy"] <= 1.0
        assert 0.0 <= metrics["macro_f1"] <= 1.0
        assert 0.0 <= metrics["roc_auc_macro"] <= 1.0
        assert len(metrics["per_class"]) == 10

    def test_confusion_matrix_shape_and_sum(self, sample_metrics_data):
        d = sample_metrics_data
        metrics = compute_classification_metrics(
            y_true=d["y_true"],
            y_pred=d["y_pred"],
            y_prob=d["probs_uncal"],
            class_names=d["class_names"],
        )

        cm = np.array(metrics["confusion_matrix"])
        assert cm.shape == (10, 10)
        assert np.sum(cm) == d["n_samples"]

    def test_per_class_support_sums_to_total(self, sample_metrics_data):
        d = sample_metrics_data
        metrics = compute_classification_metrics(
            y_true=d["y_true"],
            y_pred=d["y_pred"],
            y_prob=d["probs_uncal"],
            class_names=d["class_names"],
        )

        total_support = sum(info["support"] for info in metrics["per_class"].values())
        assert total_support == d["n_samples"]


class TestComputeReliabilityAnalysis:
    def test_reliability_structure(self, sample_metrics_data):
        d = sample_metrics_data
        rel = compute_reliability_analysis(
            y_true=d["y_true"],
            probs_uncal=d["probs_uncal"],
            probs_cal=d["probs_cal"],
            temperature=d["temperature"],
            confidence_threshold=0.70,
            abstention_threshold=0.40,
            margin_threshold=0.05,
        )

        assert "uncalibrated_metrics" in rel
        assert "calibrated_metrics" in rel
        assert "abstention_policy" in rel
        assert "confidence_distribution" in rel

        uncal = rel["uncalibrated_metrics"]
        cal = rel["calibrated_metrics"]
        assert "ece" in uncal and "nll" in uncal and "brier_score" in uncal
        assert "ece" in cal and "nll" in cal and "brier_score" in cal

        policy = rel["abstention_policy"]
        counts = policy["counts"]
        assert counts["confident"] + counts["uncertain"] + counts["abstained"] == d["n_samples"]
        assert counts["retained"] == counts["confident"] + counts["uncertain"]

        proportions = policy["proportions"]
        assert 0.0 <= proportions["coverage"] <= 1.0
        assert 0.0 <= proportions["abstention_rate"] <= 1.0

        accuracies = policy["accuracies"]
        assert 0.0 <= accuracies["full_population"] <= 1.0
        assert 0.0 <= accuracies["selective_accuracy"] <= 1.0


class TestPlantDocEvaluationHandler:
    def test_missing_plantdoc_directory_handled_gracefully(self, tmp_path):
        dummy_plantdoc_dir = tmp_path / "nonexistent_plantdoc"
        output_dir = tmp_path / "reports"
        mapping = {
            "plantdoc_label_map": {
                "_comment": "Metadata comment",
                "Tomato leaf bacterial spot": 0,
                "Tomato Early blight leaf": 1,
                "Tomato leaf": 9,
            }
        }

        res = evaluate_plantdoc_generalization(
            model=None,
            plantdoc_dir=dummy_plantdoc_dir,
            mapping=mapping,
            output_dir=output_dir,
        )

        assert res["status"] == "dataset_not_available"
        assert res["evaluated"] is False
        assert res["mapped_class_count"] == 3
        assert (output_dir / "plantdoc_status.json").exists()
        assert (output_dir / "PLANTDOC_EVALUATION.md").exists()

        content = (output_dir / "PLANTDOC_EVALUATION.md").read_text(encoding="utf-8")
        assert "PlantDoc Out-of-Domain Evaluation Protocol" in content
        assert "Tomato — Target Spot" in content


class TestScientificReportWriter:
    def test_report_generation(self, sample_metrics_data, tmp_path):
        d = sample_metrics_data
        metrics = compute_classification_metrics(
            y_true=d["y_true"],
            y_pred=d["y_pred"],
            y_prob=d["probs_uncal"],
            class_names=d["class_names"],
        )
        rel = compute_reliability_analysis(
            y_true=d["y_true"],
            probs_uncal=d["probs_uncal"],
            probs_cal=d["probs_cal"],
            temperature=d["temperature"],
        )
        calibrated_payload = {
            "checkpoint": "mobilenetv3_baseline.keras",
            "test_samples": d["n_samples"],
            "accuracy": metrics["accuracy"],
            "macro_f1": metrics["macro_f1"],
            "roc_auc_macro": metrics["roc_auc_macro"],
            "reliability_analysis": rel,
        }

        report_path = tmp_path / "SCIENTIFIC_EVALUATION.md"
        write_scientific_evaluation_report(
            output_path=report_path,
            baseline_metrics=metrics,
            calibrated_metrics=calibrated_payload,
            reliability_metrics=rel,
            n_samples=d["n_samples"],
            elapsed_time=1.5,
            model_path=Path("models/mobilenetv3_baseline.keras"),
            temperature=d["temperature"],
        )

        assert report_path.exists()
        text = report_path.read_text(encoding="utf-8")
        assert "# 🔬 Aura Farming — Phase 7: Scientific Evaluation Report" in text
        assert "Sealed Test Set" in text
        assert "Expected Calibration Error" in text
        assert "Selective Accuracy" in text
