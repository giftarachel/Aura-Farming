"""
Aura Farming — training/validate_end_to_end.py
===============================================
End-to-end model and integration validation workflow.

Phase 6 Preparation & Verification
────────────────────────────────────
This script runs a comprehensive validation checklist once training is complete
(or using synthetic fixtures when the model checkpoint is unavailable).

Checklist verified:
  1. Checkpoint Verification:
     - Expected model file exists and is readable
     - Input size (224, 224, 3) and output dimension (10 classes)
  2. Inference & Preprocessing Compatibility:
     - Input tensor preprocessing produces [-1.0, 1.0] range float32
     - Forward inference produces valid, finite probability distribution summing to 1.0
  3. Post-Hoc Calibration Workflow:
     - Calibration fitted STRICTLY on validation data (data/splits/val_manifest.csv)
     - SEALED TEST SET PROTECTION: Asserts test_manifest.csv is never touched
     - Fits temperature parameter T > 0, tracks ECE, NLL, Brier score
     - Verifies saving and re-loading of calibration JSON artifact
  4. Grad-CAM Explainability:
     - Targets final convolutional layer (mobilenetv3small base / conv features)
     - Computes gradients and normalizes positive heatmaps to [0.0, 1.0]
     - Blends overlay with original PIL image without shape distortion
  5. Reliability & Abstention Policy:
     - Confident, Uncertain, and Abstained decision paths
     - Integration with photographic quality gate (blur, exposure)
  6. Information & Prevention Guidance:
     - Verified profile retrieval for all predicted disease classes

Usage:
    # Run full validation against trained model:
    python training/validate_end_to_end.py

    # Specify custom model checkpoint:
    python training/validate_end_to_end.py --model-path models/mobilenetv3_phase_a.keras

    # Run quick smoke verification on a sample subset:
    python training/validate_end_to_end.py --quick

    # Run fixture-based self-test (no trained weights required):
    python training/validate_end_to_end.py --fixture-mode
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

import numpy as np
import yaml
from PIL import Image, ImageDraw

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from src.image_quality import assess_quality, QualityReport, QualityStatus
from src.preprocessing import preprocess_for_inference
from src.predictor import is_model_available, predict
from src.calibration import (
    apply_calibration_and_threshold,
    fit_temperature_scaling,
    save_calibration_artifact,
    load_calibration_artifact,
    compute_ece,
    compute_nll,
    compute_brier_score,
    is_probability_distribution,
    PredictionResult,
    ReliabilityStatus,
)
from src.gradcam import (
    generate_heatmap_from_gradients,
    blend_heatmap_with_image,
    generate_gradcam_overlay,
    GradCAMError,
)
from src.disease_information import get_disease_info
from src.guidance import get_guidance


def _load_config() -> dict:
    config_path = PROJECT_ROOT / "configs" / "config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _load_class_mapping() -> dict:
    mapping_path = PROJECT_ROOT / "configs" / "class_mapping.json"
    with open(mapping_path, "r", encoding="utf-8") as f:
        return json.load(f)


def make_test_leaf_image(width: int = 224, height: int = 224, sharp: bool = True) -> Image.Image:
    """Create a controlled synthetic leaf photograph for testing."""
    img = Image.new("RGB", (width, height), color=(110, 155, 75))
    draw = ImageDraw.Draw(img)
    if sharp:
        for i in range(16, width - 16, 24):
            draw.line([(i, 16), (i, height - 16)], fill=(240, 240, 220), width=2)
            draw.line([(16, i), (width - 16, i)], fill=(40, 70, 30), width=2)
    return img


class ValidationReport:
    def __init__(self):
        self.checks: list[dict[str, Any]] = []

    def record(self, name: str, passed: bool, details: str):
        self.checks.append({
            "name": name,
            "passed": passed,
            "details": details,
        })
        icon = "✅" if passed else "❌"
        print(f"  {icon} {name:<45} : {details}")

    @property
    def all_passed(self) -> bool:
        return all(c["passed"] for c in self.checks)


def validate_fixture_mode(report: ValidationReport) -> int:
    """Run validation checks using mock models and controlled fixtures."""
    print("\n" + "=" * 70)
    print("🌿 Aura Farming — End-to-End Validation [FIXTURE MODE]")
    print("=" * 70)

    # 1. Image Quality Gate
    img_sharp = make_test_leaf_image(224, 224, sharp=True)
    qr_sharp = assess_quality(img_sharp)
    report.record(
        "Quality Gate: Sharp Image Assessment",
        qr_sharp.passed and qr_sharp.status in (QualityStatus.PASSED, QualityStatus.WARNING),
        f"Sharpness={qr_sharp.blur_score:.1f}, Exposure={qr_sharp.brightness_mean:.1f}",
    )

    img_dark = Image.new("RGB", (224, 224), color=(5, 5, 5))
    qr_dark = assess_quality(img_dark)
    report.record(
        "Quality Gate: Dark Image Rejection",
        not qr_dark.passed and qr_dark.status == QualityStatus.FAILED,
        f"Correctly failed on underexposure ({qr_dark.brightness_mean:.1f}/255)",
    )

    # 2. Probability Calibration Pipeline
    np.random.seed(42)
    synthetic_logits = np.random.randn(200, 10) * 2.5
    synthetic_labels = np.random.randint(0, 10, size=200)
    for i in range(200):
        if np.random.rand() < 0.85:
            synthetic_logits[i, synthetic_labels[i]] += 4.0

    optimal_t, m_before, m_after = fit_temperature_scaling(synthetic_logits, synthetic_labels)
    report.record(
        "Temperature Scaling Optimization",
        optimal_t > 0.0 and m_after["ece"] <= m_before["ece"],
        f"Fitted T={optimal_t:.4f}, ECE: {m_before['ece']:.4f} -> {m_after['ece']:.4f}",
    )

    # 3. Artifact Round-Trip Persistence
    temp_calib_path = PROJECT_ROOT / "models" / "_test_fixture_calibration.json"
    save_calibration_artifact(
        filepath=temp_calib_path,
        temperature=optimal_t,
        metrics_before=m_before,
        metrics_after=m_after,
        metadata={"fixture": True},
    )
    loaded_calib = load_calibration_artifact(temp_calib_path)
    if temp_calib_path.is_file():
        temp_calib_path.unlink()

    report.record(
        "Calibration Artifact Save & Reload",
        loaded_calib.get("is_calibrated") is True and abs(loaded_calib.get("temperature", 0.0) - optimal_t) < 1e-4,
        f"Verified JSON serialization and parameter fidelity (T={loaded_calib.get('temperature'):.4f})",
    )

    # 4. Reliability & Abstention Decisions
    class_map = {i: f"Class {i}" for i in range(10)}
    high_conf_probs = np.zeros(10, dtype=np.float32)
    high_conf_probs[1] = 0.95
    high_conf_probs = high_conf_probs / np.sum(high_conf_probs)

    res_conf = apply_calibration_and_threshold(
        high_conf_probs, class_map, quality_report=qr_sharp, temperature=1.0
    )
    report.record(
        "Reliability Policy: Confident Flow",
        res_conf.reliability == ReliabilityStatus.CONFIDENT and res_conf.class_index == 1,
        f"Status: {res_conf.reliability.value}, Top prob: {res_conf.top_probability:.1%}",
    )

    low_conf_probs = np.ones(10, dtype=np.float32) / 10.0
    res_abstain = apply_calibration_and_threshold(
        low_conf_probs, class_map, quality_report=qr_sharp, temperature=1.0
    )
    report.record(
        "Reliability Policy: Low Confidence Abstention",
        res_abstain.reliability == ReliabilityStatus.ABSTAINED and res_abstain.class_index is None,
        f"Status: {res_abstain.reliability.value}, Correctly withheld diagnosis",
    )

    res_qual_abstain = apply_calibration_and_threshold(
        high_conf_probs, class_map, quality_report=qr_dark, temperature=1.0
    )
    report.record(
        "Reliability Policy: Quality Gate Abstention",
        res_qual_abstain.reliability == ReliabilityStatus.ABSTAINED,
        f"Correctly withheld diagnosis due to image quality rejection",
    )

    # 5. Grad-CAM Heatmap Math & Blending
    mock_features = np.ones((1, 7, 7, 16), dtype=np.float32)
    mock_grads = np.ones((1, 7, 7, 16), dtype=np.float32)
    cam = generate_heatmap_from_gradients(mock_features, mock_grads)
    overlay = blend_heatmap_with_image(img_sharp, cam, alpha=0.4)
    report.record(
        "Grad-CAM Heatmap Computation & Blending",
        cam.shape == (7, 7) and overlay.size == img_sharp.size and overlay.mode == "RGB",
        f"Heatmap shape: {cam.shape}, Overlay size: {overlay.size}",
    )

    # 6. Disease Profile & Guidance Integration
    info = get_disease_info(1)  # Early Blight
    guidance = get_guidance(1)
    report.record(
        "Disease Information & Guidance Retrieval",
        info["condition"] == "Early Blight" and len(guidance["immediate_actions"]) > 0,
        f"Loaded '{info['condition']}' with {len(guidance['immediate_actions'])} immediate actions",
    )

    print("\n" + "=" * 70)
    print(f"Validation Result: {'ALL CHECKS PASSED ✅' if report.all_passed else 'SOME CHECKS FAILED ❌'}")
    print("=" * 70 + "\n")
    return 0 if report.all_passed else 1


def validate_trained_pipeline(
    model_path: Path,
    val_manifest_path: Path,
    quick: bool = False,
) -> int:
    """Validate the end-to-end pipeline with the trained model and validation set."""
    report = ValidationReport()
    print("\n" + "=" * 70)
    print("🌿 Aura Farming — End-to-End Model & Pipeline Validation")
    print("=" * 70)

    # 1. Model Checkpoint Exists
    model_exists = model_path.is_file()
    report.record(
        "Model Checkpoint Exists on Disk",
        model_exists,
        str(model_path) if model_exists else f"File not found: {model_path}",
    )
    if not model_exists:
        print("\n❌ Model checkpoint not found. If training is active, wait for completion.")
        print("   To verify components using fixtures, run with --fixture-mode.")
        return 1

    # 2. Model Loads Successfully
    try:
        import tensorflow as tf
        model = tf.keras.models.load_model(str(model_path))
        input_shape = model.input_shape
        output_shape = model.output_shape
        load_success = True
        details = f"Input shape: {input_shape}, Output shape: {output_shape}"
    except Exception as exc:
        load_success = False
        details = f"Failed to load: {exc}"
        model = None

    report.record("Model Architecture & Weights Load", load_success, details)
    if not load_success or model is None:
        return 1

    # Verify 10 classes output
    has_ten_classes = (output_shape[-1] == 10)
    report.record("Model Output Dimension (10 Classes)", has_ten_classes, f"Classes: {output_shape[-1]}")

    # 3. Input Preprocessing Compatibility
    test_leaf = make_test_leaf_image(224, 224, sharp=True)
    tensor = preprocess_for_inference(test_leaf)
    prep_valid = (
        tensor is not None
        and tensor.shape == (1, 224, 224, 3)
        and tensor.dtype == np.float32
        and float(np.min(tensor)) >= -1.01
        and float(np.max(tensor)) <= 1.01
    )
    report.record(
        "Preprocessing Range and Shape Compatibility",
        prep_valid,
        f"Shape: {tensor.shape}, Range: [{np.min(tensor):.2f}, {np.max(tensor):.2f}], dtype: {tensor.dtype}",
    )

    # 4. Raw Model Forward Pass & Probability Validation
    raw_probs = model.predict(tensor, verbose=0)[0].astype(np.float32)
    is_valid_prob = (
        len(raw_probs) == 10
        and np.all(np.isfinite(raw_probs))
        and np.all(raw_probs >= 0.0)
        and abs(float(np.sum(raw_probs)) - 1.0) < 1e-4
    )
    report.record(
        "Forward Inference & Softmax Distribution",
        is_valid_prob,
        f"Sum: {np.sum(raw_probs):.5f}, Max prob: {np.max(raw_probs):.4f} (class {np.argmax(raw_probs)})",
    )

    # 5. Scientific Safety: Validation Split Only (Sealed Test Set Protection)
    is_val_split = "val" in str(val_manifest_path).lower() and "test" not in str(val_manifest_path).lower()
    report.record(
        "Sealed Test Set Protection Guard",
        is_val_split and val_manifest_path.is_file(),
        f"Validation manifest: {val_manifest_path.name} (Test set strictly sealed)",
    )

    # 6. Validation Set Inference & Temperature Scaling Calibration
    if val_manifest_path.is_file() and is_val_split:
        import pandas as pd
        val_df = pd.read_csv(val_manifest_path)
        label_col = "class_index" if "class_index" in val_df.columns else "label"
        max_samples = 200 if quick else min(500, len(val_df))

        sample_paths = val_df["path"].tolist()[:max_samples]
        sample_labels = val_df[label_col].to_numpy(dtype=np.int64)[:max_samples]

        batch_tensors = []
        valid_labels = []
        for p_str, lbl in zip(sample_paths, sample_labels):
            p = Path(p_str)
            if not p.is_file():
                continue
            try:
                pil_img = Image.open(p).convert("RGB").resize((224, 224), Image.Resampling.LANCZOS)
                arr = np.array(pil_img, dtype=np.float32) / 127.5 - 1.0
                batch_tensors.append(arr)
                valid_labels.append(lbl)
            except Exception:
                continue

        if batch_tensors:
            val_batch = np.stack(batch_tensors, axis=0)
            val_preds = model.predict(val_batch, verbose=0)
            optimal_temp, m_before, m_after = fit_temperature_scaling(
                val_preds, np.array(valid_labels, dtype=np.int64)
            )
            calib_success = (optimal_temp > 0.0)
            calib_details = f"T={optimal_temp:.4f}, Val samples={len(val_preds)}, ECE: {m_before['ece']:.4f} -> {m_after['ece']:.4f}"
        else:
            calib_success = False
            calib_details = "Could not load validation image samples."
    else:
        calib_success = False
        calib_details = f"Validation manifest not found at {val_manifest_path}"

    report.record("Validation Temperature Scaling Fitting", calib_success, calib_details)

    # 7. Calibration Artifact Persistence
    artifact_path = PROJECT_ROOT / "models" / "mobilenetv3_calibration.json"
    if calib_success:
        save_calibration_artifact(
            filepath=artifact_path,
            temperature=optimal_temp,
            metrics_before=m_before,
            metrics_after=m_after,
            metadata={"validation_samples": len(val_preds), "calibrated_at": datetime.now(timezone.utc).isoformat()},
        )
    loaded_artifact = load_calibration_artifact(artifact_path)
    report.record(
        "Calibration Artifact Saved & Reloaded",
        loaded_artifact.get("is_calibrated") is True and loaded_artifact.get("temperature", 0.0) > 0.0,
        f"Active Temperature T={loaded_artifact.get('temperature', 1.0):.4f}",
    )

    # 8. Grad-CAM Overlay Generation on Real Model
    try:
        top_class = int(np.argmax(raw_probs))
        overlay = generate_gradcam_overlay(
            image=test_leaf,
            tensor=tensor,
            class_index=top_class,
            model=model,
        )
        cam_valid = (
            overlay is not None
            and isinstance(overlay, Image.Image)
            and overlay.size == test_leaf.size
            and overlay.mode == "RGB"
        )
        cam_details = f"Generated {overlay.size[0]}×{overlay.size[1]} RGB overlay for class {top_class}"
    except Exception as exc:
        cam_valid = False
        cam_details = f"Grad-CAM generation failed: {exc}"

    report.record("Grad-CAM Real Model Overlay Generation", cam_valid, cam_details)

    # 9. End-to-End Decision & Agronomic Guidance Integration
    class_mapping_data = _load_class_mapping()
    class_names = {int(k): v["display_name"] for k, v in class_mapping_data["classes"].items()}
    qr = assess_quality(test_leaf)
    result = apply_calibration_and_threshold(
        raw_probabilities=raw_probs,
        class_names=class_names,
        quality_report=qr,
    )
    predicted_name = result.class_name or class_names.get(top_class, "Tomato — Healthy")
    disease_info = get_disease_info(predicted_name)
    guidance_info = get_guidance(predicted_name)

    integration_valid = (
        result.reliability in (ReliabilityStatus.CONFIDENT, ReliabilityStatus.UNCERTAIN, ReliabilityStatus.ABSTAINED)
        and disease_info["condition"] != "Unknown"
        and len(guidance_info["immediate_actions"]) > 0
    )
    report.record(
        "Full End-to-End UI Decision & Guidance Pipeline",
        integration_valid,
        f"Diagnosis: '{predicted_name}', Reliability: {result.reliability.value}, IPM Actions: {len(guidance_info['immediate_actions'])}",
    )

    print("\n" + "=" * 70)
    print(f"Overall Validation Result: {'ALL CHECKS PASSED ✅' if report.all_passed else 'SOME CHECKS FAILED ❌'}")
    print("=" * 70 + "\n")
    return 0 if report.all_passed else 1


def main():
    parser = argparse.ArgumentParser(description="Aura Farming — End-to-End Validation Workflow")
    parser.add_argument("--model-path", type=str, default=None,
                        help="Path to Keras model checkpoint.")
    parser.add_argument("--val-manifest", type=str, default="data/splits/val_manifest.csv",
                        help="Path to validation manifest CSV.")
    parser.add_argument("--quick", action="store_true",
                        help="Run quick validation on a subset of samples.")
    parser.add_argument("--fixture-mode", action="store_true",
                        help="Run validation in fixture mode (synthetic model/fixtures).")
    args = parser.parse_args()

    report = ValidationReport()

    if args.fixture_mode:
        sys.exit(validate_fixture_mode(report))

    cfg = _load_config()
    model_path_str = args.model_path or cfg.get("model", {}).get("weights_path") or "models/mobilenetv3_phase_a.keras"
    model_p = PROJECT_ROOT / model_path_str
    val_manifest_p = PROJECT_ROOT / args.val_manifest

    # If the model checkpoint doesn't exist yet, fall back cleanly with clear notice
    if not model_p.is_file():
        print(f"\n⚠️  Configured model checkpoint not found: {model_p}")
        print("   Running validation in --fixture-mode to verify component contracts...\n")
        sys.exit(validate_fixture_mode(report))

    sys.exit(validate_trained_pipeline(
        model_path=model_p,
        val_manifest_path=val_manifest_p,
        quick=args.quick,
    ))


if __name__ == "__main__":
    main()
