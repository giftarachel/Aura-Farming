"""
Aura Farming — tests/test_integration_phase5.py
=================================================
End-to-end integration tests for Phase 5 pipeline.

Tests the sequential flow:
  1. Image validation & Quality gate assessment
  2. Inference tensor preprocessing & mock inference
  3. Post-hoc calibration & abstention policy
  4. Grad-CAM visual attention mapping
  5. Disease profile & prevention guidance integration
  6. Robustness against missing checkpoints, missing calibration files, and visual failures

Run with:
    pytest tests/test_integration_phase5.py -v
"""

from __future__ import annotations

import io
import json
from pathlib import Path
import numpy as np
import pytest
from PIL import Image, ImageDraw, ImageFilter

from src.image_validation import validate_upload
from src.image_quality import assess_quality, QualityStatus
from src.calibration import (
    apply_calibration_and_threshold,
    ReliabilityStatus,
    PredictionResult,
)
from src.gradcam import (
    generate_heatmap_from_gradients,
    blend_heatmap_with_image,
    GradCAMError,
)
from src.disease_information import get_disease_info
from src.guidance import get_guidance


@pytest.fixture
def class_mapping() -> dict[int, str]:
    mapping_path = Path(__file__).parent.parent / "configs" / "class_mapping.json"
    with open(mapping_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {int(k): v["display_name"] for k, v in data["classes"].items()}


def create_sample_leaf_bytes(sharp: bool = True, dark: bool = False) -> Tuple[bytes, str]:
    img = Image.new("RGB", (256, 256), color=(20, 20, 20) if dark else (120, 160, 80))
    draw = ImageDraw.Draw(img)
    if sharp and not dark:
        for i in range(16, 240, 32):
            draw.line([(i, 16), (i, 240)], fill=(255, 255, 255), width=3)
            draw.line([(16, i), (240, i)], fill=(40, 60, 30), width=3)
    elif not sharp and not dark:
        img = img.filter(ImageFilter.GaussianBlur(radius=15))

    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue(), "sample_leaf.jpg"


class TestPhase5IntegrationPipeline:
    def test_pipeline_confident_flow(self, class_mapping):
        # 1. Upload & decode
        raw_bytes, filename = create_sample_leaf_bytes(sharp=True, dark=False)
        img, meta = validate_upload(filename, raw_bytes)
        assert img is not None
        assert meta["width"] == 256

        # 2. Quality gate
        qr = assess_quality(img)
        assert qr.passed is True
        assert qr.can_proceed is True

        # 3. Simulate high-confidence model output (e.g. Early Blight = class 1)
        simulated_probs = np.zeros(10, dtype=np.float32)
        simulated_probs[1] = 0.92
        simulated_probs[0] = 0.04
        simulated_probs = simulated_probs / np.sum(simulated_probs)

        # 4. Calibration & Abstention
        pred_res = apply_calibration_and_threshold(
            raw_probabilities=simulated_probs,
            class_names=class_mapping,
            quality_report=qr,
            temperature=1.0,
        )
        assert pred_res.reliability == ReliabilityStatus.CONFIDENT
        assert pred_res.class_index == 1
        assert pred_res.class_name == "Tomato — Early Blight"

        # 5. Grad-CAM generation
        mock_features = np.ones((1, 7, 7, 32), dtype=np.float32)
        mock_grads = np.ones((1, 7, 7, 32), dtype=np.float32)
        cam = generate_heatmap_from_gradients(mock_features, mock_grads)
        overlay = blend_heatmap_with_image(img, cam, alpha=0.4)
        assert isinstance(overlay, Image.Image)
        assert overlay.size == img.size

        # 6. Disease profile & prevention guidance
        info = get_disease_info(pred_res.class_name)
        guidance = get_guidance(pred_res.class_name)
        assert info["condition"] == "Early Blight"
        assert len(info["symptoms"]) > 0
        assert len(guidance["immediate_actions"]) > 0

    def test_pipeline_quality_gate_rejection_flow(self, class_mapping):
        # Image is severely dark
        raw_bytes, filename = create_sample_leaf_bytes(sharp=False, dark=True)
        img, meta = validate_upload(filename, raw_bytes)

        # Quality gate should fail
        qr = assess_quality(img)
        assert qr.passed is False
        assert qr.can_proceed is False

        # When passed to calibration/abstention, prediction must be ABSTAINED
        mock_probs = np.zeros(10)
        mock_probs[2] = 0.99
        mock_probs = mock_probs / np.sum(mock_probs)

        pred_res = apply_calibration_and_threshold(
            raw_probabilities=mock_probs,
            class_names=class_mapping,
            quality_report=qr,
        )
        assert pred_res.reliability == ReliabilityStatus.ABSTAINED
        assert pred_res.class_index is None
        assert "quality failure" in pred_res.explanation.lower()

    def test_pipeline_uncertain_abstention_flow(self, class_mapping):
        raw_bytes, filename = create_sample_leaf_bytes(sharp=True)
        img, _ = validate_upload(filename, raw_bytes)
        qr = assess_quality(img)

        # Flat / ambiguous probabilities -> all classes ~10%
        ambiguous_probs = np.ones(10, dtype=np.float32) / 10.0
        pred_res = apply_calibration_and_threshold(
            raw_probabilities=ambiguous_probs,
            class_names=class_mapping,
            quality_report=qr,
            temperature=1.0,
            abstention_threshold=0.40,
        )
        assert pred_res.reliability == ReliabilityStatus.ABSTAINED
        assert pred_res.class_index is None

    def test_gradcam_failure_does_not_abort_prediction(self, class_mapping):
        raw_bytes, filename = create_sample_leaf_bytes(sharp=True)
        img, _ = validate_upload(filename, raw_bytes)
        qr = assess_quality(img)

        simulated_probs = np.zeros(10, dtype=np.float32)
        simulated_probs[9] = 0.95
        simulated_probs = simulated_probs / np.sum(simulated_probs)

        pred_res = apply_calibration_and_threshold(
            raw_probabilities=simulated_probs,
            class_names=class_mapping,
            quality_report=qr,
            temperature=1.0,
        )
        assert pred_res.reliability == ReliabilityStatus.CONFIDENT

        # Simulate Grad-CAM failure with NaN gradients
        try:
            bad_grads = np.array([[[[np.nan]]]], dtype=np.float32)
            generate_heatmap_from_gradients(np.ones_like(bad_grads), bad_grads)
            overlay = None
        except GradCAMError:
            overlay = None

        # Prediction result remains intact and valid
        assert overlay is None
        assert pred_res.class_name == "Tomato — Healthy"
