"""
Aura Farming — src/calibration.py
====================================
Probability calibration and confidence-based abstention.

Phase 5 Implementation: COMPLETE
─────────────────────────────────
This module handles:
  1. Probability handling:
     - Robust logit-to-probability and probability-to-logit conversions
     - Numerical stability against NaN, Inf, and underflow
     - Normalization checks to ensure distributions sum to 1.0
  2. Temperature scaling (multiclass):
     - Single scalar parameter T > 0 fitted strictly on validation data
     - Softmax(z / T): softens overconfidence (T > 1) without altering argmax rank
  3. Calibration evaluation metrics:
     - Expected Calibration Error (ECE) with equal-width binning
     - Negative Log-Likelihood (NLL)
     - Multiclass Brier score
  4. Reliability & Abstention policy:
     - Categorizes predictions into CONFIDENT, UNCERTAIN, or ABSTAINED
     - Integrates photographic quality reports into the reliability verdict
     - Configurable thresholds for confidence, abstention, and decision margins
  5. Artifact management:
     - Safe JSON persistence and loading for calibration parameters
     - Graceful fallback when calibration artifacts are absent
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Optional, Union, Tuple, Sequence

import numpy as np
import yaml
from scipy.optimize import minimize_scalar

from src.image_quality import QualityReport, QualityStatus


# ─────────────────────────────────────────────────────────────────
# Reliability Status Enum
# ─────────────────────────────────────────────────────────────────

class ReliabilityStatus(str, Enum):
    """
    The three possible reliability states for a prediction.

    CONFIDENT:  Calibrated confidence >= confidence_threshold, quality gate
                passed with no disqualifying warnings, and distinct class margin.
    UNCERTAIN:  Calibrated confidence is between abstention_threshold and
                confidence_threshold, or image quality has warnings, or
                top-2 candidate margin is narrow. Displayed with a warning.
    ABSTAINED:  Calibrated confidence < abstention_threshold, or image
                quality gate hard-failed. The system withholds diagnosis.
    """
    CONFIDENT = "confident"
    UNCERTAIN = "uncertain"
    ABSTAINED = "abstained"


# ─────────────────────────────────────────────────────────────────
# Result dataclass
# ─────────────────────────────────────────────────────────────────

@dataclass
class PredictionResult:
    """
    Complete structured output from the prediction, calibration, and abstention pipeline.

    Attributes:
        class_index: Integer index of predicted class (or None if abstained).
        class_name: Human-readable label (or None if abstained).
        probabilities: Full calibrated probability vector (sums to 1.0).
        raw_probabilities: Original uncalibrated probability vector from model.
        top_probability: Highest calibrated probability value.
        raw_top_probability: Highest raw uncalibrated probability.
        reliability: ReliabilityStatus (CONFIDENT, UNCERTAIN, or ABSTAINED).
        temperature: Temperature parameter T applied during scaling.
        top_k: Top candidate predictions as list of (class_index, name, probability).
        explanation: Clear, human-readable justification of the reliability decision.
        quality_report: Associated image quality assessment report.
        is_placeholder: False for Phase 5 real assessments.
    """
    class_index: Optional[int] = None
    class_name: Optional[str] = None
    probabilities: Optional[np.ndarray] = None
    raw_probabilities: Optional[np.ndarray] = None
    top_probability: Optional[float] = None
    raw_top_probability: Optional[float] = None
    reliability: ReliabilityStatus = ReliabilityStatus.ABSTAINED
    temperature: float = 1.0
    top_k: list[Tuple[int, str, float]] = field(default_factory=list)
    explanation: str = ""
    quality_report: Optional[QualityReport] = None
    is_placeholder: bool = False


# ─────────────────────────────────────────────────────────────────
# Configuration helper
# ─────────────────────────────────────────────────────────────────

def _load_prediction_config() -> dict:
    """Load prediction and calibration configuration from configs/config.yaml with safe fallbacks."""
    config_path = Path(__file__).parent.parent / "configs" / "config.yaml"
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
            pred = cfg.get("prediction") or {}
            model = cfg.get("model") or {}
            eval_cfg = cfg.get("evaluation") or {}

            conf = pred.get("confidence_threshold")
            ab = pred.get("abstention_threshold")
            mar = pred.get("margin_threshold")
            calib = model.get("calibration_path")
            bins = eval_cfg.get("ece_num_bins")

            return {
                "confidence_threshold": 0.70 if conf is None else conf,
                "abstention_threshold": 0.40 if ab is None else ab,
                "margin_threshold": 0.05 if mar is None else mar,
                "calibration_path": "models/mobilenetv3_calibration.json" if calib is None else calib,
                "ece_num_bins": 10 if bins is None else bins,
            }
    except Exception:
        return {
            "confidence_threshold": 0.70,
            "abstention_threshold": 0.40,
            "margin_threshold": 0.05,
            "calibration_path": "models/mobilenetv3_calibration.json",
            "ece_num_bins": 10,
        }


# ─────────────────────────────────────────────────────────────────
# Mathematical Utilities: Logits & Softmax
# ─────────────────────────────────────────────────────────────────

def is_probability_distribution(vec: np.ndarray, tol: float = 1e-3) -> bool:
    """Check if a 1D vector represents a valid probability distribution."""
    if not isinstance(vec, np.ndarray) or vec.ndim != 1 or vec.size == 0:
        return False
    if np.any(np.isnan(vec)) or np.any(np.isinf(vec)):
        return False
    if np.any(vec < -tol) or np.any(vec > 1.0 + tol):
        return False
    return bool(abs(np.sum(vec) - 1.0) < tol)


def stable_softmax(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """
    Compute numerically stable softmax scaled by temperature T.

    softmax(z / T)_i = exp((z_i / T) - max(z / T)) / sum_j exp((z_j / T) - max(z / T))

    Args:
        logits: 1D array of shape (K,) or 2D array of shape (N, K).
        temperature: Temperature scalar T > 0.

    Returns:
        NumPy float32 array with probabilities summing to 1.0 along the last axis.

    Raises:
        ValueError: If temperature is not strictly positive, or if inputs are non-finite.
    """
    if temperature <= 0.0:
        raise ValueError(f"Temperature must be strictly positive (got {temperature}).")

    arr = np.asarray(logits, dtype=np.float64)
    if np.any(np.isnan(arr)) or np.any(np.isinf(arr)):
        raise ValueError("Logits contain NaN or Infinite values.")

    scaled = arr / temperature
    # Shift by max along last axis for numerical stability
    shifted = scaled - np.max(scaled, axis=-1, keepdims=True)
    exp_vals = np.exp(shifted)
    sum_exp = np.sum(exp_vals, axis=-1, keepdims=True)

    probs = exp_vals / sum_exp
    return probs.astype(np.float32)


def probabilities_to_logits(probs: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    Convert normalized probability vectors into logits (up to an arbitrary additive constant).

    Since softmax(log(p) / T) = softmax(z / T), taking log(p + eps) exactly recovers
    temperature-scaled probabilities when passed to stable_softmax.

    Args:
        probs: 1D or 2D array of normalized probabilities in [0, 1].
        eps: Small epsilon to prevent log(0).

    Returns:
        NumPy float64 array of unnormalized logits.
    """
    arr = np.asarray(probs, dtype=np.float64)
    clipped = np.clip(arr, eps, 1.0)
    return np.log(clipped)


def apply_temperature_scaling(
    probabilities_or_logits: np.ndarray,
    temperature: float = 1.0,
) -> np.ndarray:
    """
    Apply temperature scaling to a probability vector or raw logits.

    If the input is already a valid normalized probability vector and T == 1.0,
    the input is returned directly (no double-softmax). Otherwise, logits are
    computed and stable_softmax is applied.

    Args:
        probabilities_or_logits: 1D shape (K,) or 2D shape (N, K).
        temperature: Scalar temperature T > 0.

    Returns:
        Calibrated probabilities summing to 1.0.
    """
    arr = np.asarray(probabilities_or_logits, dtype=np.float32)

    # Check 1D case
    if arr.ndim == 1:
        if abs(temperature - 1.0) < 1e-6 and is_probability_distribution(arr):
            # Already normalized probabilities and T=1 -> return directly
            return np.clip(arr, 0.0, 1.0) / np.sum(arr)
        # Convert to logits and apply temperature scaling
        if is_probability_distribution(arr):
            logits = probabilities_to_logits(arr)
        else:
            logits = arr.astype(np.float64)
        return stable_softmax(logits, temperature=temperature)

    # Check 2D case
    if arr.ndim == 2:
        if abs(temperature - 1.0) < 1e-6:
            row_sums = np.sum(arr, axis=1)
            if np.all(np.abs(row_sums - 1.0) < 1e-3) and np.all(arr >= 0.0):
                return arr / np.sum(arr, axis=1, keepdims=True)
        # Check if first row is prob
        if np.all(arr >= 0.0) and abs(np.sum(arr[0]) - 1.0) < 1e-3:
            logits = probabilities_to_logits(arr)
        else:
            logits = arr.astype(np.float64)
        return stable_softmax(logits, temperature=temperature)

    raise ValueError(f"Expected 1D or 2D array, got shape {arr.shape}.")


# ─────────────────────────────────────────────────────────────────
# Calibration Metrics
# ─────────────────────────────────────────────────────────────────

def compute_ece(
    probabilities: np.ndarray,
    labels: np.ndarray,
    num_bins: int = 10,
) -> float:
    """
    Compute Expected Calibration Error (ECE) using equal-width binning.

    ECE divides the [0, 1] confidence range into M equal-width bins B_1, ..., B_M:
        ECE = sum_{m=1}^M (|B_m| / N) * |acc(B_m) - conf(B_m)|

    Where:
        acc(B_m) = accuracy of samples placed in bin B_m
        conf(B_m) = average predicted confidence of samples placed in bin B_m

    Args:
        probabilities: shape (N, K), predicted class probabilities.
        labels: shape (N,), ground truth integer class indices in 0..K-1.
        num_bins: Number of equal-width bins (typically 10 or 15).

    Returns:
        float: Expected Calibration Error in [0, 1]. Lower is better.
    """
    probs = np.asarray(probabilities, dtype=np.float64)
    y_true = np.asarray(labels, dtype=np.int64)

    if probs.ndim != 2:
        raise ValueError(f"Probabilities must be 2D array (N, K), got shape {probs.shape}.")
    if y_true.ndim != 1 or len(y_true) != len(probs):
        raise ValueError("Labels must be a 1D array with the same length as probabilities.")
    if len(probs) == 0:
        return 0.0

    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = (predictions == y_true).astype(np.float64)

    n_samples = len(probs)
    bin_boundaries = np.linspace(0.0, 1.0, num_bins + 1)
    ece = 0.0

    for i in range(num_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]

        # Use (lower, upper] for bins > 0, [0, upper] for first bin
        if i == 0:
            in_bin = (confidences >= bin_lower) & (confidences <= bin_upper)
        else:
            in_bin = (confidences > bin_lower) & (confidences <= bin_upper)

        bin_count = np.sum(in_bin)
        if bin_count > 0:
            bin_acc = np.mean(accuracies[in_bin])
            bin_conf = np.mean(confidences[in_bin])
            ece += (bin_count / n_samples) * abs(bin_acc - bin_conf)

    return float(ece)


def compute_nll(
    probabilities: np.ndarray,
    labels: np.ndarray,
    eps: float = 1e-12,
) -> float:
    """
    Compute Negative Log-Likelihood (cross-entropy loss).

    NLL = - (1 / N) * sum_{i=1}^N log(p_{i, y_i} + eps)

    Args:
        probabilities: shape (N, K), predicted class probabilities.
        labels: shape (N,), ground truth integer class indices.
        eps: Small constant for numerical stability.

    Returns:
        float: Mean negative log-likelihood.
    """
    probs = np.asarray(probabilities, dtype=np.float64)
    y_true = np.asarray(labels, dtype=np.int64)

    if len(probs) == 0:
        return 0.0

    n_samples = len(probs)
    # Extract predicted probability assigned to true class
    correct_conf = probs[np.arange(n_samples), y_true]
    clipped = np.clip(correct_conf, eps, 1.0)
    nll = -float(np.mean(np.log(clipped)))
    return nll


def compute_brier_score(
    probabilities: np.ndarray,
    labels: np.ndarray,
) -> float:
    """
    Compute multiclass Brier score.

    Brier = (1 / N) * sum_{i=1}^N sum_{k=1}^K (p_{i, k} - y_{i, k})^2

    Where y_{i, k} = 1 if y_i == k else 0.

    Args:
        probabilities: shape (N, K), predicted class probabilities.
        labels: shape (N,), ground truth integer class indices.

    Returns:
        float: Multiclass Brier score.
    """
    probs = np.asarray(probabilities, dtype=np.float64)
    y_true = np.asarray(labels, dtype=np.int64)

    if len(probs) == 0:
        return 0.0

    n_samples, n_classes = probs.shape
    one_hot = np.zeros((n_samples, n_classes), dtype=np.float64)
    one_hot[np.arange(n_samples), y_true] = 1.0

    brier = float(np.mean(np.sum((probs - one_hot) ** 2, axis=1)))
    return brier


# ─────────────────────────────────────────────────────────────────
# Fitting Temperature Scaling (Optimization on Validation Data)
# ─────────────────────────────────────────────────────────────────

def fit_temperature_scaling(
    logits_or_probs: np.ndarray,
    labels: np.ndarray,
    bounds: Tuple[float, float] = (0.05, 10.0),
) -> Tuple[float, dict, dict]:
    """
    Fit scalar temperature T by minimizing Negative Log-Likelihood on validation data.

    CRITICAL SCIENTIFIC RULE:
        This function must be called strictly with VALIDATION split data.
        The test split must never be used to optimize temperature.

    Args:
        logits_or_probs: shape (N, K) of logits or uncalibrated probabilities.
        labels: shape (N,) of ground truth class indices.
        bounds: Search bounds for scalar T (must be strictly positive).

    Returns:
        Tuple of (optimal_temperature, metrics_before, metrics_after).
    """
    arr = np.asarray(logits_or_probs, dtype=np.float64)
    y_true = np.asarray(labels, dtype=np.int64)

    # Convert to logits if input is probabilities
    if np.all(arr >= 0.0) and abs(np.sum(arr[0]) - 1.0) < 1e-3:
        logits = probabilities_to_logits(arr)
        probs_before = arr
    else:
        logits = arr
        probs_before = stable_softmax(logits, temperature=1.0)

    # Metrics before calibration
    metrics_before = {
        "ece": round(compute_ece(probs_before, y_true), 4),
        "nll": round(compute_nll(probs_before, y_true), 4),
        "brier": round(compute_brier_score(probs_before, y_true), 4),
    }

    # Objective function: NLL as function of temperature T
    def _loss_fn(temp: float) -> float:
        scaled_probs = stable_softmax(logits, temperature=temp)
        return compute_nll(scaled_probs, y_true)

    res = minimize_scalar(_loss_fn, bounds=bounds, method="bounded")
    optimal_temp = float(res.x)

    # Metrics after calibration
    probs_after = stable_softmax(logits, temperature=optimal_temp)
    metrics_after = {
        "ece": round(compute_ece(probs_after, y_true), 4),
        "nll": round(compute_nll(probs_after, y_true), 4),
        "brier": round(compute_brier_score(probs_after, y_true), 4),
    }

    return round(optimal_temp, 4), metrics_before, metrics_after


# ─────────────────────────────────────────────────────────────────
# Artifact Persistence: Save & Load
# ─────────────────────────────────────────────────────────────────

def save_calibration_artifact(
    filepath: Union[str, Path],
    temperature: float,
    metrics_before: dict,
    metrics_after: dict,
    metadata: Optional[dict] = None,
) -> Path:
    """
    Save calibration parameters and validation metrics to a JSON artifact.

    Args:
        filepath: Destination path (e.g. models/mobilenetv3_calibration.json).
        temperature: Fitted temperature value T.
        metrics_before: Dictionary of pre-calibration ECE, NLL, Brier score.
        metrics_after: Dictionary of post-calibration ECE, NLL, Brier score.
        metadata: Optional dictionary with dataset/model details.

    Returns:
        Path to the saved JSON artifact.
    """
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)

    data = {
        "temperature": float(temperature),
        "method": "temperature_scaling",
        "dataset_split": "validation",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "metrics_before": metrics_before,
        "metrics_after": metrics_after,
        "metadata": metadata or {},
    }

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    return path


def load_calibration_artifact(filepath: Optional[Union[str, Path]] = None) -> dict:
    """
    Load calibration parameters from disk with graceful fallback.

    If filepath is not provided, reads the configured path from config.yaml.
    If the file is absent or invalid, returns safe defaults (T = 1.0, uncalibrated).

    Args:
        filepath: Optional path to JSON artifact.

    Returns:
        Dictionary containing temperature and calibration status.
    """
    cfg = _load_prediction_config()
    target_path = Path(filepath if filepath else cfg["calibration_path"])

    # If relative, resolve from project root
    if not target_path.is_absolute():
        target_path = Path(__file__).parent.parent / target_path

    if not target_path.is_file():
        return {
            "temperature": 1.0,
            "is_calibrated": False,
            "artifact_path": str(target_path),
            "notice": "Calibration artifact not found on disk; defaulting to uncalibrated temperature T=1.0.",
        }

    try:
        with open(target_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            temp = float(data.get("temperature", 1.0))
            if temp <= 0:
                temp = 1.0
            return {
                "temperature": temp,
                "is_calibrated": True,
                "artifact_path": str(target_path),
                "metrics_before": data.get("metrics_before", {}),
                "metrics_after": data.get("metrics_after", {}),
                "created_at": data.get("created_at"),
            }
    except Exception as exc:
        return {
            "temperature": 1.0,
            "is_calibrated": False,
            "artifact_path": str(target_path),
            "notice": f"Failed to parse calibration artifact ({exc}); defaulting to T=1.0.",
        }


# ─────────────────────────────────────────────────────────────────
# Public Inference Pipeline: Calibration & Abstention
# ─────────────────────────────────────────────────────────────────

def apply_calibration_and_threshold(
    raw_probabilities: np.ndarray,
    class_names: Union[dict[int, str], Sequence[str]],
    quality_report: Optional[QualityReport] = None,
    calibration_file: Optional[Union[str, Path]] = None,
    temperature: Optional[float] = None,
    confidence_threshold: Optional[float] = None,
    abstention_threshold: Optional[float] = None,
    margin_threshold: Optional[float] = None,
) -> PredictionResult:
    """
    Apply temperature scaling and the reliability/abstention policy to model outputs.
    """
    cfg = _load_prediction_config()
    conf_thresh = float(confidence_threshold if confidence_threshold is not None else cfg["confidence_threshold"])
    abs_thresh = float(abstention_threshold if abstention_threshold is not None else cfg["abstention_threshold"])
    mar_thresh = float(margin_threshold if margin_threshold is not None else cfg["margin_threshold"])

    # Load calibration parameters or use override
    if temperature is not None:
        temp_val = float(temperature)
    else:
        calib_info = load_calibration_artifact(calibration_file)
        temp_val = float(calib_info.get("temperature", 1.0))
    temperature = temp_val

    # Validate raw probabilities input
    raw_arr = np.asarray(raw_probabilities, dtype=np.float32)
    if raw_arr.ndim != 1 or raw_arr.size == 0:
        raise ValueError(f"Expected 1D probability array, got shape {raw_arr.shape}.")

    # Normalize mapping
    if isinstance(class_names, dict):
        mapping = {int(k): str(v) for k, v in class_names.items()}
    else:
        mapping = {i: str(name) for i, name in enumerate(class_names)}

    num_classes = len(raw_arr)

    # 1. Image Quality Rejection Check
    if quality_report is not None and not quality_report.passed:
        feedback = quality_report.feedback or "Image quality did not pass validation checks."
        return PredictionResult(
            class_index=None,
            class_name=None,
            probabilities=raw_arr,
            raw_probabilities=raw_arr,
            top_probability=None,
            raw_top_probability=float(np.max(raw_arr)),
            reliability=ReliabilityStatus.ABSTAINED,
            temperature=temperature,
            top_k=[],
            explanation=(
                f"Prediction with-held due to image quality failure:\n{feedback}\n"
                "Please upload a clearer, well-lit photograph."
            ),
            quality_report=quality_report,
            is_placeholder=False,
        )

    # 2. Temperature Scaling
    calibrated_probs = apply_temperature_scaling(raw_arr, temperature=temperature)
    calibrated_probs = calibrated_probs / np.sum(calibrated_probs)  # ensure exact sum to 1

    top_idx = int(np.argmax(calibrated_probs))
    top_prob = float(calibrated_probs[top_idx])
    raw_top_prob = float(np.max(raw_arr))
    top_name = mapping.get(top_idx, f"Class {top_idx}")

    # Top-K ranking (sorted descending)
    sorted_indices = np.argsort(calibrated_probs)[::-1]
    top_k_list: list[Tuple[int, str, float]] = []
    for idx in sorted_indices[:min(3, num_classes)]:
        idx_int = int(idx)
        top_k_list.append((
            idx_int,
            mapping.get(idx_int, f"Class {idx_int}"),
            round(float(calibrated_probs[idx_int]), 4),
        ))

    # Difference between top 1 and top 2
    margin = (top_prob - float(calibrated_probs[sorted_indices[1]])) if num_classes > 1 else top_prob

    # 3. Reliability Determination
    if top_prob < abs_thresh:
        # Confidence too low -> Abstain completely
        status = ReliabilityStatus.ABSTAINED
        explanation = (
            f"The model's highest calibrated probability ({top_prob:.1%}) is below the "
            f"minimum reliability threshold ({abs_thresh:.1%}). Diagnosis cannot be reliably "
            f"established from this photograph. Please consult a plant pathologist or agricultural officer."
        )
        return_class_index = None
        return_class_name = None
    elif top_prob < conf_thresh:
        # Moderate confidence -> Uncertain
        status = ReliabilityStatus.UNCERTAIN
        explanation = (
            f"Prediction is '{top_name}' with moderate confidence ({top_prob:.1%}). "
            f"This does not exceed the high-confidence threshold ({conf_thresh:.1%}). "
            f"Secondary candidate: '{top_k_list[1][1]}' ({top_k_list[1][2]:.1%}). Use caution."
        )
        return_class_index = top_idx
        return_class_name = top_name
    elif margin < mar_thresh:
        # Top predictions are tied / ambiguous
        status = ReliabilityStatus.UNCERTAIN
        explanation = (
            f"Prediction is '{top_name}' ({top_prob:.1%}), but the distinction from "
            f"'{top_k_list[1][1]}' ({top_k_list[1][2]:.1%}) is narrow (margin: {margin:.1%} < {mar_thresh:.1%}). "
            f"Confirmation by an agricultural expert is recommended."
        )
        return_class_index = top_idx
        return_class_name = top_name
    elif quality_report is not None and quality_report.status == QualityStatus.WARNING:
        # Quality warnings exist
        status = ReliabilityStatus.UNCERTAIN
        explanation = (
            f"Prediction is '{top_name}' ({top_prob:.1%}), but photograph had quality warnings "
            f"({', '.join(quality_report.warnings)}). Prediction should be independently verified."
        )
        return_class_index = top_idx
        return_class_name = top_name
    else:
        # Confident
        status = ReliabilityStatus.CONFIDENT
        explanation = (
            f"High-confidence diagnosis: '{top_name}' ({top_prob:.1%}). "
            f"Quality gate passed and distinct margin over alternative classes."
        )
        return_class_index = top_idx
        return_class_name = top_name

    return PredictionResult(
        class_index=return_class_index,
        class_name=return_class_name,
        probabilities=calibrated_probs,
        raw_probabilities=raw_arr,
        top_probability=round(top_prob, 4),
        raw_top_probability=round(raw_top_prob, 4),
        reliability=status,
        temperature=temperature,
        top_k=top_k_list,
        explanation=explanation,
        quality_report=quality_report,
        is_placeholder=False,
    )
