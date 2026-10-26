"""
Aura Farming — src/calibration.py
====================================
Probability calibration and confidence-based abstention.

Phase 2 status: PLACEHOLDER
─────────────────────────────
This module will be fully implemented in Phase 5.

Key concepts (explained simply):

CALIBRATION:
  A neural network's softmax output of 0.90 does not necessarily
  mean it is correct 90% of the time. Temperature scaling adjusts
  these probabilities so they better reflect actual accuracy.
  This is done by dividing the model's raw scores (logits) by a
  single learned scalar T before the softmax. When T > 1, the
  probabilities become less extreme and better reflect real accuracy.

ABSTENTION:
  Rather than always returning the top class, the system checks
  whether the highest calibrated probability exceeds a threshold τ.
  If it does not, the system returns an "uncertain" status instead.
  τ is chosen on the validation set to achieve ~85% selective
  accuracy as a research target. The actual value and the resulting
  coverage/accuracy trade-off will be reported honestly in Phase 7.

Planned functionality (Phase 5):
- Load the temperature parameter T from the calibration JSON file.
- Apply temperature scaling to a raw logit vector.
- Compare the top calibrated probability against τ.
- Return a structured PredictionResult with a reliability status.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional

import numpy as np


# ─────────────────────────────────────────────────────────────────
# Reliability status
# ─────────────────────────────────────────────────────────────────

class ReliabilityStatus(str, Enum):
    """
    The three possible reliability states for a prediction.

    CONFIDENT:  Top calibrated probability ≥ threshold τ.
                The prediction label is shown to the user.
    UNCERTAIN:  Top calibrated probability < threshold τ but
                above a lower floor. The label is shown with
                a clear low-confidence warning.
    ABSTAINED:  The model cannot produce a reliable prediction.
                No disease label is shown. The user is advised
                to retake the photograph or consult an expert.
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
    Complete structured output from the prediction + calibration pipeline.

    Attributes:
        class_index: Integer index of the predicted class (or None).
        class_name: Human-readable label (or None if abstained).
        probabilities: Full calibrated probability vector.
        top_probability: Highest probability value (calibrated).
        reliability: ReliabilityStatus enum value.
        is_placeholder: True while Phase 5 is not yet implemented.
    """
    class_index: Optional[int] = None
    class_name: Optional[str] = None
    probabilities: Optional[np.ndarray] = None
    top_probability: Optional[float] = None
    reliability: ReliabilityStatus = ReliabilityStatus.ABSTAINED
    is_placeholder: bool = True


# ─────────────────────────────────────────────────────────────────
# Public interface
# ─────────────────────────────────────────────────────────────────

def apply_calibration_and_threshold(
    raw_probabilities: np.ndarray,
    class_names: dict[int, str],
) -> PredictionResult:
    """
    Apply calibration and confidence threshold to raw model probabilities.

    Phase 2: Returns a placeholder PredictionResult with ABSTAINED status.
    Phase 5: Will load calibration parameters, apply temperature scaling,
             compare to threshold τ, and return a genuine result.

    Args:
        raw_probabilities: NumPy array of shape (num_classes,) from
                           the model's softmax output.
        class_names: Mapping of integer index to display label.

    Returns:
        A PredictionResult describing the outcome.
    """
    # Phase 2 placeholder — implementation deferred to Phase 5.
    return PredictionResult(is_placeholder=True)
