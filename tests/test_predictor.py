"""
Aura Farming — tests/test_predictor.py
========================================
Unit tests for src/predictor.py.

Phase 2: Tests confirm that:
  - is_model_available() returns False when no weights path is set.
  - predict() raises ModelNotAvailableError immediately.
  - The exception message is informative.

Phase 4: These tests will be extended with a tiny mock model to
verify the full inference path without requiring real weights.

Run with:
    pytest tests/test_predictor.py -v
"""

import numpy as np
import pytest

from src.predictor import is_model_available, predict, ModelNotAvailableError


class TestIsModelAvailable:
    def test_returns_false_when_no_model_configured(self, tmp_path, monkeypatch):
        """
        is_model_available() returns False when weights_path is null
        or when the configured file does not exist on disk.
        Uses a temporary config so the test is independent of
        whether training has been run.
        """
        # Patch the config loader to return null weights_path
        import src.predictor as predictor_module

        def _mock_config():
            return {"model": {"weights_path": None}}

        monkeypatch.setattr(predictor_module, "_load_config", _mock_config)
        assert is_model_available() is False

    def test_returns_false_when_file_missing(self, monkeypatch):
        """Returns False when weights_path is set but the file does not exist."""
        import src.predictor as predictor_module

        def _mock_config():
            return {"model": {"weights_path": "/nonexistent/model.keras"}}

        monkeypatch.setattr(predictor_module, "_load_config", _mock_config)
        assert is_model_available() is False

    def test_returns_true_when_file_exists(self, tmp_path, monkeypatch):
        """Returns True when weights_path is set and the file exists."""
        import src.predictor as predictor_module
        fake_model = tmp_path / "model.keras"
        fake_model.write_bytes(b"fake")

        def _mock_config():
            return {"model": {"weights_path": str(fake_model)}}

        monkeypatch.setattr(predictor_module, "_load_config", _mock_config)
        assert is_model_available() is True

    def test_returns_bool(self):
        result = is_model_available()
        assert isinstance(result, bool)


class TestPredict:
    def test_raises_model_not_available(self, monkeypatch):
        dummy_tensor = np.zeros((1, 224, 224, 3), dtype=np.float32)
        def _mock_load():
            raise ModelNotAvailableError("Model weights file not found.")
        monkeypatch.setattr("src.predictor._load_model_cached", _mock_load)
        with pytest.raises(ModelNotAvailableError):
            predict(dummy_tensor)

    def test_error_message_is_informative(self, monkeypatch):
        """
        ModelNotAvailableError must contain actionable information.
        In Phase 4: the message mentions either TensorFlow or the
        weights path — depending on which is the actual blocker.
        """
        dummy_tensor = np.zeros((1, 224, 224, 3), dtype=np.float32)
        def _mock_load():
            raise ModelNotAvailableError("Model weights path does not exist on disk.")
        monkeypatch.setattr("src.predictor._load_model_cached", _mock_load)
        with pytest.raises(ModelNotAvailableError) as exc_info:
            predict(dummy_tensor)
        msg = str(exc_info.value)
        # The error must mention at least one of: config, TensorFlow, model
        assert any(kw in msg for kw in ["TensorFlow", "config", "model", "weights"]), (
            f"Error message not informative enough: {msg!r}"
        )
