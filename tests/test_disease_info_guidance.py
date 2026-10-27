"""
Aura Farming — tests/test_disease_info_guidance.py
===================================================
Unit tests for disease information (src/disease_information.py) and
prevention guidance (src/guidance.py).

Covers:
  - All 10 tomato conditions from class_mapping.json are present
  - Retrieval by display name, folder name, and integer index
  - Healthy vs. diseased classification distinction
  - Presence of symptoms, descriptions, and citations
  - Presence of immediate actions, prevention, and disclaimers
  - Verification that placeholder flag is False
  - Unknown class fallbacks

Run with:
    pytest tests/test_disease_info_guidance.py -v
"""

from __future__ import annotations

import json
from pathlib import Path
import pytest

from src.disease_information import (
    get_disease_info,
    get_all_disease_names,
    normalize_class_name,
)
from src.guidance import get_guidance


@pytest.fixture
def class_mapping() -> dict:
    mapping_path = Path(__file__).parent.parent / "configs" / "class_mapping.json"
    with open(mapping_path, "r", encoding="utf-8") as f:
        return json.load(f)


class TestDiseaseInformation:
    def test_all_ten_classes_covered(self, class_mapping):
        classes = class_mapping["classes"]
        assert len(classes) == 10

        for idx_str, info in classes.items():
            display_name = info["display_name"]
            folder_name = info["folder_name"]

            # Test by display name
            res_disp = get_disease_info(display_name)
            assert res_disp["is_placeholder"] is False
            assert res_disp["description"] is not None
            assert len(res_disp["symptoms"]) > 0
            assert len(res_disp["citations"]) > 0

            # Test by folder name
            res_folder = get_disease_info(folder_name)
            assert res_folder["class_name"] == display_name

            # Test by integer index
            res_idx = get_disease_info(int(idx_str))
            assert res_idx["class_name"] == display_name

    def test_healthy_vs_diseased_distinction(self):
        healthy_info = get_disease_info("Tomato — Healthy")
        assert healthy_info["is_healthy"] is True

        diseased_info = get_disease_info("Tomato — Early Blight")
        assert diseased_info["is_healthy"] is False

    def test_get_all_disease_names(self):
        names = get_all_disease_names()
        assert len(names) == 10
        assert "Tomato — Healthy" in names
        assert "Tomato — Late Blight" in names

    def test_unknown_class_fallback(self):
        res = get_disease_info("Nonexistent Potato Blight")
        assert res["condition"] == "Unknown"
        assert res["is_placeholder"] is False


class TestGuidance:
    def test_all_ten_classes_guidance_covered(self, class_mapping):
        classes = class_mapping["classes"]
        for idx_str, info in classes.items():
            display_name = info["display_name"]

            guidance = get_guidance(display_name)
            assert guidance["is_placeholder"] is False
            assert len(guidance["immediate_actions"]) > 0
            assert len(guidance["preventive_measures"]) > 0
            assert len(guidance["monitoring_advice"]) > 0
            assert "disclaimer" in guidance
            assert len(guidance["citations"]) > 0

    def test_healthy_guidance_distinction(self):
        g_healthy = get_guidance(9)
        assert g_healthy["is_healthy"] is True
        assert any("routine" in a.lower() or "maintain" in a.lower()
                   for a in g_healthy["immediate_actions"])

        g_late_blight = get_guidance(2)
        assert g_late_blight["is_healthy"] is False
        assert any("isolate" in a.lower() or "uproot" in a.lower()
                   for a in g_late_blight["immediate_actions"])

    def test_guidance_disclaimer_present(self):
        g = get_guidance("Tomato — Bacterial Spot")
        assert "Aura Farming is an educational research tool" in g["disclaimer"]
        assert "consult a qualified local agricultural extension officer" in g["disclaimer"].lower()

    def test_unknown_guidance_fallback(self):
        g = get_guidance("Unknown_Crop_123")
        assert g["condition"] == "Unknown"
        assert len(g["immediate_actions"]) > 0
