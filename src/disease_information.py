"""
Aura Farming — src/disease_information.py
============================================
Disease class descriptions and observable symptom information.

Phase 2 status: PLACEHOLDER
─────────────────────────────
This module will be fully implemented in Phase 5.

Content requirements (Phase 5):
  - Descriptions must be sourced from trustworthy agricultural
    references (e.g., CABI, FAO, university extension services).
  - Each description must cite its source.
  - No disease facts or treatment claims may be invented.
  - Content is appropriate for tomato diseases only in the
    initial version.

Placeholder behaviour:
  get_disease_info() returns a minimal stub dict that identifies
  itself as placeholder content. The UI renders this as a
  development-state notice rather than fake disease facts.
"""

from __future__ import annotations


# ─────────────────────────────────────────────────────────────────
# Public interface
# ─────────────────────────────────────────────────────────────────

def get_disease_info(class_name: str) -> dict:
    """
    Return disease description information for a predicted class.

    Phase 2: Returns a placeholder dict for all class names.
    Phase 5: Will return a structured dict with description,
             symptoms, affected_crops, and source_citation.

    Args:
        class_name: Human-readable disease class name, e.g.
                    "Tomato — Early Blight".

    Returns:
        dict with keys: class_name, description, symptoms,
        source_citation, is_placeholder.
    """
    return {
        "class_name": class_name,
        "description": None,
        "symptoms": None,
        "source_citation": None,
        "is_placeholder": True,
    }
