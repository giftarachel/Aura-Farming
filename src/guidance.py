"""
Aura Farming — src/guidance.py
=================================
Prevention and care guidance for detected diseases.

Phase 2 status: PLACEHOLDER
─────────────────────────────
This module will be fully implemented in Phase 5.

Content requirements (Phase 5):
  - Guidance must be sourced from authoritative agricultural
    publications (CABI, FAO, USDA, or peer-reviewed extension
    service resources).
  - Guidance must be general and precautionary — not specific
    pesticide dosages or guaranteed treatment claims.
  - Every guidance item must note: "Consult a qualified
    agricultural expert before applying any treatment."
  - Sources must be cited.
  - No guidance may be invented or extrapolated without a
    reliable reference.

Placeholder behaviour:
  get_guidance() returns a stub dict that identifies itself as
  placeholder content. The UI renders a development-state notice.
"""

from __future__ import annotations


def get_guidance(class_name: str) -> dict:
    """
    Return prevention and care guidance for a predicted disease class.

    Phase 2: Returns a placeholder dict for all class names.
    Phase 5: Will return sourced, carefully worded guidance.

    Args:
        class_name: Human-readable disease class name.

    Returns:
        dict with keys: class_name, guidance_points, disclaimer,
        source_citations, is_placeholder.
    """
    return {
        "class_name": class_name,
        "guidance_points": None,
        "disclaimer": (
            "Consult a qualified agricultural expert or plant pathologist "
            "before applying any treatment or management strategy."
        ),
        "source_citations": None,
        "is_placeholder": True,
    }
