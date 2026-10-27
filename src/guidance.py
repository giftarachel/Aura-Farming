"""
Aura Farming — src/guidance.py
=================================
Agronomic care, integrated pest management (IPM), and prevention guidance.

Phase 5 Implementation: COMPLETE
─────────────────────────────────
Provides educational, precautionary, and science-backed agricultural guidance
for all 10 tomato conditions in the Aura Farming dataset.

Ethical & Scientific Policy:
  - Preserves farmer safety: does NOT provide prescriptive pesticide dosages or
    unverified chemical recipes.
  - Prioritizes cultural, mechanical, and preventive biological strategies.
  - Highlights standard plant pathology disclaimers and extension officer consultation.
"""

from __future__ import annotations

from typing import Union, Dict, Any, List

from src.disease_information import normalize_class_name


# ─────────────────────────────────────────────────────────────────
# Knowledge Base: Agronomic Guidance for 10 Tomato Conditions
# ─────────────────────────────────────────────────────────────────

_GUIDANCE_KNOWLEDGE_BASE: dict[str, dict[str, Any]] = {
    "Tomato — Bacterial Spot": {
        "condition": "Bacterial Spot",
        "is_healthy": False,
        "immediate_actions": [
            "Avoid entering fields or handling tomato plants when leaves are wet to prevent spreading bacteria on hands, clothes, or tools.",
            "Sanitize pruning shears, stakes, and equipment with a 10% bleach solution or 70% alcohol between plants.",
            "Prune and safely destroy (burn or deeply bury) heavily infected lower foliage to reduce inoculum load. Do not compost infected leaves."
        ],
        "preventive_measures": [
            "Use certified pathogen-free seeds and disease-resistant hybrid varieties where available.",
            "Switch strictly to drip irrigation or furrow watering; avoid overhead sprinkler systems that splash bacteria.",
            "Practice a minimum 2-to-3 year crop rotation with non-solanaceous crops (avoid peppers, eggplants, or potatoes in the cycle).",
            "Maintain wider spacing between rows and plants to enhance airflow and rapid canopy drying."
        ],
        "monitoring_advice": "Scout weekly, particularly after rainstorms or high humidity periods, inspecting lower leaves and developing fruit.",
        "citations": [
            "UC IPM: Pest Management Guidelines — Tomato Bacterial Spot",
            "CABI PlantwisePlus: Bacterial spot management advisory",
            "Cornell Cooperative Extension: Managing Bacterial Leaf Spot in Vegetables"
        ],
    },
    "Tomato — Early Blight": {
        "condition": "Early Blight",
        "is_healthy": False,
        "immediate_actions": [
            "Prune off affected lower leaves showing concentric target spots, taking care not to shake fungal spores onto neighboring leaves.",
            "Apply a clean organic or plastic mulch around the plant base to create a physical barrier preventing soilborne spores from splashing onto foliage.",
            "Stake and tie vines to keep foliage well above the ground level."
        ],
        "preventive_measures": [
            "Water at the base of plants early in the morning using drip lines so foliage remains completely dry.",
            "Practice a 3-year crop rotation away from all nightshade family members (Solanaceae).",
            "Remove and destroy all crop residues immediately after harvest; plow under deep if turning into soil.",
            "Select cultivars with documented tolerance or resistance to Alternaria solani."
        ],
        "monitoring_advice": "Check the lowest leaves once or twice weekly as fruit begins to set, as nutrient demand increases vulnerability.",
        "citations": [
            "Penn State Extension: Early Blight of Tomato Factsheet",
            "CABI PlantwisePlus: Alternaria solani control practices",
            "UC IPM: Tomato Early Blight Cultural Controls"
        ],
    },
    "Tomato — Late Blight": {
        "condition": "Late Blight",
        "is_healthy": False,
        "immediate_actions": [
            "CRITICAL: Late blight spreads explosively via windblown sporangia. Immediately isolate the affected area.",
            "Uproot and bag severely infected plants in plastic before carrying them out of the field to prevent aerial spore release.",
            "Do NOT add infected material to home or open compost piles.",
            "Alert neighboring growers and local agricultural extension agents, as regional tracking may be active."
        ],
        "preventive_measures": [
            "Plant certified disease-free transplants and certified seed potatoes (which can serve as early-season reservoir hosts).",
            "Destroy all volunteer tomato and potato plants near the plot in early spring.",
            "Maximize sun exposure and plant in well-drained, airy locations.",
            "Consider protective preventive organic fungicides (e.g. copper formulations) only upon advice from agricultural extension during high-risk cool, damp weather."
        ],
        "monitoring_advice": "Inspect fields every 2 to 3 days during cool (15°C–20°C), foggy, or rainy spells, paying close attention to leaf undersides.",
        "citations": [
            "FAO: Global Late Blight Alert & Integrated Management",
            "Cornell University Extension: Tomato Late Blight Management Strategy",
            "CABI PlantwisePlus: Phytophthora infestans advisory"
        ],
    },
    "Tomato — Leaf Mold": {
        "condition": "Leaf Mold",
        "is_healthy": False,
        "immediate_actions": [
            "Increase greenhouse or polytunnel ventilation immediately by opening sidewalls and vents, and using horizontal airflow fans.",
            "Carefully strip out affected lower leaves and remove them in closed containers from the tunnel.",
            "Increase spacing between potted or in-ground plants to reduce relative humidity within the canopy below 85%."
        ],
        "preventive_measures": [
            "Heat and ventilate greenhouses in late afternoon to prevent condensation forming on leaves as evening temperatures drop.",
            "Never use overhead watering; keep soil surface dry using drip lines covered with mulch.",
            "Choose tomato cultivars carrying specific Cladosporium fulvum resistance genes (Cf genes) tailored to local races."
        ],
        "monitoring_advice": "Examine the undersides of older foliage for olive-green velvety sporulation, especially in dense greenhouse stands.",
        "citations": [
            "Penn State Extension: Managing Leaf Mold in High Tunnel Tomatoes",
            "UC IPM: Tomato Leaf Mold Cultural Management",
            "CABI PlantwisePlus: Passalora fulva recommendations"
        ],
    },
    "Tomato — Septoria Leaf Spot": {
        "condition": "Septoria Leaf Spot",
        "is_healthy": False,
        "immediate_actions": [
            "Remove infected lower leaves showing characteristic black-speckled circular spots and dispose of them off-site.",
            "Apply mulch (straw, compost, or plastic film) under plants to stop raindrops from splashing overwintering spores from the soil.",
            "Stake plants upright to improve ventilation and sunlight penetration."
        ],
        "preventive_measures": [
            "Enforce a strict 2-to-3 year crop rotation away from solanaceous crops and control solanaceous weeds (e.g. horsenettle, black nightshade).",
            "Clear out all end-of-season plant debris thoroughly, as Septoria overwinters on dry infected tomato stalks.",
            "Allow generous spacing (at least 60–90 cm / 2–3 feet between plants) for rapid drying after rainfall."
        ],
        "monitoring_advice": "Begin scouting lower leaves immediately after the first seasonal rainfall or heavy dews.",
        "citations": [
            "Cornell Cooperative Extension: Septoria Leaf Spot of Tomato",
            "CABI PlantwisePlus: Septoria lycopersici pest management",
            "UC IPM: Septoria Leaf Spot Cultural Controls"
        ],
    },
    "Tomato — Spider Mites (Two-spotted)": {
        "condition": "Spider Mites (Two-spotted)",
        "is_healthy": False,
        "immediate_actions": [
            "Hose down infested foliage with a strong stream of clean water to knock mites and webbing off leaf undersides.",
            "Isolate heavily infested plants; prune off and discard severely webbed branch tips.",
            "Avoid broad-spectrum synthetic insecticides (e.g. pyrethroids), which kill natural predator mites and trigger rapid spider mite flare-ups."
        ],
        "preventive_measures": [
            "Keep farm roadways and field paths wet or vegetated to suppress dust, as dusty conditions favor mite outbreaks.",
            "Encourage or release biological control agents such as predatory mites (Phytoseiulus persimilis, Neoseiulus californicus) and lady beetles.",
            "Maintain optimal plant irrigation; drought-stressed tomato plants are significantly more attractive to spider mites.",
            "Consider insecticidal soap or horticultural oil sprays during early infestation, following extension guidance."
        ],
        "monitoring_advice": "Regularly inspect leaf undersides with a 10× hand lens, looking for tiny moving yellow/green mites with two dark lateral spots.",
        "citations": [
            "UC IPM: Spider Mites in Tomatoes & Field Crops",
            "FAO: Biological Control of Spider Mites in Vegetable Production",
            "CABI PlantwisePlus: Tetranychus urticae management"
        ],
    },
    "Tomato — Target Spot": {
        "condition": "Target Spot",
        "is_healthy": False,
        "immediate_actions": [
            "Prune out symptomatic leaves showing target-like lesions to reduce airborne conidia production.",
            "Improve field drainage and avoid working in wet fields to prevent spore movement.",
            "Collect and dispose of any fallen diseased fruit immediately."
        ],
        "preventive_measures": [
            "Ensure wide row spacing and north-south row orientation for maximum sun penetration and air movement.",
            "Use drip irrigation under plastic mulch to prevent splashing soil particles.",
            "Rotate crops with non-hosts (corn, sorghum, or grasses) for at least 2 consecutive seasons.",
            "Maintain balanced nitrogen fertilization; excess nitrogen creates a dense, humid microclimate that exacerbates target spot."
        ],
        "monitoring_advice": "Monitor lower and middle leaves in warm, humid weather, checking both leaf surfaces and stem joints.",
        "citations": [
            "University of Florida IFAS: Target Spot of Tomato Management",
            "CABI PlantwisePlus: Corynespora cassiicola guidelines",
            "UC IPM: Tomato Target Spot Cultural Controls"
        ],
    },
    "Tomato — Yellow Leaf Curl Virus": {
        "condition": "Yellow Leaf Curl Virus",
        "is_healthy": False,
        "immediate_actions": [
            "Immediately rogued (pull up and destroy) infected plants showing yellow cupped leaves to prevent them from acting as virus reservoirs.",
            "Place pulled plants into sealed plastic bags before removal to trap adult whiteflies on the foliage.",
            "Install yellow sticky traps in and around the field to monitor and mass-trap adult whiteflies."
        ],
        "preventive_measures": [
            "Plant TYLCV-resistant or tolerant hybrid tomato cultivars (the single most effective defense against the virus).",
            "Use 50-mesh insect-exclusion netting in nursery seedbeds and greenhouse entries.",
            "Apply reflective silver plastic mulches in the field to repel migrating whiteflies.",
            "Implement a regional host-free period (crop holiday) of at least 2 months between tomato growing seasons to break the whitefly-virus cycle."
        ],
        "monitoring_advice": "Check the undersides of young leaves weekly for tiny whitefly adults and nymphs, using yellow sticky cards for early detection.",
        "citations": [
            "FAO: Strategy for Management of Tomato Yellow Leaf Curl Virus",
            "UC IPM: Management of TYLCV and Silverleaf Whitefly Vector",
            "CABI PlantwisePlus: Tomato yellow leaf curl virus IPM"
        ],
    },
    "Tomato — Mosaic Virus": {
        "condition": "Tomato Mosaic Virus",
        "is_healthy": False,
        "immediate_actions": [
            "Carefully rogue and destroy infected plants showing mosaic mottling or fern-like leaf distortion.",
            "Wash hands thoroughly with soap and warm water or skim milk before and after handling plants.",
            "Prohibit tobacco use (smoking, chewing) anywhere near tomato plantings, as tobacco products frequently harbor infectious mosaic viruses.",
            "Disinfect stakes, trellis wires, and pruning tools in trisodium phosphate (TSP) or 20% non-fat dry milk solution."
        ],
        "preventive_measures": [
            "Plant certified virus-free seed or use cultivars with genetic resistance to ToMV (e.g. Tm-2 or Tm-2^2 resistance genes).",
            "Use hot-water treated seed or commercial bleach seed sanitation protocols.",
            "Rotate fields away from solanaceous crops for at least 2 seasons.",
            "Minimize unnecessary handling, pruning, or brushing against plants when moving down rows."
        ],
        "monitoring_advice": "Scout during early vegetative growth for mosaic patterning, leaf distortion, or stunted shoots.",
        "citations": [
            "Penn State Extension: Tomato Mosaic Virus and Tobacco Mosaic Virus",
            "UC IPM: Tomato Mosaic Virus Sanitation Protocols",
            "CABI PlantwisePlus: Tomato mosaic virus advisory"
        ],
    },
    "Tomato — Healthy": {
        "condition": "Healthy",
        "is_healthy": True,
        "immediate_actions": [
            "Maintain current irrigation and cultural management practices.",
            "Ensure regular observation to catch any early pest or disease signs before symptoms become severe."
        ],
        "preventive_measures": [
            "Continue drip or soaker hose irrigation to maintain dry foliage and prevent fungal spore germination.",
            "Maintain balanced fertilization based on periodic soil testing; avoid excessive nitrogen which can cause soft, disease-susceptible foliage.",
            "Ensure adequate spacing and airflow throughout the canopy.",
            "Keep surrounding field edges free of alternate weed hosts."
        ],
        "monitoring_advice": "Routine scouting every 5 to 7 days is sufficient during normal growing conditions.",
        "citations": [
            "FAO: Good Agricultural Practices (GAP) for Sustainable Tomato Cultivation",
            "UC Davis: Production Guidelines for Fresh Market and Processing Tomatoes"
        ],
    },
}

DEFAULT_DISCLAIMER = (
    "Aura Farming is an educational research tool. Agricultural guidance provided here "
    "is for informational and preventive purposes only and does not constitute a guaranteed "
    "diagnosis or chemical prescription. Always consult a qualified local agricultural extension "
    "officer, agronomist, or plant pathologist before applying treatments or making crop management decisions."
)


# ─────────────────────────────────────────────────────────────────
# Public Interface
# ─────────────────────────────────────────────────────────────────

def get_guidance(class_identifier: Union[str, int]) -> dict[str, Any]:
    """
    Return agronomic prevention and management guidance for a tomato condition.

    Args:
        class_identifier: Class display name, folder name, or integer index (0–9).

    Returns:
        dict with keys:
            class_name: Standardized class display name
            condition: Condition name
            is_healthy: Boolean
            immediate_actions: List of practical non-chemical immediate actions
            preventive_measures: List of long-term IPM prevention measures
            monitoring_advice: Field scouting guidelines
            disclaimer: Official agricultural disclaimer
            source_citations: Literature references
            is_placeholder: False
    """
    canonical_name = normalize_class_name(class_identifier)

    if canonical_name in _GUIDANCE_KNOWLEDGE_BASE:
        data = dict(_GUIDANCE_KNOWLEDGE_BASE[canonical_name])
        data["class_name"] = canonical_name
        data["disclaimer"] = DEFAULT_DISCLAIMER
        data["is_placeholder"] = False
        return data

    # Fallback for unrecognized class
    return {
        "class_name": str(class_identifier),
        "condition": "Unknown",
        "is_healthy": False,
        "immediate_actions": [
            "Isolate symptomatic leaves or plants.",
            "Consult your local agricultural extension service for disease diagnosis."
        ],
        "preventive_measures": [
            "Maintain clean agronomic practices, crop rotation, and avoid overhead watering."
        ],
        "monitoring_advice": "Regularly monitor crop foliage for lesion expansion.",
        "disclaimer": DEFAULT_DISCLAIMER,
        "source_citations": [],
        "is_placeholder": False,
    }
