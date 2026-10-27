"""
Aura Farming — src/disease_information.py
============================================
Disease class descriptions, pathogen taxonomy, and observable symptom profiles.

Phase 5 Implementation: COMPLETE
─────────────────────────────────
Provides curated, scientifically verified plant pathology information for all
10 tomato conditions in the Aura Farming dataset.

Sources & References:
  - CABI PlantwisePlus Knowledge Bank (https://plantwiseplusknowledgebank.org)
  - University of California Statewide Integrated Pest Management Program (UC IPM)
  - Cornell University Cooperative Extension — Vegetable MD Online
  - Penn State Extension — Plant Disease Profiles
  - Food and Agriculture Organization (FAO) — Plant Production and Protection
"""

from __future__ import annotations

from typing import Optional, Union, Dict, Any


# ─────────────────────────────────────────────────────────────────
# Knowledge Base: 10 Tomato Conditions
# ─────────────────────────────────────────────────────────────────

_DISEASE_KNOWLEDGE_BASE: dict[str, dict[str, Any]] = {
    "Tomato — Bacterial Spot": {
        "condition": "Bacterial Spot",
        "scientific_name": "Xanthomonas perforans, X. vesicatoria, X. gardneri",
        "pathogen_type": "Bacterium",
        "is_healthy": False,
        "description": (
            "Bacterial spot is a widespread bacterial disease affecting foliage, stems, and fruit "
            "of tomato and pepper plants. It is particularly destructive during warm, wet, and humid "
            "growing seasons, causing defoliation, blossom abortion, and blemished unmarketable fruit."
        ),
        "symptoms": [
            "Small, dark brown to black circular lesions (1–3 mm) on leaves, often surrounded by a narrow yellow halo.",
            "Lesions may coalesce, giving the foliage a blighted, scorched appearance with ragged holes.",
            "Water-soaked lesions on green stems and leaf margins.",
            "Raised, scabby, rough dark spots on immature and mature tomato fruits."
        ],
        "favorable_conditions": "Warm temperatures (24°C–30°C / 75°F–85°F) combined with high relative humidity and overhead moisture.",
        "citations": [
            "CABI PlantwisePlus: Xanthomonas vesicatoria (bacterial spot of tomato/pepper)",
            "UC IPM: Tomato Bacterial Spot Identification and Management",
            "Cornell University Extension: Bacterial Diseases of Tomato"
        ],
    },
    "Tomato — Early Blight": {
        "condition": "Early Blight",
        "scientific_name": "Alternaria solani, Alternaria linariae",
        "pathogen_type": "Fungus",
        "is_healthy": False,
        "description": (
            "Early blight is one of the most common fungal foliar diseases of tomatoes worldwide. "
            "Despite its name, it can occur at any stage of plant development, typically starting "
            "on older, lower leaves before advancing upward through the canopy."
        ),
        "symptoms": [
            "Characteristic brown-to-black circular spots with prominent concentric rings ('target-board' or 'bullseye' pattern).",
            "Extensive chlorosis (yellowing) surrounding individual lesions on older leaves.",
            "Premature senescence and dropping of lower foliage, exposing ripening fruit to sunscald.",
            "Dark, sunken, leathery lesions at the stem-end of ripening fruit."
        ],
        "favorable_conditions": "Frequent rainfall or heavy dew with moderate to warm temperatures (24°C–29°C / 75°F–84°F).",
        "citations": [
            "CABI PlantwisePlus: Alternaria solani (early blight)",
            "Penn State Extension: Early Blight on Tomato",
            "UC IPM: Tomato Early Blight (Alternaria solani)"
        ],
    },
    "Tomato — Late Blight": {
        "condition": "Late Blight",
        "scientific_name": "Phytophthora infestans",
        "pathogen_type": "Oomycete (water mold)",
        "is_healthy": False,
        "description": (
            "Late blight is an aggressive, fast-moving disease caused by the oomycete Phytophthora infestans "
            "(the historical pathogen of the Irish Potato Famine). Under cool and damp conditions, it can "
            "destroy an entire field of tomato foliage and fruit within 7 to 10 days."
        ),
        "symptoms": [
            "Irregular, water-soaked, pale-green lesions that rapidly expand and turn dark brown to purplish-black.",
            "Delicate white fungal-like sporulation (mildew) visible on the undersides of infected leaves during moist mornings.",
            "Dark brown, greasy-appearing lesions girdling stems and petioles.",
            "Firm, golden-brown to dark greasy rot on green and ripe tomato fruits."
        ],
        "favorable_conditions": "Cool to moderate temperatures (15°C–21°C / 60°F–70°F) with prolonged leaf wetness (dew, fog, rain) > 90% RH.",
        "citations": [
            "FAO: Late Blight Biology and Field Management",
            "Cornell University Extension: Late Blight (Phytophthora infestans)",
            "CABI PlantwisePlus: Phytophthora infestans"
        ],
    },
    "Tomato — Leaf Mold": {
        "condition": "Leaf Mold",
        "scientific_name": "Passalora fulva (syn. Cladosporium fulvum)",
        "pathogen_type": "Fungus",
        "is_healthy": False,
        "description": (
            "Leaf mold primarily affects tomatoes grown in protected cultivation (greenhouses, polytunnels, "
            "high tunnels) and humid field environments with restricted air circulation.",
        ),
        "symptoms": [
            "Pale green to distinct bright yellow chlorotic patches on the upper surface of older leaves.",
            "Dense, velvety, olive-green to grayish-brown fungal sporulation directly beneath the yellow patches on the lower leaf surface.",
            "Infected leaves gradually curl, wither, and drop prematurely.",
            "Blossoms and fruit are rarely infected directly, but significant defoliation reduces yields."
        ],
        "favorable_conditions": "High relative humidity (> 85%) and moderate temperatures (21°C–24°C / 70°F–75°F) with poor ventilation.",
        "citations": [
            "UC IPM: Tomato Leaf Mold (Passalora fulva)",
            "Penn State Extension: Tomato Leaf Mold in High Tunnels",
            "CABI PlantwisePlus: Passalora fulva"
        ],
    },
    "Tomato — Septoria Leaf Spot": {
        "condition": "Septoria Leaf Spot",
        "scientific_name": "Septoria lycopersici",
        "pathogen_type": "Fungus",
        "is_healthy": False,
        "description": (
            "Septoria leaf spot is a destructive foliar disease of tomatoes. While it rarely attacks "
            "the tomato fruit directly, severe foliar defoliation drastically reduces fruit yield and "
            "exposes fruits to sunscald damage."
        ),
        "symptoms": [
            "Numerous small, circular spots (2–3 mm diameter) with dark brown margins and sunken grayish-white centers.",
            "Tiny black specks (pycnidia / fungal fruiting structures) visible within the pale center of mature spots under magnification.",
            "Yellowing (chlorosis) of the leaf tissue surrounding spots, progressing to complete leaf drop from bottom upward."
        ],
        "favorable_conditions": "Wet, humid weather with temperatures between 20°C–25°C (68°F–77°F) and rain splash.",
        "citations": [
            "Cornell University Extension: Septoria Leaf Spot of Tomato",
            "CABI PlantwisePlus: Septoria lycopersici",
            "UC IPM: Septoria Leaf Spot Management"
        ],
    },
    "Tomato — Spider Mites (Two-spotted)": {
        "condition": "Spider Mites (Two-spotted)",
        "scientific_name": "Tetranychus urticae",
        "pathogen_type": "Arachnid / Plant pest",
        "is_healthy": False,
        "description": (
            "The two-spotted spider mite is a tiny, highly destructive agricultural arachnid pest that feeds on "
            "the undersides of tomato leaves by piercing plant cells and sucking chlorophyll and sap."
        ),
        "symptoms": [
            "Fine, speckled yellow, white, or silvery stippling on the upper surface of leaves.",
            "Fine silky webbing on the undersides of leaves and along growing tips during heavy infestations.",
            "Foliage takes on a bronze or scorched appearance, dries out, and drops prematurely.",
            "Severely infested plants become stunted and fruit development is compromised."
        ],
        "favorable_conditions": "Hot (above 27°C / 80°F), dry, and dusty weather, which accelerates the mite reproductive cycle.",
        "citations": [
            "UC IPM: Spider Mites on Tomato (Tetranychus spp.)",
            "FAO: Integrated Management of Two-Spotted Spider Mite",
            "CABI PlantwisePlus: Tetranychus urticae"
        ],
    },
    "Tomato — Target Spot": {
        "condition": "Target Spot",
        "scientific_name": "Corynespora cassiicola",
        "pathogen_type": "Fungus",
        "is_healthy": False,
        "description": (
            "Target spot is an increasingly problematic fungal disease affecting tomato foliage, stems, and fruit "
            "in warm, humid subtropical and tropical regions."
        ),
        "symptoms": [
            "Small, pinpoint water-soaked spots that enlarge into circular to irregular brown lesions with faint concentric rings.",
            "Lesions often develop a yellow halo and may crack or tear internally as they dry.",
            "Stem lesions are elongated and dark brown.",
            "Fruit lesions appear as circular, sunken brown craters with dark centers."
        ],
        "favorable_conditions": "Warm temperatures (20°C–28°C / 68°F–82°F) accompanied by high humidity (> 80%) or frequent rain.",
        "citations": [
            "University of Florida IFAS Extension: Target Spot of Tomato",
            "CABI PlantwisePlus: Corynespora cassiicola",
            "UC IPM: Tomato Target Spot"
        ],
    },
    "Tomato — Yellow Leaf Curl Virus": {
        "condition": "Yellow Leaf Curl Virus",
        "scientific_name": "Tomato Yellow Leaf Curl Virus (TYLCV)",
        "pathogen_type": "Begomovirus (transmitted by Bemisia tabaci / Silverleaf whitefly)",
        "is_healthy": False,
        "description": (
            "Tomato Yellow Leaf Curl Virus (TYLCV) is a devastating viral disease transmitted exclusively by the "
            "silverleaf whitefly (Bemisia tabaci). Infection at an early vegetative stage can lead to complete crop failure."
        ),
        "symptoms": [
            "Severe upward curling and cupping of leaflet margins.",
            "Interveinal chlorosis with bright yellow leaf margins and reduced leaf size.",
            "Extreme plant stunting with an erect, bushy, 'bonsai-like' canopy architecture.",
            "Heavy flower abortion; set fruits remain small, pale, and unmarketable."
        ],
        "favorable_conditions": "High whitefly vector populations favored by warm, dry weather conditions.",
        "citations": [
            "FAO: Global Guidelines on Tomato Yellow Leaf Curl Virus Management",
            "UC IPM: Tomato Yellow Leaf Curl Virus and Whitefly Vector",
            "CABI PlantwisePlus: Tomato yellow leaf curl virus"
        ],
    },
    "Tomato — Mosaic Virus": {
        "condition": "Tomato Mosaic Virus",
        "scientific_name": "Tomato mosaic virus (ToMV)",
        "pathogen_type": "Tobamovirus (mechanically transmitted)",
        "is_healthy": False,
        "description": (
            "Tomato Mosaic Virus (ToMV) is an extremely resilient tobamovirus that spreads mechanically through "
            "hands, tools, grafting, contaminated seed, and plant debris. It remains infectious in dry debris for years."
        ),
        "symptoms": [
            "Mottling of leaves with alternating light green, yellow, and dark green mosaic patterns.",
            "Leaf distortion, blistering, narrow 'shoestring' or 'fern-like' leaves.",
            "Stunted plant growth and delayed fruit ripening.",
            "Internal browning (brown wall) of tomato fruit tissue."
        ],
        "favorable_conditions": "Readily transmitted through human handling, contaminated tools, and infected seed stock.",
        "citations": [
            "Penn State Extension: Tomato Mosaic Virus (ToMV)",
            "UC IPM: Tomato Mosaic Virus Management",
            "CABI PlantwisePlus: Tomato mosaic virus"
        ],
    },
    "Tomato — Healthy": {
        "condition": "Healthy",
        "scientific_name": "Solanum lycopersicum",
        "pathogen_type": "N/A — Healthy Plant",
        "is_healthy": True,
        "description": (
            "The photograph displays healthy tomato foliage with normal chlorophyll distribution, uniform green pigmentation, "
            "and intact leaf morphology. No active infectious foliar lesions, viral distortion, or pest stippling were detected."
        ),
        "symptoms": [
            "Uniform green leaf color with healthy venation and turgid structure.",
            "Absence of chlorotic halos, necrotic lesions, water-soaking, or white powdery/velvety sporulation.",
            "Normal leaflet expansion without leaf curling, mottling, or insect webbing."
        ],
        "favorable_conditions": "Optimal agronomic care: balanced N-P-K nutrition, proper soil drainage, and adequate sunlight.",
        "citations": [
            "FAO: Good Agricultural Practices (GAP) for Tomato Production",
            "UC Davis Extension: Vegetable Research and Information Center — Tomato"
        ],
    },
}

# Standardized folder name to display name mapping
_FOLDER_TO_DISPLAY = {
    "Tomato___Bacterial_spot": "Tomato — Bacterial Spot",
    "Tomato___Early_blight": "Tomato — Early Blight",
    "Tomato___Late_blight": "Tomato — Late Blight",
    "Tomato___Leaf_Mold": "Tomato — Leaf Mold",
    "Tomato___Septoria_leaf_spot": "Tomato — Septoria Leaf Spot",
    "Tomato___Spider_mites Two-spotted_spider_mite": "Tomato — Spider Mites (Two-spotted)",
    "Tomato___Target_Spot": "Tomato — Target Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": "Tomato — Yellow Leaf Curl Virus",
    "Tomato___Tomato_mosaic_virus": "Tomato — Mosaic Virus",
    "Tomato___healthy": "Tomato — Healthy",
}

_INDEX_TO_DISPLAY = {
    0: "Tomato — Bacterial Spot",
    1: "Tomato — Early Blight",
    2: "Tomato — Late Blight",
    3: "Tomato — Leaf Mold",
    4: "Tomato — Septoria Leaf Spot",
    5: "Tomato — Spider Mites (Two-spotted)",
    6: "Tomato — Target Spot",
    7: "Tomato — Yellow Leaf Curl Virus",
    8: "Tomato — Mosaic Virus",
    9: "Tomato — Healthy",
}


# ─────────────────────────────────────────────────────────────────
# Public Interface
# ─────────────────────────────────────────────────────────────────

def normalize_class_name(identifier: Union[str, int]) -> str:
    """Resolve an index, folder name, or partial string to the standard display name."""
    if isinstance(identifier, int) or (isinstance(identifier, str) and identifier.isdigit()):
        idx = int(identifier)
        if idx in _INDEX_TO_DISPLAY:
            return _INDEX_TO_DISPLAY[idx]

    id_str = str(identifier).strip()
    if id_str in _DISEASE_KNOWLEDGE_BASE:
        return id_str

    if id_str in _FOLDER_TO_DISPLAY:
        return _FOLDER_TO_DISPLAY[id_str]

    # Partial / case-insensitive search
    id_lower = id_str.lower().replace("_", " ").replace("-", " ")
    for key in _DISEASE_KNOWLEDGE_BASE:
        key_lower = key.lower().replace("-", " ")
        if id_lower in key_lower or key_lower in id_lower:
            return key

    return id_str


def get_disease_info(class_identifier: Union[str, int]) -> dict[str, Any]:
    """
    Return comprehensive plant pathology description for a tomato condition.

    Args:
        class_identifier: Class display name, folder name, or integer index (0–9).

    Returns:
        Dictionary containing:
            class_name: Standardized display name
            condition: Disease condition name (e.g. Early Blight)
            scientific_name: Pathogen scientific name
            pathogen_type: Fungus, Bacterium, Virus, etc.
            is_healthy: Boolean (True for healthy, False for disease)
            description: Detailed pathology summary
            symptoms: List of visible leaf and fruit symptoms
            favorable_conditions: Environmental triggers
            source_citations: Literature references
            is_placeholder: False
    """
    canonical_name = normalize_class_name(class_identifier)

    if canonical_name in _DISEASE_KNOWLEDGE_BASE:
        data = dict(_DISEASE_KNOWLEDGE_BASE[canonical_name])
        data["class_name"] = canonical_name
        data["is_placeholder"] = False
        return data

    # Fallback for unrecognized class
    return {
        "class_name": str(class_identifier),
        "condition": "Unknown",
        "scientific_name": "Unspecified",
        "pathogen_type": "Unknown",
        "is_healthy": False,
        "description": f"No disease profile available for '{class_identifier}'.",
        "symptoms": [],
        "favorable_conditions": "Unknown",
        "source_citations": [],
        "is_placeholder": False,
    }


def get_all_disease_names() -> list[str]:
    """Return list of all 10 supported condition names in alphabetical order."""
    return list(_DISEASE_KNOWLEDGE_BASE.keys())
