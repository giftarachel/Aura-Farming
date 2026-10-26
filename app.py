"""
Aura Farming — app.py
========================
Streamlit entry point and homepage.

Run with:
    streamlit run app.py

This is Page 1 of the application: the homepage / landing page.
It establishes the Aura Farming brand, explains the system,
guides the user to the detection page, and sets the visual tone.

No AI inference occurs on this page.
"""

import streamlit as st

# Page configuration must be the very first Streamlit call
st.set_page_config(
    page_title="Aura Farming — Crop Disease Detection",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="collapsed",
    menu_items={
        "Get Help": None,
        "Report a bug": None,
        "About": (
            "Aura Farming — An AI-Powered, Reliability-Aware Crop Disease "
            "Detection System. Research prototype."
        ),
    },
)

from src.ui_components import (
    Colors,
    BRAND_NAME,
    TAGLINE,
    HEADLINE,
    inject_global_css,
    render_navbar,
    render_section_heading,
    render_disclaimer,
    render_feature_card,
    render_workflow_step,
    render_footer,
    get_logo_svg,
)

# ─────────────────────────────────────────────────────────────────
# Global styles
# ─────────────────────────────────────────────────────────────────
inject_global_css()

# ─────────────────────────────────────────────────────────────────
# Navigation bar
# ─────────────────────────────────────────────────────────────────
render_navbar(active_page="Home")

# ─────────────────────────────────────────────────────────────────
# Hero section
# ─────────────────────────────────────────────────────────────────
hero_left, hero_right = st.columns([3, 2], gap="large")

with hero_left:
    # Tagline badge
    st.markdown(
        f'<span class="af-tagline">🌱 &nbsp;{TAGLINE}</span>',
        unsafe_allow_html=True,
    )
    # Headline
    st.markdown(
        f'<h1 class="af-hero-headline">{HEADLINE}</h1>',
        unsafe_allow_html=True,
    )
    # Description
    st.markdown(
        f"""
        <p style="font-size:1.05rem; color:{Colors.DARK_EARTH};
                  max-width:540px; margin-bottom:1.8rem; line-height:1.7;">
          Aura Farming uses artificial intelligence and computer vision to
          help identify potential crop diseases from leaf photographs,
          assess prediction reliability, and provide understandable results.
        </p>
        """,
        unsafe_allow_html=True,
    )
    # Call-to-action buttons
    cta_col, secondary_col, _ = st.columns([1.6, 1.6, 2])
    with cta_col:
        if st.button("🔍 &nbsp;Detect Crop Disease", use_container_width=True):
            st.switch_page("pages/1_Detect_Disease.py")
    with secondary_col:
        if st.button("ℹ️ &nbsp;How It Works", use_container_width=True):
            st.switch_page("pages/2_About_Aura_Farming.py")

with hero_right:
    # Botanical hero illustration using pure CSS/SVG
    st.markdown(
        f"""
        <div style="
            background: linear-gradient(
                145deg,
                {Colors.FOREST_GREEN}22 0%,
                {Colors.SAGE_GREEN}33 50%,
                {Colors.WARM_BEIGE}44 100%
            );
            border-radius: 20px;
            border: 1px solid {Colors.WARM_BEIGE};
            padding: 3rem 2rem;
            text-align: center;
            min-height: 280px;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            box-shadow: 0 4px 20px rgba(52, 78, 65, 0.10);
        ">
            <div style="font-size:5.5rem; margin-bottom:1rem;
                        filter: drop-shadow(0 4px 8px rgba(52,78,65,0.2));">
                🌿
            </div>
            <div style="color:{Colors.DEEP_OAK_GREEN}; font-weight:700;
                        font-size:1.1rem; letter-spacing:0.01em;">
                AI-Powered Leaf Analysis
            </div>
            <div style="color:{Colors.DARK_EARTH}; font-size:0.88rem;
                        margin-top:0.4rem; opacity:0.8;">
                Upload a photo → Instant analysis
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("<br>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────
# Features section
# ─────────────────────────────────────────────────────────────────
render_section_heading(
    "What Aura Farming Offers",
    "Three capabilities that set this system apart from a basic classifier.",
)

feat1, feat2, feat3 = st.columns(3, gap="medium")
with feat1:
    render_feature_card(
        icon="🔍",
        title="Smart Image Analysis",
        description=(
            "Every photograph is checked for quality before analysis. "
            "Blurry or poorly lit images are flagged so poor photographs "
            "don't produce misleading results."
        ),
    )
with feat2:
    render_feature_card(
        icon="🛡️",
        title="Reliability-Aware Predictions",
        description=(
            "The system tells you when it is not confident, rather than "
            "guessing with false certainty. Uncertain cases are flagged "
            "and expert verification is recommended."
        ),
    )
with feat3:
    render_feature_card(
        icon="🌿",
        title="Visual Explanations",
        description=(
            "Grad-CAM heatmaps highlight which leaf regions influenced "
            "the model's prediction, making AI decisions more transparent "
            "and easier to understand."
        ),
    )

st.markdown("<br>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────
# How It Works section
# ─────────────────────────────────────────────────────────────────
how_left, how_right = st.columns([1, 1], gap="large")

with how_left:
    render_section_heading("How It Works")
    render_workflow_step(
        1,
        "Upload a leaf photograph",
        "Take a clear, close-up photo of the leaf showing the area of concern.",
    )
    render_workflow_step(
        2,
        "Automatic quality check",
        "The system reviews sharpness and brightness to ensure the image "
        "is suitable for analysis.",
    )
    render_workflow_step(
        3,
        "AI analysis with reliability assessment",
        "MobileNetV3 analyses the image and provides a calibrated confidence "
        "score. If confidence is too low, the result is flagged as uncertain.",
    )
    render_workflow_step(
        4,
        "Review your results",
        "See the predicted condition, a Grad-CAM explanation heatmap, "
        "and general prevention guidance — with clear AI limitation notices.",
    )

with how_right:
    st.markdown(
        f"""
        <div class="af-card" style="margin-top:3.5rem;">
          <div style="font-size:1rem; font-weight:700;
                      color:{Colors.DEEP_OAK_GREEN}; margin-bottom:1rem;">
            🌱 Supported Conditions (Phase 4+)
          </div>
          <p style="font-size:0.9rem; color:{Colors.DARK_EARTH};
                    margin-bottom:1rem;">
            The initial version focuses on tomato leaf disease detection.
            The following classes are planned for the trained model:
          </p>
          <ul style="font-size:0.88rem; color:{Colors.DARK_EARTH};
                     padding-left:1.2rem; line-height:2;">
            <li>Bacterial Spot</li>
            <li>Early Blight</li>
            <li>Late Blight</li>
            <li>Leaf Mold</li>
            <li>Septoria Leaf Spot</li>
            <li>Spider Mites</li>
            <li>Target Spot</li>
            <li>Tomato Mosaic Virus</li>
            <li>Yellow Leaf Curl Virus</li>
            <li>Healthy Leaf</li>
          </ul>
          <p style="font-size:0.82rem; color:{Colors.DARK_EARTH};
                    opacity:0.7; margin-top:0.6rem;">
            ✱ Final class list confirmed in Phase 3 after dataset inspection.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("<br>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────
# Disclaimer
# ─────────────────────────────────────────────────────────────────
render_disclaimer()

# ─────────────────────────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────────────────────────
render_footer()
