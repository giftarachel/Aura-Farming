"""
Aura Farming — pages/2_About_Aura_Farming.py
==============================================
About page: project purpose, AI workflow, features, limitations,
and development status.

No AI inference occurs on this page.
"""

import streamlit as st

st.set_page_config(
    page_title="About — Aura Farming",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="collapsed",
)

from src.ui_components import (
    Colors,
    BRAND_NAME,
    TAGLINE,
    inject_global_css,
    render_navbar,
    render_section_heading,
    render_disclaimer,
    render_footer,
    get_logo_svg,
)

inject_global_css()
render_navbar(active_page="About")

# ─────────────────────────────────────────────────────────────────
# Page heading
# ─────────────────────────────────────────────────────────────────
render_section_heading(
    f"About {BRAND_NAME}",
    "An AI-Powered, Reliability-Aware Crop Disease Detection System",
)

# ─────────────────────────────────────────────────────────────────
# Project purpose
# ─────────────────────────────────────────────────────────────────
st.markdown(
    f"""
    <div class="af-card">
      <h3 style="color:{Colors.DEEP_OAK_GREEN}; margin-bottom:0.8rem;">
        Project Purpose
      </h3>
      <p style="font-size:0.95rem; color:{Colors.DARK_EARTH}; line-height:1.75;">
        Aura Farming is a college-level Computer Science and Engineering
        research project. Its goal is to build a web application that
        analyses photographs of crop leaves, identifies potential diseases
        using a deep learning model, and — crucially — evaluates
        <em>how reliable</em> that prediction is before presenting it
        to the user.
      </p>
      <p style="font-size:0.95rem; color:{Colors.DARK_EARTH}; line-height:1.75;
                margin-top:0.8rem;">
        The project focuses on tomato leaf disease detection as an initial
        use case. The emphasis is on scientific honesty: the system will
        tell you when it is not confident, rather than producing a
        misleading result.
      </p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown("<br>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────
# AI workflow explanation
# ─────────────────────────────────────────────────────────────────
render_section_heading("The AI Workflow")

workflow_items = [
    (
        "📤", "Image Upload and Validation",
        "You upload a photograph of a tomato leaf. The system immediately "
        "checks the file format, size, and whether the image can be decoded. "
        "Invalid or corrupted files are rejected before any processing begins.",
    ),
    (
        "🔎", "Image Quality Gate",
        "Before running the AI model, the system checks the photograph for "
        "basic quality issues: is it blurry? Is it too dark or too bright? "
        "Images that are too poor in quality are flagged so they don't "
        "produce unreliable predictions. (This feature is being built in Phase 5.)",
    ),
    (
        "🧠", "Deep Learning Classification",
        "A MobileNetV3 neural network — pretrained on millions of images "
        "and fine-tuned on tomato leaf disease photographs — analyses the "
        "validated image and produces a probability score for each possible "
        "disease class. (Connected in Phase 4.)",
    ),
    (
        "🛡️", "Reliability Assessment",
        "Raw model probabilities are not reliable confidence estimates. "
        "Aura Farming applies calibration to adjust these scores, then "
        "compares the result against a validated threshold. If confidence "
        "is too low, the system returns an 'uncertain' status rather than "
        "a potentially wrong disease label. (Phase 5.)",
    ),
    (
        "🌿", "Visual Explanation (Grad-CAM)",
        "For confident predictions, a Grad-CAM heatmap is generated and "
        "overlaid on the original photograph. This shows which parts of "
        "the leaf most influenced the model's decision — making the AI "
        "process more transparent. (Phase 5.)",
    ),
    (
        "📋", "Results and Guidance",
        "The results page displays the predicted condition, reliability "
        "status, heatmap, and general prevention information sourced from "
        "authoritative agricultural references. All results include a clear "
        "disclaimer that this is not a professional diagnosis. (Phase 5–6.)",
    ),
]

for icon, title, desc in workflow_items:
    st.markdown(
        f"""
        <div style="display:flex; align-items:flex-start; gap:1rem;
                    margin-bottom:1rem;">
          <div style="
              background-color:{Colors.FOREST_GREEN};
              color:{Colors.SOFT_WHITE};
              width:40px; height:40px;
              border-radius:10px;
              display:flex; align-items:center; justify-content:center;
              font-size:1.1rem; flex-shrink:0;
          ">{icon}</div>
          <div>
            <strong style="color:{Colors.DEEP_OAK_GREEN};
                            font-size:0.97rem;">{title}</strong>
            <p style="font-size:0.88rem; color:{Colors.DARK_EARTH};
                       margin:0.2rem 0 0 0; line-height:1.65;">{desc}</p>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("<br>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────
# Technology stack
# ─────────────────────────────────────────────────────────────────
render_section_heading("Technology Used")

tech_col1, tech_col2 = st.columns(2, gap="large")

with tech_col1:
    st.markdown(
        f"""
        <div class="af-card">
          <strong style="color:{Colors.DEEP_OAK_GREEN};">AI and Machine Learning</strong>
          <ul style="font-size:0.9rem; color:{Colors.DARK_EARTH};
                     padding-left:1.2rem; margin-top:0.6rem; line-height:2.0;">
            <li>TensorFlow / Keras — deep learning framework</li>
            <li>MobileNetV3 Small — lightweight transfer learning model</li>
            <li>Grad-CAM — model explanation visualization</li>
            <li>Temperature scaling — probability calibration</li>
          </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )

with tech_col2:
    st.markdown(
        f"""
        <div class="af-card">
          <strong style="color:{Colors.DEEP_OAK_GREEN};">Application and Data</strong>
          <ul style="font-size:0.9rem; color:{Colors.DARK_EARTH};
                     padding-left:1.2rem; margin-top:0.6rem; line-height:2.0;">
            <li>Streamlit — web application framework</li>
            <li>OpenCV and Pillow — image processing</li>
            <li>scikit-learn — metrics and evaluation</li>
            <li>PlantVillage dataset — training data (CC BY 4.0)</li>
          </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("<br>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────
# Development status
# ─────────────────────────────────────────────────────────────────
render_section_heading("Development Status")

phases = [
    ("✅", "Phase 1", "Specification and planning", "Complete"),
    ("✅", "Phase 2", "Project foundation and initial UI", "Complete"),
    ("🔲", "Phase 3", "Dataset inspection and preparation", "Upcoming"),
    ("🔲", "Phase 4", "Baseline MobileNetV3 model training", "Upcoming"),
    ("🔲", "Phase 5", "Quality gate, calibration, Grad-CAM", "Upcoming"),
    ("🔲", "Phase 6", "Full application integration", "Upcoming"),
    ("🔲", "Phase 7", "Evaluation and experiments", "Upcoming"),
    ("🔲", "Phase 8", "Final polish and documentation", "Upcoming"),
]

status_col1, status_col2 = st.columns(2, gap="medium")
for i, (icon, phase, title, status) in enumerate(phases):
    col = status_col1 if i % 2 == 0 else status_col2
    color = Colors.FOREST_GREEN if icon == "✅" else Colors.WARM_BEIGE
    text_color = Colors.SOFT_WHITE if icon == "✅" else Colors.DARK_EARTH
    with col:
        st.markdown(
            f"""
            <div style="
                display:flex; align-items:center; gap:0.8rem;
                background:{color}22;
                border:1px solid {color};
                border-radius:8px;
                padding:0.7rem 1rem;
                margin-bottom:0.6rem;
            ">
              <span style="font-size:1.1rem;">{icon}</span>
              <div>
                <strong style="font-size:0.88rem;
                                color:{Colors.DEEP_OAK_GREEN};">
                  {phase}: {title}
                </strong>
                <span style="font-size:0.8rem; color:{Colors.DARK_EARTH};
                              display:block; opacity:0.75;">{status}</span>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

st.markdown("<br>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────
# Limitations
# ─────────────────────────────────────────────────────────────────
render_section_heading("Known Limitations")

st.markdown(
    f"""
    <div class="af-card">
      <ul style="font-size:0.9rem; color:{Colors.DARK_EARTH};
                 padding-left:1.2rem; line-height:2.0;">
        <li>
          <strong>Research prototype only.</strong>
          This system has not been validated for clinical or commercial
          agricultural use.
        </li>
        <li>
          <strong>Tomato leaves only (initial version).</strong>
          The trained model will recognise only the tomato disease
          classes confirmed during Phase 3 dataset inspection.
        </li>
        <li>
          <strong>Training data is primarily laboratory-style images.</strong>
          Performance on real farm photographs may differ from
          controlled-condition test results.
        </li>
        <li>
          <strong>Probabilities are estimates, not guarantees.</strong>
          Even after calibration, confidence scores are approximations.
          A high confidence score does not guarantee a correct prediction.
        </li>
        <li>
          <strong>Grad-CAM is a visualization aid, not proof.</strong>
          The heatmap shows which image regions influenced the model —
          it does not confirm that those regions contain disease symptoms.
        </li>
        <li>
          <strong>No professional diagnosis.</strong>
          Always consult a qualified plant pathologist or agricultural
          extension officer before making treatment decisions.
        </li>
      </ul>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown("<br>", unsafe_allow_html=True)
render_disclaimer()
render_footer()
