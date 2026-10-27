"""
Aura Farming — pages/1_Detect_Disease.py
==========================================
Disease detection page: upload, validate, quality gate, inference,
probability calibration, Grad-CAM explainability, and agronomic guidance.

Phase 5 Implementation: COMPLETE
─────────────────────────────────
Workflow:
  1. Image upload via st.file_uploader.
  2. Format and size validation (src/image_validation.py).
  3. Image quality gate: blur & brightness analysis (src/image_quality.py).
  4. Real-time preview and quality checklist.
  5. Neural network inference via MobileNetV3 (src/predictor.py & src/preprocessing.py).
  6. Post-hoc probability calibration and abstention policy (src/calibration.py).
  7. Visual attention explanation with Grad-CAM (src/gradcam.py).
  8. Disease biology and IPM care guidance (src/disease_information.py & src/guidance.py).
"""

from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Optional

import streamlit as st
from PIL import Image

st.set_page_config(
    page_title="Detect Disease — Aura Farming",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="collapsed",
)

from src.ui_components import (
    Colors,
    inject_global_css,
    render_navbar,
    render_section_heading,
    render_disclaimer,
    render_dev_notice,
    render_error,
    render_warning,
    render_success,
    render_footer,
)
from src.image_validation import (
    validate_upload,
    UnsupportedFormatError,
    FileTooLargeError,
    CorruptedImageError,
    InvalidDimensionsError,
    ValidationError,
    SUPPORTED_FORMATS,
)
from src.image_quality import assess_quality, QualityReport, QualityStatus
from src.predictor import is_model_available, predict, ModelNotAvailableError, InferenceError
from src.preprocessing import preprocess_for_inference
from src.calibration import (
    apply_calibration_and_threshold,
    PredictionResult,
    ReliabilityStatus,
)
from src.gradcam import generate_gradcam_overlay, GradCAMError
from src.disease_information import get_disease_info
from src.guidance import get_guidance


# ─────────────────────────────────────────────────────────────────
# Class mapping loader
# ─────────────────────────────────────────────────────────────────

@st.cache_data
def _get_class_names() -> dict[int, str]:
    mapping_path = Path(__file__).parent.parent / "configs" / "class_mapping.json"
    with open(mapping_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {int(k): v["display_name"] for k, v in data["classes"].items()}


# ─────────────────────────────────────────────────────────────────
# Page setup
# ─────────────────────────────────────────────────────────────────
inject_global_css()
render_navbar(active_page="Detect")

# ─────────────────────────────────────────────────────────────────
# Page heading
# ─────────────────────────────────────────────────────────────────
render_section_heading(
    "Detect Crop Disease",
    "Upload a photograph of a tomato leaf to evaluate health and detect disease.",
)

# ─────────────────────────────────────────────────────────────────
# Instructions card
# ─────────────────────────────────────────────────────────────────
st.markdown(
    f"""
    <div class="af-card">
      <strong style="color:{Colors.DEEP_OAK_GREEN}; font-size:0.97rem;">
        📋 Photography & Quality Guidelines
      </strong>
      <ul style="margin-top:0.6rem; font-size:0.9rem;
                 color:{Colors.DARK_EARTH}; line-height:1.9;
                 padding-left:1.2rem;">
        <li>Photograph the leaf in <strong>balanced natural or diffused light</strong> (avoid direct harsh flash or heavy shadows).</li>
        <li>Hold the camera steady and focus cleanly on the leaf surface.</li>
        <li>Accepted formats: <strong>JPG, JPEG, PNG, WEBP</strong> (up to 20 MB).</li>
        <li>The system automatically assesses <strong>sharpness</strong> and <strong>exposure</strong> before inference.</li>
        <li>If evidence is insufficient, the system will abstain rather than guess an unreliable diagnosis.</li>
      </ul>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown("<br>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────
# Upload and validation layout
# ─────────────────────────────────────────────────────────────────
upload_col, preview_col = st.columns([1, 1], gap="large")

with upload_col:
    st.markdown(
        f'<p style="font-weight:600; color:{Colors.DEEP_OAK_GREEN}; '
        f'margin-bottom:0.3rem;">Upload your leaf photograph</p>',
        unsafe_allow_html=True,
    )
    uploaded_file = st.file_uploader(
        label="Upload leaf photograph",
        type=["jpg", "jpeg", "png", "webp"],
        accept_multiple_files=False,
        label_visibility="collapsed",
        help="Upload a leaf photograph (JPG, PNG, WEBP, max 20 MB).",
    )

# ─────────────────────────────────────────────────────────────────
# Processing logic & Session state
# ─────────────────────────────────────────────────────────────────

if "validated_image" not in st.session_state:
    st.session_state.validated_image = None
if "image_metadata" not in st.session_state:
    st.session_state.image_metadata = None
if "validation_passed" not in st.session_state:
    st.session_state.validation_passed = False
if "validation_error_msg" not in st.session_state:
    st.session_state.validation_error_msg = None
if "quality_report" not in st.session_state:
    st.session_state.quality_report = None

# Reset state when uploader is cleared
if uploaded_file is None:
    st.session_state.validated_image = None
    st.session_state.image_metadata = None
    st.session_state.validation_passed = False
    st.session_state.validation_error_msg = None
    st.session_state.quality_report = None

# Run validation and quality assessment when a file is provided
if uploaded_file is not None:
    raw_bytes = uploaded_file.read()

    try:
        image, metadata = validate_upload(uploaded_file.name, raw_bytes)
        st.session_state.validated_image = image
        st.session_state.image_metadata = metadata
        st.session_state.validation_passed = True
        st.session_state.validation_error_msg = None

        # Phase 5 Image Quality Gate
        quality_report = assess_quality(image)
        st.session_state.quality_report = quality_report

    except UnsupportedFormatError as exc:
        st.session_state.validation_passed = False
        st.session_state.validation_error_msg = str(exc)
    except FileTooLargeError as exc:
        st.session_state.validation_passed = False
        st.session_state.validation_error_msg = str(exc)
    except CorruptedImageError as exc:
        st.session_state.validation_passed = False
        st.session_state.validation_error_msg = str(exc)
    except InvalidDimensionsError as exc:
        st.session_state.validation_passed = False
        st.session_state.validation_error_msg = str(exc)
    except ValidationError as exc:
        st.session_state.validation_passed = False
        st.session_state.validation_error_msg = str(exc)
    except Exception:
        st.session_state.validation_passed = False
        st.session_state.validation_error_msg = (
            "An unexpected error occurred while reading this file. Please try a different photograph."
        )

# ─────────────────────────────────────────────────────────────────
# Left column: validation feedback & quality checks
# ─────────────────────────────────────────────────────────────────
with upload_col:
    if uploaded_file is not None:
        if not st.session_state.validation_passed:
            render_error(st.session_state.validation_error_msg)
        else:
            meta = st.session_state.image_metadata
            qr: QualityReport = st.session_state.quality_report

            # Format Sharpness label
            if qr.blur_score is not None:
                if qr.blur_score < 40.0:
                    sharpness_txt = f"❌ &nbsp;Sharpness: <strong>{qr.blur_score:.1f}</strong> (severely blurry)"
                elif qr.blur_score < 100.0:
                    sharpness_txt = f"⚠️ &nbsp;Sharpness: <strong>{qr.blur_score:.1f}</strong> (moderate blur)"
                else:
                    sharpness_txt = f"✅ &nbsp;Sharpness: <strong>{qr.blur_score:.1f}</strong> (good sharpness)"
            else:
                sharpness_txt = "ℹ️ &nbsp;Sharpness: not computed"

            # Format Brightness label
            if qr.brightness_mean is not None:
                if qr.brightness_mean < 30.0:
                    brightness_txt = f"❌ &nbsp;Exposure: <strong>{qr.brightness_mean:.1f}/255</strong> (underexposed/dark)"
                elif qr.brightness_mean > 225.0:
                    brightness_txt = f"❌ &nbsp;Exposure: <strong>{qr.brightness_mean:.1f}/255</strong> (overexposed/glare)"
                elif qr.brightness_mean < 50.0 or qr.brightness_mean > 205.0:
                    brightness_txt = f"⚠️ &nbsp;Exposure: <strong>{qr.brightness_mean:.1f}/255</strong> (borderline)"
                else:
                    brightness_txt = f"✅ &nbsp;Exposure: <strong>{qr.brightness_mean:.1f}/255</strong> (balanced)"
            else:
                brightness_txt = "ℹ️ &nbsp;Exposure: not computed"

            st.markdown(
                f"""
                <div class="af-card" style="padding:1rem 1.25rem;">
                  <strong style="font-size:0.92rem; color:{Colors.DEEP_OAK_GREEN};">
                    Image Quality Assessment
                  </strong>
                  <div style="margin-top:0.65rem; font-size:0.88rem;
                              color:{Colors.DARK_EARTH}; line-height:2.0;">
                    ✅ &nbsp;Format: <code>.{meta['extension'].upper()}</code>
                    &nbsp;&nbsp;·&nbsp;&nbsp; {meta['file_size_mb']} MB<br>
                    ✅ &nbsp;Decodable: valid image bytes<br>
                    ✅ &nbsp;Dimensions: {meta['width']} × {meta['height']} px<br>
                    {sharpness_txt}<br>
                    {brightness_txt}
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Warning / Failure feedback callouts
            if not qr.passed:
                render_error(
                    f"<strong>Quality Gate Rejection:</strong><br>{qr.feedback.replace(chr(10), '<br>')}"
                )
            elif qr.warnings:
                render_warning(
                    f"<strong>Quality Notice:</strong><br>{qr.feedback.replace(chr(10), '<br>')}"
                )

# ─────────────────────────────────────────────────────────────────
# Right column: image preview
# ─────────────────────────────────────────────────────────────────
with preview_col:
    if st.session_state.validation_passed and st.session_state.validated_image:
        st.markdown(
            f'<p style="font-weight:600; color:{Colors.DEEP_OAK_GREEN}; '
            f'margin-bottom:0.3rem;">Preview</p>',
            unsafe_allow_html=True,
        )
        st.image(
            st.session_state.validated_image,
            caption=f"{st.session_state.image_metadata['filename']}",
            use_container_width=True,
        )
    elif uploaded_file is None:
        st.markdown(
            f"""
            <div style="
                background-color:{Colors.SOFT_WHITE};
                border:1px dashed {Colors.WARM_BEIGE};
                border-radius:12px;
                padding:3rem 2rem;
                text-align:center;
                min-height:240px;
                display:flex;
                flex-direction:column;
                align-items:center;
                justify-content:center;
            ">
                <div style="font-size:3rem; margin-bottom:0.8rem; opacity:0.4;">🍃</div>
                <div style="color:{Colors.DARK_EARTH}; font-size:0.9rem; opacity:0.5;">
                    Your photograph will appear here
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

# ─────────────────────────────────────────────────────────────────
# Analyze button
# ─────────────────────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)

can_analyze = (
    st.session_state.validation_passed
    and st.session_state.quality_report is not None
    and st.session_state.quality_report.can_proceed
)

analyze_col, _ = st.columns([1, 2])
with analyze_col:
    analyze_button = st.button(
        "🔬 &nbsp;Analyze Leaf",
        use_container_width=True,
        disabled=not can_analyze,
        help="Click to analyze the leaf photograph." if can_analyze else "Ensure a valid, sharp photograph is uploaded.",
    )

# ─────────────────────────────────────────────────────────────────
# Analysis results area
# ─────────────────────────────────────────────────────────────────
if analyze_button and st.session_state.validation_passed:
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(
        f'<hr style="border:none; border-top:1px solid {Colors.WARM_BEIGE};">',
        unsafe_allow_html=True,
    )
    render_section_heading("Analysis & Diagnostic Results")

    # Check model weights availability
    if not is_model_available():
        render_dev_notice(
            "🔧 <strong>AI Model Training in Progress (Phase 4)</strong><br>"
            "The trained MobileNetV3 model weights file is not yet in <code>models/</code>.<br>"
            "The image quality gate and calibration pipelines are active. "
            "Once training is complete, end-to-end inference and Grad-CAM explanations "
            "will automatically display here."
        )
    else:
        # Preprocessing
        try:
            tensor = preprocess_for_inference(st.session_state.validated_image)
        except Exception as exc:
            tensor = None
            render_error(f"Preprocessing failed: {exc}")

        if tensor is None:
            render_dev_notice(
                "🔧 <strong>TensorFlow Not Available in Runtime</strong><br>"
                "Preprocessing returned None because TensorFlow is not installed in this Python environment.<br>"
                "To enable local neural network predictions, run in a Python 3.10–3.12 environment with TensorFlow."
            )
        else:
            # Inference & Calibration
            with st.spinner("Running deep learning inference and probability calibration..."):
                try:
                    raw_probs = predict(tensor)
                    class_names = _get_class_names()
                    result: PredictionResult = apply_calibration_and_threshold(
                        raw_probabilities=raw_probs,
                        class_names=class_names,
                        quality_report=st.session_state.quality_report,
                    )
                except Exception as exc:
                    render_error(f"Prediction failed: {exc}")
                    result = None

            if result is not None:
                # ── 1. Reliability Banner ─────────────────────────
                if result.reliability == ReliabilityStatus.CONFIDENT:
                    st.markdown(
                        f"""
                        <div style="background-color:#EBF6ED; border-left:4px solid #2E7D32;
                                    border-radius:8px; padding:1.1rem 1.4rem; margin-bottom:1.5rem;">
                          <div style="display:flex; justify-content:space-between; align-items:center;">
                            <span style="font-weight:700; color:#1B5E20; font-size:1.15rem;">
                              🌿 Confident Diagnosis: {result.class_name}
                            </span>
                            <span style="background-color:#2E7D32; color:white; font-size:0.8rem;
                                         font-weight:700; padding:0.25rem 0.65rem; border-radius:12px;">
                              CALIBRATED CONFIDENCE: {result.top_probability:.1%}
                            </span>
                          </div>
                          <p style="margin-top:0.5rem; font-size:0.9rem; color:#1E4620; margin-bottom:0;">
                            {result.explanation}
                          </p>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                elif result.reliability == ReliabilityStatus.UNCERTAIN:
                    st.markdown(
                        f"""
                        <div style="background-color:#FFF8E7; border-left:4px solid #F57C00;
                                    border-radius:8px; padding:1.1rem 1.4rem; margin-bottom:1.5rem;">
                          <div style="display:flex; justify-content:space-between; align-items:center;">
                            <span style="font-weight:700; color:#E65100; font-size:1.15rem;">
                              ⚠️ Uncertain Prediction: {result.class_name}
                            </span>
                            <span style="background-color:#F57C00; color:white; font-size:0.8rem;
                                         font-weight:700; padding:0.25rem 0.65rem; border-radius:12px;">
                              CONFIDENCE: {result.top_probability:.1%}
                            </span>
                          </div>
                          <p style="margin-top:0.5rem; font-size:0.9rem; color:#6B3600; margin-bottom:0;">
                            {result.explanation}
                          </p>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                else:  # ABSTAINED
                    st.markdown(
                        f"""
                        <div style="background-color:#FDF2E9; border-left:4px solid #D84315;
                                    border-radius:8px; padding:1.1rem 1.4rem; margin-bottom:1.5rem;">
                          <span style="font-weight:700; color:#BF360C; font-size:1.15rem;">
                            🛡️ Diagnosis Withheld (System Abstained)
                          </span>
                          <p style="margin-top:0.5rem; font-size:0.9rem; color:#4E1F11; margin-bottom:0;">
                            {result.explanation}
                          </p>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                # ── 2. Candidate Probabilities Breakdown ──────────
                if result.top_k:
                    st.markdown(
                        f'<p style="font-weight:600; color:{Colors.DEEP_OAK_GREEN}; '
                        f'margin-bottom:0.5rem;">Class Probability Distribution</p>',
                        unsafe_allow_html=True,
                    )
                    prob_cols = st.columns(len(result.top_k))
                    for col, (cls_idx, name, prob) in zip(prob_cols, result.top_k):
                        with col:
                            st.metric(label=name, value=f"{prob:.1%}")

                st.markdown("<br>", unsafe_allow_html=True)

                # ── 3. Grad-CAM Visual Explanation ────────────────
                if result.class_index is not None:
                    st.markdown(
                        f'<p style="font-weight:600; color:{Colors.DEEP_OAK_GREEN}; '
                        f'font-size:1.05rem; margin-bottom:0.3rem;">Visual Attention Explanation (Grad-CAM)</p>',
                        unsafe_allow_html=True,
                    )
                    try:
                        gradcam_overlay = generate_gradcam_overlay(
                            image=st.session_state.validated_image,
                            tensor=tensor,
                            class_index=result.class_index,
                        )
                    except GradCAMError:
                        gradcam_overlay = None
                    except Exception:
                        gradcam_overlay = None

                    if gradcam_overlay is not None:
                        cam_col1, cam_col2 = st.columns([1, 1], gap="large")
                        with cam_col1:
                            st.image(
                                st.session_state.validated_image,
                                caption="Original Photograph",
                                use_container_width=True,
                            )
                        with cam_col2:
                            st.image(
                                gradcam_overlay,
                                caption="Grad-CAM Attention Heatmap Overlay",
                                use_container_width=True,
                            )
                        st.markdown(
                            f"""
                            <p style="font-size:0.83rem; color:{Colors.DARK_EARTH}; opacity:0.85; margin-top:0.4rem;">
                              ℹ️ <strong>Scientific Note:</strong> Red/warm regions highlight image areas that had the
                              highest positive influence on the neural network's classification score.
                              Grad-CAM illustrates model focus and does not prove the presence of pathogen cells in those areas.
                            </p>
                            """,
                            unsafe_allow_html=True,
                        )
                    else:
                        st.info("ℹ️ Grad-CAM visual explanation is currently unavailable for this prediction.")

                st.markdown("<br>", unsafe_allow_html=True)

                # ── 4. Disease Biology & Prevention Guidance ───────
                if result.class_name is not None:
                    disease_info = get_disease_info(result.class_name)
                    guidance_info = get_guidance(result.class_name)

                    tab_info, tab_actions, tab_ipm = st.tabs([
                        "📖 &nbsp;Disease Profile & Symptoms",
                        "🚨 &nbsp;Immediate Field Actions",
                        "🛡️ &nbsp;Long-term IPM & Prevention",
                    ])

                    with tab_info:
                        st.markdown(
                            f"""
                            <div class="af-card" style="margin-top:0.8rem;">
                              <h4 style="color:{Colors.DEEP_OAK_GREEN}; margin-bottom:0.4rem;">
                                {disease_info['condition']}
                              </h4>
                              <p style="font-size:0.85rem; color:{Colors.DARK_EARTH}; opacity:0.8; margin-bottom:0.8rem;">
                                <strong>Pathogen:</strong> <em>{disease_info['scientific_name']}</em>
                                &nbsp;·&nbsp; <strong>Type:</strong> {disease_info['pathogen_type']}
                              </p>
                              <p style="font-size:0.92rem; color:{Colors.DARK_EARTH}; line-height:1.75;">
                                {disease_info['description']}
                              </p>
                              <strong style="color:{Colors.DEEP_OAK_GREEN}; font-size:0.9rem;">
                                Observable Symptoms:
                              </strong>
                              <ul style="font-size:0.88rem; color:{Colors.DARK_EARTH}; line-height:1.8; margin-top:0.4rem;">
                                {''.join(f'<li>{sym}</li>' for sym in disease_info['symptoms'])}
                              </ul>
                              <p style="font-size:0.85rem; color:{Colors.DARK_EARTH}; margin-top:0.6rem;">
                                <strong>Favorable Conditions:</strong> {disease_info['favorable_conditions']}
                              </p>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                    with tab_actions:
                        st.markdown(
                            f"""
                            <div class="af-card" style="margin-top:0.8rem;">
                              <h4 style="color:{Colors.DEEP_OAK_GREEN}; margin-bottom:0.6rem;">
                                Recommended Immediate Steps
                              </h4>
                              <ul style="font-size:0.9rem; color:{Colors.DARK_EARTH}; line-height:1.85;">
                                {''.join(f'<li>{act}</li>' for act in guidance_info['immediate_actions'])}
                              </ul>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                    with tab_ipm:
                        st.markdown(
                            f"""
                            <div class="af-card" style="margin-top:0.8rem;">
                              <h4 style="color:{Colors.DEEP_OAK_GREEN}; margin-bottom:0.6rem;">
                                Integrated Pest Management (IPM) & Prevention
                              </h4>
                              <ul style="font-size:0.9rem; color:{Colors.DARK_EARTH}; line-height:1.85;">
                                {''.join(f'<li>{prev}</li>' for prev in guidance_info['preventive_measures'])}
                              </ul>
                              <p style="font-size:0.88rem; color:{Colors.DARK_EARTH}; margin-top:0.8rem;">
                                <strong>Field Scouting & Monitoring:</strong> {guidance_info['monitoring_advice']}
                              </p>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

# ─────────────────────────────────────────────────────────────────
# Analyze another photograph
# ─────────────────────────────────────────────────────────────────
if st.session_state.validation_passed:
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("🔄 &nbsp;Analyze Another Photograph", use_container_width=False):
        for key in [
            "validated_image", "image_metadata",
            "validation_passed", "validation_error_msg", "quality_report",
        ]:
            st.session_state[key] = None
        st.session_state.validation_passed = False
        st.rerun()

# ─────────────────────────────────────────────────────────────────
# Disclaimer and footer
# ─────────────────────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
render_disclaimer()
render_footer()
