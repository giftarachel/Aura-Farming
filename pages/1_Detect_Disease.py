"""
Aura Farming — pages/1_Detect_Disease.py
==========================================
Disease detection page: upload, validate, and (future) analyse.

This page handles:
  1. Image upload via st.file_uploader
  2. File format and size validation (src/image_validation.py)
  3. Image quality check (Phase 5 placeholder — always passes now)
  4. Image preview
  5. Analysis trigger — currently shows a development-state notice
     because the trained model has not yet been connected.

Phase 2 status:
  - Upload, validation, preview, and error states: COMPLETE
  - Image quality assessment: PLACEHOLDER (Phase 5)
  - Model inference: PLACEHOLDER (Phase 4)
  - Calibration / abstention: PLACEHOLDER (Phase 5)
  - Grad-CAM: PLACEHOLDER (Phase 5)
  - Disease information and guidance: PLACEHOLDER (Phase 5)
"""

import io

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
from src.image_quality import assess_quality
from src.predictor import is_model_available, ModelNotAvailableError

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
    "Upload a clear photograph of a tomato leaf to begin analysis.",
)

# ─────────────────────────────────────────────────────────────────
# Instructions card
# ─────────────────────────────────────────────────────────────────
st.markdown(
    f"""
    <div class="af-card">
      <strong style="color:{Colors.DEEP_OAK_GREEN}; font-size:0.97rem;">
        📋 Before you upload
      </strong>
      <ul style="margin-top:0.6rem; font-size:0.9rem;
                 color:{Colors.DARK_EARTH}; line-height:1.9;
                 padding-left:1.2rem;">
        <li>Photograph the leaf in <strong>good natural or artificial light</strong>.</li>
        <li>Focus on the area of concern; avoid blurry or out-of-focus shots.</li>
        <li>Accepted formats: <strong>JPG, JPEG, PNG, WEBP</strong>.</li>
        <li>Maximum file size: <strong>20 MB</strong>.</li>
        <li>The system will check image quality before analysis begins.</li>
        <li>For best results, photograph a single leaf against a plain background.</li>
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
        help=(
            "Drag and drop a leaf photograph here, or click to browse. "
            "Supported formats: JPG, JPEG, PNG, WEBP. Maximum size: 20 MB."
        ),
    )

# ─────────────────────────────────────────────────────────────────
# Processing logic
# ─────────────────────────────────────────────────────────────────

# Session state keys used across reruns
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

# Reset state when the uploader is cleared
if uploaded_file is None:
    st.session_state.validated_image = None
    st.session_state.image_metadata = None
    st.session_state.validation_passed = False
    st.session_state.validation_error_msg = None
    st.session_state.quality_report = None

# Run validation when a file is present
if uploaded_file is not None:
    raw_bytes = uploaded_file.read()

    try:
        image, metadata = validate_upload(uploaded_file.name, raw_bytes)
        st.session_state.validated_image = image
        st.session_state.image_metadata = metadata
        st.session_state.validation_passed = True
        st.session_state.validation_error_msg = None

        # Phase 2 placeholder quality assessment (always passes)
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
    except Exception as exc:
        st.session_state.validation_passed = False
        st.session_state.validation_error_msg = (
            "An unexpected error occurred while reading this file. "
            "Please try a different photograph."
        )

# ─────────────────────────────────────────────────────────────────
# Left column: validation feedback
# ─────────────────────────────────────────────────────────────────
with upload_col:
    if uploaded_file is not None:
        if not st.session_state.validation_passed:
            render_error(st.session_state.validation_error_msg)
        else:
            meta = st.session_state.image_metadata
            qr = st.session_state.quality_report

            # Validation status checklist
            st.markdown(
                f"""
                <div class="af-card" style="padding:1rem 1.25rem;">
                  <strong style="font-size:0.9rem; color:{Colors.DEEP_OAK_GREEN};">
                    Image Checks
                  </strong>
                  <div style="margin-top:0.65rem; font-size:0.88rem;
                              color:{Colors.DARK_EARTH}; line-height:2.1;">
                    ✅ &nbsp;Format: <code>.{meta['extension'].upper()}</code>
                    &nbsp;&nbsp;·&nbsp;&nbsp;
                    {meta['file_size_mb']} MB<br>
                    ✅ &nbsp;Readable: image decoded successfully<br>
                    ✅ &nbsp;Dimensions: {meta['width']} × {meta['height']} px<br>
                    {'✅ &nbsp;Sharpness: check passed'
                      if (qr and qr.passed and not qr.is_placeholder)
                      else '🔧 &nbsp;Sharpness: quality gate active in Phase 5'}<br>
                    {'✅ &nbsp;Brightness: check passed'
                      if (qr and qr.passed and not qr.is_placeholder)
                      else '🔧 &nbsp;Brightness: quality gate active in Phase 5'}
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
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
                <div style="font-size:3rem; margin-bottom:0.8rem;
                            opacity:0.4;">🍃</div>
                <div style="color:{Colors.DARK_EARTH}; font-size:0.9rem;
                            opacity:0.5;">
                    Your photograph will appear here
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

# ─────────────────────────────────────────────────────────────────
# Analyze button and results area
# ─────────────────────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)

analyze_col, _ = st.columns([1, 2])
with analyze_col:
    analyze_button = st.button(
        "🔬 &nbsp;Analyze Leaf",
        use_container_width=True,
        disabled=not st.session_state.validation_passed,
        help=(
            "Upload and validate an image above to enable analysis."
            if not st.session_state.validation_passed
            else "Click to analyze the uploaded leaf photograph."
        ),
    )

# ─────────────────────────────────────────────────────────────────
# Analysis result area
# ─────────────────────────────────────────────────────────────────
if analyze_button and st.session_state.validation_passed:
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(
        f'<hr style="border:none; border-top:1px solid {Colors.WARM_BEIGE};">',
        unsafe_allow_html=True,
    )
    render_section_heading("Analysis Results")

    if not is_model_available():
        render_dev_notice(
            "🔧 <strong>AI Analysis Not Yet Available</strong><br>"
            "The trained MobileNetV3 model has not been connected to this "
            "application yet. Model training is planned for <strong>Phase 4</strong> "
            "of development.<br><br>"
            "Your image has been successfully validated and is ready for "
            "analysis once the model is available. No disease prediction "
            "has been generated."
        )

        # Show the validated image in the results area for context
        st.markdown("<br>", unsafe_allow_html=True)
        res_img_col, res_info_col = st.columns([1, 1], gap="large")
        with res_img_col:
            st.markdown(
                f'<p style="font-weight:600; color:{Colors.DEEP_OAK_GREEN};">'
                f'Uploaded Photograph</p>',
                unsafe_allow_html=True,
            )
            st.image(
                st.session_state.validated_image,
                caption="Awaiting AI model connection.",
                use_container_width=True,
            )
        with res_info_col:
            st.markdown(
                f"""
                <div class="af-card">
                  <strong style="color:{Colors.DEEP_OAK_GREEN};">
                    What happens here in Phase 4+
                  </strong>
                  <ul style="font-size:0.88rem; color:{Colors.DARK_EARTH};
                             padding-left:1.2rem; margin-top:0.8rem;
                             line-height:2.0;">
                    <li>Predicted disease class</li>
                    <li>Reliability badge (Confident / Uncertain)</li>
                    <li>Calibrated confidence score</li>
                    <li>Grad-CAM heatmap overlay</li>
                    <li>Disease description</li>
                    <li>Prevention and care guidance</li>
                  </ul>
                </div>
                """,
                unsafe_allow_html=True,
            )
    else:
        # Model is available — Phase 4+ code path
        # This branch will be fleshed out in Phase 6.
        render_dev_notice(
            "🔧 Model file detected but inference pipeline not yet "
            "connected. Full results will be shown in Phase 6."
        )

# ─────────────────────────────────────────────────────────────────
# Analyze another photograph
# ─────────────────────────────────────────────────────────────────
if st.session_state.validation_passed:
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("🔄 &nbsp;Analyze Another Photograph", use_container_width=False):
        # Clear session state and rerun to reset the page
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
