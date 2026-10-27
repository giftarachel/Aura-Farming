"""
Aura Farming — src/ui_components.py
=====================================
Centralised branding constants, CSS injection, and reusable UI blocks.

ALL color values live here. Never hardcode hex values in page files.
Import from this module to maintain a consistent visual identity.

Phase 2 status: COMPLETE — branding, layout helpers, and CSS injection.
"""

import streamlit as st


# ─────────────────────────────────────────────────────────────────
# Brand color palette
# ─────────────────────────────────────────────────────────────────

class Colors:
    """Central source of truth for the Aura Farming color palette."""
    DEEP_OAK_GREEN  = "#344E41"   # Navigation, major headings, brand elements
    FOREST_GREEN    = "#3A5A40"   # Primary buttons, active states
    SAGE_GREEN      = "#A3B18A"   # Accents, secondary badges
    WARM_BEIGE      = "#DAD7CD"   # Card borders, dividers
    CREAM           = "#F5F1E8"   # Main page background
    DARK_EARTH      = "#4A403A"   # Body text, supporting text
    SOFT_WHITE      = "#FFFEFA"   # Cards, panels, upload surfaces


# ─────────────────────────────────────────────────────────────────
# Brand text constants
# ─────────────────────────────────────────────────────────────────

BRAND_NAME      = "Aura Farming"
TAGLINE         = "Smarter Farming. Healthier Crops."
HEADLINE        = "Understand Your Crops. Protect Your Harvest."
DISCLAIMER_TEXT = (
    "Aura Farming is a research prototype. AI predictions are not a "
    "substitute for diagnosis by a qualified agricultural expert or "
    "plant pathologist. Always consult a professional before making "
    "treatment decisions."
)


# ─────────────────────────────────────────────────────────────────
# Logo SVG
# ─────────────────────────────────────────────────────────────────

def get_logo_svg(size: int = 48) -> str:
    """
    Return the Aura Farming logo as an inline SVG string.

    The logo depicts a stylised botanical leaf with a subtle node
    at its centre vein, representing the union of natural plant
    biology and AI analysis. It is simple, scalable, and suitable
    for use at any size.

    Args:
        size: Target rendering size in pixels (width = height).

    Returns:
        A self-contained SVG string safe for st.markdown injection.
    """
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" '
        f'viewBox="0 0 64 64" fill="none" role="img" aria-label="Aura Farming leaf logo">'
        f'<path d="M32 8 C18 8 8 22 8 36 C8 50 20 58 32 56 C44 58 56 50 56 36 C56 22 46 8 32 8 Z" fill="{Colors.FOREST_GREEN}" opacity="0.92"/>'
        f'<line x1="32" y1="12" x2="32" y2="54" stroke="{Colors.SAGE_GREEN}" stroke-width="2.2" stroke-linecap="round"/>'
        f'<line x1="32" y1="28" x2="18" y2="22" stroke="{Colors.SAGE_GREEN}" stroke-width="1.4" stroke-linecap="round" opacity="0.8"/>'
        f'<line x1="32" y1="38" x2="16" y2="34" stroke="{Colors.SAGE_GREEN}" stroke-width="1.4" stroke-linecap="round" opacity="0.8"/>'
        f'<line x1="32" y1="46" x2="20" y2="44" stroke="{Colors.SAGE_GREEN}" stroke-width="1.2" stroke-linecap="round" opacity="0.7"/>'
        f'<line x1="32" y1="28" x2="46" y2="22" stroke="{Colors.SAGE_GREEN}" stroke-width="1.4" stroke-linecap="round" opacity="0.8"/>'
        f'<line x1="32" y1="38" x2="48" y2="34" stroke="{Colors.SAGE_GREEN}" stroke-width="1.4" stroke-linecap="round" opacity="0.8"/>'
        f'<line x1="32" y1="46" x2="44" y2="44" stroke="{Colors.SAGE_GREEN}" stroke-width="1.2" stroke-linecap="round" opacity="0.7"/>'
        f'<circle cx="32" cy="36" r="4" fill="{Colors.SOFT_WHITE}" opacity="0.85"/>'
        f'<circle cx="32" cy="36" r="2" fill="{Colors.SAGE_GREEN}"/>'
        f'</svg>'
    )


# ─────────────────────────────────────────────────────────────────
# Global CSS injection
# ─────────────────────────────────────────────────────────────────

def inject_global_css() -> None:
    """
    Inject the Aura Farming global stylesheet into the Streamlit page.

    Call this once at the top of every page file. It applies:
    - Background and text colors from the brand palette.
    - Navigation bar styling.
    - Card component styles.
    - Button overrides.
    - Badge styles (Confident / Uncertain / Abstained).
    - Utility classes for spacing and layout.
    - Basic responsive adjustments.
    """
    st.markdown(
        f"""
        <style>
        /* ── Page background ─────────────────────────────────────── */
        .stApp {{
            background-color: {Colors.CREAM};
        }}

        /* ── Global typography ───────────────────────────────────── */
        html, body, [class*="css"] {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI",
                         Roboto, "Helvetica Neue", Arial, sans-serif;
            color: {Colors.DARK_EARTH};
            line-height: 1.65;
        }}

        h1, h2, h3, h4, h5, h6 {{
            color: {Colors.DEEP_OAK_GREEN};
            font-weight: 700;
            letter-spacing: -0.01em;
        }}

        /* ── Streamlit default header/footer cleanup ──────────────── */
        #MainMenu {{ visibility: hidden; }}
        footer {{ visibility: hidden; }}
        header {{ visibility: hidden; }}

        /* ── Scrollbar ──────────────────────────────────────────────*/
        ::-webkit-scrollbar {{ width: 6px; }}
        ::-webkit-scrollbar-track {{ background: {Colors.CREAM}; }}
        ::-webkit-scrollbar-thumb {{
            background: {Colors.SAGE_GREEN};
            border-radius: 3px;
        }}

        /* ── Primary buttons ────────────────────────────────────────*/
        .stButton > button {{
            background-color: {Colors.FOREST_GREEN} !important;
            color: {Colors.SOFT_WHITE} !important;
            border: none !important;
            border-radius: 8px !important;
            padding: 0.55rem 1.4rem !important;
            font-weight: 600 !important;
            font-size: 0.95rem !important;
            letter-spacing: 0.01em !important;
            transition: background-color 0.2s ease, box-shadow 0.2s ease !important;
            box-shadow: 0 2px 6px rgba(58, 90, 64, 0.25) !important;
        }}
        .stButton > button:hover {{
            background-color: {Colors.DEEP_OAK_GREEN} !important;
            box-shadow: 0 4px 12px rgba(52, 78, 65, 0.30) !important;
        }}
        .stButton > button:disabled {{
            background-color: {Colors.WARM_BEIGE} !important;
            color: {Colors.DARK_EARTH} !important;
            opacity: 0.55 !important;
            box-shadow: none !important;
            cursor: not-allowed !important;
        }}

        /* ── Card component ─────────────────────────────────────────*/
        .af-card {{
            background-color: {Colors.SOFT_WHITE};
            border: 1px solid {Colors.WARM_BEIGE};
            border-radius: 12px;
            padding: 1.5rem 1.75rem;
            box-shadow: 0 2px 8px rgba(52, 78, 65, 0.07);
            margin-bottom: 1.25rem;
        }}

        /* ── Feature card (homepage columns) ──────────────────────── */
        .af-feature-card {{
            background-color: {Colors.SOFT_WHITE};
            border: 1px solid {Colors.WARM_BEIGE};
            border-radius: 12px;
            padding: 1.5rem;
            box-shadow: 0 2px 8px rgba(52, 78, 65, 0.07);
            text-align: center;
            height: 100%;
        }}
        .af-feature-card .af-icon {{
            font-size: 2.2rem;
            margin-bottom: 0.6rem;
            display: block;
        }}
        .af-feature-card h4 {{
            color: {Colors.DEEP_OAK_GREEN};
            font-size: 1.05rem;
            margin-bottom: 0.4rem;
        }}
        .af-feature-card p {{
            font-size: 0.9rem;
            color: {Colors.DARK_EARTH};
            margin: 0;
        }}

        /* ── Navigation bar ─────────────────────────────────────────*/
        .af-navbar {{
            background-color: {Colors.DEEP_OAK_GREEN};
            padding: 0.85rem 2rem;
            border-radius: 10px;
            margin-bottom: 2rem;
            display: flex;
            align-items: center;
            justify-content: space-between;
        }}
        .af-navbar-brand {{
            display: flex;
            align-items: center;
            gap: 0.6rem;
            color: {Colors.SOFT_WHITE};
            font-size: 1.2rem;
            font-weight: 700;
            letter-spacing: -0.01em;
            text-decoration: none;
        }}
        .af-navbar-links {{
            display: flex;
            gap: 1.5rem;
        }}
        .af-navbar-links a {{
            color: {Colors.WARM_BEIGE};
            text-decoration: none;
            font-size: 0.9rem;
            font-weight: 500;
            transition: color 0.15s;
        }}
        .af-navbar-links a:hover {{
            color: {Colors.SOFT_WHITE};
        }}

        /* ── Section headings with accent rule ──────────────────────*/
        .af-section-heading {{
            color: {Colors.DEEP_OAK_GREEN};
            font-size: 1.6rem;
            font-weight: 700;
            margin-bottom: 0.3rem;
        }}
        .af-section-rule {{
            border: none;
            border-top: 2px solid {Colors.SAGE_GREEN};
            width: 48px;
            margin: 0.4rem 0 1.5rem 0;
        }}

        /* ── Tagline badge ───────────────────────────────────────────*/
        .af-tagline {{
            display: inline-block;
            background-color: {Colors.SAGE_GREEN};
            color: {Colors.DEEP_OAK_GREEN};
            font-size: 0.78rem;
            font-weight: 700;
            letter-spacing: 0.12em;
            text-transform: uppercase;
            padding: 0.3rem 0.8rem;
            border-radius: 20px;
            margin-bottom: 0.8rem;
        }}

        /* ── Hero headline ───────────────────────────────────────────*/
        .af-hero-headline {{
            font-size: 2.4rem;
            font-weight: 800;
            color: {Colors.DEEP_OAK_GREEN};
            line-height: 1.2;
            margin-bottom: 0.9rem;
            letter-spacing: -0.02em;
        }}
        @media (max-width: 768px) {{
            .af-hero-headline {{ font-size: 1.7rem; }}
        }}

        /* ── Reliability status badges ───────────────────────────────*/
        .af-badge-confident {{
            display: inline-flex;
            align-items: center;
            gap: 0.35rem;
            background-color: #D4EDDA;
            color: #155724;
            border: 1px solid #C3E6CB;
            border-radius: 20px;
            padding: 0.3rem 0.9rem;
            font-size: 0.85rem;
            font-weight: 600;
        }}
        .af-badge-uncertain {{
            display: inline-flex;
            align-items: center;
            gap: 0.35rem;
            background-color: #FFF3CD;
            color: #856404;
            border: 1px solid #FFE69C;
            border-radius: 20px;
            padding: 0.3rem 0.9rem;
            font-size: 0.85rem;
            font-weight: 600;
        }}
        .af-badge-abstained {{
            display: inline-flex;
            align-items: center;
            gap: 0.35rem;
            background-color: #F8D7DA;
            color: #721C24;
            border: 1px solid #F5C6CB;
            border-radius: 20px;
            padding: 0.3rem 0.9rem;
            font-size: 0.85rem;
            font-weight: 600;
        }}

        /* ── Upload zone styling ─────────────────────────────────────*/
        [data-testid="stFileUploader"] {{
            background-color: {Colors.SOFT_WHITE};
            border: 2px dashed {Colors.SAGE_GREEN};
            border-radius: 12px;
            padding: 1rem;
        }}

        /* ── Disclaimer banner ───────────────────────────────────────*/
        .af-disclaimer {{
            background-color: {Colors.WARM_BEIGE};
            border-left: 4px solid {Colors.SAGE_GREEN};
            border-radius: 8px;
            padding: 0.9rem 1.2rem;
            color: {Colors.DARK_EARTH};
            font-size: 0.87rem;
            margin: 1.5rem 0;
        }}

        /* ── Dev-state notice ────────────────────────────────────────*/
        .af-dev-notice {{
            background-color: #FFF8E7;
            border: 1px solid #FFE082;
            border-left: 4px solid #FFC107;
            border-radius: 8px;
            padding: 1rem 1.25rem;
            color: {Colors.DARK_EARTH};
            font-size: 0.92rem;
            margin: 1rem 0;
        }}

        /* ── Error / validation feedback ─────────────────────────────*/
        .af-error {{
            background-color: #FDE8E8;
            border: 1px solid #F5C6C6;
            border-left: 4px solid #E53E3E;
            border-radius: 8px;
            padding: 0.9rem 1.2rem;
            color: #742A2A;
            font-size: 0.9rem;
            margin: 0.75rem 0;
        }}
        .af-warning {{
            background-color: #FFFBEB;
            border: 1px solid #FDE68A;
            border-left: 4px solid #D97706;
            border-radius: 8px;
            padding: 0.9rem 1.2rem;
            color: #92400E;
            font-size: 0.9rem;
            margin: 0.75rem 0;
        }}
        .af-success {{
            background-color: #ECFDF5;
            border: 1px solid #A7F3D0;
            border-left: 4px solid #059669;
            border-radius: 8px;
            padding: 0.9rem 1.2rem;
            color: #064E3B;
            font-size: 0.9rem;
            margin: 0.75rem 0;
        }}

        /* ── Workflow steps ──────────────────────────────────────────*/
        .af-step {{
            display: flex;
            align-items: flex-start;
            gap: 1rem;
            margin-bottom: 1.1rem;
        }}
        .af-step-number {{
            background-color: {Colors.FOREST_GREEN};
            color: {Colors.SOFT_WHITE};
            width: 32px;
            height: 32px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: 700;
            font-size: 0.9rem;
            flex-shrink: 0;
        }}
        .af-step-text {{
            padding-top: 0.25rem;
        }}
        .af-step-text strong {{
            color: {Colors.DEEP_OAK_GREEN};
            display: block;
            font-size: 0.95rem;
        }}
        .af-step-text span {{
            font-size: 0.88rem;
            color: {Colors.DARK_EARTH};
        }}

        /* ── Footer ──────────────────────────────────────────────────*/
        .af-footer {{
            text-align: center;
            padding: 2rem 1rem 1rem 1rem;
            border-top: 1px solid {Colors.WARM_BEIGE};
            margin-top: 3rem;
            color: {Colors.DARK_EARTH};
            font-size: 0.83rem;
            opacity: 0.8;
        }}
        .af-footer strong {{
            color: {Colors.DEEP_OAK_GREEN};
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────────
# Reusable HTML component helpers
# ─────────────────────────────────────────────────────────────────

def render_navbar(active_page: str = "Home") -> None:
    """
    Render the Aura Farming navigation bar.

    Args:
        active_page: Name of the currently active page
                     ("Home", "Detect", or "About"). Used for
                     future active-link styling.
    """
    logo_svg = get_logo_svg(size=32)
    navbar_html = (
        f'<div class="af-navbar">'
        f'<div class="af-navbar-brand">{logo_svg}<span>Aura Farming</span></div>'
        f'<nav class="af-navbar-links">'
        f'<a href="/" target="_self">Home</a>'
        f'<a href="/Detect_Disease" target="_self">Detect Disease</a>'
        f'<a href="/About_Aura_Farming" target="_self">About</a>'
        f'</nav>'
        f'</div>'
    )
    st.markdown(navbar_html, unsafe_allow_html=True)


def render_section_heading(title: str, subtitle: str = "") -> None:
    """
    Render a styled section heading with a Sage Green accent rule.

    Args:
        title: The heading text.
        subtitle: Optional supporting text rendered below the rule.
    """
    st.markdown(
        f'<h2 class="af-section-heading">{title}</h2><hr class="af-section-rule">',
        unsafe_allow_html=True,
    )
    if subtitle:
        st.markdown(
            f'<p style="color:{Colors.DARK_EARTH}; margin-bottom:1.5rem;">{subtitle}</p>',
            unsafe_allow_html=True,
        )


def render_disclaimer() -> None:
    """Render the standard AI limitation disclaimer banner."""
    st.markdown(
        f'<div class="af-disclaimer">⚠️ &nbsp;<strong>Important:</strong> '
        f'{DISCLAIMER_TEXT}</div>',
        unsafe_allow_html=True,
    )


def render_dev_notice(message: str = "") -> None:
    """
    Render a development-state notice.

    Use this wherever a future feature is not yet connected.

    Args:
        message: Optional custom message. Falls back to a sensible default.
    """
    text = message or (
        "🔧 <strong>Development mode:</strong> The AI model has not yet been "
        "connected. This feature will be enabled in a later phase of development."
    )
    st.markdown(f'<div class="af-dev-notice">{text}</div>', unsafe_allow_html=True)


def render_error(message: str) -> None:
    """Render a styled error message block."""
    st.markdown(
        f'<div class="af-error">❌ &nbsp;{message}</div>',
        unsafe_allow_html=True,
    )


def render_warning(message: str) -> None:
    """Render a styled warning message block."""
    st.markdown(
        f'<div class="af-warning">⚠️ &nbsp;{message}</div>',
        unsafe_allow_html=True,
    )


def render_success(message: str) -> None:
    """Render a styled success/confirmation message block."""
    st.markdown(
        f'<div class="af-success">✅ &nbsp;{message}</div>',
        unsafe_allow_html=True,
    )


def render_footer() -> None:
    """Render the Aura Farming page footer."""
    footer_html = (
        f'<div class="af-footer">'
        f'<strong>Aura Farming</strong> &nbsp;·&nbsp; '
        f'An AI-Powered, Reliability-Aware Crop Disease Detection System<br>'
        f'<span style="margin-top:0.3rem; display:block;">'
        f'Research prototype only &nbsp;·&nbsp; '
        f'Not a validated diagnostic tool &nbsp;·&nbsp; '
        f'Consult a qualified agricultural expert for treatment decisions'
        f'</span>'
        f'</div>'
    )
    st.markdown(footer_html, unsafe_allow_html=True)


def render_feature_card(icon: str, title: str, description: str) -> None:
    """
    Render a single feature card for the homepage feature row.

    Args:
        icon: Emoji or Unicode character for the feature icon.
        title: Card heading text.
        description: Card body text (1–2 sentences).
    """
    card_html = (
        f'<div class="af-feature-card">'
        f'<span class="af-icon">{icon}</span>'
        f'<h4>{title}</h4>'
        f'<p>{description}</p>'
        f'</div>'
    )
    st.markdown(card_html, unsafe_allow_html=True)


def render_workflow_step(number: int, title: str, description: str) -> None:
    """
    Render a single numbered workflow step.

    Args:
        number: Step number (1-based).
        title: Step heading.
        description: Brief explanation of the step.
    """
    step_html = (
        f'<div class="af-step">'
        f'<div class="af-step-number">{number}</div>'
        f'<div class="af-step-text">'
        f'<strong>{title}</strong>'
        f'<span>{description}</span>'
        f'</div>'
        f'</div>'
    )
    st.markdown(step_html, unsafe_allow_html=True)
