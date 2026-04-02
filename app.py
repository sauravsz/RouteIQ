import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from io import BytesIO
from pathlib import Path
from typing import List, Optional

try:
    from st_aggrid import AgGrid, GridUpdateMode
    HAS_AGGRID = True
except ImportError:
    HAS_AGGRID = False

from src.ai_explainer import (
    generate_executive_briefing,
    get_provider_default_model,
    get_provider_model_options,
    get_supported_providers,
)
from src.optimizer import solve_transportation
from src.scenarios import load_scenario
from src.visualizations import plot_cost_heatmap, plot_network


SCENARIOS = ["baseline", "disruption", "cost_surge"]
DATA_FILE_PATH = Path(__file__).resolve().parent / "data" / "scenarios.csv"


def _apply_ui_theme(theme_mode: str) -> None:
    is_dark = theme_mode.lower() == "dark"

    bg_page = "linear-gradient(160deg, #0f172a 0%, #111827 100%)" if is_dark else "linear-gradient(160deg, #f7f9fc 0%, #eef2f7 100%)"
    bg_sidebar = "linear-gradient(180deg, #0b1220 0%, #111827 100%)" if is_dark else "linear-gradient(180deg, #ffffff 0%, #f6f8fb 100%)"
    header_bg = "rgba(15,23,42,0.95)" if is_dark else "rgba(247,249,252,0.95)"
    text_base = "#e5e7eb" if is_dark else "#1f2937"
    text_muted = "#94a3b8" if is_dark else "#64748b"
    border = "rgba(148,163,184,0.22)" if is_dark else "rgba(148,163,184,0.28)"
    panel_bg = "rgba(15,23,42,0.35)" if is_dark else "rgba(255,255,255,0.86)"
    input_bg = "rgba(15,23,42,0.5)" if is_dark else "#ffffff"
    accent = "#6366f1"
    accent_soft = "rgba(99,102,241,0.16)" if is_dark else "rgba(99,102,241,0.10)"
    select_menu_bg = "#111827" if is_dark else "#ffffff"
    grid_bg = "#0f172a" if is_dark else "#ffffff"
    grid_bg_alt = "#0b1220" if is_dark else "#f8fafc"
    grid_header = "#1e293b" if is_dark else "#eef2f7"
    grid_text = "#e5e7eb" if is_dark else "#1f2937"
    grid_text_muted = "#94a3b8" if is_dark else "#64748b"
    grid_border = "rgba(148,163,184,0.20)" if is_dark else "rgba(148,163,184,0.28)"

    css = f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Sora:wght@300;400;500;600;700&family=DM+Serif+Display:ital@0;1&display=swap');

    :root {{
        --rq-text: {text_base};
        --rq-muted: {text_muted};
        --rq-border: {border};
        --rq-panel: {panel_bg};
        --rq-input: {input_bg};
        --rq-accent: {accent};
        --rq-accent-soft: {accent_soft};

        --rq-gdg-bg-cell: {grid_bg};
        --rq-gdg-bg-cell-medium: {grid_bg_alt};
        --rq-gdg-bg-header: {grid_header};
        --rq-gdg-bg-header-hovered: {grid_header};
        --rq-gdg-bg-header-focus: {grid_header};
        --rq-gdg-bg-search: {accent_soft};
        --rq-gdg-border: {grid_border};
        --rq-gdg-h-border: {grid_border};
        --rq-gdg-text: {grid_text};
        --rq-gdg-text-muted: {grid_text_muted};
        --rq-gdg-accent: {accent};
        --rq-gdg-accent-fg: {grid_text};
    }}

    .stApp,
    [data-testid="stAppViewContainer"] {{
        background: {bg_page} !important;
        color: var(--rq-text);
        font-family: 'Sora', sans-serif;
    }}

    /* Global UI typography baseline */
    [data-testid="stAppViewContainer"] p,
    [data-testid="stAppViewContainer"] label,
    [data-testid="stAppViewContainer"] small,
    [data-testid="stAppViewContainer"] input,
    [data-testid="stAppViewContainer"] select,
    [data-testid="stAppViewContainer"] textarea,
    [data-testid="stAppViewContainer"] button,
    [data-testid="stAppViewContainer"] li,
    [data-testid="stAppViewContainer"] a {{
        font-family: 'Sora', sans-serif !important;
    }}

    /* Preserve Streamlit icon ligatures globally */
    .material-icons,
    .material-symbols-outlined,
    .material-symbols-rounded,
    [class*="material-symbols"] {{
        font-family: "Material Symbols Rounded", "Material Symbols Outlined", "Material Icons" !important;
        letter-spacing: normal !important;
        text-transform: none !important;
        line-height: 1 !important;
    }}

    [data-testid="stMain"] {{
        background: transparent !important;
    }}

    [data-testid="stHeader"] {{
        background: {header_bg} !important;
        border-bottom: 1px solid var(--rq-border) !important;
        backdrop-filter: blur(8px);
        -webkit-backdrop-filter: blur(8px);
    }}

    [data-testid="stSidebar"] {{
        background: {bg_sidebar} !important;
        border-right: 1px solid var(--rq-border) !important;
    }}

    [data-testid="stSidebar"] .block-container {{
        padding-top: 0 !important;
        padding-left: 0.85rem !important;
        padding-right: 0.85rem !important;
    }}

    [data-testid="stSidebarContent"],
    [data-testid="stSidebarContent"] > div,
    [data-testid="stSidebarUserContent"] {{
        padding-top: 0 !important;
        margin-top: 0 !important;
    }}

    [data-testid="stSidebar"] [data-testid="stVerticalBlock"] > div:first-child {{
        margin-top: 0 !important;
        padding-top: 0 !important;
    }}

    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] input,
    [data-testid="stSidebar"] textarea,
    [data-testid="stSidebar"] button,
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"],
    [data-testid="stSidebar"] [data-testid="stWidgetLabel"] {{
        font-family: 'Sora', sans-serif !important;
    }}

    /* Keep Streamlit icon ligatures on icon font to avoid text like keyboard_arrow_right */
    [data-testid="stSidebar"] .material-icons,
    [data-testid="stSidebar"] .material-symbols-outlined,
    [data-testid="stSidebar"] .material-symbols-rounded,
    [data-testid="stSidebar"] [class*="material-symbols"] {{
        font-family: "Material Symbols Rounded", "Material Symbols Outlined", "Material Icons" !important;
        letter-spacing: normal !important;
        text-transform: none !important;
        line-height: 1 !important;
    }}

    [data-testid="stSidebar"] [data-testid="stVerticalBlock"] {{
        gap: 0.45rem !important;
    }}

    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] input,
    [data-testid="stSidebar"] textarea,
    [data-testid="stSidebar"] button {{
        line-height: 1.35 !important;
    }}

    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {{
        margin-top: 0 !important;
        margin-bottom: 0.35rem !important;
    }}

    [data-testid="stSidebar"] [data-testid="stWidgetLabel"] {{
        margin-bottom: 0.35rem !important;
    }}

    [data-testid="stSidebar"] .stButton,
    [data-testid="stSidebar"] .stSelectbox,
    [data-testid="stSidebar"] .stSlider,
    [data-testid="stSidebar"] .stToggle,
    [data-testid="stSidebar"] .stTextInput,
    [data-testid="stSidebar"] .stNumberInput {{
        margin-top: 0.06rem !important;
        margin-bottom: 0.2rem !important;
    }}

    [data-testid="stSidebar"] .stToggle label {{
        margin-top: 0 !important;
        margin-bottom: 0 !important;
    }}

    [data-testid="stSidebar"] .stToggle {{
        margin-top: 0.2rem !important;
        margin-bottom: 0.25rem !important;
    }}

    [data-testid="stSidebar"] .stToggle [data-testid="stMarkdownContainer"] p {{
        margin: 0 !important;
        line-height: 1.2 !important;
        transform: translateY(2px) !important;
    }}

    [data-testid="stSidebar"] .stToggle > label {{
        align-items: center !important;
        gap: 0.55rem !important;
    }}

    [data-testid="stSidebar"] .stButton {{
        margin-top: 0 !important;
    }}

    .block-container {{
        padding-top: 4.1rem !important;
        padding-bottom: 2.2rem !important;
        max-width: 1200px;
    }}

    .rq-title {{
            font-family: 'DM Serif Display', serif;
            font-size: 2rem;
            font-weight: 400;
            letter-spacing: -0.03em;
            color: var(--rq-text);
            line-height: 1.2;
            margin-bottom: 0.2rem;
        }}

    .rq-subtitle {{
            font-family: 'Sora', sans-serif;
            font-size: 0.84rem;
            color: var(--rq-muted);
            margin-bottom: 1.4rem;
            letter-spacing: 0.01em;
        }}

    .rq-section,
    .rq-table-label,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {{
            font-family: 'Sora', sans-serif !important;
            font-size: 0.72rem !important;
            font-weight: 600 !important;
            letter-spacing: 0.09em !important;
            text-transform: uppercase !important;
            color: var(--rq-muted) !important;
        }}

    .rq-divider {{
            border: none;
            border-top: 1px solid var(--rq-border);
            margin: 1rem 0;
        }}

        .rq-side-title {{
            font-size: 0.76rem;
            font-weight: 700;
            letter-spacing: 0.1em;
            text-transform: uppercase;
            color: var(--rq-muted);
            margin-top: 0;
            margin-bottom: 0.42rem;
            line-height: 1.2;
        }}

        .rq-side-note {{
            font-size: 0.74rem;
            color: var(--rq-muted);
            margin-bottom: 0.4rem;
            line-height: 1.35;
        }}

        .rq-side-rule {{
            border: 0;
            border-top: 1px solid var(--rq-border);
            margin: 0.45rem 0 0.45rem 0;
        }}

        .rq-side-gap-xs {{
            height: 0.16rem;
        }}

        .rq-side-gap-sm {{
            height: 0.3rem;
        }}

        .rq-side-gap-md {{
            height: 0.5rem;
        }}

        [data-testid="stSidebar"] [data-testid="stExpander"] {{
            margin-top: 0.2rem !important;
            margin-bottom: 0.25rem !important;
        }}

        [data-testid="stSidebar"] [data-testid="stExpander"] summary {{
            padding-top: 0.2rem !important;
            padding-bottom: 0.2rem !important;
        }}

    .stButton > button {{
            border-radius: 10px !important;
            font-family: 'Sora', sans-serif !important;
            display: inline-flex !important;
            align-items: center !important;
            justify-content: center !important;
            gap: 0.4rem !important;
            line-height: 1 !important;
            padding-top: 0.58rem !important;
            padding-bottom: 0.58rem !important;
        }}

    .stButton > button[kind="primary"] {{
            background: var(--rq-accent) !important;
            border-color: var(--rq-accent) !important;
        }}

    .stButton > button[kind="primary"]:hover {{
            filter: brightness(1.08) !important;
        }}

    [data-baseweb="select"] > div,
    [data-testid="stTextInput"] input,
    [data-testid="stNumberInput"] input {{
            background: var(--rq-input) !important;
            border: 1px solid var(--rq-border) !important;
            border-radius: 9px !important;
            color: var(--rq-text) !important;
            font-family: 'Sora', sans-serif !important;
        }}

    [data-baseweb="select"] span,
    [data-baseweb="select"] input,
    [data-baseweb="input"] input,
    [data-baseweb="input"] textarea {{
            color: var(--rq-text) !important;
        }}

    [data-baseweb="popover"] [role="listbox"] {{
            background: {select_menu_bg} !important;
            border: 1px solid var(--rq-border) !important;
        }}

    [data-baseweb="popover"] [role="option"] {{
            color: var(--rq-text) !important;
            background: {select_menu_bg} !important;
        }}

    [data-baseweb="popover"] [role="option"][aria-selected="true"] {{
            background: var(--rq-accent-soft) !important;
        }}

    [data-testid="stMetric"] {{
            background: var(--rq-panel) !important;
            border: 1px solid var(--rq-border) !important;
            border-radius: 14px !important;
            padding: 1rem 1.2rem !important;
        }}

    [data-testid="stMetricValue"] > div {{
            color: var(--rq-text) !important;
        }}

    [data-testid="stMetricLabel"] > div,
    [data-testid="stCaptionContainer"] {{
            color: var(--rq-muted) !important;
        }}

    [data-testid="stDataEditor"],
    [data-testid="stDataFrame"],
    [data-testid="stImage"],
    [data-testid="stExpander"] > details {{
            border: 1px solid var(--rq-border) !important;
            border-radius: 12px !important;
            overflow: hidden;
    }}

    [data-testid="stDataFrame"] table th,
    [data-testid="stDataFrame"] table td {{
        text-align: center !important;
    }}

    [data-testid="stDataEditor"] [role="columnheader"],
    [data-testid="stDataEditor"] [role="gridcell"] {{
        text-align: center !important;
        justify-content: center !important;
    }}

    [data-testid="stDataEditor"],
    [data-testid="stDataFrame"],
    .stDataFrameGlideDataEditor,
    .stDataFrameGlideDataEditor > div {{
        --gdg-bg-cell: var(--rq-gdg-bg-cell) !important;
        --gdg-bg-cell-medium: var(--rq-gdg-bg-cell-medium) !important;
        --gdg-bg-header: var(--rq-gdg-bg-header) !important;
        --gdg-bg-header-hovered: var(--rq-gdg-bg-header-hovered) !important;
        --gdg-bg-header-has-focus: var(--rq-gdg-bg-header-focus) !important;
        --gdg-bg-search-result: var(--rq-gdg-bg-search) !important;
        --gdg-border-color: var(--rq-gdg-border) !important;
        --gdg-horizontal-border-color: var(--rq-gdg-h-border) !important;
        --gdg-text-dark: var(--rq-gdg-text) !important;
        --gdg-text-medium: var(--rq-gdg-text-muted) !important;
        --gdg-text-light: var(--rq-gdg-text-muted) !important;
        --gdg-accent-color: var(--rq-gdg-accent) !important;
        --gdg-accent-fg: var(--rq-gdg-accent-fg) !important;
    }}

    .stInfo {{
        background: var(--rq-accent-soft) !important;
        border-color: var(--rq-accent) !important;
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)


def _get_effective_theme_mode(theme_preference: str) -> str:
    if theme_preference == "Light":
        return "light"
    if theme_preference == "Dark":
        return "dark"
    configured_base = st.get_option("theme.base")
    if configured_base in {"light", "dark"}:
        return configured_base
    return "light"


def _figure_to_png_bytes(figure: plt.Figure) -> bytes:
    buffer = BytesIO()
    figure.savefig(buffer, format="png", dpi=170, bbox_inches="tight", facecolor=figure.get_facecolor())
    buffer.seek(0)
    return buffer.getvalue()


def _build_state_from_routes(routes_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    supply_df = (
        routes_df.groupby("factory", as_index=False)["supply"].max().rename(columns={"supply": "value"})
    )
    demand_df = (
        routes_df.groupby("warehouse", as_index=False)["demand"].max().rename(columns={"demand": "value"})
    )
    cost_matrix_df = routes_df.pivot(index="factory", columns="warehouse", values="cost").sort_index()
    return supply_df, demand_df, cost_matrix_df


def _get_editor_state(scenario_name: str, routes_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    state_key = f"editor_state_{scenario_name}"
    if state_key not in st.session_state:
        supply_df, demand_df, cost_matrix_df = _build_state_from_routes(routes_df)
        st.session_state[state_key] = {
            "supply_df": supply_df,
            "demand_df": demand_df,
            "cost_matrix_df": cost_matrix_df,
        }
    state = st.session_state[state_key]
    return state["supply_df"], state["demand_df"], state["cost_matrix_df"]


def _save_editor_state(
    scenario_name: str,
    supply_df: pd.DataFrame,
    demand_df: pd.DataFrame,
    cost_matrix_df: pd.DataFrame,
) -> None:
    state_key = f"editor_state_{scenario_name}"
    st.session_state[state_key] = {
        "supply_df": supply_df,
        "demand_df": demand_df,
        "cost_matrix_df": cost_matrix_df,
    }


def _to_optimizer_inputs(
    supply_df: pd.DataFrame,
    demand_df: pd.DataFrame,
    cost_matrix_df: pd.DataFrame,
) -> tuple[pd.DataFrame, dict, dict]:
    supply_map = {
        str(row["factory"]): float(row["value"])
        for _, row in supply_df.iterrows()
        if str(row["factory"]).strip()
    }
    demand_map = {
        str(row["warehouse"]): float(row["value"])
        for _, row in demand_df.iterrows()
        if str(row["warehouse"]).strip()
    }

    rows: list[dict] = []
    for factory, supply_value in supply_map.items():
        for warehouse, demand_value in demand_map.items():
            if factory not in cost_matrix_df.index or warehouse not in cost_matrix_df.columns:
                raise RuntimeError("Cost matrix is missing one or more factory-warehouse routes.")
            cost_value = float(cost_matrix_df.loc[factory, warehouse])
            rows.append({
                "scenario": "interactive",
                "factory": factory,
                "warehouse": warehouse,
                "supply": supply_value,
                "demand": demand_value,
                "cost": cost_value,
            })

    routes_df = pd.DataFrame(rows)
    return routes_df, supply_map, demand_map


def _format_scenario_label(scenario: str) -> str:
    return scenario.replace("_", " ").title()


def _format_provider_label(provider: str) -> str:
    normalized = provider.strip().lower()
    labels = {
        "openai": "OpenAI",
        "groq": "Groq",
        "cerebras": "Cerebras",
        "google": "Google",
    }
    return labels.get(normalized, provider[:1].upper() + provider[1:])


def _render_centered_grid(
    frame: pd.DataFrame,
    key: str,
    editable: bool,
    non_editable_cols: Optional[List[str]] = None,
    height: Optional[int] = None,
) -> pd.DataFrame:
    if HAS_AGGRID:
        frame_for_grid = frame.copy().reset_index(drop=True)
        # Guard against framework-added helper columns leaking into the grid.
        frame_for_grid = frame_for_grid[
            [column for column in frame_for_grid.columns if not str(column).startswith("_")]
        ]

        row_count = max(1, len(frame_for_grid.index))
        computed_height = height if height is not None else min(460, 38 + (row_count * 36))

        locked_columns = set(non_editable_cols or [])
        column_defs = [
            {
                "field": str(column_name),
                "editable": editable and str(column_name) not in locked_columns,
                "sortable": False,
                "filter": False,
                "resizable": False,
                "suppressMenu": True,
                "flex": 1,
                "cellStyle": {"textAlign": "center"},
                "headerClass": "rq-ag-center-header",
            }
            for column_name in frame_for_grid.columns
        ]

        grid_options = {
            "columnDefs": column_defs,
            "defaultColDef": {
                "editable": editable,
                "sortable": False,
                "filter": False,
                "resizable": False,
                "suppressMenu": True,
                "cellStyle": {"textAlign": "center"},
            },
            "headerHeight": 36,
            "rowHeight": 36,
            "animateRows": False,
            "suppressHorizontalScroll": True,
            "ensureDomOrder": True,
        }

        response = AgGrid(
            frame_for_grid,
            gridOptions=grid_options,
            update_mode=GridUpdateMode.VALUE_CHANGED,
            fit_columns_on_grid_load=True,
            allow_unsafe_jscode=False,
            theme="streamlit",
            key=key,
            height=computed_height,
            custom_css={
                ".ag-header-cell-label": {"justify-content": "center"},
                ".rq-ag-center-header .ag-header-cell-label": {"justify-content": "center"},
                ".ag-cell": {
                    "display": "flex",
                    "align-items": "center",
                    "justify-content": "center",
                },
                ".ag-cell-inline-editing input": {
                    "text-align": "center !important",
                },
                ".ag-input-field-input": {
                    "text-align": "center !important",
                },
            },
        )
        cleaned = pd.DataFrame(response["data"])
        cleaned = cleaned[[column for column in cleaned.columns if str(column) in frame.columns]]
        return cleaned

    return frame


def main() -> None:
    st.set_page_config(page_title="RouteIQ", layout="wide", page_icon="🔀")

    # ── Sidebar ──────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("<div class='rq-side-title'>Controls</div>", unsafe_allow_html=True)
        st.markdown("<div class='rq-side-gap-sm'></div>", unsafe_allow_html=True)

        auto_run = st.toggle("Auto-run on changes", value=False)
        run_clicked = st.button("▶  Run optimization", type="primary", width="stretch")

        st.markdown("<div class='rq-side-gap-md'></div>", unsafe_allow_html=True)
        st.markdown("<div class='rq-side-title'>Scenario</div>", unsafe_allow_html=True)

        st.markdown("<div class='rq-side-gap-xs'></div>", unsafe_allow_html=True)
        scenario_name = st.selectbox(
            "Scenario",
            SCENARIOS,
            index=0,
            format_func=_format_scenario_label,
            label_visibility="collapsed",
        )
        st.markdown("<div class='rq-side-gap-sm'></div>", unsafe_allow_html=True)
        cost_multiplier = st.slider("Cost multiplier", 0.5, 3.0, 1.0, 0.1)

        st.markdown("<div class='rq-side-gap-md'></div>", unsafe_allow_html=True)
        st.markdown("<div class='rq-side-title'>AI Briefing</div>", unsafe_allow_html=True)
        st.markdown("<div class='rq-side-gap-md'></div>", unsafe_allow_html=True)

        provider_options = get_supported_providers()
        default_provider = "google" if "google" in provider_options else provider_options[0]
        selected_provider = st.selectbox(
            "Provider",
            provider_options,
            index=provider_options.index(default_provider),
            format_func=_format_provider_label,
        )

        model_options = get_provider_model_options(selected_provider)
        default_model = get_provider_default_model(selected_provider)
        default_model_index = model_options.index(default_model) if default_model in model_options else 0
        selected_model = st.selectbox("Model", model_options, index=default_model_index)
        custom_model = st.text_input("Custom model override", value="", placeholder="Optional")
        active_model = custom_model.strip() or selected_model

        st.markdown("<div class='rq-side-gap-md'></div>", unsafe_allow_html=True)
        st.markdown("<div class='rq-side-title'>Network Builder</div>", unsafe_allow_html=True)
        st.markdown("<div class='rq-side-gap-md'></div>", unsafe_allow_html=True)

    effective_theme_mode = "light"
    _apply_ui_theme(effective_theme_mode)
    is_dark_mode = False

    # ── Load data ─────────────────────────────────────────────────────────
    base_routes_df, _, _ = load_scenario(str(DATA_FILE_PATH), scenario_name)
    supply_df, demand_df, cost_matrix_df = _get_editor_state(scenario_name, base_routes_df)

    # ── Network builder (sidebar, continued) ─────────────────────────────
    with st.sidebar:
        with st.expander("Factories", expanded=False):
            new_factory_name = st.text_input("Name", key=f"new_factory_{scenario_name}", placeholder="Factory name")
            new_factory_supply = st.number_input(
                "Supply", min_value=0.0, value=10.0, step=1.0, key=f"new_factory_supply_{scenario_name}"
            )
            if st.button("Add factory", key=f"add_factory_{scenario_name}", width="stretch"):
                clean_name = new_factory_name.strip()
                if not clean_name:
                    st.warning("Factory name cannot be empty.")
                elif clean_name in supply_df["factory"].tolist():
                    st.warning("Factory already exists.")
                else:
                    supply_df = pd.concat(
                        [supply_df, pd.DataFrame([{"factory": clean_name, "value": float(new_factory_supply)}])],
                        ignore_index=True,
                    )
                    default_cost = float(cost_matrix_df.values.mean()) if not cost_matrix_df.empty else 5.0
                    cost_matrix_df.loc[clean_name, cost_matrix_df.columns] = default_cost
                    _save_editor_state(scenario_name, supply_df, demand_df, cost_matrix_df)
                    st.rerun()

            remove_factory = st.selectbox(
                "Remove", options=["—"] + sorted(supply_df["factory"].tolist()),
                key=f"remove_factory_{scenario_name}",
            )
            if st.button("Remove factory", key=f"remove_factory_btn_{scenario_name}", width="stretch"):
                if remove_factory != "—" and len(supply_df) > 1:
                    supply_df = supply_df[supply_df["factory"] != remove_factory].reset_index(drop=True)
                    if remove_factory in cost_matrix_df.index:
                        cost_matrix_df = cost_matrix_df.drop(index=remove_factory)
                    _save_editor_state(scenario_name, supply_df, demand_df, cost_matrix_df)
                    st.rerun()
                elif len(supply_df) <= 1:
                    st.warning("At least one factory is required.")

            st.markdown("<div class='rq-side-gap-sm'></div>", unsafe_allow_html=True)

        with st.expander("Warehouses", expanded=False):
            new_warehouse_name = st.text_input("Name", key=f"new_wh_{scenario_name}", placeholder="Warehouse name")
            new_warehouse_demand = st.number_input(
                "Demand", min_value=0.0, value=10.0, step=1.0, key=f"new_wh_demand_{scenario_name}"
            )
            if st.button("Add warehouse", key=f"add_warehouse_{scenario_name}", width="stretch"):
                clean_name = new_warehouse_name.strip()
                if not clean_name:
                    st.warning("Warehouse name cannot be empty.")
                elif clean_name in demand_df["warehouse"].tolist():
                    st.warning("Warehouse already exists.")
                else:
                    demand_df = pd.concat(
                        [demand_df, pd.DataFrame([{"warehouse": clean_name, "value": float(new_warehouse_demand)}])],
                        ignore_index=True,
                    )
                    default_cost = float(cost_matrix_df.values.mean()) if not cost_matrix_df.empty else 5.0
                    cost_matrix_df[clean_name] = default_cost
                    _save_editor_state(scenario_name, supply_df, demand_df, cost_matrix_df)
                    st.rerun()

            remove_warehouse = st.selectbox(
                "Remove", options=["—"] + sorted(demand_df["warehouse"].tolist()),
                key=f"remove_wh_{scenario_name}",
            )
            if st.button("Remove warehouse", key=f"remove_wh_btn_{scenario_name}", width="stretch"):
                if remove_warehouse != "—" and len(demand_df) > 1:
                    demand_df = demand_df[demand_df["warehouse"] != remove_warehouse].reset_index(drop=True)
                    if remove_warehouse in cost_matrix_df.columns:
                        cost_matrix_df = cost_matrix_df.drop(columns=remove_warehouse)
                    _save_editor_state(scenario_name, supply_df, demand_df, cost_matrix_df)
                    st.rerun()
                elif len(demand_df) <= 1:
                    st.warning("At least one warehouse is required.")

    # ── Page header ───────────────────────────────────────────────────────
    st.markdown("<div class='rq-title'>RouteIQ</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='rq-subtitle'>Multi-scenario transportation optimizer — what-if analysis, network visualization, AI briefing</div>",
        unsafe_allow_html=True,
    )

    # ── Editable network inputs ───────────────────────────────────────────
    st.markdown("<p class='rq-section'>Network Inputs</p>", unsafe_allow_html=True)

    factory_list = supply_df["factory"].tolist()
    warehouse_list = demand_df["warehouse"].tolist()
    default_cost_value = float(cost_matrix_df.values.mean()) if not cost_matrix_df.empty else 5.0
    cost_matrix_df = cost_matrix_df.reindex(index=factory_list, columns=warehouse_list).fillna(default_cost_value)

    _, table_center_col, _ = st.columns([0.06, 0.88, 0.06])
    with table_center_col:
        edit_col_1, edit_col_2 = st.columns(2, gap="medium")
        with edit_col_1:
            st.markdown("<p class='rq-table-label'>Factory Supply</p>", unsafe_allow_html=True)
            supply_df = _render_centered_grid(
                supply_df,
                key=f"supply_editor_{scenario_name}",
                editable=True,
                non_editable_cols=["factory"],
            )
        with edit_col_2:
            st.markdown("<p class='rq-table-label'>Warehouse Demand</p>", unsafe_allow_html=True)
            demand_df = _render_centered_grid(
                demand_df,
                key=f"demand_editor_{scenario_name}",
                editable=True,
                non_editable_cols=["warehouse"],
            )

        st.markdown("<p class='rq-table-label' style='margin-top:1rem'>Route Cost Matrix</p>", unsafe_allow_html=True)
        cost_matrix_edit_df = cost_matrix_df.reset_index(names="factory")
        cost_matrix_edit_df = _render_centered_grid(
            cost_matrix_edit_df,
            key=f"cost_editor_{scenario_name}",
            editable=True,
            non_editable_cols=["factory"],
        )
        cost_matrix_df = cost_matrix_edit_df.set_index("factory")

    _save_editor_state(scenario_name, supply_df, demand_df, cost_matrix_df)

    should_run = auto_run or run_clicked
    results_key = f"results_state_{scenario_name}"
    result_state = st.session_state.get(results_key)

    st.markdown("<hr class='rq-divider'>", unsafe_allow_html=True)

    # ── Run / fetch results ───────────────────────────────────────────────
    if should_run:
        routes_df, supply, demand = _to_optimizer_inputs(supply_df, demand_df, cost_matrix_df)
        if cost_multiplier != 1.0:
            routes_df = routes_df.copy()
            routes_df["cost"] = routes_df["cost"] * cost_multiplier

        try:
            result_df, summary = solve_transportation(routes_df, supply, demand)
        except (RuntimeError, ValueError) as error:
            st.error(f"Optimization failed: {error}")
            return

        briefing_text = ""
        briefing_error = ""
        try:
            briefing_text = generate_executive_briefing(
                summary, scenario_name, provider=selected_provider, model=active_model
            )
        except RuntimeError as error:
            briefing_error = str(error)

        result_state = {
            "routes_df": routes_df,
            "supply": supply,
            "demand": demand,
            "result_df": result_df,
            "summary": summary,
            "briefing_text": briefing_text,
            "briefing_error": briefing_error,
            "provider": selected_provider,
            "model": active_model,
        }
        st.session_state[results_key] = result_state

    elif result_state is None:
        st.info("Adjust inputs above, then click **▶ Run optimization** in the sidebar.")
        return
    else:
        st.caption("Showing results from last run — click Run to refresh.")

    routes_df = result_state["routes_df"]
    supply    = result_state["supply"]
    demand    = result_state["demand"]
    result_df = result_state["result_df"]
    summary   = result_state["summary"]

    # ── Key metrics ───────────────────────────────────────────────────────
    st.markdown("<p class='rq-section'>Key Metrics</p>", unsafe_allow_html=True)
    metric_col_1, metric_col_2, metric_col_3 = st.columns(3, gap="medium")

    with metric_col_1:
        st.metric("Total Transportation Cost", f"{summary['total_cost']:,.2f}")
    with metric_col_2:
        most_utilized_factory, utilization_value = max(
            summary["factory_utilization"].items(), key=lambda item: item[1]
        )
        st.metric("Top Factory Utilization", f"{most_utilized_factory}", delta=f"{utilization_value:.1%} utilized")
    with metric_col_3:
        fully_filled = all(ratio >= 1.0 for ratio in summary["warehouse_fill_ratio"].values())
        st.metric("Demand Coverage", "100%" if fully_filled else "< 100%")

    # ── Charts ────────────────────────────────────────────────────────────
    st.markdown("<p class='rq-section' style='margin-top:1.6rem'>Visualizations</p>", unsafe_allow_html=True)
    chart_col_1, chart_col_2 = st.columns(2, gap="medium")

    with chart_col_1:
        st.markdown("<p class='rq-table-label'>Network Flow</p>", unsafe_allow_html=True)
        figure_network, axis_network = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
        plot_network(
            result_df,
            title=f"{scenario_name.replace('_', ' ').title()} — Network Flow",
            axis=axis_network,
            dark_mode=is_dark_mode,
        )
        st.image(_figure_to_png_bytes(figure_network), width="stretch")
        plt.close(figure_network)

    with chart_col_2:
        st.markdown("<p class='rq-table-label'>Cost Heatmap</p>", unsafe_allow_html=True)
        figure_heatmap, axis_heatmap = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
        plot_cost_heatmap(
            routes_df,
            title=f"{scenario_name.replace('_', ' ').title()} — Cost Heatmap",
            axis=axis_heatmap,
            dark_mode=is_dark_mode,
        )
        st.image(_figure_to_png_bytes(figure_heatmap), width="stretch")
        plt.close(figure_heatmap)

    # ── AI Briefing ───────────────────────────────────────────────────────
    st.markdown("<p class='rq-section' style='margin-top:1.6rem'>AI Executive Briefing</p>", unsafe_allow_html=True)
    if result_state["briefing_text"]:
        st.write(result_state["briefing_text"])
    else:
        st.warning(f"Briefing unavailable — {result_state['briefing_error']}")

    # ── Debug expander ────────────────────────────────────────────────────
    with st.expander("Optimization outputs"):
        out_col1, out_col2 = st.columns(2, gap="medium")
        with out_col1:
            st.markdown("<p class='rq-table-label'>Factory Supply</p>", unsafe_allow_html=True)
            st.json(supply)
        with out_col2:
            st.markdown("<p class='rq-table-label'>Warehouse Demand</p>", unsafe_allow_html=True)
            st.json(demand)
        st.markdown("<p class='rq-table-label' style='margin-top:0.8rem'>Optimized Route Flows</p>", unsafe_allow_html=True)
        _, output_table_col, _ = st.columns([0.06, 0.88, 0.06])
        with output_table_col:
            _render_centered_grid(
                result_df,
                key=f"optimized_output_{scenario_name}",
                editable=False,
            )


if __name__ == "__main__":
    main()
