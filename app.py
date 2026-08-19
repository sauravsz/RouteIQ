import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
import yaml
from io import BytesIO
from pathlib import Path
from typing import List, Optional

from src.ai_explainer import (
    extract_matrix_from_image,
    generate_executive_briefing,
    get_provider_default_model,
    get_provider_model_options,
    get_supported_providers,
)
from src.optimizer import solve_transportation, get_available_solvers
from src.scenarios import load_scenario
from src.visualizations import (
    plot_cost_heatmap,
    plot_network,
    plot_cost_heatmap_plotly,
    plot_network_plotly,
)
from src.features import (
    generate_pdf_report,
    generate_excel_report,
)

SCENARIOS = ["baseline", "disruption", "cost_surge"]
BASE_DIR = Path(__file__).resolve().parent
DATA_FILE_PATH = BASE_DIR / "data" / "scenarios.csv"
CONFIG_FILE_PATH = BASE_DIR / "config.yaml"

@st.cache_data
def _load_config():
    if CONFIG_FILE_PATH.exists():
        with open(CONFIG_FILE_PATH, "r") as f:
            return yaml.safe_load(f)
    return {}

@st.cache_data
def _cached_load_scenario(data_path_str: str, scenario_name: str):
    return load_scenario(data_path_str, scenario_name)

def _apply_ui_theme(theme_mode: str = "Lovable Cream") -> None:
    is_dark = theme_mode == "Dark Mode"
    is_clean = theme_mode == "Clean Light"

    bg_color = "#0b0f19" if is_dark else ("#ffffff" if is_clean else "#f7f4ed")
    sidebar_color = "#090d16" if is_dark else ("#f8fafc" if is_clean else "#f2eee5")
    text_color = "#f8fafc" if is_dark else "#1c1c1c"
    muted_color = "#94a3b8" if is_dark else "#5f5f5d"
    border_color = "rgba(148,163,184,0.18)" if is_dark else ("#e2e8f0" if is_clean else "#eceae4")
    border_interactive = "rgba(148,163,184,0.4)" if is_dark else ("rgba(15,23,42,0.4)" if is_clean else "rgba(28,28,28,0.4)")
    panel_bg = "#111827" if is_dark else ("#ffffff" if is_clean else "#f7f4ed")

    btn_bg = "#6366f1" if is_dark else "#1c1c1c"
    btn_text = "#ffffff" if is_dark else "#fcfbf8"

    gdg_bg_cell = "#111827" if is_dark else ("#ffffff" if is_clean else "#f7f4ed")
    gdg_bg_header = "#090d16" if is_dark else ("#f8fafc" if is_clean else "#f2eee5")
    gdg_text = "#f8fafc" if is_dark else "#1c1c1c"

    css = f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Geist:wght@300;400;500;600;700;800&family=Inter:wght@300;400;500;600;700;800&display=swap');

    :root {{
        --rq-bg: {bg_color};
        --rq-sidebar: {sidebar_color};
        --rq-text: {text_color};
        --rq-text-muted: {muted_color};
        --rq-border: {border_color};
        --rq-border-interactive: {border_interactive};
        --rq-panel: {panel_bg};
        --rq-font-primary: 'Geist', 'Inter', ui-sans-serif, system-ui, -apple-system, sans-serif;

        /* Glide Data Grid (st.data_editor) Overrides */
        --gdg-accent-color: {btn_bg} !important;
        --gdg-accent-light: rgba(99, 102, 241, 0.15) !important;
        --gdg-bg-cell: {gdg_bg_cell} !important;
        --gdg-bg-cell-medium: {gdg_bg_header} !important;
        --gdg-bg-header: {gdg_bg_header} !important;
        --gdg-bg-header-has-focus: {border_color} !important;
        --gdg-bg-header-hovered: {border_color} !important;
        --gdg-text-dark: {gdg_text} !important;
        --gdg-text-medium: {muted_color} !important;
        --gdg-border-color: {border_color} !important;
        --gdg-font-family: 'Geist', 'Inter', sans-serif !important;
    }}

    /* Universal Typography & Element Font Reset per DESIGN.md */
    *, html, body, .stApp, main, .main, [data-testid="stAppViewContainer"], [data-testid="stAppViewBlockContainer"], button, input, select, textarea, label, p, span, div, h1, h2, h3, h4, h5, h6 {{
        font-family: var(--rq-font-primary) !important;
    }}

    html, body, .stApp, main, .main, [data-testid="stAppViewContainer"], [data-testid="stAppViewBlockContainer"] {{
        background-color: var(--rq-bg) !important;
        background: var(--rq-bg) !important;
        color: var(--rq-text) !important;
    }}

    /* Absolute Top Header Bar & Toolbar Overrides */
    header[data-testid="stHeader"], [data-testid="stHeader"], [data-testid="stToolbar"], .stAppHeader, div[data-testid="stDecoration"], header[data-testid="stHeader"] * {{
        background-color: var(--rq-bg) !important;
        background: var(--rq-bg) !important;
        color: var(--rq-text) !important;
    }}

    /* Sidebar & all inner elements */
    section[data-testid="stSidebar"], [data-testid="stSidebar"], [data-testid="stSidebarContent"], [data-testid="stSidebarUserContent"], [data-testid="stSidebarNav"], [data-testid="stSidebarHeader"] {{
        background-color: var(--rq-sidebar) !important;
        background: var(--rq-sidebar) !important;
        border-right: 1px solid var(--rq-border) !important;
    }}

    [data-testid="stSidebar"] * {{
        color: var(--rq-text) !important;
    }}

    /* Typography Hierarchy per DESIGN.md */
    .rq-title {{
        font-family: var(--rq-font-primary) !important;
        font-size: 3.2rem !important;
        font-weight: 600 !important;
        letter-spacing: -1.5px !important;
        color: {text_color} !important;
        line-height: 1.05 !important;
        margin-bottom: 0.3rem !important;
    }}

    .rq-subtitle {{
        font-family: var(--rq-font-primary) !important;
        font-size: 1.125rem !important;
        font-weight: 400 !important;
        color: var(--rq-text-muted) !important;
        margin-bottom: 2rem !important;
        line-height: 1.38 !important;
    }}

    .rq-section {{
        font-family: var(--rq-font-primary) !important;
        font-size: 0.875rem !important;
        font-weight: 600 !important;
        letter-spacing: -0.01em !important;
        text-transform: uppercase !important;
        color: var(--rq-text) !important;
        margin-bottom: 0.8rem !important;
    }}

    .rq-side-title {{
        font-size: 0.76rem !important;
        font-weight: 600 !important;
        letter-spacing: 0.05em !important;
        text-transform: uppercase !important;
        color: var(--rq-text) !important;
        margin-top: 0 !important;
        margin-bottom: 0.42rem !important;
        line-height: 1.2 !important;
    }}

    .rq-divider {{
        border: none !important;
        border-top: 1px solid var(--rq-border) !important;
        margin: 1.5rem 0 !important;
    }}

    /* Metrics & Cards per DESIGN.md */
    [data-testid="stMetric"] {{
        background: var(--rq-panel) !important;
        border: 1px solid var(--rq-border) !important;
        border-radius: 12px !important;
        padding: 1.2rem 1.4rem !important;
        box-shadow: none !important;
    }}

    [data-testid="stMetricLabel"] p {{
        color: var(--rq-text-muted) !important;
        font-size: 0.875rem !important;
        font-weight: 400 !important;
    }}

    [data-testid="stMetricValue"] > div {{
        color: var(--rq-text) !important;
        font-size: 2rem !important;
        font-weight: 600 !important;
        letter-spacing: -0.9px !important;
    }}

    /* Primary Buttons */
    .stButton > button {{
        background-color: {btn_bg} !important;
        color: {btn_text} !important;
        border-radius: 6px !important;
        padding: 8px 16px !important;
        border: none !important;
        font-family: var(--rq-font-primary) !important;
        font-size: 1rem !important;
        font-weight: 400 !important;
        box-shadow: rgba(255,255,255,0.2) 0px 0.5px 0px 0px inset, rgba(0,0,0,0.2) 0px 0px 0px 0.5px inset, rgba(0,0,0,0.05) 0px 1px 2px 0px !important;
        transition: opacity 0.15s ease !important;
    }}

    /* Sidebar Collapse & Header Icon Buttons Override (removes black box around icons) */
    [data-testid="stSidebarCollapseButton"] button, [data-testid="stSidebarHeader"] button, [data-testid="stSidebarCollapseButton"] *, [data-testid="stHeader"] button, [data-testid="stSidebarNav"] button {{
        background-color: transparent !important;
        background: transparent !important;
        border: none !important;
        box-shadow: none !important;
        color: var(--rq-text) !important;
    }}

    [data-testid="stSidebarCollapseButton"] button:hover, [data-testid="stSidebarHeader"] button:hover {{
        background-color: rgba(28, 28, 28, 0.08) !important;
    }}

    .stButton > button:hover {{
        opacity: 0.85 !important;
        color: #ffffff !important;
    }}

    /* Download Buttons & Secondary Outline Buttons */
    .stDownloadButton > button {{
        background-color: transparent !important;
        color: {text_color} !important;
        border: 1px solid var(--rq-border-interactive) !important;
        border-radius: 6px !important;
        padding: 8px 16px !important;
        font-family: var(--rq-font-primary) !important;
        font-size: 0.875rem !important;
        font-weight: 400 !important;
    }}

    .stDownloadButton > button:hover {{
        background-color: rgba(28, 28, 28, 0.04) !important;
    }}

    /* File Uploader Dropzone Overrides */
    [data-testid="stFileUploader"], [data-testid="stFileUploaderDropzone"], section[data-testid="stFileUploaderDropzone"], div[data-testid="stFileUploaderDropzone"] {{
        background-color: var(--rq-sidebar) !important;
        background: var(--rq-sidebar) !important;
        border: 1px dashed var(--rq-border-interactive) !important;
        border-radius: 12px !important;
        color: var(--rq-text) !important;
    }}

    [data-testid="stFileUploaderDropzone"] * {{
        color: var(--rq-text) !important;
        background-color: transparent !important;
    }}

    /* Inputs, Selectboxes, Dropdowns, Textareas per DESIGN.md */
    input[type="text"], input[type="number"], input[type="password"], textarea, [data-baseweb="select"] > div, [data-baseweb="popover"], [data-baseweb="menu"], [role="listbox"] {{
        background-color: var(--rq-bg) !important;
        color: var(--rq-text) !important;
        border: 1px solid var(--rq-border) !important;
        border-radius: 6px !important;
        font-family: var(--rq-font-primary) !important;
    }}

    [data-testid="stSidebar"] div[data-baseweb="select"] > div {{
        background-color: var(--rq-sidebar) !important;
        border: 1px solid var(--rq-border) !important;
    }}

    [data-baseweb="menu"] li, [role="option"] {{
        background-color: var(--rq-bg) !important;
        color: var(--rq-text) !important;
    }}

    [data-baseweb="menu"] li:hover, [role="option"]:hover {{
        background-color: {sidebar_color} !important;
    }}

    /* Sliders & Toggles */
    [data-testid="stSlider"] * {{
        color: var(--rq-text) !important;
    }}

    div[data-baseweb="slider"] div {{
        background-color: var(--rq-text) !important;
    }}

    /* Tabs per DESIGN.md */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 1.5rem;
        border-bottom: 1px solid var(--rq-border);
        background: transparent !important;
    }}

    .stTabs [data-baseweb="tab"] {{
        color: var(--rq-text-muted) !important;
        font-weight: 400 !important;
        font-size: 1rem !important;
        font-family: var(--rq-font-primary) !important;
        background: transparent !important;
    }}

    .stTabs [aria-selected="true"] {{
        color: var(--rq-text) !important;
        font-weight: 600 !important;
        border-bottom: 2px solid var(--rq-text) !important;
    }}

    /* Expanders & Forms per DESIGN.md */
    [data-testid="stExpander"], summary[data-testid="stExpanderToggleHeader"], div[data-testid="stExpanderDetails"], [data-testid="stForm"] {{
        background-color: var(--rq-bg) !important;
        background: var(--rq-bg) !important;
        border: 1px solid var(--rq-border) !important;
        border-radius: 12px !important;
        color: var(--rq-text) !important;
        box-shadow: none !important;
    }}

    summary[data-testid="stExpanderToggleHeader"] * {{
        color: var(--rq-text) !important;
    }}

    /* Dataframe / Data Editor Table overrides */
    [data-testid="stDataFrame"], [data-testid="stDataEditor"] {{
        border: 1px solid var(--rq-border) !important;
        border-radius: 12px !important;
        overflow: hidden !important;
        background-color: var(--rq-bg) !important;
    }}

    /* Alerts (success, info, warning, error) */
    [data-testid="stNotification"], div[data-baseweb="notification"], .stAlert {{
        background-color: {sidebar_color} !important;
        border: 1px solid var(--rq-border) !important;
        color: var(--rq-text) !important;
        border-radius: 8px !important;
    }}

    /* Code Blocks */
    .stCodeBlock, [data-testid="stCodeBlock"] pre {{
        background-color: {sidebar_color} !important;
        border: 1px solid var(--rq-border) !important;
        border-radius: 8px !important;
        color: var(--rq-text) !important;
    }}

    .rq-table-label {{
        font-family: var(--rq-font-primary);
        font-size: 0.875rem;
        font-weight: 600;
        color: var(--rq-text);
        margin-bottom: 0.4rem;
    }}

    .rq-side-gap-xs {{ height: 0.16rem; }}
    .rq-side-gap-sm {{ height: 0.3rem; }}
    .rq-side-gap-md {{ height: 0.5rem; }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

def _figure_to_png_bytes(figure: plt.Figure) -> bytes:
    buffer = BytesIO()
    figure.savefig(buffer, format="png", dpi=170, bbox_inches="tight", facecolor=figure.get_facecolor())
    buffer.seek(0)
    return buffer.getvalue()

DEMAND_ROW = "Demand"
PROHIBITED_COST = 999999.0

def _parse_cell_value(val, route_label: str) -> float:
    import numpy as np
    if val is None:
        raise RuntimeError(f"Missing value for {route_label}. Fill in all matrix cells.")
    if isinstance(val, (int, float)):
        if np.isnan(val):
            raise RuntimeError(f"Missing value for {route_label}. Fill in all matrix cells.")
        return float(val)
    sval = str(val).strip().lower()
    if sval in ("x", "-", "inf", "n/a", "na", "prohibited", "none", "impossible"):
        return PROHIBITED_COST
    try:
        return float(sval)
    except ValueError:
        raise RuntimeError(f"Invalid value '{val}' for {route_label}. Enter a number or 'x' for prohibited.")

def _build_matrix_from_routes(routes_df: pd.DataFrame) -> pd.DataFrame:
    """Unified grid: rows = factories, cols = [Factory, Supply, <warehouses...>], last row = Demand."""
    supply_map = routes_df.groupby("factory")["supply"].max()
    demand_map = routes_df.groupby("warehouse")["demand"].max()
    cost_matrix = routes_df.pivot(index="factory", columns="warehouse", values="cost").sort_index()
    matrix = cost_matrix.reset_index(names="Factory")
    matrix.insert(1, "Supply", matrix["Factory"].map(supply_map).astype(float))
    demand_row = {"Factory": DEMAND_ROW, "Supply": 0.0}
    demand_row.update({w: float(demand_map[w]) for w in cost_matrix.columns})
    demand_df = pd.DataFrame([demand_row]).astype(matrix.dtypes.to_dict(), errors="ignore")
    return pd.concat([matrix, demand_df], ignore_index=True)

def _get_matrix_state(scenario_name: str, routes_df: pd.DataFrame) -> pd.DataFrame:
    state_key = f"matrix_state_{scenario_name}"
    if state_key not in st.session_state:
        st.session_state[state_key] = _build_matrix_from_routes(routes_df)
    return st.session_state[state_key]

def _build_default_assignment_matrix() -> pd.DataFrame:
    return pd.DataFrame([
        {"Agent": "Worker 1", "Task A": 9.0, "Task B": 2.0, "Task C": 7.0},
        {"Agent": "Worker 2", "Task A": 6.0, "Task B": 4.0, "Task C": 3.0},
        {"Agent": "Worker 3", "Task A": 5.0, "Task B": 8.0, "Task C": 1.0},
    ])

def _get_assignment_matrix_state(scenario_name: str) -> pd.DataFrame:
    state_key = f"assignment_matrix_state_{scenario_name}"
    if state_key not in st.session_state:
        st.session_state[state_key] = _build_default_assignment_matrix()
    return st.session_state[state_key]

def _to_assignment_inputs(
    matrix_df: pd.DataFrame,
    maximize: bool = False,
    solver_type: str = "cbc",
    timeout_seconds: int = 10,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    df = matrix_df.copy()
    df["Agent"] = df["Agent"].astype(str).str.strip()
    df = df[df["Agent"] != ""]
    if df["Agent"].duplicated().any():
        df = df.drop_duplicates(subset="Agent", keep="last")

    task_cols = [c for c in df.columns if c != "Agent"]
    if not task_cols:
        raise RuntimeError("No tasks found in assignment matrix.")
    if df.empty:
        raise RuntimeError("No agents found in assignment matrix.")

    supply_map = {row["Agent"]: 1.0 for _, row in df.iterrows()}
    demand_map = {t: 1.0 for t in task_cols}

    rows: list[dict] = []
    for _, arow in df.iterrows():
        agent = arow["Agent"]
        for t in task_cols:
            cval = _parse_cell_value(arow[t], f"assignment {agent} → {t}")
            rows.append({
                "scenario": "assignment",
                "factory": agent,
                "warehouse": t,
                "supply": 1.0,
                "demand": 1.0,
                "cost": float(cval),
            })

    routes_df = pd.DataFrame(rows)

    if maximize:
        max_c = float(routes_df["cost"].max())
        routes_to_solve = routes_df.copy()
        routes_to_solve["cost"] = max_c - routes_to_solve["cost"]
        result_df, _ = solve_transportation(
            routes_to_solve, supply_map, demand_map, solver_type=solver_type, timeout_seconds=timeout_seconds
        )
        result_df["cost"] = routes_df["cost"]
        result_df["route_cost"] = result_df["flow"] * result_df["cost"]
        summary = {
            "total_cost": float(result_df["route_cost"].sum()),
            "objective": "maximize",
            "open_lanes_count": int(result_df[result_df["flow"] > 0]["flow"].count()),
            "factory_utilization": {a: float(result_df[result_df["factory"] == a]["flow"].sum()) for a in supply_map},
            "warehouse_fill_ratio": {t: float(result_df[result_df["warehouse"] == t]["flow"].sum()) for t in demand_map},
        }
    else:
        result_df, summary = solve_transportation(
            routes_df, supply_map, demand_map, solver_type=solver_type, timeout_seconds=timeout_seconds
        )
        summary["objective"] = "minimize"

    active_df = result_df[result_df["flow"] > 0].copy()
    active_df["warehouse"] = active_df["warehouse"].replace({"Dummy_Warehouse": "Unassigned"})
    active_df["factory"] = active_df["factory"].replace({"Dummy_Factory": "Unassigned"})
    summary["active_assignments"] = active_df.to_dict(orient="records")

    return routes_df, result_df, summary

def _to_optimizer_inputs(matrix_df: pd.DataFrame) -> tuple[pd.DataFrame, dict, dict]:
    df = matrix_df.copy()
    df["Factory"] = df["Factory"].astype(str).str.strip()
    df = df[df["Factory"] != ""]
    warehouse_cols = [c for c in df.columns if c not in ("Factory", "Supply")]

    demand_mask = df["Factory"].str.lower() == DEMAND_ROW.lower()
    if not demand_mask.any():
        raise RuntimeError("Matrix is missing the Demand row.")
    demand_row = df[demand_mask].iloc[0]
    factory_df = df[~demand_mask]

    # BUG 5: deduplicate factory names (last row wins)
    if factory_df["Factory"].duplicated().any():
        factory_df = factory_df.drop_duplicates(subset="Factory", keep="last")

    # BUG 3: warn-worthy blank rows already filtered by != "", but catch NaN names
    factory_df = factory_df[factory_df["Factory"].notna()]
    if factory_df.empty:
        raise RuntimeError("No factories found in the matrix. Add at least one factory row.")

    import numpy as np

    supply_map = {}
    for _, row in factory_df.iterrows():
        sv = row["Supply"]
        if sv is None or (isinstance(sv, float) and np.isnan(sv)):
            raise RuntimeError(f"Factory '{row['Factory']}' has no Supply value.")
        supply_map[row["Factory"]] = float(sv)

    demand_map = {}
    for w in warehouse_cols:
        dv = demand_row[w]
        if dv is None or (isinstance(dv, float) and np.isnan(dv)):
            raise RuntimeError(f"Warehouse '{w}' has no Demand value.")
        demand_map[w] = float(dv)

    # Parse cell values with prohibited mask support ('x', '-', 'inf')
    rows: list[dict] = []
    for _, frow in factory_df.iterrows():
        factory = frow["Factory"]
        for w in warehouse_cols:
            cost_val = _parse_cell_value(frow[w], f"route {factory} → {w}")
            rows.append({
                "scenario": "interactive",
                "factory": factory,
                "warehouse": w,
                "supply": supply_map[factory],
                "demand": demand_map[w],
                "cost": float(cost_val),
            })

    return pd.DataFrame(rows), supply_map, demand_map

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

def main() -> None:
    st.set_page_config(page_title="RouteIQ", layout="wide", page_icon="🔀")
    theme_choice = st.session_state.get("ui_theme_selector", "Lovable Cream")
    _apply_ui_theme(theme_choice)
    is_dark_mode = (theme_choice == "Dark Mode")

    # ── Sidebar ──────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("<div class='rq-side-title'>Controls</div>", unsafe_allow_html=True)
        auto_run = st.toggle("Auto-run on changes", value=False)
        use_plotly = st.toggle("Interactive Plotly charts", value=True)
        run_clicked = st.button("▶  Run optimization", type="primary", width="stretch")

        st.markdown("<div class='rq-side-gap-md'></div>", unsafe_allow_html=True)
        st.markdown("<div class='rq-side-title'>Data Source & Scenario</div>", unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader("Upload Custom CSV", type=["csv"], help="Upload custom scenario dataset")
        uploaded_image = st.file_uploader(
            "Upload Cost Matrix Image",
            type=["png", "jpg", "jpeg", "webp"],
            help="Photo of a cost matrix table — AI extracts supply, demand and costs",
        )
        google_vision_key = st.text_input(
            "Google API Key (for image extraction)",
            type="password",
            placeholder="Required for image upload",
            key="google_vision_key",
        ) if uploaded_image is not None else ""

        has_custom_input = uploaded_file is not None or uploaded_image is not None
        scenario_options = ["custom"] if has_custom_input else SCENARIOS
        scenario_name = st.selectbox(
            "Scenario",
            scenario_options,
            index=0,
            format_func=_format_scenario_label,
        )
        
        col_mult, col_reset = st.columns([0.7, 0.3])
        with col_mult:
            cost_multiplier = st.slider("Cost multiplier", 0.5, 3.0, 1.0, 0.1)
        with col_reset:
            st.markdown("<div style='height:28px;'></div>", unsafe_allow_html=True)
            if st.button("Reset", key="reset_multiplier"):
                cost_multiplier = 1.0

        st.markdown("<div class='rq-side-gap-md'></div>", unsafe_allow_html=True)
        st.markdown("<div class='rq-side-title'>Solver & MIP Settings</div>", unsafe_allow_html=True)
        
        available_solvers = get_available_solvers()
        selected_solver = st.selectbox("Solver engine", list(available_solvers.keys()), index=0)
        solver_timeout = st.slider("Solver Timeout (sec)", 2, 60, 10, 1)
        enable_mip = st.checkbox("Enable MIP Fixed-Charge", value=False)
        fixed_lane_cost = st.number_input("Fixed Lane Cost", min_value=0.0, value=0.0, step=10.0) if enable_mip else 0.0

        st.markdown("<div class='rq-side-gap-md'></div>", unsafe_allow_html=True)
        st.markdown("<div class='rq-side-title'>AI Briefing</div>", unsafe_allow_html=True)

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
        custom_model = st.text_input("Custom model override", value="", placeholder="e.g. gpt-4")
        custom_api_key = st.text_input(
            f"{_format_provider_label(selected_provider)} API Key", 
            type="password", 
            placeholder="Optional (uses default if empty)",
        )
        active_model = custom_model.strip() or selected_model

    # ── Load data ─────────────────────────────────────────────────────────
    if uploaded_file is not None:
        custom_df = pd.read_csv(uploaded_file)
        base_routes_df, _, _ = load_scenario(custom_df, "custom")
    else:
        base_routes_df, _, _ = _cached_load_scenario(str(DATA_FILE_PATH), scenario_name)

    # ── Image extraction ──────────────────────────────────────────────────
    if uploaded_image is not None:
        image_hash = hash(uploaded_image.getvalue())
        if st.session_state.get("extracted_image_hash") != image_hash:
            try:
                with st.spinner("Extracting cost matrix from image via Gemini..."):
                    csv_text = extract_matrix_from_image(
                        uploaded_image.getvalue(),
                        mime_type=uploaded_image.type or "image/png",
                        api_key=(google_vision_key or "").strip(),
                    )
                extracted_df = pd.read_csv(BytesIO(csv_text.encode("utf-8")))
                extracted_routes, _, _ = load_scenario(extracted_df, "custom")
                st.session_state["matrix_state_custom"] = _build_matrix_from_routes(extracted_routes)
                st.session_state["extracted_image_hash"] = image_hash
                st.session_state["extracted_csv"] = csv_text
                st.session_state["force_run"] = True
                st.success("Matrix extracted — loaded into editor below.")
                with st.expander("Extracted CSV preview", expanded=False):
                    st.code(csv_text, language="csv")
                    st.download_button(
                        "Download extracted CSV",
                        data=csv_text,
                        file_name="routeiq_extracted.csv",
                        mime="text/csv",
                        key="dl_extracted_csv",
                    )
            except RuntimeError as error:
                if "NO_GOOGLE_API_KEY" in str(error):
                    st.error("No Google API key found. Paste your Gemini API key in the sidebar (AI Briefing section) and re-upload.")
                else:
                    st.error(f"Image extraction failed: {error}")

    if uploaded_image is not None and "matrix_state_custom" in st.session_state:
        scenario_name = "custom"
        matrix_df = st.session_state["matrix_state_custom"]
    else:
        matrix_df = _get_matrix_state(scenario_name, base_routes_df)

    # ── Page Header & Theme Switcher ──────────────────────────────────────
    head_col1, head_col2 = st.columns([0.75, 0.25])
    with head_col1:
        st.markdown("<div class='rq-title'>RouteIQ</div>", unsafe_allow_html=True)
        st.markdown(
            "<div class='rq-subtitle'>Multi-scenario transportation & assignment optimizer with AI executive briefing</div>",
            unsafe_allow_html=True,
        )
    with head_col2:
        st.markdown("<div style='height: 0.4rem;'></div>", unsafe_allow_html=True)
        new_theme = st.segmented_control(
            "Theme Mode",
            options=["Lovable Cream", "Dark Mode", "Clean Light"],
            default=theme_choice,
            key="ui_theme_selector",
            label_visibility="collapsed",
        )
        if new_theme and new_theme != theme_choice:
            st.rerun()

    main_tab_trans, main_tab_assign = st.tabs([
        "🔀 Transportation Optimizer",
        "🎯 Assignment Problem",
    ])

    with main_tab_trans:
            # ── Editable Network Inputs ───────────────────────────────────────
        st.markdown("<p class='rq-section'>Network Inputs</p>", unsafe_allow_html=True)

        matrix_key = f"matrix_editor_{scenario_name}"
        state_key = f"matrix_state_{scenario_name}"
        version = st.session_state.get(f"{state_key}_v", 0)
        warehouse_cols = [c for c in matrix_df.columns if c not in ("Factory", "Supply")]
        factory_names = [r for r in matrix_df["Factory"].tolist() if str(r).lower() != "demand"]

        tab_add, tab_remove = st.tabs(["Add", "Remove"])

        with tab_add:
            ac1, ac2 = st.columns(2, gap="medium")
            with ac1:
                with st.form(key=f"form_add_f_{scenario_name}", clear_on_submit=True):
                    new_factory = st.text_input("Factory name", placeholder="e.g. F4")
                    submitted_f = st.form_submit_button("Add factory", use_container_width=True)
                if submitted_f and (new_factory or "").strip():
                    name = new_factory.strip()
                    if name in matrix_df["Factory"].values:
                        st.warning(f"'{name}' already exists.")
                    else:
                        default_cost = float(matrix_df.drop(columns=["Factory", "Supply"], errors="ignore").select_dtypes("number").mean().mean() or 5.0)
                        new_row = {"Factory": name, "Supply": 0.0}
                        new_row.update({w: default_cost for w in warehouse_cols})
                        demand_idx = matrix_df.index[matrix_df["Factory"].astype(str).str.lower() == "demand"]
                        upper = matrix_df.loc[:demand_idx[0]-1] if len(demand_idx) else matrix_df.iloc[:-1]
                        lower = matrix_df.loc[demand_idx[0]:] if len(demand_idx) else matrix_df.iloc[-1:]
                        updated = pd.concat([upper, pd.DataFrame([new_row]), lower], ignore_index=True)
                        st.session_state[state_key] = updated
                        st.session_state[f"{state_key}_v"] = version + 1
                        st.rerun()
            with ac2:
                with st.form(key=f"form_add_wh_{scenario_name}", clear_on_submit=True):
                    new_warehouse = st.text_input("Warehouse name", placeholder="e.g. W5")
                    submitted_w = st.form_submit_button("Add warehouse", use_container_width=True)
                if submitted_w and (new_warehouse or "").strip():
                    name = new_warehouse.strip()
                    if name in matrix_df.columns:
                        st.warning(f"'{name}' already exists.")
                    else:
                        default_cost = float(matrix_df.drop(columns=["Factory", "Supply"], errors="ignore").select_dtypes("number").mean().mean() or 5.0)
                        updated = matrix_df.copy()
                        demand_mask = updated["Factory"].astype(str).str.lower() == "demand"
                        updated[name] = default_cost
                        updated.loc[demand_mask, name] = 0.0
                        st.session_state[state_key] = updated
                        st.session_state[f"{state_key}_v"] = version + 1
                        st.rerun()

        with tab_remove:
            rc1, rc2 = st.columns(2, gap="medium")
            with rc1:
                rm_factory = st.selectbox("Factory to remove", options=factory_names, index=None, key=f"rm_f_{scenario_name}", placeholder="Select factory...")
                if st.button("Remove factory", key=f"rm_f_btn_{scenario_name}", use_container_width=True):
                    if not rm_factory:
                        st.warning("Select a factory first.")
                    elif len(factory_names) <= 1:
                        st.warning("Cannot remove the last factory.")
                    else:
                        updated = matrix_df[matrix_df["Factory"] != rm_factory].reset_index(drop=True)
                        st.session_state[state_key] = updated
                        st.session_state[f"{state_key}_v"] = version + 1
                        st.rerun()
            with rc2:
                rm_warehouse = st.selectbox("Warehouse to remove", options=warehouse_cols, index=None, key=f"rm_wh_{scenario_name}", placeholder="Select warehouse...")
                if st.button("Remove warehouse", key=f"rm_wh_btn_{scenario_name}", use_container_width=True):
                    if not rm_warehouse:
                        st.warning("Select a warehouse first.")
                    elif len(warehouse_cols) <= 1:
                        st.warning("Cannot remove the last warehouse.")
                    else:
                        updated = matrix_df.drop(columns=[rm_warehouse])
                        st.session_state[state_key] = updated
                        st.session_state[f"{state_key}_v"] = version + 1
                        st.rerun()

        # ── Editable matrix table ─────────────────────────────────────
        _, table_center_col, _ = st.columns([0.06, 0.88, 0.06])
        with table_center_col:
            # Hide Supply value on Demand row by replacing 0 with None for display
            display_df = matrix_df.copy()
            demand_mask = display_df["Factory"].astype(str).str.lower() == "demand"
            display_df.loc[demand_mask, "Supply"] = None

            edited = st.data_editor(
                display_df,
                key=f"{matrix_key}_{version}",
                num_rows="fixed",
                hide_index=True,
                use_container_width=True,
                column_config={
                    "Factory": st.column_config.TextColumn("Factory", disabled=True),
                    "Supply": st.column_config.NumberColumn("Supply", min_value=0.0),
                },
            )
            # Restore Demand row Supply to 0 after edit (so parser ignores it)
            demand_mask_out = edited["Factory"].astype(str).str.lower() == "demand"
            edited.loc[demand_mask_out, "Supply"] = 0.0

        st.session_state[state_key] = edited
        matrix_df = edited

        should_run = auto_run or run_clicked or st.session_state.pop("force_run", False)
        results_key = f"results_state_{scenario_name}"
        result_state = st.session_state.get(results_key)

        st.markdown("<hr class='rq-divider'>", unsafe_allow_html=True)

        # ── Run / Fetch Results ───────────────────────────────────────────────
        if should_run:
            try:
                routes_df, supply, demand = _to_optimizer_inputs(matrix_df)
                if cost_multiplier != 1.0:
                    routes_df = routes_df.copy()
                    routes_df["cost"] = routes_df["cost"] * cost_multiplier

                result_df, summary = solve_transportation(
                    routes_df,
                    supply,
                    demand,
                    solver_type=selected_solver,
                    timeout_seconds=solver_timeout,
                    enable_mip=enable_mip,
                    fixed_lane_cost=fixed_lane_cost,
                )

            except (RuntimeError, ValueError) as error:
                st.error(f"Optimization failed: {error}")
                return

            briefing_text = ""
            briefing_error = ""
            try:
                briefing_text = generate_executive_briefing(
                    summary=summary,
                    scenario_name=scenario_name,
                    provider=selected_provider,
                    model=active_model,
                    api_key=custom_api_key.strip(),
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

            if "history" not in st.session_state:
                st.session_state["history"] = []
            st.session_state["history"].append({
                "scenario": scenario_name,
                "total_cost": summary["total_cost"],
                "multiplier": cost_multiplier,
                "mip_enabled": enable_mip,
            })

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

        # ── Key Metrics ───────────────────────────────────────────────────────
        st.markdown("<p class='rq-section'>Key Metrics</p>", unsafe_allow_html=True)
        metric_col_1, metric_col_2, metric_col_3 = st.columns(3, gap="medium")

        with metric_col_1:
            st.metric("Total Transportation Cost", f"${summary['total_cost']:,.2f}")
        with metric_col_2:
            if summary["factory_utilization"]:
                most_utilized_factory, utilization_value = max(
                    summary["factory_utilization"].items(), key=lambda item: item[1]
                )
            else:
                most_utilized_factory, utilization_value = "N/A", 0.0
            st.metric("Top Factory Utilization", f"{most_utilized_factory}", delta=f"{utilization_value:.1%} utilized")
        with metric_col_3:
            fully_filled = all(ratio >= 1.0 for ratio in summary["warehouse_fill_ratio"].values())
            st.metric("Demand Coverage", "100%" if fully_filled else "< 100%")

        # ── Visualizations ────────────────────────────────────────────────────
        st.markdown("<p class='rq-section' style='margin-top:1.6rem'>Visualizations</p>", unsafe_allow_html=True)
        chart_col_1, chart_col_2 = st.columns(2, gap="medium")

        if use_plotly:
            with chart_col_1:
                st.plotly_chart(plot_network_plotly(result_df, title="Interactive Network Flow"), use_container_width=True)
            with chart_col_2:
                st.plotly_chart(plot_cost_heatmap_plotly(routes_df, title="Interactive Cost Heatmap"), use_container_width=True)
        else:
            with chart_col_1:
                st.markdown("<p class='rq-table-label'>Network Flow</p>", unsafe_allow_html=True)
                figure_network, axis_network = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
                plot_network(
                    result_df,
                    title=f"{scenario_name.replace('_', ' ').title()} — Network Flow",
                    axis=axis_network,
                    dark_mode=is_dark_mode,
                )
                st.image(_figure_to_png_bytes(figure_network), use_container_width=True)
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
                st.image(_figure_to_png_bytes(figure_heatmap), use_container_width=True)
                plt.close(figure_heatmap)

        # ── AI Briefing & Exports ─────────────────────────────────────────────
        st.markdown("<p class='rq-section' style='margin-top:1.6rem'>AI Executive Briefing & Export Reports</p>", unsafe_allow_html=True)
        if result_state["briefing_text"]:
            st.write(result_state["briefing_text"])
        
            # ponytail: cache report bytes in result_state to avoid regeneration on rerun
            if "pdf_bytes" not in result_state:
                result_state["pdf_bytes"] = generate_pdf_report(summary, scenario_name, result_state["briefing_text"], result_df)
                result_state["excel_bytes"] = generate_excel_report(summary, scenario_name, result_df)

            btn_col1, btn_col2, _ = st.columns([0.25, 0.25, 0.5])
            with btn_col1:
                st.download_button(
                    "Download Executive PDF",
                    data=result_state["pdf_bytes"],
                    file_name=f"RouteIQ_{scenario_name}_report.pdf",
                    mime="application/pdf",
                )
            with btn_col2:
                st.download_button(
                    "Download Excel Dataset",
                    data=result_state["excel_bytes"],
                    file_name=f"RouteIQ_{scenario_name}_report.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )
        else:
            st.warning(f"Briefing unavailable — {result_state['briefing_error']}")

        # ── Scenario History Expander ──────────────────────────────────────────
        if "history" in st.session_state and st.session_state["history"]:
            with st.expander("Run History & Scenario Comparison"):
                history_df = pd.DataFrame(st.session_state["history"])
                st.dataframe(history_df, use_container_width=True)

    with main_tab_assign:
        st.markdown("<p class='rq-section'>Assignment Problem Matrix</p>", unsafe_allow_html=True)
        st.caption("1-to-1 matching of Agents (Workers/Machines) to Tasks (Jobs/Projects). Supply and Demand are fixed to 1.")

        asgn_obj_col, asgn_space = st.columns([0.4, 0.6])
        with asgn_obj_col:
            asgn_objective = st.radio(
                "Optimization Objective",
                options=["Minimize Cost", "Maximize Profit / Rating"],
                horizontal=True,
                key=f"asgn_obj_{scenario_name}",
            )
        is_max = "Maximize" in asgn_objective

        assign_matrix_df = _get_assignment_matrix_state(scenario_name)
        asgn_matrix_key = f"asgn_matrix_editor_{scenario_name}"
        asgn_state_key = f"assignment_matrix_state_{scenario_name}"
        asgn_version = st.session_state.get(f"{asgn_state_key}_v", 0)

        asgn_task_cols = [c for c in assign_matrix_df.columns if c != "Agent"]
        asgn_agent_names = assign_matrix_df["Agent"].tolist()

        asgn_tab_add, asgn_tab_remove = st.tabs(["Add", "Remove"])

        with asgn_tab_add:
            aac1, aac2 = st.columns(2, gap="medium")
            with aac1:
                with st.form(key=f"form_add_agent_{scenario_name}", clear_on_submit=True):
                    new_agent = st.text_input("Agent name", placeholder="e.g. Worker 4")
                    submitted_agent = st.form_submit_button("Add agent", use_container_width=True)
                if submitted_agent and (new_agent or "").strip():
                    name = new_agent.strip()
                    if name in assign_matrix_df["Agent"].values:
                        st.warning(f"'{name}' already exists.")
                    else:
                        default_c = float(assign_matrix_df.drop(columns=["Agent"], errors="ignore").select_dtypes("number").mean().mean() or 5.0)
                        new_row = {"Agent": name}
                        new_row.update({t: default_c for t in asgn_task_cols})
                        updated_asgn = pd.concat([assign_matrix_df, pd.DataFrame([new_row])], ignore_index=True)
                        st.session_state[asgn_state_key] = updated_asgn
                        st.session_state[f"{asgn_state_key}_v"] = asgn_version + 1
                        st.rerun()
            with aac2:
                with st.form(key=f"form_add_task_{scenario_name}", clear_on_submit=True):
                    new_task = st.text_input("Task name", placeholder="e.g. Task D")
                    submitted_task = st.form_submit_button("Add task", use_container_width=True)
                if submitted_task and (new_task or "").strip():
                    name = new_task.strip()
                    if name in assign_matrix_df.columns:
                        st.warning(f"'{name}' already exists.")
                    else:
                        default_c = float(assign_matrix_df.drop(columns=["Agent"], errors="ignore").select_dtypes("number").mean().mean() or 5.0)
                        updated_asgn = assign_matrix_df.copy()
                        updated_asgn[name] = default_c
                        st.session_state[asgn_state_key] = updated_asgn
                        st.session_state[f"{asgn_state_key}_v"] = asgn_version + 1
                        st.rerun()

        with asgn_tab_remove:
            arc1, arc2 = st.columns(2, gap="medium")
            with arc1:
                rm_agent = st.selectbox("Agent to remove", options=asgn_agent_names, index=None, key=f"rm_agent_{scenario_name}", placeholder="Select agent...")
                if st.button("Remove agent", key=f"rm_agent_btn_{scenario_name}", use_container_width=True):
                    if not rm_agent:
                        st.warning("Select an agent first.")
                    elif len(asgn_agent_names) <= 1:
                        st.warning("Cannot remove the last agent.")
                    else:
                        updated_asgn = assign_matrix_df[assign_matrix_df["Agent"] != rm_agent].reset_index(drop=True)
                        st.session_state[asgn_state_key] = updated_asgn
                        st.session_state[f"{asgn_state_key}_v"] = asgn_version + 1
                        st.rerun()
            with arc2:
                rm_task = st.selectbox("Task to remove", options=asgn_task_cols, index=None, key=f"rm_task_{scenario_name}", placeholder="Select task...")
                if st.button("Remove task", key=f"rm_task_btn_{scenario_name}", use_container_width=True):
                    if not rm_task:
                        st.warning("Select a task first.")
                    elif len(asgn_task_cols) <= 1:
                        st.warning("Cannot remove the last task.")
                    else:
                        updated_asgn = assign_matrix_df.drop(columns=[rm_task])
                        st.session_state[asgn_state_key] = updated_asgn
                        st.session_state[f"{asgn_state_key}_v"] = asgn_version + 1
                        st.rerun()

        # ── Editable Assignment Matrix Table ──────────────────────────
        _, asgn_table_center, _ = st.columns([0.06, 0.88, 0.06])
        with asgn_table_center:
            edited_asgn = st.data_editor(
                assign_matrix_df,
                key=f"{asgn_matrix_key}_{asgn_version}",
                num_rows="fixed",
                hide_index=True,
                use_container_width=True,
                column_config={
                    "Agent": st.column_config.TextColumn("Agent (Worker/Machine)", disabled=True),
                },
            )

        st.session_state[asgn_state_key] = edited_asgn
        assign_matrix_df = edited_asgn

        asgn_should_run = auto_run or run_clicked
        asgn_results_key = f"asgn_results_state_{scenario_name}_{'max' if is_max else 'min'}"
        asgn_result_state = st.session_state.get(asgn_results_key)

        st.markdown("<hr class='rq-divider'>", unsafe_allow_html=True)

        if asgn_should_run:
            try:
                asgn_routes_df, asgn_result_df, asgn_summary = _to_assignment_inputs(
                    assign_matrix_df, maximize=is_max, solver_type=selected_solver, timeout_seconds=solver_timeout
                )
            except (RuntimeError, ValueError) as error:
                st.error(f"Assignment optimization failed: {error}")
                asgn_result_state = None
            else:
                asgn_briefing_text = ""
                asgn_briefing_error = ""
                try:
                    asgn_briefing_text = generate_executive_briefing(
                        summary=asgn_summary,
                        scenario_name=scenario_name,
                        provider=selected_provider,
                        model=active_model,
                        api_key=custom_api_key.strip(),
                        problem_type="assignment",
                        objective="maximize" if is_max else "minimize",
                    )
                except RuntimeError as error:
                    asgn_briefing_error = str(error)

                asgn_result_state = {
                    "routes_df": asgn_routes_df,
                    "result_df": asgn_result_df,
                    "summary": asgn_summary,
                    "briefing_text": asgn_briefing_text,
                    "briefing_error": asgn_briefing_error,
                    "is_max": is_max,
                }
                st.session_state[asgn_results_key] = asgn_result_state

        if asgn_result_state is not None:
            a_routes_df = asgn_result_state["routes_df"]
            a_result_df = asgn_result_state["result_df"]
            a_summary   = asgn_result_state["summary"]
            a_is_max    = asgn_result_state["is_max"]

            # ── Key Assignment Metrics ─────────────────────────────────
            st.markdown("<p class='rq-section'>Assignment Metrics</p>", unsafe_allow_html=True)
            am1, am2, am3 = st.columns(3, gap="medium")
            with am1:
                val_label = "Total Profit / Rating" if a_is_max else "Total Assignment Cost"
                st.metric(val_label, f"${a_summary['total_cost']:,.2f}")
            with am2:
                active_count = len([p for p in a_summary.get("active_assignments", []) if p["warehouse"] != "Unassigned" and p["factory"] != "Unassigned"])
                st.metric("Assigned Pairs", f"{active_count}")
            with am3:
                unassigned_agents = [p["factory"] for p in a_summary.get("active_assignments", []) if p["warehouse"] == "Unassigned"]
                st.metric("Unassigned Agents", f"{len(unassigned_agents)}", delta=f"{', '.join(unassigned_agents)}" if unassigned_agents else "None")

            # ── Optimal Pair Allocations Table ────────────────────────
            st.markdown("<p class='rq-section' style='margin-top:1.6rem'>Optimal Matching Pairs</p>", unsafe_allow_html=True)
            active_pairs = [p for p in a_summary.get("active_assignments", []) if p["factory"] != "Unassigned" and p["warehouse"] != "Unassigned"]
            if active_pairs:
                pairs_display = pd.DataFrame([
                    {
                        "Agent": p["factory"],
                        "Assigned Task": p["warehouse"],
                        "Cost / Rating ($)": f"${p['cost']:.2f}",
                    }
                    for p in active_pairs
                ])
                st.dataframe(pairs_display, use_container_width=True)

            # ── Visualizations ────────────────────────────────────────
            st.markdown("<p class='rq-section' style='margin-top:1.6rem'>Visualizations</p>", unsafe_allow_html=True)
            achart1, achart2 = st.columns(2, gap="medium")
            if use_plotly:
                with achart1:
                    st.plotly_chart(plot_network_plotly(a_result_df, title="Assignment Flow Graph"), use_container_width=True)
                with achart2:
                    st.plotly_chart(plot_cost_heatmap_plotly(a_routes_df, title="Assignment Matrix Heatmap"), use_container_width=True)
            else:
                with achart1:
                    fig_net, ax_net = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
                    plot_network(a_result_df, title="Assignment Flow Graph", axis=ax_net, dark_mode=is_dark_mode)
                    st.image(_figure_to_png_bytes(fig_net), use_container_width=True)
                    plt.close(fig_net)
                with achart2:
                    fig_heat, ax_heat = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
                    plot_cost_heatmap(a_routes_df, title="Assignment Matrix Heatmap", axis=ax_heat, dark_mode=is_dark_mode)
                    st.image(_figure_to_png_bytes(fig_heat), use_container_width=True)
                    plt.close(fig_heat)

            # ── AI Briefing & Exports ──────────────────────────────────
            st.markdown("<p class='rq-section' style='margin-top:1.6rem'>AI Assignment Briefing & Export Reports</p>", unsafe_allow_html=True)
            if asgn_result_state["briefing_text"]:
                st.write(asgn_result_state["briefing_text"])

                if "pdf_bytes" not in asgn_result_state:
                    asgn_result_state["pdf_bytes"] = generate_pdf_report(a_summary, f"assignment_{'max' if a_is_max else 'min'}", asgn_result_state["briefing_text"], a_result_df)
                    asgn_result_state["excel_bytes"] = generate_excel_report(a_summary, f"assignment_{'max' if a_is_max else 'min'}", a_result_df)

                ab1, ab2, _ = st.columns([0.25, 0.25, 0.5])
                with ab1:
                    st.download_button("Download Executive PDF", data=asgn_result_state["pdf_bytes"], file_name=f"RouteIQ_assignment_report.pdf", mime="application/pdf", key="dl_asgn_pdf")
                with ab2:
                    st.download_button("Download Excel Dataset", data=asgn_result_state["excel_bytes"], file_name=f"RouteIQ_assignment_report.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", key="dl_asgn_excel")
            else:
                st.warning(f"Briefing unavailable — {asgn_result_state['briefing_error']}")
        else:
            st.info("Adjust assignment inputs above, then click **▶ Run optimization** in the sidebar.")

if __name__ == "__main__":
    main()
