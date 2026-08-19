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

def _apply_ui_theme(theme_mode: str) -> None:
    is_dark = theme_mode.lower() == "dark"

    bg_page = "linear-gradient(160deg, #0b0f19 0%, #111827 100%)" if is_dark else "linear-gradient(160deg, #f7f9fc 0%, #eef2f7 100%)"
    text_base = "#f8fafc" if is_dark else "#1f2937"
    text_muted = "#94a3b8" if is_dark else "#64748b"
    border = "rgba(148,163,184,0.18)" if is_dark else "rgba(148,163,184,0.28)"
    panel_bg = "rgba(15,23,42,0.65)" if is_dark else "rgba(255,255,255,0.86)"

    css = f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Sora:wght@300;400;500;600;700&family=DM+Serif+Display:ital@0;1&display=swap');

    :root {{
        --rq-text: {text_base};
        --rq-muted: {text_muted};
        --rq-border: {border};
        --rq-panel: {panel_bg};
    }}

    .stApp {{
        background: {bg_page} !important;
        color: var(--rq-text) !important;
        font-family: 'Sora', sans-serif;
    }}

    [data-testid="stSidebar"] {{
        background: #090d16 !important;
        border-right: 1px solid var(--rq-border) !important;
    }}

    .rq-title {{
        font-family: 'DM Serif Display', serif;
        font-size: 2.2rem;
        font-weight: 400;
        letter-spacing: -0.03em;
        color: #ffffff;
        line-height: 1.2;
        margin-bottom: 0.2rem;
    }}

    .rq-subtitle {{
        font-family: 'Sora', sans-serif;
        font-size: 0.86rem;
        color: var(--rq-muted);
        margin-bottom: 1.6rem;
        letter-spacing: 0.01em;
    }}

    .rq-section {{
        font-family: 'Sora', sans-serif !important;
        font-size: 0.74rem !important;
        font-weight: 700 !important;
        letter-spacing: 0.1em !important;
        text-transform: uppercase !important;
        color: #818cf8 !important;
        margin-bottom: 0.8rem;
    }}

    .rq-divider {{
        border: none;
        border-top: 1px solid var(--rq-border);
        margin: 1.2rem 0;
    }}

    .rq-side-title {{
        font-size: 0.76rem;
        font-weight: 700;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        color: #818cf8;
        margin-top: 0;
        margin-bottom: 0.42rem;
        line-height: 1.2;
    }}

    [data-testid="stMetric"] {{
        background: var(--rq-panel) !important;
        border: 1px solid var(--rq-border) !important;
        border-radius: 12px !important;
        padding: 1rem 1.2rem !important;
        box-shadow: 0 4px 12px rgba(0,0,0,0.25);
    }}

    [data-testid="stMetricValue"] > div {{
        color: #ffffff !important;
        font-weight: 600 !important;
    }}

    .rq-table-label {{
        font-family: 'Sora', sans-serif;
        font-size: 0.82rem;
        font-weight: 600;
        color: var(--rq-muted);
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

    # BUG 4: validate no NaN costs
    rows: list[dict] = []
    for _, frow in factory_df.iterrows():
        factory = frow["Factory"]
        for w in warehouse_cols:
            cost_val = frow[w]
            if cost_val is None or (isinstance(cost_val, float) and np.isnan(cost_val)):
                raise RuntimeError(f"Missing cost for route {factory} → {w}. Fill in all cost cells.")
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

    _apply_ui_theme("dark")
    is_dark_mode = True

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

    # ── Page Header ───────────────────────────────────────────────────────
    st.markdown("<div class='rq-title'>RouteIQ</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='rq-subtitle'>Multi-scenario transportation optimizer with AI executive briefing</div>",
        unsafe_allow_html=True,
    )

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

if __name__ == "__main__":
    main()
