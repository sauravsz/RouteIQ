import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
import yaml
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
from src.optimizer import solve_transportation, get_available_solvers
from src.scenarios import load_scenario
from src.visualizations import (
    plot_cost_heatmap,
    plot_network,
    plot_cost_heatmap_plotly,
    plot_network_plotly,
)
from src.features import (
    calculate_carbon_emissions,
    run_monte_carlo_simulation,
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
        auto_run = st.toggle("Auto-run on changes", value=False)
        use_plotly = st.toggle("Interactive Plotly charts", value=True)
        run_clicked = st.button("▶  Run optimization", type="primary", width="stretch")

        st.markdown("<div class='rq-side-gap-md'></div>", unsafe_allow_html=True)
        st.markdown("<div class='rq-side-title'>Data Source & Scenario</div>", unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader("Upload Custom CSV", type=["csv"], help="Upload custom scenario dataset")

        scenario_name = st.selectbox(
            "Scenario",
            SCENARIOS if uploaded_file is None else ["custom"],
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

    supply_df, demand_df, cost_matrix_df = _get_editor_state(scenario_name, base_routes_df)

    # ── Page Header ───────────────────────────────────────────────────────
    st.markdown("<div class='rq-title'>RouteIQ</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='rq-subtitle'>Multi-scenario transportation optimizer — ESG Carbon analytics, Monte Carlo simulation, AI briefing</div>",
        unsafe_allow_html=True,
    )

    # ── Editable Network Inputs ───────────────────────────────────────────
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

    # ── Run / Fetch Results ───────────────────────────────────────────────
    if should_run:
        try:
            routes_df, supply, demand = _to_optimizer_inputs(supply_df, demand_df, cost_matrix_df)
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

            result_df = calculate_carbon_emissions(result_df)
            summary["total_co2_kg"] = float(result_df["co2_emissions_kg"].sum())
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
            "total_co2_kg": summary.get("total_co2_kg", 0.0),
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
    st.markdown("<p class='rq-section'>Key Metrics & ESG Impact</p>", unsafe_allow_html=True)
    metric_col_1, metric_col_2, metric_col_3, metric_col_4 = st.columns(4, gap="medium")

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
    with metric_col_4:
        total_co2 = summary.get("total_co2_kg", 0.0)
        st.metric("Est. CO2 Emissions", f"{total_co2:,.1f} kg")

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
        
        btn_col1, btn_col2, _ = st.columns([0.25, 0.25, 0.5])
        with btn_col1:
            pdf_bytes = generate_pdf_report(summary, scenario_name, result_state["briefing_text"], result_df)
            st.download_button(
                "📄 Download Executive PDF",
                data=pdf_bytes,
                file_name=f"RouteIQ_{scenario_name}_report.pdf",
                mime="application/pdf",
            )
        with btn_col2:
            excel_bytes = generate_excel_report(summary, scenario_name, result_df)
            st.download_button(
                "📊 Download Excel Dataset",
                data=excel_bytes,
                file_name=f"RouteIQ_{scenario_name}_report.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
    else:
        st.warning(f"Briefing unavailable — {result_state['briefing_error']}")

    # ── Monte Carlo & Advanced Analytics Expander ─────────────────────────
    with st.expander("Stochastic Monte Carlo Risk Analysis"):
        st.markdown("Simulate random demand fluctuations to test network resilience.")
        mc_runs = st.slider("Simulation Runs", 10, 100, 30, 10)
        demand_std = st.slider("Demand Std Dev (%)", 0.05, 0.30, 0.15, 0.05)
        if st.button("Run Monte Carlo Risk Simulation"):
            with st.spinner("Running stochastic LP simulations..."):
                mc_df = run_monte_carlo_simulation(routes_df, supply, demand, n_simulations=mc_runs, demand_std_dev_pct=demand_std)
                st.dataframe(mc_df, use_container_width=True)
                st.success(f"Completed {mc_runs} simulations. Avg Cost: ${mc_df['total_cost'].mean():,.2f}")

    # ── Scenario History Expander ──────────────────────────────────────────
    if "history" in st.session_state and st.session_state["history"]:
        with st.expander("Run History & Scenario Comparison"):
            history_df = pd.DataFrame(st.session_state["history"])
            st.dataframe(history_df, use_container_width=True)

if __name__ == "__main__":
    main()
