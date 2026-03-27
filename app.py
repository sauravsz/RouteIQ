import matplotlib.pyplot as plt
import streamlit as st

from src.ai_explainer import generate_executive_briefing
from src.optimizer import solve_transportation
from src.scenarios import load_scenario
from src.visualizations import plot_cost_heatmap, plot_network


SCENARIOS = ["baseline", "disruption", "cost_surge"]


def main() -> None:
    st.set_page_config(page_title="RouteIQ", layout="wide")

    st.title("RouteIQ: Multi-Scenario Transportation Optimizer")
    st.caption("Interactive what-if analysis with optimization, visualizations, and AI briefing")

    st.sidebar.header("Scenario Controls")
    scenario_name = st.sidebar.selectbox("Scenario", SCENARIOS, index=0)
    cost_multiplier = st.sidebar.slider("Cost multiplier", 0.5, 3.0, 1.0, 0.1)

    routes_df, supply, demand = load_scenario("data/scenarios.csv", scenario_name)
    if cost_multiplier != 1.0:
        routes_df = routes_df.copy()
        routes_df["cost"] = routes_df["cost"] * cost_multiplier

    result_df, summary = solve_transportation(routes_df, supply, demand)

    st.subheader("Key Metrics")
    metric_col_1, metric_col_2, metric_col_3 = st.columns(3)
    with metric_col_1:
        st.metric("Total transportation cost", f"{summary['total_cost']:.2f}")
    with metric_col_2:
        most_utilized_factory, utilization_value = max(
            summary["factory_utilization"].items(), key=lambda item: item[1]
        )
        st.metric("Most utilized factory", f"{most_utilized_factory} ({utilization_value:.1%})")
    with metric_col_3:
        fully_filled = all(ratio >= 1.0 for ratio in summary["warehouse_fill_ratio"].values())
        st.metric("Demand coverage", "100%" if fully_filled else "<100%")

    chart_col_1, chart_col_2 = st.columns(2)

    with chart_col_1:
        st.markdown("### Network Flow")
        figure_network, axis_network = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
        plot_network(result_df, title=f"{scenario_name.replace('_', ' ').title()} Network Flow", axis=axis_network)
        st.pyplot(figure_network)
        plt.close(figure_network)

    with chart_col_2:
        st.markdown("### Cost Heatmap")
        figure_heatmap, axis_heatmap = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
        plot_cost_heatmap(routes_df, title=f"{scenario_name.replace('_', ' ').title()} Cost Heatmap", axis=axis_heatmap)
        st.pyplot(figure_heatmap)
        plt.close(figure_heatmap)

    st.markdown("### AI Executive Briefing")
    try:
        briefing = generate_executive_briefing(summary, scenario_name)
        st.write(briefing)
    except RuntimeError as error:
        st.warning(f"Executive briefing unavailable: {error}")

    with st.expander("Show optimization outputs"):
        st.markdown("**Factory supply**")
        st.json(supply)
        st.markdown("**Warehouse demand**")
        st.json(demand)
        st.markdown("**Optimized route flows**")
        st.dataframe(result_df, use_container_width=True)


if __name__ == "__main__":
    main()
