from src.scenarios import run_scenario
from src.ai_explainer import generate_executive_briefing
from src.visualizations import plot_cost_heatmap, plot_network
import matplotlib.pyplot as plt


SCENARIOS = ["baseline", "disruption", "cost_surge"]


def main() -> None:
    for scenario_name in SCENARIOS:
        print(f"\n=== Scenario: {scenario_name} ===")

        routes_df, result_df, summary = run_scenario("data/scenarios.csv", scenario_name)

        print("Total cost:", summary["total_cost"])

        print("\nOptimal flows:")
        print(result_df)

        print("\nSummary:")
        print(summary)

        try:
            briefing = generate_executive_briefing(summary, scenario_name)
            print("\nExecutive briefing:\n")
            print(briefing)
        except RuntimeError as error:
            # Keep local runs usable when API credentials are not configured yet.
            print(f"\nExecutive briefing unavailable: {error}")

        title_prefix = scenario_name.replace("_", " ").title()
        plot_network(result_df, title=f"{title_prefix} Network Flow")
        plot_cost_heatmap(routes_df, title=f"{title_prefix} Cost Heatmap")
        plt.show()


if __name__ == "__main__":
    main()
