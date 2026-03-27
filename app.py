from src.optimizer import solve_transportation
from src.scenarios import load_scenario
from src.ai_explainer import generate_executive_briefing
from src.visualizations import plot_cost_heatmap, plot_network
import matplotlib.pyplot as plt


def main() -> None:
    routes_df, supply, demand = load_scenario("data/scenarios.csv", "baseline")

    print("Supply:", supply)
    print("Demand:", demand)

    result_df, summary = solve_transportation(routes_df, supply, demand)

    print("\nOptimal flows:")
    print(result_df)

    print("\nSummary:")
    print(summary)

    try:
        briefing = generate_executive_briefing(summary, "baseline")
        print("\nExecutive briefing:\n")
        print(briefing)
    except RuntimeError as error:
        # Keep local runs usable when API credentials are not configured yet.
        print(f"\nExecutive briefing unavailable: {error}")

    plot_network(result_df, title="Baseline Network Flow")
    plot_cost_heatmap(routes_df, title="Baseline Cost Heatmap")
    plt.show()


if __name__ == "__main__":
    main()
