from src.optimizer import solve_transportation
from src.scenarios import load_scenario


def main() -> None:
    routes_df, supply, demand = load_scenario("data/scenarios.csv", "baseline")

    print("Supply:", supply)
    print("Demand:", demand)

    result_df, summary = solve_transportation(routes_df, supply, demand)

    print("\nOptimal flows:")
    print(result_df)

    print("\nSummary:")
    print(summary)


if __name__ == "__main__":
    main()
