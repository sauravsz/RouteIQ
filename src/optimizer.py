from typing import Dict, Tuple

import pandas as pd
from pulp import LpMinimize, LpProblem, LpStatus, LpVariable, PULP_CBC_CMD, lpSum


def solve_transportation(
    routes_df: pd.DataFrame,
    supply: Dict[str, float],
    demand: Dict[str, float],
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    factories = list(supply.keys())
    warehouses = list(demand.keys())

    # Sanity check for feasibility (total supply vs. total demand)
    total_supply = sum(supply.values())
    total_demand = sum(demand.values())
    if total_supply < total_demand:
        raise ValueError(
            f"Infeasible scenario: total supply {total_supply} < total demand {total_demand}."
        )

    # Build a cost lookup: (factory, warehouse) -> cost
    cost = {
        (row["factory"], row["warehouse"]): row["cost"]
        for _, row in routes_df.iterrows()
    }

    model = LpProblem("TransportationProblem", LpMinimize)

    # Decision variables x[(i, j)] >= 0
    x = LpVariable.dicts(
        "ship",
        ((i, j) for i in factories for j in warehouses),
        lowBound=0,
    )

    # Objective: minimize total cost
    model += lpSum(cost[(i, j)] * x[(i, j)] for i in factories for j in warehouses)

    # Supply constraints: sum_j x_ij <= S_i
    for i in factories:
        model += lpSum(x[(i, j)] for j in warehouses) <= supply[i], f"Supply_{i}"

    # Demand constraints: sum_i x_ij >= D_j
    for j in warehouses:
        model += lpSum(x[(i, j)] for i in factories) >= demand[j], f"Demand_{j}"

    # Solve
    model.solve(PULP_CBC_CMD(msg=False))

    status = LpStatus[model.status]
    if status != "Optimal":
        raise RuntimeError(f"Solver did not find an optimal solution. Status: {status}")

    # Build a tidy DataFrame of flows
    rows = []
    for i in factories:
        for j in warehouses:
            flow = x[(i, j)].value()
            rows.append(
                {
                    "factory": i,
                    "warehouse": j,
                    "flow": flow,
                    "cost": cost[(i, j)],
                    "route_cost": flow * cost[(i, j)],
                }
            )

    result_df = pd.DataFrame(rows)

    summary: Dict[str, float] = {
        "total_cost": float(result_df["route_cost"].sum()),
    }

    # Add per-factory utilization and per-warehouse fulfillment
    factory_usage = result_df.groupby("factory")["flow"].sum().to_dict()
    warehouse_received = result_df.groupby("warehouse")["flow"].sum().to_dict()

    summary["factory_utilization"] = {
        f: factory_usage.get(f, 0.0) / supply[f] if supply[f] > 0 else 0.0
        for f in factories
    }
    summary["warehouse_fill_ratio"] = {
        w: warehouse_received.get(w, 0.0) / demand[w] if demand[w] > 0 else 0.0
        for w in warehouses
    }

    return result_df, summary
