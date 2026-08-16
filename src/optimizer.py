import shutil
from typing import Dict, Tuple, Optional
import pandas as pd
from pulp import (
    LpMinimize,
    LpProblem,
    LpStatus,
    LpVariable,
    PULP_CBC_CMD,
    COIN_CMD,
    GLPK_CMD,
    lpSum,
    LpBinary,
)

def get_available_solvers() -> Dict[str, bool]:
    cbc_found = bool(shutil.which("cbc") or shutil.which("/opt/homebrew/bin/cbc"))
    glpk_found = bool(shutil.which("glpsol"))
    return {
        "cbc": cbc_found,
        "glpk": glpk_found,
        "pulp_default": True,
    }

def _resolve_solver(solver_type: str = "cbc", timeout_seconds: int = 10):
    cbc_path = shutil.which("cbc") or "/opt/homebrew/bin/cbc"
    if solver_type == "glpk" and shutil.which("glpsol"):
        return GLPK_CMD(msg=False, timeLimit=timeout_seconds)
    if shutil.which("cbc") or shutil.which(cbc_path):
        return COIN_CMD(path=cbc_path, timeLimit=timeout_seconds, msg=False)
    return PULP_CBC_CMD(timeLimit=timeout_seconds, msg=False)

def solve_transportation(
    routes_df: pd.DataFrame,
    supply: Dict[str, float],
    demand: Dict[str, float],
    solver_type: str = "cbc",
    timeout_seconds: int = 10,
    enable_mip: bool = False,
    fixed_lane_cost: float = 0.0,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    if not supply or not demand:
        raise ValueError("Supply and Demand dictionaries cannot be empty.")
    if any(s < 0 for s in supply.values()):
        raise ValueError("Supply capacities must be non-negative.")
    if any(d < 0 for d in demand.values()):
        raise ValueError("Demand values must be non-negative.")
    if (routes_df["cost"] < 0).any():
        raise ValueError("Route shipping costs must be non-negative.")

    real_factories = list(supply.keys())
    real_warehouses = list(demand.keys())

    supply_copy = dict(supply)
    demand_copy = dict(demand)
    
    cost = {
        (row["factory"], row["warehouse"]): float(row["cost"])
        for _, row in routes_df.iterrows()
    }
    
    capacity = {}
    if "capacity" in routes_df.columns:
        capacity = {
            (row["factory"], row["warehouse"]): float(row["capacity"])
            for _, row in routes_df.iterrows()
        }

    total_supply = sum(supply_copy.values())
    total_demand = sum(demand_copy.values())

    factories = list(real_factories)
    warehouses = list(real_warehouses)

    # Standard auto-balancing via 0-cost dummy nodes
    if total_supply > total_demand:
        dummy_w = "Dummy_Warehouse"
        warehouses.append(dummy_w)
        demand_copy[dummy_w] = total_supply - total_demand
        for f in real_factories:
            cost[(f, dummy_w)] = 0.0
            if capacity:
                capacity[(f, dummy_w)] = total_supply
    elif total_demand > total_supply:
        dummy_f = "Dummy_Factory"
        factories.append(dummy_f)
        supply_copy[dummy_f] = total_demand - total_supply
        for w in real_warehouses:
            cost[(dummy_f, w)] = 0.0
            if capacity:
                capacity[(dummy_f, w)] = total_demand

    model = LpProblem("TransportationProblem", LpMinimize)

    # Decision variables x[(i, j)] >= 0
    x = LpVariable.dicts(
        "ship",
        ((i, j) for i in factories for j in warehouses),
        lowBound=0,
    )

    # Optional Binary variables y[(i, j)] in {0, 1} for MIP fixed charge
    y = {}
    if enable_mip or fixed_lane_cost > 0:
        y = LpVariable.dicts(
            "open_lane",
            ((i, j) for i in factories for j in warehouses),
            cat=LpBinary,
        )

    # Objective Z
    if y and fixed_lane_cost > 0:
        model += lpSum(
            cost[(i, j)] * x[(i, j)] + fixed_lane_cost * y[(i, j)]
            for i in factories for j in warehouses
        )
    else:
        model += lpSum(cost[(i, j)] * x[(i, j)] for i in factories for j in warehouses)

    # Constraints
    for i in factories:
        model += lpSum(x[(i, j)] for j in warehouses) == supply_copy[i], f"Supply_{i}"

    for j in warehouses:
        model += lpSum(x[(i, j)] for i in factories) == demand_copy[j], f"Demand_{j}"

    # Lane capacity & MIP binary constraints
    for i in factories:
        for j in warehouses:
            max_cap = capacity.get((i, j), max(total_supply, total_demand))
            if y:
                model += x[(i, j)] <= max_cap * y[(i, j)], f"Cap_MIP_{i}_{j}"
            elif (i, j) in capacity:
                model += x[(i, j)] <= max_cap, f"Cap_{i}_{j}"

    solver_cmd = _resolve_solver(solver_type=solver_type, timeout_seconds=timeout_seconds)
    model.solve(solver_cmd)

    status = LpStatus[model.status]
    if status != "Optimal":
        raise RuntimeError(f"Solver did not find an optimal solution. Status: {status}")

    rows = []
    for i in factories:
        for j in warehouses:
            flow = float(x[(i, j)].value() or 0.0)
            is_open = float(y[(i, j)].value() or 0.0) if y else (1.0 if flow > 0 else 0.0)
            rows.append(
                {
                    "factory": i,
                    "warehouse": j,
                    "flow": flow,
                    "cost": cost[(i, j)],
                    "lane_open": is_open,
                    "route_cost": flow * cost[(i, j)] + (fixed_lane_cost * is_open if y else 0.0),
                }
            )

    result_df = pd.DataFrame(rows)

    summary: Dict[str, float] = {
        "total_cost": float(result_df["route_cost"].sum()),
        "open_lanes_count": int(result_df[result_df["lane_open"] > 0]["lane_open"].count()),
    }

    real_results = result_df[
        (result_df["factory"].isin(real_factories)) & (result_df["warehouse"].isin(real_warehouses))
    ]
    factory_usage = real_results.groupby("factory")["flow"].sum().to_dict()
    warehouse_received = real_results.groupby("warehouse")["flow"].sum().to_dict()

    summary["factory_utilization"] = {
        f: factory_usage.get(f, 0.0) / supply[f] if supply[f] > 0 else 0.0
        for f in real_factories
    }
    summary["warehouse_fill_ratio"] = {
        w: warehouse_received.get(w, 0.0) / demand[w] if demand[w] > 0 else 0.0
        for w in real_warehouses
    }

    return result_df, summary
