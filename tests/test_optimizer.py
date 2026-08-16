import pytest
import pandas as pd
from src.optimizer import solve_transportation, get_available_solvers

def test_balanced_transportation():
    routes_df = pd.DataFrame([
        {"factory": "F1", "warehouse": "W1", "cost": 4},
        {"factory": "F1", "warehouse": "W2", "cost": 6},
        {"factory": "F2", "warehouse": "W1", "cost": 5},
        {"factory": "F2", "warehouse": "W2", "cost": 3},
    ])
    supply = {"F1": 20, "F2": 30}
    demand = {"W1": 25, "W2": 25}
    res, summary = solve_transportation(routes_df, supply, demand)
    assert summary["total_cost"] == 180.0
    assert summary["factory_utilization"]["F1"] == 1.0
    assert summary["warehouse_fill_ratio"]["W1"] == 1.0

def test_unbalanced_excess_supply():
    routes_df = pd.DataFrame([
        {"factory": "F1", "warehouse": "W1", "cost": 4},
        {"factory": "F1", "warehouse": "W2", "cost": 6},
        {"factory": "F2", "warehouse": "W1", "cost": 5},
        {"factory": "F2", "warehouse": "W2", "cost": 3},
    ])
    supply = {"F1": 30, "F2": 30}
    demand = {"W1": 20, "W2": 20}
    res, summary = solve_transportation(routes_df, supply, demand)
    assert summary["total_cost"] == 140.0

def test_negative_cost_validation():
    routes_df = pd.DataFrame([{"factory": "F1", "warehouse": "W1", "cost": -5}])
    with pytest.raises(ValueError, match="Route shipping costs must be non-negative"):
        solve_transportation(routes_df, {"F1": 10}, {"W1": 10})

def test_mip_fixed_lane_cost():
    routes_df = pd.DataFrame([
        {"factory": "F1", "warehouse": "W1", "cost": 2},
        {"factory": "F1", "warehouse": "W2", "cost": 2},
    ])
    supply = {"F1": 20}
    demand = {"W1": 10, "W2": 10}
    res, summary = solve_transportation(routes_df, supply, demand, enable_mip=True, fixed_lane_cost=50.0)
    assert summary["total_cost"] == (20 * 2) + (2 * 50.0)
    assert summary["open_lanes_count"] == 2

def test_available_solvers():
    solvers = get_available_solvers()
    assert "pulp_default" in solvers

def test_empty_input_validation():
    routes_df = pd.DataFrame()
    with pytest.raises(ValueError, match="cannot be empty"):
        solve_transportation(routes_df, {}, {})
