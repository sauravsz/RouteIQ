import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.scenarios import load_scenario
from src.optimizer import solve_transportation, get_available_solvers

# Stub Streamlit to import app functions
import types
st_stub = types.ModuleType("streamlit")
st_stub.session_state = {}
st_stub.cache_data = lambda f: f
class _Col:
    def __init__(self, *a, **k): pass
st_stub.column_config = types.SimpleNamespace(TextColumn=_Col, NumberColumn=_Col)
sys.modules["streamlit"] = st_stub

import importlib.util
spec = importlib.util.spec_from_file_location("app", str(Path(__file__).resolve().parent.parent / "app.py"))
app = importlib.util.module_from_spec(spec)
spec.loader.exec_module(app)


def test_build_matrix_from_routes():
    routes_df, supply, demand = load_scenario(
        Path(__file__).resolve().parent.parent / "data" / "scenarios.csv", "baseline"
    )
    matrix = app._build_matrix_from_routes(routes_df)
    
    assert "Factory" in matrix.columns
    assert "Supply" in matrix.columns
    assert app.DEMAND_ROW in matrix["Factory"].values
    assert len(matrix) == len(supply) + 1


def test_to_optimizer_inputs_round_trip():
    routes_df, supply, demand = load_scenario(
        Path(__file__).resolve().parent.parent / "data" / "scenarios.csv", "baseline"
    )
    matrix = app._build_matrix_from_routes(routes_df)
    r_out, s_out, d_out = app._to_optimizer_inputs(matrix)

    assert s_out == {k: float(v) for k, v in supply.items()}
    assert d_out == {k: float(v) for k, v in demand.items()}
    assert len(r_out) == len(s_out) * len(d_out)


def test_nan_cost_validation():
    routes_df, _, _ = load_scenario(
        Path(__file__).resolve().parent.parent / "data" / "scenarios.csv", "baseline"
    )
    matrix = app._build_matrix_from_routes(routes_df)
    matrix.loc[0, "W1"] = np.nan

    with pytest.raises(RuntimeError, match="Missing cost for route"):
        app._to_optimizer_inputs(matrix)


def test_missing_demand_row():
    routes_df, _, _ = load_scenario(
        Path(__file__).resolve().parent.parent / "data" / "scenarios.csv", "baseline"
    )
    matrix = app._build_matrix_from_routes(routes_df)
    matrix_no_demand = matrix[matrix["Factory"] != app.DEMAND_ROW]

    with pytest.raises(RuntimeError, match="Matrix is missing the Demand row"):
        app._to_optimizer_inputs(matrix_no_demand)


def test_factory_deduplication():
    routes_df, _, _ = load_scenario(
        Path(__file__).resolve().parent.parent / "data" / "scenarios.csv", "baseline"
    )
    matrix = app._build_matrix_from_routes(routes_df)
    dup_matrix = pd.concat([matrix.iloc[:-1], matrix.iloc[:-1], matrix.iloc[-1:]], ignore_index=True)
    
    r_out, s_out, d_out = app._to_optimizer_inputs(dup_matrix)
    assert len(s_out) == len(routes_df["factory"].unique())
