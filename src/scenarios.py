from typing import Dict, Tuple, Union
import pandas as pd
from pathlib import Path
from src.optimizer import solve_transportation

def load_scenario_from_df(df: pd.DataFrame, scenario_name: str) -> Tuple[pd.DataFrame, Dict[str, float], Dict[str, float]]:
    if "scenario" in df.columns and scenario_name.lower() != "custom":
        filtered_df = df[df["scenario"] == scenario_name].copy()
    else:
        filtered_df = df.copy()

    if filtered_df.empty:
        raise ValueError(f"Scenario '{scenario_name}' not found or dataset is empty.")

    required_cols = {"factory", "warehouse", "supply", "demand", "cost"}
    missing = required_cols - set(filtered_df.columns)
    if missing:
        raise ValueError(f"Dataset missing required columns: {missing}")

    supply = filtered_df.groupby("factory")["supply"].max().to_dict()
    demand = filtered_df.groupby("warehouse")["demand"].max().to_dict()
    return filtered_df, supply, demand

def load_scenario(path_or_df: Union[str, Path, pd.DataFrame], scenario_name: str) -> Tuple[pd.DataFrame, Dict[str, float], Dict[str, float]]:
    if isinstance(path_or_df, pd.DataFrame):
        return load_scenario_from_df(path_or_df, scenario_name)
    df = pd.read_csv(path_or_df)
    return load_scenario_from_df(df, scenario_name)

def run_scenario(path_or_df: Union[str, Path, pd.DataFrame], scenario_name: str):
    routes_df, supply, demand = load_scenario(path_or_df, scenario_name)
    result_df, summary = solve_transportation(routes_df, supply, demand)
    return routes_df, result_df, summary
