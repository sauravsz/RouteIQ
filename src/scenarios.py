from typing import Dict, Tuple

import pandas as pd


def load_scenario(path: str, scenario_name: str) -> Tuple[pd.DataFrame, Dict[str, float], Dict[str, float]]:
    df = pd.read_csv(path)

    df = df[df["scenario"] == scenario_name].copy()
    if df.empty:
        raise ValueError(f"Scenario '{scenario_name}' not found in {path}.")

    # Extract unique supplies per factory
    supply = df.groupby("factory")["supply"].max().to_dict()

    # Extract unique demands per warehouse
    demand = df.groupby("warehouse")["demand"].max().to_dict()

    # Cost matrix stays at route level
    return df, supply, demand
