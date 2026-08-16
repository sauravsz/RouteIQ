import pytest
import pandas as pd
from src.features import (
    calculate_carbon_emissions,
    run_monte_carlo_simulation,
    generate_pdf_report,
    generate_excel_report,
)

def test_carbon_emissions():
    routes_df = pd.DataFrame([
        {"factory": "F1", "warehouse": "W1", "flow": 10.0, "cost": 5.0}
    ])
    df_co2 = calculate_carbon_emissions(routes_df, co2_per_unit_dist=0.2)
    assert df_co2["co2_emissions_kg"].iloc[0] == 10.0 * (5.0 * 10.0) * 0.2

def test_monte_carlo_simulation():
    routes_df = pd.DataFrame([
        {"factory": "F1", "warehouse": "W1", "cost": 4},
        {"factory": "F2", "warehouse": "W1", "cost": 3},
    ])
    supply = {"F1": 20, "F2": 20}
    demand = {"W1": 15}
    mc_df = run_monte_carlo_simulation(routes_df, supply, demand, n_simulations=5)
    assert len(mc_df) == 5
    assert "total_cost" in mc_df.columns

def test_report_generation():
    result_df = pd.DataFrame([
        {"factory": "F1", "warehouse": "W1", "flow": 10.0, "cost": 5.0, "route_cost": 50.0}
    ])
    summary = {"total_cost": 50.0}
    pdf_bytes = generate_pdf_report(summary, "baseline", "Test Briefing", result_df)
    assert len(pdf_bytes) > 0

    excel_bytes = generate_excel_report(summary, "baseline", result_df)
    assert len(excel_bytes) > 0
