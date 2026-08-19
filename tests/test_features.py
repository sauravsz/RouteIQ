import pytest
import pandas as pd
from src.features import (
    generate_pdf_report,
    generate_excel_report,
)

def test_report_generation():
    result_df = pd.DataFrame([
        {"factory": "F1", "warehouse": "W1", "flow": 10.0, "cost": 5.0, "route_cost": 50.0}
    ])
    summary = {"total_cost": 50.0}
    pdf_bytes = generate_pdf_report(summary, "baseline", "Test Briefing", result_df)
    assert len(pdf_bytes) > 0

    excel_bytes = generate_excel_report(summary, "baseline", result_df)
    assert len(excel_bytes) > 0
