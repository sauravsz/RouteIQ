import pandas as pd
from io import BytesIO
from typing import Dict, Tuple
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
def generate_pdf_report(
    summary: Dict[str, float],
    scenario_name: str,
    briefing_text: str,
    result_df: pd.DataFrame,
) -> bytes:
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, rightMargin=36, leftMargin=36, topMargin=36, bottomMargin=36)
    styles = getSampleStyleSheet()
    story = []

    title_style = ParagraphStyle('TitleStyle', parent=styles['Heading1'], fontSize=18, leading=22, textColor=colors.HexColor('#0f172a'))
    story.append(Paragraph(f"RouteIQ Executive Report — {scenario_name.title()}", title_style))
    story.append(Spacer(1, 12))

    total_cost_val = summary.get("total_cost", 0.0)
    metric_text = f"<b>Total Cost:</b> ${total_cost_val:,.2f}"
    story.append(Paragraph(metric_text, styles['Normal']))
    story.append(Spacer(1, 12))

    story.append(Paragraph("<b>Executive Briefing:</b>", styles['Heading2']))
    clean_briefing = briefing_text.replace('\n', '<br/>') if briefing_text else "No briefing generated."
    story.append(Paragraph(clean_briefing, styles['Normal']))
    story.append(Spacer(1, 16))

    story.append(Paragraph("<b>Optimized Route Allocations:</b>", styles['Heading2']))
    table_data = [["Factory", "Warehouse", "Flow", "Route Cost ($)"]]
    for _, row in result_df.iterrows():
        if row["flow"] > 0:
            table_data.append([str(row["factory"]), str(row["warehouse"]), f"{row['flow']:.1f}", f"${row['route_cost']:.2f}"])

    t = Table(table_data)
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#6366f1')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 6),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#cbd5e1')),
    ]))
    story.append(t)

    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()

def generate_excel_report(
    summary: Dict[str, float],
    scenario_name: str,
    result_df: pd.DataFrame,
) -> bytes:
    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
        summary_df = pd.DataFrame([
            {"Metric": "Scenario", "Value": scenario_name},
            {"Metric": "Total Cost ($)", "Value": summary.get("total_cost", 0.0)},
        ])
        summary_df.to_excel(writer, sheet_name="Executive Summary", index=False)
        result_df.to_excel(writer, sheet_name="Route Flows", index=False)
    buffer.seek(0)
    return buffer.getvalue()
