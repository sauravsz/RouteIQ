# RouteIQ: Transportation Optimizer + AI Explainer

RouteIQ is an interactive Streamlit app for transportation-network optimization.
It solves a classic supply-to-demand allocation problem with linear programming,
compares multiple scenarios, and generates an executive AI briefing.

## Problem Statement

Given:
- Multiple factories with fixed supply.
- Multiple warehouses with required demand.
- Route-level transportation costs.

Goal:
- Ship units from factories to warehouses to satisfy demand at minimum total cost.

## Features

- PuLP optimization model with supply and demand constraints.
- Scenario analysis in one dataset: `baseline`, `disruption`, `cost_surge`.
- Visualization layer:
  - Network flow chart.
  - Cost heatmap.
- AI executive briefing generated from optimization outputs.
- Streamlit interface:
  - Scenario selector.
  - Cost multiplier slider for what-if analysis.
  - AI provider and model selector for briefing generation.
  - Live metrics, charts, and briefing in one dashboard.

## Project Structure

```text
RouteIQ/
├── README.md
├── PHASE_OVERVIEW.md
├── app.py
├── .env.example
├── assets/
│   └── README.md
├── data/
│   └── scenarios.csv
├── scripts/
│   └── switch_provider.sh
└── src/
    ├── ai_explainer.py
    ├── optimizer.py
    ├── scenarios.py
    └── visualizations.py
```

## Tech Stack

- Python 3
- Streamlit
- PuLP
- pandas
- matplotlib / seaborn / networkx
- openai + python-dotenv

## Run Locally

### 1. Setup

```bash
git clone https://github.com/sauravsz/RouteIQ.git
cd RouteIQ
python3 -m venv .venv
source .venv/bin/activate
pip install pulp pandas matplotlib seaborn networkx streamlit openai python-dotenv
```

### 2. Configure AI Provider

```bash
cp .env.example .env
```

In `.env`:
- Set `AI_PROVIDER` to one of: `openai`, `groq`, `cerebras`, `google`.
- Fill the matching API key variable.

Quick provider switch command:

```bash
./scripts/switch_provider.sh google
```

### 3. Run App

```bash
streamlit run app.py
```

Inside the app sidebar:
- Choose scenario and cost multiplier.
- Choose AI provider and AI model used for the executive briefing.
- Optionally enter a custom model name.

## Screenshots

### Baseline Scenario
![Baseline](assets/screenshot-baseline.png)

### Disruption Scenario
![Disruption](assets/screenshot-disruption.png)

### Network Builder
![Network Builder](assets/screenshot-network-builder.png)

## Notes

- Secret safety is enforced by `.githooks/pre-commit` when enabled with:

```bash
git config core.hooksPath .githooks
```

- If the solver reports infeasible, verify total supply is at least total demand for the selected scenario.

## Status

Phases 1 to 7 are complete. Progress details are tracked in `PHASE_OVERVIEW.md`.

## Author

Saurav
