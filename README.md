# RouteIQ

RouteIQ is a transportation network optimization project that solves a classic supply-to-demand allocation problem using linear programming.

The current implementation includes:
- Scenario loading from CSV data.
- Cost-minimizing transportation optimization with PuLP.
- Network flow and route-cost visualizations.
- AI-generated executive briefings from optimization summaries.

This project is being built in phases. Phases 1 to 4 are complete.

## What Problem This Solves

Given:
- Multiple factories with limited supply.
- Multiple warehouses with required demand.
- Different shipping costs for each factory to warehouse route.

RouteIQ computes shipment quantities that satisfy demand at minimum total transportation cost.

## Current Features (Phases 1 to 4)

- Scenario-based input loader from CSV (long format).
- Linear optimization model with:
  - Supply constraints.
  - Demand constraints.
  - Non-negativity constraints.
- Summary metrics:
  - Total transportation cost.
  - Factory utilization ratios.
  - Warehouse fill ratios.
- Visual outputs:
  - Directed network flow graph.
  - Cost heatmap.
- AI explainer:
  - Structured prompt from solution summary metrics.
  - OpenAI-compatible narrative generation for executive updates.

## Project Structure

```text
RouteIQ/
├── README.md
├── PHASE_OVERVIEW.md
├── app.py
├── data/
│   └── scenarios.csv
└── src/
  ├── ai_explainer.py
  ├── __init__.py
  ├── optimizer.py
  ├── scenarios.py
  └── visualizations.py
```

## Tech Stack

- Python 3
- PuLP (linear programming)
- pandas (data handling)
- matplotlib, seaborn (visualization)
- networkx (network graph)
- openai, python-dotenv (AI briefing)

## Quick Start

### 1. Clone and enter the project

```bash
git clone https://github.com/sauravsz/RouteIQ.git
cd RouteIQ
```

### 2. Create and activate a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install pulp pandas matplotlib seaborn networkx openai python-dotenv
```

### 3.1 Configure AI Provider (Phase 4)

1. Copy `.env.example` to `.env`.
2. Set `AI_PROVIDER` to one of: `openai`, `groq`, `cerebras`, `google`.
3. Fill the matching provider API key.

Required by provider:
- `openai`: `OPENAI_API_KEY`
- `groq`: `GROQ_API_KEY`
- `cerebras`: `CEREBRAS_API_KEY`
- `google`: `GOOGLE_API_KEY`

Optional per provider:
- `*_MODEL` to override the model name.
- `*_BASE_URL` to override endpoint base URL.

### 3.2 Enable Secret-Scan Pre-Commit Hook

Run once per clone:

```bash
git config core.hooksPath .githooks
```

This enables the repo-managed hook at `.githooks/pre-commit`, which blocks commits that include key-like tokens or `.env` files.

### 3.3 Switch Provider Quickly

Use the helper script to change provider without editing `.env` manually:

```bash
./scripts/switch_provider.sh <provider> [model]
```

Examples:

```bash
./scripts/switch_provider.sh groq
./scripts/switch_provider.sh google gemini-2.5-flash-lite
```

### 4. Run the app

```bash
python3 app.py
```

What you should see:
- Console output with supply, demand, route flows, and summary metrics.
- Plot windows for:
  - Baseline Network Flow.
  - Baseline Cost Heatmap.

## Input Data Format

The scenario file is at data/scenarios.csv and uses one row per route:

```csv
scenario,factory,warehouse,supply,demand,cost
baseline,F1,W1,35,10,4
```

Rules:
- supply is repeated for each factory across its routes.
- demand is repeated for each warehouse across incoming routes.
- cost is route-specific.

## How It Works

1. app.py loads a selected scenario from data/scenarios.csv.
2. src/scenarios.py extracts:
   - Route table.
   - Factory supply dictionary.
   - Warehouse demand dictionary.
3. src/optimizer.py builds and solves the LP model.
4. src/visualizations.py renders:
   - Shipment network graph.
   - Cost heatmap.
5. src/ai_explainer.py converts summary metrics into an executive narrative.
  - Provider selection is controlled by `AI_PROVIDER` in `.env`.

## Example Output Summary

For the baseline sample data, an example run returns:
- total_cost around 177.0
- full demand coverage for all warehouses (fill ratio 1.0)

Values may change if you edit scenario inputs.

## Roadmap

- Phase 5: Multi-scenario comparison.
- Phase 6: Streamlit interactive dashboard.
- Phase 7: Final polish and publishing assets.

Progress is tracked in PHASE_OVERVIEW.md.

## Troubleshooting

- If python is not found on macOS, use python3.
- If plots do not appear in remote/headless environments, run with a GUI session or set a non-interactive backend for testing.
- If solver reports infeasible, ensure total supply is at least total demand for the selected scenario.

## Author

Saurav
