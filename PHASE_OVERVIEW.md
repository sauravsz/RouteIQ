# RouteIQ Phase Overview

Last updated: 2026-03-28

## Phase 1 - Problem Framing
- Base transportation network fixed to 3 factories (F1, F2, F3) and 4 warehouses (W1-W4).
- Objective/constraints framing established and reflected in the implemented optimizer logic.

## Phase 2 - Core Optimizer
- Project structure set up with `data/`, `src/`, and `app.py`.
- Scenario loading implemented in `src/scenarios.py`.
- PuLP transportation solver implemented in `src/optimizer.py`.
- Baseline scenario loaded from `data/scenarios.csv` and solved successfully end to end.

## Phase 3 - Visualizations
- Added `src/visualizations.py` with:
  - Network flow plot (positive-flow edges).
  - Cost heatmap (factory x warehouse).
- Updated `app.py` to generate both charts after optimization.
- Validation run succeeded with plotting code enabled.

## Phase 4 - AI Executive Explainer
- Added `src/ai_explainer.py` with:
  - Structured summary prompt builder for optimization outputs.
  - OpenAI-compatible client call for executive narrative generation.
  - Multi-provider env configuration support (`openai`, `groq`, `cerebras`, `google`).
- Updated `app.py` to request and print an executive briefing after solving.
- Added runtime fallback messaging when API key/config is missing.

## Phase 5 - Multi-Scenario Analysis
- Expanded `data/scenarios.csv` to include:
  - `baseline`
  - `disruption` (reduced F3 supply)
  - `cost_surge` (route-level cost spikes)
- Added `run_scenario` helper in `src/scenarios.py`.
- Updated `app.py` to loop across all scenarios, print total cost and summaries, generate per-scenario AI briefings, and render per-scenario charts.

## Phase 6 - Streamlit Interactive App
- Converted `app.py` from CLI flow to Streamlit UI.
- Added sidebar controls for scenario selection and cost multiplier what-if analysis.
- Embedded network and heatmap charts directly in the app view.
- Added in-app AI executive briefing panel with runtime warning fallback.

## Phase 7 - Final Polish and Publishing Assets
- Polished `README.md` for publish-ready project presentation.
- Added `assets/` folder scaffold with screenshot guidance.
- Updated run/setup instructions for Streamlit-first workflow.