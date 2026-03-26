# RouteIQ Phase Overview

Last updated: 2026-03-26

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