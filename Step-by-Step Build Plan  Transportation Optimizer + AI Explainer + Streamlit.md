# Step-by-Step Build Plan: RouteIQ + AI Explainer + Streamlit

## Overview

This guide gives a strict, chronological checklist to build a transportation problem optimizer with PuLP, scenario analysis, an AI executive explainer, and a Streamlit app.
It assumes basic comfort with Python, VS Code, and Git, and focuses on removing all design decisions so each step can simply be followed.

PuLP’s official documentation includes a classic transportation case study that can be used as a sanity-check if needed.[^1]

***

## Phase 0 – One-Time Setup

### 0.1 – Create the project folder

1. Choose or create a parent folder where you keep projects.
2. Inside it, create a new folder named, for example, `RouteIQ`.
3. Open this folder in VS Code (File → Open Folder → select `RouteIQ`).

### 0.2 – Create and activate a virtual environment

1. Open the VS Code integrated terminal (View → Terminal).
2. Run these commands (use one of the Python commands that works on your system):

```bash
# In the project root
python -m venv .venv

# Activate (macOS / Linux)
source .venv/bin/activate

# OR on Windows PowerShell
.venv\Scripts\Activate.ps1
```

3. Confirm the venv is active: your terminal prompt should show `(.venv)` at the beginning.

### 0.3 – Install required packages

1. In the same terminal, run:

```bash
pip install pulp pandas numpy matplotlib seaborn networkx streamlit python-dotenv openai
```

2. Freeze dependencies to a file:

```bash
pip freeze > requirements.txt
```

### 0.4 – Initialize a Git repository (for later GitHub push)

1. In the project root, run:

```bash
git init
git add .
git commit -m "Initial project setup"
```

(If Git is not configured, run `git config --global user.name "Your Name"` and `git config --global user.email "you@example.com"` once.)

***

## Phase 1 – Understand and Sketch the Transportation Problem

### 1.1 – Fix the example network you will use

Use exactly this small base network for your mental model and first implementation:

- Factories (supply nodes): F1, F2, F3
- Warehouses (demand nodes): W1, W2, W3, W4
- Each factory can ship to every warehouse.

### 1.2 – Draw the network on paper

1. On blank paper, draw three circles on the left labeled F1, F2, F3.
2. Draw four circles on the right labeled W1, W2, W3, W4.
3. Draw an arrow from each factory to each warehouse (12 arrows total).
4. Next to each factory, leave space to write its capacity (supply).
5. Next to each warehouse, leave space to write its demand.
6. Along a few arrows, write fake example costs (e.g., 4, 7, 2) just to visualize.

### 1.3 – Fix the mathematical formulation in your own words

On the same sheet (or another sheet), write down:

1. Decision variable: "For each factory i and warehouse j, define \(x_{ij}\) as the number of units shipped from factory i to warehouse j."
2. Objective: "Minimize total cost: sum over all routes of (cost per unit on that route × units shipped on that route)."
3. Supply constraint: "For each factory i, the sum of \(x_{ij}\) over all warehouses j is less than or equal to that factory’s capacity."
4. Demand constraint: "For each warehouse j, the sum of \(x_{ij}\) over all factories i is greater than or equal to that warehouse’s demand."
5. Non-negativity: "All \(x_{ij}\) are greater than or equal to 0."

You do not need to implement anything in code yet; just make sure you are comfortable reading this on the paper.

***

## Phase 2 – Build the Core Optimizer in Pure Python

You will now build a reusable optimizer module that:

- Reads a scenario from a CSV.
- Builds and solves a PuLP model.
- Returns clean Pandas DataFrames with flows and summary metrics.

### 2.1 – Create basic project structure

In the project root, create this structure:

```text
RouteIQ/
├── data/
│   └── scenarios.csv
├── src/
│   ├── optimizer.py
│   ├── scenarios.py
│   └── __init__.py
├── app.py              # will later be the Streamlit entry point
├── requirements.txt
└── README.md
```

Create the folders and empty files exactly as above.

### 2.2 – Design a long-form scenario CSV

Use a long format CSV (one row per route per scenario). This gives you multiple scenarios in a single file.

1. Open `data/scenarios.csv` in VS Code.
2. Paste a header row and some example baseline data like this (you can adjust numbers later):

```csv
scenario,factory,warehouse,supply,demand,cost
baseline,F1,W1,35,10,4
baseline,F1,W2,35,15,6
baseline,F1,W3,35,8,9
baseline,F1,W4,35,12,5
baseline,F2,W1,40,10,5
baseline,F2,W2,40,15,4
baseline,F2,W3,40,8,7
baseline,F2,W4,40,12,6
baseline,F3,W1,25,10,6
baseline,F3,W2,25,15,3
baseline,F3,W3,25,8,4
baseline,F3,W4,25,12,8
```

3. Interpretation rules (very important):
   - `supply` is defined per factory and scenario; the same number is repeated across that factory’s rows.
   - `demand` is defined per warehouse and scenario; the same number is repeated across that warehouse’s rows.
   - `cost` is the cost per unit for that route (factory→warehouse).

Later, you will add rows for other scenarios (e.g., `disruption`, `cost_surge`) in the same file.

### 2.3 – Implement scenario loading helper (`src/scenarios.py`)

Open `src/scenarios.py` and paste this minimal loader:

```python
import pandas as pd
from typing import Tuple, Dict


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
```

Do not modify this yet; you will connect it to the optimizer next.

### 2.4 – Implement the PuLP optimizer (`src/optimizer.py`)

Open `src/optimizer.py` and implement the core model.

1. Paste this starter code:

```python
from typing import Dict, Tuple

import pandas as pd
from pulp import LpProblem, LpMinimize, LpVariable, lpSum, LpStatus, PULP_CBC_CMD


def solve_transportation(
    routes_df: pd.DataFrame,
    supply: Dict[str, float],
    demand: Dict[str, float],
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    factories = list(supply.keys())
    warehouses = list(demand.keys())

    # Sanity check for feasibility (total supply vs. total demand)
    total_supply = sum(supply.values())
    total_demand = sum(demand.values())
    if total_supply < total_demand:
        raise ValueError(
            f"Infeasible scenario: total supply {total_supply} < total demand {total_demand}."
        )

    # Build a cost lookup: (factory, warehouse) -> cost
    cost = {
        (row["factory"], row["warehouse"]): row["cost"]
        for _, row in routes_df.iterrows()
    }

    model = LpProblem("TransportationProblem", LpMinimize)

    # Decision variables x[(i, j)] >= 0
    x = LpVariable.dicts(
        "ship",
        ((i, j) for i in factories for j in warehouses),
        lowBound=0,
    )

    # Objective: minimize total cost
    model += lpSum(cost[(i, j)] * x[(i, j)] for i in factories for j in warehouses)

    # Supply constraints: sum_j x_ij <= S_i
    for i in factories:
        model += lpSum(x[(i, j)] for j in warehouses) <= supply[i], f"Supply_{i}"

    # Demand constraints: sum_i x_ij >= D_j
    for j in warehouses:
        model += lpSum(x[(i, j)] for i in factories) >= demand[j], f"Demand_{j}"

    # Solve
    model.solve(PULP_CBC_CMD(msg=False))

    status = LpStatus[model.status]
    if status != "Optimal":
        raise RuntimeError(f"Solver did not find an optimal solution. Status: {status}")

    # Build a tidy DataFrame of flows
    rows = []
    for i in factories:
        for j in warehouses:
            flow = x[(i, j)].value()
            rows.append({
                "factory": i,
                "warehouse": j,
                "flow": flow,
                "cost": cost[(i, j)],
                "route_cost": flow * cost[(i, j)],
            })

    result_df = pd.DataFrame(rows)

    summary = {
        "total_cost": float(result_df["route_cost"].sum()),
    }

    # Add per-factory utilization and per-warehouse fulfillment
    factory_usage = result_df.groupby("factory")["flow"].sum().to_dict()
    warehouse_received = result_df.groupby("warehouse")["flow"].sum().to_dict()

    summary["factory_utilization"] = {
        f: factory_usage.get(f, 0.0) / supply[f] if supply[f] > 0 else 0.0
        for f in factories
    }
    summary["warehouse_fill_ratio"] = {
        w: warehouse_received.get(w, 0.0) / demand[w] if demand[w] > 0 else 0.0
        for w in warehouses
    }

    return result_df, summary
```

2. Save the file.

### 2.5 – Wire scenario loader and optimizer together

Create a quick script in the project root to test everything.

1. Open `app.py` and temporarily make it a plain Python script (Streamlit will come later).
2. Paste this code to test the baseline scenario:

```python
from src.scenarios import load_scenario
from src.optimizer import solve_transportation


def main():
    routes_df, supply, demand = load_scenario("data/scenarios.csv", "baseline")

    print("Supply:", supply)
    print("Demand:", demand)

    result_df, summary = solve_transportation(routes_df, supply, demand)

    print("\nOptimal flows:")
    print(result_df)

    print("\nSummary:")
    print(summary)


if __name__ == "__main__":
    main()
```

3. Run it from the terminal:

```bash
python app.py
```

4. Confirm that:
   - You see printed supply and demand dictionaries.
   - `result_df` shows flows for all 12 routes.
   - `summary` shows a `total_cost` and utilization ratios.

If you see an infeasibility error, increase total supply or reduce demands in `data/scenarios.csv` and rerun.

5. Commit this working version:

```bash
git add .
git commit -m "Add core optimizer and baseline scenario"
```

***

## Phase 3 – Visualize the Network

You will add two visualizations:

- A bipartite network graph (factories left, warehouses right) with edge thickness proportional to shipped volume.
- A cost heatmap to visually compare route costs.

### 3.1 – Implement visualization functions (`src/visualizations.py`)

1. Create a new file `src/visualizations.py`.
2. Paste this code skeleton:

```python
from typing import Dict

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns


def plot_network(result_df: pd.DataFrame, title: str = "Network Flow") -> None:
    G = nx.DiGraph()

    factories = result_df["factory"].unique().tolist()
    warehouses = result_df["warehouse"].unique().tolist()

    # Add nodes with bipartite attribute
    for f in factories:
        G.add_node(f, bipartite=0)
    for w in warehouses:
        G.add_node(w, bipartite=1)

    # Add edges only for positive flows
    for _, row in result_df.iterrows():
        if row["flow"] > 0:
            G.add_edge(row["factory"], row["warehouse"], weight=row["flow"])

    # Manual layout: factories on left, warehouses on right
    pos: Dict[str, tuple] = {}
    for idx, f in enumerate(factories):
        pos[f] = (0, idx)
    for idx, w in enumerate(warehouses):
        pos[w] = (1, idx)

    plt.figure(figsize=(8, 5))

    # Edge widths scaled by flow
    edges = G.edges(data=True)
    widths = [max(0.5, d["weight"] / 5.0) for (_, _, d) in edges]

    nx.draw(
        G,
        pos,
        with_labels=True,
        node_size=800,
        node_color=["lightblue" if n in factories else "lightgreen" for n in G.nodes()],
        width=widths,
        arrows=False,
    )

    plt.title(title)
    plt.axis("off")
    plt.tight_layout()


def plot_cost_heatmap(routes_df: pd.DataFrame, title: str = "Cost Heatmap") -> None:
    pivot = routes_df.pivot(index="factory", columns="warehouse", values="cost")

    plt.figure(figsize=(6, 4))
    sns.heatmap(pivot, annot=True, fmt=".1f", cmap="Reds")
    plt.title(title)
    plt.xlabel("Warehouse")
    plt.ylabel("Factory")
    plt.tight_layout()
```

3. Save the file.

### 3.2 – Test the plots in `app.py`

1. Modify `app.py` to display the plots using Matplotlib’s interactive window:

```python
from src.scenarios import load_scenario
from src.optimizer import solve_transportation
from src.visualizations import plot_network, plot_cost_heatmap
import matplotlib.pyplot as plt


def main():
    routes_df, supply, demand = load_scenario("data/scenarios.csv", "baseline")
    result_df, summary = solve_transportation(routes_df, supply, demand)

    print(summary)

    plot_network(result_df, title="Baseline Network Flow")
    plot_cost_heatmap(routes_df, title="Baseline Cost Heatmap")

    plt.show()


if __name__ == "__main__":
    main()
```

2. Run again:

```bash
python app.py
```

3. Confirm:
   - A network diagram appears where only edges with positive flow are shown.
   - A heatmap appears with factories as rows, warehouses as columns, and cost values.

4. Commit your changes:

```bash
git add .
git commit -m "Add network and heatmap visualizations"
```

***

## Phase 4 – Add the AI Executive Explainer

You will now:

- Summarize the optimization results into a compact Python dictionary.
- Call an OpenAI-compatible API (OpenAI, Gemini via compatible endpoint, etc.).
- Get a 2–3 paragraph executive briefing.

### 4.1 – Prepare environment variable for your API key

1. In the project root, create a file named `.env` (this will not be committed if you ignore it later).
2. Inside `.env`, add a line like:

```text
OPENAI_API_KEY=your_real_key_here
```

3. In the terminal, install `python-dotenv` (already installed earlier) and ensure `openai` is installed (it was installed in Phase 0).

### 4.2 – Implement AI explainer helper (`src/ai_explainer.py`)

1. Create `src/ai_explainer.py`.
2. Paste this code skeleton and adapt the model name / base URL later as needed:

```python
import os
from typing import Dict

from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()


def build_summary_prompt(summary: Dict, scenario_name: str) -> str:
    total_cost = summary["total_cost"]
    factory_util = summary["factory_utilization"]
    warehouse_fill = summary["warehouse_fill_ratio"]

    lines = []
    lines.append(f"Scenario: {scenario_name}")
    lines.append(f"Total transportation cost: {total_cost:.2f}")
    lines.append("Factory utilization (fraction of capacity used):")
    for f, u in factory_util.items():
        lines.append(f"  - {f}: {u:.2%}")
    lines.append("Warehouse demand fill ratios:")
    for w, r in warehouse_fill.items():
        lines.append(f"  - {w}: {r:.2%}")

    bullet_summary = "\n".join(lines)

    system_prompt = (
        "You are a senior supply chain consultant. "
        "Write a clear, non-technical 3-paragraph executive briefing for a COO. "
        "Explain what this transportation optimization solution is doing, "
        "highlight which factories are heavily utilized or underused, "
        "which warehouses are tight on supply, and what the total cost implies. "
        "Avoid formulas, keep it business-focused, and reference the scenario name."
    )

    user_prompt = (
        "Here is a structured summary of the optimized transportation plan:\n\n"
        f"{bullet_summary}\n\n"
        "Write the executive briefing now."
    )

    return system_prompt, user_prompt


def generate_executive_briefing(summary: Dict, scenario_name: str) -> str:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set in environment or .env file.")

    client = OpenAI(api_key=api_key)

    system_prompt, user_prompt = build_summary_prompt(summary, scenario_name)

    response = client.chat.completions.create(
        model="gpt-4o-mini",  # or other OpenAI-compatible model
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.3,
    )

    return response.choices.message.content
```

3. Save the file.

### 4.3 – Test the AI explainer via `app.py`

1. Modify `app.py` to call the explainer after solving:

```python
from src.scenarios import load_scenario
from src.optimizer import solve_transportation
from src.visualizations import plot_network, plot_cost_heatmap
from src.ai_explainer import generate_executive_briefing
import matplotlib.pyplot as plt


def main():
    scenario_name = "baseline"
    routes_df, supply, demand = load_scenario("data/scenarios.csv", scenario_name)
    result_df, summary = solve_transportation(routes_df, supply, demand)

    print(summary)

    plot_network(result_df, title=f"{scenario_name.capitalize()} Network Flow")
    plot_cost_heatmap(routes_df, title=f"{scenario_name.capitalize()} Cost Heatmap")

    briefing = generate_executive_briefing(summary, scenario_name)
    print("\nExecutive briefing:\n")
    print(briefing)

    plt.show()


if __name__ == "__main__":
    main()
```

2. Run:

```bash
python app.py
```

3. Confirm:
   - The optimizer runs.
   - Plots appear.
   - A multi-paragraph executive briefing is printed in the terminal.

4. Commit your changes:

```bash
git add .
git commit -m "Add AI executive explainer"
```

***

## Phase 5 – Add Multi-Scenario Analysis

You will define three scenarios inside the same `data/scenarios.csv` file and add logic to run and compare them.

### 5.1 – Extend the CSV to three scenarios

Open `data/scenarios.csv` and:

1. Keep the existing `baseline` rows.
2. Duplicate all `baseline` rows below, then change the `scenario` label to `disruption` for each duplicated row.
3. Duplicate the `baseline` rows again and change the `scenario` label to `cost_surge`.

You now have 3 × 12 = 36 rows.

4. Edit the numbers to create meaningful scenarios:

- **Scenario A – baseline**: keep as-is.
- **Scenario B – disruption**:
  - Reduce one factory’s capacity to 20 percent of baseline.
  - Example: if F2’s supply is 40 in baseline, set all `disruption` rows with `factory = F2` and `supply = 8`.
- **Scenario C – cost_surge**:
  - Multiply the cost on two chosen routes by 3.
  - For example, for `cost_surge` rows where `(factory, warehouse)` is `(F1, W2)` or `(F3, W4)`, triple the `cost` value.

Ensure that total supply is still at least total demand for all scenarios (if not, you will get the infeasible exception and should adjust supplies or demands).

### 5.2 – Add a scenario runner in `src/scenarios.py`

Append this helper to `src/scenarios.py`:

```python
from src.optimizer import solve_transportation


def run_scenario(path: str, scenario_name: str):
    routes_df, supply, demand = load_scenario(path, scenario_name)
    result_df, summary = solve_transportation(routes_df, supply, demand)
    return routes_df, result_df, summary
```

### 5.3 – Add comparative scenario logic in `app.py`

1. Replace the contents of `app.py` with:

```python
from src.scenarios import run_scenario
from src.visualizations import plot_network, plot_cost_heatmap
from src.ai_explainer import generate_executive_briefing
import matplotlib.pyplot as plt


SCENARIOS = ["baseline", "disruption", "cost_surge"]


def main():
    for scenario_name in SCENARIOS:
        print("\n=== Scenario:", scenario_name, "===")
        routes_df, result_df, summary = run_scenario("data/scenarios.csv", scenario_name)

        print("Total cost:", summary["total_cost"])

        briefing = generate_executive_briefing(summary, scenario_name)
        print("\nExecutive briefing:")
        print(briefing)

        # Optional: show plots per scenario (comment out plt.show for batch runs)
        plot_network(result_df, title=f"{scenario_name.capitalize()} Network Flow")
        plot_cost_heatmap(routes_df, title=f"{scenario_name.capitalize()} Cost Heatmap")
        plt.show()


if __name__ == "__main__":
    main()
```

2. Run:

```bash
python app.py
```

3. Confirm:
   - The script loops over `baseline`, `disruption`, and `cost_surge`.
   - You see total costs for each scenario.
   - You get separate AI briefings for each scenario.
   - You can visually see how flows and costs change.

4. Commit this milestone:

```bash
git add .
git commit -m "Add multi-scenario analysis with AI briefings"
```

***

## Phase 6 – Convert to an Interactive Streamlit App

Now that the backend logic is working, you will turn `app.py` into a Streamlit app with:

- Sidebar controls for scenario selection.
- Optional sliders to tweak supplies, demands, or cost multipliers.
- Embedded plots and the AI-generated executive briefing.

### 6.1 – Basic Streamlit layout in `app.py`

1. Replace the contents of `app.py` with the following Streamlit version:

```python
import streamlit as st
import matplotlib.pyplot as plt

from src.scenarios import load_scenario
from src.optimizer import solve_transportation
from src.visualizations import plot_network, plot_cost_heatmap
from src.ai_explainer import generate_executive_briefing


SCENARIOS = ["baseline", "disruption", "cost_surge"]


def main():
    st.set_page_config(page_title="RouteIQ", layout="wide")

    st.title("Multi-Scenario Transportation Network Optimizer")

    # Sidebar controls
    st.sidebar.header("Scenario Controls")
    scenario_name = st.sidebar.selectbox("Scenario", SCENARIOS, index=0)

    # Optional: global cost multiplier for experimentation
    cost_multiplier = st.sidebar.slider("Cost multiplier", 0.5, 3.0, 1.0, 0.1)

    # Load and optionally adjust scenario
    routes_df, supply, demand = load_scenario("data/scenarios.csv", scenario_name)

    if cost_multiplier != 1.0:
        routes_df = routes_df.copy()
        routes_df["cost"] = routes_df["cost"] * cost_multiplier

    # Solve optimization
    result_df, summary = solve_transportation(routes_df, supply, demand)

    # Summary metrics
    st.subheader("Key Metrics")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total transportation cost", f"{summary['total_cost']:.2f}")
    with col2:
        max_factory = max(summary["factory_utilization"].items(), key=lambda x: x[^1])
        st.metric("Most utilized factory", f"{max_factory} ({max_factory[^1]:.1%})")

    # Plots
    col3, col4 = st.columns(2)

    with col3:
        st.markdown("### Network Flow")
        fig1, ax1 = plt.subplots(figsize=(5, 4))
        # Use the same plotting code but direct Matplotlib to this figure
        plt.sca(ax1)
        plot_network(result_df, title="")
        st.pyplot(fig1)

    with col4:
        st.markdown("### Cost Heatmap")
        fig2, ax2 = plt.subplots(figsize=(5, 4))
        plt.sca(ax2)
        plot_cost_heatmap(routes_df, title="")
        st.pyplot(fig2)

    # AI Executive briefing
    st.markdown("### AI Executive Briefing")
    briefing = generate_executive_briefing(summary, scenario_name)
    st.write(briefing)


if __name__ == "__main__":
    main()
```

2. Run the Streamlit app:

```bash
streamlit run app.py
```

3. In your browser, verify:
   - You can switch scenarios from the sidebar.
   - The metrics, plots, and briefing update accordingly.
   - The cost multiplier slider changes total cost and flows.

4. Commit this version:

```bash
git add .
git commit -m "Convert to Streamlit app with AI briefing"
```

***

## Phase 7 – Polish and Push to GitHub

### 7.1 – Add a clear README

1. Open `README.md` and replace its contents with sections like:

```markdown
# Transportation Network Optimizer (MBA Project)

An interactive Streamlit app that solves a classic transportation problem using PuLP, compares multiple disruption scenarios, and auto-generates executive briefings using an LLM.

## Problem Statement

- Multiple factories (supply nodes) shipping to multiple warehouses/customers (demand nodes).
- Each route has a different transportation cost.
- Objective: decide how many units to ship from each factory to each warehouse to satisfy all demand at minimum total cost.

## Features

- Linear programming model implemented with PuLP.
- Scenario analysis: baseline, disruption, cost surge.
- Network flow visualization and cost heatmap.
- Executive summaries generated automatically via an LLM API.
- Streamlit front-end for interactive what-if analysis.

## How to Run Locally

1. Clone the repo and create a virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Create a `.env` file with your `OPENAI_API_KEY`.
4. Run the app:

```bash
streamlit run app.py
```

## Screenshots

(Add 1–3 screenshots here after running the app.)

## Author

- Your Name – MBA in Operations & Marketing (Project context: transportation optimization and scenario analysis)
```

2. Save the README.

### 7.2 – Capture screenshots

1. Run the Streamlit app.
2. Take at least two screenshots:
   - One showing the baseline scenario.
   - One showing the disruption or cost_surge scenario.
3. Save them into a folder such as `assets/` and reference them in the README (relative paths).

### 7.3 – Push to GitHub

1. On GitHub, create a new empty repository (without README) named `RouteIQ` (or similar).
2. Follow the GitHub instructions for an existing repository, typically:

```bash
git remote add origin https://github.com/your-username/RouteIQ.git
git branch -M main
git push -u origin main
```

3. Confirm on GitHub that:
   - Code structure is visible.
   - README renders correctly.
   - (Optional) Add a repo description and tags like `operations-research`, `supply-chain`, `streamlit`.

***

## How to Use This Plan Day-by-Day

- **Day 1–2:** Do Phase 1 thoroughly; draw and understand the math, then complete Phase 0 if not already done.
- **Days 3–5:** Implement Phases 2 and 3 (optimizer + visualization). Ensure the baseline scenario is rock-solid.
- **Days 6–8:** Add the AI explainer (Phase 4) and verify outputs are coherent.
- **Days 9–11:** Implement scenarios and comparison (Phase 5). Iterate on data to create interesting differences.
- **Days 12–14:** Convert to Streamlit and refine layout and UX (Phase 6).
- **Days 15–16:** Polish README, screenshots, and GitHub repo (Phase 7), and refine resume bullet points.

At any point, if something breaks, roll back to the last Git commit that worked and move forward again in small, tested increments.

---

## References

1. [A Transportation Problem — PuLP 3.3.0 documentation](https://coin-or.github.io/pulp/CaseStudies/a_transportation_problem.html) - The example file for this problem is found in the examples directory BeerDistributionProblem.py. Fir...

