from typing import Dict, Optional

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import seaborn as sns


def plot_network(
    result_df: pd.DataFrame,
    title: str = "Network Flow",
    axis: Optional[plt.Axes] = None,
) -> plt.Figure:
    graph = nx.DiGraph()

    factories = result_df["factory"].unique().tolist()
    warehouses = result_df["warehouse"].unique().tolist()

    for factory in factories:
        graph.add_node(factory, bipartite=0)
    for warehouse in warehouses:
        graph.add_node(warehouse, bipartite=1)

    for _, row in result_df.iterrows():
        if row["flow"] > 0:
            graph.add_edge(row["factory"], row["warehouse"], weight=row["flow"])

    # Keep a simple left-right layout so supply and demand nodes are easy to scan.
    positions: Dict[str, tuple] = {}
    for index, factory in enumerate(factories):
        positions[factory] = (0, index)
    for index, warehouse in enumerate(warehouses):
        positions[warehouse] = (1, index)

    if axis is None:
        figure, axis = plt.subplots(figsize=(8, 5), constrained_layout=True)
    else:
        figure = axis.figure

    edges = graph.edges(data=True)
    widths = [max(0.5, data["weight"] / 5.0) for _, _, data in edges]

    nx.draw(
        graph,
        positions,
        ax=axis,
        with_labels=True,
        node_size=800,
        node_color=["lightblue" if node in factories else "lightgreen" for node in graph.nodes()],
        width=widths,
        arrows=False,
    )

    axis.set_title(title)
    axis.axis("off")

    return figure


def plot_cost_heatmap(
    routes_df: pd.DataFrame,
    title: str = "Cost Heatmap",
    axis: Optional[plt.Axes] = None,
) -> plt.Figure:
    pivot = routes_df.pivot(index="factory", columns="warehouse", values="cost")

    if axis is None:
        figure, axis = plt.subplots(figsize=(6, 4), constrained_layout=True)
    else:
        figure = axis.figure

    sns.heatmap(pivot, annot=True, fmt=".1f", cmap="Reds", ax=axis)
    axis.set_title(title)
    axis.set_xlabel("Warehouse")
    axis.set_ylabel("Factory")

    return figure