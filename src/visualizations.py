from typing import Dict, Optional

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import seaborn as sns


def plot_network(
    result_df: pd.DataFrame,
    title: str = "Network Flow",
    axis: Optional[plt.Axes] = None,
    dark_mode: bool = False,
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

    text_color = "#e5e7eb" if dark_mode else "#0f172a"
    edge_color = "#94a3b8" if dark_mode else "#64748b"
    factory_color = "#60a5fa" if dark_mode else "#bfdbfe"
    warehouse_color = "#34d399" if dark_mode else "#bbf7d0"
    figure.patch.set_facecolor("#111827" if dark_mode else "white")
    axis.set_facecolor("#111827" if dark_mode else "white")

    edges = graph.edges(data=True)
    widths = [max(0.5, data["weight"] / 5.0) for _, _, data in edges]

    nx.draw(
        graph,
        positions,
        ax=axis,
        with_labels=True,
        node_size=800,
        node_color=[factory_color if node in factories else warehouse_color for node in graph.nodes()],
        edge_color=edge_color,
        font_color=text_color,
        width=widths,
        arrows=False,
    )

    axis.set_title(title, color=text_color)
    axis.axis("off")

    return figure


def plot_cost_heatmap(
    routes_df: pd.DataFrame,
    title: str = "Cost Heatmap",
    axis: Optional[plt.Axes] = None,
    dark_mode: bool = False,
) -> plt.Figure:
    pivot = routes_df.pivot(index="factory", columns="warehouse", values="cost")

    if axis is None:
        figure, axis = plt.subplots(figsize=(6, 4), constrained_layout=True)
    else:
        figure = axis.figure

    text_color = "#e5e7eb" if dark_mode else "#0f172a"
    background_color = "#111827" if dark_mode else "white"
    figure.patch.set_facecolor(background_color)
    axis.set_facecolor(background_color)

    sns.heatmap(
        pivot,
        annot=True,
        fmt=".1f",
        cmap="mako" if dark_mode else "Reds",
        ax=axis,
        annot_kws={"color": text_color if dark_mode else "#111827"},
    )
    axis.set_title(title, color=text_color)
    axis.set_xlabel("Warehouse", color=text_color)
    axis.set_ylabel("Factory", color=text_color)
    axis.tick_params(axis="x", colors=text_color)
    axis.tick_params(axis="y", colors=text_color)

    return figure