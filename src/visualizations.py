from typing import Dict

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import seaborn as sns


def plot_network(result_df: pd.DataFrame, title: str = "Network Flow") -> None:
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

    plt.figure(figsize=(8, 5))

    edges = graph.edges(data=True)
    widths = [max(0.5, data["weight"] / 5.0) for _, _, data in edges]

    nx.draw(
        graph,
        positions,
        with_labels=True,
        node_size=800,
        node_color=["lightblue" if node in factories else "lightgreen" for node in graph.nodes()],
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