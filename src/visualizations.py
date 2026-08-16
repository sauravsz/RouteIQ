from typing import Dict, Optional
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px

def plot_network_plotly(result_df: pd.DataFrame, title: str = "Network Flow") -> go.Figure:
    factories = result_df["factory"].unique().tolist()
    warehouses = result_df["warehouse"].unique().tolist()

    node_x = []
    node_y = []
    node_text = []
    node_color = []

    for idx, f in enumerate(factories):
        node_x.append(0)
        node_y.append(idx)
        node_text.append(f"Factory: {f}")
        node_color.append("#60a5fa")

    for idx, w in enumerate(warehouses):
        node_x.append(1)
        node_y.append(idx)
        node_text.append(f"Warehouse: {w}")
        node_color.append("#34d399")

    edge_x = []
    edge_y = []
    edge_hover = []

    fig = go.Figure()

    active_flows = result_df[result_df["flow"] > 0]
    for _, row in active_flows.iterrows():
        f_idx = factories.index(row["factory"])
        w_idx = warehouses.index(row["warehouse"])
        
        fig.add_trace(
            go.Scatter(
                x=[0, 1],
                y=[f_idx, w_idx],
                mode="lines",
                line=dict(width=max(1, row["flow"] / 3.0), color="#94a3b8"),
                hoverinfo="text",
                text=f"Route: {row['factory']} → {row['warehouse']}<br>Flow: {row['flow']}<br>Cost: {row['cost']:.2f}",
                showlegend=False,
            )
        )

    fig.add_trace(
        go.Scatter(
            x=node_x,
            y=node_y,
            mode="markers+text",
            marker=dict(size=35, color=node_color, line=dict(width=2, color="#1e293b")),
            text=[t.split(": ")[1] for t in node_text],
            textposition="middle center",
            hoverinfo="text",
            hovertext=node_text,
            showlegend=False,
        )
    )

    fig.update_layout(
        title=title,
        showlegend=False,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Sora, sans-serif"),
        height=400,
        margin=dict(l=20, r=20, t=40, b=20),
    )
    return fig

def plot_cost_heatmap_plotly(routes_df: pd.DataFrame, title: str = "Cost Heatmap") -> go.Figure:
    pivot = routes_df.pivot(index="factory", columns="warehouse", values="cost")
    
    fig = px.imshow(
        pivot,
        text_auto=".1f",
        aspect="auto",
        color_continuous_scale="Reds",
        title=title,
        labels=dict(x="Warehouse", y="Factory", color="Cost"),
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Sora, sans-serif"),
        height=400,
        margin=dict(l=20, r=20, t=40, b=20),
    )
    return fig

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

    positions: Dict[str, tuple] = {}
    for index, factory in enumerate(factories):
        positions[factory] = (0, index)
    for index, warehouse in enumerate(warehouses):
        positions[warehouse] = (1, index)

    standalone = axis is None
    if standalone:
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

    if standalone:
        plt.close(figure)

    return figure

def plot_cost_heatmap(
    routes_df: pd.DataFrame,
    title: str = "Cost Heatmap",
    axis: Optional[plt.Axes] = None,
    dark_mode: bool = False,
) -> plt.Figure:
    pivot = routes_df.pivot(index="factory", columns="warehouse", values="cost")

    standalone = axis is None
    if standalone:
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

    if standalone:
        plt.close(figure)

    return figure
