from typing import Dict, Optional
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px

# Lovable theme tokens
CREAM_BG = "#f7f4ed"
CHARCOAL_TEXT = "#1c1c1c"
MUTED_TEXT = "#5f5f5d"
BORDER_COLOR = "#eceae4"
FACTORY_NODE_COLOR = "#2563eb"
WAREHOUSE_NODE_COLOR = "#059669"

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
        node_color.append(FACTORY_NODE_COLOR)

    for idx, w in enumerate(warehouses):
        node_x.append(1)
        node_y.append(idx)
        node_text.append(f"Warehouse: {w}")
        node_color.append(WAREHOUSE_NODE_COLOR)

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
                line=dict(width=max(1.5, row["flow"] / 2.5), color=CHARCOAL_TEXT),
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
            marker=dict(size=36, color=node_color, line=dict(width=1.5, color=BORDER_COLOR)),
            text=[t.split(": ")[1] for t in node_text],
            textposition="middle center",
            hoverinfo="text",
            hovertext=node_text,
            textfont=dict(color="#ffffff", family="Sora, sans-serif", size=12),
            showlegend=False,
        )
    )

    fig.update_layout(
        title=dict(text=title, font=dict(family="Sora, sans-serif", size=16, color=CHARCOAL_TEXT)),
        showlegend=False,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        template="plotly_white",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Sora, sans-serif", color=CHARCOAL_TEXT),
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
        color_continuous_scale="Oranges",
        title=title,
        labels=dict(x="Warehouse", y="Factory", color="Cost"),
    )
    fig.update_layout(
        title=dict(text=title, font=dict(family="Sora, sans-serif", size=16, color=CHARCOAL_TEXT)),
        template="plotly_white",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Sora, sans-serif", color=CHARCOAL_TEXT),
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

    text_color = CHARCOAL_TEXT
    edge_color = MUTED_TEXT
    factory_color = FACTORY_NODE_COLOR
    warehouse_color = WAREHOUSE_NODE_COLOR
    figure.patch.set_facecolor(CREAM_BG)
    axis.set_facecolor(CREAM_BG)

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
        font_color="#ffffff",
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

    text_color = CHARCOAL_TEXT
    background_color = CREAM_BG
    figure.patch.set_facecolor(background_color)
    axis.set_facecolor(background_color)

    sns.heatmap(
        pivot,
        annot=True,
        fmt=".1f",
        cmap="Oranges",
        ax=axis,
        annot_kws={"color": CHARCOAL_TEXT},
    )
    axis.set_title(title, color=text_color)
    axis.set_xlabel("Warehouse", color=text_color)
    axis.set_ylabel("Factory", color=text_color)
    axis.tick_params(axis="x", colors=text_color)
    axis.tick_params(axis="y", colors=text_color)

    if standalone:
        plt.close(figure)

    return figure