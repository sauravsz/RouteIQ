from typing import Dict, Optional
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px

def _get_theme_tokens(dark_mode: bool = False):
    if dark_mode:
        return {
            "bg": "#0b0f19",
            "text": "#f8fafc",
            "muted": "#94a3b8",
            "border": "rgba(148,163,184,0.18)",
            "factory": "#60a5fa",
            "warehouse": "#34d399",
            "plotly_template": "plotly_dark",
            "heatmap_cmap": "mako",
        }
    return {
        "bg": "#f7f4ed",
        "text": "#1c1c1c",
        "muted": "#5f5f5d",
        "border": "#eceae4",
        "factory": "#2563eb",
        "warehouse": "#059669",
        "plotly_template": "plotly_white",
        "heatmap_cmap": "Oranges",
    }

def plot_network_plotly(result_df: pd.DataFrame, title: str = "Network Flow", dark_mode: bool = False) -> go.Figure:
    t = _get_theme_tokens(dark_mode)
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
        node_color.append(t["factory"])

    for idx, w in enumerate(warehouses):
        node_x.append(1)
        node_y.append(idx)
        node_text.append(f"Warehouse: {w}")
        node_color.append(t["warehouse"])

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
                line=dict(width=max(1.5, row["flow"] / 2.5), color=t["text"]),
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
            marker=dict(size=36, color=node_color, line=dict(width=1.5, color=t["border"])),
            text=[t.split(": ")[1] for t in node_text],
            textposition="middle center",
            hoverinfo="text",
            hovertext=node_text,
            textfont=dict(color="#ffffff", family="Geist, Sora, sans-serif", size=12),
            showlegend=False,
        )
    )

    fig.update_layout(
        title=dict(text=title, font=dict(family="Geist, Sora, sans-serif", size=16, color=t["text"])),
        showlegend=False,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        template=t["plotly_template"],
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Geist, Sora, sans-serif", color=t["text"]),
        height=400,
        margin=dict(l=20, r=20, t=40, b=20),
    )
    return fig

def plot_cost_heatmap_plotly(routes_df: pd.DataFrame, title: str = "Cost Heatmap", dark_mode: bool = False) -> go.Figure:
    t = _get_theme_tokens(dark_mode)
    pivot = routes_df.pivot(index="factory", columns="warehouse", values="cost")
    
    fig = px.imshow(
        pivot,
        text_auto=".1f",
        aspect="auto",
        color_continuous_scale=t["heatmap_cmap"],
        title=title,
        labels=dict(x="Warehouse", y="Factory", color="Cost"),
    )
    fig.update_layout(
        title=dict(text=title, font=dict(family="Geist, Sora, sans-serif", size=16, color=t["text"])),
        template=t["plotly_template"],
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Geist, Sora, sans-serif", color=t["text"]),
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
    t = _get_theme_tokens(dark_mode)
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

    text_color = t["text"]
    edge_color = t["muted"]
    factory_color = t["factory"]
    warehouse_color = t["warehouse"]
    figure.patch.set_facecolor(t["bg"])
    axis.set_facecolor(t["bg"])

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
    t = _get_theme_tokens(dark_mode)
    pivot = routes_df.pivot(index="factory", columns="warehouse", values="cost")

    standalone = axis is None
    if standalone:
        figure, axis = plt.subplots(figsize=(6, 4), constrained_layout=True)
    else:
        figure = axis.figure

    text_color = t["text"]
    background_color = t["bg"]
    figure.patch.set_facecolor(background_color)
    axis.set_facecolor(background_color)

    sns.heatmap(
        pivot,
        annot=True,
        fmt=".1f",
        cmap=t["heatmap_cmap"],
        ax=axis,
        annot_kws={"color": text_color},
    )
    axis.set_title(title, color=text_color)
    axis.set_xlabel("Warehouse", color=text_color)
    axis.set_ylabel("Factory", color=text_color)
    axis.tick_params(axis="x", colors=text_color)
    axis.tick_params(axis="y", colors=text_color)

    if standalone:
        plt.close(figure)

    return figure