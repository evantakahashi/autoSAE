"""Build Plotly figure JSON and render the Jinja2 template into tour.html."""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
from jinja2 import Environment, FileSystemLoader, select_autoescape


TEMPLATE_DIR = Path(__file__).parent / "templates"


def _fig_to_json(fig: go.Figure) -> str:
    return pio.to_json(fig, validate=False)


def induction_figure(scores: np.ndarray) -> go.Figure:
    n_layers, n_heads = scores.shape
    fig = go.Figure(
        data=go.Heatmap(
            z=scores,
            x=[f"H{h}" for h in range(n_heads)],
            y=[f"L{l}" for l in range(n_layers)],
            colorscale="Viridis",
            zmin=0.0,
            zmax=float(max(1e-3, scores.max())),
            colorbar=dict(title="score"),
            hovertemplate="%{y} %{x}<br>score=%{z:.3f}<extra></extra>",
        )
    )
    fig.update_layout(
        height=420,
        margin=dict(l=50, r=20, t=10, b=40),
        xaxis=dict(title="head", side="bottom"),
        yaxis=dict(title="layer", autorange="reversed"),
    )
    return fig


def attention_figure(attn: np.ndarray, tokens: list[str]) -> go.Figure:
    """attn: [n_layers, n_heads, T, T]. Build a figure with a dropdown per (layer, head)."""
    n_layers, n_heads, T, _ = attn.shape
    # Start showing L0H0.
    labels = [f"{i}:{t}" for i, t in enumerate(tokens)]
    fig = go.Figure(
        data=go.Heatmap(
            z=attn[0, 0],
            x=labels,
            y=labels,
            colorscale="Blues",
            zmin=0.0,
            zmax=1.0,
            hovertemplate="q=%{y}<br>k=%{x}<br>attn=%{z:.3f}<extra></extra>",
        )
    )

    buttons = []
    for l in range(n_layers):
        for h in range(n_heads):
            buttons.append(
                dict(
                    label=f"L{l}H{h}",
                    method="restyle",
                    args=[{"z": [attn[l, h].tolist()]}],
                )
            )
    fig.update_layout(
        height=560,
        margin=dict(l=80, r=20, t=50, b=80),
        xaxis=dict(title="key", tickangle=-40, tickfont=dict(size=10)),
        yaxis=dict(title="query", autorange="reversed", tickfont=dict(size=10)),
        updatemenus=[
            dict(
                buttons=buttons,
                direction="down",
                showactive=True,
                x=0.0,
                xanchor="left",
                y=1.18,
                yanchor="top",
            )
        ],
    )
    return fig


def resid_figure(norms: np.ndarray) -> go.Figure:
    fig = go.Figure(
        data=go.Bar(
            x=[f"L{i}" for i in range(len(norms))],
            y=norms,
            marker=dict(color="#2b6cb0"),
            hovertemplate="%{x}<br>||h||=%{y:.2f}<extra></extra>",
        )
    )
    fig.update_layout(
        height=320,
        margin=dict(l=50, r=20, t=10, b=40),
        xaxis=dict(title="layer (0 = embeddings)"),
        yaxis=dict(title="mean ‖h‖₂"),
    )
    return fig


def render_tour(
    *,
    out_path: Path,
    model_name: str,
    sae_layer: int,
    induction_scores: np.ndarray,
    induction_n_seqs: int,
    induction_seq_len: int,
    attn_prompt: str,
    attn: np.ndarray,
    attn_tokens: list[str],
    lens_prompt: str,
    lens_tokens: list[str],
    lens: list[list[list[tuple[str, float]]]],
    resid_norms: np.ndarray,
    resid_n_texts: int,
) -> None:
    env = Environment(
        loader=FileSystemLoader(str(TEMPLATE_DIR)),
        autoescape=select_autoescape(["html"]),
    )
    tpl = env.get_template("tour.html.j2")

    # Logit lens table: top-1 per (layer, position), with full top-K as tooltip.
    lens_rows = []
    for layer_entries in lens:
        row = []
        for pos_entries in layer_entries:
            top = pos_entries[0]
            tooltip = " | ".join(f"{t!r}:{p:.3f}" for t, p in pos_entries)
            row.append({
                "text": top[0].replace("\n", "\\n"),
                "prob": f"{top[1]:.2f}",
                "tooltip": tooltip,
            })
        lens_rows.append(row)

    html = tpl.render(
        timestamp=dt.datetime.now().strftime("%Y-%m-%d %H:%M"),
        model_name=model_name,
        sae_layer=sae_layer,
        induction_n_seqs=induction_n_seqs,
        induction_seq_len=induction_seq_len,
        induction_json=_fig_to_json(induction_figure(induction_scores)),
        attn_prompt=attn_prompt,
        attn_json=_fig_to_json(attention_figure(attn, attn_tokens)),
        lens_prompt=lens_prompt,
        lens_tokens=[t.replace("\n", "\\n") for t in lens_tokens],
        lens_rows=lens_rows,
        resid_json=_fig_to_json(resid_figure(resid_norms)),
        resid_n_texts=resid_n_texts,
    )
    out_path.write_text(html, encoding="utf-8")
