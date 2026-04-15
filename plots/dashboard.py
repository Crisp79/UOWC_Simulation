"""Plotting helpers for the UOWC Streamlit dashboard."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Sequence

import plotly.graph_objects as go


def _default_color_cycle() -> List[str]:
    return [
        "red",
        "blue",
        "green",
        "goldenrod",
        "crimson",
        "purple",
        "orange",
        "teal",
        "brown",
        "darkcyan",
    ]


def _default_marker_cycle() -> List[str]:
    return [
        "square",
        "circle",
        "triangle-up",
        "diamond",
        "cross",
        "x",
        "triangle-down",
        "star",
    ]


def _metric_hover_template(metric_name: str) -> str:
    metric_name = metric_name.lower()
    if metric_name in {"capacity", "ergodic capacity"}:
        return (
            "<b>%{fullData.name}</b><br>"
            "SNR: %{x:.1f} dB<br>"
            "Capacity: %{y:.4f}<extra></extra>"
        )
    if metric_name in {"ber", "bit error rate"}:
        return (
            "<b>%{fullData.name}</b><br>"
            "SNR: %{x:.1f} dB<br>"
            "BER: %{y:.2e}<extra></extra>"
        )
    return (
        "<b>%{fullData.name}</b><br>SNR: %{x:.1f} dB<br>Value: %{y:.2e}<extra></extra>"
    )


def build_instance_trace_name(label: str, suffix: str | None = None) -> str:
    if suffix:
        return f"{label} {suffix}"
    return label


def add_series_traces(
    fig: go.Figure,
    x_values: Sequence[float],
    series: Sequence[Dict[str, Any]],
    *,
    metric_key: str,
    suffix: str = "",
    log_scale: bool = False,
    colors: Sequence[str] | None = None,
    markers: Sequence[str] | None = None,
) -> go.Figure:
    """
    Add one or more series to a Plotly figure.

    Parameters
    ----------
    fig:
        Figure to modify.
    x_values:
        Shared x-axis values.
    series:
        Iterable of dicts with at least:
        - "label": display name
        - metric_key: y-values
        Optional:
        - "color"
        - "marker"
    metric_key:
        Key holding the y data.
    suffix:
        Optional suffix added to each trace label.
    log_scale:
        If True, applies a log-friendly hover template.
    colors / markers:
        Optional custom styling cycles.
    """
    colors = list(colors) if colors is not None else _default_color_cycle()
    markers = list(markers) if markers is not None else _default_marker_cycle()

    hovertemplate = _metric_hover_template(metric_key)

    for idx, item in enumerate(series):
        label = str(item.get("label", f"Series {idx + 1}"))
        y_values = item[metric_key]
        color = item.get("color", colors[idx % len(colors)])
        marker = item.get("marker", markers[idx % len(markers)])

        fig.add_trace(
            go.Scatter(
                x=x_values,
                y=y_values,
                name=build_instance_trace_name(label, suffix),
                mode="lines+markers",
                line=dict(color=color),
                marker=dict(
                    symbol=marker,
                    size=7,
                    color="rgba(0,0,0,0)",
                    line=dict(color=color, width=2),
                ),
                hovertemplate=hovertemplate,
            )
        )

    if log_scale:
        fig.update_yaxes(type="log")

    return fig


def add_combined_outage_traces(
    fig: go.Figure,
    x_values: Sequence[float],
    results: Sequence[Dict[str, Any]],
    outage_towc: Sequence[float],
    *,
    colors: Sequence[str] | None = None,
    markers: Sequence[str] | None = None,
    label_suffix: str = " + TOWC (DF)",
) -> go.Figure:
    """
    Add combined outage traces using DF combination:
        1 - (1 - P_u) * (1 - P_t)
    """
    colors = list(colors) if colors is not None else _default_color_cycle()
    markers = list(markers) if markers is not None else _default_marker_cycle()

    for idx, res in enumerate(results):
        color = res.get("color", colors[idx % len(colors)])
        marker = res.get("marker", markers[idx % len(markers)])
        combined = [
            1 - (1 - pu) * (1 - pt) for pu, pt in zip(res["outage"], outage_towc)
        ]

        fig.add_trace(
            go.Scatter(
                x=x_values,
                y=combined,
                name=f"{res['label']}{label_suffix}",
                mode="lines+markers",
                line=dict(color=color),
                marker=dict(
                    symbol=marker,
                    size=7,
                    color="rgba(0,0,0,0)",
                    line=dict(color=color, width=2),
                ),
                hovertemplate=_metric_hover_template("outage"),
            )
        )

    return fig


def build_outage_figure(
    snr_db_range: Sequence[float],
    results: Sequence[Dict[str, Any]],
    outage_towc: Sequence[float],
    avg_snr: float,
    *,
    title: str,
    height: int = 500,
) -> go.Figure:
    fig = go.Figure()
    add_series_traces(fig, snr_db_range, results, metric_key="outage")
    add_combined_outage_traces(fig, snr_db_range, results, outage_towc)

    fig.add_vline(
        x=avg_snr,
        line_dash="dash",
        line_color="grey",
        opacity=0.4,
        annotation_text=f"Avg SNR = {avg_snr:.1f} dB",
        annotation_position="top right",
    )
    fig.update_layout(
        title=title,
        xaxis_title="Average SNR (dB)",
        yaxis_title="Outage Probability",
        yaxis_type="log",
        hovermode="x unified",
        height=height,
    )
    return fig


def build_ber_figure(
    snr_db_range: Sequence[float],
    results: Sequence[Dict[str, Any]],
    *,
    title: str,
    height: int = 500,
) -> go.Figure:
    fig = go.Figure()
    add_series_traces(fig, snr_db_range, results, metric_key="ber")
    fig.update_layout(
        title=title,
        xaxis_title="Average SNR (dB)",
        yaxis_title="Bit Error Rate",
        yaxis_type="log",
        hovermode="x unified",
        height=height,
    )
    return fig


def build_capacity_figure(
    snr_db_range: Sequence[float],
    results: Sequence[Dict[str, Any]],
    *,
    title: str,
    height: int = 500,
) -> go.Figure:
    fig = go.Figure()
    add_series_traces(fig, snr_db_range, results, metric_key="cap")
    fig.update_layout(
        title=title,
        xaxis_title="Average SNR (dB)",
        yaxis_title="Ergodic Capacity",
        hovermode="x unified",
        height=height,
    )
    return fig


def build_combined_outage_figure(
    snr_db_range: Sequence[float],
    results: Sequence[Dict[str, Any]],
    outage_towc: Sequence[float],
    *,
    title: str,
    height: int = 500,
) -> go.Figure:
    fig = go.Figure()
    add_combined_outage_traces(fig, snr_db_range, results, outage_towc)
    fig.update_layout(
        title=title,
        xaxis_title="Average SNR (dB)",
        yaxis_title="Outage Probability",
        yaxis_type="log",
        hovermode="x unified",
        height=height,
    )
    return fig


def build_dashboard_figures(
    snr_db_range: Sequence[float],
    results: Sequence[Dict[str, Any]],
    outage_towc: Sequence[float],
    avg_snr: float,
    *,
    n_samples: int,
) -> Dict[str, go.Figure]:
    """
    Convenience helper to build all dashboard figures in one call.
    """
    return {
        "outage": build_outage_figure(
            snr_db_range,
            results,
            outage_towc,
            avg_snr,
            title=f"Outage Probability (N={n_samples:.0e})",
        ),
        "ber": build_ber_figure(
            snr_db_range,
            results,
            title=f"Bit Error Rate (N={n_samples:.0e})",
        ),
        "capacity": build_capacity_figure(
            snr_db_range,
            results,
            title=f"Ergodic Capacity (N={n_samples:.0e})",
        ),
        "combined": build_combined_outage_figure(
            snr_db_range,
            results,
            outage_towc,
            title=f"UOWC + TOWC Outage Probability (N={n_samples:.0e})",
        ),
    }
