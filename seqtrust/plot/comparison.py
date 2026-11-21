from __future__ import annotations

import math
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from seqtrust.plot.helpers import display_label, require_columns, save_pdf
from seqtrust.plot.style import categorical_colors, nature_style


def _panel_shape(n_panels: int) -> tuple[int, int]:
    """Return a compact subplot grid for one or more category panels."""
    if n_panels == 4:
        return 2, 2
    n_columns = min(3, n_panels)
    return math.ceil(n_panels / n_columns), n_columns


def _tool_order(
    values: pd.Series,
    requested_order: Sequence[object] | None,
) -> list[str]:
    """Resolve a stable global tool order using display-safe labels."""
    observed = list(dict.fromkeys(values.map(display_label).tolist()))
    if requested_order is None:
        return observed

    requested = list(dict.fromkeys(display_label(value) for value in requested_order))
    unknown = [tool for tool in requested if tool not in observed]
    if unknown:
        raise ValueError(f"tool_order contains tools absent from the data: {unknown}")
    return requested + [tool for tool in observed if tool not in requested]


def _numeric_plot_data(
    df: pd.DataFrame,
    numeric_cols: Sequence[str],
    tool_col: str,
    category_col: str | None,
) -> pd.DataFrame:
    """Select complete finite plotting rows without converting missing data to zero."""
    required = [tool_col, *numeric_cols]
    if category_col is not None:
        required.append(category_col)
    require_columns(df, required)

    data = df[required].copy()
    for column in numeric_cols:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    finite = np.isfinite(data[list(numeric_cols)].to_numpy(dtype=float)).all(axis=1)
    data = data.loc[finite].copy()
    if data.empty:
        raise ValueError(
            f"No rows contain complete finite values for {list(numeric_cols)}."
        )
    data[tool_col] = data[tool_col].map(display_label)
    if category_col is not None:
        data[category_col] = data[category_col].map(display_label)
    return data


class ComparisonPlotMixin:
    """Tool distributions and two-metric comparisons."""

    def tool_boxplot(
        self,
        value_col: str,
        tool_col: str,
        output_path: str | Path,
        category_col: str | None = None,
        tool_order: Sequence[object] | None = None,
        title: str | None = None,
        value_label: str | None = None,
        show_fliers: bool = True,
        overwrite: bool = True,
    ):
        """Draw horizontal tool boxplots globally or separately by category.

        Every box is calculated from the unaggregated finite values available
        for one tool. Missing and non-finite numeric values are excluded rather
        than interpreted as zero.
        """
        import matplotlib.pyplot as plt

        data = _numeric_plot_data(
            self.trust.df,
            numeric_cols=[value_col],
            tool_col=tool_col,
            category_col=category_col,
        )
        order = _tool_order(data[tool_col], tool_order)
        colors = dict(zip(order, categorical_colors(len(order))))
        categories = (
            [None]
            if category_col is None
            else list(dict.fromkeys(data[category_col].tolist()))
        )
        n_rows, n_columns = _panel_shape(len(categories))
        plot_title = title or f"Distribution of {value_col} by tool"

        with nature_style():
            fig, axes = plt.subplots(
                n_rows,
                n_columns,
                figsize=(
                    max(3.8, 3.8 * n_columns),
                    max(2.8, 0.38 * len(order) * n_rows + 1.5 * n_rows),
                ),
                layout="constrained",
                squeeze=False,
                sharex=True,
            )
            for axis, category in zip(axes.flat, categories):
                panel = (
                    data
                    if category is None
                    else data[data[category_col] == category]
                )
                panel_tools = [tool for tool in order if (panel[tool_col] == tool).any()]
                distributions = [
                    panel.loc[panel[tool_col] == tool, value_col].to_numpy(dtype=float)
                    for tool in panel_tools
                ]
                boxplot = axis.boxplot(
                    distributions,
                    vert=False,
                    patch_artist=True,
                    widths=0.58,
                    whis=1.5,
                    showfliers=show_fliers,
                    medianprops={"color": "#202020", "linewidth": 1.0},
                    whiskerprops={"color": "#555555", "linewidth": 0.75},
                    capprops={"color": "#555555", "linewidth": 0.75},
                    flierprops={
                        "marker": "o",
                        "markersize": 2.5,
                        "markerfacecolor": "none",
                        "markeredgecolor": "#555555",
                        "markeredgewidth": 0.55,
                    },
                )
                for patch, tool in zip(boxplot["boxes"], panel_tools):
                    patch.set(
                        facecolor=colors[tool],
                        edgecolor="#303030",
                        linewidth=0.65,
                        alpha=0.82,
                    )

                axis.set(
                    yticks=np.arange(1, len(panel_tools) + 1),
                    yticklabels=panel_tools,
                    xlabel=value_label or value_col,
                    ylabel=tool_col,
                    title="All data" if category is None else category,
                )
                axis.invert_yaxis()
                axis.grid(axis="x", color="#E0E0E0", linewidth=0.45)

            for axis in axes.flat[len(categories):]:
                axis.set_visible(False)
            fig.suptitle(plot_title, fontweight="bold")
            output = save_pdf(fig, output_path, title=plot_title, overwrite=overwrite)

        counts = (
            data.groupby(
                [tool_col] if category_col is None else [category_col, tool_col],
                dropna=False,
                sort=False,
            )[value_col]
            .count()
            .rename("n_observations")
            .reset_index()
        )
        return fig, {
            "plot_data": data,
            "counts": counts,
            "output_path": output,
        }

    def tool_metric_scatter(
        self,
        x_col: str,
        y_col: str,
        tool_col: str,
        output_path: str | Path,
        category_col: str | None = None,
        center: str = "mean",
        error: str | None = "sd",
        tool_order: Sequence[object] | None = None,
        title: str | None = None,
        x_label: str | None = None,
        y_label: str | None = None,
        label_tools: bool = True,
        overwrite: bool = True,
    ):
        """Compare two metrics using one summary point per tool.

        ``center`` may be ``mean`` or ``median``. ``error`` may be ``sd``,
        ``se``, or ``None``. Error bars are omitted for groups where the
        selected uncertainty cannot be estimated, such as one-row groups.
        """
        import matplotlib.pyplot as plt

        if x_col == y_col:
            raise ValueError("x_col and y_col must name two different numeric columns.")
        center = str(center).lower()
        if center not in {"mean", "median"}:
            raise ValueError("center must be either 'mean' or 'median'.")
        if error is not None:
            error = str(error).lower()
        if error not in {None, "sd", "se"}:
            raise ValueError("error must be 'sd', 'se', or None.")

        data = _numeric_plot_data(
            self.trust.df,
            numeric_cols=[x_col, y_col],
            tool_col=tool_col,
            category_col=category_col,
        )
        order = _tool_order(data[tool_col], tool_order)
        colors = dict(zip(order, categorical_colors(len(order))))
        markers = ["o", "s", "^", "D", "P", "X", "v", "<", ">"]
        group_cols = [tool_col] if category_col is None else [category_col, tool_col]
        grouped = data.groupby(group_cols, dropna=False, sort=False)
        centers = grouped[[x_col, y_col]].agg(center)
        counts = grouped.size().rename("n_observations")
        standard_deviation = grouped[[x_col, y_col]].std(ddof=1)
        if error == "se":
            uncertainty = standard_deviation.div(np.sqrt(counts), axis=0)
        elif error == "sd":
            uncertainty = standard_deviation
        else:
            uncertainty = standard_deviation * np.nan

        summary = centers.join(counts)
        summary["x_error"] = uncertainty[x_col]
        summary["y_error"] = uncertainty[y_col]
        summary = summary.reset_index().rename(
            columns={x_col: "x_value", y_col: "y_value"}
        )
        categories = (
            [None]
            if category_col is None
            else list(dict.fromkeys(data[category_col].tolist()))
        )
        n_rows, n_columns = _panel_shape(len(categories))
        plot_title = title or f"{y_col} versus {x_col} by tool"

        with nature_style():
            fig, axes = plt.subplots(
                n_rows,
                n_columns,
                figsize=(max(4.1, 3.8 * n_columns), max(3.4, 3.2 * n_rows)),
                layout="constrained",
                squeeze=False,
                sharex=True,
                sharey=True,
            )
            for axis, category in zip(axes.flat, categories):
                panel = (
                    summary
                    if category is None
                    else summary[summary[category_col] == category]
                )
                for tool_index, tool in enumerate(order):
                    row = panel[panel[tool_col] == tool]
                    if row.empty:
                        continue
                    row = row.iloc[0]
                    x_error = row["x_error"] if np.isfinite(row["x_error"]) else None
                    y_error = row["y_error"] if np.isfinite(row["y_error"]) else None
                    axis.errorbar(
                        row["x_value"],
                        row["y_value"],
                        xerr=x_error,
                        yerr=y_error,
                        fmt=markers[tool_index % len(markers)],
                        markersize=5.8,
                        markerfacecolor=colors[tool],
                        markeredgecolor="#202020",
                        markeredgewidth=0.55,
                        color=colors[tool],
                        ecolor=colors[tool],
                        elinewidth=0.7,
                        capsize=2.2,
                        alpha=0.88,
                        label=None if label_tools else tool,
                    )
                    if label_tools:
                        axis.annotate(
                            tool,
                            (row["x_value"], row["y_value"]),
                            xytext=(4, 4),
                            textcoords="offset points",
                            ha="left",
                            va="bottom",
                            fontsize=6,
                        )

                axis.set(
                    xlabel=x_label or x_col,
                    ylabel=y_label or y_col,
                    title="All data" if category is None else category,
                )
                axis.grid(color="#E0E0E0", linewidth=0.45)
                if not label_tools:
                    axis.legend(title=tool_col, loc="best")

            for axis in axes.flat[len(categories):]:
                axis.set_visible(False)
            fig.suptitle(plot_title, fontweight="bold")
            output = save_pdf(fig, output_path, title=plot_title, overwrite=overwrite)

        return fig, {
            "summary": summary,
            "center": center,
            "error": error,
            "output_path": output,
        }
