from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

from seqtrust.core.ranking import aggregate_scores
from seqtrust.plot.helpers import (
    bubble_sizes,
    display_label,
    finite_limits,
    format_number,
    save_pdf,
)
from seqtrust.plot.style import categorical_colors, nature_style


def _annotate_vertical_bars(ax, bars, values) -> None:
    for bar, value in zip(bars, values):
        offset = 3 if value >= 0 else -3
        ax.annotate(
            format_number(value),
            (bar.get_x() + bar.get_width() / 2, value),
            xytext=(0, offset),
            textcoords="offset points",
            ha="center",
            va="bottom" if value >= 0 else "top",
            fontsize=6,
        )


def _annotate_horizontal_bars(ax, bars, values) -> None:
    for bar, value in zip(bars, values):
        offset = 3 if value >= 0 else -3
        ax.annotate(
            format_number(value),
            (value, bar.get_y() + bar.get_height() / 2),
            xytext=(offset, 0),
            textcoords="offset points",
            ha="left" if value >= 0 else "right",
            va="center",
            fontsize=6,
        )


class PerformancePlotMixin:
    """Performance matrices and aggregated tool bar charts."""

    def performance_matrix(
        self,
        tool_col: str,
        category_col: str,
        value_col: str,
        output_path: str | Path,
        aggregation: str = "mean",
        top_n: int = 5,
        tools_on: str = "rows",
        title: str | None = None,
        value_label: str | None = None,
        cmap: str = "cividis",
        overwrite: bool = True,
    ):
        """Plot category-level bubble scores plus one overall bar per tool.

        Bubble area and color both encode the aggregated value. Missing
        tool-category combinations are shown as gray crosses, never as zero.
        """
        import matplotlib as mpl
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D

        tools_on = str(tools_on).lower()
        if tools_on not in {"rows", "columns"}:
            raise ValueError("tools_on must be either 'rows' or 'columns'.")

        matrix_long = aggregate_scores(
            self.trust.df,
            group_cols=[tool_col, category_col],
            value_col=value_col,
            aggregation=aggregation,
            top_n=top_n,
        )
        overall = aggregate_scores(
            self.trust.df,
            group_cols=tool_col,
            value_col=value_col,
            aggregation=aggregation,
            top_n=top_n,
        ).sort_values("score", ascending=False, kind="stable")

        matrix_long[tool_col] = matrix_long[tool_col].map(display_label)
        matrix_long[category_col] = matrix_long[category_col].map(display_label)
        overall[tool_col] = overall[tool_col].map(display_label)
        tool_order = overall[tool_col].tolist()
        category_order = list(
            dict.fromkeys(self.trust.df[category_col].map(display_label).tolist())
        )
        matrix = matrix_long.pivot(
            index=tool_col,
            columns=category_col,
            values="score",
        ).reindex(index=tool_order, columns=category_order)

        finite = matrix.to_numpy(dtype=float)
        low, high = finite_limits(finite)
        norm = mpl.colors.Normalize(vmin=low, vmax=high)
        color_map = mpl.colormaps[cmap].with_extremes(bad="#BDBDBD")
        plot_title = title or f"Tool performance ({aggregation})"
        colorbar_label = value_label or f"{aggregation} {value_col}"

        with nature_style():
            if tools_on == "rows":
                width = max(6.3, 2.2 + 0.52 * len(category_order))
                height = max(3.1, 0.34 * len(tool_order) + 1.6)
                fig = plt.figure(figsize=(width, height), layout="constrained")
                grid = fig.add_gridspec(1, 2, width_ratios=[4.2, 1.4])
                ax_matrix = fig.add_subplot(grid[0, 0])
                ax_total = fig.add_subplot(grid[0, 1], sharey=ax_matrix)

                y_index, x_index = np.indices(matrix.shape)
                valid = np.isfinite(finite)
                scatter = ax_matrix.scatter(
                    x_index[valid],
                    y_index[valid],
                    s=bubble_sizes(finite[valid]),
                    c=finite[valid],
                    cmap=color_map,
                    norm=norm,
                    edgecolors="#202020",
                    linewidths=0.45,
                )
                if (~valid).any():
                    ax_matrix.scatter(
                        x_index[~valid],
                        y_index[~valid],
                        marker="x",
                        s=14,
                        color="#8A8A8A",
                        linewidths=0.7,
                    )
                ax_matrix.set(
                    xticks=np.arange(len(category_order)),
                    xticklabels=category_order,
                    yticks=np.arange(len(tool_order)),
                    yticklabels=tool_order,
                    xlabel=category_col,
                    ylabel=tool_col,
                )
                ax_matrix.tick_params(axis="x", rotation=45)
                ax_matrix.invert_yaxis()
                ax_matrix.set_xlim(-0.6, len(category_order) - 0.4)
                ax_matrix.grid(color="#E2E2E2", linewidth=0.45)

                bars = ax_total.barh(
                    np.arange(len(tool_order)),
                    overall["score"].to_numpy(),
                    color="#0072B2",
                    edgecolor="#202020",
                    linewidth=0.45,
                    height=0.66,
                )
                ax_total.axvline(0, color="#202020", linewidth=0.55)
                ax_total.set_xlabel(f"Overall\n{colorbar_label}")
                ax_total.tick_params(axis="y", labelleft=False, left=False)
                ax_total.margins(x=0.22)
                _annotate_horizontal_bars(ax_total, bars, overall["score"])
            else:
                transposed = matrix.T
                values = transposed.to_numpy(dtype=float)
                height = max(4.1, 2.3 + 0.38 * len(category_order))
                width = max(6.3, 0.57 * len(tool_order) + 2.0)
                fig = plt.figure(figsize=(width, height), layout="constrained")
                grid = fig.add_gridspec(2, 1, height_ratios=[3.8, 1.4])
                ax_matrix = fig.add_subplot(grid[0, 0])
                ax_total = fig.add_subplot(grid[1, 0], sharex=ax_matrix)

                y_index, x_index = np.indices(transposed.shape)
                valid = np.isfinite(values)
                scatter = ax_matrix.scatter(
                    x_index[valid],
                    y_index[valid],
                    s=bubble_sizes(values[valid]),
                    c=values[valid],
                    cmap=color_map,
                    norm=norm,
                    edgecolors="#202020",
                    linewidths=0.45,
                )
                if (~valid).any():
                    ax_matrix.scatter(
                        x_index[~valid],
                        y_index[~valid],
                        marker="x",
                        s=14,
                        color="#8A8A8A",
                        linewidths=0.7,
                    )
                ax_matrix.set(
                    yticks=np.arange(len(category_order)),
                    yticklabels=category_order,
                    ylabel=category_col,
                )
                ax_matrix.tick_params(axis="x", labelbottom=False)
                ax_matrix.invert_yaxis()
                ax_matrix.grid(color="#E2E2E2", linewidth=0.45)

                bars = ax_total.bar(
                    np.arange(len(tool_order)),
                    overall["score"].to_numpy(),
                    color="#0072B2",
                    edgecolor="#202020",
                    linewidth=0.45,
                    width=0.68,
                )
                ax_total.axhline(0, color="#202020", linewidth=0.55)
                ax_total.set(
                    xticks=np.arange(len(tool_order)),
                    xticklabels=tool_order,
                    xlabel=tool_col,
                    ylabel=f"Overall\n{colorbar_label}",
                )
                ax_total.tick_params(axis="x", rotation=45)
                ax_total.margins(y=0.2)
                _annotate_vertical_bars(ax_total, bars, overall["score"])

            colorbar = fig.colorbar(scatter, ax=ax_matrix, fraction=0.045, pad=0.03)
            colorbar.set_label(colorbar_label)
            size_values = np.unique(np.linspace(low, high, 3))
            size_areas = 30 + ((size_values - low) / (high - low)) * 250
            size_handles = [
                Line2D(
                    [],
                    [],
                    linestyle="none",
                    marker="o",
                    markersize=np.sqrt(area),
                    markerfacecolor=color_map(norm(value)),
                    markeredgecolor="#202020",
                    markeredgewidth=0.45,
                    label=format_number(value),
                )
                for value, area in zip(size_values, size_areas)
            ]
            if (~np.isfinite(finite)).any():
                size_handles.append(
                    Line2D(
                        [],
                        [],
                        linestyle="none",
                        marker="x",
                        color="#8A8A8A",
                        label="Missing",
                    )
                )
            ax_matrix.legend(
                handles=size_handles,
                title="Bubble area",
                loc="upper left",
                bbox_to_anchor=(1.01, 1.0),
            )
            fig.suptitle(plot_title, fontweight="bold")
            output = save_pdf(fig, output_path, title=plot_title, overwrite=overwrite)

        return fig, {"matrix": matrix, "overall": overall, "output_path": output}

    def metric_bars(
        self,
        value_col: str,
        tool_col: str,
        output_path: str | Path,
        aggregation: str = "mean",
        top_n: int = 5,
        category_col: str | None = None,
        title: str | None = None,
        value_label: str | None = None,
        overwrite: bool = True,
    ):
        """Draw zero-based tool bars globally or one panel per category."""
        import matplotlib.pyplot as plt

        groups = [tool_col] if category_col is None else [category_col, tool_col]
        summary = aggregate_scores(
            self.trust.df,
            group_cols=groups,
            value_col=value_col,
            aggregation=aggregation,
            top_n=top_n,
        )
        summary[tool_col] = summary[tool_col].map(display_label)
        if category_col is not None:
            summary[category_col] = summary[category_col].map(display_label)
            categories = list(dict.fromkeys(summary[category_col].tolist()))
        else:
            categories = [None]

        all_tools = list(dict.fromkeys(summary[tool_col].tolist()))
        palette = categorical_colors(len(all_tools))
        tool_colors = dict(zip(all_tools, palette))
        n_panels = len(categories)
        if n_panels == 4:
            n_columns = 2
        else:
            n_columns = min(3, n_panels)
        n_rows = math.ceil(n_panels / n_columns)
        plot_title = title or f"Tool performance ({aggregation})"
        y_label = value_label or f"{aggregation} {value_col}"

        with nature_style():
            fig, axes = plt.subplots(
                n_rows,
                n_columns,
                figsize=(max(3.5, 3.25 * n_columns), max(2.8, 2.55 * n_rows)),
                layout="constrained",
                squeeze=False,
                sharey=True,
            )
            for axis, category in zip(axes.flat, categories):
                panel = summary if category is None else summary[summary[category_col] == category]
                panel = panel.sort_values("score", ascending=False, kind="stable")
                positions = np.arange(len(panel))
                bars = axis.bar(
                    positions,
                    panel["score"],
                    color=[tool_colors[tool] for tool in panel[tool_col]],
                    edgecolor="#202020",
                    linewidth=0.45,
                    width=0.7,
                )
                axis.axhline(0, color="#202020", linewidth=0.55)
                axis.set(
                    xticks=positions,
                    xticklabels=panel[tool_col],
                    ylabel=y_label,
                    title="All data" if category is None else display_label(category),
                )
                axis.tick_params(axis="x", rotation=45)
                axis.grid(axis="y", color="#E2E2E2", linewidth=0.45)
                axis.margins(y=0.18)
                _annotate_vertical_bars(axis, bars, panel["score"])

            for axis in axes.flat[n_panels:]:
                axis.set_visible(False)
            fig.suptitle(plot_title, fontweight="bold")
            output = save_pdf(fig, output_path, title=plot_title, overwrite=overwrite)

        return fig, {"summary": summary, "output_path": output}
