from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from seqtrust.core.ranking import aggregate_scores
from seqtrust.plot.helpers import display_label, format_number, save_pdf
from seqtrust.plot.style import nature_style


def _minmax(values: np.ndarray) -> np.ndarray:
    """Min-max normalize finite values while preserving missing cells."""
    result = np.full(values.shape, np.nan, dtype=float)
    finite = np.isfinite(values)
    if not finite.any():
        raise ValueError("No finite aggregated values are available for plotting.")
    lower = float(values[finite].min())
    upper = float(values[finite].max())
    if np.isclose(lower, upper):
        result[finite] = 0.5
    else:
        result[finite] = (values[finite] - lower) / (upper - lower)
    return result


class BubblePlotMixin:
    """Compact bubble summaries of one metric across tools and categories."""

    def tool_bubbles(
        self,
        tool_col: str,
        value_col: str,
        output_path: str | Path,
        aggregation: str = "mean",
        top_n: int = 5,
        category_col: str | None = None,
        tools: Sequence[object] | None = None,
        normalization: str = "global",
        bubble_size_range: tuple[float, float] = (90, 620),
        cmap: str = "cividis",
        title: str | None = None,
        value_label: str | None = None,
        annotate: bool = True,
        overwrite: bool = True,
    ):
        """Plot one aggregated bubble per tool and optional category.

        ``aggregation`` accepts ``mean``, ``max``, or ``topn``. With no
        ``category_col``, the complete DataFrame forms one row. ``global``
        normalization permits comparisons across all bubbles; ``row`` scales
        each category independently and should only be used for within-row
        comparisons. Bubble size is marker area, not radius.
        """
        import matplotlib as mpl
        import matplotlib.pyplot as plt

        normalization = str(normalization).lower()
        if normalization not in {"global", "row"}:
            raise ValueError("normalization must be either 'global' or 'row'.")
        if len(bubble_size_range) != 2:
            raise ValueError("bubble_size_range must contain (minimum, maximum).")
        size_min, size_max = map(float, bubble_size_range)
        if size_min < 0 or size_max <= size_min:
            raise ValueError(
                "bubble_size_range must satisfy 0 <= minimum < maximum."
            )

        group_cols = [tool_col]
        if category_col is not None:
            group_cols.insert(0, category_col)
        summary = aggregate_scores(
            self.trust.df,
            group_cols=group_cols,
            value_col=value_col,
            aggregation=aggregation,
            top_n=top_n,
        )
        summary[tool_col] = summary[tool_col].map(display_label)
        panel_col = category_col or "_seqtrust_category"
        if category_col is None:
            summary[panel_col] = "All data"
        else:
            summary[category_col] = summary[category_col].map(display_label)

        observed_tools = list(dict.fromkeys(summary[tool_col].tolist()))
        if tools is None:
            tool_order = observed_tools
        else:
            tool_order = list(dict.fromkeys(display_label(tool) for tool in tools))
            unavailable = [tool for tool in tool_order if tool not in observed_tools]
            if unavailable:
                raise ValueError(f"Requested tools are absent from the data: {unavailable}")
            summary = summary[summary[tool_col].isin(tool_order)].copy()
        if not tool_order:
            raise ValueError("At least one tool is required for a bubble plot.")

        category_order = list(dict.fromkeys(summary[panel_col].tolist()))
        matrix = summary.pivot(
            index=panel_col,
            columns=tool_col,
            values="score",
        ).reindex(index=category_order, columns=tool_order)
        values = matrix.to_numpy(dtype=float)
        if normalization == "global":
            scaled = _minmax(values)
        else:
            scaled = np.vstack([_minmax(row) for row in values])
        bubble_areas = size_min + scaled * (size_max - size_min)

        plot_title = title or f"Tool performance ({aggregation})"
        colorbar_label = value_label or f"Relative {value_col} ({normalization} scale)"
        color_map = mpl.colormaps[cmap].with_extremes(bad="#BDBDBD")
        norm = mpl.colors.Normalize(vmin=0, vmax=1)

        with nature_style():
            fig, ax = plt.subplots(
                figsize=(
                    max(4.4, 0.82 * len(tool_order) + 1.8),
                    max(2.6, 0.78 * len(category_order) + 1.5),
                ),
                layout="constrained",
            )
            y_positions, x_positions = np.indices(values.shape)
            finite = np.isfinite(values)
            scatter = ax.scatter(
                x_positions[finite],
                y_positions[finite],
                s=bubble_areas[finite],
                c=scaled[finite],
                cmap=color_map,
                norm=norm,
                edgecolors="white",
                linewidths=0.8,
                zorder=3,
            )
            if (~finite).any():
                ax.scatter(
                    x_positions[~finite],
                    y_positions[~finite],
                    marker="x",
                    s=24,
                    color="#777777",
                    linewidths=0.8,
                    zorder=3,
                )

            if annotate:
                for row, column in zip(*np.where(finite)):
                    rgba = color_map(norm(scaled[row, column]))
                    luminance = (
                        0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
                    )
                    ax.text(
                        column,
                        row,
                        format_number(values[row, column]),
                        ha="center",
                        va="center",
                        fontsize=6,
                        color="black" if luminance > 0.58 else "white",
                        zorder=4,
                    )

            ax.set(
                xticks=np.arange(len(tool_order)),
                xticklabels=tool_order,
                yticks=np.arange(len(category_order)),
                yticklabels=category_order,
                ylabel="" if category_col is None else category_col,
                title=plot_title,
            )
            ax.tick_params(
                axis="x",
                top=True,
                bottom=False,
                labeltop=True,
                labelbottom=False,
                rotation=40,
            )
            ax.invert_yaxis()
            ax.set_xlim(-0.6, len(tool_order) - 0.4)
            ax.set_ylim(len(category_order) - 0.55, -0.55)
            ax.set_yticks(np.arange(-0.5, len(category_order), 1), minor=True)
            ax.grid(which="minor", axis="y", color="#B8B8B8", linewidth=0.5)
            ax.tick_params(which="minor", left=False)
            for spine in ax.spines.values():
                spine.set_visible(False)

            colorbar = fig.colorbar(scatter, ax=ax, fraction=0.035, pad=0.025)
            colorbar.set_label(colorbar_label)
            output = save_pdf(fig, output_path, title=plot_title, overwrite=overwrite)

        normalized = matrix.copy()
        normalized.iloc[:, :] = scaled
        return fig, {
            "matrix": matrix,
            "normalized_matrix": normalized,
            "summary": summary,
            "normalization": normalization,
            "aggregation": aggregation,
            "output_path": output,
        }

    def metric_bubbles(
        self,
        value_cols: Sequence[str],
        tool_col: str,
        output_path: str | Path,
        tools: Sequence[object] | None = None,
        aggregation: str = "mean",
        top_n: int = 5,
        figsize: tuple[float, float] | None = None,
        bubble_size_range: tuple[float, float] = (500, 1400),
        saturation: float = 1.0,
        cmap: str | None = None,
        title: str | None = None,
        annotate: bool = True,
        overwrite: bool = True,
    ):
        """Plot one bubble per metric and tool.

        ``value_cols`` is a sequence of column names. Those names are used as
        the row labels. Rows are metrics and columns are tools.
        Each bubble is that tool's aggregated value. Color and area are
        min-max scaled inside each metric, so rows are not comparable.
        Missing values are left out of both the numerator and the count.
        """
        import matplotlib as mpl
        import matplotlib.pyplot as plt
        from matplotlib.colors import (
            LinearSegmentedColormap,
            Normalize,
            hsv_to_rgb,
            rgb_to_hsv,
            to_rgb,
        )
        from matplotlib.transforms import blended_transform_factory

        columns = [str(column) for column in value_cols]
        if not columns:
            raise ValueError("value_cols must name at least one metric.")
        if len(bubble_size_range) != 2:
            raise ValueError("bubble_size_range must contain (minimum, maximum).")
        size_min, size_max = map(float, bubble_size_range)
        if size_min < 0 or size_max <= size_min:
            raise ValueError(
                "bubble_size_range must satisfy 0 <= minimum < maximum."
            )

        pieces = []
        for column in columns:
            summary = aggregate_scores(
                self.trust.df,
                group_cols=[tool_col],
                value_col=column,
                aggregation=aggregation,
                top_n=top_n,
            )
            pieces.append(summary.set_index(tool_col)["score"].rename(column))
        means = pd.concat(pieces, axis=1)
        means.index = means.index.map(display_label)

        observed_tools = list(dict.fromkeys(means.index.tolist()))
        if tools is None:
            tool_order = observed_tools
        else:
            tool_order = list(dict.fromkeys(display_label(tool) for tool in tools))
            unavailable = [tool for tool in tool_order if tool not in observed_tools]
            if unavailable:
                raise ValueError(f"Requested tools are absent from the data: {unavailable}")
        if not tool_order:
            raise ValueError("At least one tool is required for a bubble plot.")

        matrix = means.reindex(tool_order)[columns].T
        matrix.index = columns
        values = matrix.to_numpy(dtype=float)
        scaled = np.vstack([_minmax(row) for row in values])
        bubble_areas = size_min + scaled * (size_max - size_min)

        if cmap is None:
            base_colors = ["#5C3D86", "#C4B0DE", "#F7F4FB", "#C6E6BC", "#5AAF68"]
            rgb = np.array([to_rgb(color) for color in base_colors])
            hsv = rgb_to_hsv(rgb)
            hsv[:, 1] = np.clip(hsv[:, 1] * saturation, 0, 1)
            color_map = LinearSegmentedColormap.from_list(
                "purple_green",
                hsv_to_rgb(hsv),
            )
        else:
            color_map = plt.get_cmap(cmap)
        color_norm = Normalize(0, 1)
        n_rows, n_cols = values.shape
        if figsize is None:
            figsize = (max(3.5, 0.7 * n_cols + 1.6), max(2.8, 0.62 * n_rows + 1.3))
        plot_title = title or f"Metric benchmark ({aggregation})"

        with nature_style():
            fig, ax = plt.subplots(figsize=figsize)
            y_positions, x_positions = np.indices(values.shape)
            finite = np.isfinite(values)
            ax.scatter(
                x_positions[finite],
                y_positions[finite],
                s=bubble_areas[finite],
                c=scaled[finite],
                cmap=color_map,
                norm=color_norm,
                edgecolors="white",
                linewidths=1.2,
                zorder=3,
            )
            if annotate:
                for row, column in zip(*np.where(finite)):
                    rgba = color_map(color_norm(scaled[row, column]))
                    luminance = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
                    ax.text(
                        column,
                        row,
                        f"{values[row, column]:.2f}",
                        ha="center",
                        va="center",
                        fontsize=mpl.rcParams["font.size"],
                        color="black" if luminance > 0.6 else "white",
                        zorder=4,
                    )

            ax.set_xticks(np.arange(n_cols))
            ax.set_xticklabels(
                tool_order,
                rotation=40,
                ha="left",
                rotation_mode="anchor",
            )
            ax.tick_params(
                axis="x",
                top=True,
                bottom=False,
                labeltop=True,
                labelbottom=False,
                length=0,
            )
            ax.set_yticks(np.arange(n_rows))
            ax.set_yticklabels(columns)
            ax.tick_params(axis="y", length=0, pad=8)
            ax.set_xlim(-0.5, n_cols - 0.5)
            ax.set_ylim(n_rows - 0.5, -0.5)
            for spine in ax.spines.values():
                spine.set_visible(False)
            if plot_title:
                ax.set_title(plot_title, pad=16, fontweight="bold")

            fig.tight_layout()
            fig.canvas.draw()
            renderer = fig.canvas.get_renderer()
            label_boxes = [
                tick.get_window_extent(renderer)
                for tick in ax.get_yticklabels()
                if tick.get_text()
            ]
            if label_boxes:
                to_figure = fig.transFigure.inverted()
                x_left = to_figure.transform(
                    (min(box.x0 for box in label_boxes), 0)
                )[0]
                x_right = to_figure.transform(
                    (ax.get_window_extent(renderer).x1, 0)
                )[0]
                transform = blended_transform_factory(fig.transFigure, ax.transData)
                for y_position in np.arange(-0.5, n_rows + 0.5, 1):
                    ax.plot(
                        [x_left, x_right],
                        [y_position, y_position],
                        transform=transform,
                        color="0.15",
                        linewidth=0.9,
                        linestyle=(0, (1, 1.6)),
                        clip_on=False,
                        zorder=1,
                    )
            output = save_pdf(fig, output_path, title=plot_title, overwrite=overwrite)

        return fig, {
            "matrix": matrix,
            "normalized_matrix": pd.DataFrame(scaled, index=columns, columns=tool_order),
            "aggregation": aggregation,
            "output_path": output,
        }
