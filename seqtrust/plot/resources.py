from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from seqtrust.plot.helpers import (
    bubble_sizes,
    display_label,
    format_number,
    require_columns,
    save_pdf,
)
from seqtrust.plot.style import nature_style


class ResourcePlotMixin:
    """Resource-use bubble plots for externally supplied tool metadata."""

    def resource_usage(
        self,
        resource_df: pd.DataFrame,
        tool_col: str,
        time_col: str,
        memory_col: str,
        output_path: str | Path,
        time_unit: str = "h",
        memory_unit: str = "GB",
        title: str = "Computational resource usage",
        overwrite: bool = True,
    ):
        """Plot time and memory values as separately scaled bubble areas.

        Raw values are printed above each bubble because time and memory have
        different units and their bubble areas must not be compared across rows.
        """
        import matplotlib.pyplot as plt

        if not isinstance(resource_df, pd.DataFrame):
            raise TypeError("resource_df must be a pandas DataFrame.")
        require_columns(resource_df, [tool_col, time_col, memory_col])
        data = resource_df[[tool_col, time_col, memory_col]].copy()
        if data[tool_col].duplicated().any():
            duplicates = data.loc[data[tool_col].duplicated(False), tool_col].tolist()
            raise ValueError(f"Each tool must appear once; duplicates: {duplicates[:5]}")
        data[tool_col] = data[tool_col].map(display_label)
        for column in (time_col, memory_col):
            data[column] = pd.to_numeric(data[column], errors="coerce")
            if (data[column].dropna() < 0).any():
                raise ValueError(f"{column!r} must contain non-negative values.")

        positions = np.arange(len(data))
        row_specs = [
            (time_col, 1.0, "#EE7733", time_unit),
            (memory_col, 0.0, "#44AA99", memory_unit),
        ]
        with nature_style():
            fig, ax = plt.subplots(
                figsize=(max(5.0, 0.72 * len(data) + 1.7), 2.9),
                layout="constrained",
            )
            for column, y_position, color, unit in row_specs:
                values = data[column].to_numpy(dtype=float)
                valid = np.isfinite(values)
                if valid.any():
                    sizes = bubble_sizes(values[valid], minimum=55, maximum=520)
                    ax.scatter(
                        positions[valid],
                        np.full(valid.sum(), y_position),
                        s=sizes,
                        color=color,
                        edgecolors="#202020",
                        linewidths=0.5,
                        alpha=0.82,
                    )
                    for x_position, value in zip(positions[valid], values[valid]):
                        ax.annotate(
                            f"{format_number(value)} {unit}".strip(),
                            (x_position, y_position),
                            xytext=(0, 11),
                            textcoords="offset points",
                            ha="center",
                            va="bottom",
                            fontsize=5.5,
                        )
                if (~valid).any():
                    ax.scatter(
                        positions[~valid],
                        np.full((~valid).sum(), y_position),
                        marker="x",
                        s=22,
                        color="#777777",
                        linewidths=0.8,
                    )
                    for x_position in positions[~valid]:
                        ax.annotate(
                            "Missing",
                            (x_position, y_position),
                            xytext=(0, 8),
                            textcoords="offset points",
                            ha="center",
                            va="bottom",
                            fontsize=5.5,
                            color="#555555",
                        )

            ax.set(
                xticks=positions,
                xticklabels=data[tool_col],
                yticks=[0, 1],
                yticklabels=["Memory", "Time"],
                title=title,
            )
            ax.tick_params(axis="x", rotation=45)
            ax.set_ylim(-0.42, 1.48)
            ax.grid(axis="x", color="#D7D7D7", linestyle="--", linewidth=0.55)
            ax.text(
                0.0,
                -0.23,
                "Bubble area is scaled within each resource row; labels show raw values.",
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=5.5,
                color="#555555",
            )
            output = save_pdf(fig, output_path, title=title, overwrite=overwrite)

        return fig, {"resources": data, "output_path": output}
