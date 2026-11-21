from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from seqtrust.plot.helpers import display_label, require_columns, save_pdf
from seqtrust.plot.style import nature_style


class BlatPlotMixin:
    """Matched-versus-unmatched proportions from BLAT summary output."""

    def blat_match(
        self,
        output_path: str | Path,
        summary_df: pd.DataFrame | None = None,
        result_key: str = "blat_match",
        tool_col: str = "query_group",
        matched_col: str = "n_query_matched",
        total_col: str = "n_query_total",
        title: str = "BLAT sequence matching",
        overwrite: bool = True,
    ):
        """Plot matched and unmatched query proportions for every tool."""
        import matplotlib.pyplot as plt

        if summary_df is None:
            if result_key not in self.trust.results:
                raise KeyError(
                    f"No BLAT result stored under {result_key!r}; pass summary_df "
                    "or run cal.blat_match() first."
                )
            summary_df = self.trust.results[result_key]
        if not isinstance(summary_df, pd.DataFrame):
            raise TypeError("summary_df must be a pandas DataFrame.")
        require_columns(summary_df, [tool_col, matched_col, total_col])

        data = summary_df[[tool_col, matched_col, total_col]].copy()
        data[matched_col] = pd.to_numeric(data[matched_col], errors="coerce")
        data[total_col] = pd.to_numeric(data[total_col], errors="coerce")
        if data[[matched_col, total_col]].isna().any().any():
            raise ValueError("BLAT matched and total columns must be numeric and complete.")
        if (data[total_col] <= 0).any():
            raise ValueError("BLAT total query counts must be greater than zero.")
        if (data[matched_col] < 0).any() or (data[matched_col] > data[total_col]).any():
            raise ValueError("BLAT matched counts must lie between zero and total counts.")

        data = data.groupby(tool_col, dropna=False, sort=False)[
            [matched_col, total_col]
        ].sum().reset_index()
        data[tool_col] = data[tool_col].map(display_label)
        data["matched_percent"] = data[matched_col] / data[total_col] * 100
        data["unmatched_percent"] = 100 - data["matched_percent"]
        data = data.sort_values("matched_percent", ascending=False, kind="stable")
        positions = np.arange(len(data))

        with nature_style():
            fig, ax = plt.subplots(
                figsize=(5.6, max(2.8, 0.42 * len(data) + 1.5)),
                layout="constrained",
            )
            ax.barh(
                positions,
                data["matched_percent"],
                color="#0072B2",
                edgecolor="white",
                linewidth=0.5,
                height=0.68,
                label="Matched",
            )
            ax.barh(
                positions,
                data["unmatched_percent"],
                left=data["matched_percent"],
                color="#BDBDBD",
                edgecolor="white",
                linewidth=0.5,
                height=0.68,
                label="Unmatched",
            )
            for y_position, matched, unmatched in zip(
                positions,
                data["matched_percent"],
                data["unmatched_percent"],
            ):
                if matched >= 8:
                    ax.text(
                        matched / 2,
                        y_position,
                        f"{matched:.1f}%",
                        ha="center",
                        va="center",
                        color="white",
                        fontsize=6,
                        fontweight="bold",
                    )
                else:
                    ax.text(
                        matched + 1.0,
                        y_position - 0.18,
                        f"{matched:.1f}%",
                        ha="left",
                        va="bottom",
                        color="#202020",
                        fontsize=5.5,
                    )
                if unmatched >= 8:
                    ax.text(
                        matched + unmatched / 2,
                        y_position,
                        f"{unmatched:.1f}%",
                        ha="center",
                        va="center",
                        color="#202020",
                        fontsize=6,
                    )

            ax.set(
                yticks=positions,
                yticklabels=data[tool_col],
                xlim=(0, 100),
                xlabel="Query sequences (%)",
                ylabel=tool_col,
                title=title,
            )
            ax.invert_yaxis()
            ax.set_xticks(np.arange(0, 101, 20))
            ax.grid(axis="x", color="#E0E0E0", linewidth=0.45)
            ax.legend(
                loc="upper left",
                bbox_to_anchor=(1.01, 1.0),
                ncol=1,
            )
            output = save_pdf(fig, output_path, title=title, overwrite=overwrite)

        return fig, {"proportions": data, "output_path": output}
