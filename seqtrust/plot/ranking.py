from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from seqtrust.plot.helpers import display_label, require_columns, save_pdf
from seqtrust.plot.style import nature_style


def _pairwise_rank_correlation(
    rank_matrix: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Calculate Spearman correlation using only tools shared by each pair."""
    datasets = list(rank_matrix.index)
    correlation = pd.DataFrame(np.nan, index=datasets, columns=datasets, dtype=float)
    common_counts = pd.DataFrame(0, index=datasets, columns=datasets, dtype=int)
    for first in datasets:
        for second in datasets:
            paired = pd.concat(
                [rank_matrix.loc[first], rank_matrix.loc[second]],
                axis=1,
                keys=["first", "second"],
            ).dropna()
            n_common = len(paired)
            common_counts.loc[first, second] = n_common
            if n_common < 2:
                continue
            first_rank = paired["first"].rank(method="average")
            second_rank = paired["second"].rank(method="average")
            if first_rank.nunique() < 2 or second_rank.nunique() < 2:
                continue
            correlation.loc[first, second] = first_rank.corr(second_rank)
    return correlation, common_counts


class RankingPlotMixin:
    """Correlation plots for tool rankings across datasets."""

    def rank_correlation(
        self,
        rank_df: pd.DataFrame,
        dataset_col: str,
        output_path: str | Path,
        tool_col: str = "tool",
        rank_col: str = "rank",
        title: str = "Tool-rank correlation",
        annotate_n: bool = True,
        overwrite: bool = True,
    ):
        """Plot pairwise Spearman correlations among dataset-specific ranks."""
        import matplotlib as mpl
        import matplotlib.pyplot as plt

        if not isinstance(rank_df, pd.DataFrame):
            raise TypeError("rank_df must be a pandas DataFrame.")
        require_columns(rank_df, [dataset_col, tool_col, rank_col])
        working = rank_df[[dataset_col, tool_col, rank_col]].copy()
        working[rank_col] = pd.to_numeric(working[rank_col], errors="coerce")
        working = working.dropna(subset=[rank_col])
        working[dataset_col] = working[dataset_col].map(display_label)
        working[tool_col] = working[tool_col].map(display_label)
        duplicates = working.duplicated([dataset_col, tool_col], keep=False)
        if duplicates.any():
            examples = working.loc[duplicates, [dataset_col, tool_col]].head().to_dict("records")
            raise ValueError(
                "rank_df must contain one rank per dataset-tool pair; "
                f"duplicate examples: {examples}"
            )
        if working[dataset_col].nunique() < 2:
            raise ValueError("At least two datasets are required for rank correlation.")

        rank_matrix = working.pivot(
            index=dataset_col,
            columns=tool_col,
            values=rank_col,
        )
        correlation, common_counts = _pairwise_rank_correlation(rank_matrix)
        values = correlation.to_numpy(dtype=float)
        n_datasets = len(correlation)
        cmap = mpl.colormaps["RdBu_r"].with_extremes(bad="#BDBDBD")

        with nature_style():
            size = max(3.8, 0.52 * n_datasets + 1.7)
            fig, ax = plt.subplots(figsize=(size, size), layout="constrained")
            image = ax.imshow(
                values,
                cmap=cmap,
                vmin=-1,
                vmax=1,
                interpolation="nearest",
            )
            ax.set(
                xticks=np.arange(n_datasets),
                xticklabels=correlation.columns,
                yticks=np.arange(n_datasets),
                yticklabels=correlation.index,
                xlabel="Dataset",
                ylabel="Dataset",
                title=title,
            )
            ax.tick_params(axis="x", rotation=45)
            ax.set_xticks(np.arange(-0.5, n_datasets, 1), minor=True)
            ax.set_yticks(np.arange(-0.5, n_datasets, 1), minor=True)
            ax.grid(which="minor", color="white", linewidth=0.8)
            ax.tick_params(which="minor", bottom=False, left=False)

            for row in range(n_datasets):
                for column in range(n_datasets):
                    value = values[row, column]
                    if not np.isfinite(value):
                        text = "NA"
                        color = "#202020"
                    else:
                        text = f"{value:.2f}"
                        if annotate_n:
                            text += f"\nn={common_counts.iloc[row, column]}"
                        color = "white" if abs(value) >= 0.55 else "#202020"
                    ax.text(
                        column,
                        row,
                        text,
                        ha="center",
                        va="center",
                        fontsize=5.5,
                        color=color,
                    )

            colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.03)
            colorbar.set_label("Spearman rank correlation")
            output = save_pdf(fig, output_path, title=title, overwrite=overwrite)

        return fig, {
            "correlation": correlation,
            "n_common_tools": common_counts,
            "output_path": output,
        }
