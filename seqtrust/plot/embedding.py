from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from seqtrust.plot.helpers import display_label, require_columns, save_pdf
from seqtrust.plot.style import categorical_colors, nature_style


class EmbeddingPlotMixin:
    """Two-dimensional UMAP views of stored embedding vectors."""

    def embedding_umap(
        self,
        embedding_col: str,
        color_col: str,
        output_path: str | Path,
        n_neighbors: int = 15,
        min_dist: float = 0.1,
        metric: str = "euclidean",
        random_state: int = 42,
        title: str | None = None,
        point_size: float = 14,
        alpha: float = 0.78,
        overwrite: bool = True,
    ):
        """Fit UMAP to one vector per row and color points by category."""
        import matplotlib.pyplot as plt

        require_columns(self.trust.df, [embedding_col, color_col])
        if not isinstance(n_neighbors, int) or n_neighbors < 2:
            raise ValueError("n_neighbors must be an integer of at least 2.")
        if not 0 <= min_dist <= 1:
            raise ValueError("min_dist must lie between 0 and 1.")

        vectors: list[np.ndarray] = []
        valid_positions: list[int] = []
        expected_length: int | None = None
        for position, value in enumerate(self.trust.df[embedding_col]):
            try:
                vector = np.asarray(value, dtype=float).reshape(-1)
            except (TypeError, ValueError):
                continue
            if vector.size == 0 or not np.isfinite(vector).all():
                continue
            if expected_length is None:
                expected_length = vector.size
            if vector.size != expected_length:
                raise ValueError(
                    f"Embedding at row position {position} has length {vector.size}; "
                    f"expected {expected_length}."
                )
            vectors.append(vector)
            valid_positions.append(position)

        if len(vectors) < 3:
            raise ValueError("At least three complete embedding vectors are required for UMAP.")
        n_dropped = len(self.trust.df) - len(vectors)
        if n_dropped:
            warnings.warn(
                f"Dropped {n_dropped} rows with missing or non-finite embeddings.",
                stacklevel=2,
            )

        try:
            import umap
        except ImportError as exc:
            raise ImportError(
                "embedding_umap requires umap-learn; install SeqTrust with "
                "the 'plot' optional dependencies."
            ) from exc

        effective_neighbors = min(n_neighbors, len(vectors) - 1)
        reducer = umap.UMAP(
            n_components=2,
            n_neighbors=effective_neighbors,
            min_dist=min_dist,
            metric=metric,
            random_state=random_state,
            n_jobs=1,
        )
        coordinates = reducer.fit_transform(np.stack(vectors))
        plotted = self.trust.df.iloc[valid_positions][[color_col]].copy()
        plotted["UMAP1"] = coordinates[:, 0]
        plotted["UMAP2"] = coordinates[:, 1]
        plotted["_category"] = plotted[color_col].map(display_label)

        categories = list(dict.fromkeys(plotted["_category"].tolist()))
        colors = categorical_colors(len(categories))
        markers = ["o", "s", "^", "D", "P", "X", "v", "<", ">"]
        plot_title = title or f"UMAP of {embedding_col}"

        with nature_style():
            width = 5.2 if len(categories) <= 5 else 6.2
            fig, ax = plt.subplots(figsize=(width, 4.4), layout="constrained")
            for index, (category, color) in enumerate(zip(categories, colors)):
                subset = plotted[plotted["_category"] == category]
                ax.scatter(
                    subset["UMAP1"],
                    subset["UMAP2"],
                    s=point_size,
                    color=color,
                    marker=markers[index % len(markers)],
                    alpha=alpha,
                    edgecolors="white",
                    linewidths=0.25,
                    label=category,
                    rasterized=len(plotted) > 2000,
                )
            ax.set(
                xlabel="UMAP 1",
                ylabel="UMAP 2",
                title=plot_title,
            )
            ax.set_xticks([])
            ax.set_yticks([])
            ax.legend(
                title=color_col,
                loc="center left",
                bbox_to_anchor=(1.01, 0.5),
                ncol=1 if len(categories) <= 12 else 2,
                markerscale=1.25,
            )
            output = save_pdf(fig, output_path, title=plot_title, overwrite=overwrite)

        plotted = plotted.drop(columns="_category")
        return fig, {
            "embedding": plotted,
            "n_neighbors_used": effective_neighbors,
            "random_state": random_state,
            "output_path": output,
        }
