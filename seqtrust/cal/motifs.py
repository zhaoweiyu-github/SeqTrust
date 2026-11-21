from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from pathlib import Path
PACKAGE_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_MEME_PATH = (
    PACKAGE_ROOT
    / "data"
    / "motif"
    / "HOCOMOCO"
    / "H14CORE_meme_format.meme"
)

DEFAULT_CLUSTER_PATH = (
    PACKAGE_ROOT
    / "data"
    / "motif"
    / "HOCOMOCO"
    / "motif_clusters.tsv"
)


class MotifMixin:
    """Motif scanning and derived whole-dataset motif metrics."""

    def add_motif_features(
        self,
        meme_path: str | None = None,
        motif_clusters: str | None = None,
        source: str = "HOCOMOCO",
        use_uniform_bg: bool = False,
        show_progress: bool = False,
        output_type: str = "count",
        output_col: str = "motif_features",
        threshold: float = 1e-4,
        qval_cutoff: float = 0.05,
        pval_cutoff: float = 1e-4,
        bg_freq=None,
        scan_store_key: str = "motif_scan",
    ):
        """Scan motifs once across the whole dataset.

        ``meme_path`` and ``motif_clusters`` default to ``DEFAULT_MEME_PATH``
        and ``DEFAULT_CLUSTER_PATH`` in this module. A single nucleotide
        background is estimated from all sequences unless ``bg_freq`` or
        ``use_uniform_bg`` is supplied.
        """
        from seqtrust.tools.motifs import add_motif_features as scan_all

        if meme_path is None:
            meme_path = DEFAULT_MEME_PATH
        if motif_clusters is None:
            motif_clusters = DEFAULT_CLUSTER_PATH

        frame = self.trust.df
        result, motif_names, scan_df = scan_all(
            frame,
            meme_path=meme_path,
            motif_clusters=motif_clusters,
            source=source,
            sequence_col=self.sequence_col,
            use_uniform_bg=use_uniform_bg,
            show_progress=show_progress,
            output_type=output_type,
            feature_col=output_col,
            threshold=threshold,
            qval_cutoff=qval_cutoff,
            pval_cutoff=pval_cutoff,
            bg_freq=bg_freq,
        )
        motif_names = list(motif_names)
        # The scanner preserves row order, so feature vectors line up with df.
        self.trust.df[output_col] = list(result[output_col])
        self.trust.feature_names[output_col] = motif_names

        if scan_df is not None and not scan_df.empty:
            scan_df = scan_df.copy()
            if "ID" not in scan_df.columns and scan_df.index.name == "ID":
                scan_df = scan_df.reset_index()
        else:
            scan_df = pd.DataFrame(
                columns=[
                    "ID",
                    "Matrix_id",
                    "strand",
                    "start",
                    "end",
                    "score",
                    "pval",
                    "qval",
                ]
            )
        self.trust.results[scan_store_key] = scan_df

        self._add_motif_hit_summaries(
            motif_names=motif_names,
            motif_col=output_col,
            cluster_path=motif_clusters,
        )
        self._record(
            "add_motif_features",
            source=source,
            output_col=output_col,
            output_type=output_type,
            scan_store_key=scan_store_key,
        )
        return self.trust.df

    def _add_motif_hit_summaries(
        self,
        motif_names: list[str],
        motif_col: str,
        cluster_path,
    ) -> None:
        """Add hit counts and collapsed motif-cluster counts."""
        if not motif_names or self.trust.df.empty:
            return
        features = np.stack(
            [np.asarray(value, dtype=np.float32) for value in self.trust.df[motif_col]]
        )
        hits = features > 0
        is_znf = np.array([str(name).startswith("ZNF") for name in motif_names])
        self.trust.df["motif_category_count"] = hits.sum(axis=1)
        self.trust.df["motif_category_count_noZNF"] = hits[:, ~is_znf].sum(axis=1)
        self.trust.df["motif_hit_count"] = features.sum(axis=1)
        self.trust.df["motif_hit_count_noZNF"] = features[:, ~is_znf].sum(axis=1)

        cluster_df = pd.read_csv(cluster_path, sep="\t")
        motif_to_cluster: dict[str, str] = {}
        for representative, members in zip(
            cluster_df["Representative motif"],
            cluster_df["Clustered motifs"],
        ):
            representative = str(representative)
            motif_to_cluster[representative] = representative
            for member in str(members).split(","):
                motif_to_cluster[member.strip()] = representative
        clusters = np.array(
            [motif_to_cluster.get(name, name) for name in motif_names]
        )
        unique_clusters, cluster_index = np.unique(clusters, return_inverse=True)
        membership = np.zeros(
            (len(unique_clusters), len(motif_names)),
            dtype=np.int32,
        )
        membership[cluster_index, np.arange(len(motif_names))] = 1
        cluster_hits = (membership @ hits.astype(np.int32).T) > 0
        self.trust.df["motif_cluster_category_count"] = cluster_hits.sum(axis=0)

    def add_motif_diversity(self, motif_col: str = "motif_features"):
        """Add per-sequence entropy, normalized entropy, and motif count."""
        from seqtrust.tools.motifs import calculate_motif_diversity

        if motif_col not in self.trust.df.columns:
            raise ValueError(f"Motif feature column {motif_col!r} was not found.")
        result = calculate_motif_diversity(
            self.trust.df,
            feature_col=motif_col,
        )
        for column in result:
            self.trust.df[column] = result[column].to_numpy()
        self.trust.feature_names["motif_diversity"] = list(result.columns)
        self._record("add_motif_diversity", motif_col=motif_col)
        return self.trust.df

    def add_motif_recovery(
        self,
        reference_motifs,
        motif_col: str = "motif_features",
        output_col: str = "motif_recovery_ratio",
        overlap_col: str = "motif_recovery_overlap",
    ):
        """Add the fraction of reference motifs recovered by each sequence."""
        if motif_col not in self.trust.df.columns:
            raise ValueError(f"Motif feature column {motif_col!r} was not found.")
        motif_names = list(self.trust.feature_names.get(motif_col, []))
        if not motif_names:
            raise RuntimeError(
                f"No motif names are registered for {motif_col!r}; "
                "run add_motif_features() first."
            )
        reference_motifs = list(reference_motifs)
        if not reference_motifs:
            raise ValueError("reference_motifs must not be empty.")

        name_to_index = {name: index for index, name in enumerate(motif_names)}
        missing = [name for name in reference_motifs if name not in name_to_index]
        if missing:
            warnings.warn(
                f"{len(missing)}/{len(reference_motifs)} reference motifs are "
                f"not present in the scanned motif set, for example {missing[:5]}."
            )
        reference_indices = np.asarray(
            [name_to_index[name] for name in reference_motifs if name in name_to_index],
            dtype=int,
        )
        features = np.stack(
            [np.asarray(value, dtype=np.float32) for value in self.trust.df[motif_col]]
        )
        hits = features > 0
        overlap = np.zeros_like(hits, dtype=np.int8)
        if len(reference_indices):
            overlap[:, reference_indices] = hits[:, reference_indices]
        self.trust.df[output_col] = overlap.sum(axis=1) / len(reference_motifs)
        self.trust.df[overlap_col] = list(overlap)
        self.trust.feature_names[overlap_col] = motif_names
        self._record(
            "add_motif_recovery",
            motif_col=motif_col,
            n_reference=len(reference_motifs),
            output_col=output_col,
            overlap_col=overlap_col,
        )
        return self.trust.df
