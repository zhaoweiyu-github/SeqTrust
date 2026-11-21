from __future__ import annotations

from pathlib import Path
from typing import Sequence


class BlatMixin:
    """Category-aware BLAT comparison against one dataset class."""

    def blat_match(
        self,
        category_col: str,
        dataset_class,
        query_classes: object | Sequence[object] | None = None,
        output_dir: str | Path | None = None,
        blat_exe: str | Path | None = None,
        min_score: int = 20,
        min_identity: float = 100,
        tile_size: int = 11,
        step_size: int = 5,
        min_match: int = 1,
        rep_match: int = 2253,
        keep_fasta: bool = True,
        save_summary: bool = True,
        summary_filename: str = "blat_group_summary.csv",
        store_key: str = "blat_match",
        verbose: bool = True,
    ):
        """Match selected query classes against one class used as the dataset.

        When ``query_classes`` is omitted, every class except
        ``dataset_class`` is used as a query.
        """
        from seqtrust.tools.blat import blat_group_match

        if category_col not in self.trust.df.columns:
            raise ValueError(f"Category column {category_col!r} was not found.")
        summary = blat_group_match(
            df=self.trust.df,
            database=dataset_class,
            query_group=query_classes,
            sequence_col=self.sequence_col,
            group_cols=[category_col],
            database_name=str(dataset_class),
            output_dir=output_dir,
            blat_exe=blat_exe,
            min_score=min_score,
            min_identity=min_identity,
            tile_size=tile_size,
            step_size=step_size,
            min_match=min_match,
            rep_match=rep_match,
            keep_fasta=keep_fasta,
            save_summary=save_summary,
            summary_filename=summary_filename,
            verbose=verbose,
        )
        self.trust.results[store_key] = summary
        self._record(
            "blat_match",
            category_col=category_col,
            dataset_class=dataset_class,
            query_classes=query_classes,
            store_key=store_key,
        )
        return summary
