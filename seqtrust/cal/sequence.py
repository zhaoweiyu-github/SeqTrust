from __future__ import annotations


class SequenceMetricsMixin:
    """Whole-dataset sequence composition and repeat metrics."""

    def _apply_sequence_metric(self, function, output_col: str, **kwargs):
        result = function(
            self.trust.df,
            sequence_col=self.sequence_col,
            output_col=output_col,
            **kwargs,
        )
        self.trust.df[output_col] = result[output_col]
        return self.trust.df

    def add_GC_content(self, output_col: str = "GC_content"):
        """Add the fraction of G and C bases in every sequence."""
        from seqtrust.tools.sequence_composition import calculate_gc

        result = self._apply_sequence_metric(calculate_gc, output_col)
        self._record("add_GC_content", output_col=output_col)
        return result

    def add_complexity_score(self, output_col: str = "complexity_score"):
        """Add normalized nucleotide Shannon entropy for every sequence."""
        from seqtrust.tools.sequence_composition import calculate_complexity

        result = self._apply_sequence_metric(calculate_complexity, output_col)
        self._record("add_complexity_score", output_col=output_col)
        return result

    def add_max_repeat_length(self, output_col: str = "max_repeat_length"):
        """Add the longest homopolymer run length for every sequence."""
        from seqtrust.tools.sequence_composition import calculate_max_repeat

        result = self._apply_sequence_metric(calculate_max_repeat, output_col)
        self._record("add_max_repeat_length", output_col=output_col)
        return result

    def add_max_tandem_repeat_length(
        self,
        output_col: str = "max_tandem_repeat_length",
        min_unit_size: int = 6,
        min_copies: int = 2,
    ):
        """Add the longest adjacent multi-base tandem repeat length."""
        from seqtrust.tools.sequence_composition import calculate_max_tandem_repeat

        result = self._apply_sequence_metric(
            calculate_max_tandem_repeat,
            output_col,
            min_unit_size=min_unit_size,
            min_copies=min_copies,
        )
        self._record(
            "add_max_tandem_repeat_length",
            output_col=output_col,
            min_unit_size=min_unit_size,
            min_copies=min_copies,
        )
        return result

    def add_max_dispersed_repeat_count(
        self,
        output_col: str = "max_dispersed_repeat_count",
        unit_size: int = 3,
    ):
        """Add the largest non-overlapping count of a dispersed repeat unit."""
        from seqtrust.tools.sequence_composition import calculate_max_dispersed_repeat

        result = self._apply_sequence_metric(
            calculate_max_dispersed_repeat,
            output_col,
            unit_size=unit_size,
        )
        self._record(
            "add_max_dispersed_repeat_count",
            output_col=output_col,
            unit_size=unit_size,
        )
        return result

    def add_kmer_diversity(
        self,
        output_col: str = "kmer_diversity",
        kmer_size: int = 3,
        normalize: bool = True,
    ):
        """Add per-sequence Shannon diversity of overlapping k-mers."""
        from seqtrust.tools.sequence_composition import calculate_kmer_diversity

        result = self._apply_sequence_metric(
            calculate_kmer_diversity,
            output_col,
            kmer_size=kmer_size,
            normalize=normalize,
        )
        self._record(
            "add_kmer_diversity",
            output_col=output_col,
            kmer_size=kmer_size,
            normalize=normalize,
        )
        return result
