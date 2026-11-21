from __future__ import annotations

from .attribution import AttributionMixin
from .blat import BlatMixin
from .embeddings import EmbeddingMixin
from .motifs import MotifMixin
from .sequence import SequenceMetricsMixin


class CalculationAPI(
    SequenceMetricsMixin,
    EmbeddingMixin,
    AttributionMixin,
    MotifMixin,
    BlatMixin,
):
    """Unified calculation namespace for all supported SeqTrust features."""

    def __init__(self, trust) -> None:
        self.trust = trust

    @property
    def df(self):
        """Return the live DataFrame owned by the parent SeqTrust object."""
        return self.trust.df

    @property
    def sequence_col(self) -> str:
        """Return the configured sequence column name."""
        return self.trust.sequence_col

    def _record(self, operation: str, **parameters) -> None:
        """Append one structured calculation record to the history."""
        self.trust.history.append({"operation": operation, **parameters})
