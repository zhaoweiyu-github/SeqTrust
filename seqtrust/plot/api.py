from __future__ import annotations

from seqtrust.plot.blat import BlatPlotMixin
from seqtrust.plot.bubbles import BubblePlotMixin
from seqtrust.plot.comparison import ComparisonPlotMixin
from seqtrust.plot.embedding import EmbeddingPlotMixin
from seqtrust.plot.logos import SequenceLogoMixin
from seqtrust.plot.performance import PerformancePlotMixin
from seqtrust.plot.ranking import RankingPlotMixin
from seqtrust.plot.resources import ResourcePlotMixin
from seqtrust.plot.tree import HorizontalTreeMixin


class PlotAPI(
    PerformancePlotMixin,
    BubblePlotMixin,
    ComparisonPlotMixin,
    SequenceLogoMixin,
    RankingPlotMixin,
    EmbeddingPlotMixin,
    ResourcePlotMixin,
    BlatPlotMixin,
    HorizontalTreeMixin,
):
    """Nature-style, publication-oriented plotting namespace.

    Every public method writes an opaque PDF to the explicitly supplied path
    and returns the Matplotlib figure plus the transformed plotting data.
    """

    def __init__(self, trust) -> None:
        self.trust = trust
