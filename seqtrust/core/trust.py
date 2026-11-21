from __future__ import annotations

import pickle
from pathlib import Path
from typing import Optional

import pandas as pd

from seqtrust.cal.api import CalculationAPI
from seqtrust.core.ranking import rank_tools as calculate_tool_ranks
from seqtrust.io.read_sequences import read_sequence_file
from seqtrust.plot.api import PlotAPI


class SeqTrust:
    """Top-level manager for sequence quality and trust calculations.

    The object owns the input DataFrame and exposes two namespaces:
    ``cal`` for calculations and ``plot`` for publication-style figures.
    Calculations operate on the complete DataFrame unless a method explicitly
    asks for a category column.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        sequence_col: str = "Sequence",
        copy: bool = True,
    ) -> None:
        if not isinstance(df, pd.DataFrame):
            raise TypeError("df must be a pandas DataFrame.")
        if sequence_col not in df.columns:
            raise ValueError(
                f"Missing sequence column {sequence_col!r}. "
                f"Available columns: {list(df.columns)}"
            )

        self.sequence_col = sequence_col
        self.df = df.copy() if copy else df
        self.df[self.sequence_col] = (
            self.df[self.sequence_col].astype(str).str.strip().str.upper()
        )
        self.history: list[dict] = []
        self.feature_names: dict[str, list[str]] = {}
        self.results: dict[str, object] = {}

        self.cal = CalculationAPI(self)
        self.plot = PlotAPI(self)

    @classmethod
    def from_file(
        cls,
        path: str | Path,
        sequence_col: str = "Sequence",
        sep: Optional[str] = None,
        uppercase: bool = True,
        drop_empty: bool = True,
    ) -> "SeqTrust":
        """Create a SeqTrust object from CSV, TSV, FASTA, or text input."""
        df = read_sequence_file(
            path=path,
            sequence_col=sequence_col,
            sep=sep,
            uppercase=uppercase,
            drop_empty=drop_empty,
        )
        return cls(df=df, sequence_col=sequence_col)

    def preview(self, n: int = 5) -> pd.DataFrame:
        """Return the first ``n`` rows without changing stored data."""
        return self.df.head(n)

    def rank_tools(
        self,
        value_col: str,
        tool_col: str,
        aggregation: str = "mean",
        top_n: int = 5,
        category_col: str | None = None,
        higher_is_better: bool = True,
        rank_method: str = "min",
    ) -> pd.DataFrame:
        """Rank tools by mean, maximum, or top-N mean score.

        Larger values rank first by default. When ``category_col`` is given,
        ranks are calculated independently inside each category.
        """
        return calculate_tool_ranks(
            df=self.df,
            value_col=value_col,
            tool_col=tool_col,
            aggregation=aggregation,
            top_n=top_n,
            category_col=category_col,
            higher_is_better=higher_is_better,
            rank_method=rank_method,
        )

    @property
    def info(self) -> dict:
        """Return a compact summary of the current object state."""
        return {
            "n_rows": len(self.df),
            "sequence_col": self.sequence_col,
            "columns": list(self.df.columns),
            "history": list(self.history),
            "feature_names": dict(self.feature_names),
            "result_keys": list(self.results),
        }

    def save(
        self,
        path: str | Path,
        protocol: int = pickle.HIGHEST_PROTOCOL,
    ) -> None:
        """Serialize the complete SeqTrust object to a pickle file."""
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("wb") as handle:
            pickle.dump(self, handle, protocol=protocol)

    @classmethod
    def load(cls, path: str | Path) -> "SeqTrust":
        """Restore a SeqTrust object previously written by :meth:`save`."""
        with Path(path).open("rb") as handle:
            obj = pickle.load(handle)
        if not isinstance(obj, cls):
            raise TypeError(f"Loaded object is {type(obj)}, not {cls.__name__}.")
        return obj
