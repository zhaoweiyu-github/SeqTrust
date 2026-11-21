from __future__ import annotations

from collections.abc import Sequence

import pandas as pd


VALID_AGGREGATIONS = {"mean", "max", "topn"}


def aggregate_scores(
    df: pd.DataFrame,
    group_cols: str | Sequence[str],
    value_col: str,
    aggregation: str = "mean",
    top_n: int = 5,
    output_col: str = "score",
) -> pd.DataFrame:
    """Aggregate a numeric score without treating missing values as zero.

    ``topn`` means the arithmetic mean of the largest ``top_n`` finite values
    in each group. If a group contains fewer than ``top_n`` values, all
    available values are used.
    """
    groups = [group_cols] if isinstance(group_cols, str) else list(group_cols)
    aggregation = str(aggregation).lower()
    if aggregation not in VALID_AGGREGATIONS:
        raise ValueError(
            f"aggregation must be one of {sorted(VALID_AGGREGATIONS)}; "
            f"received {aggregation!r}."
        )
    if not groups:
        raise ValueError("At least one grouping column is required.")
    if aggregation == "topn" and (not isinstance(top_n, int) or top_n < 1):
        raise ValueError("top_n must be a positive integer when aggregation='topn'.")

    required = groups + [value_col]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(
            f"Missing columns: {missing}. Available columns: {list(df.columns)}"
        )

    working = df[required].copy()
    working[value_col] = pd.to_numeric(working[value_col], errors="coerce")
    working = working.dropna(subset=[value_col])
    if working.empty:
        raise ValueError(f"Column {value_col!r} contains no finite numeric values.")

    grouped = working.groupby(groups, dropna=False, sort=False)[value_col]
    if aggregation == "mean":
        values = grouped.mean()
    elif aggregation == "max":
        values = grouped.max()
    else:
        values = grouped.apply(lambda series: series.nlargest(top_n).mean())

    result = values.rename(output_col).reset_index()
    counts = grouped.count().rename("n_observations").reset_index()
    return result.merge(counts, on=groups, how="left", validate="one_to_one")


def rank_tools(
    df: pd.DataFrame,
    value_col: str,
    tool_col: str,
    aggregation: str = "mean",
    top_n: int = 5,
    category_col: str | None = None,
    higher_is_better: bool = True,
    rank_method: str = "min",
) -> pd.DataFrame:
    """Return tool scores and ranks globally or separately by category."""
    group_cols = [tool_col] if category_col is None else [category_col, tool_col]
    result = aggregate_scores(
        df=df,
        group_cols=group_cols,
        value_col=value_col,
        aggregation=aggregation,
        top_n=top_n,
        output_col="score",
    )
    rank_groups = [] if category_col is None else [category_col]
    if rank_groups:
        result["rank"] = result.groupby(
            rank_groups,
            dropna=False,
            sort=False,
        )["score"].rank(
            ascending=not higher_is_better,
            method=rank_method,
        )
        result = result.sort_values(
            rank_groups + ["rank", tool_col],
            kind="stable",
        )
    else:
        result["rank"] = result["score"].rank(
            ascending=not higher_is_better,
            method=rank_method,
        )
        result = result.sort_values(["rank", tool_col], kind="stable")

    result["rank"] = result["rank"].astype(int)
    result["aggregation"] = aggregation
    result["top_n"] = top_n if aggregation == "topn" else pd.NA
    return result.reset_index(drop=True)
