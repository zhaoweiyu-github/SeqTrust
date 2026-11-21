from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence

import pandas as pd


def to_list_cols(cols: str | Sequence[str] | None) -> list[str]:
    if cols is None:
        return []
    if isinstance(cols, str):
        return [cols]
    return list(cols)


def group_key_to_name(key: Any) -> str:
    if isinstance(key, tuple):
        return "|".join(map(str, key))
    return str(key)


def iter_groups(
    df: pd.DataFrame,
    group_cols: str | Sequence[str] | None,
    groups: Iterable[Any] | None = None,
):
    """Yield ``(group_key, group_name, group_df)`` from a dataframe.

    ``groups`` can contain scalar values for one group column or tuples for multiple
    group columns. When no group columns are provided, the full dataframe is yielded
    as one group named ``all``.
    """
    group_cols = to_list_cols(group_cols)
    if not group_cols:
        yield "all", "all", df
        return

    missing = [c for c in group_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing group columns: {missing}")

    group_set = None if groups is None else set(groups)
    for key, sub_df in df.groupby(group_cols, dropna=False, sort=False):
        if len(group_cols) == 1 and isinstance(key, tuple):
            key = key[0]
        if group_set is not None and key not in group_set:
            continue
        yield key, group_key_to_name(key), sub_df


def split_groups(
    df: pd.DataFrame,
    group_cols: str | Sequence[str] | None,
    groups: Iterable[Any] | None = None,
) -> dict[str, pd.DataFrame]:
    return {name: sub_df.copy() for _, name, sub_df in iter_groups(df, group_cols, groups)}


def resolve_group_cols(
    explicit: str | Sequence[str] | None,
    default: list[str],
) -> list[str]:
    cols = to_list_cols(explicit)
    return cols if cols else list(default)
