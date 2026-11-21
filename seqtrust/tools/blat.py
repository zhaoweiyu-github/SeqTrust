from __future__ import annotations

import os
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
import pandas as pd

from seqtrust.core.grouping import to_list_cols


DEFAULT_BLAT_EXE = (
    "/home/wangyucheng/projects/Benchmark/Figs/Fig3/tools/ucsc/blat/blat"
)


PSL_COLS = [
    "matches",
    "misMatches",
    "repMatches",
    "nCount",
    "qNumInsert",
    "qBaseInsert",
    "tNumInsert",
    "tBaseInsert",
    "strand",
    "qName",
    "qSize",
    "qStart",
    "qEnd",
    "tName",
    "tSize",
    "tStart",
    "tEnd",
    "blockCount",
    "blockSizes",
    "qStarts",
    "tStarts",
]


def _as_list(x: Any) -> list[Any]:
    """Convert scalar/list-like input to list."""
    if isinstance(x, (list, tuple, set)):
        return list(x)
    return [x]


def _safe_name(x: Any) -> str:
    """
    Convert a group key to a filesystem-safe name.

    Examples
    --------
    "TACO"
        -> "TACO"

    ("TACO", "Complete")
        -> "TACO_Complete"
    """

    if isinstance(x, (tuple,list)):
        name="_".join(str(v) for v in x)
    else:
        name=str(x)

    name = re.sub(r"[^\w.\-]+", "_", name)
    name = name.strip("_")

    return name


def _filter_group(
    df: pd.DataFrame,
    group_cols: str | Sequence[str],
    group_value: Any,
) -> pd.DataFrame:
    """
    Filter dataframe by one group value.

    If group_cols is one column, group_value should usually be a scalar.
    If group_cols contains multiple columns, group_value should be a tuple
    with the same length as group_cols.
    """
    group_cols = to_list_cols(group_cols)

    if not group_cols:
        raise ValueError(
            "group_cols cannot be empty when filtering BLAT database/query groups."
        )

    missing = [c for c in group_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing group columns: {missing}")

    if len(group_cols) == 1:
        col = group_cols[0]

        if isinstance(group_value, tuple):
            if len(group_value) != 1:
                raise ValueError(
                    f"group_value={group_value!r} does not match one group column: {group_cols}"
                )
            group_value = group_value[0]

        return df[df[col] == group_value].copy()

    if not isinstance(group_value, tuple):
        raise ValueError(
            "When group_cols has multiple columns, group_value must be a tuple. "
            f"For example: ('toolA', 'Complete'). Got {group_value!r}."
        )

    if len(group_value) != len(group_cols):
        raise ValueError(
            f"group_value length must match group_cols. "
            f"Got group_value={group_value!r}, group_cols={group_cols}."
        )

    mask = pd.Series(True, index=df.index)

    for col, val in zip(group_cols, group_value):
        mask &= df[col] == val

    return df[mask].copy()


def _is_external_database(x: Any) -> bool:
    """
    Whether `database` is an external sequence collection rather than
    a group value of the evaluator dataframe.

    External databases are passed as a list / pandas Series / numpy
    ndarray of sequences, or a DataFrame containing the sequence
    column. Note: a tuple is always treated as a group value (for
    multi-column grouping), pass a list for external sequences.
    """
    return isinstance(x, (list, pd.Series, np.ndarray, pd.DataFrame))


def _database_to_df(database: Any, sequence_col: str) -> pd.DataFrame:
    """
    Convert an external database (sequence collection or DataFrame)
    into a DataFrame with `sequence_col`.
    """
    if isinstance(database, pd.DataFrame):
        if sequence_col not in database.columns:
            raise ValueError(
                f"Cannot find sequence column {sequence_col!r} in the database DataFrame."
            )
        return database[[sequence_col]].copy()

    return pd.DataFrame({sequence_col: list(database)})


def write_fasta(
    df: pd.DataFrame,
    seq_col: str = "Sequence",
    id_col: Optional[str] = None,
    fasta_path: str | Path | None = None,
) -> Path:
    """
    Write sequences in DataFrame to FASTA.

    Parameters
    ----------
    df
        Input dataframe.

    seq_col
        Sequence column.

    id_col
        Sequence ID column. If None, use dataframe index.

    fasta_path
        Output FASTA path.
    """
    if fasta_path is None:
        raise ValueError("fasta_path cannot be None.")

    if seq_col not in df.columns:
        raise ValueError(f"Cannot find sequence column: {seq_col}")

    fasta_path = Path(fasta_path)
    fasta_path.parent.mkdir(parents=True, exist_ok=True)

    with open(fasta_path, "w") as f:
        for idx, row in df.iterrows():
            seq = row[seq_col]

            if pd.isna(seq):
                continue

            seq = str(seq).upper()

            if id_col is None:
                seq_id = str(idx)
            else:
                seq_id = str(row[id_col])

            f.write(f">{seq_id}\n{seq}\n")

    return fasta_path


def run_blat(
    database_fa: str | Path,
    query_fa: str | Path,
    out_psl: str | Path,
    blat_exe: str | Path | None = None,
    min_score: int = 20,
    min_identity: float = 100,
    tile_size: int = 11,
    step_size: int = 5,
    min_match: int = 1,
    rep_match: int = 2253,
    no_head: bool = True,
    check: bool = True,
    verbose: bool = True,
) -> Path:
    """
    Run BLAT for DNA sequence comparison.

    Parameters
    ----------
    database_fa
        Target/database FASTA.

    query_fa
        Query FASTA.

    out_psl
        Output PSL file.
    """
    if blat_exe is None:
        blat_exe = DEFAULT_BLAT_EXE

    cmd = [
        str(blat_exe),
        str(database_fa),
        str(query_fa),
        str(out_psl),
        "-q=dna",
        f"-tileSize={tile_size}",
        f"-stepSize={step_size}",
        f"-minMatch={min_match}",
        f"-repMatch={rep_match}",
        f"-minIdentity={min_identity}",
        f"-minScore={min_score}",
    ]

    if no_head:
        cmd.append("-noHead")

    if verbose:
        print("Running:", " ".join(cmd))

    subprocess.run(cmd, check=check)

    return Path(out_psl)


def read_psl_nohead(psl_path: str | Path) -> pd.DataFrame:
    """Read BLAT PSL output without header."""
    psl_path = Path(psl_path)

    if (not psl_path.exists()) or psl_path.stat().st_size == 0:
        return pd.DataFrame(columns=PSL_COLS)

    return pd.read_csv(
        psl_path,
        sep="\t",
        header=None,
        names=PSL_COLS,
    )


def summarize_blat_match(
    psl_df: pd.DataFrame,
    query_df: pd.DataFrame,
    query_id_col: Optional[str] = None,
) -> dict[str, Any]:
    """
    Summarize BLAT matching result.

    Parameters
    ----------
    psl_df
        BLAT PSL result.

    query_df
        Query sequence dataframe.

    query_id_col
        Query ID column. If None, use index.
    """
    if query_id_col is None:
        total_queries = query_df.index.astype(str).nunique()
    else:
        total_queries = query_df[query_id_col].astype(str).nunique()

    if len(psl_df) == 0:
        return {
            "n_query_total": int(total_queries),
            "n_query_matched": 0,
            "match_ratio": 0.0,
            "n_alignments": 0,
        }

    matched_queries = psl_df["qName"].astype(str).nunique()

    return {
        "n_query_total": int(total_queries),
        "n_query_matched": int(matched_queries),
        "match_ratio": (
            float(matched_queries / total_queries)
            if total_queries > 0
            else np.nan
        ),
        "n_alignments": int(len(psl_df)),
    }


def compare_sequence_with_blat(
    database_df: pd.DataFrame,
    query_df: pd.DataFrame,
    database_seq_col: str = "Sequence",
    query_seq_col: str = "Sequence",
    database_id_col: Optional[str] = None,
    query_id_col: Optional[str] = None,
    output_dir: str | Path | None = None,
    prefix: str = "blat",
    blat_exe: str | Path | None = None,
    min_score: int = 20,
    min_identity: float = 100,
    tile_size: int = 11,
    step_size: int = 5,
    min_match: int = 1,
    rep_match: int = 2253,
    keep_fasta: bool = True,
    verbose: bool = True,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Compare query sequences against database sequences using BLAT.

    Returns
    -------
    psl_df
        Raw BLAT alignment result.

    summary
        Match summary.
    """
    if blat_exe is None:
        blat_exe = DEFAULT_BLAT_EXE

    if database_seq_col not in database_df.columns:
        raise ValueError(f"Cannot find database_seq_col: {database_seq_col}")

    if query_seq_col not in query_df.columns:
        raise ValueError(f"Cannot find query_seq_col: {query_seq_col}")

    database_fa = output_dir / f"{prefix}.database.fa"
    query_fa = output_dir / f"{prefix}.query.fa"
    out_psl = output_dir / f"{prefix}.psl"

    database_df = database_df.copy().reset_index(drop=True)
    query_df = query_df.copy().reset_index(drop=True)

    database_df = database_df.dropna(subset=[database_seq_col]).copy()
    query_df = query_df.dropna(subset=[query_seq_col]).copy()

    database_df[database_seq_col] = database_df[database_seq_col].astype(str).str.upper()
    query_df[query_seq_col] = query_df[query_seq_col].astype(str).str.upper()

    if database_id_col is None:
        database_df["_seqtrust_database_id"] = [
            f"db_{i}" for i in range(len(database_df))
        ]
        database_id_col = "_seqtrust_database_id"

    if query_id_col is None:
        query_df["_seqtrust_query_id"] = [
            f"query_{i}" for i in range(len(query_df))
        ]
        query_id_col = "_seqtrust_query_id"

    write_fasta(
        database_df,
        seq_col=database_seq_col,
        id_col=database_id_col,
        fasta_path=database_fa,
    )

    write_fasta(
        query_df,
        seq_col=query_seq_col,
        id_col=query_id_col,
        fasta_path=query_fa,
    )

    run_blat(
        database_fa=database_fa,
        query_fa=query_fa,
        out_psl=out_psl,
        blat_exe=blat_exe,
        min_score=min_score,
        min_identity=min_identity,
        tile_size=tile_size,
        step_size=step_size,
        min_match=min_match,
        rep_match=rep_match,
        verbose=verbose,
    )

    psl_df = read_psl_nohead(out_psl)

    summary = summarize_blat_match(
        psl_df=psl_df,
        query_df=query_df,
        query_id_col=query_id_col,
    )

    summary.update(
        {
            "database_fa": str(database_fa),
            "query_fa": str(query_fa),
            "psl_path": str(out_psl),
            "min_score": min_score,
            "min_identity": min_identity,
            "tile_size": tile_size,
            "step_size": step_size,
            "min_match": min_match,
            "rep_match": rep_match,
        }
    )

    if not keep_fasta:
        for path in [database_fa, query_fa]:
            if path.exists():
                path.unlink()

    return psl_df, summary


def blat_group_match(
    df: pd.DataFrame,
    database: Any,
    query_group: Any | Sequence[Any] | None = None,
    sequence_col: str | None = None,
    group_cols: str | Sequence[str] | None = None,
    database_name: str | None = None,
    output_dir: str | Path | None = None,
    blat_exe: str | Path | None = None,
    min_score: int = 20,
    min_identity: float = 100,
    tile_size: int = 11,
    step_size: int = 5,
    min_match: int = 1,
    rep_match: int = 2253,
    keep_fasta: bool = True,
    # store_psl: bool = True,
    save_summary: bool = True,
    summary_filename: str = "blat_group_summary.csv",
    verbose: bool = True,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """
    Compare one or multiple query groups against one database using BLAT.

    Parameters
    ----------
    df
        Input dataframe.

    database
        The BLAT database. Either one group of `df`:

        If group_cols has one column, examples:
            database="Natural"

        If group_cols has multiple columns, examples:
            database=("Natural", "Complete")

        ... or an external collection of sequences used directly as the
        database: a list / pandas Series / numpy ndarray of sequences,
        or a DataFrame containing `sequence_col`.

    query_group
        One query group or a list of query groups. If None, every group
        in `df` is matched against the database (the database group
        itself is excluded when `database` is a group of `df`).

        If group_cols has one column, examples:
            query_group="TACO"
            query_group=["TACO", "AdaLead"]

        If group_cols has multiple columns, examples:
            query_group=("TACO", "Complete")
            query_group=[("TACO", "Complete"), ("AdaLead", "Complete")]

    sequence_col
        Sequence column name.

    group_cols
        Group column name or multiple group columns.

    database_name
        Label used for the database in the summary table and output
        file names. If None, use the group name, or "external" when
        `database` is an external sequence collection.

    output_dir
        Directory to save FASTA, PSL and summary files.

    min_score
        BLAT -minScore threshold.

    min_identity
        BLAT -minIdentity threshold.

    Returns
    -------
    summary_df
        One row per query group.

    psl_results
        Dict: {query_group_name: psl_df}
    """
    group_cols = to_list_cols(group_cols)

    if not group_cols:
        raise ValueError(
            "group_cols is required for blat_group_match. "
            "Please provide group_cols or set evaluator.group_cols."
        )

    if sequence_col is None:
        raise ValueError(
            "sequence_col is required for blat_group_match. "
            "Please provide sequence_col or set evaluator.sequence_col."
        )

    missing = [c for c in [sequence_col] + group_cols if c not in df.columns]

    if missing:
        raise ValueError(f"Missing columns: {missing}")

    if output_dir is None:
        output_dir = tempfile.mkdtemp(prefix="seqtrust_blat_group_")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Resolve the database: either one group of `df`, or an external
    # collection of sequences (list / Series / ndarray / DataFrame).
    if _is_external_database(database):
        database_df = _database_to_df(database, sequence_col)
        database_name = _safe_name(database_name) if database_name else "external"
        database_group_value = None
    else:
        database_df = _filter_group(
            df=df,
            group_cols=group_cols,
            group_value=database,
        )
        database_name = (
            _safe_name(database_name) if database_name else _safe_name(database)
        )
        database_group_value = database

    if database_df.empty:
        raise ValueError(f"No sequences found for database={database_name!r}.")

    # Default: match every group in df against the database (excluding
    # the database group itself when `database` is a group of `df`).
    if query_group is None:
        if len(group_cols) == 1:
            query_group = list(df[group_cols[0]].unique())
        else:
            query_group = list(
                df[group_cols].drop_duplicates().itertuples(index=False, name=None)
            )
        if database_group_value is not None:
            query_group = [q for q in query_group if q != database_group_value]

    summaries: list[dict[str, Any]] = []
    # psl_results: dict[str, pd.DataFrame] = {}

    # select query_df based on group_cols and query_group
    for query in _as_list(query_group):
        query_df = _filter_group(
            df=df,
            group_cols=group_cols,
            group_value=query,
        )

        if query_df.empty:
            raise ValueError(f"No rows found for query_group={query!r}.")

        query_name = _safe_name(query)

        prefix = f"{query_name}_vs_{database_name}"

        _, summary = compare_sequence_with_blat(
            database_df=database_df,
            query_df=query_df,
            database_seq_col=sequence_col,
            query_seq_col=sequence_col,
            output_dir=output_dir,
            prefix=prefix,
            blat_exe=blat_exe,
            min_score=min_score,
            min_identity=min_identity,
            tile_size=tile_size,
            step_size=step_size,
            min_match=min_match,
            rep_match=rep_match,
            keep_fasta=keep_fasta,
            verbose=verbose,
        )

        summary.update(
            {
                "database": database_name,
                "query_group": query_name,
                "database_n": int(len(database_df)),
                "query_n": int(len(query_df)),
                "output_dir": str(output_dir),
            }
        )

        summaries.append(summary)

        # if store_psl:
        #     psl_results[query_name] = psl_df

    summary_df = pd.DataFrame(summaries)

    front_cols = [
        "database",
        "query_group",
        "database_n",
        "query_n",
        "n_query_total",
        "n_query_matched",
        "match_ratio",
        "n_alignments",
        "min_score",
        "min_identity",
        "psl_path",
        "database_fa",
        "query_fa",
        "output_dir",
    ]

    existing_front_cols = [c for c in front_cols if c in summary_df.columns]
    other_cols = [c for c in summary_df.columns if c not in existing_front_cols]
    summary_df = summary_df[existing_front_cols + other_cols]

    if save_summary:
        summary_path = output_dir / summary_filename
        summary_df.to_csv(summary_path, index=False)
        summary_df.attrs["summary_path"] = str(summary_path)

    return summary_df
