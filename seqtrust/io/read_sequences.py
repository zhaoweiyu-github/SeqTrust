from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd


DNA_ALPHABET = set("ACGTN")


def _standardize_sequence(s: str, uppercase: bool = True) -> str:
    s = str(s).strip()
    return s.upper() if uppercase else s


def read_fasta(path: str | Path, sequence_col: str = "Sequence", id_col: str = "sequence_id") -> pd.DataFrame:
    """Read FASTA/FA file into a DataFrame."""
    path = Path(path)
    records = []
    current_id = None
    chunks = []

    with path.open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if current_id is not None:
                    records.append({id_col: current_id, sequence_col: "".join(chunks)})
                current_id = line[1:].strip() or f"seq_{len(records)}"
                chunks = []
            else:
                chunks.append(line.strip())

    if current_id is not None:
        records.append({id_col: current_id, sequence_col: "".join(chunks)})

    return pd.DataFrame(records)


def read_txt_sequences(path: str | Path, sequence_col: str = "Sequence") -> pd.DataFrame:
    """
    Read plain text sequence files.

    Supported formats:
    1. One sequence per line.
    2. FASTA-like blocks where sequence records may be separated by '>' lines.
    """
    path = Path(path)
    lines = [x.strip() for x in path.read_text().splitlines() if x.strip()]

    if any(x.startswith(">") for x in lines):
        records = []
        chunks = []
        for line in lines:
            if line.startswith(">"):
                if chunks:
                    records.append("".join(chunks))
                    chunks = []
            else:
                chunks.append(line)
        if chunks:
            records.append("".join(chunks))
        return pd.DataFrame({sequence_col: records})

    return pd.DataFrame({sequence_col: lines})


def read_sequence_file(
    path: str | Path,
    sequence_col: str = "Sequence",
    sep: Optional[str] = None,
    uppercase: bool = True,
    drop_empty: bool = True,
) -> pd.DataFrame:
    """Read CSV/TSV/FASTA/TXT and return a DataFrame containing sequence_col."""
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix in {".fa", ".fasta", ".fna"}:
        df = read_fasta(path, sequence_col=sequence_col)
    elif suffix == ".csv":
        df = pd.read_csv(path)
    elif suffix in {".tsv", ".tab"}:
        df = pd.read_csv(path, sep="\t")
    elif suffix in {".txt", ".seq"}:
        df = read_txt_sequences(path, sequence_col=sequence_col)
    else:
        # fallback: infer delimiter using pandas
        df = pd.read_csv(path, sep=sep)

    if sequence_col not in df.columns:
        raise ValueError(
            f"sequence_col='{sequence_col}' not found. Available columns: {list(df.columns)}"
        )

    if drop_empty:
        df = df[df[sequence_col].notna()].copy()
    else:
        df = df.copy()

    df[sequence_col] = df[sequence_col].map(
        lambda x: _standardize_sequence(x, uppercase=uppercase)
    )

    if drop_empty:
        df = df[df[sequence_col].str.len() > 0].reset_index(drop=True)

    return df
