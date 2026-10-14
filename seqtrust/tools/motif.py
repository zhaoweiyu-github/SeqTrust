import numpy as np
import pandas as pd

from typing import List, Tuple, Optional
from collections import defaultdict
from tqdm import tqdm

from seqtrust import config


def read_meme(
    meme_path: str,
    source: str = "HOCOMOCO",
):
    """
    Read motifs from standard MEME format file.

    Parameters
    ----------
    meme_path : str
        Path to MEME motif file.

    source : str
        "HOCOMOCO" or "JASPAR".

    Returns
    -------
    motifs : list
        List of pymemesuite Motif objects.

    motif_names : list
        List of motif names.
    """

    from pymemesuite.common import MotifFile

    print(f"Reading MEME file: {meme_path}")

    with MotifFile(meme_path) as motif_file:
        motifs = list(motif_file)

    motif_names = []

    for motif in motifs:
        if source.upper() == "JASPAR":
            name = motif.name
        else:
            name = motif.accession

        if isinstance(name, bytes):
            name = name.decode()

        motif_names.append(name)

    print(f"Successfully read {len(motifs)} motifs from {source}")

    duplicated_names = set(
        [x for x in motif_names if motif_names.count(x) > 1]
    )

    if duplicated_names:
        print(f"Found duplicated motif names: {len(duplicated_names)}")
        print(list(duplicated_names)[:20])
    else:
        print("No duplicated motif names found.")

    return motifs, motif_names


def get_background(
    sequences: List[str] = None,
    use_uniform: bool = False,
):
    """
    Build background model for motif scanning.
    """

    from pymemesuite.common import Alphabet, Background, Array

    if use_uniform:
        bg_array = np.array([0.25, 0.25, 0.25, 0.25], dtype=np.float64)
        print("Using uniform background: A=0.25, C=0.25, G=0.25, T=0.25")

    else:
        if sequences is None or len(sequences) == 0:
            raise ValueError("sequences must be provided when use_uniform=False")

        counts = {
            "A": 0,
            "C": 0,
            "G": 0,
            "T": 0,
        }

        total = 0

        for seq in sequences:
            seq = str(seq).upper()
            for nuc in seq:
                if nuc in counts:
                    counts[nuc] += 1
                    total += 1

        if total == 0:
            raise ValueError("No valid A/C/G/T nucleotides found.")

        bg_array = np.array(
            [
                counts["A"],
                counts["C"],
                counts["G"],
                counts["T"],
            ],
            dtype=np.float64,
        ) / total

        print(f"Computed background from {len(sequences)} sequences:")
        print(
            f"Background: "
            f"A={bg_array[0]:.4f}, "
            f"C={bg_array[1]:.4f}, "
            f"G={bg_array[2]:.4f}, "
            f"T={bg_array[3]:.4f}, "
            f"GC={(bg_array[1] + bg_array[2]):.4f}"
        )

    bg_pymeme = Array(bg_array.tolist())
    background = Background(Alphabet.dna(), bg_pymeme)

    return bg_array, background

def background_from_freq(bg_freq):
    """
    Build a pymeme Background model from user-specified nucleotide frequencies.

    Parameters
    ----------
    bg_freq : dict or array-like
        Either a dict like {"A": 0.29, "C": 0.21, "G": 0.21, "T": 0.29},
        or an array-like of 4 frequencies in A, C, G, T order.
        Values are normalized to sum to 1.

    Returns
    -------
    bg_array : np.ndarray
        Frequencies in A, C, G, T order.
    background : pymemesuite.common.Background
    """

    from pymemesuite.common import Alphabet, Background, Array

    if isinstance(bg_freq, dict):
        bg_array = np.array(
            [bg_freq[b] for b in ("A", "C", "G", "T")],
            dtype=np.float64,
        )
    else:
        bg_array = np.asarray(list(bg_freq), dtype=np.float64)

    if bg_array.shape != (4,):
        raise ValueError(
            "bg_freq must contain exactly 4 frequencies (A, C, G, T)."
        )

    if np.any(bg_array < 0):
        raise ValueError("bg_freq values must be non-negative.")

    total = bg_array.sum()
    if total <= 0:
        raise ValueError("bg_freq values must sum to a positive value.")

    # Normalize so frequencies sum to 1
    bg_array = bg_array / total

    print(
        f"Using user-specified background: "
        f"A={bg_array[0]:.4f}, "
        f"C={bg_array[1]:.4f}, "
        f"G={bg_array[2]:.4f}, "
        f"T={bg_array[3]:.4f}, "
        f"GC={(bg_array[1] + bg_array[2]):.4f}"
    )

    bg_pymeme = Array(bg_array.tolist())
    background = Background(Alphabet.dna(), bg_pymeme)

    return bg_array, background


def scan_with_fimo(
    seq_df: pd.DataFrame,
    motifs,
    motif_names,
    bg,
    threshold: float = 1e-4,
    show_progress: bool = False,
):
    """
    Scan sequences for motif hits using FIMO.
    """

    from pymemesuite.common import Sequence
    from pymemesuite.fimo import FIMO

    if "Sequence" not in seq_df.columns:
        raise ValueError("seq_df must contain a 'Sequence' column.")

    sequences = [
        Sequence(str(seq).upper(), name=str(idx).encode())
        for idx, seq in zip(seq_df.index, seq_df["Sequence"])
    ]

    fimo = FIMO(
        both_strands=True,
        threshold=threshold,
    )

    d = defaultdict(list)

    motif_iter = tqdm(
        motifs,
        desc="Scanning motifs",
    ) if show_progress else motifs

    for idx, motif in enumerate(motif_iter):
        motif_id = motif_names[idx]

        match = fimo.score_motif(
            motif,
            sequences,
            bg,
        ).matched_elements

        for m in match:
            d["Matrix_id"].append(motif_id)
            d["ID"].append(m.source.accession.decode())
            d["strand"].append(m.strand)
            d["score"].append(m.score)
            d["pval"].append(m.pvalue)
            d["qval"].append(m.qvalue)

            if m.strand == "-":
                d["start"].append(m.stop - 1)
                d["end"].append(m.start)
            else:
                d["start"].append(m.start - 1)
                d["end"].append(m.stop)

    if len(d["ID"]) == 0:
        return pd.DataFrame(
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
        ).set_index("ID")

    return pd.DataFrame(d).set_index("ID")


def scan_motifs(
    df: pd.DataFrame,
    motifs,
    motif_names,
    sequence_col: str = "Sequence",
    bg=None,
    bg_freq=None,
    threshold: float = 1e-4,
    use_uniform_bg: bool = False,
    show_progress: bool = False,
):
    """
    Scan motifs from a sequence DataFrame.

    Parameters
    ----------
    bg
        Pre-built pymemesuite Background object. Takes precedence over
        bg_freq when provided.
    bg_freq
        User-specified background nucleotide frequencies: a dict keyed by
        A/C/G/T, or 4 values in A, C, G, T order. Only used when bg is
        None. When both are None, the background is computed from the
        input sequences (or uniform if use_uniform_bg=True).

    Returns
    -------
    motif_scans : pd.DataFrame
        FIMO scan results.
    """

    df_scan = df.copy()

    if sequence_col != "Sequence":
        df_scan = df_scan.rename(columns={sequence_col: "Sequence"})

    if "Sequence" not in df_scan.columns:
        raise ValueError(f"Cannot find sequence column: {sequence_col}")

    df_scan = df_scan.reset_index(drop=True)
    df_scan.index = df_scan.index.astype(str)
    df_scan.index.name = "ID"

    all_sequences = df_scan["Sequence"].astype(str).str.upper().tolist()

    if bg is None:
        if bg_freq is not None:
            _, bg = background_from_freq(bg_freq)
        else:
            _, bg = get_background(
                sequences=all_sequences,
                use_uniform=use_uniform_bg,
            )

    motif_scans = scan_with_fimo(
        seq_df=df_scan[["Sequence"]],
        motifs=motifs,
        motif_names=motif_names,
        bg=bg,
        threshold=threshold,
        show_progress=show_progress,
    )

    return motif_scans


def get_scan_df(
    df: pd.DataFrame,
    motifs,
    motif_names,
    sequence_col: str = "Sequence",
    threshold: float = 1e-4,
    qval_cutoff: float = 0.05,
    pval_cutoff: float = 1e-4,
    use_uniform_bg: bool = False,
    bg_freq=None,
    show_progress: bool = False,
    output_type: str = "binary",
    return_scan: bool = True,
):
    """
    Generate sequence-by-motif feature matrix.

    Parameters
    ----------
    bg_freq : dict or array-like, optional
        User-specified background nucleotide frequencies (dict keyed by
        A/C/G/T, or 4 values in A, C, G, T order). If None, the background
        is computed from the input sequences (or uniform when
        use_uniform_bg=True).
    output_type : str
        - "binary": whether each sequence has at least one hit for motif
        - "count": number of hits for motif in each sequence
        - "score_max": maximum FIMO score for motif in each sequence

    Returns
    -------
    feature_df : pd.DataFrame
        Rows are sequences, columns are motifs.
    """

    motif_scans = scan_motifs(
        df=df,
        motifs=motifs,
        motif_names=motif_names,
        sequence_col=sequence_col,
        threshold=threshold,
        bg_freq=bg_freq,
        use_uniform_bg=use_uniform_bg,
        show_progress=show_progress,
    )

    motif_scans = motif_scans[
        (motif_scans["qval"] < qval_cutoff)
        & (motif_scans["pval"] < pval_cutoff)
    ].copy()

    motif_scans = motif_scans.reset_index()

    n = len(df)

    feature_df = pd.DataFrame(
        0,
        index=[str(i) for i in range(n)],
        columns=motif_names,
        dtype=np.float32,
    )

    if len(motif_scans) == 0:
        return feature_df

    output_type = output_type.lower()

    if output_type == "binary":
        hits = (
            motif_scans
            .drop_duplicates(["ID", "Matrix_id"])
        )

        for _, row in hits.iterrows():
            feature_df.loc[row["ID"], row["Matrix_id"]] = 1.0

    elif output_type == "count":
        counts = (
            motif_scans
            .groupby(["ID", "Matrix_id"])
            .size()
            .reset_index(name="count")
        )

        for _, row in counts.iterrows():
            feature_df.loc[row["ID"], row["Matrix_id"]] = float(row["count"])

    elif output_type == "score_max":
        scores = (
            motif_scans
            .groupby(["ID", "Matrix_id"])["score"]
            .max()
            .reset_index()
        )

        for _, row in scores.iterrows():
            feature_df.loc[row["ID"], row["Matrix_id"]] = float(row["score"])

    else:
        raise ValueError(
            "output_type must be one of: 'binary', 'count', 'score_max'."
        )

    if return_scan:
        return feature_df, motif_scans

    return feature_df



def calculate_motif_features(
    df: pd.DataFrame,
    meme_path: str | None = None,
    motif_clusters: str | None = None,
    source: str = "HOCOMOCO",
    sequence_col: str = "Sequence",
    threshold: float = 1e-4,
    qval_cutoff: float = 0.05,
    pval_cutoff: float = 1e-4,
    use_uniform_bg: bool = False,
    bg_freq=None,
    show_progress: bool = False,
    output_type: str = "binary",
    feature_col: str = "motif_features",
    return_scan: bool = True,
):
    """Calculate per-sequence motif features for the given dataframe.

    ``bg_freq`` optionally provides user-specified background
    nucleotide frequencies (dict keyed by A/C/G/T, or 4 values in
    A, C, G, T order).

    Returns the dataframe with a new feature column, the motif names,
    and the positional FIMO scan table.
    """

    if meme_path is None:
        meme_path = config.DEFAULT_MEME_PATH

    if motif_clusters is None:
        motif_clusters = config.DEFAULT_CLUSTER_PATH

    df_out = df.copy()

    motifs, motif_names = read_meme(
        meme_path=meme_path,
        source=source,
    )

    motif_feature_df, motif_scan_df = get_scan_df(
        df=df_out,
        motifs=motifs,
        motif_names=motif_names,
        sequence_col=sequence_col,
        threshold=threshold,
        qval_cutoff=qval_cutoff,
        pval_cutoff=pval_cutoff,
        use_uniform_bg=use_uniform_bg,
        bg_freq=bg_freq,
        show_progress=show_progress,
        output_type=output_type,
    )

    motif_feature_df = motif_feature_df.reset_index(drop=True)
    df_out = df_out.reset_index(drop=True)

    features = motif_feature_df.to_numpy(dtype=np.float32)
    df_out[feature_col] = list(features)

    if return_scan:
        return (
            df_out,
            list(motif_names),
            motif_scan_df
        )

    return (
        df_out,
        list(motif_names)
    )
