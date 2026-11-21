import pandas as pd
import numpy as np


def calculate_gc(df: pd.DataFrame, sequence_col: str, output_col: str = "GC_content",) -> pd.DataFrame:
    seqs = df[sequence_col].astype(str)
    gc = (
        seqs.str.count("G")
        +
        seqs.str.count("C")
    )
    length = seqs.str.len()
    df[output_col] = gc / length
    return df



def calculate_complexity(df: pd.DataFrame, sequence_col: str, output_col: str = "complexity_score",) -> pd.DataFrame:

    sequences = df[sequence_col].astype(str)

    scores = []

    for seq in sequences:

        counts = {
            "A": seq.count("A"),
            "C": seq.count("C"),
            "G": seq.count("G"),
            "T": seq.count("T"),
        }

        length = sum(counts.values())

        if length == 0:
            scores.append(float("nan"))
            continue


        entropy = 0.0

        for count in counts.values():

            if count > 0:
                p = count / length
                entropy -= p * np.log2(p)


        # normalize maximum entropy log2(4)=2
        scores.append(
            entropy / 2
        )


    df[output_col] = scores

    return df


def calculate_max_repeat(df: pd.DataFrame, sequence_col: str, output_col: str = "max_repeat_length",) -> pd.DataFrame:

    sequences = df[sequence_col].astype(str)

    repeat_lengths = []

    for seq in sequences:

        if len(seq) == 0:
            repeat_lengths.append(0)
            continue

        max_run = 1
        current_run = 1

        previous = seq[0]

        for base in seq[1:]:

            if base == previous:
                current_run += 1
            else:
                max_run = max(
                    max_run,
                    current_run
                )
                current_run = 1
                previous = base

        max_run = max(
            max_run,
            current_run
        )

        repeat_lengths.append(max_run)


    df[output_col] = repeat_lengths

    return df


def calculate_max_tandem_repeat(df: pd.DataFrame, sequence_col: str, output_col: str = "max_tandem_repeat_length", min_unit_size: int = 2, min_copies: int = 2,) -> pd.DataFrame:
    """
    Longest substring made of >= min_copies consecutive copies
    of a unit whose length is >= min_unit_size.

    For a candidate unit size p, positions with seq[i] == seq[i + p]
    form runs; a run of length L corresponds to a tandem repeat tract
    of length L + p, i.e. (L + p) / p copies of the unit.
    """

    sequences = df[sequence_col].astype(str)

    tandem_lengths = []

    for seq in sequences:

        n = len(seq)

        # need at least min_copies complete copies of the smallest unit
        if n < min_unit_size * min_copies:
            tandem_lengths.append(0)
            continue

        arr = np.array(list(seq))

        best = 0

        for p in range(min_unit_size, n // min_copies + 1):

            matches = arr[:-p] == arr[p:]

            # lengths of consecutive True runs in `matches`
            changes = np.diff(
                np.concatenate(([False], matches, [False])).astype(np.int8)
            )
            run_starts = np.where(changes == 1)[0]
            run_ends = np.where(changes == -1)[0]

            if run_starts.size == 0:
                continue

            max_run = int((run_ends - run_starts).max())

            # tract length = max_run + p requires >= min_copies copies,
            # i.e. max_run >= p * (min_copies - 1)
            if max_run >= p * (min_copies - 1):
                best = max(best, max_run + p)

        tandem_lengths.append(best)


    df[output_col] = tandem_lengths

    return df


def calculate_max_dispersed_repeat(df: pd.DataFrame, sequence_col: str, output_col: str = "max_dispersed_repeat_count", unit_size: int = 3,) -> pd.DataFrame:
    """
    Maximum number of (possibly non-adjacent) occurrences of any
    unit of length `unit_size` within each sequence.

    Occurrences are counted without overlap (like str.count), i.e.
    as disjoint copies. Unlike `calculate_max_tandem_repeat`, the
    copies do NOT need to be positionally consecutive.
    """

    sequences = df[sequence_col].astype(str)

    repeat_counts = []

    for seq in sequences:

        n = len(seq)

        if n < unit_size:
            repeat_counts.append(0)
            continue

        units = {seq[i:i + unit_size] for i in range(n - unit_size + 1)}

        # str.count is non-overlapping, so copies are disjoint
        repeat_counts.append(max(seq.count(u) for u in units))

    df[output_col] = repeat_counts

    return df


def calculate_kmer_diversity(df: pd.DataFrame, sequence_col: str, output_col: str = "kmer_diversity", kmer_size: int = 3, normalize: bool = True,) -> pd.DataFrame:
    """
    Shannon entropy of the k-mer frequency spectrum of each sequence.

    K-mers are collected with an overlapping sliding window.
    High entropy means the sequence is built from diverse units;
    low entropy means a few k-mers dominate (repetitive sequence).
    When `normalize` is True, entropy is divided by the achievable
    maximum log2(min(4**kmer_size, num_windows)), giving [0, 1].
    """

    sequences = df[sequence_col].astype(str)

    scores = []

    for seq in sequences:

        n = len(seq)
        num_windows = n - kmer_size + 1

        if num_windows <= 0:
            scores.append(float("nan"))
            continue

        counts = {}

        for i in range(num_windows):
            kmer = seq[i:i + kmer_size]
            counts[kmer] = counts.get(kmer, 0) + 1

        entropy = 0.0

        for count in counts.values():
            p = count / num_windows
            entropy -= p * np.log2(p)

        if normalize:
            max_entropy = np.log2(min(4 ** kmer_size, num_windows))
            entropy = entropy / max_entropy if max_entropy > 0 else 0.0

        scores.append(entropy)

    df[output_col] = scores

    return df