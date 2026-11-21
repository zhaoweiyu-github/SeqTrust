from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd


DEFAULT_PARAMS = {
    "tau": 0.75,
    "pseudo_overlap_min": 6,
    "overlap_min": 4,
    "uniq_k": 6,
    "uniq_min_match": 5,
    "fraction_1": 0.75,
    "fraction_2": 0.75,
    "min_motif_ov": 1,
    "single_base_ratio": 1.0,
    "comb_min_gap": 1,
    "comb_max_gap": 20,
    "bridge_frac": 0.8,
    "min_seqlets": 3,
}

COLUMN_GROUPS = {
    "positive": [
        "n_pos_seqlets", "pos_seqlets_coverage", "pos_seqlets_span",
        "pos_seqlets_pair_uniq", "pos_seqlets_nonred_partner_num",
        "pos_seqlets_diversity",
    ],
    "augmented": [
        "n_motif_seqlets_added", "n_pos_seqlets_aug",
        "pos_seqlets_coverage_aug", "pos_seqlets_span_aug",
        "pos_seqlets_pair_uniq_aug", "pos_seqlets_nonred_partner_num_aug",
        "pos_seqlets_diversity_aug",
    ],
    "valid_raw": [
        "n_valid_pos_seqlets", "valid_pos_seqlets_coverage",
        "valid_pos_seqlets_span", "valid_pos_seqlets_pair_uniq",
        "valid_pos_seqlets_nonred_partner_num", "valid_pos_seqlets_diversity",
    ],
    "valid_aug": [
        "n_valid_pos_seqlets_aug", "valid_pos_seqlets_coverage_aug",
        "valid_pos_seqlets_span_aug", "valid_pos_seqlets_pair_uniq_aug",
        "valid_pos_seqlets_nonred_partner_num_aug",
        "valid_pos_seqlets_diversity_aug",
    ],
    "matched_motif": [
        "n_pos_seqlet_with_matched_motif",
        "n_pos_seqlet_with_matched_motif_aug",
        "n_valid_pos_seqlet_with_matched_motif_aug",
        "single_base_matched_motif", "single_base_matched_motif_aug",
        "valid_single_base_matched_motif_aug",
    ],
    "unique_matched_features": [
        "n_unique_matched_features", "n_unique_matched_features_aug",
        "n_valid_unique_matched_features_aug",
    ],
    "sr_matched_motif": [
        "sr_seqlet_with_matched_motif", "sr_seqlet_with_matched_motif_aug",
        "valid_sr_seqlet_with_matched_motif_aug",
    ],
    "combinatorial_motif_pairs": [
        "n_combinatorial_seqlets_motif_pairs",
        "n_combinatorial_seqlets_motif_pairs_aug",
        "n_valid_combinatorial_seqlets_motif_pairs_aug",
    ],
    "combinatorial_all_pairs": [
        "n_combinatorial_seqlet_all_pairs",
        "n_combinatorial_seqlet_all_pairs_aug",
        "n_valid_combinatorial_seqlet_all_pairs_aug",
    ],
    "adjacent_heterotypic": [
        "adjacent_heterotypic_seqlets", "adjacent_heterotypic_seqlets_aug",
        "valid_adjacent_heterotypic_seqlets_aug",
    ],
}

_INT_COLUMNS = [
    "n_pos_seqlets", "pos_seqlets_coverage", "pos_seqlets_span",
    "n_motif_seqlets_added", "n_pos_seqlets_aug", "pos_seqlets_coverage_aug",
    "pos_seqlets_span_aug", "n_valid_pos_seqlets",
    "valid_pos_seqlets_coverage", "valid_pos_seqlets_span",
    "n_valid_pos_seqlets_aug", "valid_pos_seqlets_coverage_aug",
    "valid_pos_seqlets_span_aug", "n_pos_seqlet_with_matched_motif",
    "n_pos_seqlet_with_matched_motif_aug",
    "n_valid_pos_seqlet_with_matched_motif_aug",
]
_FLOAT_COLUMNS = [
    "pos_seqlets_pair_uniq", "pos_seqlets_pair_uniq_aug",
    "valid_pos_seqlets_pair_uniq", "valid_pos_seqlets_pair_uniq_aug",
    "pos_seqlets_nonred_partner_num", "pos_seqlets_nonred_partner_num_aug",
    "valid_pos_seqlets_nonred_partner_num",
    "valid_pos_seqlets_nonred_partner_num_aug", "n_unique_matched_features",
    "n_unique_matched_features_aug", "n_valid_unique_matched_features_aug",
    "single_base_matched_motif", "single_base_matched_motif_aug",
    "valid_single_base_matched_motif_aug", "sr_seqlet_with_matched_motif",
    "sr_seqlet_with_matched_motif_aug",
    "valid_sr_seqlet_with_matched_motif_aug",
    "n_combinatorial_seqlets_motif_pairs",
    "n_combinatorial_seqlets_motif_pairs_aug",
    "n_valid_combinatorial_seqlets_motif_pairs_aug",
    "n_combinatorial_seqlet_all_pairs",
    "n_combinatorial_seqlet_all_pairs_aug",
    "n_valid_combinatorial_seqlet_all_pairs_aug",
    "adjacent_heterotypic_seqlets", "adjacent_heterotypic_seqlets_aug",
    "valid_adjacent_heterotypic_seqlets_aug", "pos_seqlets_diversity",
    "pos_seqlets_diversity_aug", "valid_pos_seqlets_diversity",
    "valid_pos_seqlets_diversity_aug",
]
ALL_COLUMNS = _INT_COLUMNS[:13] + _FLOAT_COLUMNS[:8] + _INT_COLUMNS[13:] + _FLOAT_COLUMNS[8:]
PAIR_COLUMNS = [
    "sequence_position", "sequence_index", "set_name", "seqlet_a_start",
    "seqlet_a_end", "seqlet_b_start", "seqlet_b_end", "gap_bp",
    "kmer_redundant", "motif_max_ov_bp", "motif_redundant",
    "motif_overlap", "is_combinatorial",
]

_COUNT_COLUMNS = {
    "raw": "n_pos_seqlet_with_matched_motif",
    "aug": "n_pos_seqlet_with_matched_motif_aug",
    "valid_aug": "n_valid_pos_seqlet_with_matched_motif_aug",
}
_FEATURE_COLUMNS = {
    "raw": "n_unique_matched_features",
    "aug": "n_unique_matched_features_aug",
    "valid_aug": "n_valid_unique_matched_features_aug",
}
_SINGLE_BASE_COLUMNS = {
    "raw": "single_base_matched_motif",
    "aug": "single_base_matched_motif_aug",
    "valid_aug": "valid_single_base_matched_motif_aug",
}
_SELF_REDUNDANT_COLUMNS = {
    "raw": "sr_seqlet_with_matched_motif",
    "aug": "sr_seqlet_with_matched_motif_aug",
    "valid_aug": "valid_sr_seqlet_with_matched_motif_aug",
}
_COMB_MOTIF_COLUMNS = {
    "raw": "n_combinatorial_seqlets_motif_pairs",
    "aug": "n_combinatorial_seqlets_motif_pairs_aug",
    "valid_aug": "n_valid_combinatorial_seqlets_motif_pairs_aug",
}
_COMB_ALL_COLUMNS = {
    "raw": "n_combinatorial_seqlet_all_pairs",
    "aug": "n_combinatorial_seqlet_all_pairs_aug",
    "valid_aug": "n_valid_combinatorial_seqlet_all_pairs_aug",
}
_ADJACENT_COLUMNS = {
    "raw": "adjacent_heterotypic_seqlets",
    "aug": "adjacent_heterotypic_seqlets_aug",
    "valid_aug": "valid_adjacent_heterotypic_seqlets_aug",
}


class _MotifSeqlet:
    """Seqlet-compatible representation of an attributed standalone motif."""

    def __init__(self, start, end, contrib_scores, sequence):
        self.start = int(start)
        self.end = int(end)
        self.contrib_scores = np.asarray(contrib_scores)
        self.sequence = np.asarray(sequence)
        self.own_motif = (self.start, self.end)


def _seqlet_entropy(contrib_scores) -> float:
    """Return normalized Shannon entropy of positive per-base attribution."""
    scores = np.asarray(contrib_scores)
    per_base = scores.sum(axis=1) if scores.ndim > 1 else scores
    weights = np.maximum(per_base, 0.0)
    total = weights.sum()
    if total <= 0 or len(weights) <= 1:
        return 0.0
    probabilities = weights[weights > 0] / total
    return float(-(probabilities * np.log2(probabilities)).sum() / np.log2(len(weights)))


def _coverage_span(regions) -> tuple[int, int]:
    """Return union coverage and outer span of half-open intervals."""
    regions = sorted((int(a), int(b)) for a, b in regions if b > a)
    if not regions:
        return 0, 0
    merged = [list(regions[0])]
    for start, end in regions[1:]:
        if start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return (
        sum(end - start for start, end in merged),
        max(end for _, end in regions) - min(start for start, _ in regions),
    )


def _one_hot_to_string(one_hot) -> str:
    bases = np.asarray(["A", "C", "G", "T"])
    array = np.asarray(one_hot)
    if array.ndim != 2 or array.shape[1] != 4:
        raise ValueError("Seqlet sequence arrays must have shape (length, 4).")
    return "".join(bases[np.argmax(array, axis=1)])


_RC_TABLE = str.maketrans("ACGT", "TGCA")


def _reverse_complement(sequence: str) -> str:
    return sequence.translate(_RC_TABLE)[::-1]


def _sliding_kmers(sequence: str, k: int) -> list[str]:
    sequence = sequence.upper()
    return [
        sequence[i : i + k]
        for i in range(len(sequence) - k + 1)
        if set(sequence[i : i + k]) <= set("ACGT")
    ]


def _kmer_match(first: str, second: str, min_match: int) -> bool:
    return (
        sum(a == b for a, b in zip(first, second)) >= min_match
        or sum(a == b for a, b in zip(first, _reverse_complement(second))) >= min_match
    )


def _seqlets_redundant(first: str, second: str, k: int, min_match: int) -> bool:
    second_kmers = _sliding_kmers(second, k)
    return any(
        _kmer_match(a, b, min_match)
        for a in _sliding_kmers(first, k)
        for b in second_kmers
    )


def _pair_uniqueness(sequences, k: int, min_match: int) -> tuple[float, float]:
    count = len(sequences)
    if count < 2:
        return np.nan, np.nan
    redundant = sum(
        _seqlets_redundant(sequences[i], sequences[j], k, min_match)
        for i in range(count)
        for j in range(i + 1, count)
    )
    uniqueness = 1.0 - redundant / (count * (count - 1) / 2)
    return uniqueness, (count - 1) * uniqueness


def _self_redundant_pairs(sequence: str, k: int, min_match: int):
    kmers = [(i, sequence[i : i + k]) for i in range(len(sequence) - k + 1)]
    return [
        (kmers[i][0], kmers[j][0])
        for i in range(len(kmers))
        for j in range(i + 1, len(kmers))
        if _kmer_match(kmers[i][1], kmers[j][1], min_match)
    ]


def _diversity(nonredundant_partners, count: int) -> float:
    if count < 2 or pd.isna(nonredundant_partners):
        return np.nan
    return float(nonredundant_partners) / (count - 1)


def _group_seqlets(seqlets, n_sequences: int) -> dict[int, list]:
    grouped = {i: [] for i in range(n_sequences)}
    for seqlet in seqlets:
        position = int(seqlet.example_idx)
        if position < 0 or position >= n_sequences:
            raise IndexError(
                f"seqlet.example_idx={position} is outside 0..{n_sequences - 1}."
            )
        grouped[position].append(seqlet)
    return grouped


def _suppress_overlaps(grouped, overlap_min: int | None):
    """Greedily retain the strongest seqlets when overlap exceeds a limit."""
    if overlap_min is None:
        return {key: list(value) for key, value in grouped.items()}, set()
    kept_by_sequence = {}
    removed_ids = set()
    for position, seqlets in grouped.items():
        ordered = sorted(
            seqlets,
            key=lambda item: float(np.asarray(item.contrib_scores).sum()),
            reverse=True,
        )
        kept = []
        regions = []
        for seqlet in ordered:
            start, end = int(seqlet.start), int(seqlet.end)
            if any(min(end, b) - max(start, a) > overlap_min for a, b in regions):
                removed_ids.add(id(seqlet))
            else:
                kept.append(seqlet)
                regions.append((start, end))
        kept_by_sequence[position] = kept
    return kept_by_sequence, removed_ids


def _prepare_motifs(
    motif_scan: pd.DataFrame,
    motif_list,
    n_sequences: int,
    sequence_length: int,
):
    scan = motif_scan.copy()
    if "ID" not in scan.columns:
        if scan.index.name == "ID":
            scan = scan.reset_index()
        else:
            raise ValueError("motif_scan must contain an 'ID' column or ID index.")
    required = {"ID", "start", "end"}
    missing = required - set(scan.columns)
    if missing:
        raise ValueError(f"motif_scan is missing required columns: {sorted(missing)}")
    if motif_list is not None:
        if "Matrix_id" not in scan.columns:
            raise ValueError("motif_list filtering requires motif_scan.Matrix_id.")
        allowed = {str(value) for value in motif_list}
        scan = scan[scan["Matrix_id"].astype(str).isin(allowed)]
    motifs = {i: [] for i in range(n_sequences)}
    for row in scan.itertuples(index=False):
        try:
            position = int(str(row.ID))
        except ValueError as exc:
            raise ValueError("motif_scan.ID values must be DataFrame row positions.") from exc
        if position < 0 or position >= n_sequences:
            continue
        start, end = sorted((int(row.start), int(row.end)))
        start = max(0, start)
        end = min(sequence_length, end)
        if end > start:
            motifs[position].append((start, end))
    return motifs


def _build_context(position, positive, occupied, motifs, per_base, mean_attr, sequences, params):
    """Build raw and motif-augmented positive seqlet sets for one sequence."""
    cb = per_base[position]
    candidates = []
    for start, end in motifs[position]:
        start = max(0, start)
        end = min(len(cb), end)
        if end <= start:
            continue
        if any(min(end, b) - max(start, a) >= 1 for a, b in occupied):
            continue
        score = float(cb[start:end].mean())
        if score > mean_attr[position]:
            candidates.append((score, start, end))
    pseudo = []
    pseudo_regions = []
    for _, start, end in sorted(candidates, reverse=True):
        if any(
            min(end, b) - max(start, a) >= params["pseudo_overlap_min"]
            for a, b in pseudo_regions
        ):
            continue
        pseudo_regions.append((start, end))
        pseudo.append(_MotifSeqlet(start, end, cb[start:end], sequences[position, start:end]))
    return {"pos": positive, "pseudo": pseudo, "motifs": motifs[position], "cb": cb}


def _matched_motifs(seqlets, motifs, cb, fraction_1, fraction_2):
    regions, hit_lists = [], []
    for seqlet in seqlets:
        start, end = int(seqlet.start), int(seqlet.end)
        if end <= start:
            continue
        own_motif = getattr(seqlet, "own_motif", None)
        seqlet_attr = float(cb[start:end].sum())
        hits = []
        for motif_start, motif_end in motifs:
            if own_motif == (motif_start, motif_end):
                continue
            overlap_start = max(start, motif_start)
            overlap_end = min(end, motif_end)
            overlap = overlap_end - overlap_start
            if overlap <= 0:
                continue
            if max(overlap / (end - start), overlap / (motif_end - motif_start)) < fraction_1:
                continue
            overlap_attr = float(cb[overlap_start:overlap_end].sum())
            motif_attr = float(cb[motif_start:motif_end].sum())
            fractions = [overlap_attr / value for value in (seqlet_attr, motif_attr) if value > 0]
            if fractions and max(fractions) >= fraction_2:
                hits.append((motif_start, motif_end))
        regions.append((start, end))
        hit_lists.append(hits)
    return regions, hit_lists


def _unique_features(matched, k: int, min_match: int, min_motif_overlap: int) -> int:
    parents = list(range(len(matched)))

    def find(value):
        while parents[value] != value:
            parents[value] = parents[parents[value]]
            value = parents[value]
        return value

    def union(first, second):
        first, second = find(first), find(second)
        if first != second:
            parents[first] = second

    for i in range(len(matched)):
        for j in range(i + 1, len(matched)):
            redundant = _seqlets_redundant(matched[i][2], matched[j][2], k, min_match)
            shared_motif = any(
                min(a1, b1) - max(a0, b0) >= min_motif_overlap
                for a0, a1 in matched[i][1]
                for b0, b1 in matched[j][1]
            )
            if redundant or shared_motif:
                union(i, j)
    return len({find(i) for i in range(len(matched))})


def _max_motif_overlap(first, second) -> int:
    return max(
        [0]
        + [min(a1, b1) - max(a0, b0) for a0, a1 in first for b0, b1 in second]
    )


def _count_pairs(regions, hits, sequences, params, records, position, index, set_name):
    count = 0
    for i in range(len(regions)):
        for j in range(i + 1, len(regions)):
            a0, a1 = regions[i]
            b0, b1 = regions[j]
            gap = b0 - a1
            kmer_redundant = _seqlets_redundant(
                sequences[i], sequences[j], params["uniq_k"], params["uniq_min_match"]
            )
            motif_overlap_bp = _max_motif_overlap(hits[i], hits[j])
            motif_redundant = motif_overlap_bp >= params["min_motif_ov"]
            motif_overlap = bool(hits[i]) and bool(hits[j]) and motif_overlap_bp >= 1
            combinatorial = (
                not kmer_redundant
                and not motif_redundant
                and not motif_overlap
                and params["comb_min_gap"] <= gap < params["comb_max_gap"]
            )
            count += int(combinatorial)
            if records is not None:
                records.append({
                    "sequence_position": position, "sequence_index": index,
                    "set_name": set_name, "seqlet_a_start": a0,
                    "seqlet_a_end": a1, "seqlet_b_start": b0,
                    "seqlet_b_end": b1, "gap_bp": gap,
                    "kmer_redundant": kmer_redundant,
                    "motif_max_ov_bp": motif_overlap_bp,
                    "motif_redundant": motif_redundant,
                    "motif_overlap": motif_overlap,
                    "is_combinatorial": combinatorial,
                })
    return float(count)


def _set_summary(row, prefix, seqlets, params):
    regions = [(int(item.start), int(item.end)) for item in seqlets]
    sequences = [_one_hot_to_string(item.sequence) for item in seqlets]
    coverage, span = _coverage_span(regions)
    uniqueness, partners = _pair_uniqueness(
        sequences, params["uniq_k"], params["uniq_min_match"]
    )
    if prefix == "raw":
        row.update({
            "n_pos_seqlets": len(seqlets), "pos_seqlets_coverage": coverage,
            "pos_seqlets_span": span, "pos_seqlets_pair_uniq": uniqueness,
            "pos_seqlets_nonred_partner_num": partners,
            "pos_seqlets_diversity": _diversity(partners, len(seqlets)),
        })
    elif prefix == "aug":
        row.update({
            "n_pos_seqlets_aug": len(seqlets), "pos_seqlets_coverage_aug": coverage,
            "pos_seqlets_span_aug": span, "pos_seqlets_pair_uniq_aug": uniqueness,
            "pos_seqlets_nonred_partner_num_aug": partners,
            "pos_seqlets_diversity_aug": _diversity(partners, len(seqlets)),
        })
    elif prefix == "valid":
        row.update({
            "n_valid_pos_seqlets": len(seqlets), "valid_pos_seqlets_coverage": coverage,
            "valid_pos_seqlets_span": span, "valid_pos_seqlets_pair_uniq": uniqueness,
            "valid_pos_seqlets_nonred_partner_num": partners,
            "valid_pos_seqlets_diversity": _diversity(partners, len(seqlets)),
        })
    else:
        row.update({
            "n_valid_pos_seqlets_aug": len(seqlets),
            "valid_pos_seqlets_coverage_aug": coverage,
            "valid_pos_seqlets_span_aug": span,
            "valid_pos_seqlets_pair_uniq_aug": uniqueness,
            "valid_pos_seqlets_nonred_partner_num_aug": partners,
            "valid_pos_seqlets_diversity_aug": _diversity(partners, len(seqlets)),
        })


def _sequence_metrics(ctx, sequence, params, motif_records, all_records, position, index):
    positive = ctx["pos"]
    augmented = positive + ctx["pseudo"]
    valid = [item for item in positive if _seqlet_entropy(item.contrib_scores) >= params["tau"]]
    valid_augmented = [
        item for item in augmented if _seqlet_entropy(item.contrib_scores) >= params["tau"]
    ]
    row = {"n_motif_seqlets_added": len(ctx["pseudo"])}
    _set_summary(row, "raw", positive, params)
    _set_summary(row, "aug", augmented, params)
    _set_summary(row, "valid", valid, params)
    _set_summary(row, "valid_aug", valid_augmented, params)

    set_data = {}
    for name, seqlets in {
        "raw": positive, "aug": augmented, "valid_aug": valid_augmented
    }.items():
        regions, hits = _matched_motifs(
            seqlets, ctx["motifs"], ctx["cb"], params["fraction_1"], params["fraction_2"]
        )
        strings = [_one_hot_to_string(sequence[start:end]) for start, end in regions]
        set_data[name] = (seqlets, regions, hits, strings)

    for name, (_, regions, hits, strings) in set_data.items():
        matched = [(region, hit, string) for region, hit, string in zip(regions, hits, strings) if hit]
        row[_COUNT_COLUMNS[name]] = len(matched)
        if len(matched) >= 2:
            matched.sort(key=lambda value: value[0])
            row[_FEATURE_COLUMNS[name]] = float(_unique_features(
                matched, params["uniq_k"], params["uniq_min_match"], params["min_motif_ov"]
            ))

        if matched:
            bad_count = 0
            self_redundant_count = 0
            for (start, end), motif_hits, string in matched:
                bad = False
                for motif_start, motif_end in motif_hits:
                    motif_attr = ctx["cb"][motif_start:motif_end]
                    positive_attr = motif_attr[motif_attr > 0]
                    if len(positive_attr):
                        threshold = params["single_base_ratio"] * float(positive_attr.mean())
                        bad = bad or bool((np.abs(motif_attr[motif_attr < 0]) > threshold).any())
                bad_count += int(bad)

                redundant_pairs = _self_redundant_pairs(
                    string, params["uniq_k"], params["uniq_min_match"]
                )
                unexplained = any(
                    not any(
                        min(motif_end, start + second + params["uniq_k"])
                        - max(motif_start, start + first)
                        >= params["bridge_frac"] * (second + params["uniq_k"] - first)
                        for motif_start, motif_end in motif_hits
                    )
                    for first, second in redundant_pairs
                )
                self_redundant_count += int(unexplained)
            row[_SINGLE_BASE_COLUMNS[name]] = float(bad_count)
            row[_SELF_REDUNDANT_COLUMNS[name]] = float(self_redundant_count)

        if len(matched) >= 2:
            row[_COMB_MOTIF_COLUMNS[name]] = _count_pairs(
                [item[0] for item in matched], [item[1] for item in matched],
                [item[2] for item in matched], params, motif_records,
                position, index, name,
            )
        if len(regions) >= 2:
            order = sorted(range(len(regions)), key=lambda value: regions[value])
            row[_COMB_ALL_COLUMNS[name]] = _count_pairs(
                [regions[i] for i in order], [hits[i] for i in order],
                [strings[i] for i in order], params, all_records,
                position, index, name,
            )

        if len(regions) >= params["min_seqlets"]:
            heterotypic = 0
            for i, region in enumerate(regions):
                distances = [
                    (max(0, max(region[0], other[0]) - min(region[1], other[1])), j)
                    for j, other in enumerate(regions) if j != i
                ]
                nearest = min(distances)[1]
                redundant = _seqlets_redundant(
                    strings[i], strings[nearest], params["uniq_k"], params["uniq_min_match"]
                )
                motif_overlap = _max_motif_overlap(hits[i], hits[nearest])
                shared_hit = bool(hits[i]) and bool(hits[nearest]) and motif_overlap >= 1
                heterotypic += int(
                    not redundant
                    and motif_overlap < params["min_motif_ov"]
                    and not shared_hit
                )
            row[_ADJACENT_COLUMNS[name]] = float(heterotypic)
    return row


def calculate_seqlet_features(
    df: pd.DataFrame,
    sequences: np.ndarray,
    attributions: np.ndarray,
    positive_seqlets,
    all_seqlets,
    motif_scan: pd.DataFrame,
    motif_list: Sequence[str] | None = None,
    params: dict | None = None,
    collect_pairs: bool = True,
):
    """Calculate the updated 46 seqlet metrics for every DataFrame row.

    The implementation follows ``seqeval_metrics_new.py`` but consumes the
    already aligned arrays and cached SeqTrust results directly. It therefore
    excludes the standalone script's pickle loading and command-line layers.
    """
    sequences = np.asarray(sequences, dtype=np.float32)
    attributions = np.asarray(attributions, dtype=np.float32)
    if sequences.shape != attributions.shape or sequences.ndim != 3 or sequences.shape[-1] != 4:
        raise ValueError(
            "sequences and attributions must share shape (N, L, 4); "
            f"received {sequences.shape} and {attributions.shape}."
        )
    if len(df) != len(sequences):
        raise ValueError("The DataFrame, sequences, and attributions must have equal length.")

    options = dict(DEFAULT_PARAMS)
    if params:
        unknown = sorted(set(params) - set(options))
        if unknown:
            raise ValueError(f"Unknown seqlet metric parameters: {unknown}")
        options.update(params)
    if options["uniq_min_match"] > options["uniq_k"]:
        raise ValueError("uniq_min_match cannot exceed uniq_k.")
    if options["comb_max_gap"] <= options["comb_min_gap"]:
        raise ValueError("comb_max_gap must be greater than comb_min_gap.")
    for name in ("tau", "fraction_1", "fraction_2", "bridge_frac"):
        if not 0 <= options[name] <= 1:
            raise ValueError(f"{name} must be between 0 and 1.")
    for name in (
        "pseudo_overlap_min", "uniq_k", "uniq_min_match", "min_motif_ov",
        "comb_min_gap", "comb_max_gap", "min_seqlets",
    ):
        if not isinstance(options[name], (int, np.integer)) or options[name] < 0:
            raise ValueError(f"{name} must be a non-negative integer.")
    if options["uniq_k"] < 1 or options["uniq_min_match"] < 1:
        raise ValueError("uniq_k and uniq_min_match must be at least 1.")
    if options["min_seqlets"] < 2:
        raise ValueError("min_seqlets must be at least 2.")
    if options["overlap_min"] is not None and (
        not isinstance(options["overlap_min"], (int, np.integer))
        or options["overlap_min"] < 0
    ):
        raise ValueError("overlap_min must be a non-negative integer or None.")
    if options["single_base_ratio"] < 0:
        raise ValueError("single_base_ratio must be non-negative.")

    n_sequences = len(df)
    positive_by_sequence = _group_seqlets(positive_seqlets, n_sequences)
    positive_by_sequence, removed_ids = _suppress_overlaps(
        positive_by_sequence, options["overlap_min"]
    )
    all_by_sequence = _group_seqlets(all_seqlets, n_sequences)
    if removed_ids:
        all_by_sequence = {
            key: [item for item in value if id(item) not in removed_ids]
            for key, value in all_by_sequence.items()
        }
    motifs = _prepare_motifs(
        motif_scan, motif_list, n_sequences, sequences.shape[1]
    )
    per_base = (sequences * attributions).sum(axis=2)
    mean_attr = per_base.mean(axis=1)

    columns = {name: np.zeros(n_sequences, dtype=int) for name in _INT_COLUMNS}
    columns.update({name: np.full(n_sequences, np.nan) for name in _FLOAT_COLUMNS})
    motif_records = [] if collect_pairs else None
    all_records = [] if collect_pairs else None
    for position in range(n_sequences):
        occupied = [(int(item.start), int(item.end)) for item in all_by_sequence[position]]
        context = _build_context(
            position, positive_by_sequence[position], occupied, motifs,
            per_base, mean_attr, sequences, options,
        )
        row = _sequence_metrics(
            context, sequences[position], options, motif_records, all_records,
            position, df.index[position],
        )
        for name, value in row.items():
            columns[name][position] = value

    feature_df = pd.DataFrame({name: columns[name] for name in ALL_COLUMNS}, index=df.index)
    motif_pairs = pd.DataFrame(motif_records or [], columns=PAIR_COLUMNS)
    all_pairs = pd.DataFrame(all_records or [], columns=PAIR_COLUMNS)
    details = {
        "params": options,
        "column_groups": COLUMN_GROUPS,
        "combinatorial_motif_pairs": motif_pairs,
        "combinatorial_all_pairs": all_pairs,
        "n_overlap_suppressed": len(removed_ids),
    }
    return feature_df, details
