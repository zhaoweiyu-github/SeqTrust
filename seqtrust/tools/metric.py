"""Seqlet-based biological plausibility metrics.

Computes, per sequence, the metric groups of the SeqTrust benchmark:

1. number / coverage / span / pairwise uniqueness / nonredundant partner
   number of the positive, motif-augmented positive, valid (entropy >= tau)
   and valid-augmented seqlet sets;
2. diversity = nonredundant partner number / (N - 1) on each of those sets;
3. seqlet-motif matching metrics (matched seqlet counts + unique matched
   features, union-find over k-mer / shared-motif redundancy);
4. single-base negative-attribution metric over matched seqlet-motif pairs;
5. self-redundant seqlets with matched motifs (unexplained by a bridging
   motif);
6. combinatorial seqlets-motif pairs (non-redundant, meaningfully
   separated, non-overlapping matched motifs);
7. combinatorial seqlet all-pairs (the same pair rules over the FULL
   seqlet set, so seqlets without a matched motif are eligible);
8. adjacent heterotypic seqlets (closest neighbour is heterotypic).

The public entry point is ``compute_metrics``; seqlet overlap suppression
is provided by ``suppress_overlapping_seqlets``. All thresholds default
to ``DEFAULT_PARAMS`` and can be overridden per call.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ────────────────────────────────────────────────────────────────────────
# Default thresholds
# ────────────────────────────────────────────────────────────────────────
DEFAULT_PARAMS = {
    "tau": 0.75,               # validity cut: normalized attribution entropy >= tau
    "pseudo_overlap_min": 6,   # bp NMS among motif-augmented seqlet candidates
    "overlap_min": 4,          # bp; load-time NMS of positive seqlets (None disables)
    "uniq_k": 6,               # k-mer size for the redundancy rule
    "uniq_min_match": 5,       # min matching positions (of k) to call k-mers redundant
    "fraction_1": 0.75,        # geometric overlap threshold for a matched pair
    "fraction_2": 0.75,        # attribution overlap threshold for a matched pair
    "min_motif_ov": 1,         # bp; min motif-interval overlap for a "shared" motif
    "single_base_ratio": 1.0,  # |negative attr| > ratio x mean(positive attr in motif)
    "comb_min_gap": 1,         # bp; combinatorial seqlets separated by >= this gap
    "comb_max_gap": 20,        # bp; combinatorial seqlets separated by < this gap
    "bridge_frac": 0.8,        # motif must cover >= this fraction of a redundant
                               # k-mer pair span to "explain" the self-redundancy
    "min_seqlets": 3,          # min seqlets for the adjacent-heterotypic metric
}

# column groups of the output metrics DataFrame
COLUMN_GROUPS = {
    "positive": ["n_pos_seqlets", "pos_seqlets_coverage",
                 "pos_seqlets_span", "pos_seqlets_pair_uniq",
                 "pos_seqlets_nonred_partner_num", "pos_seqlets_diversity"],
    "augmented": ["n_motif_seqlets_added", "n_pos_seqlets_aug",
                  "pos_seqlets_coverage_aug", "pos_seqlets_span_aug",
                  "pos_seqlets_pair_uniq_aug",
                  "pos_seqlets_nonred_partner_num_aug",
                  "pos_seqlets_diversity_aug"],
    "valid_raw": ["n_valid_pos_seqlets", "valid_pos_seqlets_coverage",
                  "valid_pos_seqlets_span", "valid_pos_seqlets_pair_uniq",
                  "valid_pos_seqlets_nonred_partner_num",
                  "valid_pos_seqlets_diversity"],
    "valid_aug": ["n_valid_pos_seqlets_aug", "valid_pos_seqlets_coverage_aug",
                  "valid_pos_seqlets_span_aug",
                  "valid_pos_seqlets_pair_uniq_aug",
                  "valid_pos_seqlets_nonred_partner_num_aug",
                  "valid_pos_seqlets_diversity_aug"],
    "matched_motif": ["n_pos_seqlet_with_matched_motif",
                      "n_pos_seqlet_with_matched_motif_aug",
                      "n_valid_pos_seqlet_with_matched_motif_aug",
                      "single_base_matched_motif",
                      "single_base_matched_motif_aug",
                      "valid_single_base_matched_motif_aug"],
    "unique_matched_features": ["n_unique_matched_features",
                                "n_unique_matched_features_aug",
                                "n_valid_unique_matched_features_aug"],
    "sr_matched_motif": ["sr_seqlet_with_matched_motif",
                         "sr_seqlet_with_matched_motif_aug",
                         "valid_sr_seqlet_with_matched_motif_aug"],
    "combinatorial_motif_pairs": ["n_combinatorial_seqlets_motif_pairs",
                                  "n_combinatorial_seqlets_motif_pairs_aug",
                                  "n_valid_combinatorial_seqlets_motif_pairs_aug"],
    "combinatorial_all_pairs": ["n_combinatorial_seqlet_all_pairs",
                                "n_combinatorial_seqlet_all_pairs_aug",
                                "n_valid_combinatorial_seqlet_all_pairs_aug"],
    "adjacent_heterotypic": ["adjacent_heterotypic_seqlets",
                             "adjacent_heterotypic_seqlets_aug",
                             "valid_adjacent_heterotypic_seqlets_aug"],
}

# diversity column -> (nonredundant partner number column, count column)
DIVERSITY_COLS = {
    "pos_seqlets_diversity": ("pos_seqlets_nonred_partner_num",
                              "n_pos_seqlets"),
    "pos_seqlets_diversity_aug": ("pos_seqlets_nonred_partner_num_aug",
                                  "n_pos_seqlets_aug"),
    "valid_pos_seqlets_diversity": ("valid_pos_seqlets_nonred_partner_num",
                                    "n_valid_pos_seqlets"),
    "valid_pos_seqlets_diversity_aug": (
        "valid_pos_seqlets_nonred_partner_num_aug",
        "n_valid_pos_seqlets_aug"),
}

# integer columns (default 0) and float columns (default NaN).
# ALL_COLUMNS follows the order: count/coverage/span, uniqueness,
# matched-motif counts, unique features, single-base, sr, motif-gated
# pairs, all-seqlet pairs, adjacent-heterotypic, then diversity.
_INT_COLS = [
    "n_pos_seqlets", "pos_seqlets_coverage", "pos_seqlets_span",
    "n_motif_seqlets_added", "n_pos_seqlets_aug", "pos_seqlets_coverage_aug",
    "pos_seqlets_span_aug", "n_valid_pos_seqlets", "valid_pos_seqlets_coverage",
    "valid_pos_seqlets_span", "n_valid_pos_seqlets_aug",
    "valid_pos_seqlets_coverage_aug", "valid_pos_seqlets_span_aug",
    "n_pos_seqlet_with_matched_motif", "n_pos_seqlet_with_matched_motif_aug",
    "n_valid_pos_seqlet_with_matched_motif_aug",
]
_NAN_COLS = [
    "pos_seqlets_pair_uniq", "pos_seqlets_pair_uniq_aug",
    "valid_pos_seqlets_pair_uniq", "valid_pos_seqlets_pair_uniq_aug",
    "pos_seqlets_nonred_partner_num", "pos_seqlets_nonred_partner_num_aug",
    "valid_pos_seqlets_nonred_partner_num",
    "valid_pos_seqlets_nonred_partner_num_aug",
    "n_unique_matched_features", "n_unique_matched_features_aug",
    "n_valid_unique_matched_features_aug",
    "single_base_matched_motif", "single_base_matched_motif_aug",
    "valid_single_base_matched_motif_aug",
    "sr_seqlet_with_matched_motif", "sr_seqlet_with_matched_motif_aug",
    "valid_sr_seqlet_with_matched_motif_aug",
    "n_combinatorial_seqlets_motif_pairs",
    "n_combinatorial_seqlets_motif_pairs_aug",
    "n_valid_combinatorial_seqlets_motif_pairs_aug",
    "n_combinatorial_seqlet_all_pairs",
    "n_combinatorial_seqlet_all_pairs_aug",
    "n_valid_combinatorial_seqlet_all_pairs_aug",
    "adjacent_heterotypic_seqlets", "adjacent_heterotypic_seqlets_aug",
    "valid_adjacent_heterotypic_seqlets_aug",
    "pos_seqlets_diversity", "pos_seqlets_diversity_aug",
    "valid_pos_seqlets_diversity", "valid_pos_seqlets_diversity_aug",
]
ALL_COLUMNS = (
    _INT_COLS[:13] + _NAN_COLS[:8] + _INT_COLS[13:] + _NAN_COLS[8:]
)
assert len(ALL_COLUMNS) == 46
assert set(ALL_COLUMNS) == set(_INT_COLS) | set(_NAN_COLS)

_PAIR_RECORD_COLUMNS = [
    "dataset", "seq_idx", "set_name",
    "seqlet_a_start", "seqlet_a_end", "seqlet_b_start", "seqlet_b_end",
    "gap_bp", "kmer_redundant", "motif_max_ov_bp", "motif_redundant",
    "motif_overlap", "is_combinatorial",
]


# ────────────────────────────────────────────────────────────────────────
# Small helpers
# ────────────────────────────────────────────────────────────────────────
def _seqlet_norm_entropy(contrib_scores):
    """Normalized Shannon entropy of a seqlet's per-base positive attribution.

    a_i = per-base contribution (summed over channels if 2-D);
    w_i = max(a_i, 0); p_i = w_i / sum(w);
    H = -sum p_i log2 p_i; H_norm = H / log2(L).

    Returns 0.0 for a seqlet with no positive attribution mass or L <= 1.
    """
    cs = np.asarray(contrib_scores)
    per_base = cs.sum(axis=1) if cs.ndim > 1 else cs
    w = np.maximum(per_base, 0.0)
    total = w.sum()
    if total <= 0:
        return 0.0
    p = w / total
    p = p[p > 0]
    H = float(-(p * np.log2(p)).sum())
    L = per_base.shape[0]
    return H / np.log2(L) if L > 1 else 0.0


class _MotifSeqlet:
    """Lightweight shim giving a standalone motif interval the seqlet
    interface (start / end / contrib_scores / sequence).

    contrib_scores is the 1-D per-base signed contribution over the motif
    interval; sequence is the (L, 4) one-hot of the motif interval.
    """
    __slots__ = ("start", "end", "contrib_scores", "sequence", "own_motif")

    def __init__(self, start, end, contrib_scores, sequence):
        self.start = start
        self.end = end
        self.contrib_scores = contrib_scores
        self.sequence = sequence
        self.own_motif = (start, end)  # the generating motif interval


def _coverage_span(regions):
    """(coverage, span) of a list of [start, end) regions.

    coverage = union length (bp) after merging overlapping/adjacent
    intervals; span = max(end) - min(start). Both 0 for an empty list.
    """
    regions = sorted((int(s), int(e)) for s, e in regions if e > s)
    if not regions:
        return 0, 0
    merged = [list(regions[0])]
    for s0, s1 in regions[1:]:
        if s0 <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], s1)
        else:
            merged.append([s0, s1])
    coverage = sum(e - s for s, e in merged)
    span = max(e for _, e in regions) - min(s for s, _ in regions)
    return coverage, span


def one_hot_to_str(one_hot):
    """Convert a (length, 4) one-hot array to a nucleotide string."""
    bases = np.array(["A", "C", "G", "T"])
    return "".join(bases[np.argmax(one_hot, axis=-1)])


# ── k-mer redundancy helpers ──
_RC_TABLE = str.maketrans("ACGT", "TGCA")


def reverse_complement(seq):
    """Reverse complement of a DNA string (A<->T, C<->G)."""
    return seq.translate(_RC_TABLE)[::-1]


def sliding_kmers(seq, k=6):
    """All sliding k-mers of `seq`; k-mers containing non-ACGT bases are
    skipped."""
    seq = seq.upper()
    return [seq[j:j + k] for j in range(len(seq) - k + 1)
            if set(seq[j:j + k]) <= set("ACGT")]


def is_kmer_match(kmer1, kmer2, min_match=5):
    """True if kmer1 matches kmer2 (forward) or reverse_complement(kmer2)
    at >= min_match of k positions."""
    fwd = sum(a == b for a, b in zip(kmer1, kmer2))
    if fwd >= min_match:
        return True
    rc = reverse_complement(kmer2)
    return sum(a == b for a, b in zip(kmer1, rc)) >= min_match


def seqlets_are_redundant(seqlet1, seqlet2, k=6, min_match=5):
    """True if any k-mer of seqlet1 matches any k-mer of seqlet2 (forward
    or reverse-complement, >= min_match of k positions)."""
    kmers2 = sliding_kmers(seqlet2, k=k)
    if not kmers2:
        return False
    for k1 in sliding_kmers(seqlet1, k=k):
        for k2 in kmers2:
            if is_kmer_match(k1, k2, min_match=min_match):
                return True
    return False


def _seqlet_redundancy_matrix(seqlets, k=6, min_match=5):
    """Binary symmetric redundancy matrix O: O[i, j] = True iff seqlets i
    and j share a sliding k-mer (>= min_match of k positions, forward or
    reverse-complement)."""
    n = len(seqlets)
    redundant = np.zeros((n, n), dtype=bool)
    for a in range(n):
        for b in range(a + 1, n):
            if seqlets_are_redundant(seqlets[a], seqlets[b],
                                     k=k, min_match=min_match):
                redundant[a, b] = redundant[b, a] = True
    return redundant


def calculate_seqlet_pair_uniqueness(seqlets, k=6, min_match=5):
    """Pairwise seqlet uniqueness U_pair = 1 - (#redundant pairs) / C(N, 2),
    together with the Nonredundant Partner Number = (N - 1) * U_pair.

    Returns (U_pair, nonredundant_partner_number); (NaN, NaN) for fewer
    than 2 seqlets (no pairs to test).
    """
    n = len(seqlets)
    if n < 2:
        return np.nan, np.nan
    redundant = _seqlet_redundancy_matrix(seqlets, k=k, min_match=min_match)
    n_redundant_pairs = int(np.triu(redundant, k=1).sum())
    n_pairs = n * (n - 1) // 2
    u_pair = 1.0 - n_redundant_pairs / n_pairs
    return u_pair, (n - 1) * u_pair


def _self_redundant_kmer_pairs(seq_str, k=6, min_match=5):
    """Pairs of k-mer windows WITHIN one seqlet that are redundant under
    the k-mer rule (is_kmer_match: >= min_match of k positions, forward or
    reverse-complement).

    Returns [(offset_a, offset_b)] with a < b, offsets relative to the
    seqlet start.
    """
    kmers = [(j, seq_str[j:j + k])
             for j in range(len(seq_str) - k + 1)
             if set(seq_str[j:j + k]) <= set("ACGT")]
    pairs = []
    for x in range(len(kmers)):
        for y in range(x + 1, len(kmers)):
            if is_kmer_match(kmers[x][1], kmers[y][1], min_match=min_match):
                pairs.append((kmers[x][0], kmers[y][0]))
    return pairs


def _diversity(nonred_partner_num, n_seqlets):
    """diversity = nonredundant partner number / (N - 1); NaN when N < 2.

    Numerically identical to U_pair. Stored as its own column so downstream
    code does not have to re-derive it.
    """
    if n_seqlets < 2 or nonred_partner_num != nonred_partner_num:
        return np.nan
    return float(nonred_partner_num) / (n_seqlets - 1.0)


# ────────────────────────────────────────────────────────────────────────
# Load-time positive-seqlet overlap suppression (non-mutating)
# ────────────────────────────────────────────────────────────────────────
def suppress_overlapping_seqlets(seqlets_by_sequence, pos_seqlets,
                                 overlap_min=4):
    """Greedy non-maximum suppression among positive seqlets (per sequence).

    Seqlets are ranked by total attribution mass (sum of contrib_scores);
    a seqlet overlapping a stronger kept one by MORE than `overlap_min` bp
    is removed.

    Non-mutating: returns new lists / dicts; the seqlets stored in
    ``st.results`` are never modified.

    Parameters
    ----------
    seqlets_by_sequence : dict
        Mapping of sequence index to the seqlets of that sequence.
    pos_seqlets : list
        All positive seqlets (defines the sign by object identity).
    overlap_min : int or None
        Overlap threshold in bp. None disables the suppression.

    Returns
    -------
    (kept_pos_seqlets, kept_seqlets_by_sequence, n_removed)
    """
    if overlap_min is None:
        return pos_seqlets, seqlets_by_sequence, 0

    pos_ids = {id(s) for s in pos_seqlets}
    removed = set()

    for i, sl in seqlets_by_sequence.items():
        pos = [s for s in sl if id(s) in pos_ids]
        entries = sorted(pos, key=lambda s: float(np.sum(s.contrib_scores)),
                         reverse=True)
        kept_regions = []
        for s in entries:
            s0, s1 = int(s.start), int(s.end)
            if any(min(s1, k1) - max(s0, k0) > overlap_min
                   for k0, k1 in kept_regions):
                removed.add(id(s))
                continue
            kept_regions.append((s0, s1))

    kept_pos = [s for s in pos_seqlets if id(s) not in removed]
    kept_by_sequence = {
        i: [s for s in sl if id(s) not in removed]
        for i, sl in seqlets_by_sequence.items()
    }
    return kept_pos, kept_by_sequence, len(removed)


# ────────────────────────────────────────────────────────────────────────
# Per-sequence seqlet-set construction (shared by all metric groups)
# ────────────────────────────────────────────────────────────────────────
def _build_seq_context(data, i, pseudo_overlap_min):
    """Build the kept positive seqlets + motif-augmented seqlets of sequence i.

    Motif-augmented seqlets: motif hits that are STANDALONE (no >= 1 bp
    overlap with any kept seqlet of either sign) and ATTRIBUTED (mean
    per-base signed contribution above the whole-sequence mean), greedily
    NMS-filtered among themselves (strongest-first, `pseudo_overlap_min` bp).
    """
    all_seqlets = data["seqlets_by_sequence"].get(i, [])
    pos_ids = data["pos_ids"]
    pos_seqlets = [s for s in all_seqlets if id(s) in pos_ids]
    motifs_i = data["motif_dict"].get(i, [])
    cb = data["per_pos"][i]
    seq_mean = data["seq_mean_attr"][i]

    occupied = [(int(s.start), int(s.end)) for s in all_seqlets]
    candidates = []
    for m0, m1 in motifs_i:
        m0, m1 = int(m0), int(m1)
        if m1 <= m0:
            continue
        # standalone: no >= 1 bp overlap with ANY kept seqlet
        if any(min(m1, k1) - max(m0, k0) >= 1 for k0, k1 in occupied):
            continue
        # attributed: motif mean per-base contribution > sequence mean
        mean_attr = float(cb[m0:m1].mean())
        if mean_attr <= seq_mean:
            continue
        candidates.append((mean_attr, m0, m1))

    candidates.sort(reverse=True)
    pseudo = []
    kept_pseudo_regions = []
    for mean_attr, m0, m1 in candidates:
        if any(min(m1, k1) - max(m0, k0) >= pseudo_overlap_min
               for k0, k1 in kept_pseudo_regions):
            continue
        kept_pseudo_regions.append((m0, m1))
        pseudo.append(_MotifSeqlet(
            m0, m1, cb[m0:m1].copy(), data["sequences"][i, m0:m1]))

    return {"pos": pos_seqlets, "pseudo": pseudo, "motifs": motifs_i, "cb": cb}


def _matched_motif_hits(seqlets, motifs_i, cb, fraction_1, fraction_2,
                        keep_unmatched=False):
    """Matched motif intervals per seqlet (the two-criteria test).

    A seqlet-motif pair is MATCHED iff BOTH:
      1. geometric:   max(L_ov / L_seqlet, L_ov / L_motif) >= fraction_1
      2. attribution: max(A_ov / A_seqlet, A_ov / A_motif) >= fraction_2
    where A_* are sums of the per-base signed contribution over the region.
    Motif-half-seqlets may NOT match their own generating motif.

    Returns (regs, hits_list): parallel lists of (s0, s1) intervals and
    their matched motif intervals. Seqlets with s1 <= s0 are skipped.
    By default only seqlets with >= 1 hit are recorded; keep_unmatched=True
    also records seqlets with an empty hit list (all-seqlet pair metrics
    and the adjacent-heterotypic metric).
    """
    regs, hits_list = [], []
    for s in seqlets:
        s0, s1 = int(s.start), int(s.end)
        if s1 <= s0:
            continue
        own = getattr(s, "own_motif", None)
        A_s = float(cb[s0:s1].sum())
        hits = []
        for m0, m1 in motifs_i:
            if own is not None and (m0, m1) == own:
                continue  # skip trivial self-match
            o0, o1 = max(s0, m0), min(s1, m1)
            L_ov = o1 - o0
            if L_ov <= 0:
                continue
            # criterion 1: geometric overlap
            if max(L_ov / (s1 - s0), L_ov / (m1 - m0)) < fraction_1:
                continue
            # criterion 2: attribution overlap
            A_ov = float(cb[o0:o1].sum())
            A_m = float(cb[m0:m1].sum())
            fracs = [A_ov / denom for denom in (A_s, A_m) if denom > 0]
            if not fracs or max(fracs) < fraction_2:
                continue
            hits.append((m0, m1))
        if hits or keep_unmatched:
            regs.append((s0, s1))
            hits_list.append(hits)
    return regs, hits_list


def _unique_matched_features(matched, uniq_k, uniq_min_match, min_motif_ov):
    """Number of unique matched features among matched seqlets (union-find).

    `matched`: list of ((s0, s1), hits, seq_str), sorted by interval.
    Two matched seqlets are REDUNDANT if k-mer-redundant OR their matched
    motif sets share >= 1 motif hit overlapping by >= min_motif_ov bp.
    Connected components = unique regulatory features.
    """
    n_s = len(matched)
    parent = list(range(n_s))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for a in range(n_s):
        for b in range(a + 1, n_s):
            # criterion 1: seqlet k-mer redundancy
            if seqlets_are_redundant(matched[a][2], matched[b][2],
                                     k=uniq_k, min_match=uniq_min_match):
                union(a, b)
                continue
            # criterion 2: matched motif sets share an overlapping motif hit
            if any(min(ma1, mb1) - max(ma0, mb0) >= min_motif_ov
                   for ma0, ma1 in matched[a][1]
                   for mb0, mb1 in matched[b][1]):
                union(a, b)

    return len({find(x) for x in range(n_s)})


def _max_motif_overlap(hits_a, hits_b):
    """Max bp overlap between any motif hit of A and any motif hit of B."""
    max_ov = 0
    for ma0, ma1 in hits_a:
        for mb0, mb1 in hits_b:
            max_ov = max(max_ov, min(ma1, mb1) - max(ma0, mb0))
    return max_ov


def _combinatorial_pair_flags(str_a, str_b, hits_a, hits_b, gap,
                              uniq_k, uniq_min_match, min_motif_ov,
                              min_gap, max_gap):
    """Flags for one unordered seqlet pair (a before b).

    motif_overlap is true only when BOTH seqlets have matched motifs and
    some hit pair overlaps by >= 1 bp. For the motif-gated pool every
    seqlet already has hits, so this matches the `max_motif_ov >= 1` test.
    Empty hit lists can never be motif-redundant.
    """
    kmer_red = seqlets_are_redundant(str_a, str_b, k=uniq_k,
                                     min_match=uniq_min_match)
    max_motif_ov = _max_motif_overlap(hits_a, hits_b)
    motif_red = max_motif_ov >= min_motif_ov
    motif_ov = bool(hits_a) and bool(hits_b) and max_motif_ov >= 1
    is_comb = (not kmer_red and not motif_red and not motif_ov
               and min_gap <= gap < max_gap)
    return kmer_red, max_motif_ov, motif_red, motif_ov, is_comb


# ────────────────────────────────────────────────────────────────────────
# Per-sequence metrics (all groups, one pass over the seqlet sets)
# ────────────────────────────────────────────────────────────────────────
_COUNT_COLS = {
    "raw": "n_pos_seqlet_with_matched_motif",
    "aug": "n_pos_seqlet_with_matched_motif_aug",
    "valid_aug": "n_valid_pos_seqlet_with_matched_motif_aug",
}
_FEAT_COLS = {
    "raw": "n_unique_matched_features",
    "aug": "n_unique_matched_features_aug",
    "valid_aug": "n_valid_unique_matched_features_aug",
}
_SBM_COLS = {
    "raw": "single_base_matched_motif",
    "aug": "single_base_matched_motif_aug",
    "valid_aug": "valid_single_base_matched_motif_aug",
}
_SR_COLS = {
    "raw": "sr_seqlet_with_matched_motif",
    "aug": "sr_seqlet_with_matched_motif_aug",
    "valid_aug": "valid_sr_seqlet_with_matched_motif_aug",
}
_COMB_MOTIF_COLS = {
    "raw": "n_combinatorial_seqlets_motif_pairs",
    "aug": "n_combinatorial_seqlets_motif_pairs_aug",
    "valid_aug": "n_valid_combinatorial_seqlets_motif_pairs_aug",
}
_COMB_ALL_COLS = {
    "raw": "n_combinatorial_seqlet_all_pairs",
    "aug": "n_combinatorial_seqlet_all_pairs_aug",
    "valid_aug": "n_valid_combinatorial_seqlet_all_pairs_aug",
}
_AHS_COLS = {
    "raw": "adjacent_heterotypic_seqlets",
    "aug": "adjacent_heterotypic_seqlets_aug",
    "valid_aug": "valid_adjacent_heterotypic_seqlets_aug",
}


def _append_pair_record(records, dataset, seq_idx, set_name,
                        a0, a1, b0, b1, gap, kmer_red, max_motif_ov,
                        motif_red, motif_ov, is_comb):
    records.append({
        "dataset": dataset, "seq_idx": seq_idx, "set_name": set_name,
        "seqlet_a_start": a0, "seqlet_a_end": a1,
        "seqlet_b_start": b0, "seqlet_b_end": b1,
        "gap_bp": gap,
        "kmer_redundant": kmer_red,
        "motif_max_ov_bp": max_motif_ov,
        "motif_redundant": motif_red,
        "motif_overlap": motif_ov,
        "is_combinatorial": is_comb,
    })


def _count_combinatorial_pairs(regs, hits_list, strs, params, records,
                               dataset, seq_idx, set_name):
    """Number of combinatorial pairs among `regs` (already ordered a-before-b
    by sorting on the interval). NaN-callers skip this when len < 2.

    Returns the pair count. Appends one long-form record per unordered pair
    when `records` is not None.
    """
    uniq_k = params["uniq_k"]
    uniq_min_match = params["uniq_min_match"]
    min_motif_ov = params["min_motif_ov"]
    min_gap = params["comb_min_gap"]
    max_gap = params["comb_max_gap"]
    n_pairs = 0
    for a in range(len(regs)):
        for b in range(a + 1, len(regs)):
            a0, a1 = regs[a]
            b0, b1 = regs[b]
            gap = b0 - a1
            kmer_red, max_motif_ov, motif_red, motif_ov, is_comb = \
                _combinatorial_pair_flags(
                    strs[a], strs[b], hits_list[a], hits_list[b], gap,
                    uniq_k, uniq_min_match, min_motif_ov, min_gap, max_gap)
            if is_comb:
                n_pairs += 1
            if records is not None:
                _append_pair_record(
                    records, dataset, seq_idx, set_name,
                    a0, a1, b0, b1, gap, kmer_red, max_motif_ov,
                    motif_red, motif_ov, is_comb)
    return n_pairs


def _compute_sequence_metrics(ctx, seq_one_hot, params,
                              motif_pair_records=None, all_pair_records=None,
                              dataset=None, seq_idx=None):
    """All metric columns for one sequence. Returns a dict column -> value
    (columns not applicable to this sequence are simply absent -> stay at
    their 0 / NaN default)."""
    tau = params["tau"]
    uniq_k = params["uniq_k"]
    uniq_min_match = params["uniq_min_match"]
    fraction_1 = params["fraction_1"]
    fraction_2 = params["fraction_2"]
    min_motif_ov = params["min_motif_ov"]
    ratio = params["single_base_ratio"]
    bridge_frac = params["bridge_frac"]
    min_seqlets = params["min_seqlets"]

    pos_seqlets = ctx["pos"]
    pseudo = ctx["pseudo"]
    motifs_i = ctx["motifs"]
    cb = ctx["cb"]

    aug_seqlets = pos_seqlets + pseudo
    valid_seqlets = [s for s in pos_seqlets
                     if _seqlet_norm_entropy(s.contrib_scores) >= tau]
    valid_aug_seqlets = [s for s in aug_seqlets
                         if _seqlet_norm_entropy(s.contrib_scores) >= tau]

    row = {}

    # ── 1. number / coverage / span / pairwise uniqueness / diversity ──
    raw_regions = [(int(s.start), int(s.end)) for s in pos_seqlets]
    row["n_pos_seqlets"] = len(raw_regions)
    row["pos_seqlets_coverage"], row["pos_seqlets_span"] = \
        _coverage_span(raw_regions)
    raw_seqs = [one_hot_to_str(s.sequence) for s in pos_seqlets]
    (row["pos_seqlets_pair_uniq"],
     row["pos_seqlets_nonred_partner_num"]) = \
        calculate_seqlet_pair_uniqueness(raw_seqs, k=uniq_k,
                                         min_match=uniq_min_match)
    row["pos_seqlets_diversity"] = _diversity(
        row["pos_seqlets_nonred_partner_num"], row["n_pos_seqlets"])

    row["n_motif_seqlets_added"] = len(pseudo)
    aug_regions = [(int(s.start), int(s.end)) for s in aug_seqlets]
    row["n_pos_seqlets_aug"] = len(aug_regions)
    row["pos_seqlets_coverage_aug"], row["pos_seqlets_span_aug"] = \
        _coverage_span(aug_regions)
    aug_seqs = [one_hot_to_str(s.sequence) for s in aug_seqlets]
    (row["pos_seqlets_pair_uniq_aug"],
     row["pos_seqlets_nonred_partner_num_aug"]) = \
        calculate_seqlet_pair_uniqueness(aug_seqs, k=uniq_k,
                                         min_match=uniq_min_match)
    row["pos_seqlets_diversity_aug"] = _diversity(
        row["pos_seqlets_nonred_partner_num_aug"], row["n_pos_seqlets_aug"])

    valid_regions = [(int(s.start), int(s.end)) for s in valid_seqlets]
    row["n_valid_pos_seqlets"] = len(valid_regions)
    row["valid_pos_seqlets_coverage"], row["valid_pos_seqlets_span"] = \
        _coverage_span(valid_regions)
    valid_seqs = [one_hot_to_str(s.sequence) for s in valid_seqlets]
    (row["valid_pos_seqlets_pair_uniq"],
     row["valid_pos_seqlets_nonred_partner_num"]) = \
        calculate_seqlet_pair_uniqueness(valid_seqs, k=uniq_k,
                                         min_match=uniq_min_match)
    row["valid_pos_seqlets_diversity"] = _diversity(
        row["valid_pos_seqlets_nonred_partner_num"],
        row["n_valid_pos_seqlets"])

    valid_aug_regions = [(int(s.start), int(s.end)) for s in valid_aug_seqlets]
    row["n_valid_pos_seqlets_aug"] = len(valid_aug_regions)
    (row["valid_pos_seqlets_coverage_aug"],
     row["valid_pos_seqlets_span_aug"]) = _coverage_span(valid_aug_regions)
    valid_aug_seqs = [one_hot_to_str(s.sequence) for s in valid_aug_seqlets]
    (row["valid_pos_seqlets_pair_uniq_aug"],
     row["valid_pos_seqlets_nonred_partner_num_aug"]) = \
        calculate_seqlet_pair_uniqueness(valid_aug_seqs, k=uniq_k,
                                         min_match=uniq_min_match)
    row["valid_pos_seqlets_diversity_aug"] = _diversity(
        row["valid_pos_seqlets_nonred_partner_num_aug"],
        row["n_valid_pos_seqlets_aug"])

    # ── matched-motif hits per set (computed once, shared by groups 2-8) ──
    # keep_unmatched=True records every seqlet, including those with an
    # empty hit list. Matched-only views are sliced out below.
    sets = {
        "raw": pos_seqlets,
        "aug": aug_seqlets,
        "valid_aug": valid_aug_seqlets,
    }
    set_data = {}
    for set_name, seqlets in sets.items():
        regs, hits_list = _matched_motif_hits(
            seqlets, motifs_i, cb, fraction_1, fraction_2,
            keep_unmatched=True)
        strs = [one_hot_to_str(seq_one_hot[a:b]) for a, b in regs]
        set_data[set_name] = (seqlets, regs, hits_list, strs)

    # ── 2. matched seqlet counts + unique matched features ──
    for set_name, (seqlets, regs, hits_list, strs) in set_data.items():
        matched = [(reg, hits, st)
                   for reg, hits, st in zip(regs, hits_list, strs) if hits]
        row[_COUNT_COLS[set_name]] = len(matched)
        if len(matched) >= 2:
            matched.sort(key=lambda t: t[0])
            row[_FEAT_COLS[set_name]] = float(_unique_matched_features(
                matched, uniq_k, uniq_min_match, min_motif_ov))

    # ── 3. single-base negative-attribution metric over matched pairs ──
    for set_name, (seqlets, regs, hits_list, strs) in set_data.items():
        n_matched = 0
        n_bad = 0
        for (s0, s1), hits in zip(regs, hits_list):
            if not hits:
                continue
            n_matched += 1
            bad = False
            for m0, m1 in hits:
                mb = cb[m0:m1]
                pos = mb[mb > 0]
                if len(pos) == 0:
                    continue  # threshold undefined for this motif
                thr = ratio * float(pos.mean())
                if (np.abs(mb[mb < 0]) > thr).any():
                    bad = True
                    break
            if bad:
                n_bad += 1
        if n_matched > 0:
            row[_SBM_COLS[set_name]] = float(n_bad)

    # ── 4. self-redundant seqlets with matched motifs ──
    for set_name, (seqlets, regs, hits_list, strs) in set_data.items():
        n_matched = 0
        n_sr = 0
        for (s0, s1), hits in zip(regs, hits_list):
            if not hits:
                continue
            n_matched += 1
            seq_str = one_hot_to_str(seq_one_hot[s0:s1])
            redundant_pairs = _self_redundant_kmer_pairs(
                seq_str, k=uniq_k, min_match=uniq_min_match)
            if not redundant_pairs:
                continue  # score 0: no self-redundancy
            # is every redundant pair bridged by a matched motif?
            unexplained = False
            for a_off, b_off in redundant_pairs:
                w1_start = s0 + a_off          # first window start
                w2_end = s0 + b_off + uniq_k   # second window end
                pair_len = w2_end - w1_start
                spanned = any(
                    min(m1, w2_end) - max(m0, w1_start) >= bridge_frac * pair_len
                    for m0, m1 in hits)
                if not spanned:
                    unexplained = True
                    break
            if unexplained:
                n_sr += 1
        if n_matched > 0:
            row[_SR_COLS[set_name]] = float(n_sr)

    # ── 5. combinatorial seqlets-motif pairs (motif-matched seqlets only) ──
    for set_name, (seqlets, regs, hits_list, strs) in set_data.items():
        matched = [(reg, hits, st)
                   for reg, hits, st in zip(regs, hits_list, strs) if hits]
        if len(matched) < 2:
            continue  # NaN: not applicable
        matched.sort(key=lambda t: t[0])
        row[_COMB_MOTIF_COLS[set_name]] = float(_count_combinatorial_pairs(
            [t[0] for t in matched], [t[1] for t in matched],
            [t[2] for t in matched], params, motif_pair_records,
            dataset, seq_idx, set_name))

    # ── 6. combinatorial pairs over ALL seqlets ──
    for set_name, (seqlets, regs, hits_list, strs) in set_data.items():
        if len(regs) < 2:
            continue  # NaN: not applicable
        order = sorted(range(len(regs)), key=lambda j: regs[j])
        row[_COMB_ALL_COLS[set_name]] = float(_count_combinatorial_pairs(
            [regs[j] for j in order], [hits_list[j] for j in order],
            [strs[j] for j in order], params, all_pair_records,
            dataset, seq_idx, set_name))

    # ── 7. adjacent heterotypic seqlets ──
    for set_name, (seqlets, regs, hits_list, strs) in set_data.items():
        if len(seqlets) < min_seqlets:
            continue  # NaN: not applicable
        if len(regs) < min_seqlets:
            continue  # NaN: not applicable
        n_s = len(regs)
        score_sum = 0
        for j in range(n_s):
            # closest adjacent seqlet (smallest interval distance; ties
            # broken by first encounter, i.e. original seqlet order)
            nearest, nearest_d = None, None
            for k in range(n_s):
                if k == j:
                    continue
                (a0, a1), (b0, b1) = regs[j], regs[k]
                dist = max(0, max(a0, b0) - min(a1, b1))
                if nearest_d is None or dist < nearest_d:
                    nearest, nearest_d = k, dist
            k = nearest
            if k is None:
                continue

            # condition 1: k-mer redundancy
            if seqlets_are_redundant(strs[j], strs[k], k=uniq_k,
                                     min_match=uniq_min_match):
                continue  # score 0

            # conditions 2 & 3: matched-motif overlap
            max_motif_ov = _max_motif_overlap(hits_list[j], hits_list[k])
            if max_motif_ov >= min_motif_ov:
                continue  # score 0: motif-overlap redundant
            if hits_list[j] and hits_list[k] and max_motif_ov >= 1:
                continue  # score 0: matched motif regions overlap

            score_sum += 1
        row[_AHS_COLS[set_name]] = float(score_sum)

    return row


# ────────────────────────────────────────────────────────────────────────
# Dataset-level driver
# ────────────────────────────────────────────────────────────────────────
def compute_metrics(data, params=None, collect_pairs=False,
                    show_progress=True):
    """Compute all metric columns for every sequence of one dataset.

    Parameters
    ----------
    data : dict
        Per-sequence aligned arrays with keys:

        - ``name``: dataset label (used in progress prints and pair tables)
        - ``index``: output DataFrame index (aligned to ``st.df.index``)
        - ``n``: number of sequences
        - ``sequences``: (N, L, 4) one-hot array
        - ``per_pos``: (N, L) per-base signed contribution
        - ``seq_mean_attr``: (N,) whole-sequence mean contribution
        - ``motif_dict``: sequence index -> [(start, end), ...] motif hits
        - ``seqlets_by_sequence``: sequence index -> [seqlets] (kept-only)
        - ``pos_ids``: ``{id(s)}`` of the positive seqlets

    params : dict or None
        Threshold overrides; omitted keys fall back to ``DEFAULT_PARAMS``.
    collect_pairs : bool
        Also collect the long-form combinatorial pair records.
    show_progress : bool
        Print progress every 2000 sequences.

    Returns
    -------
    (metrics_df, motif_pairs_df, all_pairs_df)
        metrics_df: per-sequence metrics DataFrame (46 columns, aligned to
        ``data["index"]``). Pair tables are empty DataFrames with the right
        columns when collect_pairs is False or no pairs were found.
    """
    name = data["name"]
    n = data["n"]
    index = data["index"]

    params = {**DEFAULT_PARAMS, **(params or {})}

    cols = {c: np.zeros(n, dtype=int) for c in _INT_COLS}
    cols.update({c: np.full(n, np.nan) for c in _NAN_COLS})
    motif_pair_records = [] if collect_pairs else None
    all_pair_records = [] if collect_pairs else None

    for i in range(n):
        ctx = _build_seq_context(data, i, params["pseudo_overlap_min"])
        row = _compute_sequence_metrics(
            ctx, data["sequences"][i], params,
            motif_pair_records=motif_pair_records,
            all_pair_records=all_pair_records,
            dataset=name, seq_idx=i)
        for k, v in row.items():
            cols[k][i] = v
        if show_progress and (i + 1) % 2000 == 0:
            print(f"  {name}: {i + 1}/{n} sequences done")

    metrics_df = pd.DataFrame({c: pd.Series(cols[c], index=index)
                               for c in ALL_COLUMNS})
    motif_pairs_df = pd.DataFrame(
        motif_pair_records if motif_pair_records else [],
        columns=_PAIR_RECORD_COLUMNS)
    all_pairs_df = pd.DataFrame(
        all_pair_records if all_pair_records else [],
        columns=_PAIR_RECORD_COLUMNS)
    return metrics_df, motif_pairs_df, all_pairs_df


def print_metrics_summary(name, df):
    """Compact per-dataset summary (means of the key metric groups)."""
    nps = df["n_pos_seqlets"]
    print(f"{name}: pos seqlets  n mean={nps.mean():.2f}, "
          f"zero={(nps == 0).sum()}/{len(nps)} | "
          f"coverage mean={df['pos_seqlets_coverage'].mean():.1f} bp | "
          f"span mean={df['pos_seqlets_span'].mean():.1f} bp")
    print(f"{name}: augmented    +{df['n_motif_seqlets_added'].mean():.2f} "
          f"motif-augmented seqlets -> n mean={df['n_pos_seqlets_aug'].mean():.2f} | "
          f"valid (raw) n mean={df['n_valid_pos_seqlets'].mean():.2f} | "
          f"valid (aug) n mean={df['n_valid_pos_seqlets_aug'].mean():.2f}")
    print(f"{name}: diversity    raw={df['pos_seqlets_diversity'].mean():.3f} | "
          f"aug={df['pos_seqlets_diversity_aug'].mean():.3f} | "
          f"valid={df['valid_pos_seqlets_diversity'].mean():.3f} | "
          f"valid-aug={df['valid_pos_seqlets_diversity_aug'].mean():.3f}")
    print(f"{name}: matched-motif seqlets  raw={df['n_pos_seqlet_with_matched_motif'].mean():.2f} | "
          f"aug={df['n_pos_seqlet_with_matched_motif_aug'].mean():.2f} | "
          f"valid-aug={df['n_valid_pos_seqlet_with_matched_motif_aug'].mean():.2f} | "
          f"unique features raw={df['n_unique_matched_features'].mean():.2f}")
    print(f"{name}: single_base={df['single_base_matched_motif'].mean():.2f} | "
          f"sr={df['sr_seqlet_with_matched_motif'].mean():.2f} | "
          f"motif-pairs={df['n_combinatorial_seqlets_motif_pairs'].mean():.2f} | "
          f"all-pairs={df['n_combinatorial_seqlet_all_pairs'].mean():.2f} | "
          f"adjacent-heterotypic={df['adjacent_heterotypic_seqlets'].mean():.2f} "
          f"(raw-set means)")
