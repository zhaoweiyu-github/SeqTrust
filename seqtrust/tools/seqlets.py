from __future__ import annotations

from typing import Tuple, Sequence

import numpy as np
import pandas as pd

def extract_modisco_seqlets(
    sequences: np.ndarray,
    attributions: np.ndarray,
    window_size: int = 10,
    flank_size: int = 2,
    target_fdr: float = 0.05,
    min_passing_windows_frac: float = 0.1,
    max_passing_windows_frac: float = 0.4,
    weak_threshold_for_counting_sign: float = 0.8,
):
    """
    Extract seqlets using modisco-lite.

    Parameters
    ----------
    sequences
        One-hot encoded sequences.

        Shape:
            (N, L, 4)

    attributions
        Attribution scores.

        Shape:
            (N, L, 4)

    window_size
        Core window size used for seqlet extraction.

    flank_size
        Number of flanking bases added to each side.

    target_fdr
        Target false discovery rate used by modisco-lite.

    min_passing_windows_frac
        Minimum fraction of windows allowed to pass.

    max_passing_windows_frac
        Maximum fraction of windows allowed to pass.

    weak_threshold_for_counting_sign
        Weak threshold used when estimating attribution sign.

    Returns
    -------
    seqlets
        All extracted seqlets.

    pos_seqlets
        Positive attribution seqlets.

    neg_seqlets
        Negative attribution seqlets.

    threshold
        Threshold estimated by modisco-lite.
    """

    # Keep optional dependency local.
    # SeqEval can still be imported without modisco-lite installed.
    try:
        from modiscolite import core
        from modiscolite import extract_seqlets

    except ImportError as e:
        raise ImportError(
            "modiscolite is required for seqlet extraction. "
            "Please install modisco-lite before calling add_seqlets()."
        ) from e


    sequences = np.asarray(
        sequences,
        dtype=np.float32,
    )

    attributions = np.asarray(
        attributions,
        dtype=np.float32,
    )


    # =========================================================
    # basic validation
    # =========================================================

    if sequences.ndim != 3:

        raise ValueError(
            "sequences must be a 3D array with shape "
            "(N, L, 4). "
            f"Got shape: {sequences.shape}"
        )


    if attributions.ndim != 3:

        raise ValueError(
            "attributions must be a 3D array with shape "
            "(N, L, 4). "
            f"Got shape: {attributions.shape}"
        )


    if sequences.shape != attributions.shape:

        raise ValueError(
            "sequences and attributions must have the same shape. "
            f"sequences={sequences.shape}, "
            f"attributions={attributions.shape}"
        )


    if sequences.shape[-1] != 4:

        raise ValueError(
            "The last dimension of sequences/attributions "
            "must be 4 for A/C/G/T. "
            f"Got shape: {sequences.shape}"
        )


    if window_size <= 0:

        raise ValueError(
            f"window_size must be > 0. "
            f"Got {window_size}."
        )


    if flank_size < 0:

        raise ValueError(
            f"flank_size must be >= 0. "
            f"Got {flank_size}."
        )


    # =========================================================
    # observed-base contribution
    # =========================================================
    #
    # Same as your original:
    #
    # contrib_scores =
    #     sequences * attributions
    #
    # For attribution already masked by one-hot, this operation
    # is effectively unchanged, but retaining it makes the
    # implementation compatible with hypothetical attribution
    # maps as well.
    # =========================================================

    contrib_scores = np.multiply(
        sequences,
        attributions,
    )


    # =========================================================
    # construct TrackSet
    # =========================================================

    track_set = core.TrackSet(
        one_hot=sequences,
        contrib_scores=contrib_scores,
        hypothetical_contribs=attributions,
    )


    # =========================================================
    # extract seqlet coordinates
    # =========================================================

    seqlet_coords, threshold = (
        extract_seqlets.extract_seqlets(
            attribution_scores=(
                contrib_scores.sum(axis=2)
            ),
            window_size=window_size,
            flank=flank_size,
            suppress=(
                int(0.5 * window_size)
                + flank_size
            ),
            target_fdr=target_fdr,
            min_passing_windows_frac=(
                min_passing_windows_frac
            ),
            max_passing_windows_frac=(
                max_passing_windows_frac
            ),
            weak_threshold_for_counting_sign=(
                weak_threshold_for_counting_sign
            ),
        )
    )


    # =========================================================
    # convert coordinates to Seqlet objects
    # =========================================================

    seqlets = track_set.create_seqlets(
        seqlet_coords
    )


    # =========================================================
    # split positive / negative seqlets
    # =========================================================

    pos_seqlets = []
    neg_seqlets = []


    for seqlet in seqlets:

        flank = int(
            0.5 * (
                len(seqlet)
                - window_size
            )
        )


        # Avoid [0:-0], which would generate an empty slice.
        if flank > 0:

            core_contrib = (
                seqlet.contrib_scores[
                    flank:-flank
                ]
            )

        else:

            core_contrib = (
                seqlet.contrib_scores
            )


        attr_sum = float(
            np.sum(core_contrib)
        )


        if attr_sum > threshold:

            pos_seqlets.append(
                seqlet
            )

        elif attr_sum < -threshold:

            neg_seqlets.append(
                seqlet
            )


    return (
        seqlets,
        pos_seqlets,
        neg_seqlets,
        threshold,
    )

def calculate_seqlet_features(
    df: pd.DataFrame,
    attributions: np.ndarray,
    seqlets,
    motif_scan: pd.DataFrame,
    motif_list: Sequence[str],
    overlap_threshold: float = 0.5,
    similarity_threshold: float = 0.7,
) -> pd.DataFrame:
    """
    Calculate seqlet-related features for each sequence.

    This function assumes:

        df.iloc[i]
            corresponds to
        attributions[i]
            corresponds to
        seqlet.example_idx == i
            corresponds to
        motif_scan ID == str(i)

    Parameters
    ----------
    df
        Sequence DataFrame.

    attributions
        Attribution array with shape:

            (N, L, 4)

    seqlets
        Seqlet objects generated by modisco-lite.

        Each seqlet should contain:

            seqlet.example_idx
            seqlet.start
            seqlet.end

    motif_scan
        Motif scanning result.

        Required fields:

            ID
            Matrix_id
            start
            end

        ID may be either a normal column or the DataFrame index.

    motif_list
        Only motifs contained in this list are considered
        when calculating motif-related seqlet features.

    overlap_threshold
        A seqlet and motif are considered overlapping when:

            overlap / seqlet_length >= overlap_threshold

        OR

            overlap / motif_length >= overlap_threshold

        Default: 0.5

    similarity_threshold
        Jaccard similarity threshold used to define
        highly similar seqlet pairs.

        similarity =
            |motifs_i ∩ motifs_j|
            ---------------------
            |motifs_i ∪ motifs_j|

        Default: 0.7

    Returns
    -------
    pd.DataFrame
        DataFrame containing:

            seqlet_count

            seqlet_overlap_motif_count

            seqlet_abs_attr_ratio

            seqlet_signed_over_positive_attr_ratio

            seqlet_motif_diversity

            seqlet_high_similarity_distance
    """

    # =====================================================
    # validation
    # =====================================================

    attributions = np.asarray(
        attributions,
        dtype=np.float32,
    )


    if attributions.ndim != 3:

        raise ValueError(
            "attributions must have shape (N, L, 4), "
            f"but got {attributions.shape}."
        )


    if attributions.shape[0] != len(df):

        raise ValueError(
            "Number of attribution maps does not match "
            "the number of sequences: "
            f"len(df)={len(df)}, "
            f"attributions.shape={attributions.shape}."
        )


    if attributions.shape[-1] != 4:

        raise ValueError(
            "The last dimension of attributions must be 4 "
            "for A/C/G/T, "
            f"but got {attributions.shape}."
        )


    if not 0 <= overlap_threshold <= 1:

        raise ValueError(
            "overlap_threshold must be between 0 and 1, "
            f"but got {overlap_threshold}."
        )


    if not 0 <= similarity_threshold <= 1:

        raise ValueError(
            "similarity_threshold must be between 0 and 1, "
            f"but got {similarity_threshold}."
        )


    if motif_list is None:

        raise ValueError(
            "motif_list must be provided."
        )


    motif_set = set(
        str(x)
        for x in motif_list
    )


    # =====================================================
    # helper:
    # merge overlapping seqlet regions
    # =====================================================

    def merge_regions(regions):

        if len(regions) == 0:
            return []


        regions = sorted(
            regions,
            key=lambda x: x[0],
        )


        merged = [
            list(regions[0])
        ]


        for start, end in regions[1:]:

            last_start, last_end = (
                merged[-1]
            )


            if start <= last_end:

                merged[-1][1] = max(
                    last_end,
                    end,
                )

            else:

                merged.append(
                    [
                        start,
                        end,
                    ]
                )


        return [
            (start, end)
            for start, end in merged
        ]


    # =====================================================
    # preprocess motif_scan
    # =====================================================

    motif_scan = motif_scan.copy()


    if "ID" not in motif_scan.columns:

        if motif_scan.index.name == "ID":

            motif_scan = (
                motif_scan
                .reset_index()
            )

        else:

            raise ValueError(
                "motif_scan must contain an 'ID' column "
                "or have an index named 'ID'."
            )


    required_motif_cols = {
        "ID",
        "Matrix_id",
        "start",
        "end",
    }


    missing_cols = (
        required_motif_cols
        -
        set(motif_scan.columns)
    )


    if missing_cols:

        raise ValueError(
            "motif_scan is missing required columns: "
            f"{sorted(missing_cols)}."
        )


    motif_scan["ID"] = (
        motif_scan["ID"]
        .astype(str)
    )


    motif_scan["Matrix_id"] = (
        motif_scan["Matrix_id"]
        .astype(str)
    )


    # =====================================================
    # only retain motifs requested by motif_list
    # =====================================================

    motif_scan = motif_scan[
        motif_scan["Matrix_id"].isin(
            motif_set
        )
    ].copy()


    # =====================================================
    # build motif dictionary
    #
    # {
    #     "0": [
    #         (start, end, motif_name),
    #         ...
    #     ],
    # }
    # =====================================================

    motif_dict = {}


    for row in motif_scan.itertuples(
        index=False
    ):

        idx = str(
            row.ID
        )


        start = int(
            row.start
        )

        end = int(
            row.end
        )

        motif_name = str(
            row.Matrix_id
        )


        if end <= start:
            continue


        motif_dict.setdefault(
            idx,
            []
        ).append(
            (
                start,
                end,
                motif_name,
            )
        )


    # =====================================================
    # build seqlet information
    # =====================================================

    seqlet_info = {
        i: {
            "regions": [],
            "seqlet_count": 0,
        }
        for i in range(len(df))
    }


    sequence_length = (
        attributions.shape[1]
    )


    for seqlet in seqlets:

        idx = int(
            seqlet.example_idx
        )


        if (
            idx < 0
            or
            idx >= len(df)
        ):

            raise IndexError(
                "seqlet.example_idx is outside the "
                "DataFrame range: "
                f"example_idx={idx}, "
                f"len(df)={len(df)}."
            )


        start = max(
            0,
            int(seqlet.start),
        )


        end = min(
            sequence_length,
            int(seqlet.end),
        )


        if end <= start:
            continue


        seqlet_info[
            idx
        ]["seqlet_count"] += 1


        seqlet_info[
            idx
        ]["regions"].append(
            (
                start,
                end,
            )
        )


    # =====================================================
    # initialize output
    # =====================================================

    n_sequences = len(df)


    seqlet_count = np.zeros(
        n_sequences,
        dtype=np.int64,
    )


    seqlet_overlap_motif_count = np.zeros(
        n_sequences,
        dtype=np.int64,
    )


    seqlet_abs_attr_ratio = np.zeros(
        n_sequences,
        dtype=np.float64,
    )


    seqlet_signed_over_positive_attr_ratio = np.zeros(
        n_sequences,
        dtype=np.float64,
    )


    # -----------------------------------------------------
    # NEW:
    # undefined cases should be NaN
    # -----------------------------------------------------

    seqlet_motif_diversity = np.full(
        n_sequences,
        np.nan,
        dtype=np.float64,
    )


    seqlet_high_similarity_distance = np.full(
        n_sequences,
        np.nan,
        dtype=np.float64,
    )


    # =====================================================
    # calculate metrics per sequence
    # =====================================================

    eps = 1e-8


    for idx in range(
        n_sequences
    ):

        info = seqlet_info[idx]

        regions = info[
            "regions"
        ]


        seqlet_count[idx] = (
            info["seqlet_count"]
        )


        # =================================================
        # no seqlet
        # =================================================

        if len(regions) == 0:
            continue


        attr = attributions[idx]


        # =================================================
        # 1. union of seqlet regions
        # =================================================

        merged_regions = (
            merge_regions(
                regions
            )
        )


        seqlet_mask = np.zeros(
            sequence_length,
            dtype=bool,
        )


        for start, end in merged_regions:

            seqlet_mask[
                start:end
            ] = True


        # =================================================
        # 2. absolute attribution ratio
        # =================================================

        abs_attr = np.abs(
            attr
        ).sum(
            axis=1
        )


        total_abs_attr = (
            abs_attr.sum()
        )


        seqlet_abs_attr = (
            abs_attr[
                seqlet_mask
            ].sum()
        )


        seqlet_abs_attr_ratio[
            idx
        ] = (
            seqlet_abs_attr
            /
            (
                total_abs_attr
                +
                eps
            )
        )


        # =================================================
        # 3. signed attr in seqlets /
        #    total positive attribution
        # =================================================

        signed_attr = (
            attr.sum(
                axis=1
            )
        )


        positive_attr = (
            np.maximum(
                attr,
                0,
            )
            .sum(
                axis=1
            )
        )


        seqlet_signed_attr = (
            signed_attr[
                seqlet_mask
            ].sum()
        )


        total_positive_attr = (
            positive_attr.sum()
        )


        seqlet_signed_over_positive_attr_ratio[
            idx
        ] = (
            seqlet_signed_attr
            /
            (
                total_positive_attr
                +
                eps
            )
        )


        # =================================================
        # 4. collect motif features for each seqlet
        # =================================================
        #
        # Only seqlets with >=1 overlapping motif are stored.
        #
        # Example:
        #
        # seqlet_motif_sets =
        #
        # [
        #     {"IRF8", "STAT2"},
        #     {"IRF8", "STAT2", "IRF4"},
        #     {"CEBPB"}
        # ]
        #
        # seqlet_centers =
        #
        # [
        #     24.0,
        #     61.0,
        #     105.0
        # ]
        #
        # The two lists are positionally aligned.
        # =================================================

        seqlet_motif_sets = []

        seqlet_centers = []


        current_motifs = (
            motif_dict.get(
                str(idx),
                []
            )
        )


        for (
            seqlet_start,
            seqlet_end,
        ) in regions:

            seqlet_length = (
                seqlet_end
                -
                seqlet_start
            )


            if seqlet_length <= 0:
                continue


            overlapping_motifs = set()


            for (
                motif_start,
                motif_end,
                motif_name,
            ) in current_motifs:

                motif_length = (
                    motif_end
                    -
                    motif_start
                )


                if motif_length <= 0:
                    continue


                overlap = (
                    min(
                        seqlet_end,
                        motif_end,
                    )
                    -
                    max(
                        seqlet_start,
                        motif_start,
                    )
                )


                if overlap <= 0:
                    continue


                seqlet_overlap_ratio = (
                    overlap
                    /
                    seqlet_length
                )


                motif_overlap_ratio = (
                    overlap
                    /
                    motif_length
                )


                if (
                    seqlet_overlap_ratio
                    >= overlap_threshold
                    or
                    motif_overlap_ratio
                    >= overlap_threshold
                ):

                    overlapping_motifs.add(
                        motif_name
                    )


            # ---------------------------------------------
            # only motif-overlapping seqlets participate
            # in motif diversity / similarity analysis
            # ---------------------------------------------

            if len(
                overlapping_motifs
            ) > 0:

                seqlet_motif_sets.append(
                    overlapping_motifs
                )


                seqlet_centers.append(
                    (
                        seqlet_start
                        +
                        seqlet_end
                    )
                    /
                    2.0
                )


        # =================================================
        # 5. seqlet overlap motif count
        # =================================================

        overlap_seqlet_count = len(
            seqlet_motif_sets
        )


        seqlet_overlap_motif_count[
            idx
        ] = (
            overlap_seqlet_count
        )


        # =================================================
        # 6. NEW:
        # seqlet motif diversity
        # =================================================
        #
        # Only defined when at least two seqlets have
        # overlapping motifs.
        #
        # consistency =
        #
        #       |intersection of all motif sets|
        #       --------------------------------
        #          |union of all motif sets|
        #
        # diversity =
        #
        #       1 - consistency
        #
        # -------------------------------------------------
        # Examples:
        #
        # [{A,B}, {A,B}]
        #
        # intersection={A,B}
        # union={A,B}
        #
        # diversity = 0
        #
        #
        # [{A,B}, {C,D}]
        #
        # intersection={}
        # union={A,B,C,D}
        #
        # diversity = 1
        # =================================================

        if overlap_seqlet_count >= 2:

            common_motifs = (
                set.intersection(
                    *seqlet_motif_sets
                )
            )


            union_motifs = (
                set.union(
                    *seqlet_motif_sets
                )
            )


            if len(union_motifs) > 0:

                motif_consistency = (
                    len(
                        common_motifs
                    )
                    /
                    len(
                        union_motifs
                    )
                )


                seqlet_motif_diversity[
                    idx
                ] = (
                    1.0
                    -
                    motif_consistency
                )


        # =================================================
        # 7. NEW:
        # highly similar seqlet distance
        # =================================================
        #
        # This metric is only defined when motif diversity
        # itself is defined, i.e. at least two seqlets have
        # motif overlap.
        #
        # Pairwise Jaccard:
        #
        #     |Mi ∩ Mj|
        #     ---------
        #     |Mi ∪ Mj|
        #
        # If similarity >= similarity_threshold:
        #
        #     distance =
        #         |center_i - center_j|
        #
        # If multiple qualifying pairs exist:
        #
        #     use mean distance.
        #
        # If no qualifying pair exists:
        #
        #     NaN
        # =================================================

        if not np.isnan(
            seqlet_motif_diversity[
                idx
            ]
        ):

            distances = []


            for i in range(
                len(seqlet_motif_sets)
            ):

                for j in range(
                    i + 1,
                    len(seqlet_motif_sets)
                ):

                    set_i = (
                        seqlet_motif_sets[i]
                    )

                    set_j = (
                        seqlet_motif_sets[j]
                    )


                    union = (
                        set_i
                        |
                        set_j
                    )


                    if len(union) == 0:
                        continue


                    intersection = (
                        set_i
                        &
                        set_j
                    )


                    similarity = (
                        len(
                            intersection
                        )
                        /
                        len(
                            union
                        )
                    )


                    if (
                        similarity
                        >=
                        similarity_threshold
                    ):

                        distance = abs(
                            seqlet_centers[i]
                            -
                            seqlet_centers[j]
                        )


                        distances.append(
                            distance
                        )


            # ---------------------------------------------
            # multiple qualifying pairs:
            # use their MEAN distance
            # ---------------------------------------------

            if len(distances) > 0:

                seqlet_high_similarity_distance[
                    idx
                ] = float(
                    np.mean(
                        distances
                    )
                )


    # =====================================================
    # output
    # =====================================================

    result = pd.DataFrame(
        {
            "seqlet_count":
                seqlet_count,

            "seqlet_overlap_motif_count":
                seqlet_overlap_motif_count,

            "seqlet_abs_attr_ratio":
                seqlet_abs_attr_ratio,

            "seqlet_signed_over_positive_attr_ratio":
                seqlet_signed_over_positive_attr_ratio,

            "seqlet_motif_diversity":
                seqlet_motif_diversity,

            "seqlet_high_similarity_distance":
                seqlet_high_similarity_distance,
        },
        index=df.index,
    )


    return result


# Preserve the historical import path while using the updated implementation.
from seqtrust.tools.seqlet_metrics import (  # noqa: E402,F401
    calculate_seqlet_features as calculate_seqlet_features,
)
