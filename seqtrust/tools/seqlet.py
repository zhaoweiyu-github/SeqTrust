from __future__ import annotations

import numpy as np

def extract_seqlets(
    sequences: np.ndarray,
    attributions: np.ndarray,
    window_size: int = 10,
    flank_size: int = 8,
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
    # SeqTrust can still be imported without modisco-lite installed.
    try:
        from modiscolite import core
        from modiscolite import extract_seqlets as modisco_extract_seqlets

    except ImportError as e:
        raise ImportError(
            "modiscolite is required for seqlet extraction. "
            "Please install modisco-lite before calling add_seqlet()."
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
        modisco_extract_seqlets.extract_seqlets(
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

def group_seqlets_by_sequence(seqlets) -> dict:
    """
    Group seqlets by their source sequence index.

    Parameters
    ----------
    seqlets
        Seqlet objects generated by modisco-lite.

        Each seqlet should contain ``seqlet.example_idx``.

    Returns
    -------
    dict
        Mapping of ``example_idx`` (int) to the list of seqlets
        extracted from that sequence.
    """

    by_sequence = {}

    for seqlet in seqlets:

        by_sequence.setdefault(
            int(seqlet.example_idx),
            [],
        ).append(
            seqlet
        )

    return by_sequence

