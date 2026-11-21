from __future__ import annotations

import numpy as np


def _one_hot_sequences(sequences, length: int) -> np.ndarray:
    """Center-crop/pad DNA strings and return an (N, L, 4) array."""
    alphabet = {"A": 0, "C": 1, "G": 2, "T": 3}
    encoded = np.zeros((len(sequences), length, 4), dtype=np.float32)
    for row, raw_sequence in enumerate(sequences):
        sequence = str(raw_sequence).upper()
        if len(sequence) > length:
            start = (len(sequence) - length) // 2
            sequence = sequence[start : start + length]
        elif len(sequence) < length:
            left = (length - len(sequence)) // 2
            sequence = "N" * left + sequence + "N" * (length - len(sequence) - left)
        for position, base in enumerate(sequence):
            base_index = alphabet.get(base)
            if base_index is not None:
                encoded[row, position, base_index] = 1.0
    return encoded


class AttributionMixin:
    """Seq2Function attribution, seqlet extraction, and seqlet features."""

    def add_seq2function_attribution(
        self,
        model_type: str,
        pretrained_path: str = "default",
        load_ckpt: bool = False,
        ckpt_path: str | None = None,
        length: int = 200,
        batch_size: int = 128,
        whether_softplus: bool = False,
        store_key: str = "seq2function_attribution",
        background_num: int = 50,
    ):
        """Calculate attribution maps once across the complete DataFrame."""
        from seqtrust.tools.embeddings import calculate_seq2function_attribution

        attribution = calculate_seq2function_attribution(
            self.trust.df,
            sequence_col=self.sequence_col,
            model_type=model_type,
            pretrained_path=pretrained_path,
            load_ckpt=load_ckpt,
            ckpt_path=ckpt_path,
            length=length,
            batch_size=batch_size,
            whether_softplus=whether_softplus,
            background_num=background_num,
        )
        self.trust.results[store_key] = {"all": attribution}
        self._record(
            "add_seq2function_attribution",
            model_type=model_type,
            store_key=store_key,
            length=length,
            batch_size=batch_size,
            background_num=background_num,
        )
        return attribution

    def add_seqlets(
        self,
        attribution_key: str = "seq2function_attribution",
        store_key: str = "seqlets",
        window_size: int = 10,
        flank_size: int = 2,
        target_fdr: float = 0.05,
        min_passing_windows_frac: float = 0.1,
        max_passing_windows_frac: float = 0.4,
        weak_threshold_for_counting_sign: float = 0.8,
    ):
        """Extract and group positive and negative modisco-lite seqlets.

        The calculation follows the reference workflow: observed-base
        contributions are ``one_hot * attributions``; modisco-lite extracts
        coordinates from their per-base sum; and the core-window attribution
        sum determines whether each seqlet is positive or negative.
        """
        from seqtrust.tools.seqlets import extract_modisco_seqlets

        if attribution_key not in self.trust.results:
            raise KeyError(
                f"Missing result {attribution_key!r}; run "
                "add_seq2function_attribution() first."
            )
        stored = self.trust.results[attribution_key]
        attributions = np.asarray(
            stored["all"] if isinstance(stored, dict) and "all" in stored else stored,
            dtype=np.float32,
        )
        if attributions.ndim != 3 or attributions.shape[-1] != 4:
            raise ValueError(
                "Attributions must have shape (N, L, 4); "
                f"received {attributions.shape}."
            )
        sequences = _one_hot_sequences(
            self.trust.df[self.sequence_col].tolist(),
            length=attributions.shape[1],
        )
        if sequences.shape != attributions.shape:
            raise ValueError(
                f"Sequence shape {sequences.shape} does not match attribution "
                f"shape {attributions.shape}."
            )

        seqlets, positive, negative, threshold = extract_modisco_seqlets(
            sequences=sequences,
            attributions=attributions,
            window_size=window_size,
            flank_size=flank_size,
            target_fdr=target_fdr,
            min_passing_windows_frac=min_passing_windows_frac,
            max_passing_windows_frac=max_passing_windows_frac,
            weak_threshold_for_counting_sign=weak_threshold_for_counting_sign,
        )
        seqlets_by_sequence: dict[int, list] = {}
        for seqlet in seqlets:
            position = int(seqlet.example_idx)
            seqlets_by_sequence.setdefault(position, []).append(seqlet)

        result = {
            "seqlets": seqlets,
            "pos_seqlets": positive,
            "neg_seqlets": negative,
            "seqlets_by_sequence": seqlets_by_sequence,
            "threshold": threshold,
            "n_total_sequences": len(self.trust.df),
            "n_sequences_with_seqlets": len(seqlets_by_sequence),
            "n_seqlets_all": len(seqlets),
            "n_seqlets_positive": len(positive),
            "n_seqlets_negative": len(negative),
            # Backward-compatible aliases used by earlier SeqTrust releases.
            "n_seqlets": len(seqlets),
            "n_pos_seqlets": len(positive),
            "n_neg_seqlets": len(negative),
        }
        self.trust.results[store_key] = result
        self._record(
            "add_seqlets",
            attribution_key=attribution_key,
            store_key=store_key,
            window_size=window_size,
            flank_size=flank_size,
            target_fdr=target_fdr,
        )
        return result

    def add_seqlet_features(
        self,
        motif_list=None,
        attribution_key: str = "seq2function_attribution",
        seqlet_key: str = "seqlets",
        seqlet_type: str = "pos_seqlets",
        motif_scan_key: str = "motif_scan",
        tau: float = 0.75,
        pseudo_overlap_min: int = 6,
        overlap_min: int | None = 4,
        uniq_k: int = 6,
        uniq_min_match: int = 5,
        fraction_1: float = 0.75,
        fraction_2: float = 0.75,
        min_motif_ov: int = 1,
        single_base_ratio: float = 1.0,
        comb_min_gap: int = 1,
        comb_max_gap: int = 20,
        bridge_frac: float = 0.8,
        min_seqlets: int = 3,
        collect_pairs: bool = True,
        detail_store_key: str = "seqlet_feature_details",
        output_prefix: str = "",
    ):
        """Add the updated biological-plausibility metrics for seqlets.

        The calculation produces 46 per-sequence metrics covering raw,
        motif-augmented, entropy-valid, motif-matched, self-redundant,
        combinatorial, and adjacent-heterotypic seqlet sets. ``motif_list``
        optionally limits the motif scan; ``None`` uses every scanned motif.
        Detailed combinatorial pair tables are stored under
        ``detail_store_key`` when ``collect_pairs`` is true.
        """
        from seqtrust.tools.seqlet_metrics import calculate_seqlet_features

        missing = [
            key
            for key in (attribution_key, seqlet_key, motif_scan_key)
            if key not in self.trust.results
        ]
        if missing:
            raise KeyError(f"Missing required SeqTrust results: {missing}.")

        stored_attribution = self.trust.results[attribution_key]
        attributions = np.asarray(
            stored_attribution["all"]
            if isinstance(stored_attribution, dict) and "all" in stored_attribution
            else stored_attribution,
            dtype=np.float32,
        )
        seqlet_result = self.trust.results[seqlet_key]
        if seqlet_type not in seqlet_result:
            raise KeyError(
                f"Unknown seqlet_type {seqlet_type!r}; "
                f"available values: {list(seqlet_result)}"
            )
        if "seqlets" not in seqlet_result:
            raise KeyError(
                f"Result {seqlet_key!r} has no complete 'seqlets' collection."
            )

        sequences = _one_hot_sequences(
            self.trust.df[self.sequence_col].tolist(),
            length=attributions.shape[1],
        )
        params = {
            "tau": tau,
            "pseudo_overlap_min": pseudo_overlap_min,
            "overlap_min": overlap_min,
            "uniq_k": uniq_k,
            "uniq_min_match": uniq_min_match,
            "fraction_1": fraction_1,
            "fraction_2": fraction_2,
            "min_motif_ov": min_motif_ov,
            "single_base_ratio": single_base_ratio,
            "comb_min_gap": comb_min_gap,
            "comb_max_gap": comb_max_gap,
            "bridge_frac": bridge_frac,
            "min_seqlets": min_seqlets,
        }

        feature_df, details = calculate_seqlet_features(
            df=self.trust.df.copy(),
            sequences=sequences,
            attributions=attributions,
            positive_seqlets=seqlet_result[seqlet_type],
            all_seqlets=seqlet_result["seqlets"],
            motif_scan=self.trust.results[motif_scan_key],
            motif_list=motif_list,
            params=params,
            collect_pairs=collect_pairs,
        )
        if output_prefix:
            feature_df = feature_df.add_prefix(output_prefix)
        for column in feature_df:
            self.trust.df[column] = feature_df[column].to_numpy()
        self.trust.results[detail_store_key] = details
        self._record(
            "add_seqlet_features",
            attribution_key=attribution_key,
            seqlet_key=seqlet_key,
            seqlet_type=seqlet_type,
            motif_scan_key=motif_scan_key,
            detail_store_key=detail_store_key,
            params=params,
            output_cols=list(feature_df.columns),
        )
        return self.trust.df
