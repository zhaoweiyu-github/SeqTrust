from __future__ import annotations

import pickle

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from seqtrust import config


def _resolve(value, fallback):
    """Return the per-call argument, falling back to the config default."""
    return fallback if value is None else value


class AnalysisMixin:
    """Core analysis functions applied to the whole record collection.

    This class is the base mixin of ``SeqTrust``; all ``add_*`` methods
    are called directly on the evaluator, e.g.::

        st = SeqTrust(df)
        st.add_seq2func_embedding(model_type="alphagenome")

    Methods rely on the evaluator attributes ``df``, ``sequence_col``,
    ``history``, ``feature_names`` and ``results``.
    """

    # ------------------------------------------------------------------
    # seq2func embedding
    # ------------------------------------------------------------------
    def add_seq2func_embedding(
        self,
        model_type: str,
        output_col: str = "xxx_embedding",
        pretrained_path: str = "default",
        load_ckpt: bool = False,
        ckpt_path: Optional[str] = None,
        length: int | None = None,
        batch_size: int | None = None,
        whether_softplus: bool = False,
    ) -> None:
        """
        Add Seq2Function trunk embedding features to ``self.df``.

        Parameters
        ----------
        model_type
            Seq2Function model type, e.g. "alphagenome", "enformer",
            "borzoi", "borzoi_transformer", "sei".

        output_col
            Column name for storing embedding vectors.

        pretrained_path
            Path of pretrained weights. "default" uses the built-in
            MODEL_PRETRAINED_MAP.

        load_ckpt
            Whether to load a fine-tuned checkpoint.

        ckpt_path
            Checkpoint path, required when load_ckpt=True.

        length
            Sequence length fed into the model.
            None falls back to ``config.DEFAULT_LENGTH``.

        batch_size
            Inference batch size.
            None falls back to ``config.DEFAULT_BATCH_SIZE``.

        whether_softplus
            Whether the model head applies softplus.
        """

        from seqtrust.tools.seq2func import calculate_seq2func_embedding

        length = _resolve(length, config.DEFAULT_LENGTH)
        batch_size = _resolve(batch_size, config.DEFAULT_BATCH_SIZE)

        result = calculate_seq2func_embedding(
            self.df,
            sequence_col=self.sequence_col,
            model_type=model_type,
            output_col=output_col,
            pretrained_path=pretrained_path,
            load_ckpt=load_ckpt,
            ckpt_path=ckpt_path,
            length=length,
            batch_size=batch_size,
            whether_softplus=whether_softplus,
        )

        self.df[output_col] = list(result[output_col])

        self.history.append(
            f"add_seq2func_embedding: "
            f"model_type={model_type}, "
            f"output_col={output_col}, "
            f"load_ckpt={load_ckpt}, "
            f"length={length}, "
            f"batch_size={batch_size}, "
            f"whether_softplus={whether_softplus}"
        )

    # ------------------------------------------------------------------
    # seq2func prediction
    # ------------------------------------------------------------------
    def add_seq2func_prediction(
        self,
        model_type: str,
        output_col: str = "xxx_prediction",
        pretrained_path: str = "default",
        load_ckpt: bool = False,
        ckpt_path: Optional[str] = None,
        length: int | None = None,
        batch_size: int | None = None,
        whether_softplus: bool = False,
    ) -> None:
        """
        Add Seq2Function scalar predictions to ``self.df``.

        Parameters
        ----------
        model_type
            Seq2Function model type, e.g. "alphagenome", "enformer",
            "borzoi", "borzoi_transformer", "sei".

        output_col
            Column name for storing predictions.

        pretrained_path
            Path of pretrained weights. "default" uses the built-in
            MODEL_PRETRAINED_MAP.

        load_ckpt
            Whether to load a fine-tuned checkpoint.

        ckpt_path
            Checkpoint path, required when load_ckpt=True.

        length
            Sequence length fed into the model.
            None falls back to ``config.DEFAULT_LENGTH``.

        batch_size
            Inference batch size.
            None falls back to ``config.DEFAULT_BATCH_SIZE``.

        whether_softplus
            Whether the model head applies softplus.
        """

        from seqtrust.tools.seq2func import calculate_seq2func_prediction

        length = _resolve(length, config.DEFAULT_LENGTH)
        batch_size = _resolve(batch_size, config.DEFAULT_BATCH_SIZE)

        result = calculate_seq2func_prediction(
            self.df,
            sequence_col=self.sequence_col,
            model_type=model_type,
            output_col=output_col,
            pretrained_path=pretrained_path,
            load_ckpt=load_ckpt,
            ckpt_path=ckpt_path,
            length=length,
            batch_size=batch_size,
            whether_softplus=whether_softplus,
        )

        self.df[output_col] = list(result[output_col])

        self.history.append(
            f"add_seq2func_prediction: "
            f"model_type={model_type}, "
            f"output_col={output_col}, "
            f"load_ckpt={load_ckpt}, "
            f"length={length}, "
            f"batch_size={batch_size}, "
            f"whether_softplus={whether_softplus}"
        )

    # ------------------------------------------------------------------
    # seq2func attribution
    # ------------------------------------------------------------------
    def add_seq2func_attribution(
        self,
        model_type: str,
        pretrained_path: str = "default",
        load_ckpt: bool = False,
        ckpt_path: Optional[str] = None,
        length: int | None = None,
        batch_size: int | None = None,
        whether_softplus: bool = False,
        store_key: str | None = None,
        background_num: int | None = None,
    ) -> None:
        """
        Calculate attribution maps for all sequences.

        The attribution array of shape (N, L, 4) is stored in::

            st.results[store_key]

        Parameters
        ----------
        model_type
            Seq2Function model type.

        pretrained_path
            Path of pretrained weights. "default" uses the built-in
            MODEL_PRETRAINED_MAP.

        load_ckpt
            Whether to load a fine-tuned checkpoint.

        ckpt_path
            Checkpoint path, required when load_ckpt=True.

        length
            Sequence length fed into the model.
            None falls back to ``config.DEFAULT_LENGTH``.

        batch_size
            Inference batch size.
            None falls back to ``config.DEFAULT_BATCH_SIZE``.

        whether_softplus
            Whether the model head applies softplus.

        store_key
            Key used to store the attribution array in ``self.results``.
            Default: "seq2func_attribution".

        background_num
            Number of dinucleotide-shuffled background sequences used
            by the gradient explainer.
            None falls back to ``config.DEFAULT_BACKGROUND_NUM``.
        """

        from seqtrust.tools.seq2func import calculate_seq2func_attribution

        length = _resolve(length, config.DEFAULT_LENGTH)
        batch_size = _resolve(batch_size, config.DEFAULT_BATCH_SIZE)
        background_num = _resolve(background_num, config.DEFAULT_BACKGROUND_NUM)

        attr = calculate_seq2func_attribution(
            self.df,
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

        key = store_key or "seq2func_attribution"

        self.results[key] = attr

        self.history.append(
            f"add_seq2func_attribution: "
            f"model_type={model_type}, "
            f"store_key={key}, "
            f"length={length}, "
            f"batch_size={batch_size}, "
            f"background_num={background_num}"
        )

    # ------------------------------------------------------------------
    # motif features
    # ------------------------------------------------------------------
    def add_motif(
        self,
        meme_path: str | None = None,
        motif_clusters: str | None = None,
        source: str = "HOCOMOCO",
        use_uniform_bg: bool = False,
        bg_freq=None,
        show_progress: bool = False,
        output_type: str = "count",  # binary / count / score_max
        output_col: str = "motif_features",
        threshold: float = 1e-4,
        qval_cutoff: float = 0.05,
        pval_cutoff: float = 1e-4,
    ) -> None:
        """
        Scan sequences with FIMO and add per-sequence motif features.

        The motif feature matrix (one row per sequence, one column per
        motif) is stored as vectors in ``self.df[output_col]``; motif
        names are stored in ``self.feature_names[output_col]`` and the
        positional scan table is stored in::

            st.results["motif_scan"]

        Parameters
        ----------
        meme_path
            MEME motif file. None uses the bundled HOCOMOCO file.

        motif_clusters
            Motif cluster file. None uses the bundled HOCOMOCO file.

        source
            Motif source, "HOCOMOCO" or "JASPAR".

        use_uniform_bg
            Whether to use a uniform A/C/G/T background.

        bg_freq
            User-specified background nucleotide frequencies: a dict
            keyed by A/C/G/T, or 4 values in A, C, G, T order.
            Takes precedence over use_uniform_bg. When None, the
            background is computed from the input sequences.

        show_progress
            Whether to show a progress bar during scanning.

        output_type
            "binary", "count" or "score_max".

        output_col
            Column name for storing motif feature vectors.

        threshold
            FIMO p-value threshold used during scanning.

        qval_cutoff
            q-value cutoff applied to scan hits.

        pval_cutoff
            p-value cutoff applied to scan hits.
        """

        from seqtrust.tools.motif import calculate_motif_features

        df_out, motif_names, scan_df = calculate_motif_features(
            self.df,
            meme_path=meme_path,
            motif_clusters=motif_clusters,
            source=source,
            sequence_col=self.sequence_col,
            use_uniform_bg=use_uniform_bg,
            bg_freq=bg_freq,
            show_progress=show_progress,
            output_type=output_type,
            feature_col=output_col,
            threshold=threshold,
            qval_cutoff=qval_cutoff,
            pval_cutoff=pval_cutoff,
        )

        self.df[output_col] = list(df_out[output_col])

        self.feature_names[output_col] = list(motif_names)

        if scan_df is not None:
            self.results["motif_scan"] = scan_df
        else:
            self.results["motif_scan"] = pd.DataFrame()

        self.history.append(
            f"add_motif: "
            f"source={source}, "
            f"output_col={output_col}, "
            f"output_type={output_type}, "
            f"use_uniform_bg={use_uniform_bg}, "
            f"bg_freq={'custom' if bg_freq is not None else None}"
        )

    # ------------------------------------------------------------------
    # seqlets
    # ------------------------------------------------------------------
    def add_seqlet(
        self,
        attribution_key: str = "seq2func_attribution",
        seqlet_key: str = "seqlets",
        window_size: int = 10,
        flank_size: int = 8,
        target_fdr: float = 0.05,
        min_passing_windows_frac: float = 0.1,
        max_passing_windows_frac: float = 0.4,
        weak_threshold_for_counting_sign: float = 0.8,
        task: str | None = None,
        save_path: str | None = None,
    ) -> None:
        """
        Extract seqlets from attribution maps with modisco-lite.

        Seqlet extraction is purely attribution-based; no motif
        information is involved. This method only produces and stores
        seqlets — no columns are added to ``self.df``. Metric columns
        derived from the seqlets (including motif-aware analyses) are
        added later by ``st.add_metrics(...)``.

        Required data must already exist in ``self.results``:

        - attribution maps, from ``st.add_seq2func_attribution(...)``

        The result dictionary is stored in::

            st.results[seqlet_key]

        containing:

            task, threshold, n_total_sequences,
            n_sequences_with_seqlets, n_seqlets, n_pos_seqlets,
            n_neg_seqlets, seqlets, pos_seqlets, neg_seqlets,
            seqlets_by_sequence

        When ``save_path`` is given, the same dictionary is also
        pickled to disk as a standalone ``*_seqlets.pkl`` file.

        Parameters
        ----------
        attribution_key
            Key of the attribution array in ``self.results``.
            Default: "seq2func_attribution".

        seqlet_key
            Key used to store extracted seqlets in ``self.results``.
            Default: "seqlets".

        window_size
            Core window size used for seqlet extraction.

        flank_size
            Flanking bases added to both sides of each seqlet.

        target_fdr
            Target FDR used by modisco-lite.

        min_passing_windows_frac
            Minimum passing-window fraction.

        max_passing_windows_frac
            Maximum passing-window fraction.

        weak_threshold_for_counting_sign
            Threshold used by modisco-lite when estimating
            positive/negative windows.

        task
            Optional label recorded in the seqlet result dictionary,
            e.g. a dataset or task name.

        save_path
            Optional output pickle path. When given, the seqlet result
            dictionary is saved to disk (parent directories are created
            automatically).
        """

        from enformer_pytorch import str_to_one_hot
        from seqtrust.tools.seqlet import (
            extract_seqlets,
            group_seqlets_by_sequence,
        )

        # =====================================================
        # 1. check required results
        # =====================================================

        if attribution_key not in self.results:

            raise KeyError(
                "Required data are missing from results: "
                f"[{attribution_key!r}]. "
                f"Available keys: {list(self.results.keys())}. "
                "Please run st.add_seq2func_attribution(...) first."
            )

        # =====================================================
        # 2. extract seqlets from attribution maps
        # =====================================================

        attributions = np.asarray(
            self.results[attribution_key],
            dtype=np.float32,
        )

        sequences = np.stack(
            [
                str_to_one_hot(seq)
                for seq in self.df[self.sequence_col]
            ],
            axis=0,
        )

        if sequences.shape != attributions.shape:

            raise ValueError(
                "Shape mismatch between sequences and attributions: "
                f"sequences.shape={sequences.shape}, "
                f"attributions.shape={attributions.shape}. "
                "Both must have the same shape (N, L, 4)."
            )

        (
            seqlets,
            pos_seqlets,
            neg_seqlets,
            threshold,
        ) = extract_seqlets(
            sequences=sequences,
            attributions=attributions,
            window_size=window_size,
            flank_size=flank_size,
            target_fdr=target_fdr,
            min_passing_windows_frac=min_passing_windows_frac,
            max_passing_windows_frac=max_passing_windows_frac,
            weak_threshold_for_counting_sign=weak_threshold_for_counting_sign,
        )

        # -----------------------------------------------------
        # group seqlets by source sequence
        # -----------------------------------------------------

        seqlets_by_sequence = group_seqlets_by_sequence(
            seqlets
        )

        seqlet_results = {
            "task": task,
            "threshold": threshold,
            "n_total_sequences": len(self.df),
            "n_sequences_with_seqlets": len(seqlets_by_sequence),
            "n_seqlets": len(seqlets),
            "n_pos_seqlets": len(pos_seqlets),
            "n_neg_seqlets": len(neg_seqlets),
            "seqlets": seqlets,
            "pos_seqlets": pos_seqlets,
            "neg_seqlets": neg_seqlets,
            "seqlets_by_sequence": seqlets_by_sequence,
        }

        self.results[seqlet_key] = seqlet_results

        print(
            f"[Seqlets] "
            f"threshold={threshold:.6g}, "
            f"total={len(seqlets)}, "
            f"positive={len(pos_seqlets)}, "
            f"negative={len(neg_seqlets)}, "
            f"{len(seqlets_by_sequence)}/{len(self.df)} "
            f"sequences have >=1 seqlet"
        )

        # =====================================================
        # 3. optionally save seqlets to a standalone pkl
        # =====================================================

        if save_path is not None:

            save_path = Path(save_path)
            save_path.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            with open(save_path, "wb") as f:
                pickle.dump(
                    seqlet_results,
                    f,
                    protocol=pickle.HIGHEST_PROTOCOL,
                )

            print(
                f"[Seqlets] Saved seqlets -> {save_path}"
            )

        # =====================================================
        # 4. history
        # =====================================================

        self.history.append(
            f"add_seqlet: "
            f"attribution_key={attribution_key}, "
            f"seqlet_key={seqlet_key}, "
            f"window_size={window_size}, "
            f"flank_size={flank_size}, "
            f"target_fdr={target_fdr}, "
            f"task={task}, "
            f"save_path={save_path}"
        )

    # ------------------------------------------------------------------
    # seqlet-based plausibility metrics
    # ------------------------------------------------------------------
    def add_metrics(
        self,
        seqlets_path: str | None = None,
        attribution_key: str = "seq2func_attribution",
        seqlet_key: str = "seqlets",
        motif_scan_key: str = "motif_scan",
        output_prefix: str = "",
        task: str | None = None,
        collect_pairs: bool = False,
        show_progress: bool = True,
        tau: float = 0.75,
        overlap_min: int | None = 4,
        pseudo_overlap_min: int = 6,
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
    ) -> None:
        """
        Compute seqlet-based plausibility metrics and add them as
        columns to ``self.df`` (46 columns per call).

        Required data must already exist in ``self.results``:

        - attribution maps, from ``st.add_seq2func_attribution(...)``
        - motif scan table, from ``st.add_motif(...)``

        Seqlets are read directly from ``st.results[seqlet_key]``
        (produced by ``st.add_seqlet(...)``); when ``seqlets_path`` is
        given, the seqlet dictionary is instead loaded from that pickle
        file, e.g. a standalone ``*_seqlets.pkl`` written by a previous
        ``st.add_seqlet(..., save_path=...)`` call.

        Per sequence, the metrics cover seven groups over the positive,
        motif-augmented and valid (entropy >= ``tau``) seqlet sets:

        1. number / coverage / span / pairwise uniqueness /
           nonredundant partner number / diversity;
        2. matched seqlet-motif counts and unique matched features;
        3. single-base negative-attribution within matched motifs;
        4. self-redundant seqlets with matched motifs;
        5. combinatorial seqlets-motif pairs;
        6. combinatorial pairs over all seqlets;
        7. adjacent heterotypic seqlets.

        Column names and groupings are available as
        ``seqtrust.tools.metric.ALL_COLUMNS`` and
        ``seqtrust.tools.metric.COLUMN_GROUPS``.

        Parameters
        ----------
        seqlets_path
            Optional path to a pickled seqlet dictionary. When None,
            seqlets are taken from ``self.results[seqlet_key]``.

        attribution_key
            Key of the attribution array in ``self.results``.
            Default: "seq2func_attribution".

        seqlet_key
            Key of the seqlet dictionary in ``self.results``.
            Default: "seqlets".

        motif_scan_key
            Key of the motif scan table in ``self.results``.
            Default: "motif_scan".

        output_prefix
            Optional prefix for generated column names.

        task
            Optional dataset label used in progress prints and in the
            pair-record tables.

        collect_pairs
            When True, the long-form combinatorial pair tables are
            stored in ``self.results["combinatorial_motif_pairs"]``
            and ``self.results["combinatorial_all_pairs"]``.

        show_progress
            Print progress and a summary of the computed metrics.

        tau
            Validity cut: normalized attribution entropy >= tau.

        overlap_min
            bp; greedy overlap suppression (NMS) among positive seqlets
            before metric computation. A seqlet overlapping a stronger
            kept one by more than this many bp is removed. None
            disables the suppression. The seqlets stored in
            ``self.results`` are never modified.

        pseudo_overlap_min
            bp; NMS among motif-augmented seqlet candidates.

        uniq_k
            k-mer size of the k-mer redundancy rule.

        uniq_min_match
            Min matching positions (of k) to call k-mers redundant.

        fraction_1
            Geometric overlap threshold for a matched seqlet-motif pair.

        fraction_2
            Attribution overlap threshold for a matched seqlet-motif
            pair.

        min_motif_ov
            bp; min motif-interval overlap for a "shared" motif.

        single_base_ratio
            A matched motif is flagged when |negative attribution| >
            ratio x mean(positive attribution in motif).

        comb_min_gap
            bp; combinatorial seqlets separated by >= this gap.

        comb_max_gap
            bp; combinatorial seqlets separated by < this gap.

        bridge_frac
            A matched motif must cover >= this fraction of a redundant
            k-mer pair span to "explain" the self-redundancy.

        min_seqlets
            Min seqlets for the adjacent-heterotypic metric.
        """

        from enformer_pytorch import str_to_one_hot
        from seqtrust.tools.metric import (
            compute_metrics,
            print_metrics_summary,
            suppress_overlapping_seqlets,
        )

        # =====================================================
        # 1. check required results
        # =====================================================

        missing_keys = [
            key
            for key in (attribution_key, motif_scan_key)
            if key not in self.results
        ]

        if missing_keys:

            raise KeyError(
                "Required data are missing from results: "
                f"{missing_keys}. "
                f"Available keys: {list(self.results.keys())}. "
                "Please run st.add_seq2func_attribution(...) and "
                "st.add_motif(...) first."
            )

        # =====================================================
        # 2. resolve seqlets: results first, pkl path otherwise
        # =====================================================

        if seqlets_path is None:

            if seqlet_key not in self.results:

                raise KeyError(
                    f"Seqlets were not found in results[{seqlet_key!r}]. "
                    f"Available keys: {list(self.results.keys())}. "
                    "Please run st.add_seqlet(...) first, or pass "
                    "seqlets_path to a pickled seqlet dictionary."
                )

            seqlet_results = self.results[seqlet_key]
            seqlet_source = f"results[{seqlet_key!r}]"

        else:

            with open(seqlets_path, "rb") as f:
                seqlet_results = pickle.load(f)

            seqlet_source = str(seqlets_path)

        missing_seqlet_keys = [
            key
            for key in ("pos_seqlets", "seqlets_by_sequence")
            if key not in seqlet_results
        ]

        if missing_seqlet_keys:

            raise KeyError(
                "The seqlet dictionary is missing required keys: "
                f"{missing_seqlet_keys}. "
                f"Available keys: {list(seqlet_results.keys())}."
            )

        # =====================================================
        # 3. sequences / attributions
        # =====================================================

        attributions = np.asarray(
            self.results[attribution_key],
            dtype=np.float32,
        )

        sequences = np.stack(
            [
                str_to_one_hot(seq)
                for seq in self.df[self.sequence_col]
            ],
            axis=0,
        )

        if sequences.shape != attributions.shape:

            raise ValueError(
                "Shape mismatch between sequences and attributions: "
                f"sequences.shape={sequences.shape}, "
                f"attributions.shape={attributions.shape}. "
                "Both must have the same shape (N, L, 4)."
            )

        # =====================================================
        # 4. overlap suppression among positive seqlets
        # =====================================================

        (
            pos_seqlets,
            seqlets_by_sequence,
            n_removed,
        ) = suppress_overlapping_seqlets(
            seqlets_by_sequence=seqlet_results["seqlets_by_sequence"],
            pos_seqlets=seqlet_results["pos_seqlets"],
            overlap_min=overlap_min,
        )

        # =====================================================
        # 5. per-base attribution + motif dict
        # =====================================================

        per_pos = (sequences * attributions).sum(axis=2)
        seq_mean_attr = per_pos.mean(axis=1)

        motif_scan = self.results[motif_scan_key]

        if not isinstance(motif_scan, pd.DataFrame):

            raise TypeError(
                f"results[{motif_scan_key!r}] must be a "
                "pandas DataFrame."
            )

        ms = motif_scan.reset_index()

        if "ID" not in ms.columns:

            raise KeyError(
                f"results[{motif_scan_key!r}] must contain an 'ID' "
                "column (sequence index) or use it as index."
            )

        ms["ID"] = ms["ID"].astype(int)
        ms["start0"] = ms[["start", "end"]].min(axis=1).astype(int)
        ms["end0"] = ms[["start", "end"]].max(axis=1).astype(int)

        motif_dict = {
            int(k): list(zip(g["start0"].values, g["end0"].values))
            for k, g in ms.groupby("ID")
        }

        # =====================================================
        # 6. compute metrics
        # =====================================================

        name = task or "dataset"

        data = {
            "name": name,
            "index": self.df.index,
            "n": len(self.df),
            "sequences": sequences,
            "per_pos": per_pos,
            "seq_mean_attr": seq_mean_attr,
            "motif_dict": motif_dict,
            "seqlets_by_sequence": seqlets_by_sequence,
            "pos_ids": {id(s) for s in pos_seqlets},
        }

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

        (
            metrics_df,
            motif_pairs_df,
            all_pairs_df,
        ) = compute_metrics(
            data,
            params,
            collect_pairs=collect_pairs,
            show_progress=show_progress,
        )

        # =====================================================
        # 7. add optional prefix
        # =====================================================

        if output_prefix:

            metrics_df = metrics_df.rename(
                columns={
                    col: f"{output_prefix}{col}"
                    for col in metrics_df.columns
                }
            )

        # =====================================================
        # 8. write into self.df
        # =====================================================

        for col in metrics_df.columns:

            self.df[col] = metrics_df[col]

        # =====================================================
        # 9. pair tables
        # =====================================================

        if collect_pairs:

            self.results["combinatorial_motif_pairs"] = motif_pairs_df
            self.results["combinatorial_all_pairs"] = all_pairs_df

        # =====================================================
        # 10. summary + history
        # =====================================================

        if show_progress:

            print(
                f"[Metrics] seqlets from {seqlet_source}: "
                f"kept {len(pos_seqlets)} positive seqlets "
                f"(overlap suppression removed {n_removed})"
            )

            print_metrics_summary(name, metrics_df)

        self.history.append(
            f"add_metrics: "
            f"seqlets_path={seqlets_path}, "
            f"attribution_key={attribution_key}, "
            f"seqlet_key={seqlet_key}, "
            f"motif_scan_key={motif_scan_key}, "
            f"task={task}, "
            f"collect_pairs={collect_pairs}, "
            f"params={params}, "
            f"output_cols={list(metrics_df.columns)}"
        )
