"""Reproduce the single-objective SeqTrust benchmark and its summary report."""

from __future__ import annotations

import contextlib
import importlib.util
import io
import time
from pathlib import Path

from seqtrust import SeqTrust
from seqtrust.reproduce_benchmark.summary_report import summary_report

_DNABERT2_MODEL = (
    "/home/wangyucheng/projects/Benchmark/Evaluation/scripts/predictors/"
    "zhihan1996/DNABERT-2-117M"
)


def _load_reference_data():
    """Load checkpoint paths and reference motifs from the tutorial loader."""
    loader_path = Path(__file__).resolve().parents[2] / "tutorials" / "load_results.py"
    spec = importlib.util.spec_from_file_location(
        "_seqtrust_tutorial_load_results",
        loader_path,
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load reference data from {loader_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    complete_ckpts_path, _offline = module.load_ckpts_path()
    motif_enrichment = module.load_motif_enrichment(verbose=False)
    return complete_ckpts_path, motif_enrichment


def _format_timing(task: str, total_seconds: float, parts: list[tuple[str, float]]) -> str:
    part_text = " | ".join(f"{name} {seconds:.1f}s" for name, seconds in parts)
    return f"[{task}] total {total_seconds / 60:.2f} min | {part_text}"


def _task_path(output_dir: Path, task: str) -> Path:
    dataset, celltype = task.rsplit("_", 1)
    return output_dir / f"{dataset}_{celltype}.pkl"


def _load_existing(task: str, path: Path):
    print(f"[{task}] computed file already exists, loading it directly")
    return SeqTrust.load(path)


def reproduce_benchmark(task_seqs_dict: dict, output_dir: str | Path) -> dict:
    """Compute SeqTrust features for each task and save one pickle per task.

    Parameters
    ----------
    task_seqs_dict :
        Mapping of ``<modality>_<cell>`` to a DataFrame with ``Sequence``
        and ``Tools`` columns.
    output_dir :
        Directory for ``<modality>_<cell>.pkl`` files.

    Returns
    -------
    dict
        Mapping of task name to a ``SeqTrust`` object. Existing pickle files
        are loaded. A task is computed only when its file is missing.

    Each computed task prints one line: total minutes, then seconds for
    creating the object, sequence metrics, model embeddings and predictions,
    motif metrics, and attribution/seqlet metrics.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {task: _task_path(output_dir, task) for task in task_seqs_dict}

    if not paths or all(path.exists() for path in paths.values()):
        return {task: _load_existing(task, path) for task, path in paths.items()}

    complete_ckpts_path, motif_enrichment = _load_reference_data()
    task_seqs_st = {}

    for task, frame in task_seqs_dict.items():
        output_path = paths[task]
        if output_path.exists():
            task_seqs_st[task] = _load_existing(task, output_path)
            continue

        dataset, celltype = task.rsplit("_", 1)
        ckpt_path = complete_ckpts_path[f"{dataset}_{celltype}"]
        captured = io.StringIO()
        total_start = time.perf_counter()
        parts: list[tuple[str, float]] = []
        try:
            with contextlib.redirect_stdout(captured):
                stage_start = time.perf_counter()
                st = SeqTrust(frame.copy(), sequence_col="Sequence")
                parts.append(("Create SeqTrust object", time.perf_counter() - stage_start))

                stage_start = time.perf_counter()
                st.cal.add_GC_content(output_col="GC_content")
                st.cal.add_complexity_score(output_col="complexity_score")
                st.cal.add_max_repeat_length(output_col="max_repeat_length")
                st.cal.add_max_tandem_repeat_length(
                    output_col="max_tandem_repeat_length",
                    min_unit_size=6,
                    min_copies=2,
                )
                st.cal.add_max_dispersed_repeat_count(
                    output_col="max_dispersed_repeat_count",
                    unit_size=3,
                )
                st.cal.add_kmer_diversity(
                    output_col="kmer_diversity",
                    kmer_size=3,
                    normalize=True,
                )
                st.cal.add_kmer_embedding(
                    output_col="kmer_embedding",
                    kmer_size=6,
                )
                parts.append(("Lightweight sequence calculations", time.perf_counter() - stage_start))

                stage_start = time.perf_counter()
                st.cal.add_seq2function_embedding(
                    model_type="alphagenome",
                    output_col="alphagenome_embedding",
                    pretrained_path="default",
                    load_ckpt=True,
                    ckpt_path=ckpt_path,
                    length=200,
                    batch_size=128,
                )
                st.cal.add_seq2function_prediction(
                    model_type="alphagenome",
                    output_col="alphagenome_prediction",
                    pretrained_path="default",
                    load_ckpt=True,
                    ckpt_path=ckpt_path,
                    length=200,
                    batch_size=128,
                )
                st.cal.add_DNABERT2_embedding(
                    model_path=_DNABERT2_MODEL,
                    output_col="dnabert2_embedding",
                    conda_env="dnabert2",
                    batch_size=32,
                    max_length=200,
                )
                parts.append(("Model-based embeddings and predictions", time.perf_counter() - stage_start))

                stage_start = time.perf_counter()
                st.cal.add_motif_features(
                    meme_path=None,
                    motif_clusters=None,
                    source="HOCOMOCO",
                    output_type="binary",
                    output_col="motif_features",
                    qval_cutoff=0.05,
                    pval_cutoff=1e-4,
                    scan_store_key="motif_scan",
                    show_progress=False,
                )
                st.cal.add_motif_diversity(motif_col="motif_features")
                st.cal.add_motif_recovery(
                    reference_motifs=motif_enrichment[f"{dataset}_{celltype}"],
                    motif_col="motif_features",
                    output_col="motif_recovery_ratio",
                    overlap_col="motif_recovery_overlap",
                )
                parts.append(("Motif scan, diversity, and recovery", time.perf_counter() - stage_start))

                stage_start = time.perf_counter()
                st.cal.add_seq2function_attribution(
                    model_type="alphagenome",
                    pretrained_path="default",
                    load_ckpt=True,
                    ckpt_path=ckpt_path,
                    length=200,
                    batch_size=1280,
                    store_key="attribution_alphagenome",
                    background_num=30,
                )
                st.cal.add_seqlets(
                    attribution_key="attribution_alphagenome",
                    store_key="seqlets",
                    window_size=10,
                    flank_size=2,
                    target_fdr=0.05,
                    min_passing_windows_frac=0.1,
                    max_passing_windows_frac=0.4,
                    weak_threshold_for_counting_sign=0.8,
                )
                st.cal.add_seqlet_features(
                    motif_list=None,
                    attribution_key="attribution_alphagenome",
                    seqlet_key="seqlets",
                    seqlet_type="pos_seqlets",
                    motif_scan_key="motif_scan",
                    tau=0.75,
                    pseudo_overlap_min=6,
                    overlap_min=4,
                    uniq_k=6,
                    uniq_min_match=5,
                    fraction_1=0.75,
                    fraction_2=0.75,
                    min_motif_ov=1,
                    single_base_ratio=1.0,
                    comb_min_gap=1,
                    comb_max_gap=20,
                    bridge_frac=0.8,
                    min_seqlets=3,
                    collect_pairs=True,
                    detail_store_key="seqlet_feature_details",
                )
                parts.append(("Attribution, seqlet extraction, and features", time.perf_counter() - stage_start))

                st.save(output_path)
        except Exception:
            print(captured.getvalue(), end="")
            raise

        task_seqs_st[task] = st
        print(_format_timing(task, time.perf_counter() - total_start, parts))

    return task_seqs_st
