"""Assemble every Plot notebook figure into one sectioned PDF report."""

from __future__ import annotations

import copy
import math
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

SECTION_ACTIVITY = "Predicted Activity mean and top selected"
SECTION_RANK = "Rank correlation across datasets & celltype"
SECTION_RESOURCE = "time and memory usage"
SECTION_SEQLETS = "Seqlets associated"
SECTION_MOTIF = "Motif recovery"
SECTION_COMPOSITION = "Sequence composition"
SECTION_LOGO = "Attribution Logo"
SECTION_UMAP = "Embedding UMAP"

SEQLET_METRIC_LABELS = {
    "n_valid_pos_seqlets_aug": "Valid seqlets",
    "n_valid_pos_seqlet_with_matched_motif_aug": "Motif-matched",
    "valid_pos_seqlets_pair_uniq_aug": "Non-redundant",
    "n_valid_unique_matched_features_aug": "Unique matched\nmotifs",
    "n_valid_combinatorial_seqlet_all_pairs_aug": "Combinatorial\npairs",
    "valid_adjacent_heterotypic_seqlets_aug": "Adjacent heterotypic\npairs",
}

COMPOSITION_COLUMNS = [
    "GC_content",
    "complexity_score",
    "max_repeat_length",
    "max_tandem_repeat_length",
    "max_dispersed_repeat_count",
]

DEFAULT_RESOURCE_DF = pd.DataFrame(
    {
        "Tools": [
            "AdaLead",
            "FastSeqProp",
            "regLM",
            "SaturationMutagenesis",
            "SimulatedAnnealing",
            "TACO",
        ],
        "Time_hours": [1.5, 0.4, 5, 0.5, 0.3, 1.5],
        "Memory_GB": [4.59, 23.07, 22.36, 3.55, 3.56, 16.10],
    }
)


def summary_report(
    task_seqs_st: dict,
    output_path: str | Path,
    *,
    resource_df: pd.DataFrame | None = None,
    random_state: int = 42,
) -> Path:
    """Write one PDF report covering every figure in the Plot notebook.

    Parameters
    ----------
    task_seqs_st :
        Mapping of ``<modality>_<cell>`` to a ``SeqTrust`` object.
        Objects in this mapping are not modified.
    output_path :
        Destination ``.pdf``. Individual figure files are not kept.
    resource_df :
        Tool time and memory table. The notebook defaults are used when
        this is omitted.
    random_state :
        Seed for the AdaLead logo sequence and for UMAP.

    Returns
    -------
    Path
        The written report.
    """
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    output_path = Path(output_path)
    if output_path.suffix.lower() != ".pdf":
        raise ValueError("output_path must end with '.pdf'.")
    if not task_seqs_st:
        raise ValueError("task_seqs_st is empty.")
    resources = DEFAULT_RESOURCE_DF if resource_df is None else resource_df
    output_path.parent.mkdir(parents=True, exist_ok=True)

    temp_dir = tempfile.TemporaryDirectory()
    pdf = PdfPages(output_path)
    try:
        info = pdf.infodict()
        info["Title"] = "SeqTrust summary report"
        info["Creator"] = "SeqTrust"
        temp = Path(temp_dir.name)
        _section_activity(pdf, temp, task_seqs_st)
        _section_rank(pdf, temp, task_seqs_st)
        _section_resource(pdf, temp, task_seqs_st, resources)
        _section_seqlets(pdf, task_seqs_st)
        _section_motif(pdf, temp, task_seqs_st)
        _section_composition(pdf, temp, task_seqs_st)
        _section_logo(pdf, temp, task_seqs_st, random_state)
        _section_umap(pdf, temp, task_seqs_st, random_state)
    finally:
        pdf.close()
        temp_dir.cleanup()

    print(f"Summary report written to {output_path}")
    return output_path


def _add_section(pdf, title: str) -> None:
    import matplotlib.pyplot as plt

    print(f"[summary] {title}")
    fig = plt.figure(figsize=(11, 2.4))
    fig.patch.set_facecolor("white")
    fig.text(
        0.05,
        0.58,
        title,
        fontsize=16,
        fontweight="bold",
        ha="left",
        va="center",
        color="#1A1A1A",
    )
    fig.add_artist(
        plt.Line2D(
            [0.05, 0.95],
            [0.28, 0.28],
            transform=fig.transFigure,
            color="#1A1A1A",
            linewidth=1.0,
        )
    )
    pdf.savefig(fig, dpi=450, facecolor="white", edgecolor="white")
    plt.close(fig)


def _add_figure(pdf, fig) -> None:
    import matplotlib.pyplot as plt

    pdf.savefig(fig, dpi=450, facecolor="white", edgecolor="white")
    plt.close(fig)


def _figure_image(source_fig):
    source_fig.set_dpi(220)
    source_fig.canvas.draw()
    return np.asarray(source_fig.canvas.buffer_rgba())


def _dataset_objects(task_seqs_st: dict) -> dict:
    """Combine cell types of one modality without changing ``task_seqs_st``."""
    grouped: dict[str, list[tuple[str, object]]] = {}
    for task, st in task_seqs_st.items():
        dataset, celltype = task.rsplit("_", 1)
        grouped.setdefault(dataset, []).append((celltype, st))

    dataset_st = {}
    for dataset, items in grouped.items():
        base = copy.copy(items[0][1])
        frames = []
        for celltype, st in items:
            frame = st.df.copy()
            frame["Cell"] = celltype
            frames.append(frame)
        base.df = pd.concat(frames, ignore_index=True)
        base.cal = type(base.cal)(base)
        base.plot = type(base.plot)(base)
        dataset_st[dataset] = base
    return dataset_st


def _section_activity(pdf, temp: Path, task_seqs_st: dict) -> None:
    _add_section(pdf, SECTION_ACTIVITY)
    for dataset, st in _dataset_objects(task_seqs_st).items():
        fig, _ = st.plot.performance_matrix(
            tool_col="Tools",
            category_col="Cell",
            value_col="alphagenome_prediction",
            aggregation="mean",
            tools_on="rows",
            output_path=temp / f"performance_matrix_mean_{dataset}.pdf",
            title=f"{dataset}: mean predicted activity",
            value_label="mean score",
        )
        _add_figure(pdf, fig)
        fig, _ = st.plot.performance_matrix(
            tool_col="Tools",
            category_col="Cell",
            value_col="alphagenome_prediction",
            aggregation="topn",
            top_n=5,
            tools_on="rows",
            output_path=temp / f"performance_matrix_topn_{dataset}.pdf",
            title=f"{dataset}: top-5 predicted activity",
            value_label="top score",
        )
        _add_figure(pdf, fig)


def _rank_frame(task_seqs_st: dict, aggregation: str, top_n: int = 5) -> pd.DataFrame:
    frames = []
    for task, st in task_seqs_st.items():
        ranking = st.rank_tools(
            value_col="alphagenome_prediction",
            tool_col="Tools",
            aggregation=aggregation,
            top_n=top_n,
        )
        frames.append(ranking.assign(dataset=task))
    return pd.concat(frames, ignore_index=True)


def _section_rank(pdf, temp: Path, task_seqs_st: dict) -> None:
    _add_section(pdf, SECTION_RANK)
    plot = next(iter(task_seqs_st.values())).plot
    fig, _ = plot.rank_correlation(
        rank_df=_rank_frame(task_seqs_st, "mean"),
        dataset_col="dataset",
        tool_col="Tools",
        rank_col="rank",
        output_path=temp / "rank_correlation_mean.pdf",
        title="Tool-rank correlation (mean)",
    )
    _add_figure(pdf, fig)
    fig, _ = plot.rank_correlation(
        rank_df=_rank_frame(task_seqs_st, "topn", top_n=5),
        dataset_col="dataset",
        tool_col="Tools",
        rank_col="rank",
        output_path=temp / "rank_correlation_topn.pdf",
        title="Tool-rank correlation (top 5)",
    )
    _add_figure(pdf, fig)


def _section_resource(pdf, temp: Path, task_seqs_st: dict, resource_df: pd.DataFrame) -> None:
    _add_section(pdf, SECTION_RESOURCE)
    fig, _ = next(iter(task_seqs_st.values())).plot.resource_usage(
        resource_df=resource_df,
        tool_col="Tools",
        time_col="Time_hours",
        memory_col="Memory_GB",
        output_path=temp / "resource_usage.pdf",
        time_unit="h",
        memory_unit="GB",
    )
    _add_figure(pdf, fig)


def _section_seqlets(pdf, task_seqs_st: dict) -> None:
    import matplotlib.pyplot as plt

    from seqtrust.plot.helpers import display_label
    from seqtrust.plot.style import categorical_colors, nature_style

    _add_section(pdf, SECTION_SEQLETS)
    tasks = list(task_seqs_st.items())
    tool_order = list(dict.fromkeys(
        display_label(tool)
        for _, st in tasks
        for tool in st.df["Tools"]
    ))
    colors = dict(zip(tool_order, categorical_colors(len(tool_order))))
    for metric, metric_label in SEQLET_METRIC_LABELS.items():
        metric_title = metric_label.replace("\n", " ")
        with nature_style():
            fig, axes = plt.subplots(
                1,
                len(tasks),
                figsize=(max(2.2, 1.7 * len(tasks)), max(2.6, 0.32 * len(tool_order) + 1.2)),
                layout="constrained",
                squeeze=False,
                sharex=True,
                sharey=True,
            )
            for axis, (task, st) in zip(axes.flat, tasks):
                data = st.df[["Tools", metric]].dropna().copy()
                data[metric] = pd.to_numeric(data[metric], errors="coerce")
                data = data[np.isfinite(data[metric])]
                data["Tools"] = data["Tools"].map(display_label)
                panel_tools = [tool for tool in tool_order if (data["Tools"] == tool).any()]
                boxplot = axis.boxplot(
                    [
                        data.loc[data["Tools"] == tool, metric].to_numpy(dtype=float)
                        for tool in panel_tools
                    ],
                    positions=[tool_order.index(tool) + 1 for tool in panel_tools],
                    vert=False,
                    patch_artist=True,
                    widths=0.5,
                    whis=1.5,
                    showfliers=True,
                    medianprops={"color": "#202020", "linewidth": 0.8},
                    whiskerprops={"color": "#555555", "linewidth": 0.6},
                    capprops={"color": "#555555", "linewidth": 0.6},
                    flierprops={
                        "marker": "o",
                        "markersize": 1.8,
                        "markerfacecolor": "none",
                        "markeredgecolor": "#555555",
                        "markeredgewidth": 0.4,
                    },
                )
                for patch, tool in zip(boxplot["boxes"], panel_tools):
                    patch.set(
                        facecolor=colors[tool],
                        edgecolor="#303030",
                        linewidth=0.5,
                        alpha=0.82,
                    )
                axis.set_title(task, fontsize=6)
                axis.grid(axis="x", color="#E0E0E0", linewidth=0.4)
            axes.flat[0].set_yticks(np.arange(1, len(tool_order) + 1))
            axes.flat[0].set_yticklabels(tool_order)
            axes.flat[0].invert_yaxis()
            axes.flat[0].set_ylabel("Tools")
            fig.supxlabel(metric_label)
            fig.suptitle(f"Distribution of {metric_title} by tool", fontweight="bold")
        _add_figure(pdf, fig)


def _section_motif(pdf, temp: Path, task_seqs_st: dict) -> None:
    _add_section(pdf, SECTION_MOTIF)
    for task, st in task_seqs_st.items():
        fig, _ = st.plot.metric_bars(
            value_col="motif_recovery_ratio",
            tool_col="Tools",
            aggregation="mean",
            category_col=None,
            output_path=temp / f"motif_recovery_ratio_{task}.pdf",
            title=f"Motif recovery: {task}",
        )
        _add_figure(pdf, fig)


def _section_composition(pdf, temp: Path, task_seqs_st: dict) -> None:
    _add_section(pdf, SECTION_COMPOSITION)
    for task, st in task_seqs_st.items():
        fig, _ = st.plot.metric_bubbles(
            value_cols=COMPOSITION_COLUMNS,
            tool_col="Tools",
            aggregation="mean",
            output_path=temp / f"sequence_composition_{task}.pdf",
            figsize=(4.7, 4.7),
            title=f"Sequence composition: {task}",
        )
        _add_figure(pdf, fig)


def _section_logo(pdf, temp: Path, task_seqs_st: dict, random_state: int) -> None:
    import matplotlib.pyplot as plt

    _add_section(pdf, SECTION_LOGO)
    tasks = list(task_seqs_st.items())
    rng = np.random.default_rng(random_state)
    fig, axes = plt.subplots(
        len(tasks),
        1,
        figsize=(12, 2.3 * len(tasks)),
        layout="constrained",
        squeeze=False,
    )
    for axis, (task, st) in zip(axes.flat, tasks):
        pool = st.df.loc[st.df["Tools"].eq("AdaLead")]
        if pool.empty:
            raise ValueError(f"{task} has no AdaLead sequences.")
        counts = pool[st.sequence_col].map(pool[st.sequence_col].value_counts())
        unique_pool = pool.loc[counts.eq(1)]
        if unique_pool.empty:
            raise ValueError(f"{task} has no unique AdaLead sequence.")
        sequence = unique_pool.loc[rng.choice(unique_pool.index.to_numpy()), st.sequence_col]
        logo_fig, _ = st.plot.sequence_logo(
            filters={"Tools": "AdaLead", "Sequence": sequence},
            attribution_key="attribution_alphagenome",
            output_path=temp / f"sequence_logo_{task}.pdf",
            pos_threshold=0.02,
            neg_threshold=-1,
            break_regions=[(60, 80), (5, 45)],
            highlight_regions=[(20, 35)],
            title=f"Attribution logo: {task}",
        )
        axis.imshow(_figure_image(logo_fig))
        axis.set_axis_off()
        plt.close(logo_fig)
    _add_figure(pdf, fig)


def _section_umap(pdf, temp: Path, task_seqs_st: dict, random_state: int) -> None:
    import matplotlib.pyplot as plt

    _add_section(pdf, SECTION_UMAP)
    tasks = list(task_seqs_st.items())
    n_cols = 3
    n_rows = math.ceil(len(tasks) / n_cols)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(5.6 * n_cols, 4.6 * n_rows),
        layout="constrained",
        squeeze=False,
    )
    for axis, (task, st) in zip(axes.flat, tasks):
        umap_fig, _ = st.plot.embedding_umap(
            embedding_col="dnabert2_embedding",
            color_col="Tools",
            n_neighbors=10,
            min_dist=0.10,
            metric="euclidean",
            random_state=random_state,
            title=f"UMAP: {task}",
            output_path=temp / f"embedding_umap_{task}.pdf",
        )
        axis.imshow(_figure_image(umap_fig))
        axis.set_axis_off()
        plt.close(umap_fig)
    for axis in axes.flat[len(tasks):]:
        axis.set_axis_off()
    _add_figure(pdf, fig)
