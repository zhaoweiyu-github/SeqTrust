from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from seqtrust.plot.helpers import require_columns, save_pdf
from seqtrust.plot.style import nature_style


BASE_COLORS = {
    "A": "#109648",
    "C": "#255C99",
    "G": "#F28E2B",
    "T": "#E15759",
}


def _align_sequence(sequence: object, length: int) -> str:
    """Center-crop or N-pad a sequence to the attribution length."""
    cleaned = "".join(str(sequence).split()).upper()
    if len(cleaned) > length:
        start = (len(cleaned) - length) // 2
        return cleaned[start : start + length]
    if len(cleaned) < length:
        left = (length - len(cleaned)) // 2
        return "N" * left + cleaned + "N" * (length - len(cleaned) - left)
    return cleaned


def _normalize_regions(
    regions: Sequence[tuple[int, int]] | tuple[int, int] | None,
    length: int,
    *,
    strict_inside: bool = True,
) -> list[tuple[int, int]]:
    """Validate and merge inclusive sequence regions."""
    if regions is None:
        return []
    if (
        len(regions) == 2
        and np.isscalar(regions[0])
        and np.isscalar(regions[1])
    ):
        regions = [regions]

    validated: list[tuple[int, int]] = []
    for region in regions:
        if len(region) != 2:
            raise ValueError("Each region must be an inclusive (start, end) pair.")
        start, end = region
        if not isinstance(start, (int, np.integer)) or not isinstance(
            end, (int, np.integer)
        ):
            raise TypeError("Region boundaries must be integers.")
        start, end = int(start), int(end)
        if start > end:
            raise ValueError(f"Region start exceeds end: {(start, end)}")
        if strict_inside:
            if start <= 0 or end >= length - 1:
                raise ValueError(
                    f"Break region {(start, end)} must lie inside "
                    f"(0, {length - 1})."
                )
        elif start < 0 or end >= length:
            raise ValueError(
                f"Region {(start, end)} lies outside [0, {length - 1}]."
            )
        validated.append((start, end))

    merged: list[list[int]] = []
    for start, end in sorted(validated):
        if not merged or start > merged[-1][1] + 1:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    return [(start, end) for start, end in merged]


def _compress_positions(
    logo_contrib: np.ndarray,
    break_regions: Sequence[tuple[int, int]],
    break_gap: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[tuple[float, int, int]]]:
    """Remove break regions and return bidirectional position mappings."""
    length = len(logo_contrib)
    original_to_display = np.full(length, -1, dtype=int)
    display_to_original: list[int] = []
    display_parts: list[np.ndarray] = []
    break_info: list[tuple[float, int, int]] = []
    source_cursor = 0
    display_cursor = 0

    for break_start, break_end in break_regions:
        kept = np.arange(source_cursor, break_start, dtype=int)
        if kept.size:
            displayed = np.arange(display_cursor, display_cursor + kept.size)
            original_to_display[kept] = displayed
            display_to_original.extend(kept.tolist())
            display_parts.append(logo_contrib[kept])
            display_cursor += kept.size

        gap_start = display_cursor
        display_parts.append(np.zeros((break_gap, 4), dtype=float))
        display_to_original.extend([-1] * break_gap)
        break_info.append(
            (gap_start + (break_gap - 1) / 2, break_start, break_end)
        )
        display_cursor += break_gap
        source_cursor = break_end + 1

    kept = np.arange(source_cursor, length, dtype=int)
    if kept.size:
        displayed = np.arange(display_cursor, display_cursor + kept.size)
        original_to_display[kept] = displayed
        display_to_original.extend(kept.tolist())
        display_parts.append(logo_contrib[kept])

    return (
        np.vstack(display_parts),
        original_to_display,
        np.asarray(display_to_original, dtype=int),
        break_info,
    )


class SequenceLogoMixin:
    """Attribution-weighted sequence logos selected by DataFrame filters."""

    def sequence_logo(
        self,
        filters: Mapping[str, object],
        output_path: str | Path,
        contribution_col: str | None = None,
        attribution_key: str = "seq2function_attribution",
        sequence_col: str | None = None,
        pos_threshold: float = 0.10,
        neg_threshold: float = -0.10,
        break_regions: Sequence[tuple[int, int]] | tuple[int, int] | None = None,
        break_gap: int = 2,
        highlight_regions: Sequence[tuple[int, int]] | tuple[int, int] | None = None,
        show_break_labels: bool = True,
        show_position_axis: bool = True,
        title: str | None = None,
        figsize: tuple[float, float] | None = None,
        ylim: tuple[float, float] | None = None,
        overwrite: bool = True,
    ):
        """Select exactly one sequence with equality filters and plot its logo.

        ``filters`` maps DataFrame column names to values and all conditions are
        combined with equality/AND. Contributions may be stored directly in a
        DataFrame column or in ``trust.results[attribution_key]`` aligned with
        DataFrame row positions. The contribution matrix must have shape
        ``(sequence_length, 4)`` in A/C/G/T order.
        """
        import matplotlib as mpl
        import matplotlib.pyplot as plt

        try:
            import logomaker
        except ImportError as exc:
            raise ImportError(
                "sequence_logo requires logomaker; install SeqTrust with the "
                "'plot' optional dependencies."
            ) from exc

        if not isinstance(filters, Mapping) or not filters:
            raise ValueError("filters must be a non-empty mapping of column to value.")
        require_columns(self.trust.df, list(filters))
        mask = pd.Series(True, index=self.trust.df.index, dtype=bool)
        for column, expected in filters.items():
            comparison = self.trust.df[column].eq(expected)
            mask &= comparison.fillna(False).astype(bool)
        positions = np.flatnonzero(mask.to_numpy())
        if len(positions) == 0:
            raise ValueError(f"No sequence matches all equality filters: {dict(filters)}")
        if len(positions) > 1:
            raise ValueError(
                f"Filters match {len(positions)} rows; add conditions until exactly "
                "one sequence is selected."
            )
        row_position = int(positions[0])
        selected = self.trust.df.iloc[row_position]

        sequence_col = sequence_col or self.trust.sequence_col
        require_columns(self.trust.df, [sequence_col])
        if contribution_col is not None:
            require_columns(self.trust.df, [contribution_col])
            contribution = selected[contribution_col]
            contribution_source = f"df.{contribution_col}"
        else:
            if attribution_key not in self.trust.results:
                raise KeyError(
                    f"No result stored under {attribution_key!r}; pass "
                    "contribution_col or calculate/store attributions first."
                )
            stored = self.trust.results[attribution_key]
            attribution = stored.get("all") if isinstance(stored, dict) else stored
            attribution = np.asarray(attribution)
            if attribution.ndim != 3 or attribution.shape[-1] != 4:
                raise ValueError(
                    "Stored attribution must have shape (N, L, 4); "
                    f"received {attribution.shape}."
                )
            if attribution.shape[0] != len(self.trust.df):
                raise ValueError(
                    "Stored attribution row count does not match trust.df: "
                    f"{attribution.shape[0]} versus {len(self.trust.df)}."
                )
            contribution = attribution[row_position]
            contribution_source = f"results[{attribution_key!r}]"

        contribution = np.asarray(contribution, dtype=float)
        if contribution.ndim != 2 or contribution.shape[1] != 4:
            raise ValueError(
                "Selected contribution must have shape (L, 4); "
                f"received {contribution.shape}."
            )
        if contribution.shape[0] == 0:
            raise ValueError("Selected contribution matrix has zero sequence length.")
        if not np.isfinite(contribution).all():
            raise ValueError("Selected contribution contains NaN or infinite values.")
        if pos_threshold < 0 or neg_threshold > 0:
            raise ValueError(
                "pos_threshold must be non-negative and neg_threshold non-positive."
            )
        if not isinstance(break_gap, (int, np.integer)) or break_gap < 1:
            raise ValueError("break_gap must be a positive integer.")
        if ylim is not None and (len(ylim) != 2 or ylim[0] >= ylim[1]):
            raise ValueError("ylim must be an increasing (minimum, maximum) pair.")

        length = contribution.shape[0]
        aligned_sequence = _align_sequence(selected[sequence_col], length)
        base_labels = np.asarray(list(aligned_sequence))
        point_colors = np.asarray(
            [BASE_COLORS.get(base, "#777777") for base in base_labels]
        )
        per_position = contribution.sum(axis=1)
        logo_mask = (per_position >= pos_threshold) | (
            per_position <= neg_threshold
        )
        logo_contribution = np.where(logo_mask[:, None], contribution, 0.0)
        normalized_breaks = _normalize_regions(break_regions, length)
        (
            display_logo,
            original_to_display,
            display_to_original,
            break_info,
        ) = _compress_positions(logo_contribution, normalized_breaks, break_gap)
        kept = original_to_display >= 0
        low_positions = np.flatnonzero((~logo_mask) & kept)
        low_display = original_to_display[low_positions]
        display_length = len(display_logo)

        highlight_regions = _normalize_regions(
            highlight_regions,
            length,
            strict_inside=False,
        )
        if figsize is None:
            figsize = (max(6.5, min(14.0, display_length * 0.075)), 2.8)
        plot_title = title or "Sequence attribution logo"

        with nature_style():
            fig, ax = plt.subplots(figsize=figsize, layout="constrained")
            for start, end in highlight_regions:
                visible = original_to_display[start : end + 1]
                visible = visible[visible >= 0]
                if not visible.size:
                    continue
                split_at = np.where(np.diff(visible) > 1)[0] + 1
                for run in np.split(visible, split_at):
                    ax.axvspan(
                        run[0] - 0.5,
                        run[-1] + 0.5,
                        color="#F3E7A0",
                        alpha=0.35,
                        linewidth=0,
                        zorder=0,
                    )

            ax.vlines(
                low_display,
                0,
                per_position[low_positions],
                colors=point_colors[low_positions],
                linewidths=0.65,
                alpha=0.18,
                zorder=1,
            )
            ax.scatter(
                low_display,
                per_position[low_positions],
                c=point_colors[low_positions].tolist(),
                s=7,
                alpha=0.30,
                edgecolors="none",
                zorder=2,
            )
            logo_frame = pd.DataFrame(display_logo, columns=["A", "C", "G", "T"])
            logomaker.Logo(
                logo_frame,
                ax=ax,
                center_values=False,
                color_scheme=BASE_COLORS,
                shade_below=0,
                fade_below=0,
            )
            ax.axhline(0, color="#202020", linewidth=0.65, zorder=4)
            ax.set_xlim(-0.5, display_length - 0.5)

            if ylim is None:
                visible_contribution = contribution[kept]
                positive_max = max(
                    float(np.clip(visible_contribution, 0, None).sum(axis=1).max()),
                    0.0,
                )
                negative_min = min(
                    float(np.clip(visible_contribution, None, 0).sum(axis=1).min()),
                    0.0,
                )
                span = max(positive_max - negative_min, 0.2)
                ax.set_ylim(
                    min(-0.1, negative_min - 0.08 * span),
                    max(0.1, positive_max + 0.08 * span),
                )
            else:
                ax.set_ylim(*ylim)

            y_min, y_max = ax.get_ylim()
            zero_axis = (0 - y_min) / (y_max - y_min) if y_min <= 0 <= y_max else 0.5
            break_transform = ax.get_xaxis_transform()
            for center, start, end in break_info:
                for offset in (-0.23, 0.23):
                    ax.plot(
                        [center + offset - 0.12, center + offset + 0.12],
                        [zero_axis - 0.05, zero_axis + 0.05],
                        transform=break_transform,
                        color="#202020",
                        linewidth=1.1,
                        solid_capstyle="round",
                        zorder=9,
                    )
                if show_break_labels:
                    ax.text(
                        center,
                        zero_axis + 0.075,
                        f"{start}-{end}",
                        transform=break_transform,
                        ha="center",
                        va="bottom",
                        fontsize=6,
                        color="#202020",
                        zorder=10,
                    )

            if show_position_axis:
                ax.set_xlabel("Position")
                ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(10, integer=True))

                def original_position(value, _):
                    display_position = int(round(value))
                    if not np.isclose(value, display_position, atol=0.15):
                        return ""
                    if not 0 <= display_position < display_length:
                        return ""
                    position = display_to_original[display_position]
                    return "" if position < 0 else str(position)

                ax.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(original_position))
            else:
                ax.set_xticks([])
                ax.spines["bottom"].set_visible(False)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.set(ylabel="Contribution", title=plot_title)
            output = save_pdf(fig, output_path, title=plot_title, overwrite=overwrite)

        return fig, {
            "selected_row": selected.copy(),
            "row_position": row_position,
            "filters": dict(filters),
            "contribution_source": contribution_source,
            "contribution": contribution,
            "logo_matrix": logo_frame,
            "aligned_sequence": aligned_sequence,
            "display_to_original": display_to_original,
            "output_path": output,
        }
