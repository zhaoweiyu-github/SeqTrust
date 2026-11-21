from __future__ import annotations

import os
import uuid
from pathlib import Path

import numpy as np
import pandas as pd


def require_columns(df, columns) -> None:
    """Raise a clear error when required plotting columns are absent."""
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(
            f"Missing columns: {missing}. Available columns: {list(df.columns)}"
        )


def display_label(value) -> str:
    """Create an explicit label for missing categorical values."""
    missing = pd.isna(value)
    if np.ndim(missing) == 0 and bool(missing):
        return "Missing"
    return str(value)


def finite_limits(values) -> tuple[float, float]:
    """Return non-degenerate limits over finite values."""
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if array.size == 0:
        raise ValueError("No finite values are available for plotting.")
    lower = float(array.min())
    upper = float(array.max())
    if lower == upper:
        pad = abs(lower) * 0.05 or 0.5
        return lower - pad, upper + pad
    return lower, upper


def bubble_sizes(values, minimum: float = 30, maximum: float = 280) -> np.ndarray:
    """Map values linearly to marker area, not radius."""
    array = np.asarray(values, dtype=float)
    lower, upper = finite_limits(array)
    normalized = (array - lower) / (upper - lower)
    return minimum + normalized * (maximum - minimum)


def save_pdf(
    fig,
    output_path: str | Path,
    *,
    title: str,
    overwrite: bool = True,
) -> Path:
    """Atomically save one opaque, font-editable PDF at an explicit path."""
    path = Path(output_path)
    if path.suffix.lower() != ".pdf":
        raise ValueError("output_path must end with '.pdf'.")
    if path.exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.stem}.{uuid.uuid4().hex}.tmp.pdf")
    try:
        fig.savefig(
            temporary,
            format="pdf",
            dpi=450,
            bbox_inches=None,
            facecolor="white",
            edgecolor="white",
            transparent=False,
            metadata={
                "Title": title,
                "Creator": "SeqTrust",
                "Subject": "Publication-style scientific figure",
            },
        )
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return path


def format_number(value: float) -> str:
    """Format numeric annotations compactly while preserving meaningful scale."""
    magnitude = abs(float(value))
    if magnitude >= 1000 or (0 < magnitude < 0.01):
        return f"{value:.2g}"
    if magnitude >= 100:
        return f"{value:.0f}"
    if magnitude >= 10:
        return f"{value:.1f}"
    return f"{value:.2f}".rstrip("0").rstrip(".")
