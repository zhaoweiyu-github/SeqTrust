from __future__ import annotations

from pathlib import Path


NATURE_STYLE = Path(__file__).with_name("nature.mplstyle")

# Paul Tol muted palette: restrained categorical colors for print figures.
TOL_MUTED = [
    "#CC6677",
    "#332288",
    "#DDCC77",
    "#117733",
    "#88CCEE",
    "#882255",
    "#44AA99",
    "#999933",
    "#AA4499",
]

OKABE_ITO_ON_WHITE = [
    "#0072B2",
    "#D55E00",
    "#009E73",
    "#CC79A7",
    "#000000",
]


def nature_style():
    """Return a scoped Matplotlib context using the bundled Nature-style preset."""
    import matplotlib.pyplot as plt

    return plt.style.context(NATURE_STYLE)


def categorical_colors(n_categories: int) -> list[str]:
    """Return deterministic qualitative colors without interpolating a palette."""
    if n_categories <= len(TOL_MUTED):
        return TOL_MUTED[:n_categories]

    import matplotlib.pyplot as plt

    colors: list[str] = []
    for cmap_name in ("tab20", "tab20b", "tab20c"):
        cmap = plt.get_cmap(cmap_name)
        colors.extend(cmap(index) for index in range(cmap.N))
    if n_categories > len(colors):
        raise ValueError(
            f"At most {len(colors)} categorical colors are supported by the "
            "default palette; provide fewer categories or split the figure."
        )
    return colors[:n_categories]
