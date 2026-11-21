from __future__ import annotations

import textwrap
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from seqtrust.plot.helpers import save_pdf
from seqtrust.plot.style import nature_style


@dataclass
class _TreeNode:
    """Internal representation of one hierarchy node."""

    label: str
    edge_label: str | None = None
    children: list["_TreeNode"] = field(default_factory=list)
    node_id: str = ""
    parent_id: str | None = None
    depth: int = 0
    x: float = 0.0
    y: float = 0.0


def _label(value: object) -> str:
    """Convert an external node label to a validated string."""
    label = str(value).strip()
    if not label:
        raise ValueError("Tree node labels must not be empty.")
    return label


def _children_from_nested(value: object) -> list[_TreeNode]:
    """Convert nested mapping/list values into child nodes."""
    if value is None:
        return []
    if isinstance(value, Mapping):
        if "label" in value:
            return [_node_from_spec(value)]
        return [_node_from_nested(label, children) for label, children in value.items()]
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        children: list[_TreeNode] = []
        for item in value:
            if isinstance(item, Mapping):
                if "label" in item:
                    children.append(_node_from_spec(item))
                elif len(item) == 1:
                    label, nested = next(iter(item.items()))
                    children.append(_node_from_nested(label, nested))
                else:
                    raise ValueError(
                        "Nested child mappings must contain 'label' or one label key."
                    )
            elif isinstance(item, Sequence) and not isinstance(item, (str, bytes)):
                raise ValueError(
                    "Nested node children cannot be paths; pass root-to-leaf paths "
                    "as the top-level hierarchy instead."
                )
            else:
                children.append(_TreeNode(label=_label(item)))
        return children
    return [_TreeNode(label=_label(value))]


def _node_from_nested(label: object, children: object) -> _TreeNode:
    """Build a node whose descendants are expressed as nested values."""
    return _TreeNode(
        label=_label(label),
        children=_children_from_nested(children),
    )


def _node_from_spec(spec: Mapping[str, object]) -> _TreeNode:
    """Build a node from a label/children/edge_label configuration object."""
    if "label" not in spec:
        raise ValueError("Every node configuration must contain a 'label' field.")
    allowed = {"label", "children", "edge_label"}
    unknown = sorted(set(spec) - allowed)
    if unknown:
        raise ValueError(f"Unknown node configuration fields: {unknown}")
    edge_label = spec.get("edge_label")
    return _TreeNode(
        label=_label(spec["label"]),
        edge_label=None if edge_label is None else _label(edge_label),
        children=_children_from_nested(spec.get("children", [])),
    )


def _tree_from_paths(
    hierarchy: Sequence[object],
    root_label: str | None,
) -> _TreeNode:
    """Merge root-to-leaf paths into one ordered tree."""
    if not hierarchy:
        raise ValueError("hierarchy must contain at least one path or node.")
    if all(
        not isinstance(item, Sequence) or isinstance(item, (str, bytes))
        for item in hierarchy
    ):
        paths = [list(hierarchy)]
    else:
        paths = []
        for path in hierarchy:
            if not isinstance(path, Sequence) or isinstance(path, (str, bytes)):
                raise TypeError("Every hierarchy path must be a sequence of labels.")
            paths.append(list(path))
    if any(not path for path in paths):
        raise ValueError("Hierarchy paths must not be empty.")
    if any(
        isinstance(item, (Mapping, Sequence)) and not isinstance(item, (str, bytes))
        for path in paths
        for item in path
    ):
        raise TypeError("Path entries must be scalar node labels.")

    normalized = [[_label(item) for item in path] for path in paths]
    if root_label is None:
        root_name = normalized[0][0]
        if any(path[0] != root_name for path in normalized):
            raise ValueError(
                "All paths must share their first root label, or root_label must "
                "be supplied to create a common root."
            )
        branches = [path[1:] for path in normalized]
    else:
        root_name = _label(root_label)
        branches = [path[1:] if path[0] == root_name else path for path in normalized]

    root = _TreeNode(label=root_name)
    for path in branches:
        parent = root
        for node_label in path:
            child = next(
                (candidate for candidate in parent.children if candidate.label == node_label),
                None,
            )
            if child is None:
                child = _TreeNode(label=node_label)
                parent.children.append(child)
            parent = child
    return root


def _parse_hierarchy(hierarchy: object, root_label: str | None) -> _TreeNode:
    """Accept path lists, explicit node specs, or compact nested mappings."""
    if isinstance(hierarchy, Mapping):
        if "label" in hierarchy:
            if root_label is not None:
                raise ValueError(
                    "root_label is unnecessary when hierarchy already has a root node."
                )
            return _node_from_spec(hierarchy)
        if len(hierarchy) == 1 and root_label is None:
            label, children = next(iter(hierarchy.items()))
            return _node_from_nested(label, children)
        if root_label is None:
            raise ValueError(
                "A nested mapping with multiple top-level keys requires root_label."
            )
        return _TreeNode(
            label=_label(root_label),
            children=[
                _node_from_nested(label, children)
                for label, children in hierarchy.items()
            ],
        )
    if isinstance(hierarchy, Sequence) and not isinstance(hierarchy, (str, bytes)):
        return _tree_from_paths(hierarchy, root_label)
    raise TypeError(
        "hierarchy must be root-to-leaf paths, a node specification, or a "
        "nested mapping."
    )


def _assign_coordinates(
    root: _TreeNode,
    level_step: float,
    leaf_gap: float,
) -> tuple[list[_TreeNode], list[tuple[_TreeNode, _TreeNode]], int, int]:
    """Assign stable left-to-right coordinates using leaf-centered parents."""
    nodes: list[_TreeNode] = []
    edges: list[tuple[_TreeNode, _TreeNode]] = []
    next_leaf = 0
    max_depth = 0

    def visit(
        node: _TreeNode,
        depth: int,
        node_id: str,
        parent_id: str | None,
    ) -> float:
        nonlocal next_leaf, max_depth
        node.node_id = node_id
        node.parent_id = parent_id
        node.depth = depth
        node.x = depth * level_step
        max_depth = max(max_depth, depth)
        nodes.append(node)
        if not node.children:
            node.y = next_leaf * leaf_gap
            next_leaf += 1
            return node.y

        child_positions = []
        for index, child in enumerate(node.children):
            edges.append((node, child))
            child_positions.append(
                visit(child, depth + 1, f"{node_id}.{index}", node_id)
            )
        node.y = float(np.mean(child_positions))
        return node.y

    visit(root, 0, "0", None)
    return nodes, edges, max_depth, max(next_leaf, 1)


class HorizontalTreeMixin:
    """Horizontal trees generated from externally supplied hierarchy data."""

    def horizontal_tree(
        self,
        hierarchy: object,
        output_path: str | Path,
        root_label: str | None = None,
        title: str | None = None,
        node_width: float = 2.15,
        node_height: float = 0.66,
        level_gap: float = 1.25,
        leaf_gap: float = 1.05,
        max_label_chars: int = 22,
        font_size: float = 7.0,
        edge_font_size: float = 6.0,
        cmap: str = "cividis",
        show_arrows: bool = True,
        overwrite: bool = True,
    ):
        """Draw a left-to-right hierarchy and save it as a PDF.

        ``hierarchy`` accepts root-to-leaf paths, for example
        ``[["Root", "A", "A1"], ["Root", "B", "B1"]]``; a compact
        nested mapping such as ``{"Root": {"A": ["A1", "A2"]}}``; or an
        explicit node configuration with ``label``, ``children``, and optional
        ``edge_label`` fields.
        """
        import matplotlib as mpl
        import matplotlib.pyplot as plt
        from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

        numeric_options = {
            "node_width": node_width,
            "node_height": node_height,
            "level_gap": level_gap,
            "leaf_gap": leaf_gap,
            "font_size": font_size,
            "edge_font_size": edge_font_size,
        }
        for name, value in numeric_options.items():
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be a positive finite number.")
        if not isinstance(max_label_chars, int) or max_label_chars < 4:
            raise ValueError("max_label_chars must be an integer of at least 4.")

        root = _parse_hierarchy(hierarchy, root_label)
        level_step = node_width + level_gap
        nodes, edges, max_depth, leaf_count = _assign_coordinates(
            root,
            level_step=level_step,
            leaf_gap=leaf_gap,
        )
        depth_colors = mpl.colormaps[cmap](
            np.linspace(0.18, 0.82, max_depth + 1)
        )
        figure_width = max(5.2, 1.6 + (max_depth + 1) * 2.35)
        figure_height = max(2.8, 1.25 + leaf_count * 0.78)
        plot_title = title or "Horizontal decision tree"

        with nature_style():
            fig, ax = plt.subplots(
                figsize=(figure_width, figure_height),
                layout="constrained",
            )

            # Draw connectors before nodes so boxes cover connector endpoints.
            for parent, child in edges:
                start_x = parent.x + node_width / 2
                end_x = child.x - node_width / 2
                elbow_x = (start_x + end_x) / 2
                ax.plot(
                    [start_x, elbow_x, elbow_x],
                    [parent.y, parent.y, child.y],
                    color="#606060",
                    linewidth=0.75,
                    solid_capstyle="round",
                    zorder=1,
                )
                if show_arrows:
                    arrow = FancyArrowPatch(
                        (elbow_x, child.y),
                        (end_x, child.y),
                        arrowstyle="-|>",
                        mutation_scale=8,
                        linewidth=0.75,
                        color="#606060",
                        shrinkA=0,
                        shrinkB=0,
                        zorder=1,
                    )
                    ax.add_patch(arrow)
                else:
                    ax.plot(
                        [elbow_x, end_x],
                        [child.y, child.y],
                        color="#606060",
                        linewidth=0.75,
                        zorder=1,
                    )
                if child.edge_label:
                    ax.text(
                        (elbow_x + end_x) / 2,
                        child.y - node_height * 0.62,
                        child.edge_label,
                        ha="center",
                        va="bottom",
                        fontsize=edge_font_size,
                        color="#404040",
                        zorder=2,
                    )

            for node in nodes:
                facecolor = depth_colors[node.depth]
                luminance = (
                    0.299 * facecolor[0]
                    + 0.587 * facecolor[1]
                    + 0.114 * facecolor[2]
                )
                box = FancyBboxPatch(
                    (node.x - node_width / 2, node.y - node_height / 2),
                    node_width,
                    node_height,
                    boxstyle="round,pad=0.025,rounding_size=0.08",
                    facecolor=facecolor,
                    edgecolor="#202020",
                    linewidth=0.65,
                    zorder=3,
                )
                ax.add_patch(box)
                ax.text(
                    node.x,
                    node.y,
                    textwrap.fill(node.label, width=max_label_chars),
                    ha="center",
                    va="center",
                    fontsize=font_size,
                    color="black" if luminance > 0.58 else "white",
                    zorder=4,
                )

            x_margin = max(0.55, level_gap * 0.35)
            y_margin = max(0.65, node_height)
            ax.set_xlim(
                -node_width / 2 - x_margin,
                max_depth * level_step + node_width / 2 + x_margin,
            )
            ax.set_ylim((leaf_count - 1) * leaf_gap + y_margin, -y_margin)
            ax.set_title(plot_title, fontweight="bold")
            ax.set_axis_off()
            output = save_pdf(fig, output_path, title=plot_title, overwrite=overwrite)

        node_table = pd.DataFrame(
            [
                {
                    "node_id": node.node_id,
                    "parent_id": node.parent_id,
                    "label": node.label,
                    "depth": node.depth,
                    "x": node.x,
                    "y": node.y,
                }
                for node in nodes
            ]
        )
        edge_table = pd.DataFrame(
            [
                {
                    "parent_id": parent.node_id,
                    "child_id": child.node_id,
                    "edge_label": child.edge_label,
                }
                for parent, child in edges
            ],
            columns=["parent_id", "child_id", "edge_label"],
        )
        return fig, {
            "nodes": node_table,
            "edges": edge_table,
            "output_path": output,
        }
