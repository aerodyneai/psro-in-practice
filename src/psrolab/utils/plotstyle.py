"""Shared matplotlib styling for book figures.

Called at the top of every ``_fig*`` helper in the experiment scripts so that
figures written to ``docs/figures/`` share fonts, sizes, and DPI regardless of
which chapter produced them. Individual figures still choose their own axes
layouts and palettes; this module only sets the shared rcParams.
"""

from __future__ import annotations

import matplotlib.pyplot as plt

_APPLIED = False


def apply_style() -> None:
    global _APPLIED
    if _APPLIED:
        return
    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.labelsize": 11,
            "axes.titlesize": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "legend.frameon": False,
            "figure.dpi": 100,
            "savefig.dpi": 200,
            "savefig.bbox": "tight",
        }
    )
    _APPLIED = True
