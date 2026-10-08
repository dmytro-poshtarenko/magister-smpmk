"""Спільне оформлення рисунків для звіту (світла тема, шрифт Arial)."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Категоріальна палітра у фіксованому порядку (перевірена на розрізнення при дальтонізмі).
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

CM = 1 / 2.54


def setup() -> None:
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 10,
            "axes.labelsize": 10.5,
            "axes.labelcolor": INK_2,
            "axes.edgecolor": AXIS,
            "axes.linewidth": 0.8,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": GRID,
            "grid.linewidth": 0.6,
            "xtick.color": INK_2,
            "ytick.color": INK_2,
            "xtick.major.size": 0,
            "ytick.major.size": 0,
            "legend.frameon": False,
            "legend.fontsize": 9.5,
            "figure.dpi": 100,
            "savefig.dpi": 200,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.08,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.formatter.use_locale": False,
            "mathtext.fontset": "custom",
            "mathtext.rm": "Arial",
            "mathtext.it": "Arial:italic",
            "mathtext.bf": "Arial:bold",
            "mathtext.sf": "Arial",
            "mathtext.cal": "Arial",
        }
    )


def comma(x: float, digits: int = 2) -> str:
    """Число з десятковою комою, як прийнято в українських текстах."""
    return f"{x:.{digits}f}".replace(".", ",")


def comma_formatter(digits: int = 0):
    from matplotlib.ticker import FuncFormatter

    return FuncFormatter(lambda v, _: comma(v, digits))
