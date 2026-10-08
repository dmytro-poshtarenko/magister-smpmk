"""Досліди 1 і 2 лабораторної роботи № 1.

Дослід 1. Генерування масивів вхідних даних X1…X5 (2, 3, 6, 9 і 18 показників,
100 спостережень), формування модельованого показника y як зваженої суми
показників і синтез моделей двома алгоритмами (МНК і багаторядний МГУА).

Дослід 2. Сортування спостережень кожного масиву за зростанням модуля
відхилення |y − ȳ| і повторний синтез моделей. Додатково розглянуто сортування
за відхиленням без модуля, як записано у формулі методички.

Запуск: python experiments.py (результати — у data/ та outputs/).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotting as pl
from asm import first_layer_models, structural_variety, synthesize_gmdh, synthesize_ls
from mvd import MVD, distribution_law, sort_observations
from scipy import stats

ROOT = Path(__file__).parent
DATA = ROOT / "data"
OUT = ROOT / "outputs"

SEED = 2026
N_OBS = 100
N_TRAIN = 70  # навчальна вибірка A — перші 70 спостережень, перевірна B — решта 30
SIZES = {"X1": 2, "X2": 3, "X3": 6, "X4": 9, "X5": 18}
# Вагові коефіцієнти (характеристика впливовості факторів x1…x18).
WEIGHTS = np.array([3.0, 2.5, 2.0, 1.8, 1.5, 1.2, 1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.25, 0.2, 0.15, 0.1])
NORMAL = (50.0, 10.0)  # непарні показники: нормальний закон N(50; 10)
UNIFORM = (10.0, 90.0)  # парні показники: рівномірний закон U(10; 90)
SORTINGS = {"abs_dev": "за |Δy|", "dev": "за Δy"}


def generate_base(seed: int = SEED) -> tuple[np.ndarray, list[str]]:
    """Генерує 18 показників по 100 спостережень із додатними значеннями.

    Аналог інструмента «Генерація випадкових чисел» надбудови «Аналіз даних» Excel:
    непарні показники мають нормальний розподіл, парні — рівномірний.
    """
    rng = np.random.default_rng(seed)
    X = np.empty((N_OBS, len(WEIGHTS)))
    laws = []
    for j in range(len(WEIGHTS)):
        if j % 2 == 0:
            X[:, j] = rng.normal(*NORMAL, N_OBS)
            laws.append("N(50; 10)")
        else:
            X[:, j] = rng.uniform(*UNIFORM, N_OBS)
            laws.append("U(10; 90)")
    assert X.min() > 0, "усі значення показників мають бути додатними"
    return X, laws


def build_arrays(X: np.ndarray) -> dict[str, MVD]:
    """Масиви X1 ⊂ X2 ⊂ … ⊂ X5: кожен наступний доповнює попередній новими показниками."""
    arrays = {}
    for name, m in SIZES.items():
        Xm = X[:, :m]
        y = Xm @ WEIGHTS[:m]
        arrays[name] = MVD(Xm, y, [f"x{j + 1}" for j in range(m)])
    return arrays


def synthesize(mvd: MVD) -> tuple:
    ls = synthesize_ls(mvd.X, mvd.y, N_TRAIN)
    gmdh = synthesize_gmdh(mvd.X, mvd.y, N_TRAIN, names=mvd.names)
    return ls, gmdh


def result_row(name: str, mvd: MVD, ls, gmdh, experiment: str) -> dict:
    m = mvd.shape[1]
    w = WEIGHTS[:m]
    return {
        "experiment": experiment,
        "array": name,
        "m": m,
        "y_mean": mvd.y.mean(),
        "y_std": mvd.y.std(),
        "ls_sx_train": ls.sx_train,
        "ls_sx_check": ls.sx_check,
        "ls_max_dw": float(np.max(np.abs(ls.model.c - w))),
        "ls_a0": ls.model.c0,
        "gmdh_layers": gmdh.layers,
        "gmdh_models": gmdh.n_models,
        "gmdh_sx_train": gmdh.sx_train,
        "gmdh_sx_check": gmdh.sx_check,
        "gmdh_criterion": gmdh.criterion,
        "gmdh_factors": len(gmdh.inputs_used),
        "gmdh_max_dw": float(np.max(np.abs(gmdh.model.c - w))),
        "gmdh_a0": gmdh.model.c0,
        "first_layer_models": first_layer_models(m),
        "structural_variety": structural_variety(m),
    }


def main() -> dict:
    DATA.mkdir(exist_ok=True)
    OUT.mkdir(exist_ok=True)
    pl.setup()

    X, laws = generate_base()
    arrays = build_arrays(X)

    # Закон розподілу кожного показника (за X5, що містить усі 18 показників).
    dist_rows = []
    for j in range(X.shape[1]):
        d = distribution_law(X[:, j])
        dist_rows.append({"indicator": f"x{j + 1}", "generated": laws[j], "weight": WEIGHTS[j], **d})
    dist = pd.DataFrame(dist_rows)
    dist.to_csv(OUT / "exp1_distribution.csv", index=False)

    rows, weights, histories = [], {}, {}
    for name, mvd in arrays.items():
        mvd.to_csv(DATA / f"exp1_{name}.csv")
        ls, gmdh = synthesize(mvd)
        rows.append(result_row(name, mvd, ls, gmdh, "1"))
        weights[(name, "1")] = (ls.model, gmdh.model)
        histories[(name, "1")] = gmdh.history
        for key in SORTINGS:
            sorted_mvd = sort_observations(mvd, key)
            if key == "abs_dev":
                table = pd.DataFrame(sorted_mvd.X, columns=sorted_mvd.names)
                table.insert(0, "obs", sorted_mvd.index)
                table["y"] = sorted_mvd.y
                table["abs_dy"] = np.abs(sorted_mvd.y - mvd.y.mean())
                table.round(4).to_csv(DATA / f"exp2_{name}.csv", index=False)
            ls2, gmdh2 = synthesize(sorted_mvd)
            exp = "2" if key == "abs_dev" else "2*"
            rows.append(result_row(name, sorted_mvd, ls2, gmdh2, exp))
            weights[(name, exp)] = (ls2.model, gmdh2.model)
            histories[(name, exp)] = gmdh2.history

    results = pd.DataFrame(rows)
    results.to_csv(OUT / "exp_results.csv", index=False)

    hist_rows = [
        {"array": a, "experiment": e, **h} for (a, e), hist in histories.items() for h in hist
    ]
    pd.DataFrame(hist_rows).to_csv(OUT / "gmdh_history.csv", index=False)

    w_rows = []
    for (name, exp), (ls_model, g_model) in weights.items():
        for j in range(len(ls_model.c)):
            w_rows.append(
                {
                    "array": name,
                    "experiment": exp,
                    "indicator": f"x{j + 1}",
                    "true": WEIGHTS[j],
                    "ls": ls_model.c[j],
                    "gmdh": g_model.c[j],
                }
            )
        w_rows.append({"array": name, "experiment": exp, "indicator": "a0", "true": 0.0, "ls": ls_model.c0, "gmdh": g_model.c0})
    wdf = pd.DataFrame(w_rows)
    wdf.to_csv(OUT / "exp_weights.csv", index=False)

    # Чутливість МГУА до свободи вибору F для X5.
    x5 = arrays["X5"]
    f_rows = []
    for F in (2, 4, 9, 18, 36):
        g = synthesize_gmdh(x5.X, x5.y, N_TRAIN, freedom=F)
        f_rows.append({"F": F, "layers": g.layers, "models": g.n_models, "sx_check": g.sx_check, "criterion": g.criterion, "factors": len(g.inputs_used)})
    pd.DataFrame(f_rows).to_csv(OUT / "gmdh_freedom_X5.csv", index=False)

    plot_distributions(X)
    plot_history(pd.DataFrame(hist_rows))
    plot_weights(wdf, "X5")
    plot_experiments(results)

    summary = {
        "seed": SEED,
        "n_obs": N_OBS,
        "n_train": N_TRAIN,
        "min_value": float(X.min()),
        "max_value": float(X.max()),
    }
    (OUT / "exp_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


# --- Рисунки ------------------------------------------------------------------------------------


def plot_distributions(X: np.ndarray) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(15 * pl.CM, 6.2 * pl.CM), sharey=False)
    specs = [
        (0, "x1", "нормальний закон N(50; 10)", stats.norm(*NORMAL)),
        (1, "x2", "рівномірний закон U(10; 90)", stats.uniform(UNIFORM[0], UNIFORM[1] - UNIFORM[0])),
    ]
    for ax, (j, name, label, dist) in zip(axes, specs):
        x = X[:, j]
        bins = np.linspace(x.min(), x.max(), 11)
        ax.hist(x, bins=bins, density=True, color=pl.SERIES[0], alpha=0.85, edgecolor="white", linewidth=1.5, label="емпірична частота")
        grid = np.linspace(x.min() - 5, x.max() + 5, 300)
        ax.plot(grid, dist.pdf(grid), color=pl.SERIES[1], linewidth=2, label="теоретична щільність")
        ax.set_xlabel(f"значення показника {name}")
        ax.set_title(f"{name}: {label}", fontsize=10, color=pl.INK, loc="left")
        ax.yaxis.set_major_formatter(pl.comma_formatter(3))
    axes[0].set_ylabel("щільність")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(OUT / "fig1_distributions.png")
    plt.close(fig)


def plot_history(hist: pd.DataFrame) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(15 * pl.CM, 7.5 * pl.CM))
    h1 = hist[hist.experiment == "1"]
    for k, name in enumerate(["X2", "X3", "X4", "X5"]):
        h = h1[h1.array == name]
        m = SIZES[name]
        ax.plot(h.layer, h.criterion, marker="o", markersize=4.5, linewidth=1.8, color=pl.SERIES[k], label=f"{name} (m = {m})")
        stop = h[h["stop"].fillna(False).astype(bool)] if "stop" in h else h.iloc[0:0]
        ax.plot(stop.layer, stop.criterion, linestyle="none", marker="o", markersize=7, markerfacecolor="white", markeredgecolor=pl.SERIES[k], markeredgewidth=1.6)
    ax.set_yscale("log")
    ax.set_xlabel("номер шару МГУА")
    ax.set_ylabel("критерій регулярності на вибірці B")
    ax.set_xticks(range(1, int(h1.layer.max()) + 1))
    ax.legend(loc="lower left", ncol=2)
    fig.tight_layout()
    fig.savefig(OUT / "fig2_gmdh_layers.png")
    plt.close(fig)


def plot_weights(wdf: pd.DataFrame, array: str) -> None:
    import matplotlib.pyplot as plt

    d1 = wdf[(wdf.array == array) & (wdf.experiment == "1") & (wdf.indicator != "a0")]
    d2 = wdf[(wdf.array == array) & (wdf.experiment == "2") & (wdf.indicator != "a0")]
    x = np.arange(len(d1))
    width = 0.27
    fig, ax = plt.subplots(figsize=(15 * pl.CM, 7.5 * pl.CM))
    ax.bar(x - width, d1.true, width * 0.9, color=pl.SERIES[0], label="задані ваги (МНК відтворює їх точно)")
    ax.bar(x, d1.gmdh, width * 0.9, color=pl.SERIES[1], label="МГУА, дослід 1")
    ax.bar(x + width, d2.gmdh, width * 0.9, color=pl.SERIES[2], label="МГУА, дослід 2")
    ax.axhline(0, color=pl.AXIS, linewidth=0.8)
    ax.set_xticks(x, d1.indicator, fontsize=9)
    ax.set_ylabel("ваговий коефіцієнт")
    ax.yaxis.set_major_formatter(pl.comma_formatter(1))
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(OUT / "fig3_weights_X5.png")
    plt.close(fig)


def plot_experiments(results: pd.DataFrame) -> None:
    import matplotlib.pyplot as plt

    names = list(SIZES)
    x = np.arange(len(names))
    width = 0.27
    fig, ax = plt.subplots(figsize=(15 * pl.CM, 7.2 * pl.CM))
    series = [("1", "дослід 1 (початковий порядок)"), ("2", "дослід 2 (сортування за |Δy|)"), ("2*", "сортування за Δy без модуля")]
    for k, (exp, label) in enumerate(series):
        vals = results[results.experiment == exp].set_index("array").loc[names, "gmdh_sx_check"].to_numpy()
        ax.bar(x + (k - 1) * width, vals, width * 0.9, color=pl.SERIES[k], label=label)
    for i, name in enumerate(names):
        if results[(results.array == name)].gmdh_sx_check.max() < 0.01:
            ax.text(i, 0.6, "≈ 0 у всіх\nвипадках", ha="center", va="bottom", fontsize=9, color=pl.INK_2)
    ax.set_xticks(x, [f"{n}\nm = {SIZES[n]}" for n in names])
    ax.set_ylabel("СКВ моделі МГУА на вибірці B")
    ax.yaxis.set_major_formatter(pl.comma_formatter(0))
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(OUT / "fig4_experiments.png")
    plt.close(fig)


if __name__ == "__main__":
    print(json.dumps(main(), ensure_ascii=False, indent=2))
