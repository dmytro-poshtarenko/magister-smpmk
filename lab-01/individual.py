"""Тестування модуля первинної обробки МВД за індивідуальними даними.

Індивідуальний МВД (seed = 1125) описує 120 об'єктів трьох груп і містить
12 показників:

* x1, x3, x5 — нормальні показники, середні значення яких залежать від групи об'єкта;
* x2, x4, x8 — рівномірні показники, x11 — нормальний показник;
* x6 і x9 — сталі показники (неінформативні);
* x12 — квазісталий показник (коливання ±0,05 навколо 50);
* x7 і x10 — показники, що майже дублюють x2 і x4 (вимірювання тієї самої
  властивості іншим способом).

Модельований показник y = 2,0·x1 + 1,5·x2 + 1,2·x3 + 1,0·x4 + 0,8·x5 + 0,5·x11 + ε,
де ε — похибка вимірювання з нормальним розподілом N(0; 2).

Скрипт застосовує модуль mvd.py, а потім порівнює якість моделей, синтезованих
за різними варіантами сформованого МВД.

Запуск: python individual.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotting as pl
from asm import synthesize_gmdh, synthesize_ls
from mvd import (
    MVD,
    cluster_indicators,
    cluster_observations,
    distribution_law,
    remove_non_informative,
    sort_observations,
    standardize,
    stratified_order,
)

ROOT = Path(__file__).parent
DATA = ROOT / "data"
OUT = ROOT / "outputs"

SEED = 1125
N_OBS = 120
GROUP_SIZES = (45, 40, 35)
TRAIN_SHARE = 0.7
CV_MIN = 0.01  # поріг коефіцієнта варіації для квазісталих показників
R_MIN = 0.9  # поріг |r| для об'єднання показників у кластер
TRUE_WEIGHTS = {"x1": 2.0, "x2": 1.5, "x3": 1.2, "x4": 1.0, "x5": 0.8, "x11": 0.5}


def generate_individual(seed: int = SEED) -> tuple[MVD, np.ndarray]:
    rng = np.random.default_rng(seed)
    group = np.repeat(np.arange(len(GROUP_SIZES)), GROUP_SIZES)
    rng.shuffle(group)
    n = N_OBS
    cols = {
        "x1": rng.normal(np.array([40, 55, 70])[group], 4),
        "x2": rng.uniform(10, 90, n),
        "x3": rng.normal(np.array([20, 32, 26])[group], 3),
        "x4": rng.uniform(5, 25, n),
        "x5": rng.normal(np.array([60, 45, 78])[group], 5),
        "x6": np.full(n, 10.0),
    }
    cols["x7"] = 0.8 * cols["x2"] + rng.normal(0, 2, n)
    cols["x8"] = rng.uniform(0, 50, n)
    cols["x9"] = np.full(n, 3.5)
    cols["x10"] = 1.5 * cols["x4"] + rng.normal(0, 1, n)
    cols["x11"] = rng.normal(15, 3, n)
    cols["x12"] = 50 + rng.uniform(-0.05, 0.05, n)
    y = sum(w * cols[name] for name, w in TRUE_WEIGHTS.items()) + rng.normal(0, 2, n)
    names = list(cols)
    X = np.column_stack([cols[c] for c in names])
    return MVD(X.round(4), y.round(4), names), group


def interleaved_order(n: int, train_share: float = TRAIN_SHARE) -> tuple[np.ndarray, int]:
    """Чергування: з кожних десяти спостережень три (3-тє, 6-те, 9-те) ідуть у вибірку B."""
    pos = np.arange(n)
    check = pos[np.isin(pos % 10, (2, 5, 8))]
    train = pos[~np.isin(pos % 10, (2, 5, 8))]
    return np.concatenate([train, check]), len(train)


def run_variant(label: str, mvd: MVD, n_train: int, note: str = "") -> dict:
    ls = synthesize_ls(mvd.X, mvd.y, n_train)
    gm = synthesize_gmdh(mvd.X, mvd.y, n_train, names=mvd.names)
    return {
        "variant": label,
        "note": note,
        "m": mvd.shape[1],
        "n_train": n_train,
        "ls_sx_train": ls.sx_train if ls else None,
        "ls_sx_check": ls.sx_check if ls else None,
        "gmdh_sx_train": gm.sx_train,
        "gmdh_sx_check": gm.sx_check,
        "gmdh_layers": gm.layers,
        "gmdh_models": gm.n_models,
        "gmdh_inputs": ", ".join(mvd.names[j] for j in gm.inputs_used),
    }


def main() -> dict:
    DATA.mkdir(exist_ok=True)
    OUT.mkdir(exist_ok=True)
    pl.setup()

    raw, group = generate_individual()
    raw.to_csv(DATA / "individual_mvd.csv")
    n_train = round(N_OBS * TRAIN_SHARE)

    # 1. Закон розподілу кожного показника.
    law_rows = []
    for j, name in enumerate(raw.names):
        x = raw.X[:, j]
        row = {"indicator": name, "mean": x.mean(), "std": x.std(ddof=1), "cv": x.std(ddof=1) / abs(x.mean())}
        if np.ptp(x) > 0:
            row |= distribution_law(x)
        law_rows.append(row)

    # 2. Видалення неінформативних показників.
    clean, removed = remove_non_informative(raw, CV_MIN)

    # 3. Кластеризація показників.
    ind = cluster_indicators(clean, R_MIN)
    groups = [
        {"members": [clean.names[j] for j in g["members"]], "representative": clean.names[g["representative"]]}
        for g in ind["groups"]
    ]
    reduced = clean.columns(sorted(g["representative"] for g in ind["groups"]))

    # 4. Кластеризація спостережень (за показниками, що лишилися після скорочення).
    obs = cluster_observations(reduced, seed=0)
    crosstab = pd.crosstab(pd.Series(group + 1, name="group"), pd.Series(obs["labels"] + 1, name="cluster"))

    for row in law_rows:
        name = row["indicator"]
        if name in removed:
            row["decision"] = f"видалено: {removed[name]}"
        else:
            g = next(g for g in groups if name in g["members"])
            if len(g["members"]) == 1:
                row["decision"] = "залишено"
            elif g["representative"] == name:
                row["decision"] = "представник кластера " + "{" + ", ".join(g["members"]) + "}"
            else:
                row["decision"] = f"видалено: дублює {g['representative']}"
    laws = pd.DataFrame(law_rows)
    laws.to_csv(OUT / "individual_laws.csv", index=False)

    # 5. Вплив способу формування МВД на результати моделювання.
    variants = [run_variant("V0", raw, n_train, "первинний МВД, початковий порядок")]
    variants.append(run_variant("V1", clean, n_train, "без неінформативних показників, початковий порядок"))
    variants.append(run_variant("V2", sort_observations(clean, "abs_dev"), n_train, "сортування за |Δy|, послідовний поділ"))
    srt = sort_observations(clean, "abs_dev")
    order, n_a = interleaved_order(len(srt.y))
    variants.append(run_variant("V3", srt.rows(order), n_a, "сортування за |Δy|, чергування спостережень"))
    order, n_a = stratified_order(obs["labels"], TRAIN_SHARE)
    variants.append(run_variant("V4", clean.rows(order), n_a, "стратифікація за кластерами спостережень"))
    variants.append(run_variant("V5", reduced.rows(order), n_a, "V4 + представники кластерів показників"))
    var_df = pd.DataFrame(variants)
    var_df.to_csv(OUT / "individual_variants.csv", index=False)

    # Оброблений МВД: без неінформативних і дубльованих показників, відсортований за |Δy|.
    sort_observations(reduced, "abs_dev").to_csv(DATA / "individual_mvd_processed.csv")

    plot_clusters(clean, ind, reduced, obs)
    plot_variants(var_df)

    summary = {
        "removed": removed,
        "indicator_groups": groups,
        "reduced": reduced.names,
        "k": obs["k"],
        "silhouette": {int(k): round(v, 4) for k, v in obs["silhouette"].items()},
        "sizes": obs["sizes"].tolist(),
        "centers": {name: obs["centers"][:, j].round(2).tolist() for j, name in enumerate(reduced.names)},
        "y_means": obs["y_means"].round(2).tolist(),
        "crosstab": crosstab.to_dict(),
        "y_mean": float(raw.y.mean()),
        "y_std": float(raw.y.std()),
    }
    (OUT / "individual_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=int))
    return summary


def plot_clusters(clean: MVD, ind: dict, reduced: MVD, obs: dict) -> None:
    import matplotlib.pyplot as plt
    from scipy.cluster.hierarchy import dendrogram

    fig, axes = plt.subplots(1, 2, figsize=(15.5 * pl.CM, 7.2 * pl.CM), gridspec_kw={"width_ratios": [1, 1.1]})
    ax = axes[0]
    dendrogram(
        ind["linkage"],
        labels=clean.names,
        ax=ax,
        color_threshold=0,
        above_threshold_color=pl.INK_2,
        leaf_font_size=9,
    )
    ax.axhline(1 - R_MIN, color=pl.SERIES[1], linewidth=1.6)
    ax.text(ax.get_xlim()[0] + 2, 1 - R_MIN + 0.015, "поріг d = 0,1 (|r| = 0,9)", ha="left", va="bottom", fontsize=8.5, color=pl.INK_2, bbox={"facecolor": "white", "edgecolor": "none", "pad": 1.5})
    ax.set_ylabel("відстань d = 1 − |r|")
    ax.set_yticks(np.arange(0, 1.01, 0.2))
    ax.yaxis.set_major_formatter(pl.comma_formatter(1))
    ax.grid(axis="x", visible=False)
    ax.set_title("а) кластеризація показників", fontsize=10, color=pl.INK, loc="left")

    ax = axes[1]
    Z = standardize(reduced.X)
    _, _, vt = np.linalg.svd(Z, full_matrices=False)
    pcs = Z @ vt[:2].T
    for c in range(obs["k"]):
        sel = obs["labels"] == c
        ax.scatter(pcs[sel, 0], pcs[sel, 1], s=22, color=pl.SERIES[c], edgecolor="white", linewidth=0.6, label=f"кластер {c + 1} ({sel.sum()})")
    ax.set_xlabel("перша головна компонента")
    ax.set_ylabel("друга головна компонента")
    ax.xaxis.set_major_formatter(pl.comma_formatter(0))
    ax.yaxis.set_major_formatter(pl.comma_formatter(0))
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3, fontsize=8.5, handletextpad=0.2, columnspacing=0.8)
    ax.set_title("б) кластери спостережень", fontsize=10, color=pl.INK, loc="left")
    fig.tight_layout()
    fig.savefig(OUT / "fig5_clusters.png")
    plt.close(fig)


def plot_variants(var_df: pd.DataFrame) -> None:
    import matplotlib.pyplot as plt

    labels = var_df.variant.tolist()
    x = np.arange(len(labels))
    width = 0.38
    fig, ax = plt.subplots(figsize=(15 * pl.CM, 7 * pl.CM))
    ls_vals = var_df.ls_sx_check.astype(float).to_numpy()
    ax.bar(x - width / 2, np.nan_to_num(ls_vals), width * 0.9, color=pl.SERIES[0], label="МНК")
    ax.bar(x + width / 2, var_df.gmdh_sx_check, width * 0.9, color=pl.SERIES[1], label="МГУА")
    for i, v in enumerate(ls_vals):
        if np.isnan(v):
            ax.text(i - width / 2, 0.3, "модель не\nсинтезовано", ha="center", va="bottom", fontsize=8, color=pl.INK_2)
    ax.set_xticks(x, labels)
    ax.set_ylabel("СКВ на перевірній вибірці B")
    ax.yaxis.set_major_formatter(pl.comma_formatter(0))
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(OUT / "fig6_variants.png")
    plt.close(fig)


if __name__ == "__main__":
    print(json.dumps(main(), ensure_ascii=False, indent=2, default=int))
