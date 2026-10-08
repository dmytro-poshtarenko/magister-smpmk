"""Тестування АСМ у процесі формування переваг ОПР за індивідуальними даними.

Індивідуальний первинний опис — 60 значень функції F з однаковим кроком
(seed = 7). Ряд отримано з логістичної залежності з випадковою складовою:

    u_{t+1} = r·u_t·(1 − u_t/10) + ε_t,   F_t = 10 + u_t,   r = 3,9,   ε_t ~ N(0; 0,05).

Звідси F_{n+1} = (10 − 20r) + 3r·F_n − 0,1r·F_n² + ε, тобто справжня структура
моделі відома, і результат синтезу можна перевірити.

ОПР задає порогові значення характеристик (корисна модель: S_x ≤ 0,5, горизонт ≥ 2
кроки за відносної похибки ≤ 5 %) і порівнює критерії попарно за шкалою Сааті.
Модуль синтезує сім моделей-кандидатів, обчислює їхні характеристики і впорядковує
моделі для трьох профілів переваг ОПР.

Запуск: python individual.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotting as pl
from kg_asm import DecisionSupport, ModelSynthesizer, ModelTester, Preferences, TimeSeries, standard_candidates

ROOT = Path(__file__).parent
DATA = ROOT / "data"
OUT = ROOT / "outputs"

SEED = 7
N = 60
R = 3.9
SIGMA = 0.05
U0 = 3.0
SX_MAX = 0.5
HORIZON_MIN = 2
DELTA_MAX = 0.05
MAX_STEPS = 15

CANDIDATES = standard_candidates()
DESCRIPTION = {
    "M1": "лінійна, одне попереднє значення",
    "M2": "лінійна, два попередні значення",
    "M3": "опорна модель прикладу методички",
    "M4": "квадратична, одне попереднє значення",
    "M5": "повний поліном 2-го степеня, два значення",
    "M6": "поліном методички для двох значень",
    "M7": "повний поліном 2-го степеня, три значення",
}
PROFILES = {
    "ОПР-1 «Планувальник»": np.array([[1, 1 / 3, 3], [3, 1, 5], [1 / 3, 1 / 5, 1]]),
    "ОПР-2 «Аналітик»": np.array([[1, 3, 5], [1 / 3, 1, 3], [1 / 5, 1 / 3, 1]]),
    "ОПР-3 «Інженер супроводу»": np.array([[1, 1, 1 / 3], [1, 1, 1 / 3], [3, 3, 1]]),
}


def generate(seed: int = SEED) -> TimeSeries:
    rng = np.random.default_rng(seed)
    u = np.empty(N)
    u[0] = U0
    for t in range(1, N):
        u[t] = R * u[t - 1] * (1 - u[t - 1] / 10) + rng.normal(0, SIGMA)
        u[t] = min(max(u[t], 0.1), 9.9)  # захист від виходу за межі області визначення
    return TimeSeries(np.round(10 + u, 3), "F")


def main() -> dict:
    DATA.mkdir(exist_ok=True)
    OUT.mkdir(exist_ok=True)
    pl.setup()
    series = generate()
    pd.DataFrame({"T": np.arange(1, N + 1), "F": series.values}).to_csv(DATA / "individual_series.csv", index=False)

    rankings = {}
    evaluated = None
    weights = {}
    for name, matrix in PROFILES.items():
        prefs = Preferences(name, matrix, SX_MAX, HORIZON_MIN, DELTA_MAX)
        ds = DecisionSupport(prefs, MAX_STEPS)
        rows = ds.evaluate(series, CANDIDATES)
        ranked = ds.rank(rows)
        rankings[name] = [{"name": r["name"], "rank": r["rank"], "utility": r["utility"], "useful": r["useful"]} for r in ranked]
        weights[name] = {"weights": prefs.weights.tolist(), **{k: v for k, v in prefs.ahp.items() if k != "weights"}}
        if evaluated is None:
            evaluated = ranked

    # Характеристики моделей-кандидатів (не залежать від профілю ОПР).
    tester = ModelTester(DELTA_MAX)
    cand_rows = []
    horizon_curves = {}
    for ref in CANDIDATES:
        r = next(x for x in evaluated if x["name"] == ref.name)
        h, steps = tester.horizon(series, ref, MAX_STEPS)
        horizon_curves[ref.name] = [(s.steps, s.error) for s in steps]
        cand_rows.append({
            "name": ref.name,
            "description": DESCRIPTION[ref.name],
            "terms": ", ".join(t.label() for t in ref.terms),
            "size": ref.size,
            "sx": r["sx"],
            "horizon": h,
            "useful": r["useful"],
            "coefficients": r["model"].coefficients.tolist(),
            "formula": ref.formula(r["model"].coefficients),
        })
    pd.DataFrame(cand_rows).to_csv(OUT / "candidates.csv", index=False)

    rank_rows = []
    for name, ranked in rankings.items():
        for r in ranked:
            rank_rows.append({"profile": name, **r})
    pd.DataFrame(rank_rows).to_csv(OUT / "rankings.csv", index=False)

    # Справжні коефіцієнти логістичної залежності для перевірки M4.
    true_m4 = [10 - 20 * R, 3 * R, -0.1 * R]
    m4 = ModelSynthesizer().synthesize(series, CANDIDATES[3])

    summary = {
        "seed": SEED, "n": N, "r": R, "sigma": SIGMA,
        "min": float(series.values.min()), "max": float(series.values.max()), "mean": float(series.values.mean()),
        "sx_max": SX_MAX, "horizon_min": HORIZON_MIN, "delta_max": DELTA_MAX,
        "weights": weights,
        "winners": {name: ranked[0]["name"] for name, ranked in rankings.items()},
        "m4_true": true_m4, "m4_fitted": m4.coefficients.tolist(),
        "m4_forecast_next": float(m4.predict(series.values)),
    }
    (OUT / "individual_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))

    plot_series(series, m4)
    plot_horizon(horizon_curves)
    plot_utilities(rankings)
    return summary


def plot_series(series: TimeSeries, m4) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(15 * pl.CM, 6.5 * pl.CM))
    t = np.arange(1, len(series) + 1)
    ax.plot(t, series.values, color=pl.SERIES[0], linewidth=1.6, marker="o", markersize=3.5, label="первинний опис F")
    fit = m4.fitted(series)
    ax.plot(t[1:], fit, color=pl.SERIES[1], linewidth=1.2, linestyle="none", marker="x", markersize=4.5, label="розрахунок за моделлю M4")
    ax.set_xlabel("момент часу T")
    ax.set_ylabel("значення функції F")
    ax.yaxis.set_major_formatter(pl.comma_formatter(0))
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2)
    fig.tight_layout()
    fig.savefig(OUT / "fig_l2_series.png")
    plt.close(fig)


def plot_horizon(curves: dict) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(15 * pl.CM, 7.2 * pl.CM))
    shown = ["M4", "M5", "M6", "M7"]
    for k, name in enumerate(shown):
        pts = curves[name]
        ax.plot([s for s, _ in pts], [100 * e for _, e in pts], marker="o", markersize=4.5, linewidth=1.8, color=pl.SERIES[k], label=name)
    ax.axhline(100 * DELTA_MAX, color=pl.INK_2, linewidth=1.0)
    ax.text(0.6, 100 * DELTA_MAX * 1.12, "прийнятна похибка 5 %", fontsize=8.5, color=pl.INK_2)
    ax.set_yscale("log")
    ax.set_xlabel("кількість кроків прогнозу k (навчання за n − k точками)")
    ax.set_ylabel("відносна похибка n-ї точки, %")
    ax.set_xticks(range(1, MAX_STEPS + 1))
    ax.legend(loc="upper left", ncol=4)
    fig.tight_layout()
    fig.savefig(OUT / "fig_l2_horizon.png")
    plt.close(fig)


def plot_utilities(rankings: dict) -> None:
    import matplotlib.pyplot as plt

    names = ["M4", "M5", "M6", "M7"]
    x = np.arange(len(names))
    width = 0.27
    fig, ax = plt.subplots(figsize=(15 * pl.CM, 6.8 * pl.CM))
    for k, (profile, ranked) in enumerate(rankings.items()):
        util = {r["name"]: r["utility"] for r in ranked}
        ax.bar(x + (k - 1) * width, [util[n] for n in names], width * 0.9, color=pl.SERIES[k], label=profile)
    ax.set_xticks(x, names)
    ax.set_ylabel("корисність U")
    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_formatter(pl.comma_formatter(1))
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3, fontsize=8.5)
    fig.tight_layout()
    fig.savefig(OUT / "fig_l2_utilities.png")
    plt.close(fig)


if __name__ == "__main__":
    print(json.dumps(main(), ensure_ascii=False, indent=2))
