"""Досліди лабораторної роботи № 3 за індивідуальними даними.

1. Синтезатор (завдання 1): модель стану об'єкта моніторингу y = f(x1…x6) за
   квартальним МВД, порівняння одношарових і багатошарових АСМ.
2. Ієрархія моделей (завдання 2): верхній рівень — модель середньорічної концентрації,
   нижній — моделі кварталів, вибір пари за критерієм балансу, каскадно-регресійна
   модель (2) і довготривалий прогноз на роки екзаменаційної вибірки.
3. Стійкість висновків: обидва досліди повторюються на 20 реалізаціях даних (seed 0…19).

Запуск: python experiments.py [--seeds 20]
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import plotting as pl
from hierarchy import (CASCADE, AnnualModel, Series, actual, annual_mvd, balance, choose_pair, fit_seasonal, forecast,
                       moments_forecast, rank_combinations, split_counts, synthesize_seasonal)
from monitoring import (CHECK_LAST, FIRST_YEAR, INDICATORS, LAST_YEAR, MPC, SEED, TRAIN_LAST, generate, split_sizes)
from synthesizer import AlgorithmConstructor, LinearASM, ModelSynthesizer, MultilayerASM, MultilayerModel, rmse

ROOT = Path(__file__).parent
DATA = ROOT / "data"
OUT = ROOT / "outputs"

SX_LIMIT = 0.1 * MPC  # граничне S_x — 10 % ГДК, мг/дм³
K_BEST = 3  # кращих комбінацій сезонних моделей
N_BEST = 2  # кращих моделей верхнього рівня (n < k)
METHODS = {
    "moments": "«мова моментів»: середні за кварталами",
    "single": "одношарова: моделі кварталів без річного рівня",
    "balance": "дворівнева: моделі кварталів, вибрані за критерієм балансу",
    "cascade": "дворівнева каскадно-регресійна модель",
    "cascade_lin": "каскадно-регресійна модель (2), МНК",
}


def asm_by_name(name: str):
    constructor = AlgorithmConstructor()
    return next(a for a in constructor.single_layer() + constructor.multilayer() if a.name == name)


# --- Завдання 1: синтезатор на квартальному МВД ------------------------------------------------------------


def run_synthesizer(q: pd.DataFrame) -> dict:
    names = list(INDICATORS)
    n_train, n_check = split_sizes(q["year"])
    return ModelSynthesizer(SX_LIMIT).run(q[names].to_numpy(), q["y"].to_numpy(), names, n_train, n_check)


# --- Завдання 2: ієрархія моделей ----------------------------------------------------------------------


def run_hierarchy(q: pd.DataFrame, plan: pd.DataFrame) -> dict:
    s = Series.from_frames(q, plan)
    X, y, names, years = annual_mvd(s)
    n_train, n_check = split_counts(years, TRAIN_LAST, CHECK_LAST)
    annual_run = ModelSynthesizer(SX_LIMIT).run(X, y, names, n_train, n_check)
    ranked = sorted((r for _, r in annual_run["log"]), key=lambda r: r.sx["B"])
    # n кращих різних моделей: АСМ, що дали ту саму модель (однакові прогнози), не повторюються.
    annual_models, seen = [], []
    for r in ranked:
        fitted = r.model.predict(annual_run["data"].X)
        if any(np.allclose(fitted, other) for other in seen):
            continue
        seen.append(fitted)
        annual_models.append(AnnualModel(r.asm, r.model, annual_run["data"].names, r.sx))
        if len(annual_models) == N_BEST:
            break
    # Тип АСМ верхнього рівня разом з архітектурою: багатошаровий АСМ на нижньому рівні
    # обмежується кількістю шарів кращої річної моделі.
    type_asm = asm_by_name(ranked[0].asm)
    if isinstance(type_asm, MultilayerASM):
        type_asm.max_layers = ranked[0].layers

    groups = synthesize_seasonal(s, TRAIN_LAST, CHECK_LAST)
    combos = rank_combinations(s, TRAIN_LAST, CHECK_LAST, groups)
    pairs = choose_pair(s, TRAIN_LAST, CHECK_LAST, combos[:K_BEST], annual_models)
    chosen = pairs[0]
    cascade = [fit_seasonal(s, t, CASCADE, type_asm, TRAIN_LAST, CHECK_LAST) for t in range(4)]
    cascade_lin = [fit_seasonal(s, t, CASCADE, LinearASM(), TRAIN_LAST, CHECK_LAST) for t in range(4)]

    horizon = LAST_YEAR - CHECK_LAST
    upper = chosen["annual"]
    forecasts = {
        "moments": moments_forecast(s, TRAIN_LAST, CHECK_LAST, horizon),
        "single": forecast(s, CHECK_LAST, horizon, [g[0] for g in groups], upper),
        "balance": forecast(s, CHECK_LAST, horizon, chosen["seasonal"], upper),
        "cascade": forecast(s, CHECK_LAST, horizon, cascade, upper),
        "cascade_lin": forecast(s, CHECK_LAST, horizon, cascade_lin, upper),
    }
    y_true, Y_true = actual(s, CHECK_LAST, horizon)
    metrics = {}
    for key, f in forecasts.items():
        metrics[key] = {
            "sx_quarterly": rmse(y_true, f.quarterly),
            "sx_annual": rmse(Y_true, f.quarterly.mean(axis=1)),
            "balance": balance(f) if key != "moments" else None,
        }
    metrics["upper"] = {"sx_annual": rmse(Y_true, forecasts["cascade"].annual)}
    return {
        "series": s, "annual_run": annual_run, "annual_models": annual_models, "type": type_asm.name,
        "type_layers": ranked[0].layers,
        "groups": groups, "combos": combos, "pairs": pairs, "chosen": chosen, "cascade": cascade,
        "cascade_lin": cascade_lin, "forecasts": forecasts, "metrics": metrics, "actual": (y_true, Y_true),
    }


# --- Стійкість на кількох реалізаціях ---------------------------------------------------------------------------


def robustness(seeds: int) -> pd.DataFrame:
    rows = []
    for seed in range(seeds):
        q, plan = generate(seed)
        syn = run_synthesizer(q)
        hier = run_hierarchy(q, plan)
        row = {"seed": seed, "output": syn["output"].asm if syn["output"] else "—", "type": hier["type"]}
        row.update({f"asm:{r.asm}": r.sx["C"] for _, r in syn["log"]})
        row.update({f"fc:{k}": hier["metrics"][k]["sx_quarterly"] for k in METHODS})
        rows.append(row)
        print(f"seed {seed}: вихід синтезатора — {row['output']}, тип АСМ верхнього рівня — {row['type']}")
    return pd.DataFrame(rows)


def describe(table: pd.DataFrame, prefix: str) -> list[dict]:
    cols = [c for c in table.columns if c.startswith(prefix)]
    best = table[cols].to_numpy().argmin(axis=1)
    return [{
        "name": c[len(prefix):], "mean": float(table[c].mean()), "median": float(table[c].median()),
        "min": float(table[c].min()), "max": float(table[c].max()), "wins": int((best == j).sum()),
        "below_limit": int((table[c] <= SX_LIMIT).sum()),
    } for j, c in enumerate(cols)]


# --- Підсумок і рисунки ---------------------------------------------------------------------------------


def synthesizer_summary(syn: dict) -> dict:
    data = syn["data"]
    log = []
    for stage, r in syn["log"]:
        log.append({"stage": stage, "asm": r.asm, "model": r.model.name, "params": r.params, "layers": r.layers,
                    "sx": r.sx, "accepted": r.accepted})
    multilayer = {}
    for _, r in syn["log"]:
        if isinstance(r.model, MultilayerModel):
            multilayer[r.asm] = {
                "history": [{k: v for k, v in h.items()} for h in r.model.history],
                "neurons": [{"key": n.key, "layer": n.layer, "inputs": list(n.inputs), "kind": n.kind,
                             "params": n.partial.params} for n in r.model.used()],
                "indicators": r.model.indicators,
            }
    return {
        "names": data.names, "removed": data.removed, "groups": data.groups, "r_y": data.r_y,
        "n_train": data.n_train, "n_check": data.n_check, "n_exam": len(data.y) - data.n_train - data.n_check,
        "log": log, "output": syn["output"].asm if syn["output"] else None, "signals": syn["signals"],
        "multilayer": multilayer,
    }


def hierarchy_summary(h: dict) -> dict:
    run = h["annual_run"]
    linear = next(r for _, r in run["log"] if r.asm == "МНК")
    return {
        "annual_names": run["data"].names, "annual_removed": run["data"].removed,
        "annual_r_y": run["data"].r_y,
        "annual_split": [run["data"].n_train, run["data"].n_check, len(run["data"].y) - run["data"].n_train - run["data"].n_check],
        "annual_log": [{"asm": r.asm, "model": r.model.name, "params": r.params, "layers": r.layers, "sx": r.sx}
                       for _, r in run["log"]],
        "annual_linear_coef": linear.model.coef.tolist(),
        "annual_models": [{"asm": a.asm, "sx": a.sx} for a in h["annual_models"]],
        "type": h["type"], "type_layers": h["type_layers"],
        "seasonal": [[{"inputs": list(m.inputs), "structure": m.structure, "sx": m.sx, "latex": m.latex()} for m in g]
                     for g in h["groups"]],
        "combos": [{"structures": [m.structure for m in c["seasonal"]], "sx_annual": c["sx_annual"],
                    "sx_quarterly": c["sx_quarterly"]} for c in h["combos"][:K_BEST]],
        "n_combos": len(h["combos"]),
        "pairs": [{"k": p["k"], "n": p["n"], "annual": p["annual"].asm, "balance": p["balance"],
                   "sx_annual": p["sx_annual"]} for p in h["pairs"]],
        "chosen": {"k": h["chosen"]["k"], "n": h["chosen"]["n"], "annual": h["chosen"]["annual"].asm,
                   "structures": [m.structure for m in h["chosen"]["seasonal"]]},
        "single": [g[0].structure for g in h["groups"]],
        "cascade": [{"model": m.model.name, "params": m.model.params, "layers": m.model.layers, "sx": m.sx,
                     "neurons": [{"key": n.key, "inputs": list(n.inputs), "kind": n.kind, "params": n.partial.params,
                                  "coef": n.partial.coef.tolist() if n.partial.coef is not None else None}
                                 for n in m.model.used()] if isinstance(m.model, MultilayerModel) else []}
                    for m in h["cascade"]],
        "cascade_lin": [{"latex": m.latex(), "coef": m.model.coef.tolist(), "sx": m.sx} for m in h["cascade_lin"]],
        "metrics": h["metrics"],
        "years": h["forecasts"]["cascade"].years.tolist(),
        "forecast_quarterly": {k: f.quarterly.tolist() for k, f in h["forecasts"].items()},
        "forecast_annual": h["forecasts"]["cascade"].annual.tolist(),
        "actual_quarterly": h["actual"][0].tolist(), "actual_annual": h["actual"][1].tolist(),
    }


def plot_data(q: pd.DataFrame) -> None:
    import matplotlib.pyplot as plt

    t = q["year"] + (q["quarter"] - 0.5) / 4
    fig, axes = plt.subplots(3, 1, figsize=(16 * pl.CM, 15 * pl.CM), sharex=True)
    panels = [("y", "концентрація\nамоній-іонів, мг/дм³", 1), ("x3", "витрата води, м³/с", 0), ("x4", "скид амонійного\nазоту, т", 0)]
    for ax, (col, label, digits) in zip(axes, panels):
        ax.plot(t, q[col], color=pl.SERIES[0], linewidth=1.3)
        ax.set_ylabel(label)
        ax.yaxis.set_major_formatter(pl.comma_formatter(digits))
        for edge in (TRAIN_LAST + 1, CHECK_LAST + 1):
            ax.axvline(edge, color=pl.INK_2, linewidth=0.9, linestyle="--")
    from matplotlib.ticker import MultipleLocator
    axes[0].yaxis.set_major_locator(MultipleLocator(0.2))
    axes[0].axhline(MPC, color=pl.SERIES[1], linewidth=1.1)
    axes[0].text(1.01, MPC, "ГДК", transform=axes[0].get_yaxis_transform(), va="center", fontsize=9, color=pl.INK_2)
    top = axes[0].get_ylim()[1]
    for x0, x1, name in [(FIRST_YEAR, TRAIN_LAST + 1, "A"), (TRAIN_LAST + 1, CHECK_LAST + 1, "B"), (CHECK_LAST + 1, LAST_YEAR + 1, "C")]:
        axes[0].text((x0 + x1) / 2, top * 0.97, f"вибірка {name}", ha="center", va="top", fontsize=9, color=pl.INK_2)
    axes[-1].set_xlabel("рік")
    fig.tight_layout()
    fig.savefig(OUT / "fig_l3_data.png")
    plt.close(fig)


def plot_asm(summary: dict) -> None:
    import matplotlib.pyplot as plt

    log = summary["log"]
    labels = [r["asm"] for r in log]
    values = [r["sx"]["C"] for r in log]
    colors = [pl.SERIES[0] if r["layers"] == 1 and "багатошаровий" not in r["asm"] else pl.SERIES[1] for r in log]
    fig, ax = plt.subplots(figsize=(16 * pl.CM, 7 * pl.CM))
    y = np.arange(len(labels))[::-1]
    ax.barh(y, values, color=colors, height=0.6)
    for yi, v in zip(y, values):
        ax.text(v + 0.001, yi, pl.comma(v, 4), va="center", fontsize=9, color=pl.INK)
    ax.axvline(SX_LIMIT, color=pl.INK_2, linewidth=1.1, linestyle="--")
    ax.text(SX_LIMIT + 0.001, y[0] + 0.55, "граничне значення $S_x$", fontsize=8.5, color=pl.INK_2)
    ax.set_yticks(y, labels)
    ax.set_xlabel("$S_x$ на екзаменаційній вибірці C, мг/дм³")
    ax.xaxis.set_major_formatter(pl.comma_formatter(2))
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0, max(values) * 1.18)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=pl.SERIES[0], label="одношарові АСМ"), Patch(color=pl.SERIES[1], label="багатошарові АСМ")],
              loc="lower right")
    fig.tight_layout()
    fig.savefig(OUT / "fig_l3_asm.png")
    plt.close(fig)


def plot_layers(history: list[dict]) -> None:
    import matplotlib.pyplot as plt

    layers = [h["layer"] for h in history]
    fig, ax = plt.subplots(figsize=(15 * pl.CM, 7 * pl.CM))
    for k, (part, label) in enumerate([("sx_A", "навчальна вибірка A"), ("sx_B", "перевірна вибірка B"), ("sx_C", "екзаменаційна вибірка C")]):
        ax.plot(layers, [h[part] for h in history], marker="o", markersize=5, linewidth=1.8, color=pl.SERIES[k], label=label)
    stop = [h["layer"] for h in history if h.get("stop")]
    if stop:
        ax.axvline(stop[0], color=pl.INK_2, linewidth=0.9, linestyle="--")
        ax.text(stop[0] - 0.08, ax.get_ylim()[1] * 0.97, "зупинка", ha="right", va="top", fontsize=8.5, color=pl.INK_2)
    ax.axhline(SX_LIMIT, color=pl.MUTED, linewidth=0.9)
    ax.text(layers[-1], SX_LIMIT * 1.04, "граничне значення $S_x$", ha="right", va="bottom", fontsize=8.5, color=pl.INK_2)
    ax.set_xticks(layers)
    ax.set_xlabel("номер шару")
    ax.set_ylabel("$S_x$ кращої моделі шару, мг/дм³")
    ax.yaxis.set_major_formatter(pl.comma_formatter(2))
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3)
    fig.tight_layout()
    fig.savefig(OUT / "fig_l3_layers.png")
    plt.close(fig)


def plot_structure(model: MultilayerModel) -> None:
    """Граф структури багатошарової моделі (graphviz)."""
    used = model.used()
    inputs = sorted({s for n in used for s in n.inputs if s in model.names})
    lines = [
        "digraph G {",
        'graph [rankdir=LR, fontname="Arial", dpi=200, nodesep=0.25, ranksep=0.55];',
        'node [fontname="Arial", fontsize=12, color=black];',
        'edge [color=black, arrowsize=0.7];',
    ]
    for name in inputs:
        lines.append(f'"{name}" [shape=box, label="{name}"];')
    by_layer: dict[int, list[str]] = {}
    for n in used:
        lines.append(f'"{n.key}" [shape=ellipse, label="{n.key}\\n{n.kind}"];')
        by_layer.setdefault(n.layer, []).append(n.key)
        for s in n.inputs:
            lines.append(f'"{s}" -> "{n.key}";')
    lines.append("{rank=same; " + " ".join(f'"{x}";' for x in inputs) + "}")
    for keys in by_layer.values():
        lines.append("{rank=same; " + " ".join(f'"{k}";' for k in keys) + "}")
    lines.append(f'"{model.output}" -> "y" ; "y" [shape=plaintext, label="ŷ"];')
    lines.append("}")
    dot = OUT / "structure.dot"
    dot.write_text("\n".join(lines), encoding="utf-8")
    subprocess.run(["dot", "-Tpng", "-o", str(OUT / "fig_l3_structure.png"), str(dot)], check=True)


def plot_forecast(h: dict) -> None:
    import matplotlib.pyplot as plt

    s = h["series"]
    i0 = s.index(TRAIN_LAST)
    hist_years = s.years[i0 + 1:]
    t_hist = (hist_years[:, None] + (np.arange(4) + 0.5) / 4).ravel()
    years = h["forecasts"]["cascade"].years
    t_fc = (years[:, None] + (np.arange(4) + 0.5) / 4).ravel()
    fig, axes = plt.subplots(2, 1, figsize=(16 * pl.CM, 13 * pl.CM), sharex=True, gridspec_kw={"height_ratios": [1.4, 1]})
    ax = axes[0]
    ax.plot(t_hist, s.y[i0 + 1:].ravel(), color=pl.INK, linewidth=1.3, label="фактичні значення")
    show = [("cascade", pl.SERIES[0], "-"), ("single", pl.SERIES[1], "-"), ("moments", pl.SERIES[3], "--")]
    for key, color, style in show:
        ax.plot(t_fc, h["forecasts"][key].quarterly.ravel(), color=color, linewidth=1.6, linestyle=style, label=METHODS[key])
    ax.set_ylabel("концентрація, мг/дм³")
    ax.yaxis.set_major_formatter(pl.comma_formatter(1))
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.05), ncol=2, fontsize=8.5)
    ax2 = axes[1]
    ax2.plot(hist_years + 0.5, s.Y[i0 + 1:], color=pl.INK, linewidth=1.3, marker="o", markersize=3.5, label="фактичне $Y_T$")
    ax2.plot(years + 0.5, h["forecasts"]["cascade"].annual, color=pl.SERIES[0], linewidth=1.6, marker="s", markersize=4,
             label="прогноз моделі верхнього рівня")
    ax2.plot(years + 0.5, h["forecasts"]["single"].quarterly.mean(axis=1), color=pl.SERIES[1], linewidth=1.6, marker="^",
             markersize=4, label="середнє одношарового прогнозу")
    ax2.set_ylabel("середньорічна\nконцентрація, мг/дм³", fontsize=9.5)
    ax2.set_xlabel("рік")
    ax2.yaxis.set_major_formatter(pl.comma_formatter(1))
    ax2.legend(loc="upper center", bbox_to_anchor=(0.5, -0.32), ncol=2, fontsize=8.5)
    for a in axes:
        a.axvline(CHECK_LAST + 1, color=pl.INK_2, linewidth=0.9, linestyle="--")
    axes[0].text(CHECK_LAST + 1.1, axes[0].get_ylim()[1] * 0.98, "початок прогнозу", fontsize=8.5, va="top", color=pl.INK_2)
    fig.tight_layout()
    fig.savefig(OUT / "fig_l3_forecast.png")
    plt.close(fig)


def main() -> dict:
    parser = argparse.ArgumentParser(description="Досліди лабораторної роботи № 3")
    parser.add_argument("--seeds", type=int, default=20, help="кількість реалізацій для перевірки стійкості (0 — не перевіряти)")
    args = parser.parse_args()
    DATA.mkdir(exist_ok=True)
    OUT.mkdir(exist_ok=True)
    pl.setup()

    q, plan = generate(SEED)
    q.round(4).to_csv(DATA / "monitoring.csv", index=False)
    plan.round(4).to_csv(DATA / "annual_plan.csv", index=False)

    syn = run_synthesizer(q)
    hier = run_hierarchy(q, plan)
    summary = {
        "seed": SEED, "sx_limit": SX_LIMIT, "mpc": MPC, "k_best": K_BEST, "n_best": N_BEST,
        "years": [FIRST_YEAR, TRAIN_LAST, CHECK_LAST, LAST_YEAR],
        "stats": q[list(INDICATORS) + ["y"]].describe().loc[["mean", "std", "min", "max"]].to_dict(),
        "synthesizer": synthesizer_summary(syn),
        "hierarchy": hierarchy_summary(hier),
        "methods": METHODS,
    }
    if args.seeds:
        table = robustness(args.seeds)
        table.to_csv(OUT / "seeds.csv", index=False)
        summary["robustness"] = {"seeds": args.seeds, "asm": describe(table, "asm:"), "forecast": describe(table, "fc:"),
                                 "outputs": table["output"].value_counts().to_dict(),
                                 "types": table["type"].value_counts().to_dict()}
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=float))

    plot_data(q)
    plot_asm(summary["synthesizer"])
    output = syn["output"] or min((r for _, r in syn["log"]), key=lambda r: r.sx["B"])
    if isinstance(output.model, MultilayerModel):
        plot_layers(output.model.history)
        plot_structure(output.model)
    plot_forecast(hier)
    return summary


if __name__ == "__main__":
    result = main()
    print(json.dumps({k: result[k] for k in ("seed", "sx_limit")}, ensure_ascii=False))
