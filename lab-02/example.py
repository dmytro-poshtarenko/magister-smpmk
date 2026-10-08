"""Перевірка АСМ на розв'язаному прикладі з частини 2.2 методички (сторінки 46–51).

Первинний опис: F = 10, 15, 13, 19, 14, 18, 17, 11 (T = 1…8), опорна модель
F = a0 + a1·x1 + a2·x2 + a3·x1·x2, де x1 — раніше, а x2 — пізніше з двох попередніх
значень (так складено умовні рівняння методички). Скрипт порівнює надруковані
в методичці числа з розрахованими модулем kg_asm.py.

Запуск: python example.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from kg_asm import GaussSystem, ModelTester, PredictiveModel, ReferenceModel, Term, TimeSeries, gauss_elimination, term

ROOT = Path(__file__).parent
DATA = ROOT / "data"
OUT = ROOT / "outputs"

# x1 = F2 (раніше значення), x2 = F1 (пізніше), як в умовних рівняннях методички.
REFERENCE = ReferenceModel("опорна модель методички", [Term(), term(2), term(1), term(1, 2)])


def main() -> dict:
    OUT.mkdir(exist_ok=True)
    f = pd.read_csv(DATA / "p47-pervynnyi-opys-funktsii.csv")
    series = TimeSeries(f.F.dropna().to_numpy())

    system = GaussSystem.build(series, REFERENCE)
    G, g = system.normal()

    printed = pd.read_csv(DATA / "p50-normalni-rivniannia-hausa.csv")
    G_printed = printed[["c0", "c1", "c2", "c3"]].to_numpy(dtype=float)
    g_printed = printed["y"].to_numpy(dtype=float)
    a_printed = pd.read_csv(DATA / "p50-koefitsiienty-modeli.csv").iloc[0].to_numpy(dtype=float)

    a_ours = system.solve()
    a_printed_system = gauss_elimination(G_printed, g_printed)
    a_lstsq = np.linalg.lstsq(system.A, system.b, rcond=None)[0]

    variants = {
        "надруковані коефіцієнти": a_printed,
        "розв'язок надрукованої системи": a_printed_system,
        "розв'язок повної системи (МНК)": a_ours,
    }
    rows = []
    for name, a in variants.items():
        model = PredictiveModel(REFERENCE, a)
        rows.append({
            "variant": name,
            "a0": a[0], "a1": a[1], "a2": a[2], "a3": a[3],
            "sx": ModelTester.sx(model, series),
            "forecast_T9": model.predict(series.values),
        })
    coef = pd.DataFrame(rows)
    coef.to_csv(OUT / "example_coefficients.csv", index=False)

    equations = pd.DataFrame(system.A[:, 1:3], columns=["x1", "x2"])
    equations.insert(0, "y", system.b)
    equations["x1x2"] = system.A[:, 3]
    equations.to_csv(OUT / "example_conditional_equations.csv", index=False)

    normal = pd.DataFrame({
        "row": [1, 2, 3, 4],
        "printed_rhs": g_printed, "full_rhs": g,
        **{f"printed_c{k}": G_printed[:, k] for k in range(4)},
        **{f"full_c{k}": G[:, k] for k in range(4)},
    })
    normal.to_csv(OUT / "example_normal_equations.csv", index=False)

    # Горизонт прогнозування для прикладу (прийнятна похибка 10 %).
    tester = ModelTester(delta_max=0.10)
    horizon, steps = tester.horizon(series, REFERENCE)
    # Таблиця етапу 6 методички: 2,298 + 1,61·19 + 0,786·14 − 0,09·19·14.
    stage6 = 2.298 + 1.61 * 19 + 0.786 * 14 - 0.09 * 19 * 14

    summary = {
        "equations": int(system.equations),
        "unknowns": REFERENCE.size,
        "normal_matches_numpy": bool(np.allclose(a_ours, a_lstsq)),
        "horizon_delta": 0.10,
        "horizon": horizon,
        "horizon_steps": [s.__dict__ for s in steps],
        "stage6_printed_formula_value": stage6,
        "stage6_printed_value": 17.6,
    }
    (OUT / "example_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


if __name__ == "__main__":
    print(json.dumps(main(), ensure_ascii=False, indent=2))
