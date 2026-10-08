"""Індивідуальні дані до лабораторної роботи № 3: об'єкт моніторингу.

Об'єкт моніторингу — умовний гідрохімічний пост на малій річці нижче скиду
очисних споруд; модельований показник — середньоквартальна концентрація
амоній-іонів. Як і в роботах № 1 і № 2, набір власний: його згенеровано з
фіксованим seed, тож результати відтворюються. Ряд охоплює 32 роки по 4 квартали.

Механізм формування даних (невідомий синтезатору):

* E_T — річний ліміт скиду амонійного азоту, т: спадає через модернізацію очисних
  споруд і коливається з циклом 7 років; відомий заздалегідь (плановий показник);
* W_T — водність року (логарифмічна шкала), авторегресія 1-го порядку;
* x1 — температура повітря, °C; x2 — сума опадів, мм; x3 — витрата води, м³/с
  (сезонна норма × водність року × вплив опадів); x4 — скид амонійного азоту за
  квартал, т (частка річного ліміту); x5 — кількість відборів проб за квартал
  (стала, неінформативний показник); x6 — температура води, °C (майже лінійна
  функція x1, дублює її);
* y = 0,15 + 0,1268·x4 / x3 · exp(−0,03·(x6 − 10)) · (1 + ε), де 0,1268 переводить
  т/квартал і м³/с у мг/дм³, а множник з температурою води описує нітрифікацію.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 3
FIRST_YEAR, LAST_YEAR = 1993, 2024
TRAIN_LAST, CHECK_LAST = 2012, 2018  # A: 1993–2012, B: 2013–2018, C: 2019–2024

AIR = np.array([-2.0, 14.5, 19.5, 4.5])  # сезонна норма температури повітря, °C
RAIN = np.array([105.0, 165.0, 170.0, 130.0])  # норма опадів за квартал, мм
FLOW = np.array([11.0, 13.0, 4.5, 6.5])  # норма витрати води, м³/с
SHARE = np.array([0.24, 0.25, 0.26, 0.25])  # частка річного ліміту скиду за квартал
SAMPLES = 3.0  # відборів проб за квартал
MPC = 0.5  # ГДК амоній-іонів для рибогосподарських водойм, мг/дм³

INDICATORS = {
    "x1": "Температура повітря, °C",
    "x2": "Сума опадів, мм",
    "x3": "Витрата води, м³/с",
    "x4": "Скид амонійного азоту, т",
    "x5": "Кількість відборів проб",
    "x6": "Температура води, °C",
}
TARGET = "Концентрація амоній-іонів, мг/дм³"


def generate(seed: int = SEED) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Повертає квартальний МВД і річні планові та прихованi показники."""
    rng = np.random.default_rng(seed)
    years = np.arange(FIRST_YEAR, LAST_YEAR + 1)
    k = np.arange(len(years))
    limit = 160 - 2.2 * k + 10 * np.sin(2 * np.pi * k / 7) + rng.normal(0, 4, len(years))
    wet = np.zeros(len(years))
    for i in range(1, len(years)):
        wet[i] = 0.6 * wet[i - 1] + rng.normal(0, 0.15)

    rows = []
    for i, year in enumerate(years):
        for t in range(4):
            x1 = AIR[t] + 0.04 * i + rng.normal(0, 1.3)
            x2 = max(RAIN[t] + rng.normal(0, 30), 15.0)
            x3 = FLOW[t] * np.exp(wet[i] + 0.003 * (x2 - RAIN[t]) + rng.normal(0, 0.08))
            x4 = limit[i] * SHARE[t] * (1 + rng.normal(0, 0.05))
            x6 = max(0.85 * x1 + 4 + rng.normal(0, 0.8), 0.2)
            y = 0.15 + 0.1268 * x4 / x3 * np.exp(-0.03 * (x6 - 10)) * (1 + rng.normal(0, 0.05))
            rows.append({"year": year, "quarter": t + 1, "x1": x1, "x2": x2, "x3": x3, "x4": x4,
                         "x5": SAMPLES, "x6": x6, "y": y})
    quarterly = pd.DataFrame(rows)
    annual = pd.DataFrame({"year": years, "E": limit, "W": wet})
    return quarterly, annual


def split_sizes(years: pd.Series) -> tuple[int, int]:
    """Кількість спостережень у вибірках A і B для хронологічного поділу."""
    return int((years <= TRAIN_LAST).sum()), int(((years > TRAIN_LAST) & (years <= CHECK_LAST)).sum())


def annual_table(quarterly: pd.DataFrame, plan: pd.DataFrame) -> pd.DataFrame:
    """Середньорічні значення: Y_T — середня концентрація року, E_T — річний ліміт скиду."""
    g = quarterly.groupby("year")
    table = pd.DataFrame({
        "year": g.size().index,
        "Y": g["y"].mean().to_numpy(),
        "x1": g["x1"].mean().to_numpy(),
        "x2": g["x2"].sum().to_numpy(),
        "x3": g["x3"].mean().to_numpy(),
        "x4": g["x4"].sum().to_numpy(),
    })
    return table.merge(plan[["year", "E"]], on="year")


def main() -> None:
    parser = argparse.ArgumentParser(description="Генерує індивідуальні дані об'єкта моніторингу")
    parser.add_argument("--out", type=Path, default=Path(__file__).with_name("data"))
    args = parser.parse_args()
    args.out.mkdir(exist_ok=True)
    quarterly, plan = generate()
    quarterly.round(4).to_csv(args.out / "monitoring.csv", index=False)
    plan.round(4).to_csv(args.out / "annual_plan.csv", index=False)
    print(quarterly.describe().round(3).to_string())


if __name__ == "__main__":
    main()
