"""Ієрархія моделей об'єкта моніторингу і дворівневий довготривалий прогноз.

Реалізує алгоритм рис. 1 методички:

1. таблиця вхідних даних — квартальний МВД об'єкта моніторингу;
2. виявлення інформативності та відбір вхідних параметрів — агрегат A1 синтезатора;
3. визначення типу АСМ за середньорічними показниками — синтезатор на річному МВД
   (верхній рівень: моделі середньорічної концентрації Y_T, n кращих);
4. визначення групи кращих посезонних моделей — моделі кварталів за квартальними
   значеннями (нижній рівень), характеристики комбінацій — за річними значеннями;
5. визначення моделі верхнього рівня як комбінації сезонних моделей за балансом
   сезонних і річних прогнозів.

Після вибору пари будується каскадно-регресійна модель (2) методички
y_{t,T} = f_t(y_{t−1,T}, Y_T), t = 1…4, де f_t синтезує АСМ того типу, що виявився кращим
на верхньому рівні, а Y_T — прогноз моделі верхнього рівня. Прогноз рекурентний
(рис. 2 методички): прогноз кварталу t подається на вхід моделі кварталу t + 1, прогноз
4-го кварталу — на вхід моделі 1-го кварталу наступного року (y_{0,T} = y_{4,T−1}).
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from itertools import product

import numpy as np
import pandas as pd

from synthesizer import ASM, Dataset, LinearASM, LinearModel, Model, rmse

ANNUAL_NAMES = ["Y_prev", "E", "N"]
ANNUAL_LATEX = {"Y_prev": "Y_{T-1}", "E": "E_{T}", "N": "N_{T}"}
SAMPLES_PER_YEAR = 12.0
SEASONAL_STRUCTURES = (("y_prev",), ("y_lag",), ("y_prev", "y_lag"))  # без річних входів
CASCADE = ("y_prev", "Y")  # формула (2) методички


@dataclass
class Series:
    """Квартальні та річні ряди об'єкта: y[i, t] — квартал t року i, Y[i] — середнє за рік."""

    years: np.ndarray
    y: np.ndarray
    Y: np.ndarray
    E: np.ndarray

    @classmethod
    def from_frames(cls, quarterly: pd.DataFrame, plan: pd.DataFrame) -> Series:
        q = quarterly.pivot(index="year", columns="quarter", values="y")
        E = plan.set_index("year").loc[q.index, "E"].to_numpy()
        y = q.to_numpy()
        return cls(q.index.to_numpy(), y, y.mean(axis=1), E)

    def index(self, year: int) -> int:
        return int(np.flatnonzero(self.years == year)[0])


def split_counts(years: np.ndarray, train_last: int, check_last: int) -> tuple[int, int]:
    return int((years <= train_last).sum()), int(((years > train_last) & (years <= check_last)).sum())


# --- Верхній рівень ---------------------------------------------------------------------------------


def annual_mvd(s: Series) -> tuple[np.ndarray, np.ndarray, list[str], np.ndarray]:
    """Річний МВД: Y_{T−1}, плановий ліміт скиду E_T і кількість відборів проб N_T (стала)."""
    X = np.column_stack([s.Y[:-1], s.E[1:], np.full(len(s.Y) - 1, SAMPLES_PER_YEAR)])
    return X, s.Y[1:], list(ANNUAL_NAMES), s.years[1:]


@dataclass
class AnnualModel:
    """Модель верхнього рівня, синтезована синтезатором за річним МВД."""

    asm: str
    model: Model
    names: list[str]
    sx: dict[str, float]

    def predict(self, Y_prev: float, E: float) -> float:
        values = {"Y_prev": Y_prev, "E": E, "N": SAMPLES_PER_YEAR}
        return float(self.model.predict(np.array([[values[name] for name in self.names]]))[0])


# --- Нижній рівень -----------------------------------------------------------------------------------


def seasonal_columns(s: Series, t: int) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
    """Входи і вихід моделі кварталу t (0…3) для років з другого по останній."""
    prev = s.y[1:, t - 1] if t > 0 else s.y[:-1, 3]
    return {"y_prev": prev, "y_lag": s.y[:-1, t], "Y": s.Y[1:]}, s.y[1:, t], s.years[1:]


def symbol(t: int, name: str) -> str:
    """Позначення входу моделі кварталу t (1…4) у LaTeX."""
    if name == "y_prev":
        return "y_{4,T-1}" if t == 1 else f"y_{{{t - 1},T}}"
    return f"y_{{{t},T-1}}" if name == "y_lag" else "Y_{T}"


@dataclass
class SeasonalModel:
    """Модель кварталу t, синтезована АСМ за квартальним МВД з вибраними входами."""

    quarter: int
    inputs: tuple[str, ...]
    model: Model
    sx: dict[str, float]

    def predict(self, values: dict[str, float]) -> float:
        return float(self.model.predict(np.array([[values[name] for name in self.inputs]]))[0])

    @property
    def structure(self) -> str:
        return ", ".join(symbol(self.quarter + 1, name) for name in self.inputs)

    def latex(self, digits: int = 3) -> str:
        """Формула лінійної моделі з числовими коефіцієнтами."""
        if not isinstance(self.model, LinearModel):
            raise TypeError("формула записується лише для лінійної моделі")
        c = self.model.coef
        terms = [f"{c[0]:.{digits}f}"]
        for coef, name in zip(c[1:], self.inputs):
            terms.append(f"{'-' if coef < 0 else '+'} {abs(coef):.{digits}f}\\,{symbol(self.quarter + 1, name)}")
        return f"y_{{{self.quarter + 1},T}} = " + " ".join(terms)


def fit_seasonal(s: Series, t: int, inputs: tuple[str, ...], asm: ASM, train_last: int, check_last: int) -> SeasonalModel:
    columns, target, years = seasonal_columns(s, t)
    n_train, n_check = split_counts(years, train_last, check_last)
    data = Dataset(np.column_stack([columns[name] for name in inputs]), target, list(inputs), n_train, n_check)
    model = copy.deepcopy(asm).synthesize(data)
    sx = {part: rmse(data.part(part)[1], model.predict(data.part(part)[0])) for part in "ABC"}
    return SeasonalModel(t, inputs, model, sx)


def synthesize_seasonal(s: Series, train_last: int, check_last: int, structures=SEASONAL_STRUCTURES,
                        asm: ASM | None = None) -> list[list[SeasonalModel]]:
    """Моделі кожного кварталу для всіх структур, ранжовані за S_x на вибірці B."""
    asm = asm or LinearASM()
    groups = []
    for t in range(4):
        models = [fit_seasonal(s, t, inputs, asm, train_last, check_last) for inputs in structures]
        groups.append(sorted(models, key=lambda m: m.sx["B"]))
    return groups


# --- Дворівневий прогноз і критерій балансу -----------------------------------------------------------------


@dataclass
class Forecast:
    years: np.ndarray
    quarterly: np.ndarray  # (роки, 4)
    annual: np.ndarray  # прогноз моделі верхнього рівня (nan, якщо її немає)


def forecast(s: Series, last_known: int, horizon: int, seasonal: list[SeasonalModel],
             annual: AnnualModel | None) -> Forecast:
    """Рекурентний прогноз на horizon років після року last_known (фактичні дані — до нього включно)."""
    i0 = s.index(last_known)
    y_prev, last_year, Y_prev = s.y[i0, 3], s.y[i0].copy(), s.Y[i0]
    quarterly, annual_values = [], []
    for h in range(1, horizon + 1):
        Y_hat = annual.predict(Y_prev, s.E[i0 + h]) if annual is not None else np.nan
        row = []
        for t in range(4):
            y_prev = seasonal[t].predict({"y_prev": y_prev, "y_lag": last_year[t], "Y": Y_hat})
            row.append(y_prev)
        quarterly.append(row)
        annual_values.append(Y_hat)
        last_year, Y_prev = np.array(row), Y_hat
    return Forecast(s.years[i0 + 1: i0 + 1 + horizon], np.array(quarterly), np.array(annual_values))


def balance(f: Forecast) -> float:
    """Критерій балансу: СКВ між середнім із сезонних прогнозів і річним прогнозом."""
    return rmse(f.quarterly.mean(axis=1), f.annual)


def actual(s: Series, last_known: int, horizon: int) -> tuple[np.ndarray, np.ndarray]:
    i0 = s.index(last_known)
    return s.y[i0 + 1: i0 + 1 + horizon], s.Y[i0 + 1: i0 + 1 + horizon]


def rank_combinations(s: Series, train_last: int, check_last: int,
                      groups: list[list[SeasonalModel]]) -> list[dict]:
    """Комбінації моделей кварталів, ранжовані за точністю прогнозу річних значень на вибірці B."""
    horizon = check_last - train_last
    y_true, Y_true = actual(s, train_last, horizon)
    rows = []
    for combo in product(*groups):
        f = forecast(s, train_last, horizon, list(combo), None)
        rows.append({"seasonal": list(combo), "forecast": f,
                     "sx_annual": rmse(Y_true, f.quarterly.mean(axis=1)), "sx_quarterly": rmse(y_true, f.quarterly)})
    return sorted(rows, key=lambda r: r["sx_annual"])


def choose_pair(s: Series, train_last: int, check_last: int, combos: list[dict],
                annual_models: list[AnnualModel]) -> list[dict]:
    """Пари (комбінація сезонних моделей, модель верхнього рівня), ранжовані за критерієм балансу на B."""
    horizon = check_last - train_last
    rows = []
    for k, combo in enumerate(combos, start=1):
        for n, annual in enumerate(annual_models, start=1):
            f = forecast(s, train_last, horizon, combo["seasonal"], annual)
            rows.append({"k": k, "n": n, "seasonal": combo["seasonal"], "annual": annual, "balance": balance(f),
                         "sx_annual": combo["sx_annual"], "sx_quarterly": combo["sx_quarterly"]})
    return sorted(rows, key=lambda r: r["balance"])


def moments_forecast(s: Series, train_last: int, last_known: int, horizon: int) -> Forecast:
    """«Мова моментів»: у кожному кварталі прогноз — середнє за навчальні роки."""
    means = s.y[s.years <= train_last].mean(axis=0)
    i0 = s.index(last_known)
    return Forecast(s.years[i0 + 1: i0 + 1 + horizon], np.tile(means, (horizon, 1)), np.full(horizon, np.nan))
