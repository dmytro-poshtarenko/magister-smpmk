"""Алгоритм синтезу прогнозних моделей (АСМ) на основі полінома Колмогорова-Габора.

Модуль реалізує етапи, описані в частині 2.2 методички, і класи, показані на
діаграмі класів лабораторної роботи № 2:

1. ``ReferenceModel`` — опорна модель: набір членів полінома Колмогорова-Габора;
2. ``TimeSeries`` — первинний опис функції (відомі значення через однаковий крок);
3. ``GaussSystem`` — система умовних рівнянь Гауса і її нормалізація;
4. ``gauss_elimination`` — розв'язання системи нормальних рівнянь методом Гауса;
5. ``ModelSynthesizer`` — синтез (навчання) моделі ``PredictiveModel``;
6. ``ModelTester`` — характеристики моделі: СКВ і горизонт прогнозування;
7. ``Preferences`` і ``DecisionSupport`` — формування переваг особи, що приймає
   рішення (ОПР), і вибір моделі за цими перевагами (метод аналізу ієрархій).

Позначення: F1 — останнє відоме значення функції (F_n), F2 — передостаннє (F_{n−1}) і т. д.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations_with_replacement

import numpy as np

# --- Опорна модель -------------------------------------------------------------------------------


@dataclass(frozen=True)
class Term:
    """Член полінома: добуток степенів попередніх значень, наприклад F1²·F2.

    powers — пари (лаг, степінь); порожній кортеж означає вільний член.
    """

    powers: tuple[tuple[int, int], ...] = ()

    @property
    def degree(self) -> int:
        return sum(p for _, p in self.powers)

    @property
    def max_lag(self) -> int:
        return max((lag for lag, _ in self.powers), default=0)

    def value(self, history: np.ndarray) -> float:
        """Значення члена за історією; history[-1] = F1, history[-2] = F2, …"""
        result = 1.0
        for lag, power in self.powers:
            result *= history[-lag] ** power
        return result

    def label(self) -> str:
        if not self.powers:
            return "1"
        sup = str.maketrans("0123456789", "⁰¹²³⁴⁵⁶⁷⁸⁹")
        return "·".join(f"F{lag}" + (str(p).translate(sup) if p > 1 else "") for lag, p in self.powers)

    def latex(self) -> str:
        if not self.powers:
            return ""
        return "".join(f"F_{{{lag}}}" + (f"^{{{p}}}" if p > 1 else "") for lag, p in self.powers)


def term(*factors: int) -> Term:
    """term(1, 1, 2) → F1²·F2: кожен аргумент — лаг одного множника."""
    counts: dict[int, int] = {}
    for lag in factors:
        counts[lag] = counts.get(lag, 0) + 1
    return Term(tuple(sorted(counts.items())))


@dataclass
class ReferenceModel:
    """Опорна модель — сукупність членів полінома Колмогорова-Габора."""

    name: str
    terms: list[Term]

    @property
    def lags(self) -> int:
        return max(t.max_lag for t in self.terms)

    @property
    def size(self) -> int:
        """Кількість невідомих коефіцієнтів."""
        return len(self.terms)

    def row(self, history: np.ndarray) -> np.ndarray:
        return np.array([t.value(history) for t in self.terms])

    def formula(self, coefficients: np.ndarray | None = None, digits: int = 3) -> str:
        text = ""
        for k, t in enumerate(self.terms):
            if coefficients is None:
                sign, coef = "+", f"a{k}"
            else:
                sign, coef = ("−" if coefficients[k] < 0 else "+"), f"{abs(coefficients[k]):.{digits}f}"
            body = coef if not t.powers else f"{coef}·{t.label()}"
            text += (("−" if sign == "−" else "") + body) if k == 0 else f" {sign} {body}"
        return "F = " + text

    def latex(self) -> str:
        parts = [f"a_{{{k}}}{t.latex()}" for k, t in enumerate(self.terms)]
        return "F_{n+1} = " + " + ".join(parts)

    @classmethod
    def kolmogorov_gabor(cls, lags: int, degree: int, name: str | None = None) -> ReferenceModel:
        """Повний поліном Колмогорова-Габора заданого степеня від lags попередніх значень."""
        terms = [Term()]
        for d in range(1, degree + 1):
            for combo in combinations_with_replacement(range(1, lags + 1), d):
                terms.append(term(*combo))
        return cls(name or f"КГ(лагів {lags}, степінь {degree})", terms)


def standard_candidates() -> list[ReferenceModel]:
    """Опорні моделі-кандидати M1–M7, з яких ОПР обирає модель."""
    return [
        ReferenceModel("M1", [Term(), term(1)]),
        ReferenceModel("M2", [Term(), term(1), term(2)]),
        ReferenceModel("M3", [Term(), term(1), term(2), term(1, 2)]),
        ReferenceModel("M4", [Term(), term(1), term(1, 1)]),
        ReferenceModel.kolmogorov_gabor(2, 2, "M5"),
        ReferenceModel("M6", [Term(), term(1), term(2), term(1, 2), term(1, 1, 2), term(1, 2, 2)]),
        ReferenceModel.kolmogorov_gabor(3, 2, "M7"),
    ]


def parse_terms(spec: str, name: str = "задана модель") -> ReferenceModel:
    """Опорна модель з рядка на кшталт "1; F1; F2; F1*F2; F1^2"."""
    terms = []
    for item in spec.replace(" ", "").split(";"):
        if not item:
            continue
        if item == "1":
            terms.append(Term())
            continue
        lags: list[int] = []
        for factor in item.split("*"):
            base, _, power = factor.partition("^")
            if not base.upper().startswith("F"):
                raise ValueError(f"незрозумілий множник {factor!r}")
            lags += [int(base[1:])] * int(power or 1)
        terms.append(term(*lags))
    return ReferenceModel(name, terms)


# --- Первинний опис і система рівнянь ------------------------------------------------------------


@dataclass
class TimeSeries:
    """Первинний опис функції: значення через однаковий крок дискретизації."""

    values: np.ndarray
    name: str = "F"

    def __post_init__(self) -> None:
        self.values = np.asarray(self.values, dtype=float)

    def __len__(self) -> int:
        return len(self.values)

    def head(self, n: int) -> TimeSeries:
        return TimeSeries(self.values[:n], self.name)


@dataclass
class GaussSystem:
    """Система умовних рівнянь Гауса: A·a = b (рядок — одна ділянка функції)."""

    A: np.ndarray
    b: np.ndarray

    @classmethod
    def build(cls, series: TimeSeries, reference: ReferenceModel) -> GaussSystem:
        """Кожна ділянка з lags + 1 сусідніх точок дає одне умовне рівняння."""
        lags = reference.lags
        v = series.values
        rows = [reference.row(v[t - lags:t]) for t in range(lags, len(v))]
        return cls(np.array(rows), v[lags:].copy())

    @property
    def equations(self) -> int:
        return len(self.b)

    def normal(self, divide: bool = False) -> tuple[np.ndarray, np.ndarray]:
        """Нормалізація: кожне умовне рівняння множать на коефіцієнт при k-му невідомому
        і додають усі рівняння; так отримують k-те нормальне рівняння.

        divide=True ділить рівняння на кількість умовних рівнянь (розв'язок не змінюється).
        """
        G = self.A.T @ self.A
        g = self.A.T @ self.b
        if divide:
            G, g = G / self.equations, g / self.equations
        return G, g

    def solve(self) -> np.ndarray:
        G, g = self.normal()
        return gauss_elimination(G, g)


def gauss_elimination(M: np.ndarray, v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Розв'язок системи M·x = v методом Гауса з вибором головного елемента."""
    M = np.array(M, dtype=float)
    v = np.array(v, dtype=float)
    n = len(v)
    scale = np.abs(M).max()
    for k in range(n):
        p = k + int(np.argmax(np.abs(M[k:, k])))
        if abs(M[p, k]) <= eps * scale:
            raise np.linalg.LinAlgError("система нормальних рівнянь вироджена")
        if p != k:
            M[[k, p]] = M[[p, k]]
            v[[k, p]] = v[[p, k]]
        for i in range(k + 1, n):
            f = M[i, k] / M[k, k]
            M[i, k:] -= f * M[k, k:]
            v[i] -= f * v[k]
    x = np.zeros(n)
    for k in range(n - 1, -1, -1):
        x[k] = (v[k] - M[k, k + 1:] @ x[k + 1:]) / M[k, k]
    return x


# --- Модель, синтез і випробування ------------------------------------------------------------------


@dataclass
class PredictiveModel:
    """Навчена прогнозна модель: опорна модель з визначеними коефіцієнтами."""

    reference: ReferenceModel
    coefficients: np.ndarray

    def predict(self, history: np.ndarray) -> float:
        return float(self.reference.row(np.asarray(history, dtype=float)) @ self.coefficients)

    def forecast(self, history: np.ndarray, steps: int) -> np.ndarray:
        """Рекурсивний прогноз на steps кроків уперед: прогнозоване значення стає історією."""
        h = list(np.asarray(history, dtype=float))
        out = []
        for _ in range(steps):
            nxt = self.predict(np.array(h))
            out.append(nxt)
            h.append(nxt)
        return np.array(out)

    def fitted(self, series: TimeSeries) -> np.ndarray:
        """Розраховані значення для точок, за якими модель навчалася."""
        lags = self.reference.lags
        v = series.values
        return np.array([self.predict(v[t - lags:t]) for t in range(lags, len(v))])


class ModelSynthesizer:
    """Синтез моделі: умовні рівняння → нормальні рівняння → коефіцієнти."""

    def synthesize(self, series: TimeSeries, reference: ReferenceModel) -> PredictiveModel:
        system = GaussSystem.build(series, reference)
        if system.equations < reference.size:
            raise ValueError(
                f"для {reference.size} коефіцієнтів потрібно щонайменше {reference.size} умовних рівнянь, "
                f"а є {system.equations}"
            )
        return PredictiveModel(reference, system.solve())


@dataclass
class HorizonStep:
    steps: int  # на скільки кроків прогнозували
    train_points: int  # за скількома точками навчали
    predicted: list[float]
    actual: list[float]
    error: float  # відносна похибка прогнозу останньої (n-ї) точки
    acceptable: bool


class ModelTester:
    """Випробовувач моделей: середнє квадратичне відхилення і горизонт прогнозування."""

    def __init__(self, delta_max: float = 0.05, synthesizer: ModelSynthesizer | None = None) -> None:
        self.delta_max = delta_max  # прийнятна відносна похибка прогнозу
        self.synthesizer = synthesizer or ModelSynthesizer()

    @staticmethod
    def sx(model: PredictiveModel, series: TimeSeries) -> float:
        """S_x = sqrt(sum((Y_експ − Y_розр)²) / n) за точками, за якими модель навчалася."""
        y = series.values[model.reference.lags:]
        return float(np.sqrt(np.mean((y - model.fitted(series)) ** 2)))

    def horizon(self, series: TimeSeries, reference: ReferenceModel, max_steps: int = 12) -> tuple[int, list[HorizonStep]]:
        """Горизонт прогнозування за методичкою: навчання за n − k точками і прогноз k точок.

        Ітерації k = 1, 2, … тривають, доки похибка прогнозу n-ї точки прийнятна і точок
        вистачає для навчання. Горизонт дорівнює останньому k з прийнятною похибкою.
        """
        n = len(series)
        steps: list[HorizonStep] = []
        horizon = 0
        for k in range(1, max_steps + 1):
            train = series.head(n - k)
            if len(train) - reference.lags < reference.size:
                break  # дослідження обмежене кількістю точок
            try:
                model = self.synthesizer.synthesize(train, reference)
            except np.linalg.LinAlgError:
                break  # за меншою кількістю точок система вироджена
            pred = model.forecast(train.values, k)
            actual = series.values[n - k:]
            error = abs(pred[-1] - actual[-1]) / abs(actual[-1])
            ok = bool(error <= self.delta_max)
            steps.append(HorizonStep(k, n - k, pred.tolist(), actual.tolist(), float(error), ok))
            if not ok:
                break
            horizon = k
        return horizon, steps


# --- Переваги ОПР і вибір моделі ------------------------------------------------------------------

RANDOM_INDEX = {1: 0.0, 2: 0.0, 3: 0.58, 4: 0.90, 5: 1.12, 6: 1.24, 7: 1.32, 8: 1.41, 9: 1.45}


def ahp_weights(matrix: np.ndarray) -> dict:
    """Ваги критеріїв за матрицею парних порівнянь Сааті (власний вектор λ_max)."""
    M = np.asarray(matrix, dtype=float)
    vals, vecs = np.linalg.eig(M)
    k = int(np.argmax(vals.real))
    w = np.abs(vecs[:, k].real)
    w = w / w.sum()
    n = len(M)
    lam = float(vals[k].real)
    ci = (lam - n) / (n - 1) if n > 1 else 0.0
    ri = RANDOM_INDEX.get(n, 1.49)
    cr = ci / ri if ri else 0.0
    return {"weights": w, "lambda_max": lam, "CI": ci, "CR": cr}


@dataclass
class Preferences:
    """Переваги ОПР: порогові значення характеристик і парні порівняння критеріїв.

    Критерії: точність (менше S_x), горизонт прогнозування (більше), простота (менше коефіцієнтів).
    """

    name: str
    pairwise: np.ndarray
    sx_max: float
    horizon_min: int
    delta_max: float = 0.05
    criteria: tuple[str, ...] = ("точність", "горизонт", "простота")
    _ahp: dict = field(default=None, init=False, repr=False)

    @property
    def ahp(self) -> dict:
        if self._ahp is None:
            self._ahp = ahp_weights(self.pairwise)
        return self._ahp

    @property
    def weights(self) -> np.ndarray:
        return self.ahp["weights"]


class DecisionSupport:
    """Синтезує моделі-кандидати, оцінює їх і впорядковує за перевагами ОПР."""

    def __init__(self, preferences: Preferences, max_steps: int = 12) -> None:
        self.preferences = preferences
        self.synthesizer = ModelSynthesizer()
        self.tester = ModelTester(preferences.delta_max, self.synthesizer)
        self.max_steps = max_steps

    def evaluate(self, series: TimeSeries, candidates: list[ReferenceModel]) -> list[dict]:
        rows = []
        for ref in candidates:
            row = {"name": ref.name, "reference": ref, "size": ref.size}
            try:
                model = self.synthesizer.synthesize(series, ref)
            except (np.linalg.LinAlgError, ValueError) as exc:
                # Вироджена система або нестача даних: модель не синтезовано.
                rows.append(row | {"model": None, "sx": float("inf"), "horizon": 0, "error": str(exc)})
                continue
            h, _ = self.tester.horizon(series, ref, self.max_steps)
            rows.append(row | {"model": model, "sx": self.tester.sx(model, series), "horizon": h})
        return rows

    def rank(self, rows: list[dict]) -> list[dict]:
        """Корисні моделі (характеристики не гірші за пороги ОПР) йдуть першими, далі за корисністю.

        Корисність U = Σ w_i·u_i, де u_i — значення критеріїв, нормовані на [0; 1] за
        діапазоном серед корисних моделей (саме з них ОПР обирає модель).
        """
        p = self.preferences
        for r in rows:
            r["useful"] = bool(r["sx"] <= p.sx_max and r["horizon"] >= p.horizon_min)
        base = [r for r in rows if r["useful"]]
        if len(base) < 2:
            base = rows
        sx = np.array([r["sx"] for r in base])
        hz = np.array([r["horizon"] for r in base], dtype=float)
        sz = np.array([r["size"] for r in base], dtype=float)

        def scale(value: float, x: np.ndarray, higher_better: bool) -> float:
            span = x.max() - x.min()
            if span == 0:
                return 1.0
            u = (value - x.min()) / span if higher_better else (x.max() - value) / span
            return float(np.clip(u, 0.0, 1.0))

        for r in rows:
            u = np.array([scale(r["sx"], sx, False), scale(r["horizon"], hz, True), scale(r["size"], sz, False)])
            r["u"] = u
            r["utility"] = float(u @ p.weights)
        ranked = sorted(rows, key=lambda r: (not r["useful"], -r["utility"]))
        for k, r in enumerate(ranked, start=1):
            r["rank"] = k
        return ranked
