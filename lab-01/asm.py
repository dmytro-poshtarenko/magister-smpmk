"""Алгоритми синтезу моделей (АСМ) для лабораторної роботи № 1.

Модуль містить два алгоритми, що синтезують модель модельованого показника y
за масивом вхідних даних (МВД) X:

* ``synthesize_ls`` — лінійна опорна модель y = a0 + a1*x1 + ... + am*xm,
  коефіцієнти якої знаходять із системи нормальних рівнянь Гауса (метод
  найменших квадратів, МНК);
* ``synthesize_gmdh`` — багаторядний алгоритм МГУА (метод групового врахування
  аргументів): на кожному шарі будуються часткові описи z = a0 + a1*u + a2*v
  для всіх пар вхідних сигналів, кращі F моделей відбираються за зовнішнім
  критерієм регулярності на перевірній вибірці B і передаються на наступний
  шар разом із показниками МВД, що не ввійшли до кращих моделей.

Обидва алгоритми навчаються на перших ``n_train`` спостереженнях (навчальна
вибірка A), а решта спостережень утворює перевірну вибірку B. Тому порядок
спостережень у МВД впливає на результат синтезу.

Часткові описи лінійні, тому будь-яка модель МГУА лишається лінійною функцією
вихідних показників. Модуль зберігає для кожної моделі її розгорнутий вигляд
y = c0 + c1*x1 + ... + cm*xm, і вагові коефіцієнти моделі можна прямо порівняти
із заданими ваговими коефіцієнтами показників.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from math import comb

import numpy as np


@dataclass
class LinearForm:
    """Модель у розгорнутому вигляді y = c0 + c @ x."""

    c0: float
    c: np.ndarray

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.c0 + X @ self.c


@dataclass
class SynthesisResult:
    """Результат синтезу моделі за одним МВД."""

    algorithm: str
    model: LinearForm
    sx_train: float  # СКВ на навчальній вибірці A
    sx_check: float  # СКВ на перевірній вибірці B
    sx_all: float  # СКВ на всьому МВД
    criterion: float  # критерій регулярності на вибірці B
    n_models: int  # скільки моделей-кандидатів згенеровано
    layers: int = 1
    inputs_used: list[int] = field(default_factory=list)  # індекси показників у моделі
    history: list[dict] = field(default_factory=list)  # перебіг по шарах (для МГУА)


def rmse(y: np.ndarray, y_hat: np.ndarray) -> float:
    """Середнє квадратичне відхилення S_x = sqrt(sum((y - y_hat)^2) / n)."""
    return float(np.sqrt(np.mean((y - y_hat) ** 2)))


def regularity(y: np.ndarray, y_hat: np.ndarray) -> float:
    """Зовнішній критерій регулярності: sum((y - y_hat)^2) / sum((y - mean(y))^2).

    Критерій безрозмірний: 0 відповідає точній моделі, 1 — моделі, яка не краща
    за середнє значення y на перевірній вибірці.
    """
    denom = float(np.sum((y - y.mean()) ** 2))
    return float(np.sum((y - y_hat) ** 2)) / denom


def solve_normal_equations(F: np.ndarray, y: np.ndarray, max_cond: float = 1e13) -> np.ndarray | None:
    """Розв'язує систему нормальних рівнянь Гауса (F^T F) a = F^T y.

    Кожне умовне рівняння множиться на коефіцієнт при відповідному невідомому,
    рівняння додаються і діляться на їх кількість. Повертає ``None``, якщо система
    вироджена (наприклад, МВД містить сталий показник, що дублює вільний член).
    """
    n = F.shape[0]
    G = F.T @ F / n
    b = F.T @ y / n
    if not np.all(np.isfinite(G)) or np.linalg.cond(G) > max_cond:
        return None
    return np.linalg.solve(G, b)


def _design(X: np.ndarray) -> np.ndarray:
    return np.column_stack([np.ones(X.shape[0]), X])


def synthesize_ls(X: np.ndarray, y: np.ndarray, n_train: int) -> SynthesisResult | None:
    """МНК-синтез лінійної опорної моделі за всіма показниками МВД."""
    A = slice(0, n_train)
    B = slice(n_train, None)
    a = solve_normal_equations(_design(X[A]), y[A])
    if a is None:
        return None
    model = LinearForm(float(a[0]), a[1:].copy())
    y_hat = model.predict(X)
    return SynthesisResult(
        algorithm="МНК",
        model=model,
        sx_train=rmse(y[A], y_hat[A]),
        sx_check=rmse(y[B], y_hat[B]),
        sx_all=rmse(y, y_hat),
        criterion=regularity(y[B], y_hat[B]),
        n_models=1,
        inputs_used=list(range(X.shape[1])),
    )


@dataclass
class _Signal:
    """Вхідний сигнал шару МГУА: показник МВД або вихід моделі попереднього шару."""

    values: np.ndarray
    form: LinearForm
    inputs: frozenset[int]
    label: str


def synthesize_gmdh(
    X: np.ndarray,
    y: np.ndarray,
    n_train: int,
    freedom: int | None = None,
    max_layers: int = 20,
    min_gain: float = 0.01,
    names: list[str] | None = None,
) -> SynthesisResult:
    """Багаторядний алгоритм МГУА з лінійними частковими описами.

    freedom — свобода вибору F: скільки кращих моделей шару передається далі
    (за замовчуванням дорівнює кількості показників МВД).
    min_gain — мінімальне відносне зменшення критерію, за якого синтез
    продовжується на наступному шарі (критерій зупинки).
    """
    n, m = X.shape
    names = names or [f"x{j + 1}" for j in range(m)]
    A = slice(0, n_train)
    B = slice(n_train, None)
    F = freedom or m

    signals = [
        _Signal(X[:, j], LinearForm(0.0, np.eye(m)[j]), frozenset([j]), names[j])
        for j in range(m)
    ]
    history: list[dict] = []
    best_prev: dict | None = None
    total_models = 0

    for layer in range(1, max_layers + 1):
        if len(signals) < 2:
            break
        candidates = []
        for u, v in combinations(signals, 2):
            if np.allclose(u.values, v.values):
                continue
            Fm = np.column_stack([np.ones(n), u.values, v.values])
            a = solve_normal_equations(Fm[A], y[A])
            if a is None:
                continue
            values = Fm @ a
            form = LinearForm(
                float(a[0] + a[1] * u.form.c0 + a[2] * v.form.c0),
                a[1] * u.form.c + a[2] * v.form.c,
            )
            candidates.append(
                {
                    "values": values,
                    "form": form,
                    "inputs": u.inputs | v.inputs,
                    "label": f"({u.label}, {v.label})",
                    "criterion": regularity(y[B], values[B]),
                }
            )
        total_models += len(candidates)
        if not candidates:
            break
        candidates.sort(key=lambda c: c["criterion"])
        best = candidates[0]
        history.append(
            {
                "layer": layer,
                "inputs": len(signals),
                "models": len(candidates),
                "criterion": best["criterion"],
                "sx_check": rmse(y[B], best["values"][B]),
                "factors": len(best["inputs"]),
            }
        )
        if best_prev is not None and best["criterion"] > best_prev["criterion"] * (1 - min_gain):
            history[-1]["stop"] = True
            break
        best_prev = best | {"layer": layer}
        selected = candidates[:F]
        used = frozenset().union(*(c["inputs"] for c in selected))
        signals = [
            _Signal(c["values"], c["form"], c["inputs"], f"z{layer}.{k + 1}")
            for k, c in enumerate(selected)
        ]
        signals += [
            _Signal(X[:, j], LinearForm(0.0, np.eye(m)[j]), frozenset([j]), names[j])
            for j in range(m)
            if j not in used
        ]

    assert best_prev is not None, "МВД має містити щонайменше два показники"
    model = best_prev["form"]
    y_hat = model.predict(X)
    return SynthesisResult(
        algorithm="МГУА",
        model=model,
        sx_train=rmse(y[A], y_hat[A]),
        sx_check=rmse(y[B], y_hat[B]),
        sx_all=rmse(y, y_hat),
        criterion=regularity(y[B], y_hat[B]),
        n_models=total_models,
        layers=best_prev["layer"],
        inputs_used=sorted(best_prev["inputs"]),
        history=history,
    )


def structural_variety(m: int) -> int:
    """Кількість різних структур лінійної моделі з m показників: 2^m - 1."""
    return 2**m - 1


def first_layer_models(m: int) -> int:
    """Кількість часткових описів першого шару МГУА: C(m, 2)."""
    return comb(m, 2)
