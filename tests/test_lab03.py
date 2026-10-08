"""Тести синтезатора багатошарових моделей і ієрархії моделей (лабораторна робота № 3)."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

LAB = Path(__file__).resolve().parents[1] / "lab-03"
sys.path.insert(0, str(LAB))

from hierarchy import Series, forecast, synthesize_seasonal
from monitoring import generate, split_sizes
from synthesizer import (
    AlgorithmConstructor,
    Controller,
    Dataset,
    LinearASM,
    ModelSynthesizer,
    ModelTester,
    MultilayerASM,
    MVDFormer,
    PerceptronASM,
    TestResult,
    fit_partial,
)


def make_data(n=60, seed=1):
    rng = np.random.default_rng(seed)
    x1, x2, x3 = rng.uniform(1, 3, (3, n))
    X = np.column_stack([x1, x2, x3, np.full(n, 5.0), 2 * x1 + 0.01 * rng.normal(size=n)])
    return X, ["x1", "x2", "x3", "const", "dup"]


def test_former_removes_constant_and_duplicate():
    X, names = make_data()
    y = X[:, 0] + X[:, 1]
    data = MVDFormer().form(X, y, names, 40, 10)
    assert data.removed["const"] == "сталий"
    # З пари дублікатів лишається показник, сильніше корельований з y.
    kept, dropped = ("x1", "dup") if "x1" in data.names else ("dup", "x1")
    assert data.removed[dropped] == f"дублює {kept}"
    assert sorted(data.names) == sorted([kept, "x2", "x3"])
    assert [len(data.part(p)[1]) for p in "ABC"] == [40, 10, 10]


def test_linear_asm_recovers_coefficients():
    X, names = make_data()
    y = 1.5 + 2 * X[:, 0] - 0.5 * X[:, 2]
    data = Dataset(X[:, :3], y, names[:3], 40, 10)
    model = LinearASM().synthesize(data)
    np.testing.assert_allclose(model.coef, [1.5, 2, 0, -0.5], atol=1e-9)


def test_multilayer_builds_product_of_three_indicators():
    X, names = make_data(n=90)
    y = X[:, 0] * X[:, 1] * X[:, 2]
    data = Dataset(X[:, :3], y, names[:3], 60, 15)
    model = MultilayerASM(freedom=4, forms="LBQ").synthesize(data)
    first = model.history[0]
    # Етап 4: до кожного сигналу шару 2 додаються показники, яких немає в його структурі.
    for key, missing in first["next"]["indicators"].items():
        assert set(missing).isdisjoint(model.nodes[key].indicators)
    assert model.layers >= 2
    assert model.indicators == ["x1", "x2", "x3"]
    tester = ModelTester()
    linear = tester.test(LinearASM(), LinearASM().synthesize(data), data).sx["C"]
    assert tester.test(MultilayerASM(), model, data).sx["C"] < 0.5 * linear


def test_multilayer_stops_when_layer_does_not_improve():
    X, names = make_data()
    y = 3 + X[:, 0] - X[:, 1]  # лінійна залежність відтворюється вже першим шаром
    data = Dataset(X[:, :3], y, names[:3], 40, 10)
    model = MultilayerASM(freedom=3, forms="L").synthesize(data)
    assert model.history[-1].get("stop")
    assert model.layers == len(model.history) - 1


def test_partial_bounds_limit_extrapolation():
    u = np.linspace(0, 1, 50)
    v = u ** 2
    y = 1 + u + v
    partial = fit_partial("Q", u, v, y, 50, bounds=(0.5, 3.5))
    assert partial.predict(np.array([10.0]), np.array([100.0]))[0] == 3.5


def test_perceptron_fits_smooth_nonlinearity():
    x = np.linspace(-2, 2, 80)
    y = np.tanh(1.5 * x)
    data = Dataset(x[:, None], y, ["x"], 60, 10)
    model = PerceptronASM(hidden=(2,), restarts=2).synthesize(data)
    assert np.max(np.abs(model.predict(x[:, None]) - y)) < 0.05


def test_controller_and_synthesizer_signal():
    X, names = make_data()
    y = X[:, 0] + X[:, 1]
    loose = ModelSynthesizer(sx_limit=1.0).run(X, y, names, 40, 10, try_all=False)
    assert loose["output"].asm == "МНК" and not loose["signals"]
    assert {stage for stage, _ in loose["log"]} == {"одношарові АСМ"}
    constructor = AlgorithmConstructor(freedoms=(3,), hidden=(2,))
    noisy = y + np.random.default_rng(0).normal(0, 0.1, len(y))
    strict = ModelSynthesizer(sx_limit=0.0, constructor=constructor).run(X, noisy, names, 40, 10)
    assert strict["output"] is None
    assert len(strict["signals"]) == 2
    result = TestResult("АСМ", "одношаровий", None, {"A": 0.1, "B": 0.1, "C": 0.2}, 3, 1)
    assert Controller(0.25).check(result) and not Controller(0.15).check(result)


def test_monitoring_data_is_reproducible():
    q1, plan1 = generate(3)
    q2, _ = generate(3)
    pd.testing.assert_frame_equal(q1, q2)
    assert q1["x5"].nunique() == 1
    assert split_sizes(q1["year"]) == (80, 24)
    assert len(plan1) == 32


def test_seasonal_models_on_annual_mean_are_balanced():
    q, plan = generate(3)
    s = Series.from_frames(q, plan)
    groups = synthesize_seasonal(s, 2012, 2018, structures=[("Y",)])
    values = {"Y": 0.7, "y_prev": 0.0, "y_lag": 0.0}
    # МНК з однаковим регресором Y_T для всіх кварталів: сума коефіцієнтів відтворює 4·Y_T.
    assert np.mean([g[0].predict(values) for g in groups]) == pytest.approx(0.7)


def test_forecast_feeds_fourth_quarter_back():
    years = np.arange(2000, 2004)
    y = np.tile([1.0, 2.0, 3.0, 4.0], (4, 1))
    s = Series(years, y, y.mean(axis=1), np.zeros(4))

    class Shift:
        """Модель, що повторює попередній квартал: перевіряє зв'язок y_{0,T} = y_{4,T−1}."""

        def __init__(self, quarter):
            self.quarter = quarter

        def predict(self, values):
            return values["y_prev"]

    f = forecast(s, 2001, 2, [Shift(t) for t in range(4)], None)
    np.testing.assert_allclose(f.quarterly, 4.0)
    assert f.years.tolist() == [2002, 2003]
