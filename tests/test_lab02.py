"""Тести алгоритму синтезу прогнозних моделей (лабораторна робота № 2)."""

import io
import sys
from pathlib import Path

import numpy as np
import pytest

LAB = Path(__file__).resolve().parents[1] / "lab-02"
sys.path.insert(0, str(LAB))

from cli import SynthesisCLI
from kg_asm import (
    DecisionSupport,
    GaussSystem,
    ModelSynthesizer,
    ModelTester,
    Preferences,
    ReferenceModel,
    Term,
    TimeSeries,
    ahp_weights,
    gauss_elimination,
    parse_terms,
    term,
)

EXAMPLE = TimeSeries([10, 15, 13, 19, 14, 18, 17, 11])
EXAMPLE_REF = ReferenceModel("методичка", [Term(), term(2), term(1), term(1, 2)])


def logistic(n=40, r=3.6):
    u = [3.0]
    for _ in range(n - 1):
        u.append(r * u[-1] * (1 - u[-1] / 10))
    return TimeSeries(10 + np.array(u))


def test_gauss_elimination_matches_numpy():
    rng = np.random.default_rng(0)
    M = rng.normal(size=(5, 5)) + 5 * np.eye(5)
    v = rng.normal(size=5)
    assert np.allclose(gauss_elimination(M, v), np.linalg.solve(M, v))


def test_gauss_elimination_detects_singular_system():
    with pytest.raises(np.linalg.LinAlgError):
        gauss_elimination(np.array([[1.0, 2.0], [2.0, 4.0]]), np.array([1.0, 2.0]))


def test_example_normal_equations_and_coefficients():
    system = GaussSystem.build(EXAMPLE, EXAMPLE_REF)
    assert system.equations == 6
    G, g = system.normal()
    assert g.tolist() == [92, 1375, 1453, 21551]
    assert G[3, 3] == 349430
    assert np.allclose(system.solve(), [-73.864, 6.831, 5.744, -0.441], atol=1e-3)


def test_noise_free_logistic_series_is_recovered_exactly():
    series = logistic()
    ref = ReferenceModel("M4", [Term(), term(1), term(1, 1)])
    model = ModelSynthesizer().synthesize(series, ref)
    r = 3.6
    assert np.allclose(model.coefficients, [10 - 20 * r, 3 * r, -0.1 * r], atol=1e-6)
    assert ModelTester.sx(model, series) < 1e-8
    horizon, _ = ModelTester(0.05).horizon(series, ref, max_steps=6)
    assert horizon == 6


def test_horizon_is_limited_by_data():
    tester = ModelTester(0.5)
    horizon, steps = tester.horizon(EXAMPLE, EXAMPLE_REF, max_steps=10)
    # Для 4 коефіцієнтів потрібно щонайменше 4 умовні рівняння, тобто 6 точок.
    assert all(s.train_points >= 6 for s in steps)
    assert horizon <= 2


def test_kolmogorov_gabor_term_counts():
    assert ReferenceModel.kolmogorov_gabor(2, 2).size == 6
    assert ReferenceModel.kolmogorov_gabor(3, 2).size == 10
    assert ReferenceModel.kolmogorov_gabor(2, 3).size == 10


def test_parse_terms():
    ref = parse_terms("1; F1; F1^2*F2")
    assert [t.label() for t in ref.terms] == ["1", "F1", "F1²·F2"]
    assert ref.lags == 2


def test_ahp_weights_for_consistent_matrix():
    res = ahp_weights(np.array([[1, 2, 4], [1 / 2, 1, 2], [1 / 4, 1 / 2, 1]]))
    assert np.allclose(res["weights"], [4 / 7, 2 / 7, 1 / 7])
    assert res["CR"] == pytest.approx(0, abs=1e-9)


def test_decision_support_puts_useful_models_first():
    series = logistic(50, 3.7)
    prefs = Preferences("тест", np.array([[1, 1 / 3, 3], [3, 1, 5], [1 / 3, 1 / 5, 1]]), sx_max=0.5, horizon_min=2)
    ds = DecisionSupport(prefs, max_steps=6)
    cands = [ReferenceModel("лінійна", [Term(), term(1)]), ReferenceModel("квадратична", [Term(), term(1), term(1, 1)])]
    ranked = ds.rank(ds.evaluate(series, cands))
    assert ranked[0]["name"] == "квадратична" and ranked[0]["useful"]
    assert not ranked[1]["useful"]


def test_cli_select_and_inconsistent_judgements(tmp_path):
    csv = tmp_path / "f.csv"
    series = logistic(45, 3.7)
    csv.write_text("T,F\n" + "\n".join(f"{i},{v}" for i, v in enumerate(series.values, 1)))
    out = io.StringIO()
    assert SynthesisCLI(out).run([str(csv), "--select", "--max-steps", "5"]) == 0
    assert "1. " in out.getvalue()
    out = io.StringIO()
    assert SynthesisCLI(out).run([str(csv), "--select", "--pairwise", "1,5,1/5; 1/5,1,3; 5,1/3,1"]) == 2
