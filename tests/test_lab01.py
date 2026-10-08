"""Тести модулів лабораторної роботи № 1."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lab-01"))

from asm import rmse, solve_normal_equations, synthesize_gmdh, synthesize_ls
from mvd import (
    MVD,
    cluster_indicators,
    cluster_observations,
    distribution_law,
    remove_non_informative,
    sort_observations,
    stratified_order,
)


def make_linear(n=100, m=5, seed=1):
    rng = np.random.default_rng(seed)
    X = rng.normal(50, 10, (n, m))
    w = np.linspace(2.0, 0.5, m)
    return X, X @ w, w


def test_least_squares_recovers_weights_exactly():
    X, y, w = make_linear()
    res = synthesize_ls(X, y, 70)
    assert np.allclose(res.model.c, w, atol=1e-9)
    assert res.sx_check < 1e-9


def test_normal_equations_detect_constant_indicator():
    X, y, _ = make_linear()
    X = np.column_stack([X, np.full(len(y), 7.0)])
    F = np.column_stack([np.ones(len(y)), X])
    assert solve_normal_equations(F, y) is None
    assert synthesize_ls(X, y, 70) is None


def test_gmdh_model_form_matches_its_predictions():
    X, y, _ = make_linear(m=6)
    res = synthesize_gmdh(X, y, 70)
    # Розгорнутий вигляд моделі має відтворювати її прогноз на всьому МВД.
    assert res.sx_all == pytest.approx(rmse(y, res.model.predict(X)))
    assert res.layers >= 2


def test_gmdh_exact_for_two_indicators():
    X, y, w = make_linear(m=2)
    res = synthesize_gmdh(X, y, 70)
    assert res.layers == 1 and res.n_models == 1
    assert np.allclose(res.model.c, w, atol=1e-9)


def test_remove_constant_and_quasi_constant():
    rng = np.random.default_rng(0)
    X = np.column_stack([rng.normal(size=50), np.full(50, 3.0), 100 + rng.uniform(-0.01, 0.01, 50)])
    mvd = MVD(X, rng.normal(size=50), ["a", "b", "c"])
    clean, removed = remove_non_informative(mvd, cv_min=0.01)
    assert clean.names == ["a"]
    assert removed["b"] == "сталий"
    assert removed["c"].startswith("квазісталий")


def test_distribution_law_on_large_samples():
    rng = np.random.default_rng(3)
    assert distribution_law(rng.normal(50, 10, 400))["law"] == "нормальний"
    assert distribution_law(rng.uniform(10, 90, 400))["law"] == "рівномірний"


def test_sort_by_absolute_deviation():
    y = np.array([5.0, 1.0, 9.0, 4.0, 6.0])
    mvd = MVD(np.arange(10.0).reshape(5, 2), y, ["a", "b"])
    out = sort_observations(mvd, "abs_dev")
    dev = np.abs(out.y - y.mean())
    assert np.all(np.diff(dev) >= 0)
    assert out.index.tolist() == [1, 4, 5, 2, 3]  # |Δy| = 0, 1, 1, 4, 4


def test_kmeans_finds_separated_groups():
    rng = np.random.default_rng(5)
    centers = np.array([[0, 0], [10, 0], [0, 10]])
    X = np.vstack([c + rng.normal(0, 0.5, (30, 2)) for c in centers])
    res = cluster_observations(MVD(X, X.sum(axis=1), ["a", "b"]), seed=1)
    assert res["k"] == 3
    assert sorted(res["sizes"].tolist()) == [30, 30, 30]


def test_indicator_clusters_join_duplicates():
    rng = np.random.default_rng(7)
    a = rng.normal(size=200)
    b = rng.normal(size=200)
    X = np.column_stack([a, 2 * a + rng.normal(0, 0.05, 200), b])
    res = cluster_indicators(MVD(X, a + b, ["a", "a2", "b"]), r_min=0.9)
    members = [g["members"] for g in res["groups"]]
    assert [0, 1] in members and [2] in members


def test_stratified_order_keeps_every_cluster_in_both_samples():
    labels = np.repeat([0, 1, 2], [40, 30, 30])
    order, n_train = stratified_order(labels, 0.7)
    assert sorted(order.tolist()) == list(range(100))
    for c in range(3):
        assert np.any(labels[order[:n_train]] == c)
        assert np.any(labels[order[n_train:]] == c)
