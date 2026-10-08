"""Модуль первинної обробки масиву вхідних даних (МВД).

Завдання до самостійного виконання лабораторної роботи № 1: модуль виконує
первинну обробку МВД шляхом

* визначення закону розподілу кожного показника (критерій згоди Пірсона χ²
  для нормального і рівномірного законів);
* видалення неінформативних показників, значення яких не змінюються
  (сталих, а за потреби і квазісталих з малим коефіцієнтом варіації);
* сортування спостережень (за модулем відхилення модельованого показника від
  середнього, за самим відхиленням або за значенням y);
* кластеризації спостережень (алгоритм k-середніх, кількість кластерів
  обирається за коефіцієнтом силуету);
* кластеризації показників (ієрархічна кластеризація за відстанню 1 − |r|,
  у кожному кластері лишається показник, найтісніше пов'язаний з y).

Запуск з командного рядка:

    python mvd.py data/individual_mvd.csv --target y --cv-min 0.01 --r-min 0.9
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy import stats
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform


@dataclass
class MVD:
    """Масив вхідних даних: показники X, модельований показник y, номери спостережень."""

    X: np.ndarray
    y: np.ndarray
    names: list[str]
    index: np.ndarray = field(default=None)  # номери спостережень у первинному описі (з 1)

    def __post_init__(self) -> None:
        self.X = np.asarray(self.X, dtype=float)
        self.y = np.asarray(self.y, dtype=float)
        if self.index is None:
            self.index = np.arange(1, len(self.y) + 1)

    @property
    def shape(self) -> tuple[int, int]:
        return self.X.shape

    def columns(self, keep: list[int]) -> MVD:
        return MVD(self.X[:, keep], self.y, [self.names[j] for j in keep], self.index)

    def rows(self, order: np.ndarray) -> MVD:
        return MVD(self.X[order], self.y[order], list(self.names), self.index[order])

    @classmethod
    def from_csv(cls, path: str | Path, target: str = "y") -> MVD:
        data = np.genfromtxt(path, delimiter=",", names=True, encoding="utf-8")
        cols = list(data.dtype.names)
        names = [c for c in cols if c not in (target, "obs")]
        X = np.column_stack([data[c] for c in names])
        index = data["obs"].astype(int) if "obs" in cols else None
        return cls(X, data[target], names, index)

    def to_csv(self, path: str | Path, target: str = "y", digits: int = 4) -> None:
        header = ",".join(["obs", *self.names, target])
        table = np.column_stack([self.index, self.X, self.y])
        fmt = ["%d"] + [f"%.{digits}f"] * (table.shape[1] - 1)
        np.savetxt(path, table, delimiter=",", header=header, comments="", fmt=fmt, encoding="utf-8")


# --- Закон розподілу -----------------------------------------------------------------


def log_likelihoods(x: np.ndarray) -> dict[str, float]:
    """Логарифми функції правдоподібності нормального і рівномірного законів.

    Параметри обох законів оцінено методом максимальної правдоподібності; кожен
    закон має два параметри, тому порівняння правдоподібностей рівносильне
    порівнянню за інформаційним критерієм Акаїке.
    """
    n = len(x)
    sigma2 = x.var()  # оцінка максимальної правдоподібності (ділення на n)
    return {
        "нормальний": float(-n / 2 * np.log(2 * np.pi * sigma2) - n / 2),
        "рівномірний": float(-n * np.log(np.ptp(x))),
    }


def uniform_ks_pvalue(x: np.ndarray) -> float:
    """Критерій Колмогорова — Смирнова для рівномірного закону з межами, оціненими за вибіркою."""
    n = len(x)
    r = np.ptp(x)
    a, b = x.min() - r / (n - 1), x.max() + r / (n - 1)  # незміщені оцінки меж
    return float(stats.kstest(x, "uniform", args=(a, b - a)).pvalue)


def distribution_law(x: np.ndarray, alpha: float = 0.05) -> dict:
    """Визначає закон розподілу показника (нормальний чи рівномірний).

    Закон обирається за максимумом правдоподібності, а згоду вибраного закону з
    даними перевіряє критерій Шапіро — Уїлка (нормальний закон) або критерій
    Колмогорова — Смирнова (рівномірний закон) на рівні значущості alpha.
    """
    ll = log_likelihoods(x)
    p_normal = float(stats.shapiro(x).pvalue)
    p_uniform = uniform_ks_pvalue(x)
    law = max(ll, key=ll.get)
    p_law = p_normal if law == "нормальний" else p_uniform
    return {
        "mean": float(x.mean()),
        "std": float(x.std(ddof=1)),
        "skew": float(stats.skew(x)),
        "kurtosis": float(stats.kurtosis(x)),
        "loglik_normal": ll["нормальний"],
        "loglik_uniform": ll["рівномірний"],
        "p_normal": p_normal,
        "p_uniform": p_uniform,
        "law": law,
        "confirmed": bool(p_law >= alpha),
    }


# --- Неінформативні показники ----------------------------------------------------------


def find_non_informative(X: np.ndarray, cv_min: float = 0.0) -> dict[int, str]:
    """Повертає {індекс показника: причина} для показників, що не несуть інформації.

    Сталий показник має однакові значення в усіх спостереженнях. Квазісталим
    вважається показник із коефіцієнтом варіації std / |mean| < cv_min.
    """
    found: dict[int, str] = {}
    for j in range(X.shape[1]):
        col = X[:, j]
        if np.ptp(col) == 0:
            found[j] = "сталий"
            continue
        mean = abs(col.mean())
        if cv_min > 0 and mean > 0 and col.std(ddof=1) / mean < cv_min:
            found[j] = f"квазісталий (V = {col.std(ddof=1) / mean:.2e})"
    return found


def remove_non_informative(mvd: MVD, cv_min: float = 0.0) -> tuple[MVD, dict[str, str]]:
    found = find_non_informative(mvd.X, cv_min)
    keep = [j for j in range(mvd.shape[1]) if j not in found]
    return mvd.columns(keep), {mvd.names[j]: reason for j, reason in found.items()}


# --- Сортування спостережень ----------------------------------------------------------------


def sort_observations(mvd: MVD, key: str = "abs_dev") -> MVD:
    """Сортує спостереження за зростанням ключа.

    abs_dev — модуль відхилення |y − ȳ| (дослід 2), dev — відхилення y − ȳ без
    модуля (так записано формулу в методичці), y — значення модельованого показника.
    """
    dev = mvd.y - mvd.y.mean()
    keys = {"abs_dev": np.abs(dev), "dev": dev, "y": mvd.y}
    return mvd.rows(np.argsort(keys[key], kind="stable"))


# --- Кластеризація спостережень -----------------------------------------------------------


def standardize(X: np.ndarray) -> np.ndarray:
    return (X - X.mean(axis=0)) / X.std(axis=0)


def kmeans(Z: np.ndarray, k: int, n_init: int = 20, max_iter: int = 300, seed: int = 0):
    """Алгоритм k-середніх (Ллойда) з початковими центрами k-means++."""
    rng = np.random.default_rng(seed)
    best = None
    for _ in range(n_init):
        centers = [Z[rng.integers(len(Z))]]
        for _ in range(1, k):
            d2 = np.min(((Z[:, None, :] - np.array(centers)[None]) ** 2).sum(-1), axis=1)
            centers.append(Z[rng.choice(len(Z), p=d2 / d2.sum())])
        centers = np.array(centers)
        for _ in range(max_iter):
            labels = np.argmin(((Z[:, None, :] - centers[None]) ** 2).sum(-1), axis=1)
            new = np.array([Z[labels == c].mean(axis=0) if np.any(labels == c) else centers[c] for c in range(k)])
            if np.allclose(new, centers):
                break
            centers = new
        inertia = float(((Z - centers[labels]) ** 2).sum())
        if best is None or inertia < best[2]:
            best = (labels, centers, inertia)
    labels, centers, inertia = best
    # Нумеруємо кластери за порядком першої появи, щоб результат не залежав від випадку.
    order = list(dict.fromkeys(labels.tolist()))
    remap = {old: new for new, old in enumerate(order)}
    return np.array([remap[c] for c in labels]), centers[order], inertia


def silhouette(Z: np.ndarray, labels: np.ndarray) -> float:
    """Середній коефіцієнт силуету s = (b − a) / max(a, b)."""
    D = np.sqrt(((Z[:, None, :] - Z[None]) ** 2).sum(-1))
    s = np.zeros(len(Z))
    for i in range(len(Z)):
        own = labels == labels[i]
        if own.sum() == 1:
            continue
        a = D[i, own].sum() / (own.sum() - 1)
        b = min(D[i, labels == c].mean() for c in set(labels.tolist()) if c != labels[i])
        s[i] = (b - a) / max(a, b)
    return float(s.mean())


def cluster_observations(mvd: MVD, k_values=range(2, 7), seed: int = 0) -> dict:
    """Кластеризує спостереження і обирає кількість кластерів за силуетом."""
    Z = standardize(mvd.X)
    scores = {}
    runs = {}
    for k in k_values:
        labels, _, inertia = kmeans(Z, k, seed=seed)
        scores[k] = silhouette(Z, labels)
        runs[k] = (labels, inertia)
    k_best = max(scores, key=scores.get)
    labels = runs[k_best][0]
    centers = np.array([mvd.X[labels == c].mean(axis=0) for c in range(k_best)])
    sizes = np.bincount(labels, minlength=k_best)
    y_means = np.array([mvd.y[labels == c].mean() for c in range(k_best)])
    return {
        "k": k_best,
        "silhouette": scores,
        "inertia": {k: runs[k][1] for k in runs},
        "labels": labels,
        "centers": centers,
        "sizes": sizes,
        "y_means": y_means,
    }


def stratified_order(labels: np.ndarray, train_share: float = 0.7) -> tuple[np.ndarray, int]:
    """Порядок спостережень, за якого навчальна вибірка A містить частку кожного кластера.

    Повертає (order, n_train): перші n_train спостережень утворюють вибірку A,
    решта — перевірну вибірку B. Усередині кластера спостереження чергуються,
    тож у B потрапляє кожне третє–четверте спостереження кластера.
    """
    train, check = [], []
    for c in np.unique(labels):
        members = np.flatnonzero(labels == c)
        n_check = round(len(members) * (1 - train_share))
        picks = set(np.linspace(0, len(members) - 1, n_check + 2)[1:-1].round().astype(int).tolist()) if n_check else set()
        for pos, i in enumerate(members):
            (check if pos in picks else train).append(i)
    return np.array(train + check), len(train)


# --- Кластеризація показників -------------------------------------------------------------


def cluster_indicators(mvd: MVD, r_min: float = 0.9) -> dict:
    """Об'єднує в кластери показники, кореляція між якими за модулем не менша r_min.

    Відстань між показниками d = 1 − |r|, зв'язок — середній (average linkage).
    Представником кластера стає показник із найбільшим |r| з модельованим показником y.
    """
    R = np.corrcoef(mvd.X, rowvar=False)
    D = 1 - np.abs(R)
    np.fill_diagonal(D, 0)
    Z = linkage(squareform(D, checks=False), method="average")
    labels = fcluster(Z, t=1 - r_min, criterion="distance")
    r_y = np.array([np.corrcoef(mvd.X[:, j], mvd.y)[0, 1] for j in range(mvd.shape[1])])
    groups = []
    for c in np.unique(labels):
        members = np.flatnonzero(labels == c).tolist()
        rep = max(members, key=lambda j: abs(r_y[j]))
        groups.append({"members": members, "representative": rep})
    groups.sort(key=lambda g: g["members"][0])
    return {"linkage": Z, "labels": labels, "groups": groups, "r_y": r_y, "R": R}


# --- Повний цикл первинної обробки ----------------------------------------------------------


def preprocess(
    mvd: MVD,
    cv_min: float = 0.01,
    r_min: float = 0.9,
    sort_key: str | None = "abs_dev",
    reduce_indicators: bool = True,
    seed: int = 0,
) -> tuple[MVD, dict]:
    """Виконує всі кроки первинної обробки і повертає оброблений МВД та звіт."""
    report: dict = {"input_shape": mvd.shape}
    report["laws"] = {name: distribution_law(mvd.X[:, j]) for j, name in enumerate(mvd.names) if np.ptp(mvd.X[:, j]) > 0}
    clean, removed = remove_non_informative(mvd, cv_min)
    report["removed"] = removed
    ind = cluster_indicators(clean, r_min)
    report["indicator_groups"] = [
        {"members": [clean.names[j] for j in g["members"]], "representative": clean.names[g["representative"]]}
        for g in ind["groups"]
    ]
    if reduce_indicators:
        clean = clean.columns(sorted(g["representative"] for g in ind["groups"]))
    obs = cluster_observations(clean, seed=seed)
    report["observation_clusters"] = {
        "k": obs["k"],
        "silhouette": {int(k): round(v, 4) for k, v in obs["silhouette"].items()},
        "sizes": obs["sizes"].tolist(),
        "y_means": obs["y_means"].round(3).tolist(),
    }
    if sort_key:
        clean = sort_observations(clean, sort_key)
    report["output_shape"] = clean.shape
    report["output_names"] = clean.names
    return clean, report


def main() -> None:
    parser = argparse.ArgumentParser(description="Первинна обробка масиву вхідних даних")
    parser.add_argument("csv", help="CSV зі стовпцями показників і модельованого показника")
    parser.add_argument("--target", default="y", help="назва стовпця модельованого показника")
    parser.add_argument("--cv-min", type=float, default=0.01, help="поріг коефіцієнта варіації квазісталих показників")
    parser.add_argument("--r-min", type=float, default=0.9, help="поріг |r| для об'єднання показників у кластер")
    parser.add_argument("--sort", default="abs_dev", choices=["abs_dev", "dev", "y", "none"])
    parser.add_argument("--out", help="куди зберегти оброблений МВД (CSV)")
    args = parser.parse_args()

    mvd = MVD.from_csv(args.csv, args.target)
    result, report = preprocess(mvd, args.cv_min, args.r_min, None if args.sort == "none" else args.sort)
    if args.out:
        result.to_csv(args.out, args.target)
    print(json.dumps(report, ensure_ascii=False, indent=2, default=float))


if __name__ == "__main__":
    main()
