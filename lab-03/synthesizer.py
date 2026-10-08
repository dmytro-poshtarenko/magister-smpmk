"""Синтезатор багатошарових моделей (лабораторна робота № 3).

Структура відповідає рисунку 3.1 методички:

* A1 ``MVDFormer`` — формує масив вхідних даних (МВД) з первинного опису: видаляє
  неінформативні та дубльовані показники функціями модуля первинної обробки з роботи № 1
  і ділить спостереження на навчальну A, перевірну B та екзаменаційну C вибірки;
* A2 ``TestScenario`` — сценарій випробування: які АСМ і в якій послідовності
  отримують МВД;
* A31…A3n — алгоритми синтезу моделей: ``LinearASM`` (МНК), ``PerceptronASM``
  (нейромережа з одним прихованим шаром), ``PolynomialASM`` (повний поліном
  Колмогорова-Габора 2-го степеня), ``MultilayerASM`` (багатошаровий синтез);
* A4 ``ModelTester`` — випробовує модель і обчислює її показник якості S_x;
* A5 ``Controller`` — порівнює S_x на екзаменаційній вибірці з граничним значенням;
* A6 ``AlgorithmConstructor`` — конструює АСМ з одношаровою і багатошаровою
  структурою; багатошарові АСМ подаються на випробування, якщо жодна одношарова
  модель не пройшла контролер;
* ``ModelSynthesizer`` — поєднує агрегати в один цикл синтезу.

Вибірка A служить для оцінювання коефіцієнтів, B — для вибору структури (кількість
нейронів перцептрона, відбір і зупинка в багатошаровому синтезі), C — лише для
випробування, тому S_x(C) є чесною оцінкою якості прогнозу.

``MultilayerASM`` реалізує етапи 1–4 структурування МВД із завдання роботи:
моделі шару синтезуються для пар вхідних сигналів, ранжуються за S_x на вибірці B,
а входами наступного шару стають виходи кращих моделей, показники МВД, що не ввійшли
до кращих моделей, і виходи моделей шару n − 1, не використані кращими моделями.
Синтез зупиняється, коли краща модель нового шару не краща за кращу модель
попереднього шару.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lab-01"))
from mvd import MVD, cluster_indicators, remove_non_informative  # noqa: E402  (первинна обробка з роботи № 1)


def rmse(y: np.ndarray, y_hat: np.ndarray) -> float:
    """Середнє квадратичне відхилення S_x = sqrt(sum((y − ŷ)²) / n)."""
    return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(y_hat)) ** 2)))


# --- A1. Формувач МВД --------------------------------------------------------------------------------


@dataclass
class Dataset:
    """Сформований МВД: показники, модельований показник і поділ на вибірки A, B, C."""

    X: np.ndarray
    y: np.ndarray
    names: list[str]
    n_train: int
    n_check: int
    removed: dict[str, str] = field(default_factory=dict)
    groups: list[dict] = field(default_factory=list)
    r_y: dict[str, float] = field(default_factory=dict)

    def _rows(self, part: str) -> slice:
        a, b = self.n_train, self.n_train + self.n_check
        return {"A": slice(0, a), "B": slice(a, b), "C": slice(b, None)}[part]

    def part(self, name: str) -> tuple[np.ndarray, np.ndarray]:
        rows = self._rows(name)
        return self.X[rows], self.y[rows]

    @property
    def A(self) -> tuple[np.ndarray, np.ndarray]:
        return self.part("A")

    @property
    def B(self) -> tuple[np.ndarray, np.ndarray]:
        return self.part("B")

    @property
    def C(self) -> tuple[np.ndarray, np.ndarray]:
        return self.part("C")


class MVDFormer:
    """A1: формування МВД з первинного опису.

    Неінформативні (сталі, квазісталі) і дубльовані (|r| ≥ r_min) показники
    визначаються лише за навчальною вибіркою A, щоб відбір не використовував
    дані, на яких модель потім випробовується.
    """

    def __init__(self, cv_min: float = 0.01, r_min: float = 0.9) -> None:
        self.cv_min = cv_min
        self.r_min = r_min

    def form(self, X: np.ndarray, y: np.ndarray, names: list[str], n_train: int, n_check: int) -> Dataset:
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        train = MVD(X[:n_train], y[:n_train], list(names))
        _, removed = remove_non_informative(train, self.cv_min)
        kept = [j for j, name in enumerate(names) if name not in removed]
        groups: list[dict] = []
        keep = kept
        if len(kept) > 1:
            ind = cluster_indicators(train.columns(kept), self.r_min)
            for g in ind["groups"]:
                members = [names[kept[j]] for j in g["members"]]
                rep = names[kept[g["representative"]]]
                groups.append({"members": members, "representative": rep})
                for name in members:
                    if name != rep:
                        removed[name] = f"дублює {rep}"
            keep = sorted(kept[g["representative"]] for g in ind["groups"])
        r_y = {names[j]: float(np.corrcoef(X[:n_train, j], y[:n_train])[0, 1]) for j in kept}
        return Dataset(X[:, keep], y, [names[j] for j in keep], n_train, n_check, removed, groups, r_y)


# --- A3. Алгоритми синтезу моделей ---------------------------------------------------------------------


class Model:
    """Синтезована модель: обчислює модельований показник за показниками МВД."""

    name: str = "модель"
    layers: int = 1

    @property
    def params(self) -> int:  # pragma: no cover - інтерфейс
        raise NotImplementedError

    def predict(self, X: np.ndarray) -> np.ndarray:  # pragma: no cover - інтерфейс
        raise NotImplementedError


class ASM:
    """Алгоритм синтезу моделей (агрегат A3x)."""

    name = "АСМ"
    kind = "одношаровий"

    def synthesize(self, data: Dataset) -> Model:  # pragma: no cover - інтерфейс
        raise NotImplementedError


def _design(X: np.ndarray) -> np.ndarray:
    return np.column_stack([np.ones(len(X)), X])


def poly2_features(X: np.ndarray) -> np.ndarray:
    """Члени повного полінома Колмогорова-Габора 2-го степеня (без вільного члена)."""
    m = X.shape[1]
    cols = [X[:, j] for j in range(m)]
    cols += [X[:, i] * X[:, j] for i in range(m) for j in range(i, m)]
    return np.column_stack(cols)


@dataclass
class LinearModel(Model):
    coef: np.ndarray
    name: str = "лінійна модель"

    @property
    def params(self) -> int:
        return len(self.coef)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return _design(X) @ self.coef


class LinearASM(ASM):
    """A31: лінійна модель за всіма показниками МВД, коефіцієнти — МНК."""

    name = "МНК"

    def synthesize(self, data: Dataset) -> Model:
        XA, yA = data.A
        coef = np.linalg.lstsq(_design(XA), yA, rcond=None)[0]
        return LinearModel(coef, "лінійна модель (МНК)")


@dataclass
class PolynomialModel(Model):
    coef: np.ndarray
    name: str = "поліном КГ 2-го степеня"

    @property
    def params(self) -> int:
        return len(self.coef)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return _design(poly2_features(X)) @ self.coef


class PolynomialASM(ASM):
    """A33: повний поліном Колмогорова-Габора 2-го степеня за всіма показниками МВД."""

    name = "поліном КГ"

    def synthesize(self, data: Dataset) -> Model:
        XA, yA = data.A
        coef = np.linalg.lstsq(_design(poly2_features(XA)), yA, rcond=None)[0]
        return PolynomialModel(coef, "поліном Колмогорова-Габора 2-го степеня")


@dataclass
class PerceptronModel(Model):
    """Нейромережа: входи → прихований шар з активацією tanh → лінійний вихід."""

    W: np.ndarray
    b: np.ndarray
    v: np.ndarray
    c: float
    x_mean: np.ndarray
    x_std: np.ndarray
    y_mean: float
    y_std: float
    name: str = "перцептрон"

    @property
    def params(self) -> int:
        return self.W.size + self.b.size + self.v.size + 1

    def predict(self, X: np.ndarray) -> np.ndarray:
        Z = (X - self.x_mean) / self.x_std
        H = np.tanh(Z @ self.W + self.b)
        return (H @ self.v + self.c) * self.y_std + self.y_mean


def fit_perceptron(XA: np.ndarray, yA: np.ndarray, h: int, l2: float = 1e-3, restarts: int = 6,
                   seed: int = 0) -> PerceptronModel:
    """Навчає перцептрон m–h–1 на вибірці A: мінімум СКВ з L2-регуляризацією методом L-BFGS
    з кількох випадкових початкових точок (seed фіксований, результат відтворюваний)."""
    xm, xs = XA.mean(axis=0), XA.std(axis=0)
    xs = np.where(xs > 0, xs, 1.0)
    ym, ys = float(yA.mean()), float(yA.std())
    Z, t = (XA - xm) / xs, (yA - ym) / ys
    n, m = Z.shape

    def unpack(p):
        return p[: m * h].reshape(m, h), p[m * h: m * h + h], p[m * h + h: m * h + 2 * h], p[-1]

    def loss(p):
        W, b, v, c = unpack(p)
        H = np.tanh(Z @ W + b)
        r = H @ v + c - t
        gH = np.outer(r, v) * (1 - H ** 2) * (2 / n)
        grad = np.concatenate([
            (Z.T @ gH + 2 * l2 * W).ravel(),
            gH.sum(axis=0),
            H.T @ r * (2 / n) + 2 * l2 * v,
            [r.sum() * (2 / n)],
        ])
        return float(np.mean(r ** 2) + l2 * (np.sum(W ** 2) + np.sum(v ** 2))), grad

    rng = np.random.default_rng(seed)
    best = None
    for _ in range(restarts):
        res = minimize(loss, rng.normal(0, 0.5, m * h + 2 * h + 1), jac=True, method="L-BFGS-B",
                       options={"maxiter": 3000})
        if best is None or res.fun < best.fun:
            best = res
    W, b, v, c = unpack(best.x)
    return PerceptronModel(W, b, v, float(c), xm, xs, ym, ys, f"перцептрон {m}–{h}–1")


class PerceptronASM(ASM):
    """A32: перцептрон з одним прихованим шаром (класична нейромережа).

    Кількість нейронів прихованого шару вибирається за S_x на вибірці B.
    """

    name = "перцептрон"

    def __init__(self, hidden: tuple[int, ...] = (2, 3, 4, 6), l2: float = 1e-3, restarts: int = 6, seed: int = 0) -> None:
        self.hidden = hidden
        self.l2 = l2
        self.restarts = restarts
        self.seed = seed

    def synthesize(self, data: Dataset) -> Model:
        XA, yA = data.A
        XB, yB = data.B
        models = [fit_perceptron(XA, yA, h, self.l2, self.restarts, self.seed) for h in self.hidden]
        return min(models, key=lambda mdl: rmse(yB, mdl.predict(XB)))


# Часткові описи (нейрони багатошарової моделі) різної структури для пари сигналів u, v.
PARTIAL_FORMS = {
    "L": "a0 + a1·u + a2·v",
    "B": "a0 + a1·u + a2·v + a3·u·v",
    "Q": "a0 + a1·u + a2·v + a3·u·v + a4·u² + a5·v²",
    "N": "c + v1·tanh(w11·u + w12·v + b1) + v2·tanh(w21·u + w22·v + b2)",
}


def partial_design(kind: str, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    cols = [np.ones_like(u), u, v]
    if kind in ("B", "Q"):
        cols.append(u * v)
    if kind == "Q":
        cols += [u ** 2, v ** 2]
    return np.column_stack(cols)


@dataclass
class Partial:
    """Частковий опис — нейрон багатошарової моделі з двома входами.

    Поліноміальні нейрони (L, B, Q) оцінюються МНК, нейрон N — перцептрон 2–2–1.
    bounds обмежує вихід нейрона діапазоном модельованого показника на вибірці A,
    розширеним на задану частку: композиція поліномів інакше «вибухає» при екстраполяції.
    """

    kind: str
    coef: np.ndarray | None = None
    net: PerceptronModel | None = None
    bounds: tuple[float, float] | None = None

    @property
    def params(self) -> int:
        return self.net.params if self.net is not None else len(self.coef)

    def predict(self, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        if self.net is not None:
            z = self.net.predict(np.column_stack([u, v]))
        else:
            z = partial_design(self.kind, u, v) @ self.coef
        return np.clip(z, *self.bounds) if self.bounds else z


def fit_partial(kind: str, u: np.ndarray, v: np.ndarray, y: np.ndarray, a: int,
                bounds: tuple[float, float] | None, seed: int = 0) -> Partial:
    if kind == "N":
        net = fit_perceptron(np.column_stack([u[:a], v[:a]]), y[:a], 2, restarts=3, seed=seed)
        return Partial(kind, net=net, bounds=bounds)
    coef = np.linalg.lstsq(partial_design(kind, u[:a], v[:a]), y[:a], rcond=None)[0]
    return Partial(kind, coef=coef, bounds=bounds)


@dataclass
class Node:
    """Нейрон багатошарової моделі: частковий опис двох вхідних сигналів."""

    key: str
    layer: int
    inputs: tuple[str, ...] = ()
    partial: Partial | None = None
    indicators: frozenset[str] = frozenset()  # показники МВД у структурі нейрона

    @property
    def kind(self) -> str:
        return self.partial.kind if self.partial else ""


@dataclass
class MultilayerModel(Model):
    """Багатошарова модель: мережа часткових описів, обчислювана від входів до виходу."""

    nodes: dict[str, Node]
    output: str
    names: list[str]
    history: list[dict]
    name: str = "багатошарова модель"

    def used(self) -> list[Node]:
        """Нейрони, що входять до структури моделі, впорядковані за шарами."""
        seen, stack = {}, [self.output]
        while stack:
            key = stack.pop()
            if key in seen:
                continue
            seen[key] = self.nodes[key]
            stack.extend(self.nodes[key].inputs)
        return sorted((n for n in seen.values() if n.layer > 0), key=lambda n: (n.layer, n.key))

    @property
    def params(self) -> int:
        return sum(n.partial.params for n in self.used())

    @property
    def layers(self) -> int:
        return self.nodes[self.output].layer

    @property
    def indicators(self) -> list[str]:
        return [name for name in self.names if name in self.nodes[self.output].indicators]

    def predict(self, X: np.ndarray) -> np.ndarray:
        cache = {name: X[:, j] for j, name in enumerate(self.names)}

        def value(key: str) -> np.ndarray:
            if key not in cache:
                node = self.nodes[key]
                cache[key] = node.partial.predict(value(node.inputs[0]), value(node.inputs[1]))
            return cache[key]

        return value(self.output)


class MultilayerASM(ASM):
    """Багатошаровий синтез моделей (етапи 1–4 завдання роботи).

    freedom — скільки кращих моделей шару передається на наступний шар (свобода
    вибору F); min_gain — мінімальне відносне зменшення S_x(B) кращої моделі шару,
    за якого синтез продовжується (критерій зупинки); forms — структури нейронів
    (див. PARTIAL_FORMS); margin — розширення діапазону виходу нейрона (None — без обмеження).
    """

    name = "багатошаровий синтез"
    kind = "багатошаровий"

    def __init__(self, freedom: int = 6, max_layers: int = 8, min_gain: float = 0.01, forms: str = "LBQN",
                 margin: float | None = 0.2) -> None:
        self.freedom = freedom
        self.max_layers = max_layers
        self.min_gain = min_gain
        self.forms = forms
        self.margin = margin

    def synthesize(self, data: Dataset) -> Model:
        a, b = data.n_train, data.n_train + data.n_check
        y = data.y
        bounds = None
        if self.margin is not None:
            lo, hi = float(y[:a].min()), float(y[:a].max())
            bounds = (lo - self.margin * (hi - lo), hi + self.margin * (hi - lo))
        values = {name: data.X[:, j] for j, name in enumerate(data.names)}
        nodes = {name: Node(name, 0, indicators=frozenset([name])) for name in data.names}
        pairs = list(combinations(data.names, 2))
        history: list[dict] = []
        best_prev: tuple[float, str] | None = None
        prev_selected: list[str] = []

        for layer in range(1, self.max_layers + 1):
            # Етап 1: синтез моделей різної структури для всіх пар вхідних сигналів.
            cands = []
            for u, v in pairs:
                for kind in self.forms:
                    partial = fit_partial(kind, values[u], values[v], y, a, bounds)
                    z = partial.predict(values[u], values[v])
                    cands.append((rmse(y[a:b], z[a:b]), u, v, partial, z))
            if not cands:
                break
            # Етапи 2–3: ранжування за S_x на вибірці B і критерій зупинки.
            cands.sort(key=lambda c: c[0])
            sx_b, z_best = cands[0][0], cands[0][4]
            record = {
                "layer": layer, "pairs": len(pairs), "models": len(cands),
                "sx_A": rmse(y[:a], z_best[:a]), "sx_B": sx_b, "sx_C": rmse(y[b:], z_best[b:]),
                "best": f"{cands[0][3].kind}({cands[0][1]}, {cands[0][2]})",
            }
            history.append(record)
            if best_prev is not None and sx_b > best_prev[0] * (1 - self.min_gain):
                record["stop"] = True
                break
            selected = []
            for k, (_, u, v, partial, z) in enumerate(cands[: self.freedom], start=1):
                key = f"z{layer}.{k}"
                nodes[key] = Node(key, layer, (u, v), partial, nodes[u].indicators | nodes[v].indicators)
                values[key] = z
                selected.append(key)
            best_prev = (sx_b, selected[0])
            used_signals = {s for key in selected for s in nodes[key].inputs}
            # Етап 4: виходи моделей шару n − 1, не використані кращими моделями шару n.
            free_previous = [key for key in prev_selected if key not in used_signals]
            signals = selected + free_previous
            # Етапи 2 і 4: входи наступного шару — виходи кращих моделей, невикористані виходи
            # шару n − 1 і для кожного з них показники МВД, що не ввійшли до його структури.
            missing = {s: [x for x in data.names if x not in nodes[s].indicators] for s in signals}
            pairs = list(combinations(signals, 2)) + [(s, x) for s in signals for x in missing[s]]
            record["next"] = {"best": selected, "previous": free_previous, "indicators": missing}
            prev_selected = selected

        if best_prev is None:
            raise ValueError("МВД має містити щонайменше два показники")
        return MultilayerModel(nodes, best_prev[1], list(data.names), history, f"багатошарова модель (F = {self.freedom})")


# --- A4, A5, A6 і сценарій ---------------------------------------------------------------------------


@dataclass
class TestResult:
    __test__ = False  # не тестовий клас pytest

    asm: str
    kind: str
    model: Model
    sx: dict[str, float]
    params: int
    layers: int
    accepted: bool = False


class ModelTester:
    """A4: випробовування моделі — СКВ на вибірках A, B і C."""

    def test(self, asm: ASM, model: Model, data: Dataset) -> TestResult:
        sx = {}
        for part in "ABC":
            X, y = data.part(part)
            sx[part] = rmse(y, model.predict(X)) if len(y) else float("nan")
        return TestResult(asm.name, asm.kind, model, sx, model.params, model.layers)


class Controller:
    """A5: пропускає модель на вихід, якщо S_x на екзаменаційній вибірці C не перевищує граничного значення."""

    def __init__(self, sx_limit: float) -> None:
        self.sx_limit = sx_limit

    def check(self, result: TestResult) -> bool:
        result.accepted = result.sx["C"] <= self.sx_limit
        return result.accepted


class AlgorithmConstructor:
    """A6: конструює АСМ з одношаровою і багатошаровою структурою."""

    def __init__(self, freedoms: tuple[int, ...] = (4, 6, 8), hidden: tuple[int, ...] = (2, 3, 4, 6)) -> None:
        self.freedoms = freedoms
        self.hidden = hidden

    def single_layer(self) -> list[ASM]:
        return [LinearASM(), PerceptronASM(self.hidden), PolynomialASM()]

    def multilayer(self) -> list[ASM]:
        algorithms = []
        for F in self.freedoms:
            asm = MultilayerASM(freedom=F)
            asm.name = f"багатошаровий, F = {F}"
            algorithms.append(asm)
        return algorithms


class TestScenario:
    """A2: сценарій випробування — послідовність етапів і АСМ, на вхід яких подається МВД."""

    __test__ = False  # не тестовий клас pytest

    def __init__(self, constructor: AlgorithmConstructor) -> None:
        self.constructor = constructor

    def stages(self):
        yield "одношарові АСМ", self.constructor.single_layer()
        yield "багатошарові АСМ", self.constructor.multilayer()


class ModelSynthesizer:
    """Синтезатор моделей: A1 → A2 → A3x → A4 → A5, за сигналом A5 — A6.

    Виходом синтезатора є перша в порядку сценарію модель, що пройшла контролер.
    Якщо на етапі одношарових АСМ таких немає, контролер подає сигнал конструктору,
    і випробовуються багатошарові АСМ. З try_all=True випробовуються всі АСМ
    сценарію (для порівняння), але вихід визначається так само.
    """

    def __init__(self, sx_limit: float, former: MVDFormer | None = None,
                 constructor: AlgorithmConstructor | None = None) -> None:
        self.former = former or MVDFormer()
        self.scenario = TestScenario(constructor or AlgorithmConstructor())
        self.tester = ModelTester()
        self.controller = Controller(sx_limit)

    def run(self, X: np.ndarray, y: np.ndarray, names: list[str], n_train: int, n_check: int,
            try_all: bool = True) -> dict:
        data = self.former.form(X, y, names, n_train, n_check)
        log: list[tuple[str, TestResult]] = []
        output: TestResult | None = None
        signals: list[str] = []
        for stage, algorithms in self.scenario.stages():
            if output is not None and not try_all:
                break
            for asm in algorithms:
                result = self.tester.test(asm, asm.synthesize(data), data)
                self.controller.check(result)
                log.append((stage, result))
                if result.accepted and output is None:
                    output = result
            if output is None:
                signals.append(f"{stage}: жодна модель не пройшла контролер — сигнал конструктору A6")
        return {"data": data, "log": log, "output": output, "signals": signals}
