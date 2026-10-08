"""Граничний клас SynthesisCLI: інтерфейс командного рядка синтезатора прогнозних моделей.

Синтез заданої опорної моделі:

    python cli.py data/individual_series.csv --model "1; F1; F1^2"

Вибір моделі за перевагами ОПР (матриця парних порівнянь критеріїв «точність»,
«горизонт», «простота» за шкалою Сааті, рядки через «;»):

    python cli.py data/individual_series.csv --select --pairwise "1,1/3,3; 3,1,5; 1/3,1/5,1" --sx-max 0.5 --h-min 2
"""

from __future__ import annotations

import argparse
import sys
from fractions import Fraction

import numpy as np
from kg_asm import (
    DecisionSupport,
    ModelSynthesizer,
    ModelTester,
    Preferences,
    TimeSeries,
    parse_terms,
    standard_candidates,
)


class SynthesisCLI:
    """Приймає запит користувача, викликає керуючі класи і показує результат."""

    def __init__(self, out=sys.stdout) -> None:
        self.out = out

    def run(self, argv: list[str] | None = None) -> int:
        args = self._parser().parse_args(argv)
        series = self.load_series(args.csv, args.column)
        if args.select:
            return self._select(series, args)
        return self._synthesize(series, args)

    @staticmethod
    def load_series(path: str, column: str = "F") -> TimeSeries:
        data = np.genfromtxt(path, delimiter=",", names=True, encoding="utf-8")
        values = data[column]
        return TimeSeries(values[~np.isnan(values)], column)

    @staticmethod
    def parse_pairwise(spec: str) -> np.ndarray:
        rows = [r for r in spec.replace(" ", "").split(";") if r]
        return np.array([[float(Fraction(x)) for x in r.split(",")] for r in rows])

    def _synthesize(self, series: TimeSeries, args) -> int:
        reference = parse_terms(args.model)
        model = ModelSynthesizer().synthesize(series, reference)
        tester = ModelTester(args.delta)
        horizon, _ = tester.horizon(series, reference, args.max_steps)
        self.show(
            f"{model.reference.formula(model.coefficients)}\n"
            f"S_x = {tester.sx(model, series):.4f}; горизонт прогнозування = {horizon} кроків (δ = {args.delta:.0%})\n"
            f"прогноз наступного значення: {model.predict(series.values):.3f}"
        )
        return 0

    def _select(self, series: TimeSeries, args) -> int:
        prefs = Preferences("ОПР", self.parse_pairwise(args.pairwise), args.sx_max, args.h_min, args.delta)
        ahp = prefs.ahp
        self.show("ваги критеріїв: " + ", ".join(f"{c} {w:.3f}" for c, w in zip(prefs.criteria, ahp["weights"])) + f"; CR = {ahp['CR']:.3f}")
        if ahp["CR"] >= 0.1:
            self.show("судження неузгоджені (CR ≥ 0,1): перегляньте парні порівняння")
            return 2
        ds = DecisionSupport(prefs, args.max_steps)
        for r in ds.rank(ds.evaluate(series, standard_candidates())):
            mark = "корисна" if r["useful"] else ("не синтезовано: " + r["error"] if r.get("model") is None else "не корисна")
            self.show(f"{r['rank']}. {r['name']}: S_x = {r['sx']:.3f}, H = {r['horizon']}, коефіцієнтів {r['size']}, U = {r['utility']:.3f} ({mark})")
        return 0

    def show(self, text: str) -> None:
        print(text, file=self.out)

    @staticmethod
    def _parser() -> argparse.ArgumentParser:
        p = argparse.ArgumentParser(description="Синтезатор прогнозних моделей на основі полінома Колмогорова-Габора")
        p.add_argument("csv", help="CSV з первинним описом функції")
        p.add_argument("--column", default="F", help="стовпець зі значеннями функції")
        p.add_argument("--model", default="1; F1; F2; F1*F2", help="члени опорної моделі через «;»")
        p.add_argument("--select", action="store_true", help="обрати модель за перевагами ОПР")
        p.add_argument("--pairwise", default="1,1/3,3; 3,1,5; 1/3,1/5,1", help="матриця парних порівнянь критеріїв")
        p.add_argument("--sx-max", type=float, default=0.5, help="найбільше допустиме S_x")
        p.add_argument("--h-min", type=int, default=2, help="найменший допустимий горизонт")
        p.add_argument("--delta", type=float, default=0.05, help="прийнятна відносна похибка прогнозу")
        p.add_argument("--max-steps", type=int, default=15, help="найбільший перевірюваний горизонт")
        return p


if __name__ == "__main__":
    sys.exit(SynthesisCLI().run())
