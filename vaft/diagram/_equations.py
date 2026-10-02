"""Equations a diagram shows, taken from the formula catalog rather than restated."""

from __future__ import annotations


def formula_equation(function) -> str:
    """The defining equation of a ``vaft.formula`` function, as its catalog entry gives it.

    The figures show exactly the first of ``FormulaSpec.definitions``, so an
    equation on a diagram cannot drift from the one the formula documents,
    the reference page shows and ``vaft.formula.show`` renders.
    """
    from vaft.formula.catalog import CATEGORIES, describe

    package, _, category = getattr(function, "__module__", "").rpartition(".")
    name = getattr(function, "__name__", repr(function))
    if package != "vaft.formula" or category not in CATEGORIES:
        raise ValueError(f"{name} is not a vaft.formula function, so it documents no equation")
    definitions = describe(f"{category}.{name}").definitions
    if not definitions:
        raise ValueError(f"{name} documents no $$...$$ equation")
    return " ".join(definitions[0].split())
