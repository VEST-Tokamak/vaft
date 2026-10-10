"""Attach a domain-owned analytic Jacobian to a formula without wrapping it (#1874).

:func:`jacobian` is a decorator that records ``d f / d x`` on the function
object and returns that *same* object: the formula's signature, values,
return type and cost are unchanged, and nothing runs on an ordinary call.
Only the opt-in :func:`vaft.formula.sensitivity.propagate_formula_uncertainty`
reads it.

The Jacobian callable takes the formula's arguments by name and returns one
row per output and one column per argument it names in ``wrt``, in that order.
A formula carrying one must document it in an ``Uncertainty propagation``
docstring section; the catalog reports a formula that does not.
"""

from __future__ import annotations

from typing import Callable, Optional, Tuple

_ATTRIBUTE = "__vaft_jacobian__"


def jacobian(derivative: Callable[..., object], *, wrt: Tuple[str, ...]):
    """Record ``derivative`` as the analytic Jacobian of the decorated formula with respect to ``wrt``."""

    def attach(function):
        setattr(function, _ATTRIBUTE, (derivative, tuple(wrt)))
        return function

    return attach


def analytic_jacobian(function) -> Optional[Tuple[Callable[..., object], Tuple[str, ...]]]:
    """The ``(derivative, wrt)`` a formula carries, or ``None``."""
    return getattr(function, _ATTRIBUTE, None)
