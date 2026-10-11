"""Attach a domain-owned analytic Jacobian and validity domain to a formula without wrapping it (#1874).

:func:`jacobian` is a decorator that records ``d f / d x`` -- and, optionally,
the domain where the formula is defined -- on the function object and returns
that *same* object: the formula's signature, values, return type and cost are
unchanged, and nothing runs on an ordinary call.  Only the opt-in
:func:`vaft.formula.sensitivity.propagate_formula_uncertainty` reads them.

The Jacobian callable receives every argument of the call by name (uncertain
and fixed alike, so it should accept ``**_`` for the ones it ignores) and
returns one row per output and one column per name in ``wrt``, in that order.
``domain``, when given, receives the same arguments and returns ``True``
where the formula is physically defined: a quotient that is finite for a
negative denominator still has no meaning there.  A formula carrying a
Jacobian must document it in an ``Uncertainty propagation`` docstring
section; the catalog reports a formula that does not.
"""

from __future__ import annotations

from typing import Callable, Optional, Tuple

_ATTRIBUTE = "__vaft_jacobian__"
_DOMAIN = "__vaft_domain__"


def jacobian(derivative: Callable[..., object], *, wrt: Tuple[str, ...],
             domain: Optional[Callable[..., bool]] = None):
    """Record ``derivative`` (and ``domain``) on the decorated formula, returning the formula itself."""

    def attach(function):
        setattr(function, _ATTRIBUTE, (derivative, tuple(wrt)))
        if domain is not None:
            setattr(function, _DOMAIN, domain)
        return function

    return attach


def analytic_jacobian(function) -> Optional[Tuple[Callable[..., object], Tuple[str, ...]]]:
    """The ``(derivative, wrt)`` a formula carries, or ``None``."""
    return getattr(function, _ATTRIBUTE, None)


def formula_domain(function) -> Optional[Callable[..., bool]]:
    """The validity-domain predicate a formula carries, or ``None``."""
    return getattr(function, _DOMAIN, None)
