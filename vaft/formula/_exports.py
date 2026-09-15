"""How a formula submodule decides what it publishes.

A hand-written ``__all__`` is a second list to keep in step with the code, and
it does not stay in step: #711 found seven functions that `develop` had added
to ``equilibrium`` and ``utils`` while a branch held a stale copy, which
dropped them out of ``vaft.formula``'s resolvable surface while the catalog
still listed them.

Deriving the list removes that failure mode for the case it actually happens
in -- someone adds a function -- while keeping the one thing the hand-written
list was for: not re-exporting a module's own imports, so ``np``, ``Union``
and ``curve_fit`` stay out of ``vaft.formula.__all__`` (#368).

Constants stay explicit.  A float carries no ``__module__``, so nothing can
tell ``MU0`` defined here from ``MU0`` imported from ``.constants``; naming
them is the honest way to say which is which.
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping


def public_names(namespace: Mapping[str, Any], *, constants: Iterable[str] = ()) -> list[str]:
    """The functions and classes this module defines, plus the constants it names.

    Parameters
    ----------
    namespace : mapping
        The module's own ``globals()``, called at the bottom of the module so
        every definition is already bound [-].
    constants : iterable of str, optional
        Module-level values that carry no ``__module__`` and so cannot be
        detected -- floats, tuples, dataclass instances [-].

    Returns
    -------
    list of str
        Sorted names, suitable as ``__all__`` [-].

    Raises
    ------
    KeyError
        The namespace has no ``__name__``, so it is not a module's globals.
    NameError
        A named constant is not bound in the namespace, which would make
        ``from <module> import *`` raise at import time instead.

    Applicability
    -------------
    Machine-independent.

    Provenance
    ----------
    .. [1] Issue #368 for excluding a module's own imports, and #711 for the
       stale-list failure this removes.
    """
    module = namespace["__name__"]
    missing = [name for name in constants if name not in namespace]
    if missing:
        raise NameError(
            f"{module}: public_names was given constants that are not defined here: "
            f"{sorted(missing)}"
        )
    defined = {
        name
        for name, value in namespace.items()
        if not name.startswith("_") and getattr(value, "__module__", None) == module
    }
    return sorted(defined | set(constants))
