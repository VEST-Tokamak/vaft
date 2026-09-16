"""Record every IMAS-DD path a plot builder reads (issue #439 declaration tests).

A test utility, not a test: ``test_plot_recipe_reads.py`` imports it by bare
name (``test/`` is on ``sys.path`` through ``conftest.py``).  Two recorders:

* :func:`accessor_reads` wraps :func:`vaft.plot.backend.access.accessor_for`,
  so every ``get``/``count``/``has`` call is recorded whatever object answers
  it -- an OMAS ODS or a native IMAS entry alike.
* :func:`ods_reads` wraps ``omas.ODS`` subscripting, membership and
  iteration, so the reads of an OMAS-bound helper that never goes through the
  accessor are recorded too.  It is best-effort: omas internals that read
  ``omas_data`` directly (``time()``, ``flat()``, ``paths()``) leave no trace.

:func:`covered` is the coverage rule: a recorded path is declared when it
matches a template exactly (``{i}`` stands for any index) or is a container
or root above a declared leaf.  :func:`suggest` turns a recording into the
templates a declaration would need, for deriving a recipe's ``reads``.
"""

from __future__ import annotations

import contextlib
import dataclasses
import re
from typing import Any, Iterator, Mapping, Sequence

import numpy as np

_PLACEHOLDER = re.compile(r"\{[a-z]\}")
_DIGITS = re.compile(r"(?<=\.)(\d+|:)(?=\.|$)")
_LETTERS = "ijklmn"

#: A declared leaf that the ODS decodes into a subtree: reads below it count
#: as reads of it.
_DECODED_LEAVES = ("code.parameters",)


class Recorded:
    """The concrete paths read, in order, without duplicates."""

    def __init__(self) -> None:
        self.paths: list[str] = []
        self._seen: set[str] = set()

    def add(self, path: str) -> None:
        if path and path not in self._seen:
            self._seen.add(path)
            self.paths.append(path)


@contextlib.contextmanager
def accessor_reads(monkeypatch) -> Iterator[Recorded]:
    """Every path read through :mod:`vaft.plot.backend.access`."""
    from vaft.plot.backend import access

    recorded = Recorded()
    real = access.accessor_for

    class _Recording:
        def __init__(self, inner: Any) -> None:
            self._inner = inner

        def get(self, obj: Any, path: str, default: Any = None) -> Any:
            recorded.add(path)
            return self._inner.get(obj, path, default)

        def count(self, obj: Any, path: str) -> int:
            recorded.add(path)
            return self._inner.count(obj, path)

        def has(self, obj: Any, path: str) -> bool:
            recorded.add(path)
            return self._inner.has(obj, path)

    monkeypatch.setattr(access, "accessor_for", lambda obj: _Recording(real(obj)))
    yield recorded


@contextlib.contextmanager
def ods_reads(monkeypatch) -> Iterator[Recorded]:
    """Every path read on any ``omas.ODS``: subscripts, ``in``, ``keys()``, ``len()``."""
    from omas import ODS

    recorded = Recorded()

    def full(ods: Any, key: Any) -> str:
        head = str(getattr(ods, "location", "") or "")
        tail = key if isinstance(key, str) else ".".join(str(k) for k in (key if isinstance(key, (list, tuple)) else (key,)))
        # A sub-ODS handed an absolute path (``slice_ods["equilibrium.time_slice.0.q"]``
        # from inside a private copy) is read from its top: keep the key.
        if head and tail and tail.split(".")[0] == head.split(".")[0]:
            return tail
        return f"{head}.{tail}" if head and tail else (head or tail)

    real_getitem, real_contains = ODS.__getitem__, ODS.__contains__

    # omas' own traversal (consistency checks, deep copies, the recursion
    # behind one dotted read) calls ``__getitem__(key, cocos_and_coords)``
    # with the flag given; a caller's read leaves it at its default.  Only
    # the latter is a read of the plot's input.
    def getitem(self, key, *args, **kwargs):
        if not args and "cocos_and_coords" not in kwargs:
            recorded.add(full(self, key))
        return real_getitem(self, key, *args, **kwargs)

    def contains(self, key, *args, **kwargs):
        recorded.add(full(self, key))
        return real_contains(self, key, *args, **kwargs)

    monkeypatch.setattr(ODS, "__getitem__", getitem)
    monkeypatch.setattr(ODS, "__contains__", contains)
    for name in ("keys", "__len__", "__iter__"):
        real = getattr(ODS, name)

        def wrapped(self, *args, _real=real, **kwargs):
            recorded.add(str(getattr(self, "location", "") or ""))
            return _real(self, *args, **kwargs)

        monkeypatch.setattr(ODS, name, wrapped)
    yield recorded


def generalise(path: str) -> str:
    """``magnetics.ip.0.data`` -> ``magnetics.ip.{i}.data`` (letters in order)."""
    counter = iter(_LETTERS)
    return _DIGITS.sub(lambda m: "{" + next(counter) + "}", path)


def suggest(recorded: Recorded) -> tuple[str, ...]:
    """The distinct templates a recording would need declared, sorted."""
    return tuple(sorted({generalise(p) for p in recorded.paths if "." in p}))


def _pattern(template: str) -> re.Pattern:
    escaped = re.escape(template)
    escaped = escaped.replace(r"\{", "{").replace(r"\}", "}")
    return re.compile("^" + _PLACEHOLDER.sub(r"(?:\\d+|:)", escaped) + "$")


def _prefixes(template: str) -> list[str]:
    parts = template.split(".")
    return [".".join(parts[:n]) for n in range(1, len(parts))]


def covered(path: str, templates: Sequence[str]) -> bool:
    """Whether ``path`` is declared by ``templates`` (or lies above a declared leaf).

    A bare IDS root in ``templates`` (``"pf_active"``) declares the whole IDS
    -- what a builder that deep-copies it consumes -- and covers every path
    beneath it.  omas spells a structure-array index ``element[0]`` in some
    internal locations; that is read as ``element.0``.
    """
    concrete = path.rstrip(".").replace("[", ".").replace("]", "")
    for template in templates:
        if "." not in template:
            if concrete == template or concrete.startswith(template + "."):
                return True
            continue
        if _pattern(template).match(concrete):
            return True
        # An element of a declared array leaf (``vacuum_toroidal_field.b0.3``).
        if re.match(_pattern(template).pattern[:-1] + r"\.\d+$", concrete):
            return True
        if any(_pattern(prefix).match(concrete) for prefix in _prefixes(template)):
            return True
        for leaf in _DECODED_LEAVES:
            if template.endswith(leaf) and _pattern(template).match(concrete.split(leaf)[0] + leaf):
                return True
    return False


#: Read by every builder for the entry label a model carries; never a plot's input.
SHOT_LABEL = "dataset_description.data_entry.pulse"


def undeclared(recorded: Recorded, templates: Sequence[str], ignored: Mapping[str, str] = {}) -> list[str]:
    """The recorded paths neither declared nor explicitly ignored."""
    declared = (*templates, SHOT_LABEL)
    return [
        path for path in recorded.paths
        if not covered(path, declared)
        and not any(_pattern(t).match(path) for t in ignored)
    ]


def assert_models_equal(a: Any, b: Any, where: str = "") -> None:
    """Field-by-field equality of two view models (the equivalence suite's rule)."""
    if dataclasses.is_dataclass(a):
        assert type(a) is type(b), f"{where}: {type(a).__name__} vs {type(b).__name__}"
        for f in dataclasses.fields(a):
            assert_models_equal(getattr(a, f.name), getattr(b, f.name), f"{where}.{f.name}")
    elif isinstance(a, np.ndarray):
        assert np.shape(a) == np.shape(b), f"{where}: shapes {np.shape(a)} vs {np.shape(b)}"
        if np.asarray(a).dtype.kind in "fiu":
            assert np.allclose(a, b, equal_nan=True), where
        else:
            assert np.array_equal(a, b), where
    elif isinstance(a, Mapping):
        assert dict(a) == dict(b), where
    elif isinstance(a, (tuple, list)):
        assert len(a) == len(b), f"{where}: {len(a)} vs {len(b)} items"
        for i, (x, y) in enumerate(zip(a, b)):
            assert_models_equal(x, y, f"{where}[{i}]")
    else:
        assert a == b, f"{where}: {a!r} vs {b!r}"
