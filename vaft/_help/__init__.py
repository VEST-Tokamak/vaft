"""Capability-oriented help: ``vaft.help()`` and ``vaft help`` (#1203).

``vaft.help()`` answers *what can VAFT do here, with which defaults, and what
is configured* -- without the reader knowing the package layout.  It reads;
it never changes runtime state (no backend switch, no environment variable,
no file, no network), and it never prints a secret.

    >>> import vaft
    >>> vaft.help()                       # overview of topics
    >>> vaft.help("database")             # defaults + HSDS configuration status
    >>> vaft.help("formula", "q95")       # delegated to vaft.formula.catalog

The result is data (:class:`~vaft._help._model.HelpPage`, with
:meth:`~vaft._help._model.HelpPage.as_dict`) that renders as text when
printed and as Markdown in a notebook.  ``import vaft``, ``vaft.help`` and
the overview import no subsystem; a topic imports only the subsystem it
describes, when it is asked for.  Help itself writes nothing, but the first
import of a subsystem can carry its libraries' own import-time effects
(Matplotlib building its font cache for ``plot``, omas compiling helpers for
``omas``/``imas``/``process``).
"""

from __future__ import annotations

from importlib import import_module

from ._model import DEFAULT_KINDS, Default, HelpPage, Section, Topic
from ._registry import TOPICS

__all__ = ["DEFAULT_KINDS", "Default", "HelpPage", "Section", "Topic", "help", "topics"]


def _resolve(target: str):
    module, _, name = target.partition(":")
    return getattr(import_module(module), name)


def topics() -> tuple[str, ...]:
    """Names accepted by :func:`help`, in display order."""
    return tuple(TOPICS)


def help(topic: str | None = None, item: str | None = None, *, probe: bool = False):
    """Describe VAFT, one topic of it, or one item within a topic.

    Parameters
    ----------
    topic : str, optional
        One of :func:`topics`; ``None`` gives the overview [-].
    item : str, optional
        An entry within the topic -- a formula, process, plot query, source,
        check, sample shot or command.  It is answered by the topic's own
        describe/catalog function, whose object is returned as is [-].
    probe : bool, optional
        For ``"code"``: look for each external code's executable under its
        ``*HOME`` instead of reporting only whether the variable is set.
        Nothing is launched either way; other topics ignore it [-].

    Returns
    -------
    HelpPage or object
        A :class:`HelpPage` for a topic; the subsystem's own description for
        an item.

    Raises
    ------
    KeyError
        Unknown topic, or an item the topic's catalog does not know.
    ValueError
        ``item`` given for a topic without a per-item view.

    Notes
    -----
    A topic whose subsystem cannot be imported or described here (an optional
    dependency missing, a run under ``python -OO``) still returns its page,
    with the reason under ``warnings``.
    """
    name = "overview" if topic is None else str(topic).strip().lower()
    if name not in TOPICS:
        raise KeyError(f"unknown help topic {topic!r}; choose from: {', '.join(TOPICS)}")
    entry = TOPICS[name]
    if item is not None:
        if not entry.item:
            raise ValueError(f"help topic {name!r} has no per-item view; try vaft.help({name!r})")
        return _resolve(entry.item)(str(item))

    fields: dict = {}
    warnings: tuple[str, ...] = ()
    try:
        fields = dict(_resolve(entry.provider)(entry, probe=probe))
    except Exception as error:  # noqa: BLE001 - help must describe even a broken environment
        warnings = (f"{name} details unavailable here: {type(error).__name__}: {error}",)
    return HelpPage(
        topic=name,
        summary=entry.summary,
        entry_points=entry.entry_points,
        defaults=tuple(fields.get("defaults", ())),
        sections=tuple(fields.get("sections", ())),
        optional=tuple(fields.get("optional", ())),
        cli=entry.cli,
        setup=entry.setup + tuple(fields.get("setup", ())),
        see_also=entry.see_also,
        warnings=warnings + tuple(fields.get("warnings", ())),
    )
