"""Colour intent tokens: what a recipe means by a colour, resolved when drawn (issue #709).

A recipe used to write ``style={"color": "#e41a1c"}`` and mean "this is the
plasma boundary".  The literal defeated every theme (an explicit colour is
a plain keyword to Matplotlib, so the prop cycle never reached it) and
scattered the meaning across eighty sites.  A recipe now writes the
meaning -- ``feature:boundary`` -- and the colour is looked up when the
figure is drawn, from the theme in force or, without one, from the table
of what those literals always were.  So with no theme every figure comes
out exactly as before, and under a theme every series follows it.

Tokens are plain strings of the form ``<namespace>:<key>``; anything else
(``"k"``, ``"0.4"``, a hex, an RGBA tuple, ``"none"``, ``"C3"``) passes
through untouched, so nothing is forced to convert.  The namespaces:

``palette:<n>``
    the n-th distinguishing colour of a set of series or diagnostics;
``role:<measured|reconstructed|reference>``
    the measurement itself, a reconstruction's prediction of it, a
    reference curve drawn against members;
``feature:<wall|limiter|boundary|axis|coil|passive>``
    a machine or plasma feature drawn for what it is;
``state:<enabled|disabled|missing>``
    the state of a constraint channel;
``emphasis:<strong|medium|low|lower|faint|alert>``
    the graded greys of guides and annotations, and the one that warns.

Deliberately not tokens: an invalid channel's grey (renderer policy,
:data:`vaft.plot.style.INVALID_COLOR`), a fit's colour (it shares its
band's palette slot), ``C<n>`` (Matplotlib's current cycle, which already
follows a theme), a camera overlay's colour (it must contrast with a
photograph, not with a theme), colormaps and continuous maps.

The active theme is read through a context variable that the presentation
context sets for the whole render, so a composite's members resolve like
its own artists and no signature carries a theme.  This module imports no
Matplotlib and no model: recipes take only strings from it.
"""

from __future__ import annotations

import contextlib
import contextvars
from typing import Any, Iterator, Mapping

__all__ = [
    "DEFAULT_COLOURS",
    "DEFAULT_PALETTE",
    "TOKEN_NAMESPACES",
    "active_theme",
    "is_colour_token",
    "palette",
    "resolve_color",
    "resolve_style",
    "themed",
]

#: The distinguishing colours a set of series or diagnostics take without a
#: theme, in the order the top view's diagnostic table first used them
#: (ColorBrewer Set1 and Dark2), so ``palette:0`` .. ``palette:9`` are what
#: those sites always drew.
DEFAULT_PALETTE: tuple[str, ...] = (
    "#377eb8", "#ff7f00", "#984ea3", "#4daf4a", "#a65628",
    "#f781bf", "#999999", "#e6ab02", "#e41a1c", "#66a61e",
)

#: Every non-palette token and the literal it stood for before it was a token.
DEFAULT_COLOURS: Mapping[str, str] = {
    "role:measured": "black",
    "role:reconstructed": "red",
    "role:reference": "k",
    "feature:wall": "0.4",
    "feature:limiter": "k",
    "feature:boundary": "#e41a1c",
    "feature:axis": "k",
    "feature:coil": "#d62728",
    "feature:passive": "0.55",
    "state:enabled": "black",
    "state:disabled": "tab:orange",
    "state:missing": "tab:red",
    "emphasis:strong": "0.35",
    "emphasis:medium": "0.4",
    "emphasis:low": "0.5",
    "emphasis:lower": "0.6",
    "emphasis:faint": "0.75",
    "emphasis:alert": "tab:red",
}

TOKEN_NAMESPACES: tuple[str, ...] = ("palette", "role", "feature", "state", "emphasis")

#: The style keys that carry a colour.
_COLOUR_KEYS = ("color", "markerfacecolor", "markeredgecolor")

_ACTIVE_THEME: contextvars.ContextVar[Any] = contextvars.ContextVar("vaft_plot_active_theme", default=None)


def palette(n: int) -> str:
    """The token for the ``n``-th distinguishing colour."""
    return f"palette:{int(n)}"


def is_colour_token(value: Any) -> bool:
    """Whether ``value`` is a ``<namespace>:<key>`` token this module resolves."""
    if not isinstance(value, str) or ":" not in value:
        return False
    namespace, _, key = value.partition(":")
    return namespace in TOKEN_NAMESPACES and bool(key)


def active_theme() -> Any:
    """The theme in force for the render under way, or ``None``."""
    return _ACTIVE_THEME.get()


@contextlib.contextmanager
def themed(theme: Any) -> Iterator[None]:
    """Make ``theme`` the one colours resolve against until the block ends."""
    token = _ACTIVE_THEME.set(theme)
    try:
        yield
    finally:
        _ACTIVE_THEME.reset(token)


def resolve_color(value: Any, theme: Any = "active") -> Any:
    """``value`` as a colour: a token looked up, anything else unchanged.

    ``theme`` is the theme to resolve against; the default reads the one in
    force, ``None`` resolves against the table of today's literals.  A
    palette token takes the theme's palette (its colour cycle unless it
    names one); another token takes the theme's override for it, which
    may be a colour or a style patch whose colour is taken, else the
    default.  An unknown key in a known namespace is an error: a typo must
    not draw Matplotlib's silent fallback.
    """
    if not is_colour_token(value):
        return value
    if theme == "active":
        theme = active_theme()
    namespace, _, key = value.partition(":")
    if namespace == "palette":
        try:
            index = int(key)
        except ValueError:
            raise ValueError(f"palette token needs an index: {value!r}") from None
        colours = tuple(getattr(theme, "palette", None) or getattr(theme, "colors", None) or ()) or DEFAULT_PALETTE
        return colours[index % len(colours)]
    override = (getattr(theme, "intents", None) or {}).get(value)
    if override is not None:
        return override["color"] if isinstance(override, Mapping) else override
    try:
        return DEFAULT_COLOURS[value]
    except KeyError:
        raise ValueError(
            f"unknown colour intent {value!r}; {namespace}: tokens are "
            + ", ".join(sorted(k for k in DEFAULT_COLOURS if k.startswith(namespace + ":")))
        ) from None


def resolve_style(style: Mapping[str, Any], theme: Any = "active") -> dict[str, Any]:
    """A copy of ``style`` with its colour tokens resolved.

    ``color``, ``markerfacecolor`` and ``markeredgecolor`` are resolved.
    When the theme's override for the ``color`` token is a style patch, its
    other keys are laid over the style too -- a monochrome theme may give a
    role a dash as well as a grey -- except ``markerfacecolor``, so a hollow
    marker stays hollow.
    """
    if theme == "active":
        theme = active_theme()
    resolved = dict(style)
    token = resolved.get("color")
    for key in _COLOUR_KEYS:
        if key in resolved:
            resolved[key] = resolve_color(resolved[key], theme)
    if is_colour_token(token):
        override = (getattr(theme, "intents", None) or {}).get(token)
        if isinstance(override, Mapping):
            for key, patch in override.items():
                if key not in ("color", "markerfacecolor"):
                    resolved[key] = patch
    return resolved
