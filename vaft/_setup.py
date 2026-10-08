"""Explicit runtime preparation: ``vaft.setup()`` (#1203).

``vaft.setup()`` changes non-scientific runtime conveniences -- today, which
Matplotlib backend a notebook draws with -- and only when called.  Importing
VAFT never runs it.  It reports what it chose and why as a :class:`VAFTSetup`.

Profiles:

``auto``
    In a Jupyter or VS Code kernel, behave like ``notebook``; anywhere else,
    change nothing and report.
``notebook``
    In a kernel with ``ipympl`` installed, switch to the live ``ipympl``
    backend (pan/zoom, canvas sliders).  Without ``ipympl``, or outside a
    kernel, change nothing and say why.
``batch``
    Keep execution headless: when no backend has been chosen yet and
    ``MPLBACKEND`` is not set, select ``Agg``.  The caller's environment is
    left alone -- unless Matplotlib had to fall back to a temporary
    configuration directory because its default is not writable, in which
    case ``MPLCONFIGDIR`` is exported so that child processes share this
    run's font cache instead of each rebuilding one, and ``details`` says so.
``database``
    Diagnose HSDS configuration and point to ``vaft hsds configure``.  It never
    prompts, never writes a file and never contacts the server.

What is never touched: an explicit ``MPLBACKEND`` (other than the inline one
Jupyter itself exports), a backend already selected in the process (``Agg``
included, and ``%matplotlib inline`` where Matplotlib 3.9+ records it as
``inline``; older releases cannot tell it from Jupyter's default), the
process environment (``MPLCONFIGDIR`` above excepted, and reported), and every
scientific choice -- COCOS, coordinates, filtering, fitting, slice selection.
Switching to ``ipympl`` does what ``%matplotlib widget`` does, including
interactive mode (``rcParams["interactive"]``).  Calling ``setup`` again with
the same profile is a no-op.
"""

from __future__ import annotations

import importlib.util
import os
import tempfile
from dataclasses import asdict, dataclass

__all__ = ["PROFILES", "VAFTSetup", "setup"]

PROFILES = ("auto", "notebook", "batch", "database")

#: The backend ipykernel exports as ``MPLBACKEND`` in every kernel; its
#: presence is Jupyter's default, not the user's choice.
_INLINE = "module://matplotlib_inline.backend_inline"
_IPYMPL = "module://ipympl.backend_nbagg"
_KERNELS = ("jupyter", "vscode")


def _is_ipympl(backend: str | None) -> bool:
    """``ipympl`` under any of its names.

    Matplotlib's backend registry (3.9+) reports it as ``widget`` or
    ``ipympl`` after ``%matplotlib widget``; older releases as the module path.
    :data:`vaft.plot.environment._LIVE_BACKENDS` knows only the module path,
    so the live check is completed here.
    """
    name = (backend or "").lower()
    return name in ("widget", "ipympl") or name.startswith("module://ipympl")


@dataclass(frozen=True)
class VAFTSetup:
    """What :func:`setup` found, chose, and why.

    ``backend`` is the Matplotlib backend after the call (``None`` when setup
    left it unchosen); ``previous_backend`` what it was before.  ``reasons``
    explain each choice; ``details`` carries diagnostic rows (the ``database``
    profile).
    """

    profile: str
    environment: str
    backend: str | None = None
    previous_backend: str | None = None
    live_figures: bool | None = None
    changed: bool = False
    reasons: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    details: tuple[tuple[str, str], ...] = ()

    def as_dict(self) -> dict:
        data = asdict(self)
        for key in ("reasons", "warnings"):
            data[key] = list(data[key])
        data["details"] = [list(row) for row in self.details]
        return data

    def _rows(self) -> list[tuple[str, str]]:
        rows = [("profile", self.profile), ("environment", self.environment)]
        if self.profile != "database":
            before, after = self.previous_backend or "not chosen", self.backend or "not chosen"
            rows.append(("backend", f"{before} -> {after}" if self.changed else after))
            if self.live_figures is not None:
                rows.append(("live figures", "enabled" if self.live_figures else "no"))
        rows += list(self.details)
        return rows

    def render(self) -> str:
        rows = self._rows()
        width = max(len(label) for label, _ in rows)
        lines = ["VAFT setup"] + [f"  {label.ljust(width)}  {value}" for label, value in rows]
        lines += [f"  because: {reason}" for reason in self.reasons]
        lines += [f"  warning: {warning}" for warning in self.warnings]
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.render()

    def __repr__(self) -> str:
        return self.render()

    def _repr_markdown_(self) -> str:
        out = ["### VAFT setup", "", "| | |", "|---|---|"]
        out += [f"| {label} | `{value}` |" for label, value in self._rows()]
        out += [""] + [f"- because: {reason}" for reason in self.reasons]
        out += [f"- **warning:** {warning}" for warning in self.warnings]
        return "\n".join(out)


# -- environment ---------------------------------------------------------------
def _kernel_kind() -> str:
    """Frontend kind without making Matplotlib pick a backend.

    :func:`vaft.plot.environment.detect_environment` resolves the backend, so it
    is only consulted once a kernel is known to be present.
    """
    from ._help._providers import _frontend_kind

    return _frontend_kind()


def _chosen_backend() -> str | None:
    """The backend already selected in this process, or ``None`` if unresolved."""
    import matplotlib

    chosen = matplotlib.rcParams._get_backend_or_none()
    return str(chosen).lower() if chosen is not None else None


def _explicit_backend() -> str | None:
    """``MPLBACKEND`` when the user set it (not Jupyter's inline default)."""
    value = os.environ.get("MPLBACKEND", "").strip()
    return value if value and value.lower() != _INLINE else None


def _has_ipympl() -> bool:
    return importlib.util.find_spec("ipympl") is not None


def _switch_to_ipympl() -> None:
    """Enable ``ipympl`` the way ``%matplotlib widget`` does.

    The backend is named by its module path rather than ``widget``: with
    Matplotlib 3.9+, ``widget`` becomes the backend's recorded name, and
    :func:`vaft.plot.environment.detect_environment` recognises only the
    module path as a live canvas -- so VAFT's own interactive plots would
    otherwise fall back to static figures.  An IPython too old to accept a
    module path in ``%matplotlib`` gets ``widget``.
    """
    try:
        from IPython import get_ipython

        shell = get_ipython()
    except ImportError:  # pragma: no cover - IPython is a dependency
        shell = None
    if shell is not None:
        try:
            shell.run_line_magic("matplotlib", _IPYMPL)
        except (KeyError, ValueError, RuntimeError):
            shell.run_line_magic("matplotlib", "widget")
    else:  # pragma: no cover - a kernel always has a shell
        import matplotlib

        matplotlib.use(_IPYMPL)


def _use_agg() -> None:
    """Select ``Agg`` without touching the environment.

    Not :func:`vaft.plot.environment.use_non_interactive_backend`: that one
    also exports ``MPLCONFIGDIR`` to a fresh temporary directory, which is
    inert for this process (Matplotlib is already imported and its
    configuration directory resolved) but makes every child process rebuild
    the font cache there, silently (cold review 0.8.0 delta-absorb-2 F6).
    """
    import matplotlib

    matplotlib.use("Agg", force=False)


def _fallback_config_dir() -> str | None:
    """The temporary configuration directory Matplotlib fell back to, or ``None``.

    When its default directory (``~/.matplotlib``, or the XDG one) is not
    writable, Matplotlib creates ``<tmp>/matplotlib-*`` for the process and
    says so in a warning.  Nothing set ``MPLCONFIGDIR`` in that case, or
    Matplotlib would have used it.
    """
    if os.environ.get("MPLCONFIGDIR"):
        return None
    import matplotlib

    configdir = os.fspath(matplotlib.get_configdir())
    parent, name = os.path.split(configdir)
    temp = tempfile.gettempdir()
    if name.startswith("matplotlib-") and parent in (temp, os.path.realpath(temp)):
        return configdir
    return None


# -- profiles --------------------------------------------------------------------
def _notebook(profile: str, kind: str) -> VAFTSetup:
    before = _chosen_backend()
    explicit = _explicit_backend()
    if kind not in _KERNELS:
        return VAFTSetup(
            profile, kind, backend=before, previous_backend=before,
            reasons=(f"not a notebook kernel ({kind}); the backend is left as it is",),
        )
    if explicit is not None:
        return VAFTSetup(
            profile, kind, backend=before, previous_backend=before,
            reasons=(f"MPLBACKEND={explicit} is set explicitly and is respected",),
        )
    if before is not None and before not in (_INLINE, "agg") and not _is_ipympl(before):
        return VAFTSetup(
            profile, kind, backend=before, previous_backend=before,
            reasons=(f"backend {before} was already chosen in this session and is respected",),
        )
    if before == "agg":
        return VAFTSetup(
            profile, kind, backend=before, previous_backend=before, live_figures=False,
            reasons=("the Agg (headless) backend is already selected and is respected",),
        )

    from vaft.plot.environment import detect_environment

    env = detect_environment()
    if env.live_figures or _is_ipympl(env.backend):
        return VAFTSetup(
            profile, env.kind, backend=env.backend, previous_backend=before, live_figures=True,
            reasons=(f"{env.backend} already draws live figures",),
        )
    if not _has_ipympl():
        return VAFTSetup(
            profile, env.kind, backend=env.backend, previous_backend=before, live_figures=False,
            reasons=("figures stay static (inline)",),
            warnings=("install ipympl (pip install ipympl) for live figures with pan/zoom",),
        )
    try:
        _switch_to_ipympl()
    except Exception as error:  # noqa: BLE001 - an installed but broken ipympl/ipywidgets
        return VAFTSetup(
            profile, env.kind, backend=_chosen_backend() or env.backend, previous_backend=before,
            live_figures=False,
            reasons=("figures stay static (inline)",),
            warnings=(f"ipympl is installed but could not be enabled: {type(error).__name__}: {error}",),
        )
    after = detect_environment()
    return VAFTSetup(
        profile, after.kind, backend=after.backend, previous_backend=env.backend,
        live_figures=after.live_figures or _is_ipympl(after.backend), changed=after.backend != env.backend,
        reasons=(f"{env.kind} kernel detected and ipympl is installed",),
    )


def _batch(profile: str, kind: str) -> VAFTSetup:
    before = _chosen_backend()
    explicit = _explicit_backend()
    if explicit is not None:
        reason = f"MPLBACKEND={explicit} is set explicitly and is respected"
    elif before == _INLINE:
        reason = "the Jupyter inline default is in place; figures render to the notebook, not a window"
    elif before is not None:
        reason = f"backend {before} was already chosen in this session and is respected"
    else:
        _use_agg()
        after = _chosen_backend()
        reasons = ["no backend was chosen yet; Agg keeps the run headless"]
        details: list[tuple[str, str]] = []
        fallback = _fallback_config_dir()
        if fallback is not None:
            # Export the directory Matplotlib already chose so child
            # processes share one font cache; a writable default is never
            # overridden and the environment stays as the caller left it.
            os.environ["MPLCONFIGDIR"] = fallback
            details.append(("MPLCONFIGDIR", fallback))
            reasons.append(
                "Matplotlib's default configuration directory is not writable; "
                "MPLCONFIGDIR is exported so child processes share this run's cache"
            )
        return VAFTSetup(
            profile, kind, backend=after, previous_backend=None, live_figures=False,
            changed=after is not None, reasons=tuple(reasons), details=tuple(details),
        )
    return VAFTSetup(profile, kind, backend=before, previous_backend=before, reasons=(reason,))


def _database(profile: str, kind: str) -> VAFTSetup:
    from ._help._providers import database

    page = database(None)
    details: list[tuple[str, str]] = []
    reasons: list[str] = []
    for section in page["sections"]:
        if section.title == "HSDS configuration":
            details += [(f"hsds {label}", value) for label, value in section.rows]
            reasons.append(section.note)
    details += [(d.name, d.value) for d in page["defaults"] if d.kind == "data-access"]
    return VAFTSetup(
        profile, kind, details=tuple(details),
        reasons=(*reasons, "diagnosis only: nothing was written, prompted or contacted"),
    )


def setup(profile: str = "auto") -> VAFTSetup:
    """Prepare VAFT's non-scientific runtime environment, explicitly.

    Parameters
    ----------
    profile : {"auto", "notebook", "batch", "database"}, optional
        What to prepare; see the module documentation [-].

    Returns
    -------
    VAFTSetup
        The environment, the backend before and after, whether anything
        changed, and why.

    Raises
    ------
    ValueError
        Unknown profile.
    """
    name = str(profile).strip().lower()
    if name not in PROFILES:
        raise ValueError(f"unknown setup profile {profile!r}; choose from: {', '.join(PROFILES)}")
    kind = _kernel_kind()
    if name == "database":
        return _database(name, kind)
    if name == "batch":
        return _batch(name, kind)
    return _notebook(name, kind)
