"""The one place the GUI imports Panel."""

from __future__ import annotations

from typing import Any

INSTALL_HINT = "reinstall VAFT (`pip install vaft`), or `pip install panel`"


def require_panel() -> Any:
    """``panel``, or an ImportError saying how to get it back.

    Panel is a VAFT dependency, so this fails only in an environment that
    lost it, or one installed with ``--no-deps``.
    """
    try:
        import panel
    except ImportError as error:
        raise ImportError(
            f"The VAFT GUI needs the panel package, which VAFT depends on but this environment lacks; {INSTALL_HINT}."
        ) from error
    return panel
