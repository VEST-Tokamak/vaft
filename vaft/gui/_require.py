"""The one place the GUI imports Panel."""

from __future__ import annotations

from typing import Any

INSTALL_HINT = "install it with `pip install 'vaft[gui]'` (or `pip install panel`)"


def require_panel() -> Any:
    """``panel``, or an ImportError naming ``vaft[gui]``."""
    try:
        import panel
    except ImportError as error:
        raise ImportError(
            f"The VAFT GUI needs the panel package, which is optional; {INSTALL_HINT}."
        ) from error
    return panel
