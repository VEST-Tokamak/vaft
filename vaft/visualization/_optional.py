"""Import the optional 3-D libraries, naming the extra that installs them."""

from __future__ import annotations

from typing import Any

__all__ = ["require_k3d", "require_pyvista"]


def require_pyvista() -> Any:
    """``pyvista``, or an ImportError naming ``vaft[vtk]``."""
    try:
        import pyvista
    except ImportError as error:
        raise ImportError(
            "VTK/ParaView export needs the pyvista package, which is optional; "
            "install it with `pip install vaft[vtk]` (or `pip install pyvista`)."
        ) from error
    return pyvista


def require_k3d() -> Any:
    """``k3d``, or an ImportError naming ``vaft[jupyter3d]``."""
    try:
        import k3d
    except ImportError as error:
        raise ImportError(
            "Interactive 3-D notebook views need the k3d package, which is optional; "
            "install it with `pip install vaft[jupyter3d]` (or `pip install k3d`)."
        ) from error
    return k3d
