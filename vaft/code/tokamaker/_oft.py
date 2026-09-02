"""Private helpers for locating and importing the Open FUSION Toolkit (OFT).

TokaMaker is the first *in-process* adapter in ``vaft.code``: unlike the
subprocess-based EFIT/CHEASE/GPEC/TES codes there is no ``$XHOME/bin/<exe>``
to resolve. Instead the compiled toolkit is imported as the Python package
``OpenFUSIONToolkit`` (a ctypes shim over ``liboftpy``), so "availability"
means "importable in this interpreter".

Every import is deferred to :func:`import_oft` so that ``vaft.code.tokamaker``
itself (and ``from vaft.code import *``) works without OFT installed, matching
the lazy-import house style used elsewhere in the package.
"""

from __future__ import annotations

import importlib
import logging
import os
import sys
from types import SimpleNamespace

_log = logging.getLogger(__name__)

# Environment variables honoured when locating OFT. The first two are read by
# OpenFUSIONToolkit itself to find the compiled ``liboftpy`` library; the last
# is the convention used by the upstream tutorials, pointing at an OFT release
# (or source) root whose ``python``/``src/python`` directory is importable.
OFT_LIBRARY_DIR_ENV = "OFT_LIBRARY_DIR"
OFT_INSTALL_DIR_ENV = "OFT_INSTALL_DIR"
OFT_ROOTPATH_ENV = "OFT_ROOTPATH"

_MISSING_OFT_MESSAGE = (
    "OpenFUSIONToolkit (TokaMaker) is not importable in this environment. "
    "Install the Python package from a compiled Open FUSION Toolkit checkout "
    "or release, e.g. `pip install -e <OFT_ROOT>/src/python`. If the import "
    "fails because the compiled library cannot be located, point "
    f"${OFT_LIBRARY_DIR_ENV} (directory containing liboftpy) or "
    f"${OFT_INSTALL_DIR_ENV} (install root containing bin/liboftpy) at a "
    f"built toolkit. Alternatively set ${OFT_ROOTPATH_ENV} to an OFT release "
    "root and the package directory is added to sys.path automatically."
)


def _import_modules() -> SimpleNamespace:
    root = importlib.import_module("OpenFUSIONToolkit")
    tokamaker = importlib.import_module("OpenFUSIONToolkit.TokaMaker")
    meshing = importlib.import_module("OpenFUSIONToolkit.TokaMaker.meshing")
    util = importlib.import_module("OpenFUSIONToolkit.TokaMaker.util")
    return SimpleNamespace(
        OFT_env=root.OFT_env,
        TokaMaker=tokamaker.TokaMaker,
        meshing=meshing,
        util=util,
    )


def import_oft() -> SimpleNamespace:
    """Import OpenFUSIONToolkit lazily, honouring ``$OFT_ROOTPATH``.

    Returns a namespace with ``OFT_env`` (runtime class), ``TokaMaker`` (solver
    class), and the ``meshing`` / ``util`` modules. Raises an actionable
    ``ImportError`` when the toolkit (or its compiled library) is unavailable.
    """
    try:
        return _import_modules()
    except ImportError as exc:
        first_error = exc

    rootpath = os.environ.get(OFT_ROOTPATH_ENV, "").strip()
    if rootpath:
        candidates = [
            os.path.join(rootpath, "python"),      # release-tarball layout
            os.path.join(rootpath, "src", "python"),  # source-tree layout
        ]
        added = [path for path in candidates if os.path.isdir(path) and path not in sys.path]
        sys.path.extend(added)
        if added:
            try:
                return _import_modules()
            except ImportError as exc:
                first_error = exc

    raise ImportError(f"{_MISSING_OFT_MESSAGE} (import failed with: {first_error})") from first_error


def get_oft_env(nthreads: int = 2):
    """Return the per-interpreter ``OFT_env``, creating it on first use.

    ``OFT_env`` is a hard singleton: constructing it twice in one Python kernel
    raises ``RuntimeError``, and the first instance is exposed as the class
    attribute ``OFT_env.instance``. ``nthreads`` therefore only takes effect on
    the first call in a given kernel; later calls with a different value reuse
    the existing environment and log a warning.
    """
    oft = import_oft()
    instance = getattr(oft.OFT_env, "instance", None)
    if instance is not None:
        if nthreads != getattr(instance, "nthreads", nthreads):
            _log.warning(
                "OFT_env already initialised with nthreads=%s; requested "
                "nthreads=%s is ignored (one OFT_env per Python kernel).",
                getattr(instance, "nthreads", "?"),
                nthreads,
            )
        return instance
    return oft.OFT_env(nthreads=nthreads)
