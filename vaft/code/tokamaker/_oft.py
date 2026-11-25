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

# The upstream wheel (``pip install 'vaft[tokamaker]'``) needs no environment at
# all: the package loads the ``liboftpy`` bundled next to it.  For a source
# build, ``$OFT_ROOTPATH`` names its install root (``make install`` prefix,
# holding ``python/`` and ``bin/``), the convention of the upstream tutorials;
# the package directory is then added to sys.path and finds ``../../bin`` itself.
OFT_ROOTPATH_ENV = "OFT_ROOTPATH"

#: The oldest OFT release VAFT supports (``pip install 'openfusiontoolkit>=26.9'``).
#: Releases before it rebuild F(psi) wrongly for F0 < 0 in the F*F' profile mode
#: TokaMaker uses (OFT issue #321, fixed in #322 on 2026-07-30 and first released
#: in 26.9): the equilibrium is right, but FPOL, q and B_T are not.
MIN_OFT_VERSION = (26, 9)

#: Set to ``1`` to run a negative-F0 case on an OFT that does not report a
#: version (an uninstalled source tree) after checking it contains upstream #322.
ALLOW_UNVERSIONED_OFT_ENV = "VAFT_TOKAMAKER_ALLOW_UNVERSIONED_OFT"

_MISSING_OFT_MESSAGE = (
    "OpenFUSIONToolkit (TokaMaker) is not importable in this environment. "
    "Install the upstream release: `pip install 'vaft[tokamaker]'` (the "
    "`openfusiontoolkit>=26.9` wheel, which bundles its compiled libraries). "
    f"To use a source build instead, set ${OFT_ROOTPATH_ENV} to its install "
    "root (the directory holding python/ and bin/) and the package directory "
    "is added to sys.path automatically."
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


def setup_model(mygs, pts, lc, reg, *, cond_dict=None, coil_dict=None, order: int = 2, F0: float = 0.0) -> None:
    """Mesh, regions and FE space in one call, for the OFT version installed.

    OFT ``main`` after v26.9 (upstream #350, 2026-11-09; in no release yet)
    added ``TokaMaker.setup_model`` and deprecated the ``setup_mesh`` /
    ``setup_regions`` / ``setup`` sequence (it warns and will remove it).
    Releases up to v26.9 only have the sequence.  Either way
    ``mygs.settings`` must already hold the run settings: ``setup_model``
    applies them during the FE setup, as ``setup`` did.  ``setup_regions`` is
    called on the old path only when region dictionaries are given, which is
    what the fixed-boundary bridge relied on.
    """
    if hasattr(mygs, "setup_model"):
        mygs.setup_model(r=pts, lc=lc, reg=reg, cond_dict=cond_dict, coil_dict=coil_dict, order=order, F0=F0)
        return
    mygs.setup_mesh(pts, lc, reg)
    if cond_dict is not None or coil_dict is not None:
        mygs.setup_regions(cond_dict=cond_dict if cond_dict is not None else {},
                           coil_dict=coil_dict if coil_dict is not None else {})
    mygs.setup(order=order, F0=F0)


def oft_version() -> tuple[int, int] | None:
    """``(major, minor)`` of the importable OpenFUSIONToolkit, or ``None`` if it reports none.

    Read from ``OpenFUSIONToolkit.__version__``.  A wheel or an installed
    (``make install``) build carries e.g. ``"26.9"``, or ``"26.9.dev412"`` for a
    development build of ``main``.  Only the release numbers are kept, so a
    development build of 26.9 -- the version was set after #322 merged --
    counts as 26.9.  An uninstalled source tree carries the unconfigured
    ``"@CMAKE_PROJECT_VERSION@..."`` placeholder, and an old fork none at all:
    both give ``None``.
    """
    import re

    root = importlib.import_module("OpenFUSIONToolkit")
    match = re.match(r"^\s*(\d+)\.(\d+)", str(getattr(root, "__version__", "")))
    return (int(match.group(1)), int(match.group(2))) if match else None


def require_negative_f0_support(f0: float | None) -> None:
    """Refuse a F0 < 0 solve on an OFT that writes the wrong F(psi), q and B_T for it.

    A positive (or unset) F0 is unaffected by OFT issue #321 and always passes.
    For F0 < 0 the OFT version must be at least :data:`MIN_OFT_VERSION`.  An
    OFT that reports no version passes only when
    ``$VAFT_TOKAMAKER_ALLOW_UNVERSIONED_OFT=1`` says the caller checked it.

    Raises
    ------
    RuntimeError
        F0 < 0 on an OFT older than 26.9, or on an unversioned one without the
        override.
    """
    if f0 is None or not f0 < 0.0:
        return
    version = oft_version()
    if version is not None and version >= MIN_OFT_VERSION:
        return
    if version is None and os.environ.get(ALLOW_UNVERSIONED_OFT_ENV, "").strip() == "1":
        _log.warning("F0 < 0 on an OpenFUSIONToolkit that reports no version, allowed by $%s",
                     ALLOW_UNVERSIONED_OFT_ENV)
        return
    found = "no version" if version is None else "version %d.%d" % version
    raise RuntimeError(
        f"F0 = {f0:.6g} < 0 (B_T < 0) needs OpenFUSIONToolkit >= {MIN_OFT_VERSION[0]}.{MIN_OFT_VERSION[1]}, "
        f"found {found}. Older TokaMaker rebuilds F(psi) as a reflection about F0 instead of with its "
        "sign (OFT issue #321, fixed in #322), so the written FPOL, q and B_T would be wrong although the "
        "solve converges. Install `pip install 'openfusiontoolkit>=26.9'`"
        + (f", or set ${ALLOW_UNVERSIONED_OFT_ENV}=1 if this source build contains upstream #322."
           if version is None else ".")
    )
