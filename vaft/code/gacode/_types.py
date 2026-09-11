"""Dataclasses and constants shared by every GACODE-suite backend.

GACODE is a suite -- NEO, TGLF, CGYRO and their tools share one source tree,
one build and one profile format -- so the runtime contract lives here, at the
suite level, and each backend adds only what is specific to it.  That is the
same shape ``vaft.code.gpec`` uses for DCON/RDCON/STRIDE/GPEC, and it is why
this is not a top-level ``vaft.code.neo``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from ..base import CodeConfig

#: The VAFT-side installation root, matching $GPECHOME, $CHEASEHOME and friends.
GACODE_HOME_ENV = "GACODEHOME"

#: GACODE's own root variable.  VAFT reads it as a fallback and *sets* it for
#: the subprocess, but never redefines what it means: a tree configured for a
#: plain shell through ``shared/bin/gacode_setup`` keeps working unchanged.
GACODE_ROOT_ENV = "GACODE_ROOT"

#: Selects ``platform/build/make.inc.$GACODE_PLATFORM`` at build time and
#: ``platform/exec/exec.$GACODE_PLATFORM`` at run time.  There is no default
#: that is right anywhere, so it is resolved explicitly and never guessed.
GACODE_PLATFORM_ENV = "GACODE_PLATFORM"

GACODE_COMPATIBILITY_ENVS: tuple[str, ...] = (GACODE_ROOT_ENV,)

#: Suite members that build from this tree.  Only ``neo`` has a VAFT adapter
#: today; the rest are listed because the runtime resolves any of them and
#: because issue #553 adds TGLF next.
SUITE_CODES: tuple[str, ...] = ("neo", "tglf", "cgyro")

#: Backends VAFT can actually prepare, run and parse.
SUPPORTED_CODES = frozenset({"neo"})


@dataclass(frozen=True)
class GACODEConfig(CodeConfig):
    """Runtime configuration shared by every GACODE backend.

    Subclasses :class:`vaft.code.base.CodeConfig`, so ``executable``,
    ``workdir``, ``args``, ``env`` and ``timeout`` mean what they mean
    everywhere else in :mod:`vaft.code`.  ``executable`` overrides the launcher
    path outright and is the escape hatch for an installation this resolution
    order does not describe.

    Attributes
    ----------
    home : str, optional
        Installation root.  Falls back to ``$GACODEHOME``, then to
        ``$GACODE_ROOT``.
    platform : str, optional
        Platform tag.  Falls back to ``$GACODE_PLATFORM``.  A wrong value fails
        inside a shell script without naming itself, which is why
        :func:`vaft.code.gacode.gacode_platform` resolves it up front and lists
        what the installation actually provides.
    n_mpi : int
        MPI tasks passed to the launcher's ``-n``.
    n_omp : int
        OpenMP threads passed to the launcher's ``-nomp``.
    """

    home: Optional[str] = None
    platform: Optional[str] = None
    n_mpi: int = 1
    n_omp: int = 1

    def __post_init__(self) -> None:
        if int(self.n_mpi) < 1:
            raise ValueError(f"n_mpi must be at least 1; got {self.n_mpi!r}")
        if int(self.n_omp) < 1:
            raise ValueError(f"n_omp must be at least 1; got {self.n_omp!r}")
