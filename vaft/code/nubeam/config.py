"""Configuration and the fixed-width path budget for the NUBEAM adapter."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from vaft.code.base import CodeConfig

#: Environment variable naming the NUBEAM installation root.
NUBEAM_HOME_ENV = "NUBEAMHOME"

#: Serial NUBEAM, relative to ``$NUBEAMHOME``.
NUBEAM_HOME_EXECUTABLE = Path("bin/nubeam_comp_exec")

#: The Plasma State generator. The NTCC archive ships no main program for it;
#: this is built from ``plasma_state_test.f90`` by ``external/nubeam/macos.sh``.
NUBEAM_GENERATOR_EXECUTABLE = Path("bin/plasma_state_test")

#: Merges NUBEAM's own state-change output into a full Plasma State.
NUBEAM_UPDATE_STATE_EXECUTABLE = Path("bin/update_state")

#: Width of the buffer every NUBEAM filename passes through.
#:
#: ``subroutine echo`` in ``nubeam_comp_exec.F90`` -- whose declared job is to
#: "prepend workpath if necessary" -- composes each input and output filename
#: in a local ``character*140 zfile``. A longer result is truncated with no
#: diagnostic, and the run then fails wherever the truncated name fails to
#: open. In practice that is
#: ``?plasma_state_get: file open failure: "<truncated path>"``, which names
#: the input state rather than the path length and sends you looking in
#: entirely the wrong place.
#:
#: Note that ``workpath`` itself is declared ``character*200``; 140 is the
#: narrower buffer downstream and is the one that actually binds.
NUBEAM_PATH_BUFFER_CHARS = 140

#: The longest filename NUBEAM appends to the run id.
#:
#: ``<runid>_debug_nbi_ptcl_state_cpu0_1.cdf``, written on every step while
#: ``nltest_output = 1`` -- which the namelist shipped with the distribution
#: sets. Budgeting against the shorter non-debug name would leave a run that
#: works until someone enables test output.
NUBEAM_LONGEST_OUTPUT_SUFFIX = "_debug_nbi_ptcl_state_cpu0_1.cdf"


def workdir_budget(
    runid: str, *, buffer_chars: int = NUBEAM_PATH_BUFFER_CHARS
) -> int:
    """Longest work-directory path NUBEAM can be handed for this run id.

    ``zfile`` has to hold the work directory, the separator NUBEAM appends to
    it, and the longest filename derived from *runid*::

        len(workdir) + 1 + len(runid) + len(suffix)  <=  buffer_chars

    The budget therefore shrinks as the run id grows, which is why this is a
    function of *runid* rather than one constant.
    """
    return buffer_chars - 1 - len(runid) - len(NUBEAM_LONGEST_OUTPUT_SUFFIX)


@dataclass(frozen=True)
class NUBEAMConfig(CodeConfig):
    """Runtime configuration for a NUBEAM run.

    The work directory is passed to :func:`prepare_nubeam_inputs`, not carried
    here, and is subject to :func:`workdir_budget`. For a scratch run, wrap the
    call in :func:`vaft.compat.short_temporary_directory` with
    ``max_length=config.workdir_budget``.
    """

    #: Run identifier. NUBEAM derives every output filename from it, so it
    #: also consumes part of the path budget above.
    runid: str = "NUBEAM"

    #: ``NUBEAM_REPEAT_COUNT``: ``<count>x<step seconds>``.
    repeat_count: str = "1x0.001"

    #: ``NUBEAM_POSTPROC``. ``summary_test`` adds printed diagnostics that the
    #: Plasma State output does not carry.
    postproc: str = "summary_test"

    #: Run the FRANTIC neutral-transport model (``FRANTIC_INIT`` at init,
    #: ``FRANTIC_ACTION`` at step), matching the shipped reference cases.
    frantic: bool = True

    #: Number of gas-grid zones passed as ``FRANTIC_INIT``.
    frantic_zones: int = 50

    #: PREACT reaction tables and ADAS data. ``nubeam_comp_exec`` calls
    #: ``bad_exit`` when either ``PREACTDIR`` or ``ADASDIR`` is blank, so these
    #: are required, not optional. Both default to ``$NUBEAMHOME/share/*``.
    preact_dir: Optional[Path] = None
    adas_dir: Optional[Path] = None

    #: Overrides for the two executables the adapter needs besides NUBEAM.
    generator_executable: Optional[str] = None
    update_state_executable: Optional[str] = None

    #: Monte Carlo particle count override, applied to ``nptcls``/``nptclf``.
    #: ``None`` keeps whatever the supplied namelist sets.
    nptcls: Optional[int] = None

    #: Buffer width to budget against. Exposed only so a future NUBEAM build
    #: with a wider buffer does not need a code change here.
    path_buffer_chars: int = NUBEAM_PATH_BUFFER_CHARS

    @property
    def workdir_budget(self) -> int:
        """Longest permissible work-directory path for this configuration."""
        return workdir_budget(self.runid, buffer_chars=self.path_buffer_chars)
