"""Staging of a NUBEAM run directory, and the ``inputf`` rewrite.

Nothing here shells out. The one text transformation NUBEAM staging needs --
pointing ``inputf`` at the equilibrium actually being run -- is done in Python
on purpose; see :func:`rewrite_inputf_equilibrium`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import re
import shutil
from typing import Optional, Sequence

from vaft.code.base import CodeInputs

from .config import NUBEAMConfig

#: ``inputf`` is positional, and ``plasma_state_test.f90`` reads it as::
#:
#:     read(10,*) time0,time1   ! 1
#:     read(10,*) fgname        ! 2  G-EQDSK
#:     read(10,*) fmname        ! 3  machine description
#:     read(10,*) fsname        ! 4  shot configuration
#:     read(10,*) fpsname       ! 5  Plasma State to write
#:     read(10,*) runID         ! 6
#:     read(10,*) fprofil       ! 7  profile mode
#:
#: These are 1-based line numbers, matching how the file reads.
INPUTF_EQUILIBRIUM_LINE = 2
INPUTF_STATE_LINE = 5
INPUTF_RUNID_LINE = 6

#: Files a NUBEAM run directory needs beyond the machine description.
REQUIRED_INPUTS = (
    "inputf",
    "profiles",
    "nubeam_init.dat",
    "nubeam_init_files.dat",
    "nubeam_step.dat",
    "nubeam_step_files.dat",
)


class NUBEAMInputError(ValueError):
    """Raised when a NUBEAM run cannot be staged from the given inputs."""


@dataclass
class NUBEAMInputs(CodeInputs):
    """A staged NUBEAM run directory."""

    inputf: Optional[Path] = None
    gfile: Optional[Path] = None
    plasma_state: Optional[Path] = None
    runid: str = "NUBEAM"
    manifests: tuple[Path, ...] = field(default_factory=tuple)


def _split_keeping_terminators(text: str) -> list[str]:
    return text.splitlines(keepends=True)


def rewrite_inputf_equilibrium(text: str, gfile_name: str) -> str:
    """Return *text* with the ``inputf`` G-EQDSK line pointing at *gfile_name*.

    Deliberately Python rather than ``sed``. The equivalent one-liner is not
    portable in either direction:

    * GNU ``sed`` spells a line replacement ``2c\\TEXT``; BSD ``sed`` requires
      the text on its own line after a backslash, and rejects the GNU form with
      "extra characters after \\ at the end of c command".
    * ``sed -i`` takes an optional backup suffix on BSD and none on GNU, so
      ``sed -i ''`` -- correct on macOS -- makes GNU ``sed`` read ``''`` as the
      script and the real script as a filename.

    ``server-smoke.sh`` uses the GNU spelling, which is why it is Linux-only.
    Doing it here instead means the adapter behaves identically on every
    platform and needs no external tool at all.

    Line terminators are preserved exactly, including CRLF and a missing final
    newline: NUBEAM reads this file with Fortran list-directed input, and the
    surrounding lines are not ours to reformat.
    """
    lines = _split_keeping_terminators(text)
    if len(lines) < INPUTF_EQUILIBRIUM_LINE:
        raise NUBEAMInputError(
            f"inputf has {len(lines)} lines; the G-EQDSK name is expected on "
            f"line {INPUTF_EQUILIBRIUM_LINE}"
        )

    index = INPUTF_EQUILIBRIUM_LINE - 1
    original = lines[index]
    # Keep whatever terminator the file already used.
    terminator = original[len(original.rstrip("\r\n")):]
    # Fortran list-directed input stops at the first blank, so the trailing
    # "! ..." on these lines is already ignored by the reader. Keep it: it is
    # what makes the staged file readable next to the original.
    lines[index] = f"{gfile_name}\t\t\t! EQDSK file{terminator}"
    return "".join(lines)


def _inputf_field(text: str, line_number: int, description: str) -> str:
    lines = text.splitlines()
    if len(lines) < line_number:
        raise NUBEAMInputError(
            f"inputf has {len(lines)} lines; {description} is expected on "
            f"line {line_number}"
        )
    value = lines[line_number - 1].split("!", 1)[0].strip()
    if not value:
        raise NUBEAMInputError(
            f"inputf line {line_number} is blank; expected {description}"
        )
    return value.split()[0]


def inputf_state_filename(text: str) -> str:
    """Name of the Plasma State ``plasma_state_test`` will write."""
    return _inputf_field(text, INPUTF_STATE_LINE, "the output Plasma State name")


def inputf_runid(text: str) -> str:
    """Run identifier declared by ``inputf``."""
    return _inputf_field(text, INPUTF_RUNID_LINE, "the run id")


def check_workdir_length(workdir: Path, config: NUBEAMConfig) -> None:
    """Refuse a work directory NUBEAM would silently truncate.

    Checked before anything runs, because the symptom otherwise appears much
    later and names the wrong cause -- see ``NUBEAM_PATH_BUFFER_CHARS``.
    """
    budget = config.workdir_budget
    actual = len(str(workdir))
    if actual <= budget:
        return
    raise NUBEAMInputError(
        f"The NUBEAM work directory is {actual} characters, which exceeds the "
        f"{budget} available for run id {config.runid!r}. NUBEAM composes every "
        f"filename in a {config.path_buffer_chars}-character Fortran buffer "
        "(`character*140 zfile` in `subroutine echo`, nubeam_comp_exec.F90), so "
        "a longer path is truncated with no diagnostic and the run fails later "
        "with a misleading file-open error. Use a shorter work directory, or a "
        f"shorter run id. Path was: {workdir}"
    )


def _apply_particle_count(namelist: Path, nptcls: int) -> None:
    """Rewrite ``nptcls``/``nptclf`` in a NUBEAM namelist.

    A targeted regular expression rather than f90nml: the shipped namelists
    carry extensive ``!!`` commentary that documents every knob, and a
    round-trip through a namelist writer would discard all of it.
    """
    if nptcls < 100:
        raise NUBEAMInputError(
            f"nubeam_comp_exec rejects nptcls < 100; got {nptcls}"
        )
    text = namelist.read_text(encoding="utf-8")
    pattern = re.compile(
        r"^(?P<indent>[ \t]*)(?P<key>nptcls|nptclf)(?P<gap>[ \t]*=[ \t]*)\d+",
        re.MULTILINE,
    )
    updated = pattern.sub(
        lambda m: f"{m.group('indent')}{m.group('key')}{m.group('gap')}{nptcls}",
        text,
    )
    namelist.write_text(updated, encoding="utf-8")


def prepare_nubeam_inputs(
    input_dir: str | Path,
    *,
    gfile: str | Path,
    workdir: str | Path,
    config: Optional[NUBEAMConfig] = None,
) -> NUBEAMInputs:
    """Stage a NUBEAM run directory from a case directory and an equilibrium.

    *input_dir* holds the case as NUBEAM expects it: ``inputf``, ``profiles``,
    the four ``nubeam_*.dat`` namelists, and one ``mdescr_*.dat`` /
    ``sconfig_*.dat`` pair describing the machine.
    """
    config = config or NUBEAMConfig()
    source = Path(input_dir).expanduser()
    equilibrium = Path(gfile).expanduser()
    target = Path(workdir).expanduser()

    if not source.is_dir():
        raise NUBEAMInputError(f"NUBEAM input directory does not exist: {source}")
    if not equilibrium.is_file():
        raise NUBEAMInputError(f"G-EQDSK file does not exist: {equilibrium}")

    missing = [name for name in REQUIRED_INPUTS if not (source / name).is_file()]
    if missing:
        raise NUBEAMInputError(
            f"NUBEAM input directory {source} is missing: {', '.join(missing)}"
        )

    descriptors: Sequence[Path] = sorted(
        [*source.glob("mdescr_*.dat"), *source.glob("sconfig_*.dat")]
    )
    if len(descriptors) < 2:
        raise NUBEAMInputError(
            f"No mdescr_*.dat / sconfig_*.dat machine description in {source}"
        )

    check_workdir_length(target, config)
    target.mkdir(parents=True, exist_ok=True)

    staged: list[Path] = []
    for name in REQUIRED_INPUTS:
        destination = target / name
        shutil.copy2(source / name, destination)
        staged.append(destination)
    for descriptor in descriptors:
        destination = target / descriptor.name
        shutil.copy2(descriptor, destination)
        staged.append(destination)

    # One fixed name, so the staged inputf does not depend on what the
    # equilibrium happened to be called upstream.
    staged_gfile = target / "equilibrium.gfile"
    shutil.copy2(equilibrium, staged_gfile)
    staged.append(staged_gfile)

    inputf = target / "inputf"
    original = inputf.read_text(encoding="utf-8")
    inputf.write_text(
        rewrite_inputf_equilibrium(original, staged_gfile.name), encoding="utf-8"
    )
    rewritten = inputf.read_text(encoding="utf-8")

    if config.nptcls is not None:
        _apply_particle_count(target / "nubeam_init.dat", config.nptcls)

    return NUBEAMInputs(
        workdir=target,
        files=tuple(staged),
        inputf=inputf,
        gfile=staged_gfile,
        plasma_state=target / inputf_state_filename(rewritten),
        runid=inputf_runid(rewritten),
    )
