"""Running PENTRC in a completed ideal-GPEC cell.

PENTRC is the GPEC suite's neoclassical toroidal viscous torque calculation.
:mod:`vaft.code.pentrc` reads what it produced; this is what produces it.

**Where it runs, and why not as a fifth suite module.**  PENTRC needs three
inputs at once: DCON's ``euler.bin``, the ``gpec_xclebsch_n<n>.out`` displacement
ideal GPEC writes, and the ``.kin`` profiles.  The ideal-GPEC cell is the only
directory in VAFT's per-module layout that holds all three -- ``euler.bin`` is
brought there by :func:`~vaft.code.gpec.stage_dcon_products`, because GPEC reads
its own vacuum handshake files out of ``dcon_dir`` -- so PENTRC runs there, the
way GPEC's own examples run it.  A ``pentrc`` module directory would have to
re-stage from two other cells, and :class:`~vaft.code.gpec._solvers.Solver`'s
``companion_executables()`` takes no config, so PENTRC cannot be made a
*conditional* companion of ``gpec`` without changing that protocol.

**The one convention that has to be got right.**  ``pentrc.in``'s ``jac_in``
must be the Jacobian the displacement file was decomposed in.  That is GPEC's
``jac_out``, which is not necessarily DCON's ``jac_type``, and PENTRC reads
``""`` or ``"default"`` as DCON's (``pentrc/inputs.f90:631-632``).  The
displacement file states its own in its header (``gpec/gpout.f:5514``), so
:func:`peq_jacobian` reads it there and :func:`prepare_pentrc_run` writes what
it found.  The legacy runner hard-coded ``jac_in="hamada"``, which is correct
only while ``equil.in`` says ``jac_type='hamada'`` *and* ``gpec.in`` leaves
``jac_out`` empty -- two files away from the one being written.

Typical use, after :func:`~vaft.code.gpec.run_gpec_suite_case` has completed::

    from vaft.code.gpec import PENTRCOptions, run_pentrc
    from vaft.code import pentrc

    record = run_pentrc(
        gpec_cell,
        mode=1,
        options=PENTRCOptions(
            methods=("fgar", "tgar"),
            main_ion="deuterium",
            impurity="carbon",
            collision_operator="harmonic",
        ),
        kinetic_file="profiles.kin",
        config=config,
    )
    with pentrc.read_pentrc_output(record.outputs[0]) as run:
        psi_norm, torque = pentrc.torque_profile(run, "fgar")
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

from ...compat import is_executable
from . import _runtime as rt
from ._types import GPECModuleRun, GPECSuiteConfig, PENTRCOptions

#: The module name PENTRC records itself under in a :class:`GPECModuleRun`.
#:
#: Deliberately not added to :data:`~vaft.code.gpec.SUPPORTED_MODULES`: PENTRC is
#: not a suite module here (see this module's docstring), and putting the name
#: there would make ``modules=("dcon", "gpec", "pentrc")`` look supported by
#: :func:`~vaft.code.gpec.prepare_gpec_suite_case`, which would then try to give
#: it a directory of its own.
PENTRC_MODULE = "pentrc"


def pentrc_output_name(mode: int) -> str:
    """PENTRC's netCDF output for one toroidal mode (``pentrc/torque.F90:1905``)."""
    return f"pentrc_output_n{int(mode)}.nc"


#: Every Jacobian word GPEC and PENTRC know.
#:
#: GPEC's own set is ``equil/equil.f:108-131``; PENTRC adds ``polar``
#: (``pentrc/inputs.f90:404-418``).  Both stop the program on anything else, so
#: this is the whole vocabulary rather than a convenience list.
JACOBIAN_NAMES: tuple[str, ...] = (
    "hamada",
    "pest",
    "equal_arc",
    "boozer",
    "park",
    "polar",
    "other",
)


def _resolve_jacobian_word(word: str, path: Path) -> str:
    """Undo the header's field width, or refuse a word nothing recognises.

    GPEC writes the Jacobian with ``(1/,1x,a13,a8)`` (``gpec/gpout.f:5762``), so
    the word is **truncated to eight characters**: ``equal_arc`` -- the one name
    in :data:`JACOBIAN_NAMES` that is longer -- reaches the file as
    ``equal_ar``, and PENTRC stops on that because its own ``SELECT CASE`` has no
    such branch.  Resolved by prefix against the known set, and only when the
    prefix picks out exactly one name.
    """
    if word in JACOBIAN_NAMES:
        return word
    candidates = [name for name in JACOBIAN_NAMES if name.startswith(word)]
    if len(candidates) == 1:
        return candidates[0]
    raise ValueError(
        f"{path} states jac_out={word!r}, which is not a Jacobian GPEC or PENTRC "
        f"knows ({', '.join(JACOBIAN_NAMES)})"
        + (
            f" and is a prefix of {len(candidates)} of them ({', '.join(candidates)}), "
            "so the eight-character header field cannot be undone"
            if candidates
            else ""
        )
    )


def peq_jacobian(path: Path | str) -> str:
    """The Jacobian a ``gpec_xclebsch_n<n>.out`` was decomposed in, as ``jac_in``.

    GPEC writes ``jac_out = <word>`` into the file's own header
    (``gpec/gpout.f:5760-5762``).  An empty word there means GPEC output in DCON's
    working coordinate system, which is exactly what PENTRC reads ``"default"``
    as (``pentrc/inputs.f90:631-632``) -- so that is what comes back, rather
    than the empty string, because ``"default"`` says in the namelist what is
    actually meant.

    **The header truncates.** The format is ``a8``, so ``equal_arc`` arrives as
    ``equal_ar`` and PENTRC would stop on it; the word is resolved against
    :data:`JACOBIAN_NAMES` before it is returned.

    Parameters
    ----------
    path : Path
        A ``gpec_xclebsch_n<n>.out`` written by ideal GPEC [-].

    Returns
    -------
    str
        The ``jac_in`` value to write, e.g. ``"hamada"`` or ``"default"`` [-].

    Raises
    ------
    ValueError
        The file carries no ``jac_out`` header line, so nothing here knows what
        coordinates its harmonics are in.  Guessing would be the legacy defect
        this function exists to remove.
    """
    path = Path(path)
    # The header is four short lines; the table behind it is megabytes.
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        head = "".join(line for _, line in zip(range(12), handle))
    match = re.search(r"^\s*jac_out\s*=\s*(\S*)\s*$", head, re.MULTILINE)
    if match is None:
        raise ValueError(
            f"{path} carries no 'jac_out =' header line, so the coordinate system its "
            "harmonics are in is unknown; PENTRC's jac_in cannot be derived from it. "
            "GPEC writes that line for every gpec_xclebsch output (gpec/gpout.f:5514)"
        )
    word = match.group(1).strip().strip("\"'")
    return _resolve_jacobian_word(word, path) if word else "default"


#: What ``jsurf_in`` must be for a ``gpec_xclebsch`` displacement.
#:
#: Zero, always, and **not** ``gpec.in``'s ``jsurf_out``: the displacement is
#: written through ``gpeq_bcoordsout`` without the optional ``ji`` argument, and
#: that routine hard-codes ``jout = 0`` (``gpec/gpeq.f:830-834``).  So a run with
#: ``jsurf_out=1`` -- which appends an area-weighted *control* output -- still
#: writes an unweighted displacement, and a ``pentrc.in`` that copied
#: ``jsurf_out`` would tell PENTRC the file is a flux when it is a field.
PEQ_JSURF_IN = 0

#: The ``gpec.in`` key whose value the displacement's toroidal angle follows.
#:
#: ``gpeq_bcoordsout`` takes ``tout = tmag_out`` when the caller passes no
#: ``ti``, which is how ``gpout_xclebsch`` calls it
#: (``gpec/gpout.f:5738-5740``, ``gpec/gpeq.f:826-829``).  GPEC's own code
#: default is 1 (``gpec/gpec.f:136``).
PEQ_TMAG_SOURCE_KEY = "tmag_out"
PEQ_TMAG_DEFAULT = 1


def peq_toroidal_angle(run_dir: Path | str) -> int:
    """``tmag_in`` for the displacement in *run_dir*, read off its own ``gpec.in``.

    Derived rather than defaulted, for the same reason ``jac_in`` is read off the
    file's header: the convention belongs to the run that wrote the displacement,
    and a ``pentrc.in`` that assumed the common value would transform the
    harmonics under the wrong angle without saying so.

    Raises
    ------
    FileNotFoundError
        No ``gpec.in`` beside the displacement, so the angle the file is in is
        not recoverable.  Guessing the usual value is the shape of defect this
        module exists to remove.
    ValueError
        The ``gpec.in`` has no ``&GPEC_OUTPUT`` group, for the same reason: a
        ``tmag_out`` sitting in some other group is not the one GPEC read.
    """
    namelist = Path(run_dir) / "gpec.in"
    if not namelist.is_file():
        raise FileNotFoundError(
            f"no gpec.in in {run_dir}, so the toroidal angle its "
            "gpec_xclebsch displacement is written in is unknown; PENTRC's "
            f"tmag_in cannot be derived. GPEC takes it from {PEQ_TMAG_SOURCE_KEY} "
            "(gpec/gpeq.f:826-829)"
        )
    written = rt.read_namelist_group(namelist, "gpec_output").get(PEQ_TMAG_SOURCE_KEY)
    if written is None:
        return PEQ_TMAG_DEFAULT
    return int(str(written).strip())


#: Torque methods PENTRC runs unless the namelist turns them off.
#:
#: ``fgar_flag=.true.`` is PENTRC's own code default
#: (``pentrc/pentrc_interface.f90:62``); every other ``<method>_flag`` defaults
#: to ``.false.`` (``:63-80``).  So a template that lacks ``fgar_flag`` cannot
#: express a run *without* FGAR, and the key is required even when the method
#: is not selected, while a method that is off by default is simply not written
#: into a template that does not declare it.
PENTRC_METHODS_ON_BY_DEFAULT: tuple[str, ...] = ("fgar",)


def _method_flags(selected: set[str]) -> tuple[dict[str, bool], dict[str, bool]]:
    """The ``<method>_flag`` assignments, split into required and optional.

    Every method is written where the template declares it, so the namelist
    beside a result states the whole selection.  Required are the selected
    ones -- a method asked for that the template cannot name is a run that
    does not do what it says -- and the ones PENTRC runs by default, because
    leaving those undeclared is a silent extra calculation.  The rest are
    GPEC's own defaults restated: GPEC's ``input/pentrc.in`` declares 9 of the
    18, and a template is entitled to leave the others out (cold review 0.8.0
    delta-squash F2).
    """
    from ..pentrc import TORQUE_METHODS

    required: dict[str, bool] = {}
    optional: dict[str, bool] = {}
    for method in TORQUE_METHODS:
        wanted = method in selected
        target = required if wanted or method in PENTRC_METHODS_ON_BY_DEFAULT else optional
        target[f"{method}_flag"] = wanted
    return required, optional


def _namelist_replacements(
    options: PENTRCOptions,
    *,
    kinetic_file: str,
    jac_in: str,
    tmag_in: int,
) -> tuple[dict[str, object], dict[str, object]]:
    """Every ``pentrc.in`` key this run sets, as ``(required, optional)``.

    *required* must be in the template; *optional* is written where the
    template declares the key and skipped where it does not -- see
    :func:`_method_flags`.
    """
    selected = {str(name) for name in options.methods}
    replacements: dict[str, object] = {
        "kinetic_file": kinetic_file,
        # Left empty on purpose: PENTRC then names it
        # `gpec_xclebsch_n<nn>.out` from the mode it read off euler.bin
        # (`pentrc/pentrc_interface.f90:237-239`), so it cannot be pointed at
        # another mode's displacement by a stale namelist.
        "peq_file": "",
        "idconfile": "euler.bin",
        "data_dir": options.data_dir,
        "jac_in": jac_in,
        # The other two halves of the same contract as `jac_in`: which angle the
        # harmonics are in, and whether they are area-weighted.
        "tmag_in": int(tmag_in),
        "jsurf_in": PEQ_JSURF_IN,
        "nl": int(options.bounce_harmonics),
        "electron": bool(options.electron),
        "nutype": options.collision_operator,
        "moment": options.moment,
        "pentrc_threads": int(options.threads),
        "output_ascii": bool(options.output_ascii),
        "output_netcdf": bool(options.output_netcdf),
    }
    replacements.update(options.species_namelist)
    # Every method, not only the requested ones: the packaged template ships
    # them all off, and writing the whole set means the namelist beside a result
    # states the entire selection.
    method_flags, optional = _method_flags(selected)
    replacements.update(method_flags)
    for name, flag in PENTRCOptions.GRID_FLAGS.items():
        replacements[flag] = name in set(options.grids)
    if options.psi_limits is not None:
        replacements["psilims"] = tuple(float(value) for value in options.psi_limits)
    if options.artificial_factors is not None:
        replacements.update(
            {name: float(value) for name, value in options.artificial_factors.items()}
        )
    return replacements, optional


def validate_pentrc_inputs(run_dir: Path | str, mode: int) -> list[str]:
    """Describe what stops PENTRC running in ``run_dir``.

    Returns an empty list when it can run.  Every reason names the file and what
    writes it, because each missing input points at a different earlier step:
    ``euler.bin`` at DCON, the displacement at ideal GPEC's ``xclebsch_flag``
    *and* its ``ascii_flag`` (PENTRC reads the ASCII table, not the netCDF), and
    ``pentrc.in`` at :func:`prepare_pentrc_run`.

    Parameters
    ----------
    run_dir : Path
        The ideal-GPEC cell PENTRC would run in [-].
    mode : int
        Toroidal mode number, which names the displacement file [-].

    Returns
    -------
    list of str
        One reason per missing prerequisite [-].
    """
    run_dir = Path(run_dir)
    problems: list[str] = []
    if not (run_dir / "pentrc.in").is_file():
        problems.append(
            f"missing pentrc.in in {run_dir}: PENTRC reads it from its working "
            "directory (pentrc/pentrc_interface.f90:205)"
        )
    if not (run_dir / "euler.bin").is_file():
        problems.append(
            "missing euler.bin: DCON's eigenfunctions, brought to the ideal-GPEC cell "
            "by stage_dcon_products"
        )
    displacement = run_dir / f"gpec_xclebsch_n{int(mode)}.out"
    if not displacement.is_file():
        problems.append(
            f"missing {displacement.name}: ideal GPEC writes it under xclebsch_flag "
            "*and* ascii_flag (gpec/gpout.f:5508-5511), and PENTRC reads the ASCII "
            "table rather than the netCDF one"
        )
    return problems


def prepare_pentrc_run(
    run_dir: Path | str,
    *,
    mode: int,
    options: PENTRCOptions,
    kinetic_file: Path | str,
    config: GPECSuiteConfig | None = None,
) -> Path:
    """Write ``pentrc.in`` into a completed ideal-GPEC cell, and return it.

    The kinetic profiles are staged into *run_dir* if they are not already
    there, because PENTRC resolves ``kinetic_file`` relative to its working
    directory and a namelist naming a path outside the cell makes the result
    unreproducible once the tree is moved.

    Parameters
    ----------
    run_dir : Path
        A completed ideal-GPEC cell [-].
    mode : int
        Toroidal mode number [-].
    options : PENTRCOptions
        What to compute, and for which plasma [-].
    kinetic_file : Path
        A GPEC ``.kin``; write one with
        :func:`vaft.data.kinetic_profiles.write_kin` [-].
    config : GPECSuiteConfig, optional
        Only its ``templates_dir`` is used here, to locate ``pentrc.in``;
        defaults to the packaged template [-].

    Returns
    -------
    Path
        The ``pentrc.in`` that was written [-].

    Raises
    ------
    FileNotFoundError
        *run_dir* or *kinetic_file* is not there.
    ValueError
        The displacement file for *mode* does not state its Jacobian, so
        ``jac_in`` cannot be derived -- see :func:`peq_jacobian`.
    """
    run_dir = Path(run_dir)
    if not run_dir.is_dir():
        raise FileNotFoundError(f"ideal-GPEC cell not found: {run_dir}")
    staged = _stage_kinetic_file(run_dir, kinetic_file)
    displacement = run_dir / f"gpec_xclebsch_n{int(mode)}.out"
    if not displacement.is_file():
        raise FileNotFoundError(
            f"{displacement} is not there, so its Jacobian cannot be read and PENTRC "
            "has no displacement to integrate. Run ideal GPEC for this mode with "
            "xclebsch_flag and ascii_flag on first"
        )
    template = rt.template_dir(config or GPECSuiteConfig()) / "pentrc.in"
    if not template.is_file():
        raise FileNotFoundError(f"pentrc.in template not found: {template}")
    # What is about to be overwritten, kept once. On a threshold run this file is
    # what GPEC's own threshold models read, and the species and profiles it names
    # are not recoverable from anything else in the cell.
    existing = run_dir / "pentrc.in"
    prior = run_dir / PRIOR_PENTRC_INPUT
    if existing.is_file() and not prior.is_file():
        prior.write_bytes(existing.read_bytes())
    replacements, optional = _namelist_replacements(
        options,
        kinetic_file=staged.name,
        jac_in=peq_jacobian(displacement),
        tmag_in=peq_toroidal_angle(run_dir),
    )
    return rt.write_template(template, existing, replacements, optional=optional)


#: Log lines that mean PENTRC stopped without finishing, with exit status 0.
#:
#: Fortran's ``STOP "ERROR: ..."`` prints the string and exits **0** under
#: gfortran -- only an integer stop-code sets the status -- and PENTRC uses that
#: form for every quadrature it gives up on (``pentrc/energy.f90:225,280``,
#: ``pentrc/pitch.f90:279``).  So the return code cannot tell such a run from a
#: clean one, and with a stale output file beside it neither can the directory.
PENTRC_ERROR_LINE = re.compile(r"^\s*ERROR\b")

#: Log lines that mean an integration did not converge but PENTRC carried on.
#:
#: LSODE reports a repeated corrector failure and PENTRC checks only
#: ``istate == -1`` (``pentrc/energy.f90:224``), so ``istate == -5`` leaves the
#: run going with a result nothing has vouched for.  Counted onto the record
#: rather than turned into a verdict: whether such a profile is usable is a
#: physics judgement, and the alternative -- silence -- is what hid it.
PENTRC_NONCONVERGENCE_LINE = re.compile(
    r"(corrector convergence failed|too many steps)", re.IGNORECASE
)

#: How many non-convergence lines to count before reporting "at least".
#:
#: A diverging run writes the same line millions of times; the count is for a
#: reader, not for a histogram.
PENTRC_NONCONVERGENCE_SCAN_LIMIT = 1000


def scan_pentrc_log(path: Path | str) -> tuple[str | None, int]:
    """The first fatal line in a PENTRC log, and how many non-convergence lines.

    Streamed rather than read, because a log of a diverging run is hundreds of
    megabytes of one repeated message.  Counting stops at
    :data:`PENTRC_NONCONVERGENCE_SCAN_LIMIT`; the caller says "at least" from
    there.

    Returns
    -------
    tuple
        ``(fatal_line, nonconvergence_count)``; the first element is ``None``
        when the log carries no ``ERROR`` line [-].
    """
    fatal: str | None = None
    warnings = 0
    path = Path(path)
    if not path.is_file():
        return None, 0
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if fatal is None and PENTRC_ERROR_LINE.search(line):
                fatal = line.strip()
            elif PENTRC_NONCONVERGENCE_LINE.search(line):
                if warnings < PENTRC_NONCONVERGENCE_SCAN_LIMIT:
                    warnings += 1
                elif fatal is not None:
                    break
    return fatal, warnings


#: Where a ``pentrc.in`` that GPEC's threshold models read is kept.
#:
#: :func:`prepare_pentrc_run` writes the torque run's own ``pentrc.in`` over the
#: top -- same file name, different contents -- so without this copy the input the
#: thresholds were computed from is gone, and with it which species and which
#: profiles they used.
THRESHOLD_PENTRC_INPUT = "pentrc_threshold.in"

#: Where an existing ``pentrc.in`` that :func:`prepare_pentrc_run` replaced is kept.
#:
#: The first one only: a second torque run would otherwise overwrite the record
#: with its own predecessor, which is the thing being preserved.
PRIOR_PENTRC_INPUT = "pentrc_prior.in"


def write_threshold_pentrc_input(
    run_dir: Path | str,
    *,
    options: PENTRCOptions,
    kinetic_file: Path | str,
    config: GPECSuiteConfig | None = None,
) -> Path:
    """Write the ``pentrc.in`` **ideal GPEC** reads for a penetration threshold.

    Not the torque run's file.  When ``singthresh_callen_flag`` or
    ``singthresh_slayer_flag`` is on, GPEC calls PENTRC's initialiser itself --
    ``initialize_pentrc(op_kin=.TRUE., op_deq=.TRUE., op_peq=.FALSE.)``
    (``gpec/gpec.f:571-573``) -- which reads ``pentrc.in`` out of the working
    directory for the **profiles and the species** and not for the displacement.
    So this file has to exist *before* GPEC runs, when no displacement has been
    written yet and :func:`prepare_pentrc_run` cannot run at all.

    The species are not decoration here: Callen's critical width is built from the
    ion gyroradius, ``sqrt(2 T / (mi mp)) / (zi e bt0 / (mi mp))``
    (``gpec/gpout.f:1751-1753``), so ``mi`` and ``zi`` are in the threshold.

    Every torque method is written **off**.  GPEC's threshold path runs none of
    them, and a namelist left claiming a selection would describe a torque
    calculation that never happened.  A copy is kept as
    :data:`THRESHOLD_PENTRC_INPUT`, because the torque run overwrites ``pentrc.in``.

    Parameters
    ----------
    run_dir : Path
        The ideal-GPEC cell that will run, already prepared [-].
    options : PENTRCOptions
        Read for the species, the collision operator and ``data_dir``; its
        ``methods`` are not written on [-].
    kinetic_file : Path
        A GPEC ``.kin``, staged into the cell the way PENTRC resolves it [-].
    config : GPECSuiteConfig, optional
        Only its ``templates_dir`` is used, to locate ``pentrc.in`` [-].

    Returns
    -------
    Path
        The ``pentrc.in`` that was written [-].

    Raises
    ------
    FileNotFoundError
        *run_dir* or *kinetic_file* is not there, or no ``pentrc.in`` template.
    """
    run_dir = Path(run_dir)
    if not run_dir.is_dir():
        raise FileNotFoundError(f"ideal-GPEC cell not found: {run_dir}")
    staged = _stage_kinetic_file(run_dir, kinetic_file)
    template = rt.template_dir(config or GPECSuiteConfig()) / "pentrc.in"
    if not template.is_file():
        raise FileNotFoundError(f"pentrc.in template not found: {template}")

    replacements: dict[str, object] = {
        "kinetic_file": staged.name,
        "idconfile": "euler.bin",
        "data_dir": options.data_dir,
        "nutype": options.collision_operator,
        "nl": int(options.bounce_harmonics),
    }
    replacements.update(options.species_namelist)
    # None selected: only the flag PENTRC turns on by itself has to be in the
    # template; the others are restated where declared.
    method_flags, optional = _method_flags(set())
    replacements.update(method_flags)
    written = rt.write_template(template, run_dir / "pentrc.in", replacements, optional=optional)
    (run_dir / THRESHOLD_PENTRC_INPUT).write_bytes(written.read_bytes())
    return written


def _stage_kinetic_file(run_dir: Path, kinetic_file: Path | str) -> Path:
    """Put the profiles beside the namelist that names them, and return them.

    PENTRC resolves ``kinetic_file`` relative to its working directory, so a
    namelist naming a path outside the cell makes the result unreproducible once
    the tree is moved.
    """
    kinetic = Path(kinetic_file).expanduser()
    if not kinetic.is_file():
        raise FileNotFoundError(f"kinetic profile file not found: {kinetic}")
    staged = run_dir / kinetic.name
    if staged.resolve() != kinetic.resolve():
        staged.write_bytes(kinetic.read_bytes())
    return staged


def run_pentrc(
    run_dir: Path | str,
    *,
    mode: int,
    options: PENTRCOptions,
    kinetic_file: Path | str,
    config: GPECSuiteConfig | None = None,
    allow_unbounded_runtime: bool = False,
) -> GPECModuleRun:
    """Prepare and run PENTRC in a completed ideal-GPEC cell.

    Parameters
    ----------
    run_dir : Path
        A completed ideal-GPEC cell [-].
    mode : int
        Toroidal mode number [-].
    options : PENTRCOptions
        What to compute, and for which plasma [-].
    kinetic_file : Path
        A GPEC ``.kin`` [-].
    config : GPECSuiteConfig, optional
        Where the executable and the ``pentrc.in`` template come from, plus the
        subprocess timeout and environment.  ``run_mode`` is honoured the way
        the suite honours it: ``prepare_only`` writes the namelist and stops,
        ``strict`` raises where the others report [-].
    allow_unbounded_runtime : bool, optional
        Permit ``config.timeout=None``.  Off by default: a diverging PENTRC
        prints rather than stops, so an unbounded run reports nothing and grows a
        log without limit.  Pass it when a scheduler bounds the process [-].

    Returns
    -------
    GPECModuleRun
        ``module="pentrc"``, with the status, the log and the output found.
        ``status`` is ``completed`` on success, ``prepared`` under
        ``prepare_only``, ``skipped`` when PENTRC is not installed or an input
        is missing, and ``failed`` when it ran and did not produce its output [-].

    Raises
    ------
    FileNotFoundError, RuntimeError
        Under ``run_mode="strict"``, for a missing executable and a missing
        input respectively.
    """
    run_dir = Path(run_dir)
    config = config or GPECSuiteConfig()
    policy = rt.run_policy(config)
    record = GPECModuleRun(PENTRC_MODULE, int(mode), run_dir)
    if policy != "prepare_only" and config.timeout is None and not allow_unbounded_runtime:
        # Refused rather than honoured, because this module has no other brake.
        # PENTRC does not stop when an integration will not converge -- it prints
        # -- so `timeout=None` is not "run as long as it takes", it is "fill the
        # disk and never report". A scheduler with its own walltime is the case
        # the opt-in exists for.
        raise ValueError(
            "run_pentrc needs a timeout: config.timeout is None, and a PENTRC whose "
            "integration diverges neither stops nor fails -- it writes the same LSODE "
            "line until something else kills it (measured: 223 MB in 300 s). Set "
            "GPECSuiteConfig(timeout=...), or pass allow_unbounded_runtime=True if "
            "something outside this process bounds the run"
        )

    # The executable is resolved *before* the namelist is written, so a run that
    # cannot happen leaves the directory as it was. That matters here more than
    # it does for a suite module: the `pentrc.in` being overwritten is the one
    # GPEC's own penetration thresholds read (`gpec/gpec.f:620-623`), so
    # clobbering it on a skipped run would destroy the record of what those
    # thresholds were computed from.
    executable = None
    if policy != "prepare_only":
        executable = rt.optional_executable(config, PENTRC_MODULE)
        if executable is None or not is_executable(executable):
            reason = (
                rt.unconfigured_reason()
                if executable is None
                else f"missing or non-executable pentrc: {executable}"
            )
            if policy == "strict":
                raise FileNotFoundError(reason)
            record.status = "skipped"
            record.reason = reason
            return record

    # The displacement is checked before the namelist is written, for the same
    # reason the executable is: `prepare_pentrc_run` needs it for `jac_in` and
    # raises without it, which under `prepare_only` / `run_if_available` would
    # be an exception where the docstring promises `skipped` -- after the kin
    # file had already been staged into the cell (cold review 0.8.0
    # delta-squash F4). `strict` keeps the raise, from the preparation itself.
    if policy != "strict":
        missing = [
            reason for reason in validate_pentrc_inputs(run_dir, mode)
            if f"gpec_xclebsch_n{int(mode)}.out" in reason
        ]
        if missing:
            record.status = "skipped"
            record.reason = f"cannot run PENTRC in {run_dir}: {'; '.join(missing)}"
            return record

    prepare_pentrc_run(
        run_dir, mode=mode, options=options, kinetic_file=kinetic_file, config=config
    )
    if policy == "prepare_only":
        record.reason = "run_mode=prepare_only"
        return record

    problems = validate_pentrc_inputs(run_dir, mode)
    if problems:
        reason = f"cannot run PENTRC in {run_dir}: {'; '.join(problems)}"
        if policy == "strict":
            raise RuntimeError(reason)
        record.status = "skipped"
        record.reason = reason
        return record

    output = run_dir / pentrc_output_name(mode)
    # Removed before the launch, not checked after it. PENTRC's own failure exit
    # is `STOP "ERROR: ..."`, which gfortran gives status 0, so a stale netCDF
    # from an earlier run of the same cell would be reported as this run's output
    # with this run's status. Deleting it makes the file's presence mean what the
    # check below assumes it means.
    if output.exists():
        output.unlink()
    log_path = run_dir / "pentrc.log"
    try:
        returncode, log_path = rt.run_subprocess(
            executable, run_dir, log_path, config=config
        )
    except subprocess.TimeoutExpired as expired:
        # A timeout is a `failed` record, not an exception. PENTRC does not stop
        # when its integration will not converge -- it prints: LSODE's energy
        # corrector failing repeatedly wrote 223 MB of one repeated message in
        # the 300 s it was given, on a real case. The caller needs that in the run's
        # record beside the runs that worked, and there is no partial output to
        # salvage: the netCDF is written at the end.
        record.status = "failed"
        record.returncode = None
        record.commands = (str(executable),)
        record.logs = (log_path,) if log_path.is_file() else ()
        log_bytes = log_path.stat().st_size if log_path.is_file() else 0
        if isinstance(expired, rt.GPECLimitStop) and not expired.is_time_limit:
            # A memory stop or a launch never admitted is not "did not finish within".
            record.reason = f"{expired.reason}; its log is {log_bytes} bytes"
        else:
            record.reason = f"pentrc did not finish within {expired.timeout:g} s; its log is {log_bytes} bytes"
        if policy == "strict":
            raise
        return record
    record.returncode = returncode
    record.commands = (str(executable),)
    record.logs = (log_path,)
    record.outputs = (output,) if output.is_file() else ()
    fatal, nonconvergence = scan_pentrc_log(log_path)
    unconverged = (
        ""
        if not nonconvergence
        else (
            f"; LSODE reported non-convergence on "
            + (
                f"at least {nonconvergence}"
                if nonconvergence >= PENTRC_NONCONVERGENCE_SCAN_LIMIT
                else str(nonconvergence)
            )
            + " line(s), and PENTRC checks only istate==-1 so the run went on"
        )
    )
    if returncode != 0:
        record.status = "failed"
        record.reason = f"pentrc exited {returncode}{unconverged}"
    elif fatal is not None:
        # Exit 0 and a fatal line: `STOP "ERROR: ..."` is status 0 under
        # gfortran, so the log is the only place this run says it stopped.
        record.status = "failed"
        record.reason = (
            f"pentrc exited 0 and stopped on {fatal!r} ({log_path.name}); a Fortran "
            f"STOP with a character code exits 0{unconverged}"
        )
        # Whatever is in the cell is not this run's product.
        record.outputs = ()
    elif not output.is_file():
        # Exit 0 with no netCDF is the `output_netcdf=f` case and nothing else,
        # and PENTRCOptions refuses both output forms off -- so it is a real
        # failure rather than a legitimate quiet run.
        record.status = "failed"
        record.reason = f"pentrc exited 0 but wrote no {output.name}{unconverged}"
    else:
        record.status = "completed"
        # Reported on a *completed* record rather than turned into a failure:
        # LSODE can complain and recover, and whether the profile it then wrote
        # is usable is a physics judgement rather than this function's.
        record.reason = unconverged.lstrip("; ")
    return record
