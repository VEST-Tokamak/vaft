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


def peq_jacobian(path: Path | str) -> str:
    """The Jacobian a ``gpec_xclebsch_n<n>.out`` was decomposed in, as ``jac_in``.

    GPEC writes ``jac_out = <word>`` into the file's own header
    (``gpec/gpout.f:5514``).  An empty word there means GPEC output in DCON's
    working coordinate system, which is exactly what PENTRC reads ``"default"``
    as (``pentrc/inputs.f90:631-632``) -- so that is what comes back, rather
    than the empty string, because ``"default"`` says in the namelist what is
    actually meant.

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
    return word or "default"


def _namelist_replacements(
    options: PENTRCOptions,
    *,
    kinetic_file: str,
    jac_in: str,
) -> dict[str, object]:
    """Every ``pentrc.in`` key this run sets, and what it sets it to."""
    from ..pentrc import TORQUE_METHODS

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
    for method in TORQUE_METHODS:
        replacements[f"{method}_flag"] = method in selected
    for name, flag in PENTRCOptions.GRID_FLAGS.items():
        replacements[flag] = name in set(options.grids)
    if options.psi_limits is not None:
        replacements["psilims"] = tuple(float(value) for value in options.psi_limits)
    if options.artificial_factors is not None:
        replacements.update(
            {name: float(value) for name, value in options.artificial_factors.items()}
        )
    return replacements


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
    kinetic = Path(kinetic_file).expanduser()
    if not kinetic.is_file():
        raise FileNotFoundError(f"kinetic profile file not found: {kinetic}")
    staged = run_dir / kinetic.name
    if staged.resolve() != kinetic.resolve():
        staged.write_bytes(kinetic.read_bytes())

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
    return rt.write_template(
        template,
        run_dir / "pentrc.in",
        _namelist_replacements(
            options,
            kinetic_file=staged.name,
            jac_in=peq_jacobian(displacement),
        ),
    )


def run_pentrc(
    run_dir: Path | str,
    *,
    mode: int,
    options: PENTRCOptions,
    kinetic_file: Path | str,
    config: GPECSuiteConfig | None = None,
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

    returncode, log_path = rt.run_subprocess(
        executable, run_dir, run_dir / "pentrc.log", config=config
    )
    output = run_dir / pentrc_output_name(mode)
    record.returncode = returncode
    record.commands = (str(executable),)
    record.logs = (log_path,)
    record.outputs = (output,) if output.is_file() else ()
    if returncode != 0:
        record.status = "failed"
        record.reason = f"pentrc exited {returncode}"
    elif not output.is_file():
        # Exit 0 with no netCDF is the `output_netcdf=f` case and nothing else,
        # and PENTRCOptions refuses both output forms off -- so it is a real
        # failure rather than a legitimate quiet run.
        record.status = "failed"
        record.reason = f"pentrc exited 0 but wrote no {output.name}"
    else:
        record.status = "completed"
    return record
