"""Per-solver namelist templating, output discovery, and success checks.

Each solver owns exactly the parts of the suite that actually differ between
DCON/RDCON/STRIDE/GPEC: which namelist(s) it writes, which companion
executable (if any) chains after it, which files it produces, and what
"the run actually worked" means for its output. The shared plumbing in
``_runtime.py`` (subprocess execution, env, executable resolution, directory
layout) stays solver-agnostic and is not duplicated here.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shutil
from typing import TYPE_CHECKING, Protocol

from . import _runtime as rt

if TYPE_CHECKING:
    from ._types import GPECCaseInputs, GPECSuiteConfig


@dataclass(frozen=True)
class SolverContext:
    """Everything a solver's ``prepare()`` needs to write its namelist(s)."""

    run_dir: Path
    template_dir: Path
    coil_data_dir: Path
    eq_filename: str
    mode: int
    inputs: "GPECCaseInputs"
    config: "GPECSuiteConfig"
    dcon_dir: Path


class Solver(Protocol):
    name: str

    def prepare(self, ctx: SolverContext) -> None:
        """Write this solver's namelist(s) and companion input files into ``ctx.run_dir``."""
        ...

    def output_patterns(self, mode: int) -> tuple[str, ...]:
        """Filenames this solver (and any companion it chains) is expected to produce."""
        ...

    def companion_executables(self) -> tuple[str, ...]:
        """Executable names to run after this solver succeeds, in order."""
        ...

    def companion_outputs(self, mode: int) -> tuple[str, ...]:
        """The subset of ``output_patterns`` a companion produces, not this solver.

        Companions are resolved with :func:`_runtime.optional_executable`, so an
        installation may legitimately not have one -- which makes these files
        legitimately absent, and means their absence must neither be reported as
        a failure nor force an otherwise-complete cell to be solved again.
        """
        ...

    def check_success(self, run_dir: Path, mode: int) -> tuple[bool, str]:
        """Whether a completed run actually produced usable physics output."""
        ...

    def stability_output(self, mode: int) -> str | None:
        """The output carrying this run's free-boundary energy, or ``None``.

        ``None`` for a solver that computes no such energy, and so can never
        report an equilibrium as stable.
        """
        ...


def free_boundary_stable(run_dir: Path, filename: str) -> bool | None:
    """Whether ``filename``'s own energies say every free-boundary mode is stable.

    A stable discharge produces no unstable-mode output, so any check that keys
    on the presence of such output reads the desirable outcome as a failure
    (issue #423). All three ideal solvers decide stability the same way --
    ``Re(total1) < 0`` is unstable, anything else is stable
    (``dcon/dcon.F:306-315``, ``rdcon/dcon.f:450-458``,
    ``stride/stride.F:376-385``) -- and all three put that number in their
    netCDF: RDCON and STRIDE as a ``total1`` global attribute
    (``rdcon/rdcon_netcdf.f:155``, ``stride/stride_netcdf.f:148``), DCON as the
    ``W_t_eigenvalue`` entry labelled by mode 1.

    Taken from the output rather than from the solver's log, which states the
    same verdict in prose but -- in DCON and RDCON -- only under ``verbose``
    (``dcon/dcon.F:309``, ``rdcon/dcon.f:452``): a Fortran default VAFT does not
    set, and one an operator quietening a scan may switch off without ever
    connecting that to a shot's stability being misreported.

    ``None`` when this run computed no free-boundary energy, which is not the
    same as being unstable. STRIDE writes the attribute only when it ran
    ``free_run`` (``stride/stride_netcdf.f:144-149``), so there its absence is
    the whole signal; RDCON writes it under ``vac_flag`` alone
    (``rdcon/rdcon_netcdf.f:152``) and stores an exact zero when it skipped
    ``free_run`` anyway (``rdcon/dcon.f:431-438``), which is that sentinel
    rather than a marginally stable equilibrium.
    """
    path = run_dir / filename
    if not path.exists():
        return None
    try:
        import xarray as xr

        from ._netcdf import complex_scalar_attr, complex_var, least_stable_eigenvalue

        with xr.open_dataset(path) as ds:
            energy = complex_scalar_attr(ds, "total1")
            if energy is None:
                energy = least_stable_eigenvalue(
                    complex_var(ds, "W_t_eigenvalue"),
                    ds["mode"].values if "mode" in ds.variables else None,
                )
    except Exception:
        # An unreadable file says nothing about stability. Reporting that as
        # "not stable" would be a verdict this function did not establish; the
        # callers that care about integrity check it themselves.
        return None
    if energy is None or energy.real == 0.0:
        return None
    return energy.real > 0.0


def required_outputs(solver: Solver, mode: int) -> tuple[str, ...]:
    """The files ``solver`` itself must produce for its run directory to be complete.

    Derived from ``output_patterns`` rather than maintained as a second list: a
    solver that grows a new output should not have to remember to add it in two
    places, and the two lists drifting apart is exactly how a companion's file
    ends up being treated as mandatory again.
    """
    companions = set(solver.companion_outputs(mode))
    return tuple(name for name in solver.output_patterns(mode) if name not in companions)


def missing_companion_outputs(solver: Solver, run_dir: Path, mode: int) -> tuple[str, ...]:
    """Companion-produced files this run directory does not have."""
    return tuple(name for name in solver.companion_outputs(mode) if not (run_dir / name).exists())


def _check_nc_variable(
    run_dir: Path, filename: str, variable: str | None = None
) -> tuple[bool, str]:
    """Check that ``filename`` really carries readable ``variable`` data.

    A solver killed mid-write leaves a file whose header is complete but
    whose data is not, so presence of the variable *name* is not enough:
    xarray opens lazily, and a truncated classic netCDF reads back as
    silent zeros for the missing tail rather than raising. This therefore
    forces the data to materialize and, for the classic format, requires
    the file to be at least as large as its own header says its variables
    need. Compressed HDF5-based files are legitimately smaller than their
    logical size, so the size floor is applied only to classic files.

    ``variable`` names a leaf whose presence is also required. Pass ``None``
    to check integrity alone, for a file whose variable names VAFT does not
    otherwise depend on -- naming a leaf GPEC might not write would turn a
    healthy run into a spurious failure.
    """
    path = run_dir / filename
    if not path.exists():
        return False, f"missing output: {filename}"
    try:
        import numpy as np
        import xarray as xr

        with xr.open_dataset(path) as ds:
            if variable is not None and variable not in ds.variables:
                return False, f"{filename} is missing expected variable {variable!r}"
            # Materialize values; HDF5-backed truncation raises here. Read the
            # named leaf when there is one, else the largest, so the check
            # stays bounded rather than loading an entire grid needlessly.
            if variable is not None:
                values = np.asarray(ds[variable].values)
                # Defining the output variables and then failing before filling
                # them leaves the name in place backed by fill values, which the
                # truncation checks below cannot see: the file is exactly as
                # long as its header says it should be.
                if values.size and not bool(np.isfinite(values.astype(float, copy=False)).any()):
                    return False, f"{filename}'s {variable!r} holds no finite value"
            elif ds.variables:
                largest = max(ds.variables, key=lambda name: int(np.prod(ds[name].shape)))
                np.asarray(ds[largest].values)

            with open(path, "rb") as handle:
                classic = handle.read(3) == b"CDF"
            if classic:
                needed = sum(
                    int(np.prod(var.shape)) * int(var.dtype.itemsize)
                    for var in ds.variables.values()
                )
                actual = path.stat().st_size
                if actual < needed:
                    return False, (
                        f"{filename} is truncated: {actual} bytes on disk, but its "
                        f"header declares at least {needed} bytes of variable data"
                    )
    except Exception as exc:  # pragma: no cover - defensive, exercised via real .nc files only
        return False, f"could not read {filename}: {exc}"
    return True, ""


def _defines_variable(run_dir: Path, filename: str, variable: str) -> bool:
    """Whether ``filename`` defines ``variable`` at all, regardless of its contents."""
    try:
        import xarray as xr

        with xr.open_dataset(run_dir / filename) as ds:
            return variable in ds.variables
    except Exception:
        return False


def _check_matching_output(run_dir: Path, filename: str) -> tuple[bool, str]:
    """Verify an RDCON/STRIDE output, allowing a stable run to have no ``Delta_prime``.

    ``Delta_prime`` is the tearing-stability matrix, and both solvers define it
    only when the solve found rational surfaces to match across
    (``rdcon/rdcon_netcdf.f:317``, ``stride/stride_netcdf.f:212``). An
    equilibrium the solver itself declares free-boundary stable therefore has
    none to write, and demanding it would fail every stable cell (issue #423).

    The rescue is narrow on purpose. The file still has to pass the same
    integrity check on its own -- present, openable, not truncated -- and the
    stable verdict still has to come from the file's own energies. Only the one
    reason "this healthy output does not name ``Delta_prime``" is forgiven; a
    missing, truncated or unreadable output is still a failure, and is still
    reported with the reason that made it one.
    """
    ok, reason = _check_nc_variable(run_dir, filename, "Delta_prime")
    if ok:
        return True, ""
    intact, intact_reason = _check_nc_variable(run_dir, filename)
    if not intact:
        # The file itself is the problem. Reporting the absent ``Delta_prime``
        # here would point the reader at the tearing solve rather than at the
        # damage that is actually stopping the read.
        return False, intact_reason
    if _defines_variable(run_dir, filename, "Delta_prime"):
        # Written but unusable is a defect in the tearing solve, not the
        # absence a stable equilibrium legitimately produces.
        return ok, reason
    if free_boundary_stable(run_dir, filename) is True:
        return True, ""
    return ok, reason


class DCONSolver:
    name = "dcon"

    def prepare(self, ctx: SolverContext) -> None:
        rt.write_template(
            ctx.template_dir / "dcon.in",
            ctx.run_dir / "dcon.in",
            {
                "nn": ctx.mode,
                "sas_flag": ctx.config.dcon.sas_flag,
                "qhigh": ctx.config.dcon.qhigh,
                "psiedge": ctx.config.dcon.psiedge,
                "mer_flag": ctx.config.dcon.mer_flag,
                "bal_flag": ctx.config.dcon.bal_flag,
                "thmax0": ctx.config.dcon.thmax0,
            },
        )
        shutil.copy2(ctx.template_dir / "match.in", ctx.run_dir / "match.in")

    def output_patterns(self, mode: int) -> tuple[str, ...]:
        return (
            "euler.bin",
            "psi_in.bin",
            "vacuum.bin",
            f"dcon_output_n{mode}.nc",
            "dcon.out",
            "match.out",
            "solutions.bin",
        )

    def companion_executables(self) -> tuple[str, ...]:
        return ("match",)

    def companion_outputs(self, mode: int) -> tuple[str, ...]:
        # `match` reads DCON's euler.bin and writes the reconstructed ideal
        # solution (match/ideal.f:378-389); DCON writes neither file itself.
        return ("match.out", "solutions.bin")

    def check_success(self, run_dir: Path, mode: int) -> tuple[bool, str]:
        return _check_nc_variable(run_dir, f"dcon_output_n{mode}.nc", "W_t_eigenvalue")

    def stability_output(self, mode: int) -> str | None:
        return f"dcon_output_n{mode}.nc"


class RDCONSolver:
    name = "rdcon"

    def prepare(self, ctx: SolverContext) -> None:
        rt.write_template(ctx.template_dir / "rdcon.in", ctx.run_dir / "rdcon.in", {"nn": ctx.mode})
        # rmatch.in has no per-run templated keys (no `nn`): it reads whatever
        # RDCON just wrote into this same directory (vmat.bin, etc).
        shutil.copy2(ctx.template_dir / "rmatch.in", ctx.run_dir / "rmatch.in")

    def output_patterns(self, mode: int) -> tuple[str, ...]:
        return (
            f"rdcon_output_n{mode}.nc",
            "delta_gw.out",
            "dcon.out",
            "globalsol.bin",
            "vmat.bin",
        )

    def companion_executables(self) -> tuple[str, ...]:
        return ("rmatch",)

    def companion_outputs(self, mode: int) -> tuple[str, ...]:
        # globalsol.bin is rmatch's (rmatch/match.f:1372). vmat.bin and
        # delta_gw.out only look like companion output: RDCON writes both
        # itself (rdcon/sing.f:73, rdcon/gal.f:1419).
        return ("globalsol.bin",)

    def check_success(self, run_dir: Path, mode: int) -> tuple[bool, str]:
        return _check_matching_output(run_dir, self.stability_output(mode))

    def stability_output(self, mode: int) -> str | None:
        return f"rdcon_output_n{mode}.nc"


class STRIDESolver:
    name = "stride"

    def prepare(self, ctx: SolverContext) -> None:
        rt.write_template(ctx.template_dir / "stride.in", ctx.run_dir / "stride.in", {"nn": ctx.mode})

    def output_patterns(self, mode: int) -> tuple[str, ...]:
        return (f"stride_output_n{mode}.nc", "stride.out", "delta_prime.out")

    def companion_executables(self) -> tuple[str, ...]:
        return ()

    def companion_outputs(self, mode: int) -> tuple[str, ...]:
        return ()

    def check_success(self, run_dir: Path, mode: int) -> tuple[bool, str]:
        return _check_matching_output(run_dir, self.stability_output(mode))

    def stability_output(self, mode: int) -> str | None:
        return f"stride_output_n{mode}.nc"


class IdealGPECSolver:
    name = "gpec"

    def prepare(self, ctx: SolverContext) -> None:
        rt.write_template(
            ctx.template_dir / "gpec.in",
            ctx.run_dir / "gpec.in",
            {"dcon_dir": str(ctx.dcon_dir), "coil_flag": ctx.config.gpec.coil_flag},
        )
        shutil.copy2(ctx.template_dir / "vac.in", ctx.run_dir / "vac.in")
        # Precedence: an explicit coil.in wins over canonical coil_specs,
        # which wins over the packaged template copied verbatim.
        if ctx.inputs.coil_in:
            shutil.copy2(Path(ctx.inputs.coil_in).expanduser(), ctx.run_dir / "coil.in")
        elif ctx.config.gpec.coil_specs is not None:
            from ._coil_input import resolve_coil_inputs, stage_coil_data, write_coil_in

            options = ctx.config.gpec
            specs = tuple(options.coil_specs)
            coil_config, ip_direction, bt_direction = resolve_coil_inputs(
                options.machine,
                options.coil_config,
                options.ip_direction,
                options.bt_direction,
                [spec.name for spec in specs],
            )
            coil_dir = ctx.run_dir / "coil"
            # coil.in first: it carries the current-count validation, so a
            # rejected request leaves no half-staged coil/ tree behind.
            write_coil_in(
                ctx.template_dir / "coil.in",
                ctx.run_dir / "coil.in",
                data_dir=coil_dir.resolve(),
                specs=specs,
                machine=options.machine,
                coil_config=coil_config,
                ip_direction=ip_direction,
                bt_direction=bt_direction,
            )
            stage_coil_data(
                [coil_config[spec.name] for spec in specs], coil_dir, machine=options.machine
            )
        else:
            # The packaged template is VEST's; IdealGPECOptions refuses any other
            # machine without coil_specs, so this branch is VEST by construction.
            from vaft.machine_mapping.conventions import VEST_GPEC_COIL_DIRECTIONS

            from ._coil_input import DEFAULT_MACHINE

            assert ctx.config.gpec.machine == DEFAULT_MACHINE
            rt.write_template(
                ctx.template_dir / "coil.in",
                ctx.run_dir / "coil.in",
                {
                    "data_dir": str(ctx.coil_data_dir.resolve()),
                    "machine": DEFAULT_MACHINE,
                    "coil_num": 3,
                    **VEST_GPEC_COIL_DIRECTIONS,
                },
            )

    def output_patterns(self, mode: int) -> tuple[str, ...]:
        return (
            f"gpec_control_output_n{mode}.nc",
            f"gpec_profile_output_n{mode}.nc",
            f"gpec_cylindrical_output_n{mode}.nc",
            f"gpec_response_n{mode}.out",
            f"gpec_bnormal_pest_n{mode}.out",
        )

    def companion_executables(self) -> tuple[str, ...]:
        return ()

    def companion_outputs(self, mode: int) -> tuple[str, ...]:
        return ()

    def check_success(self, run_dir: Path, mode: int) -> tuple[bool, str]:
        core = (
            f"gpec_control_output_n{mode}.nc",
            f"gpec_profile_output_n{mode}.nc",
            f"gpec_cylindrical_output_n{mode}.nc",
        )
        missing = [name for name in core if not (run_dir / name).exists()]
        if missing:
            return False, f"missing outputs: {', '.join(missing)}"
        # Every core file must be verified, not just the first: GPEC writes
        # them in sequence, so a run killed mid-write leaves an intact
        # control.nc beside a truncated profile or cylindrical one. `R` is the
        # grid coordinate the cylindrical reader itself requires; the profile
        # file is integrity-checked only, since VAFT reads no named leaf from
        # it and inventing one could fail a healthy run.
        for filename, variable in (
            (f"gpec_control_output_n{mode}.nc", "b_n"),
            (f"gpec_cylindrical_output_n{mode}.nc", "R"),
            (f"gpec_profile_output_n{mode}.nc", None),
        ):
            ok, reason = _check_nc_variable(run_dir, filename, variable)
            if not ok:
                return False, reason
        return True, ""

    def stability_output(self, mode: int) -> str | None:
        # Ideal GPEC computes a perturbed-field response to an applied
        # spectrum, not a free-boundary energy: it has no stability verdict of
        # its own to report, and takes DCON's as given.
        return None


SOLVERS: dict[str, Solver] = {
    "dcon": DCONSolver(),
    "rdcon": RDCONSolver(),
    "stride": STRIDESolver(),
    "gpec": IdealGPECSolver(),
}
