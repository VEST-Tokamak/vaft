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

    def check_success(self, run_dir: Path, mode: int) -> tuple[bool, str]:
        """Whether a completed run actually produced usable physics output."""
        ...


def _check_nc_variable(run_dir: Path, filename: str, variable: str) -> tuple[bool, str]:
    path = run_dir / filename
    if not path.exists():
        return False, f"missing output: {filename}"
    try:
        import xarray as xr

        with xr.open_dataset(path) as ds:
            if variable not in ds.variables:
                return False, f"{filename} is missing expected variable {variable!r}"
    except Exception as exc:  # pragma: no cover - defensive, exercised via real .nc files only
        return False, f"could not read {filename}: {exc}"
    return True, ""


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

    def check_success(self, run_dir: Path, mode: int) -> tuple[bool, str]:
        return _check_nc_variable(run_dir, f"dcon_output_n{mode}.nc", "W_t_eigenvalue")


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

    def check_success(self, run_dir: Path, mode: int) -> tuple[bool, str]:
        return _check_nc_variable(run_dir, f"rdcon_output_n{mode}.nc", "Delta_prime")


class STRIDESolver:
    name = "stride"

    def prepare(self, ctx: SolverContext) -> None:
        rt.write_template(ctx.template_dir / "stride.in", ctx.run_dir / "stride.in", {"nn": ctx.mode})

    def output_patterns(self, mode: int) -> tuple[str, ...]:
        return (f"stride_output_n{mode}.nc", "stride.out", "delta_prime.out")

    def companion_executables(self) -> tuple[str, ...]:
        return ()

    def check_success(self, run_dir: Path, mode: int) -> tuple[bool, str]:
        return _check_nc_variable(run_dir, f"stride_output_n{mode}.nc", "Delta_prime")


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
        elif ctx.config.gpec.coil_specs:
            from ._coil_input import stage_coil_data, write_coil_in
            from vaft.machine_mapping.coil_geometry_3d import load_vest_3d_coil_config

            specs = tuple(ctx.config.gpec.coil_specs)
            config = load_vest_3d_coil_config(coil_sets=[spec.name for spec in specs])
            coil_dir = ctx.run_dir / "coil"
            stage_coil_data(
                [config[spec.name] for spec in specs], coil_dir
            )
            write_coil_in(
                ctx.template_dir / "coil.in",
                ctx.run_dir / "coil.in",
                data_dir=coil_dir.resolve(),
                specs=specs,
            )
        else:
            rt.write_template(
                ctx.template_dir / "coil.in",
                ctx.run_dir / "coil.in",
                {"data_dir": str(ctx.coil_data_dir.resolve()), "machine": "vest", "coil_num": 3},
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

    def check_success(self, run_dir: Path, mode: int) -> tuple[bool, str]:
        core = (
            f"gpec_control_output_n{mode}.nc",
            f"gpec_profile_output_n{mode}.nc",
            f"gpec_cylindrical_output_n{mode}.nc",
        )
        missing = [name for name in core if not (run_dir / name).exists()]
        if missing:
            return False, f"missing outputs: {', '.join(missing)}"
        return _check_nc_variable(run_dir, f"gpec_control_output_n{mode}.nc", "b_n")


SOLVERS: dict[str, Solver] = {
    "dcon": DCONSolver(),
    "rdcon": RDCONSolver(),
    "stride": STRIDESolver(),
    "gpec": IdealGPECSolver(),
}
