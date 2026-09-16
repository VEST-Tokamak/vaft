"""Is the diamagnetic constraint reachable at all, and does pressure follow? (#386)

    PYTHONPATH=$PWD EFITHOME=~/git/efit/build-linux \\
        python workflow/efit_diamagnetic_weight/diamagnetic_weight_scan.py \\
            --output /scratch/dia-weight --shots 39915,41524,41672 --workers 8

Why this is not #663 run again
------------------------------

#663 swept ``objective_scales["diamagnetic_flux"]`` from x1 to x10,000 on all
three reference shots and concluded the family was inactive within a produced
solution.  That conclusion cannot be read as a statement about the diagnostic,
because the sweep never left the region where the row is inactive by
construction.  EFIT processes a statistical row as ``FWT/sigma``:

    SIGDLC = weight * legacy_weight_scale * 1000       vaft/code/efit/kfile.py
           = 1 * 1e4 * 1e3 = 1e7 mWb
    sigdia = 1e-3 * abs(SIGDLC) = 1e4 Wb               data_input.F90:2297
    fwtdlc = fwtdlc / sigdia**nsq,  nsq = 1            data_input.F90:2782, :109
    row    = fwtdlc * rspdlc,  rhs = -fwtdlc * diamag  response_matrix.F90:2192

so the row weight VEST submits is ``FWTDLC/sigdia = 1e-4``.  The stored 3 %
measurement error would give ``2.3e4``.  That is a gap of **8.4 decades**, and
the x1..x10,000 ladder covers four of them -- all inside the inactive region.

This study moves the sigma instead, through
``constraints.uncertainty_scales["diamagnetic_flux"]``, which divides SIGDLC
for that family alone and leaves every other submitted weight and uncertainty
where it was.  Ten rungs, one decade apart, carry the processed row weight from
``1e-4`` to ``1e5`` and bracket the stored-error point.

``uncertainty_mode="standard_deviation"`` is deliberately **not** used as a
rung.  It replaces the legacy uncertainty for *every* family at once, so a run
under it differs from the baseline on five axes and attributes nothing -- the
same one-axis-at-a-time rule #588 established for the seed study.  Where each
rung sits relative to the real measurement error is reported per slice instead,
as ``sigma_ratio_to_stored``.

What is being asked
-------------------

Three questions, in order, and the third only matters if the first two answer
the way #386 expects:

1. **Reachability.**  At what processed row weight does the diamagnetic
   residual first stop being ignored?  Reported as ``w_activation`` on the
   residual itself, not on geometry, because a constraint can be active and
   still not move the boundary.
2. **Consequence.**  Does the reconstructed pressure follow?  ``p_axis``,
   ``wmhd`` and ``betap`` on slices common to every rung -- population change
   is reported separately and never mixed in, which is the correction #663 had
   to publish about its own first pass.
3. **Direction.**  Does it move *toward* the independent kinetic measurement?
   On 39915 only, through ``thomson_pressure_check``, which is the one shot
   whose Thomson samples fall inside its EFIT window.

The classification thresholds are #663's, unchanged and reused from its module,
so the two studies compose rather than competing.

Everything here is characterization.  No production default is changed:
``uncertainty_scales`` ships at 1.0 for every family, and the baseline rung is
today's configuration exactly.
"""

from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import math
import os
import sys
import warnings
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

SCHEMA = 1
REPOSITORY = Path(__file__).resolve().parents[2]
CONSTRAINT_STUDY = (
    REPOSITORY
    / "workflow"
    / "efit_constraint_information"
    / "constraint_information_study.py"
)
PROFILE_STUDY = (
    REPOSITORY / "workflow" / "efit_profile_models" / "profile_model_study.py"
)
SEED_STUDY = REPOSITORY / "workflow" / "efit_numerics" / "seed_basin.py"
THOMSON_CHECK = Path(__file__).resolve().parent / "thomson_pressure_check.py"
REFERENCE_SET = REPOSITORY / "test" / "data" / "efit_reference_set.json"
DEFAULT_TABLE = REPOSITORY / "test" / "data" / "efit_diamagnetic_weight.json"

#: The rung that is today's production configuration.
BASELINE = "legacy_sigma"

#: The shot whose Thomson samples fall inside its own EFIT window, and so the
#: only one that can answer question 3 (``workflow/efit_reference_set``).
KINETIC_SHOT = 39915


@dataclass(frozen=True)
class Rung:
    """One position on the ladder: a divisor on the diamagnetic sigma."""

    name: str
    uncertainty_scale: float
    purpose: str = ""

    @property
    def submitted_sigma_mwb(self) -> float:
        """SIGDLC as this rung writes it, for a unit channel weight."""
        return 1.0e4 * 1.0e3 / self.uncertainty_scale

    @property
    def processed_row_weight(self) -> float:
        """``FWTDLC/sigdia``, the weight EFIT actually gives the row."""
        return 1.0 / (1.0e-3 * self.submitted_sigma_mwb)


LADDER: tuple[Rung, ...] = (
    Rung(
        BASELINE,
        1.0,
        purpose="today's production configuration; processed row weight 1e-4",
    ),
) + tuple(
    Rung(
        f"sigma_x1e{exponent}",
        float(10**exponent),
        purpose=(
            "the far end of the #663 ladder, in row-weight terms"
            if exponent == 4
            else (
                "near the stored 3 % measurement error"
                if exponent == 8
                else "one decade of processed row weight"
            )
        ),
    )
    for exponent in range(1, 10)
)


def _module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def scientific_for(rung: Rung, profile_study: Any):
    """The frozen #579/#663 configuration with one family's sigma scaled."""
    from vaft.code.efit.config import DIAGNOSTIC_GROUPS

    base = profile_study.fixed_scientific_config()
    scales = {name: 1.0 for name in DIAGNOSTIC_GROUPS}
    scales["diamagnetic_flux"] = float(rung.uncertainty_scale)
    constraints = replace(
        base.constraints,
        group_weights={},
        uncertainty_scales=scales,
    )
    return replace(base, constraints=constraints)


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _stored_sigma_wb(constraints: Any, shot_times: np.ndarray) -> dict[int, float]:
    """The measurement's own error bar per slice, in Wb, from the ODS.

    This is what the rungs are measured against.  It is read from the frozen
    constraints rather than from any run, so it is the same number for every
    rung by construction.
    """
    stored: dict[int, float] = {}
    for index, time_s in enumerate(shot_times):
        path = f"equilibrium.time_slice.{index}.constraints.diamagnetic_flux"
        try:
            value = abs(float(constraints[f"{path}.measured_error_upper"]))
        except Exception:
            continue
        if math.isfinite(value) and value > 0.0:
            stored[int(round(float(time_s) * 1000.0))] = value
    return stored


def _kfile_for(workdir: Path, shot: int, afile_name: str) -> Path:
    """The k-file that produced one a-file, by the shared name suffix."""
    return workdir / "kfile" / f"k0{shot}{afile_name.split(f'a0{shot}', 1)[1]}"


def _submitted_row(kfile: Path) -> dict[str, Any]:
    """What the k-file actually asked EFIT for, and what EFIT will make of it.

    ``DFLUX`` and ``SIGDLC`` are written in mWb (``data_input.F90:2296-2297``
    converts both with ``1e-3``), so the processed row weight is
    ``FWTDLC / (1e-3 * |SIGDLC|)``.  This is read from the file rather than
    recomputed from the rung, so a rung that failed to write what it intended
    shows up as a disagreement instead of being assumed correct.
    """
    import f90nml

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        namelist = f90nml.read(kfile)["in1"]
    sigma = abs(float(namelist["sigdlc"])) * 1.0e-3
    weight = float(namelist["fwtdlc"])
    return {
        "measured_wb": float(namelist["dflux"]) * 1.0e-3,
        "sigma_submitted_wb": sigma,
        "submitted_weight": weight,
        "processed_row_weight": (weight / sigma) if sigma > 0.0 else None,
    }


def diamagnetic_rows(
    run: Mapping[str, Any],
    *,
    workdir: Path,
    shot: int,
    stored_sigma: Mapping[int, float],
) -> list[dict[str, Any]]:
    """What the diamagnetic row did on each slice, and what pressure did.

    The row is reconstructed from the a-file and the k-file, which EFIT always
    writes.  The m-file would carry ``fwtdia`` and ``chidflux`` directly, but
    it is NetCDF and this build ships with ``ENABLE_NETCDF=OFF``, so depending
    on it would leave the study silently measuring nothing.  Where #663's
    ``constraint_audit`` *is* available it is read as a cross-check and any
    disagreement is recorded rather than averaged away.

    ``cdflux`` is EFIT's exact forward flux (``beta_li.F90:787,832``), not the
    linearised row the fit is driven by, so ``residual_wb`` is the same
    quantity ``chidflux`` scores (``beta_li.F90:840``).

    Both chi-squares are kept.  ``chi_squared`` is ``(residual/sigma)^2``, the
    m-file normalisation with the submitted weight divided out;
    ``chi_squared_reweighted`` is the solver's own contribution
    ``(processed weight * residual)^2``.  #663 had to publish a correction for
    treating those as one quantity.
    """
    from vaft.data import read_aeqdsk

    rows = []
    for item in run["slices"]:
        time_ms = int(round(float(item["time_ms"])))
        record: dict[str, Any] = {
            "time_ms": time_ms,
            "outcome": item.get("outcome"),
            "sigma_stored_wb": stored_sigma.get(time_ms),
            "row_source": None,
        }
        afile = item.get("afile")
        if afile is not None:
            scalars = afile.get("scalars", {})
            for name in ("betap", "li", "wmhd", "chisq", "area", "volume"):
                record[name] = _finite(scalars.get(name))
            kfile = _kfile_for(workdir, shot, afile["path"])
            reconstructed = _finite(
                read_aeqdsk(workdir / afile["path"]).scalars.get("cdflux")
            )
            if kfile.is_file() and reconstructed is not None:
                record.update(_submitted_row(kfile))
                record["row_source"] = "afile+kfile"
                record["reconstructed_wb"] = reconstructed
                residual = reconstructed - record["measured_wb"]
                record["residual_wb"] = residual
                sigma = record["sigma_submitted_wb"]
                weight = record["processed_row_weight"]
                if sigma:
                    record["chi_squared"] = (residual / sigma) ** 2
                if weight is not None:
                    record["chi_squared_reweighted"] = (weight * residual) ** 2
                if record["measured_wb"]:
                    record["residual_relative"] = residual / abs(record["measured_wb"])
                stored = record["sigma_stored_wb"]
                if sigma and stored:
                    record["sigma_ratio_to_stored"] = sigma / stored
        # The m-file, when this build can write one, says the same thing from
        # EFIT's own arrays.  Disagreement is a fact about the run, not noise.
        family = (item.get("constraint_audit") or {}).get("families", {}).get(
            "diamagnetic_flux"
        )
        if family and family.get("processed_weight"):
            mfile_weight = _finite(family["processed_weight"][0])
            record["mfile_processed_row_weight"] = mfile_weight
            record["mfile_chi_squared"] = (
                _finite(family["chi2_per_channel"][0])
                if family.get("chi2_per_channel")
                else None
            )
            submitted = record.get("processed_row_weight")
            if mfile_weight and submitted:
                record["row_weight_agreement"] = mfile_weight / submitted
        gfile = item.get("gfile")
        if gfile is not None:
            pressure = gfile.get("profiles", {}).get("pressure") or []
            record["p_axis"] = _finite(pressure[0]) if pressure else None
            record["p_edge"] = _finite(pressure[-1]) if pressure else None
        rows.append(record)
    return rows


def summarize_rung(
    rows: Sequence[Mapping[str, Any]], profile_study: Any
) -> dict[str, Any]:
    """Spreads over the slices that produced an equilibrium."""
    produced = [row for row in rows if row.get("outcome") in ("accepted", "flagged")]
    accepted = [row for row in produced if row.get("outcome") == "accepted"]
    return {
        "slices": len(rows),
        "produced": len(produced),
        "accepted": len(accepted),
        # A row that could not be read is not a row that did not respond.
        "diamagnetic_row_read": sum(
            1 for row in produced if row.get("row_source") is not None
        ),
        "processed_row_weight": profile_study.spread(
            row.get("processed_row_weight") for row in produced
        ),
        "sigma_ratio_to_stored": profile_study.spread(
            row.get("sigma_ratio_to_stored") for row in produced
        ),
        "diamagnetic_residual_relative": profile_study.spread(
            (row.get("residual_relative") for row in produced), absolute=True
        ),
        "diamagnetic_chi_squared": profile_study.spread(
            row.get("chi_squared") for row in produced
        ),
        **{
            name: profile_study.spread(row.get(name) for row in produced)
            for name in ("p_axis", "wmhd", "betap", "li", "chisq")
        },
    }


def common_slice_change(
    rung_rows: Sequence[Mapping[str, Any]],
    baseline_rows: Sequence[Mapping[str, Any]],
    profile_study: Any,
) -> dict[str, Any]:
    """Relative change on slices both runs produced, and nothing else.

    #663's first pass reported a smaller aggregate residual at high weight and
    had to correct itself: the smaller number was population selection, not a
    better fit.  Comparing only common slices is what makes that mistake
    impossible rather than merely unlikely.
    """
    produced = lambda rows: {
        row["time_ms"]: row
        for row in rows
        if row.get("outcome") in ("accepted", "flagged")
    }
    left, right = produced(rung_rows), produced(baseline_rows)
    common = sorted(set(left) & set(right))
    changes: dict[str, list[float]] = {
        name: [] for name in ("p_axis", "wmhd", "betap", "li", "residual_relative")
    }
    for time_ms in common:
        for name in changes:
            before, after = right[time_ms].get(name), left[time_ms].get(name)
            if before is None or after is None or before == 0.0:
                continue
            changes[name].append((after - before) / abs(before))
    return {
        "common_slices": len(common),
        "relative_change": {
            name: profile_study.spread(values, absolute=True)
            for name, values in changes.items()
        },
        "signed_relative_change": {
            name: profile_study.spread(values) for name, values in changes.items()
        },
    }


#: A common-slice relative change below this is numerical noise, not a
#: response.  #663 measured invariance at 1e-7 through its whole ladder and a
#: real but sub-threshold 0.5 % response at one slice, so the floor has to sit
#: between those: 1e-3 separates them by three orders either way.
RESPONSE_FLOOR = 1.0e-3


def branch_response(comparison: Mapping[str, Any]) -> bool:
    """Did the *equilibrium* move, as opposed to the population shrinking?

    #663's ``material_response`` is the right predicate for "did removing this
    family change anything", and its thresholds are reused verbatim below.
    But it counts a 10-percentage-point acceptance change as a material
    response, and on this ladder acceptance falls simply because high weights
    reject slices.  Folding that into ``w_branch`` would report a branch
    transition every time solutions were lost, which is the one confusion
    ``w_fail`` exists to prevent.

    So the branch test keeps #663's geometry thresholds -- 5 mm of LCFS, 2 % of
    area or volume, all measured on slices both runs produced -- and drops the
    acceptance term, which is reported separately as ``w_fail``.
    """
    geometry = comparison["geometry"]
    lcfs = geometry["lcfs_rms_mm"]["median"] or 0.0
    area = geometry["absolute_relative_change"]["area"]["median"] or 0.0
    volume = geometry["absolute_relative_change"]["volume"]["median"] or 0.0
    return lcfs >= 5.0 or area >= 0.02 or volume >= 0.02


def classify_reachability(shot_block: Mapping[str, Any]) -> dict[str, Any]:
    """Where the ladder crosses from inactive to active to broken.

    The three scales #663's interpretation asked for, measured rather than
    assumed: first response on the same branch, transition to another branch,
    and loss of solution.
    """
    rungs = shot_block["rungs"]
    baseline = rungs[BASELINE]
    ordered = [item for item in LADDER if item.name in rungs]
    baseline_produced = baseline["summary"]["produced"]

    # A ladder on which nothing was reconstructed has measured nothing.  It
    # must not be reported as "the constraint is unreachable": that sentence
    # is the study's most consequential finding and an empty run would be
    # asserting it for free.
    silent = {
        "response_floor_relative": RESPONSE_FLOOR,
        "w_activation": None,
        "w_branch": None,
        "w_fail": None,
        "reachable_band": None,
    }
    if not baseline_produced:
        return {
            **silent,
            "verdict": (
                "the baseline rung produced no equilibrium, so this shot "
                "measured nothing about the diamagnetic constraint"
            ),
        }
    if not any(
        rungs[item.name]["summary"]["diamagnetic_row_read"] for item in ordered
    ):
        return {
            **silent,
            "verdict": (
                "no rung's diamagnetic row could be read back from its "
                "outputs, so nothing here is evidence about the constraint"
            ),
        }

    activation = branch = failure = None
    for rung in ordered:
        if rung.name == BASELINE:
            continue
        block = rungs[rung.name]
        change = block["versus_baseline"]
        residual = change["relative_change"]["residual_relative"]["median"]
        if (
            activation is None
            and residual is not None
            and residual >= RESPONSE_FLOOR
        ):
            activation = rung.name
        if branch is None and block.get("branch_response"):
            branch = rung.name
        if (
            failure is None
            and baseline_produced
            and block["summary"]["produced"] <= 0.5 * baseline_produced
        ):
            failure = rung.name
    return {
        "response_floor_relative": RESPONSE_FLOOR,
        "w_activation": activation,
        "w_branch": branch,
        "w_fail": failure,
        "reachable_band": (
            None
            if activation is None
            else [activation, failure or ordered[-1].name]
        ),
        "verdict": _verdict(activation, branch, failure),
    }


def _verdict(activation, branch, failure) -> str:
    if activation is None and failure is None:
        return (
            "the diamagnetic residual never responded anywhere on the ladder: "
            "the constraint is unreachable by weighting alone and the pressure "
            "deficit is not a weighting problem"
        )
    if activation is None:
        return (
            f"solutions were lost at {failure} before the residual ever "
            "responded: raising the weight removes reconstructions without "
            "improving the one it is supposed to constrain"
        )
    if failure is not None and failure == activation:
        return (
            f"the residual first responded at {activation}, the same rung that "
            "lost half the solutions: there is no usable band"
        )
    return (
        f"the residual first responded at {activation}"
        + (f", the branch moved at {branch}" if branch else "")
        + (f", and solutions were lost at {failure}" if failure else "")
    )


def markdown(payload: Mapping[str, Any]) -> str:
    lines = [
        "# Is the diamagnetic constraint reachable? (#386)",
        "",
        f"Run at {payload['run_at']}.",
        "",
        "The processed row weight is `FWTDLC/sigdia` (`data_input.F90:2782`, "
        "`nsq = 1`). Production sits at `1e-4`; the stored 3 % measurement "
        "error implies `2.3e4`. Each rung divides the diamagnetic sigma alone.",
        "",
    ]
    for shot, block in sorted(payload["shots"].items()):
        lines += [
            f"## {shot}",
            "",
            "| rung | row weight | sigma/stored | produced | accepted | "
            "row read | abs residual | p_axis [Pa] | beta_p |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
        for rung in LADDER:
            item = block["rungs"].get(rung.name)
            if item is None:
                continue
            s = item["summary"]
            cell = lambda spread: (
                "-" if spread["median"] is None else f"{spread['median']:.4g}"
            )
            lines.append(
                f"| {rung.name} | {cell(s['processed_row_weight'])} | "
                f"{cell(s['sigma_ratio_to_stored'])} | {s['produced']} | "
                f"{s['accepted']} | {s['diamagnetic_row_read']} | "
                f"{cell(s['diamagnetic_residual_relative'])} | "
                f"{cell(s['p_axis'])} | {cell(s['betap'])} |"
            )
        classification = block["classification"]
        lines += [
            "",
            f"**Verdict.** {classification['verdict']}.",
            "",
            f"- `w_activation`: {classification['w_activation'] or 'never'}",
            f"- `w_branch`: {classification['w_branch'] or 'never'}",
            f"- `w_fail`: {classification['w_fail'] or 'never'}",
            "",
        ]
        kinetic = block.get("kinetic_cross_check")
        if kinetic:
            lines += [
                "### Against Thomson, which the fit never used",
                "",
                kinetic.get("verdict", ""),
                "",
            ]
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--table", type=Path, default=DEFAULT_TABLE)
    parser.add_argument("--markdown", type=Path, default=None)
    parser.add_argument("--shots", default="39915,41524,41672")
    parser.add_argument(
        "--rungs",
        default=None,
        help="comma-separated rung names; default: the whole ladder",
    )
    parser.add_argument("--tables", default=None)
    parser.add_argument("--efit-home", default=None)
    parser.add_argument("--tstep", type=float, default=0.001)
    parser.add_argument("--average-window", type=float, default=0.0005)
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help=(
            "run this many rungs at once. Runtimes from a run with "
            "--workers > 1 are not comparable between rungs."
        ),
    )
    args = parser.parse_args(argv)
    if args.efit_home:
        os.environ["EFITHOME"] = str(Path(args.efit_home).expanduser())

    from vaft.code.efit.toolchain import resolve_toolchain, toolchain_identities
    from vaft.data.resources import data_path

    constraint_study = _module(CONSTRAINT_STUDY, "dia_constraint_support")
    profile_study = _module(PROFILE_STUDY, "dia_profile_support")
    seed_study = _module(SEED_STUDY, "dia_seed_support")
    thomson = _module(THOMSON_CHECK, "dia_thomson_support")

    resolved = resolve_toolchain()
    if resolved.get("efit") is None:
        print("no efit executable: set EFITHOME", file=sys.stderr)
        return 2

    rungs = LADDER
    if args.rungs:
        requested = [name.strip() for name in args.rungs.split(",") if name.strip()]
        by_name = {item.name: item for item in LADDER}
        unknown = sorted(set(requested) - set(by_name))
        if unknown:
            print(f"unknown rung(s): {', '.join(unknown)}", file=sys.stderr)
            return 2
        if BASELINE not in requested:
            requested = [BASELINE] + requested
        rungs = tuple(by_name[name] for name in requested)

    output = args.output.expanduser()
    output.mkdir(parents=True, exist_ok=True)
    source_tables = (
        Path(args.tables).expanduser()
        if args.tables
        else Path(data_path("efit")).resolve()
    )
    tables, table_record = profile_study.prepare_vest_tables(
        source_tables, output / "tables"
    )
    table_dir = str(tables.resolve()) + "/"

    reference = json.loads(REFERENCE_SET.read_text(encoding="utf-8"))
    products = {
        int(item["shot"]): item["files"]["product"]["path"]
        for item in reference["shots"]
        if item["files"]["product"] and item["files"]["product"]["exists"]
    }

    base = profile_study.fixed_scientific_config()
    payload: dict[str, Any] = {
        "schema_version": SCHEMA,
        "issue": 386,
        "run_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "toolchain": toolchain_identities(resolved),
        "table": table_record,
        "fixed_scientific": base.to_dict(),
        "fixed_scientific_sha256": base.sha256,
        "row_weight_arithmetic": {
            "submitted_sigdlc_mwb_at_baseline": LADDER[0].submitted_sigma_mwb,
            "processed_row_weight_at_baseline": LADDER[0].processed_row_weight,
            "stored_error_implied_row_weight": 1.0 / (0.03 * 1.448e-3),
            "decades_between": math.log10(
                (1.0 / (0.03 * 1.448e-3)) / LADDER[0].processed_row_weight
            ),
            "decades_covered_by_issue_663": 4.0,
        },
        "classification_thresholds": {
            "lcfs_mm": 5,
            "area_or_volume_relative": 0.02,
            "other_family_relative": 0.20,
            "acceptance_percentage_points": 10,
            "response_floor_relative": RESPONSE_FLOOR,
        },
        "ladder": [
            {
                **item.__dict__,
                "submitted_sigdlc_mwb": item.submitted_sigma_mwb,
                "processed_row_weight": item.processed_row_weight,
            }
            for item in rungs
        ],
        "shots": {},
    }

    for shot in [int(value) for value in args.shots.split(",")]:
        if shot not in products:
            print(f"{shot}: no packaged pre-EFIT product; skipped", file=sys.stderr)
            continue
        print(f"{shot}: building the shared frozen constraints", flush=True)
        constraints, times, window, baseline_module = seed_study.prepare_shot(
            shot,
            Path(data_path(products[shot])),
            workdir=output / f"shot_{shot}" / "constraints",
            tables=table_dir,
            tstep=args.tstep,
            average_window=args.average_window,
        )
        phase_by_time, threshold = profile_study._phase_map(
            constraints, times, base.initialization.current_threshold
        )
        stored_sigma = _stored_sigma_wb(constraints, times)

        def execute(rung: Rung) -> tuple[Rung, dict[str, Any], Path, Any]:
            scientific = scientific_for(rung, profile_study)
            workdir = output / f"shot_{shot}" / rung.name
            run = profile_study.run_model(
                copy.deepcopy(constraints),
                shot=shot,
                times=times,
                workdir=workdir,
                executable=str(resolved["efit"]),
                scientific=scientific,
                baseline_module=baseline_module,
                phase_by_time=phase_by_time,
            )
            return rung, run, workdir, scientific

        if args.workers > 1:
            with ThreadPoolExecutor(max_workers=args.workers) as pool:
                results = list(pool.map(execute, rungs))
        else:
            results = [execute(rung) for rung in rungs]

        block: dict[str, Any] = {
            "window": {
                "start": float(window.start),
                "end": float(window.end),
                "requested": int(times.size),
            },
            "phase_dcurrent_dt_threshold": threshold,
            "rungs": {},
        }
        runs: dict[str, dict[str, Any]] = {}
        for rung, run, workdir, scientific in results:
            constraint_study.enrich_run(run, workdir, shot, scientific)
            rows = diamagnetic_rows(
                run, workdir=workdir, shot=shot, stored_sigma=stored_sigma
            )
            runs[rung.name] = run
            block["rungs"][rung.name] = {
                "specification": {
                    **rung.__dict__,
                    "submitted_sigdlc_mwb": rung.submitted_sigma_mwb,
                    "processed_row_weight": rung.processed_row_weight,
                },
                "scientific_sha256": scientific.sha256,
                "seconds": run["seconds"],
                "summary": summarize_rung(rows, profile_study),
                "diamagnetic": rows,
            }
            summary = block["rungs"][rung.name]["summary"]
            print(
                f"  {rung.name}: {summary['produced']} produced, "
                f"{summary['accepted']} accepted, "
                f"row weight {summary['processed_row_weight']['median']}",
                flush=True,
            )

        baseline_rows = block["rungs"][BASELINE]["diamagnetic"]
        for rung in rungs:
            if rung.name == BASELINE:
                continue
            item = block["rungs"][rung.name]
            item["versus_baseline"] = common_slice_change(
                item["diamagnetic"], baseline_rows, profile_study
            )
            item["geometry_versus_baseline"] = constraint_study.comparison(
                runs[rung.name], runs[BASELINE], profile_study
            )
            # Kept for comparability with #663, which classifies on it.
            item["material_response"] = constraint_study.material_response(
                item["geometry_versus_baseline"], excluded_family="diamagnetic_flux"
            )
            item["branch_response"] = branch_response(
                item["geometry_versus_baseline"]
            )
        block["classification"] = classify_reachability(block)
        print(f"  {shot}: {block['classification']['verdict']}", flush=True)

        if shot == KINETIC_SHOT:
            block["kinetic_cross_check"] = thomson.compare_ladder(
                {rung.name: workdir for rung, _, workdir, _ in results},
                shot=shot,
                output=output / f"shot_{shot}",
            )

        payload["shots"][str(shot)] = block

    destination = args.table.expanduser()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(payload, indent=1, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {destination}")
    report = markdown(payload)
    if args.markdown:
        args.markdown.expanduser().write_text(report, encoding="utf-8")
        print(f"wrote {args.markdown}")
    else:
        print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
