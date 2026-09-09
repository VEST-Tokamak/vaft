"""Audit, ablate, and reweight VEST EFIT constraint families (issue #663)."""

from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import math
import os
import sys
import warnings
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

SCHEMA = 1
CHI2_NUMERICAL_FLOOR = 1.0e-18
REPOSITORY = Path(__file__).resolve().parents[2]
PROFILE_STUDY = (
    REPOSITORY / "workflow" / "efit_profile_models" / "profile_model_study.py"
)
SEED_STUDY = REPOSITORY / "workflow" / "efit_numerics" / "seed_basin.py"
REFERENCE_SET = REPOSITORY / "test" / "data" / "efit_reference_set.json"
BASELINE = "baseline"
PLASMA_PHASES = frozenset(("ramp_up", "quasi_stationary", "ramp_down"))

FAMILIES = {
    "plasma_current": ("plasma", "cpasma", "sigpasma", "fwtpasma", "chipasma"),
    "flux_loop": ("silopt", "csilop", "sigsil", "fwtsi", "saisil"),
    "bpol_probe": ("expmpi", "cmpr2", "sigmpi", "fwtmp2", "saimpi"),
    "diamagnetic_flux": ("diamag", "cdflux", "sigdia", "fwtdia", "chidflux"),
    "pf_current": ("fccurt", "ccbrsp", "sigfcc", "fwtfc", "chifcc"),
}


@dataclass(frozen=True)
class Variant:
    name: str
    scales: Mapping[str, float]
    use_relations: bool = True
    wall_current_mode: str = "measured"
    passive_structure_mode: str = "fixed_currents"
    group: str = "baseline"
    purpose: str = ""


def _units(**updates: float) -> dict[str, float]:
    values = {name: 1.0 for name in FAMILIES}
    values.update(updates)
    return values


DIAMAGNETIC_FINE_SCALES = tuple(range(1_000, 10_001, 1_000))


COMPLETE_VARIANTS = (
    Variant(BASELINE, _units(), purpose="all constraint families at unit scale"),
    Variant("no_ip", _units(plasma_current=0), group="ablation", purpose="remove Ip"),
    Variant(
        "no_flux_loops",
        _units(flux_loop=0),
        group="ablation",
        purpose="remove flux loops",
    ),
    Variant(
        "no_bpol_probes",
        _units(bpol_probe=0),
        group="ablation",
        purpose="remove B-pol probes",
    ),
    Variant(
        "no_diamagnetic_flux",
        _units(diamagnetic_flux=0),
        group="ablation",
        purpose="remove diamagnetic flux",
    ),
    Variant(
        "no_pf_penalty",
        _units(pf_current=0),
        group="ablation",
        purpose="remove PF-current penalty",
    ),
    Variant(
        "no_pf_relations",
        _units(),
        use_relations=False,
        group="pf_structure",
        purpose="remove soft structural PF-relation rows",
    ),
    Variant(
        "no_pf_penalty_or_relations",
        _units(pf_current=0),
        use_relations=False,
        group="pf_structure",
        purpose="remove both PF controls",
    ),
    Variant(
        "zero_vcurrt",
        _units(),
        wall_current_mode="disabled",
        group="passive",
        purpose="zero VCURRT, retain passive response",
    ),
    Variant(
        "no_passive_response",
        _units(),
        passive_structure_mode="disabled",
        group="passive",
        purpose="disable passive response",
    ),
    Variant("flux_loops_x10", _units(flux_loop=10), group="strength"),
    Variant("flux_loops_x100", _units(flux_loop=100), group="strength"),
    Variant("bpol_probes_x10", _units(bpol_probe=10), group="strength"),
    Variant("bpol_probes_x100", _units(bpol_probe=100), group="strength"),
    Variant("diamagnetic_flux_x10", _units(diamagnetic_flux=10), group="strength"),
    Variant("diamagnetic_flux_x100", _units(diamagnetic_flux=100), group="strength"),
    *(
        Variant(
            f"diamagnetic_flux_x{scale}",
            _units(diamagnetic_flux=scale),
            group="strength",
        )
        for scale in DIAMAGNETIC_FINE_SCALES
    ),
    Variant("ip_x0p1", _units(plasma_current=0.1), group="strength"),
    Variant("ip_x0p01", _units(plasma_current=0.01), group="strength"),
    Variant("pf_penalty_x0p1", _units(pf_current=0.1), group="strength"),
    Variant("pf_penalty_x0p01", _units(pf_current=0.01), group="strength"),
    Variant(
        "ip_only",
        _units(flux_loop=0, bpol_probe=0, diamagnetic_flux=0),
        group="build_up",
    ),
    Variant(
        "flux_loops_only",
        _units(plasma_current=0, bpol_probe=0, diamagnetic_flux=0),
        group="build_up",
    ),
    Variant(
        "bpol_probes_only",
        _units(plasma_current=0, flux_loop=0, diamagnetic_flux=0),
        group="build_up",
    ),
    Variant(
        "diamagnetic_flux_only",
        _units(plasma_current=0, flux_loop=0, bpol_probe=0),
        group="build_up",
    ),
    Variant(
        "flux_loops_plus_bpol",
        _units(plasma_current=0, diamagnetic_flux=0),
        group="build_up",
    ),
    Variant(
        "ip_plus_flux_loops_plus_bpol", _units(diamagnetic_flux=0), group="build_up"
    ),
)

CONFIRMATION_NAMES = {
    BASELINE,
    "no_ip",
    "no_flux_loops",
    "no_bpol_probes",
    "no_diamagnetic_flux",
    "no_pf_penalty",
}


def _module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def selected_variants(matrix: str, names: str | None = None) -> tuple[Variant, ...]:
    variants = COMPLETE_VARIANTS
    if matrix == "confirmation":
        variants = tuple(item for item in variants if item.name in CONFIRMATION_NAMES)
    if names:
        requested = [value.strip() for value in names.split(",") if value.strip()]
        by_name = {item.name: item for item in COMPLETE_VARIANTS}
        unknown = sorted(set(requested) - set(by_name))
        if unknown:
            raise ValueError(f"unknown variant(s): {', '.join(unknown)}")
        variants = tuple(by_name[name] for name in requested)
        if BASELINE not in requested:
            variants = (by_name[BASELINE],) + variants
    return variants


def scientific_for(variant: Variant, profile_study: Any):
    base = profile_study.fixed_scientific_config()
    constraints = replace(
        base.constraints,
        group_weights={},
        objective_scales=variant.scales,
        use_coil_relation_constraints=variant.use_relations,
        wall_current_mode=variant.wall_current_mode,
        passive_structure_mode=variant.passive_structure_mode,
    )
    return replace(base, constraints=constraints)


def _array(data: Any, name: str) -> np.ndarray:
    if name not in data:
        return np.asarray([], dtype=float)
    return np.asarray(data[name].data, dtype=float).reshape(-1)


def _json_values(values: np.ndarray) -> list[float | None]:
    return [float(value) if math.isfinite(float(value)) else None for value in values]


def _metric(values: Iterable[float], *, absolute: bool = False) -> dict[str, Any]:
    array = np.asarray(list(values), dtype=float)
    array = array[np.isfinite(array)]
    if absolute:
        array = np.abs(array)
    if not array.size:
        return {
            "n": 0,
            "bias": None,
            "rms": None,
            "median_absolute": None,
            "max_absolute": None,
        }
    return {
        "n": int(array.size),
        "bias": float(array.mean()),
        "rms": float(np.sqrt(np.mean(array**2))),
        "median_absolute": float(np.median(np.abs(array))),
        "max_absolute": float(np.max(np.abs(array))),
    }


def _family_record(data: Any, family: str, scale: float) -> dict[str, Any]:
    measured_name, reconstructed_name, sigma_name, weight_name, chi_name = FAMILIES[
        family
    ]
    measured = _array(data, measured_name)
    reconstructed = _array(data, reconstructed_name)
    uncertainty = _array(data, sigma_name)
    weight = _array(data, weight_name)
    chi = _array(data, chi_name)
    count = min(
        measured.size, reconstructed.size, uncertainty.size, weight.size, chi.size
    )
    measured, reconstructed = measured[:count], reconstructed[:count]
    uncertainty, weight, chi = uncertainty[:count], weight[:count], chi[:count]
    residual = reconstructed - measured
    active = np.isfinite(weight) & (weight != 0.0)
    finite_residual = residual[active & np.isfinite(residual)]
    recomputed = np.square(residual * weight)
    return {
        "measured": _json_values(measured),
        "reconstructed": _json_values(reconstructed),
        "physical_residual": _json_values(residual),
        "uncertainty": _json_values(uncertainty),
        "objective_scale": float(scale),
        "processed_weight": _json_values(weight),
        "chi2_per_channel": _json_values(chi),
        "chi2_recomputed_per_channel": _json_values(recomputed),
        "active_channels": int(active.sum()),
        "residual_metrics": _metric(finite_residual),
        "uncertainty_metrics": _metric(uncertainty[active], absolute=True),
        "processed_weight_metrics": _metric(weight[active], absolute=True),
        "chi2_sum": float(np.nansum(chi)),
        "chi2_recomputed_sum": float(np.nansum(recomputed[active])),
    }


def _read_kfile(path: Path) -> Mapping[str, Any]:
    import f90nml

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return f90nml.read(path)["in1"]


def ip_accounting_record(
    *,
    ip_measured: float,
    ipmhd: float | None,
    vessel_current_sum: float,
    sigma_ip: float | None,
    chipasma_reported: float,
) -> dict[str, float | None]:
    """Reproduce EFIT's post-fit Ip accounting, including fixed VCURRT."""
    predicted = None
    predicted_from_vcurrt = None
    if sigma_ip not in (None, 0.0):
        predicted_from_vcurrt = (vessel_current_sum / float(sigma_ip)) ** 2
    if ipmhd is not None and sigma_ip not in (None, 0.0):
        predicted = ((ip_measured - ipmhd - vessel_current_sum) / float(sigma_ip)) ** 2
    relative_error = None
    if predicted_from_vcurrt is not None and chipasma_reported != 0.0:
        relative_error = abs(predicted_from_vcurrt - chipasma_reported) / abs(
            chipasma_reported
        )
    return {
        "ip_measured_kfile": ip_measured,
        "ipmhd_afile": ipmhd,
        "ipmhd_minus_ipmeas": None if ipmhd is None else ipmhd - ip_measured,
        "vessel_current_sum": vessel_current_sum,
        "sigma_ip": sigma_ip,
        "chipasma_reported": chipasma_reported,
        "chipasma_predicted_from_vcurrt_only": predicted_from_vcurrt,
        "chipasma_predicted_with_printed_ip_residual": predicted,
        "prediction_relative_error": relative_error,
    }


def audit_mfile(
    path: Path,
    *,
    kfile: Path,
    afile: Mapping[str, Any] | None,
    scientific: Any,
) -> dict[str, Any]:
    from vaft.data import read_meqdsk

    data = read_meqdsk(path)
    families = {
        name: _family_record(data, name, scientific.constraints.objective_scales[name])
        for name in FAMILIES
    }
    total = float(sum(item["chi2_sum"] for item in families.values()))
    non_ip = total - families["plasma_current"]["chi2_sum"]
    for name, item in families.items():
        item["share_total_chi2"] = None if total == 0.0 else item["chi2_sum"] / total
        item["share_non_ip_chi2"] = (
            None
            if name == "plasma_current" or non_ip == 0.0
            else item["chi2_sum"] / non_ip
        )

    kdata = _read_kfile(kfile)
    vcurrt = np.asarray(kdata.get("vcurrt", []), dtype=float).reshape(-1)
    ip_measured = float(kdata["plasma"])
    ipmhd = None
    if afile is not None:
        value = afile.get("scalars", {}).get("ipmhd")
        if value is not None and math.isfinite(float(value)):
            ipmhd = float(value)
    sigma = _array(data, "sigpasma")
    sigma_ip = float(sigma[0]) if sigma.size and sigma[0] != 0 else None
    wall_sum = float(np.sum(vcurrt))
    reported = families["plasma_current"]["chi2_sum"]
    ip_accounting = ip_accounting_record(
        ip_measured=ip_measured,
        ipmhd=ipmhd,
        vessel_current_sum=wall_sum,
        sigma_ip=sigma_ip,
        chipasma_reported=reported,
    )

    pf_reconstructed = np.asarray(
        [
            value if value is not None else np.nan
            for value in families["pf_current"]["reconstructed"]
        ],
        dtype=float,
    )
    relation = np.asarray([], dtype=float)
    matrix = np.asarray(scientific.constraints.coil_constraint_matrix, dtype=float)
    targets = np.asarray(scientific.constraints.coil_constraint_targets, dtype=float)
    if pf_reconstructed.size == matrix.shape[0]:
        relation = matrix.T @ pf_reconstructed - targets

    return {
        "path": path.name,
        "families": families,
        "family_chi2_total": total,
        "family_chi2_excluding_ip": non_ip,
        "mfile_totals": {
            name: (
                float(np.nansum(_array(data, name)))
                if _array(data, name).size
                else None
            )
            for name in ("chifin", "chitot")
        },
        "ip_accounting": ip_accounting,
        "pf_relations": {
            "enabled": scientific.constraints.use_coil_relation_constraints,
            "residual": _json_values(relation),
            "metrics": _metric(relation),
        },
    }


def enrich_run(run: dict[str, Any], workdir: Path, shot: int, scientific: Any) -> None:
    for item in run["slices"]:
        if item.get("mfile") is None:
            item["constraint_audit"] = None
            continue
        filename = item["mfile"]["path"]
        suffix = filename.split(f"m0{shot}", 1)[1]
        kfile = workdir / "kfile" / f"k0{shot}{suffix}"
        mfile = workdir / filename
        if not kfile.is_file() or not mfile.is_file():
            item["constraint_audit"] = None
            continue
        item["constraint_audit"] = audit_mfile(
            mfile, kfile=kfile, afile=item.get("afile"), scientific=scientific
        )


def summarize(run: Mapping[str, Any], profile_study: Any) -> dict[str, Any]:
    summary = profile_study.summarize_run(run)
    plasma = [
        item
        for item in run["slices"]
        if item["phase"] in PLASMA_PHASES and item.get("constraint_audit")
    ]
    summary["constraint_families"] = {}
    for family in FAMILIES:
        records = [item["constraint_audit"]["families"][family] for item in plasma]
        summary["constraint_families"][family] = {
            "chi2_sum": profile_study.spread(item["chi2_sum"] for item in records),
            "share_total_chi2": profile_study.spread(
                item["share_total_chi2"] for item in records
            ),
            "share_non_ip_chi2": profile_study.spread(
                item["share_non_ip_chi2"] for item in records
            ),
            "residual_rms": profile_study.spread(
                item["residual_metrics"]["rms"] for item in records
            ),
            "processed_weight_median": profile_study.spread(
                item["processed_weight_metrics"]["median_absolute"] for item in records
            ),
            "active_channels": profile_study.spread(
                item["active_channels"] for item in records
            ),
        }
    summary["ip_accounting"] = {
        "prediction_relative_error": profile_study.spread(
            item["constraint_audit"]["ip_accounting"]["prediction_relative_error"]
            for item in plasma
        ),
        "ipmhd_minus_ipmeas": profile_study.spread(
            (
                item["constraint_audit"]["ip_accounting"]["ipmhd_minus_ipmeas"]
                for item in plasma
            ),
            absolute=True,
        ),
        "vessel_current_sum": profile_study.spread(
            (
                item["constraint_audit"]["ip_accounting"]["vessel_current_sum"]
                for item in plasma
            ),
            absolute=True,
        ),
    }
    summary["pf_relation_residual"] = profile_study.spread(
        item["constraint_audit"]["pf_relations"]["metrics"]["rms"] for item in plasma
    )
    return summary


def _safe_relative(after: float | None, before: float | None) -> float | None:
    if (
        after is None
        or before is None
        or not math.isfinite(float(after))
        or not math.isfinite(float(before))
    ):
        return None
    if float(before) == 0.0:
        return 0.0 if float(after) == 0.0 else 1.0
    return abs(float(after) - float(before)) / abs(float(before))


def comparison(
    candidate: Mapping[str, Any], baseline: Mapping[str, Any], profile_study: Any
) -> dict[str, Any]:
    geometry = profile_study.compare_runs(candidate, baseline)
    baseline_by_time = {int(item["time_ms"]): item for item in baseline["slices"]}
    family_changes: dict[str, list[float]] = {name: [] for name in FAMILIES}
    residual_changes: dict[str, list[float]] = {name: [] for name in FAMILIES}
    baseline_chi2: dict[str, list[float]] = {name: [] for name in FAMILIES}
    pf_current_changes = []
    for item in candidate["slices"]:
        reference = baseline_by_time.get(int(item["time_ms"]))
        if not reference or item["phase"] not in PLASMA_PHASES:
            continue
        left, right = item.get("constraint_audit"), reference.get("constraint_audit")
        if left is None or right is None:
            continue
        for family in FAMILIES:
            a, b = left["families"][family], right["families"][family]
            baseline_chi2[family].append(float(b["chi2_sum"]))
            change = _safe_relative(a["chi2_sum"], b["chi2_sum"])
            residual = _safe_relative(
                a["residual_metrics"]["rms"], b["residual_metrics"]["rms"]
            )
            if change is not None:
                family_changes[family].append(change)
            if residual is not None:
                residual_changes[family].append(residual)
        before = np.asarray(
            [
                v
                for v in right["families"]["pf_current"]["reconstructed"]
                if v is not None
            ]
        )
        after = np.asarray(
            [
                v
                for v in left["families"]["pf_current"]["reconstructed"]
                if v is not None
            ]
        )
        if before.size and before.size == after.size:
            denominator = float(np.sqrt(np.mean(before**2)))
            if denominator:
                pf_current_changes.append(
                    float(np.sqrt(np.mean((after - before) ** 2))) / denominator
                )

    baseline_plasma = [
        item for item in baseline["slices"] if item["phase"] in PLASMA_PHASES
    ]
    candidate_plasma = [
        item for item in candidate["slices"] if item["phase"] in PLASMA_PHASES
    ]
    base_rate = sum(item["outcome"] == "accepted" for item in baseline_plasma) / max(
        len(baseline_plasma), 1
    )
    candidate_rate = sum(
        item["outcome"] == "accepted" for item in candidate_plasma
    ) / max(len(candidate_plasma), 1)
    result = {
        "geometry": geometry,
        "acceptance_change_percentage_points": 100.0 * (candidate_rate - base_rate),
        "family_chi2_absolute_relative_change": {
            name: profile_study.spread(values)
            for name, values in family_changes.items()
        },
        "family_residual_rms_absolute_relative_change": {
            name: profile_study.spread(values)
            for name, values in residual_changes.items()
        },
        "baseline_family_chi2": {
            name: profile_study.spread(values) for name, values in baseline_chi2.items()
        },
        "pf_reconstructed_current_relative_rms_change": profile_study.spread(
            pf_current_changes
        ),
    }
    result["material_response"] = material_response(result)
    return result


def material_response(
    item: Mapping[str, Any], *, excluded_family: str | None = None
) -> bool:
    geometry = item["geometry"]
    lcfs = geometry["lcfs_rms_mm"]["median"] or 0.0
    area = geometry["absolute_relative_change"]["area"]["median"] or 0.0
    volume = geometry["absolute_relative_change"]["volume"]["median"] or 0.0
    acceptance = abs(float(item["acceptance_change_percentage_points"]))
    other = []
    for group in (
        "family_chi2_absolute_relative_change",
        "family_residual_rms_absolute_relative_change",
    ):
        for family, values in item[group].items():
            if family != excluded_family and values["median"] is not None:
                if (
                    group == "family_chi2_absolute_relative_change"
                    and (
                        item.get("baseline_family_chi2", {})
                        .get(family, {})
                        .get("median")
                        or 0.0
                    )
                    < CHI2_NUMERICAL_FLOOR
                ):
                    continue
                other.append(float(values["median"]))
    return (
        lcfs >= 5.0
        or area >= 0.02
        or volume >= 0.02
        or acceptance >= 10.0
        or any(value >= 0.20 for value in other)
    )


ABLATIONS = {
    "plasma_current": "no_ip",
    "flux_loop": "no_flux_loops",
    "bpol_probe": "no_bpol_probes",
    "diamagnetic_flux": "no_diamagnetic_flux",
    "pf_current": "no_pf_penalty",
}
HIGH_STRENGTH = {
    "flux_loop": "flux_loops_x100",
    "bpol_probe": "bpol_probes_x100",
    "diamagnetic_flux": "diamagnetic_flux_x10000",
}


def classify(shot_block: Mapping[str, Any]) -> dict[str, Any]:
    baseline_summary = shot_block["variants"][BASELINE]["summary"]
    comparisons = shot_block["comparisons"]
    result = {}
    for family, ablation in ABLATIONS.items():
        labels = []
        ablation_result = comparisons.get(ablation)
        ablation_available = ablation_result is not None
        discriminating = bool(
            ablation_result
            and material_response(ablation_result, excluded_family=family)
        )
        if discriminating:
            labels.append("discriminating")
        base_share = baseline_summary["constraint_families"][family][
            "share_total_chi2"
        ]["median"]
        high = comparisons.get(HIGH_STRENGTH.get(family, ""))
        if (
            base_share is not None
            and base_share < 0.01
            and high
            and material_response(high, excluded_family=family)
        ):
            labels.append("overwhelmed")
        elif (
            ablation_available
            and not discriminating
            and (high is None or not material_response(high, excluded_family=family))
        ):
            labels.append("inactive/redundant")
        if family == "pf_current":
            pf_tests = [
                comparisons.get(name)
                for name in (
                    "no_pf_penalty",
                    "no_pf_relations",
                    "no_pf_penalty_or_relations",
                )
            ]
            if any(
                item
                and (
                    (
                        item["pf_reconstructed_current_relative_rms_change"]["median"]
                        or 0.0
                    )
                    >= 0.02
                    or (item["geometry"]["lcfs_rms_mm"]["median"] or 0.0) >= 5.0
                    or abs(float(item["acceptance_change_percentage_points"])) >= 10.0
                )
                for item in pf_tests
            ):
                labels.append("structural anchor")
        if family == "plasma_current":
            identity = baseline_summary["ip_accounting"]["prediction_relative_error"][
                "max"
            ]
            passive = shot_block.get("passive_attribution", {})
            passive_agrees = passive.get("agreement", True)
            # NetCDF stores chipasma at float32 precision; the three-shot
            # verification ceiling is 5.02e-8 after that quantization.
            if identity is not None and identity < 5.1e-8 and passive_agrees:
                labels.append("accounting-confounded")
        result[family] = {
            "classification": labels or ["unclassified"],
            "baseline_chi2_share_median": base_share,
            "ablation": ablation,
            "ablation_material_response": discriminating,
            "high_strength_test": HIGH_STRENGTH.get(family),
        }
    return result


def markdown(payload: Mapping[str, Any]) -> str:
    def cell(value: Any) -> str:
        return "–" if value is None else f"{float(value):.6g}"

    lines = ["# EFIT constraint-information study (#663)", ""]
    lines.append(
        "All variants retain the issue-#579 `(2,2)` zero-edge model, qualified seed, numerical controls, input data, grid, and cadence."
    )
    lines.extend(
        [
            "",
            "## Baseline objective audit",
            "",
            "| shot | family | chi² median | total share median | residual RMS median | active channels median |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for shot, block in payload["shots"].items():
        base = block["variants"][BASELINE]["summary"]
        for family, item in base["constraint_families"].items():
            lines.append(
                f"| {shot} | `{family}` | {cell(item['chi2_sum']['median'])} | {cell(item['share_total_chi2']['median'])} | {cell(item['residual_rms']['median'])} | {cell(item['active_channels']['median'])} |"
            )
    lines.extend(
        [
            "",
            "## Classification",
            "",
            "| shot | family | classification | baseline share | ablation material? |",
            "|---|---|---|---:|---|",
        ]
    )
    for shot, block in payload["shots"].items():
        for family, item in block["classification"].items():
            lines.append(
                f"| {shot} | `{family}` | {', '.join(item['classification'])} | {cell(item['baseline_chi2_share_median'])} | {item['ablation_material_response']} |"
            )
    lines.extend(
        [
            "",
            "## Variant effects",
            "",
            "| shot | variant | plasma produced/requested | acceptance Δ [pp] | LCFS RMS [mm] | |Δarea| | |Δvolume| |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for shot, block in payload["shots"].items():
        for name, candidate in block["variants"].items():
            if name == BASELINE:
                continue
            summary = candidate["summary"]
            item = block["comparisons"][name]
            geom = item["geometry"]
            lines.append(
                f"| {shot} | `{name}` | {summary['plasma_produced']}/{summary['plasma_requested']} | {cell(item['acceptance_change_percentage_points'])} | {cell(geom['lcfs_rms_mm']['median'])} | {cell(geom['absolute_relative_change']['area']['median'])} | {cell(geom['absolute_relative_change']['volume']['median'])} |"
            )
    lines.extend(
        [
            "",
            "## Ip/VCURRT accounting",
            "",
            "| shot | max relative prediction error | median |ipmhd-ipmeas| [A] | median |ΣVCURRT| [A] |",
            "|---|---:|---:|---:|",
        ]
    )
    for shot, block in payload["shots"].items():
        item = block["variants"][BASELINE]["summary"]["ip_accounting"]
        lines.append(
            f"| {shot} | {cell(item['prediction_relative_error']['max'])} | {cell(item['ipmhd_minus_ipmeas']['median'])} | {cell(item['vessel_current_sum']['median'])} |"
        )
    passive_rows = [
        (shot, block.get("passive_attribution"))
        for shot, block in payload["shots"].items()
        if block.get("passive_attribution")
    ]
    if passive_rows:
        lines.extend(
            [
                "",
                "The zero-`VCURRT` and fully disabled passive-response variants "
                + (
                    "agree within the material-response thresholds."
                    if all(item["agreement"] for _, item in passive_rows)
                    else "do not agree within the material-response thresholds; sole attribution to `VCURRT` is therefore withheld."
                ),
                "",
            ]
        )
    lines.extend(
        [
            "",
            "Family chi² is the direct sum of EFIT's per-channel m-file arrays. Shares use the sum of the five audited families; the non-Ip shares are also retained in JSON. Classification thresholds are encoded in `material_response()`.",
            "",
        ]
    )
    return "\n".join(lines)


def plots(payload: Mapping[str, Any], output: Path) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    created = []
    colors = dict(zip(FAMILIES, plt.cm.tab10.colors))
    for shot, block in payload["shots"].items():
        rows = [
            item
            for item in block["variants"][BASELINE]["run"]["slices"]
            if item.get("constraint_audit")
        ]
        times = np.asarray([item["time_ms"] for item in rows])
        fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
        ax = axes[0]
        bottom = np.zeros(times.size)
        by_family = {}
        for family in FAMILIES:
            values = np.asarray(
                [
                    max(item["constraint_audit"]["families"][family]["chi2_sum"], 1e-30)
                    for item in rows
                ]
            )
            by_family[family] = values
            ax.fill_between(
                times,
                bottom + 1e-30,
                bottom + values,
                label=family,
                color=colors[family],
                alpha=0.8,
            )
            bottom += values
        ax.set_yscale("log")
        ax.set_ylabel("family chi²")
        ax.legend(ncol=3, fontsize=8)
        ax.set_title(f"Shot {shot}: stacked objective contributions")
        for family in FAMILIES:
            if family != "plasma_current":
                axes[1].plot(
                    times,
                    by_family[family],
                    label=family,
                    color=colors[family],
                )
        axes[1].set_yscale("log")
        axes[1].set_xlabel("time [ms]")
        axes[1].set_ylabel("non-Ip family chi²")
        axes[1].legend(ncol=2, fontsize=8)
        path = output / f"shot_{shot}_chi2_contributions.png"
        fig.tight_layout()
        fig.savefig(path, dpi=160)
        plt.close(fig)
        created.append(path.name)

    first_shot = next(iter(payload["shots"]))
    block = payload["shots"][first_shot]
    rows = [
        item
        for item in block["variants"][BASELINE]["run"]["slices"]
        if item.get("constraint_audit") and item["phase"] in PLASMA_PHASES
    ]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    residual_data, weight_data, labels = [], [], []
    for family in FAMILIES:
        residuals, weights = [], []
        for row in rows:
            item = row["constraint_audit"]["families"][family]
            residuals.extend(abs(v) for v in item["physical_residual"] if v is not None)
            weights.extend(
                abs(v) for v in item["processed_weight"] if v not in (None, 0.0)
            )
        residual_data.append(np.maximum(residuals, 1e-30))
        weight_data.append(np.maximum(weights, 1e-30))
        labels.append(family.replace("_", "\n"))
    axes[0].boxplot(residual_data, tick_labels=labels, showfliers=False)
    axes[0].set_yscale("log")
    axes[0].set_ylabel("|physical residual|")
    axes[1].boxplot(weight_data, tick_labels=labels, showfliers=False)
    axes[1].set_yscale("log")
    axes[1].set_ylabel("processed m-file weight")
    path = output / "residual_and_weight_distributions.png"
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    created.append(path.name)

    ablations = [name for name in ABLATIONS.values() if name in block["comparisons"]]
    heat = []
    for name in ablations:
        item = block["comparisons"][name]
        geom = item["geometry"]
        heat.append(
            [
                (geom["lcfs_rms_mm"]["median"] or 0) / 5,
                (geom["absolute_relative_change"]["area"]["median"] or 0) / 0.02,
                (geom["absolute_relative_change"]["volume"]["median"] or 0) / 0.02,
                abs(item["acceptance_change_percentage_points"]) / 10,
            ]
        )
    fig, ax = plt.subplots(figsize=(8, max(3, 0.55 * len(ablations))))
    if heat:
        image = ax.imshow(heat, aspect="auto", cmap="magma")
        ax.set_xticks(
            range(4), ["LCFS / 5 mm", "area / 2%", "volume / 2%", "acceptance / 10 pp"]
        )
        ax.set_yticks(range(len(ablations)), ablations)
        fig.colorbar(image, ax=ax, label="classification-threshold multiple")
    else:
        ax.text(0.5, 0.5, "No ablation variants selected", ha="center", va="center")
        ax.set_axis_off()
    path = output / "constraint_ablation_heatmap.png"
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    created.append(path.name)

    fig, ax = plt.subplots(figsize=(9, 5))
    for shot, shot_block in payload["shots"].items():
        records = [
            item
            for item in shot_block["variants"][BASELINE]["run"]["slices"]
            if item.get("constraint_audit")
        ]
        ax.plot(
            [item["time_ms"] for item in records],
            [
                item["constraint_audit"]["ip_accounting"]["chipasma_reported"]
                for item in records
            ],
            label=f"{shot} reported",
        )
        ax.plot(
            [item["time_ms"] for item in records],
            [
                item["constraint_audit"]["ip_accounting"][
                    "chipasma_predicted_from_vcurrt_only"
                ]
                for item in records
            ],
            "--",
            label=f"{shot} VCURRT prediction",
        )
    ax.set_yscale("log")
    ax.set_xlabel("time [ms]")
    ax.set_ylabel("chipasma")
    ax.legend(fontsize=8, ncol=2)
    path = output / "ip_vcurrt_accounting.png"
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    created.append(path.name)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    if rows:
        middle = rows[len(rows) // 2]["constraint_audit"]
        pf = middle["families"]["pf_current"]
        axes[0].plot(pf["measured"], "o-", label="measured")
        axes[0].plot(pf["reconstructed"], "x--", label="reconstructed")
        axes[0].set_xlabel("PF channel")
        axes[0].set_ylabel("current [A]")
        axes[0].legend()
        relation_times = [item["time_ms"] for item in rows]
        relation_rms = [
            item["constraint_audit"]["pf_relations"]["metrics"]["rms"] for item in rows
        ]
        axes[1].plot(relation_times, relation_rms)
        if any(value not in (None, 0.0) for value in relation_rms):
            axes[1].set_yscale("symlog", linthresh=1e-12)
        else:
            axes[1].set_ylim(-1.0, 1.0)
            axes[1].text(
                0.5,
                0.6,
                "relation residual is exactly zero",
                ha="center",
                transform=axes[1].transAxes,
            )
        axes[1].set_xlabel("time [ms]")
        axes[1].set_ylabel("RMS CCOILSᵀI-XCOILS [A]")
    path = output / "pf_currents_and_relations.png"
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    created.append(path.name)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for shot, shot_block in payload["shots"].items():
        scales = []
        produced = []
        acceptance_change = []
        lcfs = []
        for scale in DIAMAGNETIC_FINE_SCALES:
            name = f"diamagnetic_flux_x{scale}"
            if name not in shot_block["variants"]:
                continue
            scales.append(scale)
            produced.append(shot_block["variants"][name]["summary"]["plasma_produced"])
            candidate = shot_block["comparisons"][name]
            acceptance_change.append(candidate["acceptance_change_percentage_points"])
            lcfs.append(candidate["geometry"]["lcfs_rms_mm"]["median"])
        if not scales:
            continue
        axes[0].plot(scales, produced, "o-", label=f"{shot} produced")
        axes[0].plot(
            scales,
            acceptance_change,
            "x--",
            label=f"{shot} acceptance Δ [pp]",
        )
        axes[1].plot(
            scales,
            [np.nan if value is None else max(value, 1e-12) for value in lcfs],
            "o-",
            label=shot,
        )
    axes[0].axhline(0.0, color="black", linewidth=0.6)
    axes[0].set_xlabel("diamagnetic objective scale")
    axes[0].set_ylabel("count or percentage points")
    axes[0].legend(fontsize=8)
    axes[1].set_yscale("log")
    axes[1].set_xlabel("diamagnetic objective scale")
    axes[1].set_ylabel("median common-slice LCFS shift [mm]")
    axes[1].legend(fontsize=8)
    path = output / "diamagnetic_strength_sweep.png"
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    created.append(path.name)
    return created


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--shots", default="41672")
    parser.add_argument(
        "--matrix", choices=("complete", "confirmation"), default="complete"
    )
    parser.add_argument("--variants", default=None)
    parser.add_argument("--tables", default=None)
    parser.add_argument("--packaged-envelope", action="store_true")
    parser.add_argument("--efit-home", default=None)
    parser.add_argument("--tstep", type=float, default=0.001)
    parser.add_argument("--average-window", type=float, default=0.0005)
    args = parser.parse_args(argv)
    if args.efit_home:
        os.environ["EFITHOME"] = str(Path(args.efit_home).expanduser())

    from vaft.code.efit.toolchain import resolve_toolchain, toolchain_identities
    from vaft.data.resources import data_path

    profile_study = _module(PROFILE_STUDY, "constraint_profile_support")
    seed_study = _module(SEED_STUDY, "constraint_seed_support")
    resolved = resolve_toolchain()
    if resolved.get("efit") is None:
        print("no efit executable: set EFITHOME", file=sys.stderr)
        return 2
    output = args.output.expanduser()
    output.mkdir(parents=True, exist_ok=True)
    source_tables = (
        Path(args.tables).expanduser()
        if args.tables
        else Path(data_path("efit")).resolve()
    )
    if args.packaged_envelope:
        tables, table_record = (
            source_tables,
            {"source": str(source_tables), "policy": "packaged envelope"},
        )
    else:
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
    variants = selected_variants(args.matrix, args.variants)
    base = profile_study.fixed_scientific_config()
    payload: dict[str, Any] = {
        "schema_version": SCHEMA,
        "issue": 663,
        "run_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "matrix": args.matrix,
        "toolchain": toolchain_identities(resolved),
        "table": table_record,
        "fixed_scientific": base.to_dict(),
        "fixed_scientific_sha256": base.sha256,
        "classification_thresholds": {
            "lcfs_mm": 5,
            "area_or_volume_relative": 0.02,
            "other_family_relative": 0.20,
            "acceptance_percentage_points": 10,
        },
        "variants": [
            {**item.__dict__, "scales": dict(item.scales)} for item in variants
        ],
        "shots": {},
    }
    for shot in [int(value) for value in args.shots.split(",")]:
        if shot not in products:
            raise ValueError(f"{shot}: no packaged pre-EFIT reference product")
        print(f"{shot}: building shared frozen constraints", flush=True)
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
        block: dict[str, Any] = {
            "window": {
                "start": float(window.start),
                "end": float(window.end),
                "requested": int(times.size),
            },
            "phase_dcurrent_dt_threshold": threshold,
            "variants": {},
            "comparisons": {},
        }
        for variant in variants:
            print(f"  {variant.name}: {variant.purpose or variant.group}", flush=True)
            scientific = scientific_for(variant, profile_study)
            workdir = output / f"shot_{shot}" / variant.name
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
            enrich_run(run, workdir, shot, scientific)
            block["variants"][variant.name] = {
                "specification": {**variant.__dict__, "scales": dict(variant.scales)},
                "summary": summarize(run, profile_study),
                "run": run,
            }
            summary = block["variants"][variant.name]["summary"]
            print(
                f"    {summary['plasma_produced']}/{summary['plasma_requested']} plasma outputs; {summary['seconds']:.1f} s",
                flush=True,
            )
        baseline_run = block["variants"][BASELINE]["run"]
        for variant in variants:
            if variant.name != BASELINE:
                block["comparisons"][variant.name] = comparison(
                    block["variants"][variant.name]["run"], baseline_run, profile_study
                )
        if {"zero_vcurrt", "no_passive_response"} <= set(block["variants"]):
            passive_comparison = comparison(
                block["variants"]["no_passive_response"]["run"],
                block["variants"]["zero_vcurrt"]["run"],
                profile_study,
            )
            block["passive_attribution"] = {
                "comparison": passive_comparison,
                "agreement": not material_response(passive_comparison),
            }
        block["classification"] = classify(block)
        payload["shots"][str(shot)] = block
        (output / "constraint_information.json").write_text(
            json.dumps(payload, indent=1, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
    payload["plots"] = plots(payload, output)
    (output / "constraint_information.json").write_text(
        json.dumps(payload, indent=1, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    report = markdown(payload)
    (output / "constraint_information.md").write_text(report, encoding="utf-8")
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
