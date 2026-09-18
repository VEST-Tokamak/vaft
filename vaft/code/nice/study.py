"""Whole-window and matched-slice analysis helpers for NICE/EFIT studies."""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from .config import NiceConfig, NiceDiagnostic, NiceResult
from .inputs import prepare_nice_inputs
from .runner import run_nice


def physical_channel_signature(
    channels: Iterable[NiceDiagnostic],
) -> tuple[tuple[Any, ...], ...]:
    """Stable identity of a physical diagnostic set, excluding values/weights."""
    return tuple(
        sorted(
            (
                item.family,
                item.ods_path,
                item.identifier,
                bool(item.enabled),
                item.reason,
            )
            for item in channels
        )
    )


def assert_same_physical_channels(
    left: Iterable[NiceDiagnostic], right: Iterable[NiceDiagnostic]
) -> None:
    if physical_channel_signature(left) != physical_channel_signature(right):
        raise ValueError(
            "EFIT and NICE cases do not use the same physical diagnostic channel set"
        )


def compare_diagnostic_residuals(
    nice_residuals: Iterable[dict[str, Any]],
    efit_residuals: Iterable[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Match residuals by physical channel, never by solver array position."""

    def indexed(rows):
        return {
            (row.get("family"), row.get("ods_path") or row.get("identifier")): row
            for row in rows
        }

    nice, efit = indexed(nice_residuals), indexed(efit_residuals)
    if set(nice) != set(efit):
        raise ValueError(
            "NICE and EFIT residual tables contain different physical channels"
        )
    return [
        {
            "family": key[0],
            "channel": key[1],
            "nice_physical": float(nice[key]["residual"]),
            "efit_physical": float(efit[key]["residual"]),
            "nice_normalized": float(nice[key]["normalized_residual"]),
            "efit_normalized": float(efit[key]["normalized_residual"]),
        }
        for key in sorted(nice)
    ]


def constraint_family_configs(base: NiceConfig) -> dict[str, NiceConfig]:
    """The issue #666 family-addition cases (diamagnetics is provenance-only)."""
    return {
        "Core": replace(
            base,
            include_flux_loops=False,
            include_bpol_probes=False,
            include_diamagnetic_flux=False,
        ),
        "Core + flux loops": replace(
            base,
            include_flux_loops=True,
            include_bpol_probes=False,
            include_diamagnetic_flux=False,
        ),
        "Core + B-pol": replace(
            base,
            include_flux_loops=False,
            include_bpol_probes=True,
            include_diamagnetic_flux=False,
        ),
        "Core + flux loops + B-pol": replace(
            base,
            include_flux_loops=True,
            include_bpol_probes=True,
            include_diamagnetic_flux=False,
        ),
        "Core + diamagnetic": replace(
            base,
            include_flux_loops=False,
            include_bpol_probes=False,
            include_diamagnetic_flux=True,
        ),
        "Full": replace(
            base,
            include_flux_loops=True,
            include_bpol_probes=True,
            include_diamagnetic_flux=True,
        ),
    }


def _slice_index(ods: Any, time: float) -> int:
    values = np.asarray(ods["equilibrium.time"], float)
    return int(np.argmin(np.abs(values - time)))


def _value(ods: Any, path: str, default=float("nan")) -> float:
    try:
        return float(ods[path])
    except Exception:
        return float(default)


def _resample_closed(r, z, count: int = 256) -> np.ndarray:
    points = np.column_stack((np.asarray(r, float), np.asarray(z, float)))
    if len(points) < 3:
        return np.empty((0, 2))
    points = np.vstack((points, points[0]))
    length = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
    if length[-1] == 0:
        return np.empty((0, 2))
    target = np.linspace(0.0, length[-1], count, endpoint=False)
    return np.column_stack(
        (
            np.interp(target, length, points[:, 0]),
            np.interp(target, length, points[:, 1]),
        )
    )


def lcfs_rms_displacement(left_r, left_z, right_r, right_z) -> float:
    """Orientation/starting-index invariant RMS LCFS point displacement [m]."""
    left, right = _resample_closed(left_r, left_z), _resample_closed(right_r, right_z)
    if not len(left) or not len(right):
        return float("nan")
    candidates = []
    for curve in (right, right[::-1]):
        candidates.extend(
            np.sqrt(
                np.mean(np.sum((left - np.roll(curve, shift, axis=0)) ** 2, axis=1))
            )
            for shift in range(len(curve))
        )
    return float(min(candidates))


def compare_equilibria(nice_ods: Any, efit_ods: Any, time: float) -> dict[str, Any]:
    """Compare common physical quantities on the closest matched slice."""
    ni, ei = _slice_index(nice_ods, time), _slice_index(efit_ods, time)
    nb, eb = f"equilibrium.time_slice.{ni}", f"equilibrium.time_slice.{ei}"
    paths = {
        "axis_r_m": "global_quantities.magnetic_axis.r",
        "axis_z_m": "global_quantities.magnetic_axis.z",
        "ip_A": "global_quantities.ip",
        "beta_pol": "global_quantities.beta_pol",
        "li_3": "global_quantities.li_3",
        "area_m2": "global_quantities.area",
        "volume_m3": "global_quantities.volume",
        "stored_energy_J": "global_quantities.energy_mhd",
        "elongation": "boundary.elongation",
        "triangularity_upper": "boundary.triangularity_upper",
        "triangularity_lower": "boundary.triangularity_lower",
    }
    values = {}
    for name, leaf in paths.items():
        n, e = _value(nice_ods, f"{nb}.{leaf}"), _value(efit_ods, f"{eb}.{leaf}")
        values[name] = {"nice": n, "efit": e, "difference": n - e}
    try:
        values["lcfs_rms_m"] = lcfs_rms_displacement(
            nice_ods[f"{nb}.boundary.outline.r"],
            nice_ods[f"{nb}.boundary.outline.z"],
            efit_ods[f"{eb}.boundary.outline.r"],
            efit_ods[f"{eb}.boundary.outline.z"],
        )
    except Exception:
        values["lcfs_rms_m"] = float("nan")
    profiles = {}
    for name in ("pressure", "dpressure_dpsi", "f_df_dpsi", "j_tor", "q"):
        try:
            nv = np.asarray(nice_ods[f"{nb}.profiles_1d.{name}"], float)
            ev = np.asarray(efit_ods[f"{eb}.profiles_1d.{name}"], float)
            grid = np.linspace(0.0, 1.0, 101)
            delta = np.interp(grid, np.linspace(0, 1, len(nv)), nv) - np.interp(
                grid, np.linspace(0, 1, len(ev)), ev
            )
            profiles[name] = {
                "rms": float(np.sqrt(np.mean(delta**2))),
                "max_abs": float(np.max(np.abs(delta))),
            }
        except Exception:
            profiles[name] = {"rms": float("nan"), "max_abs": float("nan")}
    return {
        "time_s": float(time),
        "nice_index": ni,
        "efit_index": ei,
        "quantities": values,
        "profiles": profiles,
    }


def run_nice_window(
    ods: Any, times: Sequence[float], config: NiceConfig
) -> tuple[NiceResult, ...]:
    """Run every requested time and retain explicit failure records."""
    results = []
    for index, time in enumerate(times):
        case = replace(
            config,
            time=float(time),
            time_index=None,
            workdir=Path(config.workdir) / f"slice_{index:04d}_{time:.6f}",
        )
        try:
            results.append(run_nice(prepare_nice_inputs(ods, case), case))
        except Exception as exc:
            results.append(
                NiceResult(
                    None,
                    Path(case.workdir),
                    termination_reason=f"{type(exc).__name__}: {exc}",
                )
            )
    return tuple(results)


def summarize_window(
    times: Sequence[float], results: Sequence[NiceResult]
) -> dict[str, Any]:
    requested = len(times)
    produced = sum(result.ods is not None for result in results)
    converged = sum(result.converged is True for result in results)
    usable = sum(result.scientifically_usable is True for result in results)
    reasons = {}
    for result in results:
        reasons[result.termination_reason] = (
            reasons.get(result.termination_reason, 0) + 1
        )
    return {
        "requested": requested,
        "produced": produced,
        "converged": converged,
        "scientifically_usable": usable,
        "fractions": {
            key: value / requested if requested else 0.0
            for key, value in {
                "produced": produced,
                "converged": converged,
                "scientifically_usable": usable,
            }.items()
        },
        "failed_or_missing_times_s": [
            float(t)
            for t, r in zip(times, results)
            if r.ods is None or r.converged is not True
        ],
        "termination_reasons": reasons,
        "nonlinear_iterations": [
            r.nonlinear_iterations
            for r in results
            if r.nonlinear_iterations is not None
        ],
    }


def write_study_report(path: str | Path, payload: dict[str, Any]) -> Path:
    """Write a deterministic, machine-readable report used by plotting/notebooks."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=True), encoding="utf-8"
    )
    return target


def write_window_report(
    output_dir: str | Path,
    shot: int,
    times: Sequence[float],
    results: Sequence[NiceResult],
    efit_ods: Any | None = None,
) -> dict[str, Path]:
    """Generate the compact per-shot JSON summary and NICE/EFIT trace overlay."""
    import matplotlib.pyplot as plt

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    summary = summarize_window(times, results)
    first_provenance = next((r.provenance for r in results if r.provenance), {})
    summary["provenance"] = {
        key: first_provenance.get(key)
        for key in (
            "nice_source_revision",
            "vaft_revision",
            "input_snapshot_hash",
            "build_options",
            "profile_basis",
            "solver_tolerances",
            "initialization_method",
            "cocos_in",
            "cocos_out",
            "passive_current_treatment",
            "passive_current_hash",
            "diagnostic_channel_set_hash",
        )
    }
    summary["provenance"]["slice_native_input_hashes"] = [
        {
            "time_s": float(time),
            "native_input_hash": result.provenance.get("native_input_hash"),
        }
        for time, result in zip(times, results)
    ]
    comparisons = []
    if efit_ods is not None:
        for time, result in zip(times, results):
            if result.ods is not None:
                comparisons.append(
                    compare_equilibria(result.ods, efit_ods, float(time))
                )
    summary.update({"shot": int(shot), "slice_comparisons": comparisons})
    json_file = write_study_report(output / f"nice_{shot}_summary.json", summary)

    fields = (
        ("axis_r_m", "Magnetic axis R [m]"),
        ("axis_z_m", "Magnetic axis Z [m]"),
        ("ip_A", "Ip [A]"),
        ("beta_pol", "beta_p"),
        ("li_3", "li(3)"),
        ("area_m2", "Area [m2]"),
        ("volume_m3", "Volume [m3]"),
    )
    figure, axes = plt.subplots(4, 2, figsize=(10, 12), sharex=True)
    for axis, (field, label) in zip(axes.flat, fields):
        nt, nv, et, ev, ft, fv = [], [], [], [], [], []
        for time, result in zip(times, results):
            leaf = {
                "axis_r_m": "global_quantities.magnetic_axis.r",
                "axis_z_m": "global_quantities.magnetic_axis.z",
                "ip_A": "global_quantities.ip",
                "beta_pol": "global_quantities.beta_pol",
                "li_3": "global_quantities.li_3",
                "area_m2": "global_quantities.area",
                "volume_m3": "global_quantities.volume",
            }[field]
            if efit_ods is not None:
                eindex = _slice_index(efit_ods, float(time))
                et.append(float(time))
                ev.append(_value(efit_ods, f"equilibrium.time_slice.{eindex}.{leaf}"))
            if result.ods is None:
                if efit_ods is not None:
                    ft.append(float(time))
                    fv.append(ev[-1])
                continue
            index = _slice_index(result.ods, float(time))
            nt.append(float(time))
            nv.append(_value(result.ods, f"equilibrium.time_slice.{index}.{leaf}"))
        axis.plot(nt, nv, "o-", label="NICE")
        if et:
            axis.plot(et, ev, "s--", label="EFIT")
        if ft:
            axis.plot(ft, fv, "rx", label="NICE failed")
        axis.set_ylabel(label)
        axis.grid(True, alpha=0.3)
    axes.flat[-1].axis("off")
    axes.flat[0].legend()
    for axis in axes[-1, :]:
        axis.set_xlabel("time [s]")
    figure.suptitle(f"Shot {shot}: NICE / EFIT reconstruction overview")
    figure.tight_layout()
    plot_file = output / f"nice_{shot}_traces.png"
    figure.savefig(plot_file, dpi=150)
    plt.close(figure)
    return {"summary": json_file, "traces": plot_file}
