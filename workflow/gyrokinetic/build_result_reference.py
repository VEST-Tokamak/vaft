"""Reproduce native nonlinear fields and qualified diagnostics without a solver.

python -m workflow.gyrokinetic.build_result_reference --run RUN --out OUTPUT
Default RUN is the compact offline real-output fixture. A short fixture validates
execution/reconstruction, not saturated transport or convergence.
"""

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from vaft.code.gacode.cgyro import collect_cgyro_outputs, read_field_history
from vaft.code.gacode.output_audit import mapping_transformations, output_audit
from vaft.code.gacode.result_identity import digest, plain
from vaft.plot.gyrokinetics import plot_flux_trace, plot_gyrokinetics_field_potential
from vaft.process.gyrokinetics import (
    reduce_time_series,
    zonal_turbulence_intensity,
)
from workflow.gyrokinetic.capture_reference import sha


def build(
    run,
    out,
    *,
    window=None,
    stride=5,
    video=True,
    fps=10.0,
    time_unit="native",
    time_normalization=None,
):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    run, out = Path(run), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    history = read_field_history(run, time_window=window, stride=stride)
    all_fields = read_field_history(run, time_window=window)
    coefficients = all_fields.read()
    intensity = zonal_turbulence_intensity(
        coefficients,
        all_fields.ky,
        weighting="unit-weight native saved-theta potential intensity; special radial slot excluded",
    )
    native = collect_cgyro_outputs(run)
    from vaft.code.gacode.cgyro import CgyroTimeNormalization

    normalization = None
    time_evidence = None
    if time_normalization is not None:
        time_evidence = json.loads(Path(time_normalization).read_text(encoding="utf-8"))
        source = time_evidence["source"]
        # The retained absolute scales apply only to the verified native input.
        input_name = source["native_input_file"]
        if Path(input_name).name != input_name:
            raise ValueError("normalization input file must be a basename")
        if sha(run / input_name) != source["native_input_sha256"]:
            raise ValueError("time normalization belongs to a different native input")
        record = json.loads((run / "record.json").read_text(encoding="utf-8"))
        if record.get("state_identity") != time_evidence["state_identity"]:
            raise ValueError("time normalization belongs to a different plasma state")
        normalization = CgyroTimeNormalization.from_metadata(
            time_evidence["normalization"]
        )
    elif time_unit != "native":
        normalization = native.time_normalization
    time_options = {"time_unit": time_unit, "time_normalization": normalization}
    trace = native.flux_time_trace()
    if trace is None:
        raise ValueError("native flux history unavailable")
    ft = native.time[: trace.shape[-1]]
    lo, hi = float(all_fields.time[0]), float(all_fields.time[-1])
    charges = np.asarray(native.equilibrium["species"]["z"])
    electron = int(np.flatnonzero(charges < 0)[0])
    ion = charges > 0
    qi = trace[ion, 1].sum(axis=0)
    qe = trace[electron, 1]
    statistics = {
        k: reduce_time_series(ft, v, window=(lo, hi))
        for k, v in {"Qi": qi, "Qe": qe, "Qtotal": qi + qe}.items()
    }
    diagnostics = {
        "zonal fraction": {
            "time": all_fields.time,
            "values": intensity.fraction,
            "unit": "1",
        },
        "Qi": {"time": ft, "values": qi, "unit": "Q_GB"},
        "Qe": {"time": ft, "values": qe, "unit": "Q_GB"},
    }
    manifest = {
        p.name: {"sha256": sha(p), "bytes": p.stat().st_size}
        for p in run.iterdir()
        if p.is_file() and not p.name.startswith("bin.cgyro.restart")
    }
    history = replace(
        history,
        metadata={
            **history.metadata,
            "source_identity": digest(manifest),
            "native_file_sha256": manifest.get("bin.cgyro.kxky_phi", {}).get("sha256"),
        },
    )
    provenance = {
        "schema_version": 1,
        "display_time_unit": time_unit,
        "time_domain": "gyrokinetic elapsed time from solver origin",
        "time_normalization_evidence": time_evidence,
        "experimental_trajectory_equivalence": False,
        "source_directory": str(run.resolve()),
        "source_files": manifest,
        "source_identity": digest(manifest),
        "field": dict(history.metadata),
        "window": [lo, hi],
        "weighting": intensity.weighting,
        "statistics": statistics,
        "qualification": {
            "real_native_output": True,
            "upstream_reconstruction_oracle": "b493397; see fixture oracle.json",
            "numerical_convergence": "unverified",
            "saturated_transport": "unverified; selected window only",
        },
        "missing_fields": [
            name
            for name, suffix in [("a_parallel", "apar"), ("b_parallel", "bpar")]
            if not (run / f"bin.cgyro.kxky_{suffix}").exists()
        ],
    }
    # Quantify the difference from the legacy all-slot, unit-weight fraction.
    old = all_fields.raw[..., all_fields.records]
    power = np.mean(abs(old) ** 2, axis=1)
    total = power.sum(axis=(0, 1))
    legacy = np.divide(
        power[:, 0].sum(axis=0), total, out=np.full_like(total, np.nan), where=total > 0
    )
    provenance["legacy_fraction_max_abs_difference"] = float(
        np.nanmax(abs(legacy - intensity.fraction))
    )
    provenance["legacy_fraction_difference_reason"] = (
        "exclude upstream special first radial slot; use one saved theta"
    )
    from vaft.code.gacode.cgyro.locality import radial_correlation_length

    provenance["locality_correlation"] = radial_correlation_length(
        native, window=(lo, hi)
    )
    for component in ("total", "zonal", "nonzonal"):
        fig, _ = plot_gyrokinetics_field_potential(
            history,
            component=component,
            frame_index=len(history.time) // 2,
            diagnostics=diagnostics,
            **time_options,
        )
        fig.savefig(out / f"field_{component}.png", dpi=120)
        plt.close(fig)
    fig, _ = plot_flux_trace(ft, {"Qi": qi, "Qe": qe}, window=(lo, hi), **time_options)
    fig.savefig(out / "flux_window.png", dpi=120)
    plt.close(fig)
    if video:
        movie = plot_gyrokinetics_field_potential(
            history, animation=True, diagnostics=diagnostics, fps=fps, **time_options
        )
        movie.save(out / "field_total.mp4")
        movie.save(out / "field_total.webm")
    (out / "reference.json").write_text(
        json.dumps(plain(provenance), indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    (out / "output_audit.json").write_text(
        json.dumps(
            {"outputs": output_audit(), "transformations": mapping_transformations()},
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return provenance


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--run",
        type=Path,
        default=Path(__file__).resolve().parents[2]
        / "test/data/gacode/cgyro_nonlinear_39915_em",
    )
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--window", nargs=2, type=float)
    p.add_argument("--stride", type=int, default=5)
    p.add_argument("--fps", type=float, default=10.0)
    p.add_argument("--skip-video", action="store_true")
    p.add_argument("--time-unit", choices=("native", "s", "ms"), default="native")
    p.add_argument(
        "--time-normalization",
        type=Path,
        help="Retained normalization evidence, tied to native input and state identity",
    )
    a = p.parse_args()
    build(
        a.run,
        a.out,
        window=a.window,
        stride=a.stride,
        video=not a.skip_video,
        fps=a.fps,
        time_unit=a.time_unit,
        time_normalization=a.time_normalization,
    )
