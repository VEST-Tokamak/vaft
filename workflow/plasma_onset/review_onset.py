"""Review figures for the onset primitives (issue #409).

One page per shot: raw and filtered plasma current with the principal-pulse
onset and the pickup scale, the H-alpha record with the threshold, hold and
sustained onset, and the runs each detector rejected.  Reads either a
packaged pipeline sample (``--shot 39915``) or a corpus file written by the
raw-database scan (``--npz path``; keys ``t_ip, ip, t_ha, ha``).  The rules
and the shared plasma-analysis range come from the ``plasma_timing`` policy
in ``vest.yaml``; a packaged shot also gets the composed
``vaft.omas.plasma_timing`` verdict -- source, agreement, fallback reason --
as a third panel.

    python workflow/plasma_onset/review_onset.py --shot 39915 --out figures/
"""
from __future__ import annotations

import argparse
import gzip
import shutil
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from vaft.machine_mapping.utils import PlasmaTimingPolicy, resolve_plasma_timing_policy
from vaft.process.onset import (
    active_window,
    median_smooth,
    principal_pulse_onset,
    sustained_excess_onset,
    zero_phase_lowpass,
)


def load_ods(shot: int):
    from omas import load_omas_json

    from vaft.data import resources

    source = resources.data_path(f"samples/{shot}/source/pipeline-until-efit.json.gz")
    with gzip.open(source, "rt") as handle, tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as plain:
        shutil.copyfileobj(handle, plain)
        path = plain.name
    try:
        return load_omas_json(path, consistency_check=False)
    finally:
        Path(path).unlink(missing_ok=True)


def load_sample(ods) -> dict[str, np.ndarray]:
    """The plasma current and the primary H-alpha line, found by label."""
    from vaft.omas.plasma_timing import halpha_sources

    primary = halpha_sources()[0]
    return {
        "t_ip": np.asarray(ods["magnetics.ip.0.time"], float),
        "ip": np.asarray(ods["magnetics.ip.0.data"], float),
        "t_ha": np.asarray(ods["spectrometer_uv.time"], float),
        "ha": np.asarray(ods[primary.base + ".data"], float),
    }


def load_npz(path: str) -> dict[str, np.ndarray]:
    d = np.load(path)
    return {k: np.asarray(d[k], float) for k in ("t_ip", "ip", "t_ha", "ha")}


def _subset(rule: dict, func) -> dict:
    """The part of a policy rule a single-onset detector accepts."""
    import inspect

    accepted = inspect.signature(func).parameters
    return {k: v for k, v in rule.items() if k in accepted}


def review(data: dict[str, np.ndarray], title: str, out: Path, *, policy: PlasmaTimingPolicy | None = None, timing=None) -> Path:
    policy = policy or resolve_plasma_timing_policy()
    window = (policy.baseline_start, policy.window.tend)
    baseline_end = policy.window.tstart
    panels = 3 if timing is not None else 2
    fig, axes = plt.subplots(panels, 1, figsize=(11, 3.5 * panels), sharex=True)
    for ax, key, label, kind in ((axes[0], "ip", "plasma current [kA]", "ip"), (axes[1], "ha", "H-alpha [a.u.]", "ha")):
        t, y = data["t_" + key], data[key]
        m = (t >= window[0]) & (t < window[1])
        t, y = t[m], y[m]
        if t.size < 10:
            ax.set_title(f"{label}: no samples in window")
            continue
        ref = t < baseline_end  # the baseline stretch, the detectors' reference_mask
        rule = policy.ip if kind == "ip" else policy.h_alpha
        threshold_label = f"threshold max({100 * rule.get('fraction', 0.02):g} % peak, {rule.get('sigma', 5.0):g}σ)"
        scale = 1e3 if kind == "ip" else 1.0
        ax.plot(t * 1e3, y / scale, "0.6", lw=0.5, label="raw")
        if kind == "ip":
            fs = 1.0 / np.median(np.diff(t))
            rec = principal_pulse_onset(t, y, reference_mask=ref, search_mask=~ref, **_subset(rule, principal_pulse_onset))
            cutoff = rule.get("cutoff_hz")
            if cutoff:
                ax.plot(t * 1e3, zero_phase_lowpass(y, cutoff, fs) / scale, "tab:blue", lw=1, label=f"zero-phase LP {cutoff / 1e3:g} kHz")
            ax.axhline(rec.evidence["threshold"] / scale, color="0.4", ls=":", label=threshold_label)
            if rec.evidence.get("pickup_scale"):
                ax.axhline((rec.evidence["baseline_median"] + rec.evidence["pickup_scale"]) / scale, color="tab:red", ls=":", lw=0.8, label="pickup scale")
        else:
            rec = sustained_excess_onset(t, y, reference_mask=ref, search_mask=~ref, **_subset(rule, sustained_excess_onset))
            kernel = int(rule.get("prefilter_samples", 1))
            ax.plot(t * 1e3, median_smooth(y, kernel) if kernel > 1 else y, "k", lw=0.8, label=f"median-{kernel}")
            ax.axhline(rec.evidence["threshold"], color="0.4", ls=":", label=threshold_label)
        for tr, why, feats in rec.rejected[:12]:
            ax.axvspan(feats.start_time * 1e3, feats.end_time * 1e3, color="tab:orange", alpha=0.25)
        if rec.found:
            ax.axvline(rec.time * 1e3, color="tab:green", lw=1.5, label=f"onset {rec.time * 1e3:.2f} ms ({rec.method})")
        else:
            ax.text(0.02, 0.9, "no onset: " + ", ".join(rec.flags), transform=ax.transAxes, color="tab:red", fontsize=9)
        # the active window with the same rule: principal pulse for Ip, envelope for H-alpha
        win = active_window(t, y, reference_mask=ref, search_mask=~ref, **rule)
        if win.found:
            ax.axvspan(win.start * 1e3, win.end * 1e3, color="tab:green", alpha=0.08,
                       label=f"window {win.start * 1e3:.1f}-{win.end * 1e3:.1f} ms" + (f" {list(win.flags)}" if win.flags else ""))
            ax.axvline(win.end * 1e3, color="tab:green", lw=1.0, ls="--")
        ax.set_ylabel(label)
        ax.legend(fontsize=7, loc="upper left")
    if timing is not None:
        ax = axes[2]
        ax.axis("off")
        if isinstance(timing, str):
            ax.text(0.0, 1.0, timing, transform=ax.transAxes, va="top", fontsize=8, color="tab:red", family="monospace")
            axes[-1].set_xlabel("time [ms]")
            fig.tight_layout()
            out.mkdir(parents=True, exist_ok=True)
            path = out / f"onset_review_{title.replace(' ', '_').replace('#', '')}.png"
            fig.savefig(path, dpi=110)
            plt.close(fig)
            return path
        rec = timing.record()
        lines = [
            f"plasma_timing: onset {rec['onset']} s, offset {rec['offset']} s from {rec['source']}  [{rec['agreement']}]",
            f"optical: {None if rec['optical'] is None else (rec['optical']['start'], rec['optical']['end'])}  "
            f"ip: {None if rec['ip'] is None else (rec['ip']['start'], rec['ip']['end'])}  delta onset/offset [ms]: "
            f"{None if rec['onset_delta_s'] is None else round(rec['onset_delta_s'] * 1e3, 2)} / "
            f"{None if rec['offset_delta_s'] is None else round(rec['offset_delta_s'] * 1e3, 2)}",
            f"flags: {rec['flags']}   fallback: {rec['fallback_reason']}",
            f"span: {rec['span']}",
        ]
        for c in rec["candidates"]:
            m = c["metrics"]
            lines.append(
                f"{c['source']['role']} ch{c['source']['channel']}/{c['source']['line']}: usable={c['usable']} reason={c['reason']} "
                f"MAD={m.get('baseline_mad', float('nan')):.2e} peak/sigma={m.get('peak_over_sigma', float('nan')):.0f} "
                f"railed={m.get('railed_fraction', float('nan')):.4f} resampled={c['notes'].get('resampled')}"
            )
        ax.text(0.0, 1.0, "\n".join(lines), transform=ax.transAxes, va="top", fontsize=7.5, family="monospace")
    axes[-1].set_xlabel("time [ms]")
    fig.suptitle(f"{title}: rejected runs shaded orange, onset and offset in green, active window tinted", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"onset_review_{title.replace(' ', '_').replace('#', '')}.png"
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shot", type=int, action="append", default=[], help="packaged sample shot(s)")
    parser.add_argument("--npz", action="append", default=[], help="corpus file(s) from the raw-database scan")
    parser.add_argument("--out", type=Path, default=Path("onset_review"))
    args = parser.parse_args(argv)
    policy = resolve_plasma_timing_policy()
    for shot in args.shot:
        from vaft.omas.plasma_timing import PlasmaTimingError, plasma_timing

        ods = load_ods(shot)
        try:
            timing = plasma_timing(ods, policy=policy)
        except PlasmaTimingError as exc:
            timing = f"plasma_timing refused: {exc}"
        print(review(load_sample(ods), f"#{shot}", args.out, policy=policy, timing=timing))
    for path in args.npz:
        print(review(load_npz(path), Path(path).stem, args.out, policy=policy))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
