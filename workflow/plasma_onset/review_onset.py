"""Review figures for the onset primitives (issue #409).

One page per shot: raw and filtered plasma current with the principal-pulse
onset and the pickup scale, the H-alpha record with the threshold, hold and
sustained onset, and the runs each detector rejected.  Reads either a
packaged pipeline sample (``--shot 39915``) or a corpus file written by the
raw-database scan (``--npz path``; keys ``t_ip, ip, t_ha, ha``).

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

from vaft.process.onset import active_window, principal_pulse_onset, sustained_excess_onset

HALPHA = "spectrometer_uv.channel.0.processed_line.0.intensity"


def load_sample(shot: int) -> dict[str, np.ndarray]:
    from omas import load_omas_json

    from vaft.data import resources

    source = resources.data_path(f"samples/{shot}/source/pipeline-until-efit.json.gz")
    with gzip.open(source, "rt") as handle, tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as plain:
        shutil.copyfileobj(handle, plain)
        path = plain.name
    try:
        ods = load_omas_json(path, consistency_check=False)
    finally:
        Path(path).unlink(missing_ok=True)
    return {
        "t_ip": np.asarray(ods["magnetics.ip.0.time"], float),
        "ip": np.asarray(ods["magnetics.ip.0.data"], float),
        "t_ha": np.asarray(ods["spectrometer_uv.time"], float),
        "ha": np.asarray(ods[HALPHA + ".data"], float),
    }


def load_npz(path: str) -> dict[str, np.ndarray]:
    d = np.load(path)
    return {k: np.asarray(d[k], float) for k in ("t_ip", "ip", "t_ha", "ha")}


def review(data: dict[str, np.ndarray], title: str, out: Path, *, window=(0.26, 0.36), reference_end=0.29) -> Path:
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    for ax, key, label, kind in ((axes[0], "ip", "plasma current [kA]", "ip"), (axes[1], "ha", "H-alpha [a.u.]", "ha")):
        t, y = data["t_" + key], data[key]
        m = (t >= window[0]) & (t < window[1])
        t, y = t[m], y[m]
        if t.size < 10:
            ax.set_title(f"{label}: no samples in window")
            continue
        ref = t < reference_end
        scale = 1e3 if kind == "ip" else 1.0
        ax.plot(t * 1e3, y / scale, "0.6", lw=0.5, label="raw")
        if kind == "ip":
            fs = 1.0 / np.median(np.diff(t))
            rec = principal_pulse_onset(t, y, reference_mask=ref, cutoff_hz=2000.0, fs=fs)
            from vaft.process.onset import zero_phase_lowpass

            ax.plot(t * 1e3, zero_phase_lowpass(y, 2000.0, fs) / scale, "tab:blue", lw=1, label="zero-phase LP 2 kHz")
            ax.axhline(rec.evidence["threshold"] / scale, color="0.4", ls=":", label="threshold max(2 % peak, 5σ)")
            if rec.evidence.get("pickup_scale"):
                ax.axhline((rec.evidence["baseline_median"] + rec.evidence["pickup_scale"]) / scale, color="tab:red", ls=":", lw=0.8, label="pickup scale")
        else:
            rec = sustained_excess_onset(t, y, reference_mask=ref, prefilter_samples=5, hold_s=5e-4, min_width_s=1e-3, min_prominence_sigma=10.0, min_integral_fraction=0.01)
            from vaft.process.onset import median_smooth

            ax.plot(t * 1e3, median_smooth(y, 5), "k", lw=0.8, label="median-5")
            ax.axhline(rec.evidence["threshold"], color="0.4", ls=":", label="threshold max(2 % peak, 5σ)")
        for tr, why, feats in rec.rejected[:12]:
            ax.axvspan(feats.start_time * 1e3, feats.end_time * 1e3, color="tab:orange", alpha=0.25)
        if rec.found:
            ax.axvline(rec.time * 1e3, color="tab:green", lw=1.5, label=f"onset {rec.time * 1e3:.2f} ms ({rec.method})")
        else:
            ax.text(0.02, 0.9, "no onset: " + ", ".join(rec.flags), transform=ax.transAxes, color="tab:red", fontsize=9)
        # the active window: principal pulse for Ip, envelope of every segment for H-alpha
        if kind == "ip":
            win = active_window(t, y, principal_only=True, pickup_floor=3.0, cutoff_hz=2000.0, fs=fs,
                                reference_mask=ref, hold_s=5e-4, min_width_s=1e-3, end_fraction=0.10)
        else:
            win = active_window(t, y, prefilter_samples=5, hold_s=5e-4, min_width_s=1e-3,
                                min_prominence_sigma=10.0, min_integral_fraction=0.01, reference_mask=ref)
        if win.found:
            ax.axvspan(win.start * 1e3, win.end * 1e3, color="tab:green", alpha=0.08,
                       label=f"window {win.start * 1e3:.1f}-{win.end * 1e3:.1f} ms" + (f" {list(win.flags)}" if win.flags else ""))
            ax.axvline(win.end * 1e3, color="tab:green", lw=1.0, ls="--")
        ax.set_ylabel(label)
        ax.legend(fontsize=7, loc="upper left")
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
    for shot in args.shot:
        print(review(load_sample(shot), f"#{shot}", args.out))
    for path in args.npz:
        print(review(load_npz(path), Path(path).stem, args.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
