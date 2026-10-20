"""Population views of storage-neutral equilibrium monitoring tables.

Group by backend: a server product and its HSDS observation are not independent
attempts. These report trends, not causal attribution or new quality thresholds.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

COLORS = {"usable": "#2b8c4b", "physical_failed": "#d73027", "numerical_failed": "#f28e2b",
          "output_failed": "#9467bd", "runtime_failed": "#8c564b", "unrecorded": "#999999",
          "pass": "#2b8c4b", "fail": "#d73027", "warn": "#f28e2b", "not_available": "#bdbdbd",
          "indeterminate": "#9467bd", "evaluated": "#377eb8",
          "stage_success": "#377eb8", "stage_no_output": "#9467bd",
          "stage_failed": "#8c564b", "stage_unavailable": "#bdbdbd", "stage_ok": "#377eb8"}

OUTCOMES = ("usable", "physical_failed", "numerical_failed", "output_failed",
            "runtime_failed", "unrecorded", "stage_success", "stage_no_output",
            "stage_failed", "stage_unavailable", "stage_ok")


def write_monitoring_figures(summary, directory):
    """Write outcome trends, time-point maps and check-status distributions."""
    import matplotlib.pyplot as plt
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    written = []
    for backend in summary.shots.backend.dropna().unique():
        shots = summary.shots[summary.shots.backend == backend]
        attempts = summary.attempts[summary.attempts.backend == backend]
        checks = summary.checks[summary.checks.backend == backend]
        fig, axes = plt.subplots(2, 1, figsize=(9, 7), layout="constrained")
        if len(attempts):
            bins = (attempts.shot.astype(int) // 100) * 100
            counts = attempts.groupby([bins.rename("shot_bin"), "overall_status"]).size().unstack(fill_value=0)
            fractions = counts.div(counts.sum(axis=1), axis=0)
            fractions.plot.bar(stacked=True, ax=axes[0], width=1, color=[COLORS.get(c, "#333333") for c in fractions.columns])
            axes[0].set_ylabel("Fraction of observed time points")
            axes[0].set_xlabel("Shot bin (100 shots)")
            axes[0].legend(fontsize="small", loc="upper left", bbox_to_anchor=(1, 1))
            axes[0].set_title(f"{backend}: observed outcomes; unrecorded is unknown")
            positions = {shot: i for i, shot in enumerate(sorted(attempts.shot.unique()))}
            values = [COLORS.get(s, "#333333") for s in attempts.overall_status]
            axes[1].scatter(attempts.time_s, attempts.shot.map(positions), c=values, s=8)
            tick_positions = np.unique(np.linspace(0, len(positions)-1, min(10, len(positions))).astype(int))
            axes[1].set_yticks(tick_positions, [str(list(positions)[i]) for i in tick_positions])
            axes[1].set_xlabel("Attempt time [s]")
            axes[1].set_ylabel("Shot (ordered rows)")
            from matplotlib.lines import Line2D
            axes[1].legend(handles=[Line2D([], [], marker='o', linestyle='', color=COLORS[name], label=name)
                                    for i, name in enumerate(OUTCOMES) if name in set(attempts.overall_status)],
                           fontsize="small", loc="upper left", bbox_to_anchor=(1, 1))
        else:
            for ax in axes:
                ax.text(.5, .5, "No observed time points", ha="center", transform=ax.transAxes)
        n_missing = int((shots.equilibria_present.fillna(0) == 0).sum())
        fig.suptitle(f"{backend}: {len(shots)} requested shots; {n_missing} without equilibrium")
        path = directory / f"outcomes_{backend}.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        written.append(path)
        if len(checks):
            counts = checks.groupby(["check", "status"]).size().unstack(fill_value=0)
            fig, ax = plt.subplots(figsize=(9, max(3, len(counts)*.28)), layout="constrained")
            counts.div(counts.sum(axis=1), axis=0).plot.barh(stacked=True, ax=ax,
                color=[COLORS.get(c, "#333333") for c in counts.columns])
            ax.set_xlabel("Fraction of evaluated records (missing evidence retained)")
            ax.set_title(f"{backend}: verification and validation status")
            ax.legend(fontsize="small", loc="upper left", bbox_to_anchor=(1, 1))
            path = directory / f"validation_{backend}.png"
            fig.savefig(path, dpi=150)
            plt.close(fig)
            written.append(path)
        # Channel residual matrix: channel labels include family to avoid mixing
        # identically named sensors from different constraint families.
        channels = summary.channels[(summary.channels.backend == backend) &
                                    (summary.channels.scope == "equilibrium_fit")].copy()
        if len(channels):
            channels["z"] = channels.details_json.map(lambda text: json.loads(text).get("z"))
            channels["label"] = channels.family_name.astype(str) + ":" + channels.channel.astype(str)
            matrix = channels.pivot_table(index="label", columns="shot", values="z", aggfunc="median")
            if matrix.size:
                fig, ax = plt.subplots(figsize=(10, max(3, min(14, len(matrix)*.15))), layout="constrained")
                limit = np.nanquantile(np.abs(matrix.to_numpy(float)), .95)
                if not np.isfinite(limit) or limit == 0:
                    limit = 1
                mesh = ax.imshow(matrix.to_numpy(float), aspect="auto", cmap="RdBu_r", vmin=-limit, vmax=limit)
                ticks = np.unique(np.linspace(0, len(matrix.columns)-1, min(10, len(matrix.columns))).astype(int))
                ax.set_xticks(ticks, [str(matrix.columns[i]) for i in ticks])
                ax.set_yticks(range(len(matrix)), matrix.index, fontsize="x-small")
                ax.set_xlabel("Shot (ordered columns)")
                ax.set_title(f"{backend}: median fitted residual by channel and shot")
                fig.colorbar(mesh, ax=ax, label="Signed residual / uncertainty (z)")
                path = directory / f"channel_residuals_{backend}.png"
                fig.savefig(path, dpi=150)
                plt.close(fig)
                written.append(path)
    return written
