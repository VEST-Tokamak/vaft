"""Population renderers for the equilibrium-quality cohort tables (#1644).

They draw the tables :mod:`vaft.validation.equilibrium_quality` builds -- the
per-slice cohort table, its channel-level constraint points and its census --
and compute nothing scientific of their own: residuals, ``z`` and the verdicts
come from those tables.  Like :mod:`vaft.plot.population` they take a table,
not a view model, follow the renderer contract (``ax=None``, ``show=False``,
return ``(Figure, Axes)``), and give each cohort a fixed colour *and* marker so
identity never rests on colour alone.

The measured-versus-reconstructed view keeps ``y = x`` as the reference and
reports deviation from it (bias and RMS of ``z``), not a fitted regression.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = [
    "COHORT_STYLE",
    "equilibrium_quality_measured_vs_reconstructed",
    "equilibrium_quality_reduced_chi2",
    "equilibrium_quality_residual_distribution",
    "equilibrium_quality_selection_funnel",
    "equilibrium_quality_validation_matrix",
]

#: Cohort -> (colour, marker, label), in report order (validated palette of
#: vaft.plot.population; the failed cohort is the neutral grey).
COHORT_STYLE = {
    "good": ("#2a78d6", "o", "good"),
    "admissible": ("#eb6834", "s", "admissible-only"),
    "unreconstructible": ("#8b8a82", "x", "unreconstructible attempt"),
}

FAMILY_TITLES = {
    "bpol_probe": "Poloidal probes",
    "flux_loop": "Flux loops",
    "pf_current": "PF currents",
    "ip": "Plasma current",
    "diamagnetic_flux": "Diamagnetic flux",
}

RULE_TITLES = {
    "rule_convergence": "convergence",
    "rule_pressure_nonnegative": "pressure ≥ 0",
    "rule_beta_p_positive": "β_p > 0",
    "rule_w_positive": "W > 0",
    "rule_q95": "q95 > floor",
    "rule_ip_ratio": "Ip ratio",
    "rule_probe_fit": "probe fit",
    "rule_loop_fit": "flux-loop fit",
    "rule_ip_fit": "Ip fit",
    "rule_dia_fit": "diamagnetic fit",
    "virial_status": "virial",
    "grad_shafranov_status": "Grad–Shafranov",
    "thomson_status": "Thomson (separate)",
}


def _axes(ax, figsize):
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    return fig, ax


def _finish(show: bool):
    if show:
        import matplotlib.pyplot as plt

        plt.show()


def _cohorts(table: pd.DataFrame, by: str):
    for cohort, (color, marker, label) in COHORT_STYLE.items():
        mask = (table[by] == cohort).to_numpy() if by in table else np.zeros(len(table), bool)
        if mask.any():
            yield cohort, mask, color, marker, label


def equilibrium_quality_measured_vs_reconstructed(points: pd.DataFrame, *, family: str, by: str = "quality_label",
                                                  fitted_only: bool = True, ax=None, show: bool = False,
                                                  figsize=(5.2, 5.0)):
    """Measured (x) against reconstructed (y) for one constraint family, by cohort, with ``y = x``.

    ``points`` are :func:`~vaft.validation.equilibrium_quality.equilibrium_quality_constraint_points`
    rows joined with each slice's ``quality_label``.  The legend gives, per
    cohort, the number of channels and the bias and RMS of ``z`` -- deviation
    from the identity line, not a regression.
    """
    fig, ax = _axes(ax, figsize)
    data = points[points["family"] == family]
    if fitted_only and "fitted" in data:
        data = data[data["fitted"].astype(bool)]
    x = pd.to_numeric(data["measured"], errors="coerce").to_numpy(float)
    y = pd.to_numeric(data["reconstructed"], errors="coerce").to_numpy(float)
    z = pd.to_numeric(data.get("z", pd.Series(np.nan, index=data.index)), errors="coerce").to_numpy(float)
    finite = np.isfinite(x) & np.isfinite(y)
    for _cohort, mask, color, marker, label in _cohorts(data, by):
        use = mask & finite
        if not use.any():
            continue
        zc = z[use][np.isfinite(z[use])]
        stats = f", z bias {zc.mean():+.2f}, z rms {np.sqrt(np.mean(zc ** 2)):.2f}" if zc.size else ""
        # Failed attempts underneath and faint, good on top: the largest cohort
        # must not hide the one the figure is about.
        layer = {"good": 3, "admissible": 2}.get(_cohort, 1)
        ax.scatter(x[use], y[use], s=14 if layer > 1 else 8, marker=marker, color=color,
                   alpha=0.8 if layer == 3 else 0.55 if layer == 2 else 0.2, zorder=layer,
                   linewidths=1.0 if marker == "x" else 0.4, edgecolors=None if marker == "x" else "white",
                   label=f"{label} (n={int(use.sum())}{stats})")
    if finite.any():
        lo = float(np.nanmin(np.r_[x[finite], y[finite]]))
        hi = float(np.nanmax(np.r_[x[finite], y[finite]]))
        pad = 0.05 * (hi - lo or 1.0)
        ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color="#1a1a19", lw=1.0, ls="--", label="y = x")
        ax.set_xlim(lo - pad, hi + pad)
        ax.set_ylim(lo - pad, hi + pad)
    unit = data["unit"].iloc[0] if "unit" in data and len(data) else ""
    ax.set_xlabel(f"measured [{unit}]" if unit else "measured")
    ax.set_ylabel(f"reconstructed [{unit}]" if unit else "reconstructed")
    ax.set_title(FAMILY_TITLES.get(family, family))
    ax.set_aspect("equal", adjustable="box")
    ax.legend(fontsize=7, loc="upper left", frameon=False)
    _finish(show)
    return fig, ax


def equilibrium_quality_residual_distribution(points: pd.DataFrame, *, family: str, by: str = "quality_label",
                                              ax=None, show: bool = False, figsize=(5.2, 3.6)):
    """Empirical CDF of the normalised residual ``|z|`` of one family, by cohort.

    The dotted lines at |z| = 2 and 3 are the outlier levels the fit-quality
    metrics use; ``z`` is the table's, never recomputed.
    """
    fig, ax = _axes(ax, figsize)
    data = points[(points["family"] == family) & points["fitted"].astype(bool)] if "fitted" in points else \
        points[points["family"] == family]
    z = np.abs(pd.to_numeric(data["z"], errors="coerce").to_numpy(float))
    for _cohort, mask, color, _marker, label in _cohorts(data, by):
        values = np.sort(z[mask & np.isfinite(z)])
        if values.size:
            ax.step(values, np.arange(1, values.size + 1) / values.size, where="post", color=color,
                    lw=1.6, label=f"{label} (n={values.size})")
    for level in (2.0, 3.0):
        ax.axvline(level, color="#8b8a82", lw=0.8, ls=":")
    ax.set_xscale("symlog", linthresh=1.0)
    ax.set_xlim(left=0.0)
    ax.set_xlabel("|z| (normalised residual)")
    ax.set_ylabel("fraction of channels ≤ |z|")
    ax.set_ylim(0, 1.02)
    ax.set_title(FAMILY_TITLES.get(family, family))
    ax.legend(fontsize=7, frameon=False, loc="lower right")
    _finish(show)
    return fig, ax


def equilibrium_quality_reduced_chi2(table: pd.DataFrame, *, column: str = "probe_reduced_chi2",
                                     by: str = "quality_label", band=(0.5, 2.0), ax=None, show: bool = False,
                                     figsize=(5.2, 3.6)):
    """Empirical CDF of a family's reduced chi-square over slices, by cohort, with the study's band shaded."""
    fig, ax = _axes(ax, figsize)
    values = pd.to_numeric(table[column], errors="coerce").to_numpy(float)
    for _cohort, mask, color, _marker, label in _cohorts(table, by):
        v = np.sort(values[mask & np.isfinite(values) & (values > 0)])
        if v.size:
            ax.step(v, np.arange(1, v.size + 1) / v.size, where="post", color=color, lw=1.6,
                    label=f"{label} (n={v.size})")
    if band is not None:
        ax.axvspan(band[0], band[1], color="#e9e8e2", zorder=0, label=f"study band {band[0]}–{band[1]}")
    ax.set_xscale("log")
    ax.set_xlabel(column.replace("_", " "))
    ax.set_ylabel("fraction of slices ≤ value")
    ax.set_ylim(0, 1.02)
    ax.legend(fontsize=7, frameon=False, loc="lower right")
    _finish(show)
    return fig, ax


def equilibrium_quality_validation_matrix(census: dict, *, ax=None, show: bool = False, figsize=(6.8, 6.4)):
    """The rule × cohort matrix: each cell's fail fraction as colour, pass / fail / n as text.

    ``census`` is :func:`~vaft.validation.equilibrium_quality.equilibrium_quality_failure_census`.
    Cells where nothing could be graded are hatched; Thomson is drawn as its own
    row, below a rule line, because it never decides ``good``.
    """
    from matplotlib.patches import Rectangle

    fig, ax = _axes(ax, figsize)
    matrix = census["matrix"]
    rules = list(matrix)
    cohorts = [c for c in COHORT_STYLE]
    fail = np.full((len(rules), len(cohorts)), np.nan)
    for i, rule in enumerate(rules):
        for j, cohort in enumerate(cohorts):
            cell = matrix[rule].get(cohort) or {}
            graded = (cell.get("pass") or 0) + (cell.get("fail") or 0)
            if cell.get("n") and graded == graded and graded > 0:
                fail[i, j] = (cell.get("fail") or 0) / graded
    image = ax.imshow(np.ma.masked_invalid(fail), cmap="OrRd", vmin=0, vmax=1, aspect="auto")
    for i, rule in enumerate(rules):
        for j, cohort in enumerate(cohorts):
            cell = matrix[rule].get(cohort) or {}
            n = cell.get("n") or 0
            if not n:
                continue
            p, f = cell.get("pass") or 0.0, cell.get("fail") or 0.0
            na = 1.0 - p - f
            text = f"{p:.0%} / {f:.0%}" + (f"\n(n/a {na:.0%})" if na > 0.005 else "")
            dark = np.isfinite(fail[i, j]) and fail[i, j] > 0.6
            ax.text(j, i, text, ha="center", va="center", fontsize=6.5, color="white" if dark else "#1a1a19")
            if not np.isfinite(fail[i, j]):
                ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, hatch="///",
                                       edgecolor="#b8b7ae", lw=0))
    if "thomson_status" in rules:
        ax.axhline(rules.index("thomson_status") - 0.5, color="#1a1a19", lw=1.2)
    ax.set_xticks(range(len(cohorts)))
    ax.set_xticklabels([f"{COHORT_STYLE[c][2]}\n(n={(matrix[rules[0]].get(c) or {}).get('n', 0)})" for c in cohorts],
                       fontsize=8)
    ax.set_yticks(range(len(rules)))
    ax.set_yticklabels([RULE_TITLES.get(r, r) for r in rules], fontsize=8)
    ax.set_title("Rule verdicts by cohort (pass / fail; colour = fail share of graded)", fontsize=9)
    fig.colorbar(image, ax=ax, fraction=0.04, pad=0.02, label="fail fraction")
    _finish(show)
    return fig, ax


def equilibrium_quality_selection_funnel(funnel: dict, *, ax=None, show: bool = False, figsize=(6.4, 4.2)):
    """Per EFIT-quality cohort: confinement candidates, then what each downstream rule removes, then selected.

    ``funnel`` is :func:`~vaft.validation.equilibrium_quality.equilibrium_quality_confinement_funnel`.
    One horizontal bar per cohort, split into the slices each rule removes in
    sequence and the slices selected -- so a ``good`` equilibrium rejected by a
    stationarity rule reads as exactly that.
    """
    fig, ax = _axes(ax, figsize)
    cohorts = list(funnel["cohorts"])
    rules = funnel["rules"]
    shades = ("#5c5b55", "#8b8a82", "#b8b7ae", "#d8d7cf", "#e9e8e2")
    for i, cohort in enumerate(cohorts):
        entry = funnel["cohorts"][cohort]
        removed = {row["rule"]: row["removed_in_sequence"] for row in entry["exclusions"]}
        left = 0
        for j, rule in enumerate(rules):
            width = removed.get(rule, 0)
            if width:
                ax.barh(i, width, left=left, color=shades[j % len(shades)], edgecolor="white")
                ax.text(left + width / 2, i, str(width), ha="center", va="center", fontsize=7, color="white" if j < 2 else "#1a1a19")
            left += width
        color = COHORT_STYLE.get(cohort, ("#2a78d6", "o", cohort))[0]
        ax.barh(i, entry["selected"], left=left, color=color, edgecolor="white")
        ax.text(left + entry["selected"] / 2 if entry["selected"] else left, i, f"{entry['selected']} selected",
                ha="center", va="center", fontsize=7, color="white" if entry["selected"] else "#1a1a19")
    from matplotlib.patches import Patch

    handles = [Patch(color=shades[j % len(shades)], label=f"removed by {rule}") for j, rule in enumerate(rules)]
    # Below the axes: inside, it would cover the widest cohort's bar.
    ax.legend(handles=handles, fontsize=7, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)
    ax.set_yticks(range(len(cohorts)))
    ax.set_yticklabels([f"{COHORT_STYLE.get(c, (None, None, c))[2]}\n(n={funnel['cohorts'][c]['candidates']})" for c in cohorts],
                       fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("confinement candidate slices")
    ax.set_title("EFIT quality → confinement selection", fontsize=9)
    _finish(show)
    return fig, ax
