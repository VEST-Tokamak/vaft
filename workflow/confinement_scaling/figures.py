"""Lane D figures for the conference atlas (#548; handed to Lane V, #1456).

``vaft.plot`` is frozen, so these are workflow functions. Each returns
``(fig, axes)``, so ``notebooks/conference_operational_space_atlas.ipynb`` can
call them directly. They draw only from Lane D's files and Lane M's public data:

- ``population_figure``: VEST over the ITPA DB5.2.3 standard set (SELDB5),
  with the spherical tokamaks (NSTX, MAST, START) apart from the conventional
  ones. It reuses ``vaft.plot.population.confinement_population``.
- ``predicted_vs_measured_figure`` and ``h_factor_figure``: tau_E against
  IPB98(y,2) and NSTX2006L, from Lane M's ``predict_confinement_time`` and plot
  functions. VEST is ohmic L-mode, so its IPB98 H factor is a location, not a
  performance claim.
- ``exponent_figure``: fitted engineering exponents (VEST free and closure fits)
  against NSTX and IPB98(y,2), plus the Kadomtsev-completed mu_rho, drawn as
  *assumed* and marked undetermined where (1 + aP)/sigma < 2.

VEST rows are the primary selection of ``fit.py`` (decided on #1490), with
W = W_mhd and P = P_net (radiation not subtracted, as DB5 PLTH).

Usage::

    python figures.py --atlas ~/runs/campaign/atlas/confinement \\
        --out ~/runs/campaign/atlas/confinement/figures
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fit import SELECTIONS, _git, select  # noqa: E402

SPHERICAL = ("NSTX", "MAST", "START")
LABELS = {"H98y2": "IPB98(y,2)", "NSTX2006L": "NSTX 2006 L-mode", "NSTX2006H": "NSTX 2006 H-mode",
          "ITER89P": "ITER89-P"}
VEST_LABEL = "VEST (ohmic, this work)"
#: Distinct open markers for the reference scalings, so each can be told apart.
REFERENCE_MARKERS = ("s", "D", "^", "v", "P")
#: IPB98(y,2) engineering exponents at fixed epsilon (ITER Physics Basis 1999).
IPB98 = {"i_p": 0.93, "b_t": 0.15, "p_net": -0.69, "n_e": 0.41}
UNDETERMINED_SIGMA = 2.0


def population_table(db5: pd.DataFrame, confinement: pd.DataFrame, selection: str = "primary") -> pd.DataFrame:
    """DB5 standard set plus the selected VEST rows, with a ``population`` label column."""
    from vaft.data.public.schema import CONFINEMENT_COLUMNS

    vest = confinement.loc[select(confinement, SELECTIONS[selection])]
    vest = vest.loc[np.isfinite(vest["tau_e_th_s"]) & (vest["tau_e_th_s"] > 0), list(CONFINEMENT_COLUMNS)]
    std = db5.loc[db5["selected"].astype(bool), list(CONFINEMENT_COLUMNS)]
    table = pd.concat([std, vest], ignore_index=True)
    machine = table["machine"].astype(str)
    table["population"] = np.where(
        machine == "VEST", "VEST (ohmic, this work)",
        np.where(machine.isin(SPHERICAL), machine + " (DB5)", "conventional tokamaks (DB5)"))
    return table


def population_figure(table: pd.DataFrame, *, figsize=(11.0, 4.6)):
    """tau_E against I_p and against P_loss: VEST among the DB5 population."""
    import matplotlib.pyplot as plt

    from vaft.plot.population import confinement_population

    fig, axes = plt.subplots(1, 2, figsize=figsize, constrained_layout=True)
    for ax, x in zip(axes, ("i_p_A", "p_loss_W")):
        confinement_population(table, x=x, y="tau_e_th_s", by="population",
                               highlight=VEST_LABEL, max_groups=4, ax=ax)
    axes[0].set_title("Thermal confinement time against plasma current")
    axes[1].set_title("... against loss power (VEST: P_OH - dW/dt)")
    return fig, axes


def predicted_vs_measured_figure(table: pd.DataFrame, scalings=("H98y2", "NSTX2006L"), *, figsize=(11.0, 5.2)):
    """Measured against scaling-predicted tau_E, one panel per scaling."""
    import matplotlib.pyplot as plt

    from vaft.data.public import predict_confinement_time
    from vaft.plot.population import confinement_predicted_vs_measured

    fig, axes = plt.subplots(1, len(scalings), figsize=figsize, constrained_layout=True)
    for ax, scaling in zip(np.atleast_1d(axes), scalings):
        predicted = predict_confinement_time(table, scaling)
        confinement_predicted_vs_measured(table, predicted, scaling_label=LABELS.get(scaling, scaling),
                                          by="population", highlight=VEST_LABEL, max_groups=4, ax=ax)
        ax.set_title(f"{LABELS.get(scaling, scaling)}: {_vest_coverage(table, predicted)}", fontsize=9)
    return fig, axes


def _vest_coverage(table, predicted) -> str:
    """'VEST n/N rows': a scaling with a density term drops rows without Thomson n_e."""
    vest = (table["machine"] == "VEST").to_numpy()
    shown = int(np.sum(vest & np.isfinite(predicted.to_numpy(float))))
    note = " (rows with Thomson n_e)" if shown < vest.sum() else ""
    return f"VEST {shown}/{int(vest.sum())} rows{note}"


def h_factor_figure(table: pd.DataFrame, scalings=("H98y2", "NSTX2006L"), *, figsize=(11.0, 4.6)):
    """H-factor distributions per population, one panel per scaling."""
    import matplotlib.pyplot as plt

    from vaft.data.public import h_factor
    from vaft.plot.population import confinement_h_factor_distribution

    fig, axes = plt.subplots(1, len(scalings), figsize=figsize, constrained_layout=True)
    for ax, scaling in zip(np.atleast_1d(axes), scalings):
        h = h_factor(table, scaling)
        confinement_h_factor_distribution(table, h, scaling_label=LABELS.get(scaling, scaling),
                                          by="population", highlight=VEST_LABEL, ax=ax)
        ax.set_title(f"{LABELS.get(scaling, scaling)}: {_vest_coverage(table, h)}", fontsize=9)
    return fig, axes


def exponent_table(closures: pd.DataFrame, nstx: pd.DataFrame, data: str = "primary:A") -> pd.DataFrame:
    """One row per scaling: aI, aB, aP (with errors where fitted) and the completed mu_rho."""
    rows = []
    sub = closures.loc[closures["data"] == data]
    for _, r in sub.iterrows():
        undetermined = (r["model"] == "free"
                        and abs(r.get("one_plus_aP_over_se", np.inf)) < UNDETERMINED_SIGMA)
        rows.append({
            "label": f"VEST {r['model']}" + ("" if r["model"] == "free" else f" (mu_rho = {r['mu_rho_imposed']:g})"),
            "kind": "vest_free" if r["model"] == "free" else "vest_closure",
            **{f"a_{k}": r.get(f"a_{k}", np.nan) for k in ("i_p", "b_t", "p_net")},
            **{f"se_{k}": r.get(f"se_{k}", np.nan) for k in ("i_p", "b_t", "p_net")},
            "mu_rho": r["mu_rho_imposed"] if r["model"] != "free" else r.get("mu_rho_completed", np.nan),
            "mu_rho_se": 0.0 if r["model"] != "free" else r.get("mu_rho_completed_se", np.nan),
            "mu_rho_undetermined": bool(undetermined),
        })
    for _, r in nstx.loc[~nstx["scaling"].astype(str).str.startswith("VEST")].iterrows():
        rows.append({"label": r["scaling"], "kind": "reference",
                     **{f"a_{k}": r[f"a_{k}"] for k in ("i_p", "b_t", "p_net")},
                     "mu_rho": r["mu_rho_completed"], "mu_rho_se": 0.0, "mu_rho_undetermined": False})
    rows.append({"label": "IPB98(y,2)", "kind": "reference",
                 **{f"a_{k}": IPB98[k] for k in ("i_p", "b_t", "p_net")}, "mu_rho": -2.70, "mu_rho_se": 0.0,
                 "mu_rho_undetermined": False})
    return pd.DataFrame(rows)


def exponent_figure(exponents: pd.DataFrame, *, figsize=(12.5, 5.2)):
    """Engineering exponents and the completed mu_rho, VEST fits against references."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=figsize, constrained_layout=True,
                             gridspec_kw={"width_ratios": [2.0, 1.25]})
    ax = axes[0]
    terms = [("i_p", r"$\alpha_I$"), ("b_t", r"$\alpha_B$"), ("p_net", r"$\alpha_P$")]
    n = len(exponents)
    offsets = np.linspace(-0.32, 0.32, n) if n > 1 else np.zeros(1)
    closure_colors = iter(("C1", "C2", "C4", "C5"))
    reference_markers = iter(REFERENCE_MARKERS)
    styles = {}
    for i, (_, r) in enumerate(exponents.iterrows()):
        if r["kind"] == "vest_free":
            st = dict(marker="*", ms=14, color="k", zorder=5)
        elif r["kind"] == "vest_closure":
            st = dict(marker="o", ms=6, color=next(closure_colors))
        else:
            st = dict(marker=next(reference_markers), ms=7, color="0.4", mfc="none", mew=1.3)
        styles[r["label"]] = st
        xs = np.arange(len(terms)) + offsets[i]
        ys = [r[f"a_{k}"] for k, _ in terms]
        es = [r.get(f"se_{k}", np.nan) for k, _ in terms]
        es = [e if np.isfinite(e) else 0.0 for e in es]
        ax.errorbar(xs, ys, yerr=es, ls="none", label=r["label"], capsize=2, **st)
    ax.axhline(0.0, color="0.7", lw=0.8)
    ax.set_xticks(range(len(terms)), [t for _, t in terms], fontsize=12)
    ax.set_ylabel("engineering exponent (VEST: ±1σ, clustered by shot)")
    ax.set_title(r"$\tau_E \propto I_p^{\alpha_I} B_T^{\alpha_B} P^{\alpha_P}$"
                 "  (VEST: density-free, primary selection)", fontsize=10)
    ax.legend(fontsize=7, loc="upper right", ncol=2)

    ax = axes[1]
    ys = np.arange(len(exponents))[::-1]
    for y, (_, r) in zip(ys, exponents.iterrows()):
        st = {k: v for k, v in styles[r["label"]].items() if k != "zorder"}
        if r["mu_rho_undetermined"] or not np.isfinite(float(r["mu_rho"])):
            ax.text(-4.0, y, r"undetermined: $(1+\alpha_P)/\sigma < $" + f"{UNDETERMINED_SIGMA:g}",
                    ha="center", va="center", fontsize=8, color="k",
                    bbox=dict(boxstyle="round", fc="white", ec="0.6"))
            continue
        se = float(r["mu_rho_se"]) if np.isfinite(float(r["mu_rho_se"])) else 0.0
        ax.errorbar(float(r["mu_rho"]), y, xerr=se or None, ls="none", capsize=2, **st)
    ax.axvspan(-3.0, -2.0, color="0.92", zorder=0)
    for x, name in ((-2.0, "Bohm"), (-3.0, "gyro-Bohm")):
        ax.axvline(x, color="0.6", lw=0.8, ls=":")
        ax.text(x, 1.01, name, transform=ax.get_xaxis_transform(), ha="center", va="bottom", fontsize=8)
    ax.set_yticks(ys, exponents["label"], fontsize=7)
    ax.set_xlim(-7.0, 0.0)
    ax.set_ylim(-0.7, len(exponents) - 0.3)
    ax.set_xlabel(r"$\mu_\rho$ in $\Omega_i\tau_E \propto \rho_*^{\mu_\rho}\beta^{\mu_\beta}\nu_*^{\mu_\nu}$"
                  "\nKadomtsev-completed: the size exponent is ASSUMED", fontsize=9)
    ax.set_title("closures: imposed; references: completed", fontsize=9, pad=16)
    return fig, axes


def main(argv=None) -> int:
    import matplotlib

    matplotlib.use("Agg")
    from vaft.data.public import normalize_db5, read_db5

    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--atlas", required=True, help="Lane D atlas directory (table.csv, closures/)")
    p.add_argument("--out", required=True)
    args = p.parse_args(argv)
    atlas = Path(args.atlas).expanduser()
    out = Path(args.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)

    confinement = pd.read_csv(atlas / "table.csv")
    closures = pd.read_csv(atlas / "closures" / "closures.csv")
    nstx = pd.read_csv(atlas / "closures" / "nstx_comparison.csv")
    table = population_table(normalize_db5(read_db5()), confinement)
    exponents = exponent_table(closures, nstx)

    figures = {
        "confinement_population": population_figure(table)[0],
        "confinement_predicted_vs_measured": predicted_vs_measured_figure(table)[0],
        "confinement_h_factor": h_factor_figure(table)[0],
        "confinement_exponents": exponent_figure(exponents)[0],
    }
    written = []
    for name, fig in figures.items():
        for ext in ("png", "pdf"):
            path = out / f"{name}.{ext}"
            fig.savefig(path, dpi=200)
            written.append(path.name)
    exponents.to_csv(out / "exponents.csv", index=False)
    inputs = [atlas / "table.csv", atlas / "closures" / "closures.csv", atlas / "closures" / "nstx_comparison.csv"]
    manifest = {
        "generated_at": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "command": " ".join(sys.argv), "vaft_git": _git("rev-parse", "HEAD"),
        "vaft_dirty": bool(_git("status", "--porcelain")),
        "inputs": {str(f): hashlib.sha256(f.read_bytes()).hexdigest() for f in inputs},
        "db5": "ITPA DB5.2.3 standard set (SELDB5), vaft.data.public.read_db5",
        "vest_rows": int((table["machine"] == "VEST").sum()), "files": written,
        "notes": ("VEST: primary selection, W_mhd, P_net = P_OH - dW/dt (radiation not subtracted, as DB5 "
                  "PLTH), ohmic L-mode; IPB98 H factors locate VEST, they are not a performance claim. mu_rho "
                  "is Kadomtsev-completed (assumed size exponent)."),
    }
    (out / "MANIFEST.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps({k: manifest[k] for k in ("vest_rows", "files")}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
