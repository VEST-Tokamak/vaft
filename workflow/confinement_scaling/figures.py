"""Lane D figures for the conference atlas (#548; handed to Lane V, #1456).

``vaft.plot`` is frozen, so these are workflow functions. Each returns
``(fig, axes)``, so ``notebooks/conference_operational_space_atlas.ipynb`` can
call them directly. They draw only from Lane D's files and Lane M's public data:

- ``population_figure``: VEST over the ITPA DB5.2.3 standard set (SELDB5),
  with the spherical tokamaks (NSTX, MAST, START) apart from the conventional
  ones. It reuses ``vaft.plot.population.confinement_population``.
- ``predicted_vs_measured_figure`` and ``h_factor_figure``: tau_E against
  IPB98(y,2) and NSTX2006L by default, or any of ``ALL_SCALINGS``. That covers the
  five of ``vaft.formula`` (via Lane M's ``predict_confinement_time``) and the
  ohmic/L-mode ones of ``extra_scalings.py`` (neo-Alcator, Goldston 1984, ITER97-L).
  The plot functions are Lane M's.
- Variants: by machine (the seven largest, the rest as "Other"), and every DB5 row
  instead of the standard set. VEST is ohmic L-mode, so its IPB98 H factor is a location, not a
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
import extra_scalings  # noqa: E402
from fit import SELECTIONS, _git, select  # noqa: E402

SPHERICAL = ("NSTX", "MAST", "START")
LABELS = {"H98y2": "IPB98(y,2)", "NSTX2006L": "NSTX 2006 L-mode", "NSTX2006H": "NSTX 2006 H-mode",
          "ITER89P": "ITER89-P (L-mode)", "Kurskiev2022": "Kurskiev 2022 (ST H-mode)"}
LABELS.update(extra_scalings.LABELS)
#: Every confinement scaling vaft.formula carries: the _SCALING_COEFS power laws plus
#: the ohmic and L-mode ones extra_scalings.py maps onto table columns (#670);
#: ohmic, then L-mode, then H-mode.
ALL_SCALINGS = ("NeoAlcator", "Goldston84OhmicL", "Goldston84L", "ITER89P", "ITER97L", "NSTX2006L",
                "H98y2", "NSTX2006H", "Kurskiev2022")


#: Where each scaling was fitted and for which regime, as a compact label tag:
#: database ("single" machine, "multi"-machine; "ST" when spherical tokamaks only)
#: and confinement mode. Goldston 1984 and ITER89-P/97-L are multi-machine L-mode
#: (neo-Alcator: ohmic) compilations; NSTX 2006 is NSTX alone; Kurskiev 2022 is the
#: multi-machine ST H-mode set; IPB98(y,2) the ITPA ELMy H-mode set.
SCALING_TAGS = {
    "NeoAlcator": ("multi", "ohmic"),
    "Goldston84OhmicL": ("multi", "ohmic+L"),
    "Goldston84L": ("multi", "L"),
    "ITER89P": ("multi", "L"),
    "ITER97L": ("multi", "L"),
    "NSTX2006L": ("single ST", "L"),
    "H98y2": ("multi", "H"),
    "NSTX2006H": ("single ST", "H"),
    "Kurskiev2022": ("multi ST", "H"),
}
#: Box colour per database tag of SCALING_TAGS.
DATABASE_COLOURS = {"multi": "#9ecae1", "single ST": "#fdae6b", "multi ST": "#a1d99b"}
#: Box hatch per confinement mode of SCALING_TAGS. A new mode takes the next unused
#: pattern of SPARE_HATCHES (dots first), so the encoding extends without a redesign.
MODE_HATCHES = {"ohmic": "", "ohmic+L": "\\\\", "L": "//", "H": "--"}
SPARE_HATCHES = ("..", "xx", "oo", "++")


def mode_hatch(mode: str) -> str:
    """Hatch pattern of a confinement mode; an unlisted mode gets a spare pattern, stably."""
    if mode in MODE_HATCHES:
        return MODE_HATCHES[mode]
    return SPARE_HATCHES[sum(map(ord, mode)) % len(SPARE_HATCHES)]


def tagged_label(scaling: str) -> str:
    """'ITER97-L (Kaye 1997)  [multi | L]': the scaling's label with its database and mode."""
    database, mode = SCALING_TAGS.get(scaling, ("?", "?"))
    return f"{LABELS.get(scaling, scaling)}  [{database} | {mode}]"


def predict_tau(table: pd.DataFrame, scaling: str) -> pd.Series:
    """Predicted tau_E of any scaling, from vaft.data.public or extra_scalings [s]."""
    from vaft.data.public import predict_confinement_time

    if scaling in extra_scalings.NAMES:
        return extra_scalings.predict(table, scaling)
    return predict_confinement_time(table, scaling)


def h_factor_of(table: pd.DataFrame, scaling: str) -> pd.Series:
    """tau_e_th_s over the scaling's prediction [-]."""
    tau = pd.to_numeric(table["tau_e_th_s"], errors="coerce")
    return (tau / predict_tau(table, scaling)).rename(f"h_{scaling}")
#: Machines coloured individually when grouping by machine (the largest by row count);
#: the rest fold into "Other". vaft.plot.population has seven colours, and more groups
#: would repeat them.
MACHINE_GROUPS = 7
VEST_LABEL = "VEST (ohmic, this work)"
#: Distinct open markers for the reference scalings, so each can be told apart.
REFERENCE_MARKERS = ("s", "D", "^", "v", "P")
#: IPB98(y,2) engineering exponents at fixed epsilon (ITER Physics Basis 1999).
IPB98 = {"i_p": 0.93, "b_t": 0.15, "p_net": -0.69, "n_e": 0.41}
UNDETERMINED_SIGMA = 2.0


def population_table(db5: pd.DataFrame, confinement: pd.DataFrame, selection: str = "primary",
                     *, scope: str = "standard", grouping: str = "spherical") -> pd.DataFrame:
    """DB5 rows plus the selected VEST rows, with a ``population`` label column.

    ``scope``: ``"standard"`` keeps the DB5 standard set (SELDB5), ``"all"`` every
    DB5 row. ``grouping``: ``"spherical"`` labels NSTX, MAST and START apart from
    one conventional-tokamak group; ``"machine"`` labels every machine.
    """
    from vaft.data.public.schema import CONFINEMENT_COLUMNS

    if scope not in ("standard", "all") or grouping not in ("spherical", "machine"):
        raise ValueError(f"scope must be standard/all and grouping spherical/machine; got {scope}, {grouping}")
    vest = confinement.loc[select(confinement, SELECTIONS[selection])]
    keep = list(CONFINEMENT_COLUMNS) + (["thomson_consistent"] if "thomson_consistent" in vest else [])
    vest = vest.loc[np.isfinite(vest["tau_e_th_s"]) & (vest["tau_e_th_s"] > 0), keep]
    rows = db5 if scope == "all" else db5.loc[db5["selected"].astype(bool)]
    table = pd.concat([rows[list(CONFINEMENT_COLUMNS)], vest], ignore_index=True)
    machine = table["machine"].astype(str)
    if grouping == "machine":
        table["population"] = machine
    else:
        table["population"] = np.where(machine.isin(SPHERICAL), machine + " (DB5)",
                                       "conventional tokamaks (DB5)")
    table.loc[machine == "VEST", "population"] = VEST_LABEL
    return table


def population_figure(table: pd.DataFrame, *, max_groups: int = 4, figsize=(11.0, 4.6)):
    """tau_E against I_p and against P_loss: VEST among the DB5 population."""
    import matplotlib.pyplot as plt

    from vaft.plot.population import confinement_population

    fig, axes = plt.subplots(1, 2, figsize=figsize, constrained_layout=True)
    # VEST sits bottom left and the I_p panel's upper left is empty; the two panels
    # draw the same series, so the P_loss panel, which has no empty corner, keeps none.
    for ax, x in zip(axes, ("i_p_A", "p_loss_W")):
        confinement_population(table, x=x, y="tau_e_th_s", by="population",
                               highlight=VEST_LABEL, max_groups=max_groups, ax=ax)
        mark_thomson_inconsistent(ax, table, x, "tau_e_th_s", legend_loc="upper left")
    if axes[1].get_legend() is not None:
        axes[1].get_legend().remove()
    axes[0].set_title(r"$\tau_E$ against $I_p$")
    axes[1].set_title(r"$\tau_E$ against $P_{loss}$ (VEST: $P_{OH} - dW/dt$)")
    return fig, axes


#: Legend text of the ring drawn around a VEST point whose EFIT pressure is outside
#: [1, 2] p_e of its Thomson profile (Lane K criteria v2, #1521).
THOMSON_RING_LABEL = "VEST: EFIT p outside [1, 2] p_e (#1521)"


def thomson_inconsistent(table: pd.DataFrame) -> np.ndarray:
    """VEST rows whose criteria-v2 verdict is False (NaN, no Thomson, is not marked)."""
    if "thomson_consistent" not in table:
        return np.zeros(len(table), dtype=bool)
    verdict = table["thomson_consistent"].map(
        {True: False, False: True, "True": False, "False": True, "true": False, "false": True})
    return ((table["machine"] == "VEST") & verdict.fillna(False).astype(bool)).to_numpy()


def mark_thomson_inconsistent(ax, table: pd.DataFrame, x, y: str, *, legend_loc: str = "best") -> None:
    """Ring the VEST points of ``ax`` whose stored energy disagrees with Thomson.

    ``x`` is a column name or an array aligned with ``table``. The ring is drawn
    only when some row is marked; the legend is redrawn in the style of
    ``vaft.plot.population`` (x-small, frameless) so the ring joins it.
    """
    mask = thomson_inconsistent(table)
    if not mask.any():
        return
    import matplotlib

    from vaft.plot.population import LABELS as POPULATION_UNITS

    def values(v):
        if not isinstance(v, str):
            return np.asarray(v, float)
        # The population plots draw a column in its display unit (MA, MW, ...).
        scale = POPULATION_UNITS.get(v, (None, None, 1.0))[2]
        return pd.to_numeric(table[v], errors="coerce").to_numpy(float) * scale

    xs, ys = values(x), values(y)
    shown = mask & np.isfinite(xs) & np.isfinite(ys)
    if not shown.any():
        return
    size = 2.2 * float(matplotlib.rcParams["lines.markersize"]) ** 2
    ax.scatter(xs[shown], ys[shown], s=size, facecolors="none", edgecolors="crimson",
               linewidths=1.2, zorder=6, label=f"{THOMSON_RING_LABEL} ({int(shown.sum())})")
    if ax.get_legend() is not None:
        ax.legend(fontsize="x-small", markerscale=1.5, frameon=False, loc=legend_loc)


def _grid(n: int, panel=(5.4, 5.0), figsize=None):
    """A figure with n panels, at most three per row; unused panels hidden."""
    import matplotlib.pyplot as plt

    ncols = min(3, n)
    nrows = int(np.ceil(n / ncols))
    size = figsize if figsize is not None else (panel[0] * ncols, panel[1] * nrows)
    fig, axes = plt.subplots(nrows, ncols, figsize=size, constrained_layout=True, squeeze=False)
    flat = axes.ravel()
    for ax in flat[n:]:
        ax.set_visible(False)
    return fig, flat[:n]


def predicted_vs_measured_figure(table: pd.DataFrame, scalings=("H98y2", "NSTX2006L"), *, max_groups: int = 4,
                                 figsize=None):
    """Measured against scaling-predicted tau_E, one panel per scaling."""

    from vaft.plot.population import confinement_predicted_vs_measured

    fig, axes = _grid(len(scalings), figsize=figsize)
    for ax, scaling in zip(axes, scalings):
        predicted = predict_tau(table, scaling)
        confinement_predicted_vs_measured(table, predicted, scaling_label=LABELS.get(scaling, scaling),
                                          by="population", highlight=VEST_LABEL, max_groups=max_groups, ax=ax)
        mark_thomson_inconsistent(ax, table, predicted.to_numpy(float), "tau_e_th_s", legend_loc="upper left")
        ax.set_title(f"{LABELS.get(scaling, scaling)}: {_vest_coverage(table, predicted)}", fontsize="small")
    return fig, axes


def _vest_coverage(table, predicted) -> str:
    """'VEST n/N rows': a scaling with a density term drops rows without Thomson n_e."""
    vest = (table["machine"] == "VEST").to_numpy()
    shown = int(np.sum(vest & np.isfinite(predicted.to_numpy(float))))
    note = " (Thomson n_e)" if shown < vest.sum() else ""
    return f"VEST {shown}/{int(vest.sum())} rows{note}"


def h_factor_figure(table: pd.DataFrame, scalings=("H98y2", "NSTX2006L")):
    """H-factor distributions per population, one panel per scaling."""
    from vaft.plot.population import confinement_h_factor_distribution

    n_groups = int(table["population"].nunique())
    fig, axes = _grid(len(scalings), panel=(5.6, max(4.6, 0.33 * n_groups + 1.5)))
    for ax, scaling in zip(axes, scalings):
        h = h_factor_of(table, scaling)
        confinement_h_factor_distribution(table, h, scaling_label=LABELS.get(scaling, scaling),
                                          by="population", highlight=VEST_LABEL, ax=ax)
        ax.set_title(f"{LABELS.get(scaling, scaling)}: {_vest_coverage(table, h)}", fontsize="small")
    return fig, axes


def vest_h_factor_figure(table: pd.DataFrame, scalings=ALL_SCALINGS, *, figsize=(9.5, 5.6)):
    """VEST alone: the H factor against each scaling, one box per scaling.

    ``table`` holds VEST rows only (e.g. the primary selection); a scaling with a
    density term uses the rows with a Thomson density. The axis names only the
    scaling: the box colour is where it was fitted and the hatch its confinement
    mode (SCALING_TAGS), each with its own legend.
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    values, labels, colours, hatches = [], [], [], []
    for name in scalings:
        h = h_factor_of(table, name).replace([np.inf, -np.inf], np.nan).dropna()
        values.append(h.to_numpy(float))
        database, mode = SCALING_TAGS.get(name, ("?", "?"))
        labels.append(LABELS.get(name, name))
        colours.append(DATABASE_COLOURS.get(database, "0.85"))
        hatches.append(mode_hatch(mode))
    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    # Top to bottom in the order given (ALL_SCALINGS runs ohmic, L, H).
    positions = np.arange(len(values))[::-1]
    boxes = ax.boxplot(values, positions=positions, vert=False, widths=0.6, patch_artist=True,
                       medianprops=dict(color="k"), flierprops=dict(markersize=3))
    for patch, colour, hatch in zip(boxes["boxes"], colours, hatches):
        patch.set_facecolor(colour)
        patch.set_hatch(hatch)
    ax.set_yticks(positions, labels, fontsize="small")
    ax.axvline(1.0, color="k", ls="--", lw=1)
    ax.set_xscale("log")
    ax.set_xlabel(r"$H = \tau_{E,th}/\tau_{E,scaling}$")
    ax.grid(alpha=0.25, which="both", axis="x")
    ax.set_title("VEST (ohmic): H against each scaling", fontsize="medium")
    tags = [SCALING_TAGS.get(name, ("?", "?")) for name in scalings]
    databases = list(dict.fromkeys(db for db, _ in tags))
    modes = list(dict.fromkeys(mode for _, mode in tags))
    # Short labels under a "fit database" title: the first legend is placed with
    # add_artist, which the layout engine does not make room for.
    fit_handles = [Patch(facecolor=DATABASE_COLOURS.get(db, "0.85"), edgecolor="k", label=db) for db in databases]
    mode_handles = [Patch(facecolor="white", edgecolor="k", hatch=mode_hatch(m),
                          label=f"{m}-mode" if len(m) == 1 else m) for m in modes]
    # Both legends sit outside the axes, right: no H range is free of boxes or outliers.
    first = ax.legend(handles=fit_handles, fontsize="x-small", loc="upper left", bbox_to_anchor=(1.01, 1.0),
                      frameon=False, title="fit database", title_fontsize="x-small", alignment="left")
    ax.add_artist(first)
    ax.legend(handles=mode_handles, fontsize="x-small", loc="lower left", bbox_to_anchor=(1.01, 0.0),
              frameon=False, title="regime", title_fontsize="x-small", alignment="left")
    return fig, ax


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
                             gridspec_kw={"width_ratios": [1.0, 1.15]})
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
    ax.set_xticks(range(len(terms)), [t for _, t in terms], fontsize="large")
    # The right panel names every series beside its marker, so it is the legend.
    ax.set_ylabel("exponent (VEST ±1σ, by shot)")
    ax.set_title(r"$\tau_E \propto I_p^{\alpha_I} B_T^{\alpha_B} P^{\alpha_P}$", fontsize="medium")

    ax = axes[1]
    ys = np.arange(len(exponents))[::-1]
    for y, (_, r) in zip(ys, exponents.iterrows()):
        st = {k: v for k, v in styles[r["label"]].items() if k != "zorder"}
        if r["mu_rho_undetermined"] or not np.isfinite(float(r["mu_rho"])):
            ax.text(-6.8, y, r"undetermined: $(1+\alpha_P)/\sigma < $" + f"{UNDETERMINED_SIGMA:g}",
                    ha="left", va="center", fontsize="x-small", color="k",
                    bbox=dict(boxstyle="round", fc="white", ec="0.6"))
            continue
        se = float(r["mu_rho_se"]) if np.isfinite(float(r["mu_rho_se"])) else 0.0
        ax.errorbar(float(r["mu_rho"]), y, xerr=se or None, ls="none", capsize=2, **st)
    ax.axvspan(-3.0, -2.0, color="0.92", zorder=0)
    for x, name, align in ((-2.0, "Bohm", "left"), (-3.0, "gyro-Bohm", "right")):
        ax.axvline(x, color="0.6", lw=0.8, ls=":")
        ax.text(x, 1.01, name, transform=ax.get_xaxis_transform(), ha=align, va="bottom", fontsize="x-small")
    ax.set_yticks(ys, exponents["label"], fontsize="x-small")
    ax.set_xlim(-7.0, 0.0)
    ax.set_ylim(-0.7, len(exponents) - 0.3)
    ax.set_xlabel(r"$\mu_\rho$ in $\Omega_i\tau_E \propto \rho_*^{\mu_\rho}\beta^{\mu_\beta}\nu_*^{\mu_\nu}$"
                  "\nKadomtsev-completed: size exponent ASSUMED", fontsize="small", loc="right")
    return fig, axes


#: The conference set: (name, scalings) drawn in vaft.plot's slide format (#1572).
SLIDE_SCALINGS = ("ITER97L", "H98y2")


def slide_figures(table: pd.DataFrame, exponents: pd.DataFrame, *, theme: str = "minimal",
                  fmt: str = "slide") -> dict:
    """The conference figures in a ``vaft.plot`` presentation format: name -> figure.

    Width and height ceiling, type size and line scale come from the format; the
    figures are drawn inside its rc context, so the relative font sizes of the
    functions above scale with it. VEST points whose EFIT pressure is outside
    [1, 2] p_e of Thomson are ringed.
    """
    from vaft.plot.presentation import resolve_presentation

    pres = resolve_presentation(fmt, theme)
    if pres is None or pres.format is None:
        raise ValueError(f"slide_figures needs a presentation format such as 'slide'; got {fmt!r}")
    width, ceiling = pres.format.width_in, pres.format.max_height_in
    with pres.context():
        return {
            "tau_population": population_figure(table, figsize=(width, min(0.5 * width, ceiling)))[0],
            "tau_predicted_vs_measured": predicted_vs_measured_figure(
                table, SLIDE_SCALINGS, figsize=(width, min(0.52 * width, ceiling)))[0],
            "exponents": exponent_figure(exponents, figsize=(width, ceiling))[0],
            "vest_h_factor": vest_h_factor_figure(table.loc[table["machine"] == "VEST"],
                                                  figsize=(width, ceiling))[0],
        }


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
    db5 = normalize_db5(read_db5())
    table = population_table(db5, confinement)
    by_machine = population_table(db5, confinement, grouping="machine")
    all_rows = population_table(db5, confinement, scope="all", grouping="machine")
    exponents = exponent_table(closures, nstx)

    figures = {
        "confinement_population": population_figure(table)[0],
        "confinement_predicted_vs_measured": predicted_vs_measured_figure(table)[0],
        "confinement_h_factor": h_factor_figure(table)[0],
        "confinement_exponents": exponent_figure(exponents)[0],
        # Every machine coloured, standard set and every DB5 row.
        "confinement_population_by_machine": population_figure(by_machine, max_groups=MACHINE_GROUPS)[0],
        "confinement_population_all_db5": population_figure(all_rows, max_groups=MACHINE_GROUPS)[0],
        # Every scaling vaft.formula carries.
        "confinement_predicted_vs_measured_all_scalings":
            predicted_vs_measured_figure(by_machine, ALL_SCALINGS, max_groups=MACHINE_GROUPS)[0],
        "confinement_h_factor_all_scalings": h_factor_figure(by_machine, ALL_SCALINGS)[0],
        # VEST alone against every scaling, tagged by fit database and mode.
        "confinement_vest_h_factor": vest_h_factor_figure(table.loc[table["machine"] == "VEST"])[0],
    }
    written = []
    for name, fig in figures.items():
        for ext in ("png", "pdf"):
            path = out / f"{name}.{ext}"
            fig.savefig(path, dpi=200)
            written.append(path.name)
    slide_dir = out / "slide"
    slide_dir.mkdir(exist_ok=True)
    for name, fig in slide_figures(table, exponents).items():
        for ext in ("png", "pdf"):
            path = slide_dir / f"{name}.{ext}"
            fig.savefig(path, dpi=200)
            written.append(f"slide/{path.name}")
    exponents.to_csv(out / "exponents.csv", index=False)
    inputs = [atlas / "table.csv", atlas / "closures" / "closures.csv", atlas / "closures" / "nstx_comparison.csv"]
    manifest = {
        "generated_at": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "command": " ".join(sys.argv), "vaft_git": _git("rev-parse", "HEAD"),
        "vaft_dirty": bool(_git("status", "--porcelain")),
        "inputs": {str(f): hashlib.sha256(f.read_bytes()).hexdigest() for f in inputs},
        "db5": ("ITPA DB5.2.3 via vaft.data.public.read_db5: the standard set (SELDB5), except "
                "confinement_population_all_db5, which shows every row"),
        "scalings_all": list(ALL_SCALINGS),
        "vest_rows": int((table["machine"] == "VEST").sum()), "files": written,
        "notes": ("VEST: primary selection, W_mhd, P_net = P_OH - dW/dt (radiation not subtracted, as DB5 "
                  "PLTH), ohmic L-mode; IPB98 H factors locate VEST, they are not a performance claim. mu_rho "
                  "is Kadomtsev-completed (assumed size exponent)."),
    }
    (out / "MANIFEST.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({k: manifest[k] for k in ("vest_rows", "files")}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
