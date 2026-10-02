# Confinement scaling (lane D, #548)

This directory builds the VEST Tier A confinement table on a validated ohmic power balance (layer A of the #548 hierarchy) and fits it (layers B and C). It also runs errors in variables, Kadomtsev completion, the dimensionless form and the Bohm / −2.5 / gyro-Bohm comparison (layers D–F), with NSTX as an explicit comparison. Progress and decisions are recorded in the Lane D log on GitHub.

| script | what it does |
|---|---|
| `build_table.py` | One row per magnetics state key of Lane K's state key contract v1 (#1454). It writes Lane M's `CONFINEMENT_COLUMNS` (`vaft.data.public.schema`) followed by Lane D extension columns: three stored energies, every power-balance term, slice-quality evidence and rule flags. |
| `closures.py` | Runs four analyses: an errors-in-variables fit of W against P_OH (independent of W) and P_net (for comparison; it carries W noise through dW/dt), over a grid of the assumed equation error of W and measurement error of P; the Kadomtsev-completed size exponent and dimensionless indices, with the cluster covariance propagated and the size exponent labelled `assumed_not_measured`; closures μ_ρ = −2, −2.5 and −3 as linear constraints, compared by RMS, AIC/BIC, a cluster Wald test, leave-one-shot-out error and bootstrap spread; and NSTX (Buxton 2019, Kaye 2006) as a comparison, never a prior. |
| `figures.py` | Figure functions for Lane V's conference notebook (`vaft.plot` is frozen). Each returns `(fig, axes)`: VEST over the ITPA DB5.2.3 standard set with the spherical tokamaks broken out; τ_E measured against IPB98(y,2) and NSTX2006L; the H-factor distributions; and the exponent comparison with the Kadomtsev-completed μ_ρ, marked as assumed and as undetermined where (1+α_P)/σ < 2. The CLI renders PNG and PDF. |
| `extra_scalings.py` | The ohmic and L-mode scalings `vaft.formula` lacks, transcribed from the papers and evaluated in their own units: neo-Alcator (Goldston 1984 eq. 3), Goldston 1984 L-mode (eq. 6), their eq. (11) quadrature, and ITER97-L (Kaye 1997). They belong in `vaft.formula` under #670. |
| `extensions.py` | Tests what the exponents measure: B_T split into TF current (B0R0) and plasma position (R_geo); shot fixed effects (within-shot) against shot means (between-shot); a campaign offset for the two Tier A shot blocks; and a direct dimensionless regression on the Thomson subset (no T elimination through P). |
| `fit.py` | Fits τ_E = C I_p^aI B_T^aB P_net^aP, density-free (A, primary) and with the Thomson line density (B) on the same subset. It reports errors clustered by shot, a shot bootstrap, leave-one-shot-out refits, influence, a Huber fit and identifiability (VIF, condition number, correlations, log spread). Each fit runs on the primary and the sensitivity selection. |

## Inputs (read only)

- `~/runs/campaign/atlas/v1/state.csv`, Lane K: the #1331 Tier A good + admissible EFIT slices.
- `~/runs/campaign/filedb`, the campaign FileDB:
  - `omas/efit/magnetic/<shot>`: W_mhd, li_3 and geometry;
  - `omas/diagnostics/<shot>`: measured Ip and the inboard flux-loop voltage;
  - `omas/core_profiles/<shot>`: the Thomson n_e that gives the z = 0 line density;
  - `omas/electron_efit/<shot>`: W_kin.

## Definitions

- **P_OH** = I_p,meas · V_res. V_res is the measured inboard-midplane loop voltage minus the internal inductive voltage (1/I_p) dW_int/dt, with W_int = μ0 R0 li_3 I_p²/4 (Romero 2010 eq. 24). The function is `vaft.process.confinement.resistive_loop_voltage`, which also returns R_p.
- **The EFIT boundary-flux path** (`compute_voltage_consumption`) is kept only as a comparison column. The Tier A products store ψ in Wb, and their early slices jump by tens of volts.
- **The Spitzer path** is a comparison only, and runs only with `--spitzer` and an explicit Z_eff and ln Λ (#1188).
- **dW/dt and dli_3/dt** are computed only over the shot's labelled (good/admissible) slices, using a local linear fit over 3 ms. The other slices of a product are unreconstructed, with W down to −75 kJ.
- **Stored energy, three columns:**
  - `w_mhd_J` = 1.5∫p dV of the magnetics EFIT. This is `w_th_J`, the energy τ_E uses.
  - `w_kin_J`: the same integral for the paired electron_kinetic EFIT.
  - `w_e_ts_J` = 1.5∫p_e dV of the Thomson fit alone, mapped through its own ρ_pol onto the EFIT grid inside the LCFS.
- **P_net** = P_OH − dW/dt. This is `p_loss_W`; radiation is not subtracted, as in DB5 `PLTH`.
- **P_transport** is NaN: VEST maps no bolometer.
- **τ_E** = W_mhd / P_net. The kinetic counterpart is `tau_e_kin_s`, which uses the magnetics dW/dt.
- **EFIT verdicts** come from Lane K's state table under criteria v2 (#1521), as two separate columns:
  - `efit_quality` (good / admissible / ...) grades the fit alone;
  - `thomson_consistent` is True when the EFIT pressure lies within [1, 2]·p_e of the Thomson fit, False outside, and NaN where no Thomson profile was matched.

Units of every extension column are in `EXTENSION_UNITS` (`build_table.py`) and in the product's `schema/` files.

## Slice quality

Every row carries its evidence:
- `ip_rate_1_s`;
- `ip_change_per_tau`, which is |dI_p/dt|/I_p · τ_E;
- `dwdt_fraction`, which is |dW/dt|/P_OH.

The `rule_*` and `accepted` columns apply the provisional working thresholds in `MANIFEST.json`. `threshold_sweep.csv` gives the accepted count and the number of shots on a threshold grid. The thresholds are not adopted until the sweep has been read; VEST discharges have no flat top, so stationarity is judged by rate, not by phase.

## Selections (decided 2026-10-02 on #1490)

| | \|ΔI_p\| per τ_E | \|dW/dt\|/P_OH | \|I_p\| |
|---|---|---|---|
| primary | ≤ 0.20 | ≤ 1.0 | ≥ 30 kA |
| sensitivity | ≤ 0.05 | ≤ 0.5 | ≥ 30 kA |

`fit.py` re-derives both selections from the evidence columns, so changing them does not need a rebuild.

## Run

On vestserver, use an isolated worktree with the import shim and assert `vaft.__file__`:

```bash
python build_table.py --state ~/runs/campaign/atlas/v1/state.csv \
    --filedb ~/runs/campaign/filedb --out ~/runs/campaign/atlas/confinement --spitzer
```

Outputs:
- `table.csv`, `exclusions.csv`, `threshold_sweep.csv`;
- `series/<shot>.csv`, the per-shot power balance on the labelled EFIT slices;
- `schema/table.schema.json` and `MANIFEST.json`;
- `failures.csv`, written only if a shot fails.

Then fit:

```bash
python fit.py --table ~/runs/campaign/atlas/confinement/table.csv --out ~/runs/campaign/atlas/confinement/fits
```

This writes `coefficients.csv`, `summary.csv`, `influence.csv`, `identifiability.json` and `MANIFEST.json`. Then run the closures:

```bash
python closures.py --table ~/runs/campaign/atlas/confinement/table.csv --out ~/runs/campaign/atlas/confinement/closures
```

This writes `closures.csv`, `odr_scan.csv`, `nstx_comparison.csv` and `MANIFEST.json`. Every dimensionless index divides by 1 + α_P, so read `one_plus_aP_over_se` before any μ: within about 2σ of zero, the completed indices are undetermined.

Then the extensions:

```bash
python extensions.py --table ~/runs/campaign/atlas/confinement/table.csv --out ~/runs/campaign/atlas/confinement/extensions
```

Then the figures:

```bash
python figures.py --atlas ~/runs/campaign/atlas/confinement --out ~/runs/campaign/atlas/confinement/figures
```

The TGLF saturation-rule link (#1482) reads Lane T's `transport/` and `transport_sensitivity/` products beside the table:

```bash
python tglf_link.py --atlas ~/runs/campaign/atlas --out ~/runs/campaign/atlas/confinement/tglf_link
```

This writes `power_link.csv` (P = (Q_e + Q_i) dV/dr per state, r/a and TGLF configuration, beside P_OH and P_net) and `power_link_summary_r0.8.csv`. It is a consistency test, not an identity: the flow through r carries only the power deposited inside r.

Nothing is written to the FileDB.
