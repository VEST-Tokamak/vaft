# Confinement scaling (lane D, #548)

This directory builds the VEST Tier A confinement table on a validated ohmic power balance, the first layer (A) of the #548 hierarchy. Regression, identifiability, Kadomtsev completion and the Bohm/gyro-Bohm comparison come in follow-up PRs. Progress and decisions are recorded in the Lane D log on GitHub.

| script | what it does |
|---|---|
| `build_table.py` | One row per magnetics state key of Lane K's state key contract v1 (#1454). It writes Lane M's `CONFINEMENT_COLUMNS` (`vaft.data.public.schema`) followed by Lane D extension columns: both stored energies, every power-balance term, slice-quality evidence and rule flags. |

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
- **P_net** = P_OH − dW/dt. This is `p_loss_W`; radiation is not subtracted, as in DB5 `PLTH`.
- **P_transport** is NaN: VEST maps no bolometer.
- **τ_E** = W_mhd / P_net. The kinetic counterpart is `tau_e_kin_s`, which uses the magnetics dW/dt.

## Slice quality

Every row carries its evidence:
- `ip_rate_1_s`;
- `ip_change_per_tau`, which is |dI_p/dt|/I_p · τ_E;
- `dwdt_fraction`, which is |dW/dt|/P_OH.

The `rule_*` and `accepted` columns apply the provisional working thresholds in `MANIFEST.json`. `threshold_sweep.csv` gives the accepted count and the number of shots on a threshold grid. The thresholds are not adopted until the sweep has been read; VEST discharges have no flat top, so stationarity is judged by rate, not by phase.

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

Nothing is written to the FileDB.
