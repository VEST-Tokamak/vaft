# Resistive Z_eff atlas (#1214, Lane Z)

`build_zeff.py` infers the resistive effective charge `Zeff_resistive` for the
Lane K Tier A states (contract #1454). It matches the plasma resistance closed
by Romero's transformer balance against a parallel-conductivity model. The
result is **model-inferred, not a composition measurement**, and nothing is
written to `core_profiles.zeff`.

## Inputs (read only)

- `--filedb`: the campaign FileDB. Each window reads one EFIT product
  (`omas/efit/magnetic/<shot>/output/efit.json.gz`).
- `--atlas`: the Lane K state atlas (`state.csv`, `core_profiles/<shot>.json.gz`).

## Outputs (`--out`)

| file | one row per |
|---|---|
| `zeff.csv` | window: status, estimate, uncertainty, the three models, the largest sensitivity per input, and the reason when there is no estimate |
| `slices.csv` | Lane K key `(shot, time_efit_s, efit_lineage)` in a window: V_B, V_I, V_R, R_p^obs, and R_p^model at the fit |
| `sensitivity.csv` | re-fit: T_e, n_e and li_3 x(1 ± 0.1), alternative conductivity model |
| `schema/*.schema.json`, `MANIFEST.json` | generated from the column dictionaries; input hashes, commit and settings |

## Defaults and what they mean

| setting | value | why |
|---|---|---|
| window | longest run of consecutive graded (good/admissible) magnetics slices | the balance differentiates in time |
| nominal model | Redl | #1214 Sec. 3: the nominal model at spherical-tokamak trapped fractions |
| `ln_lambda` | Sauter's per-surface value | — |
| bounds | `[1, 8]` | a minimum on a bound is reported as `bound_hit`, never clipped |
| `I_ni` | `0` | stated Ohmic assumption; there is no NBI, ECCD or helicity injection in Tier A |
| bootstrap | none | it needs T_i, which Tier A only has as an assumption |
| smoothing | none | EFIT slices are 1 ms apart, so even a first-order local fit has no redundancy |
| `--inductive-max` | 1.0 | slices where |V_I| ≥ |V_B| are not fitted |
| `--max-excluded-current` | 0.05 | states with more than 5 % of the current outside the profiles' support are not fitted: the model omits the cold edge, so Z_eff would be biased high |
| `--stationary-rate` | 20 /s | `flattop` needs median \|dln I_p/dt\| and \|dln L_i/dt\| both below it; otherwise `ip_stationary_li_evolving`, `ramp_up` or `decay`. A small \|V_I\|/\|V_B\| is not used: the two V_I terms cancel in the decay (#1514) |

**Boundary voltage cross-check (Lane D, #548).** The observed path uses
Romero's EFIT ψ_B (#1214 Sec. 4). Every fitted window is also re-fitted with
V_B taken from the inboard midplane flux loop #10 (R = 0.091 m), which is
what Lane D's `resistive_loop_voltage` uses. The loop sits inside the LCFS,
so the flux between the loop and the LCFS biases it. The shift is reported
as `zeff_flux_loop` / `delta_boundary_voltage_source`, as a
`boundary_voltage_source` row in `sensitivity.csv`, and per slice as
`v_loop_fl10_v`.

A window that cannot be inferred is still a row, with
`status = not_identifiable` and the reason (#1214 Sec. 10).

## Run (vestserver)

```bash
cd ~/work/lane-z && PYTHONPATH=~/work/lane-z-shim python3 workflow/resistive_zeff/build_zeff.py \
    --filedb ~/runs/campaign/filedb --atlas ~/runs/campaign/atlas/v1 \
    --out ~/runs/campaign/atlas/zeff
```
