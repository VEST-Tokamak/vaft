# Kinetic state (lane K)

These scripts build the conference atlas's kinetic-state tables from the #1331 Tier A EFIT campaign.

The state key contract is defined in #1454: rows are keyed by `(shot, time_efit_s, efit_lineage)` and every row carries `efit_quality`. The slice-level calculations live in `vaft.validation.kinetic_state`. The Thomson comparison is the sampler that `criteria.py`'s Thomson verdict grades, `vaft.validation.equilibrium.thomson_pressure_samples`.

| script | reads | writes |
|---|---|---|
| `build_state.py` | the campaign FileDB and `tierA_analysis.json` (slice grades) | `state.csv`, `profiles.csv`, `schema/*.schema.json`, `MANIFEST.json` |
| `summarize.py` | `state.csv` and the analysis JSONs | `summary.json`, `figures/pressure_consistency.{png,pdf}` |
| `build_ti.py` | `state.csv`, `profiles.csv` and the FileDB | `ti_inferred.csv`, `ti_state.csv`, `ti_summary.json`, `core_profiles/<shot>.json.gz` |

On vestserver the outputs go to `~/runs/campaign/atlas/v1/`. Nothing is written to the FileDB.

```bash
python3 workflow/kinetic_state/build_state.py --filedb ~/runs/campaign/filedb \
    --analysis ~/runs/campaign/tierA_analysis.json --out ~/runs/campaign/atlas/v1
python3 workflow/kinetic_state/summarize.py --atlas ~/runs/campaign/atlas/v1 \
    --analysis ~/runs/campaign/tierA_analysis.json \
    --analysis-tite017 ~/runs/campaign/tierA_analysis_tite017.json
```

## Reading the ratios

- `R_sum = Σ p_EFIT / Σ p_e` over the Thomson channels inside the LCFS.
  - This is `exp(-log_ratio)` of the `thomson_pressure` check.
  - `build_state.py` stores the analysis JSON's value beside it as `criteria_log_ratio`.
- `R_W` is the volume integral of p_EFIT over the volume integral of p_e.
  - It covers the channels' ψ_N span, with p_e taken from the `core_profiles` fit at the matched time.
  - `R_W_full` covers the whole plasma and therefore extrapolates that fit.
- **Magnetics-only rows** are held-out validation: Thomson was not used in the fit.
- **Electron-kinetic rows** are fit consistency: Thomson was fitted, with Ti = Te.
- **`efit_quality` is fit quality only** (criteria version 2, 2026-10-02). Thomson does not gate `good` rows, so `R_sum` is free on both `good` and `admissible` rows.
- **`thomson_consistent`** is the criteria-v2 physical-consistency verdict of the row's own `efit_setting`, reported per magnetics row and never used to select one.
  - It grades `criteria_log_ratio`, the weight scan's g-file comparison (band `1 ≤ p/p_e ≤ 2`). It is not the row's `R_sum` from the matched profiles, which can differ; both are kept.
  - In `state.csv` it is `true`, `false` or empty (no Thomson verdict).
  - The ceiling follows from `p = p_e (1 + f_i T_i/T_e)` with no fast ions, `T_i ≤ T_e` and `f_i = n_i,tot/n_e ≤ 1`.
  - Version 1 used `[1, 3]` and gated `good` rows on it, so atlases built before 2026-10-02 differ.

## Inferred T_i (#1426)

`build_ti.py` runs after `build_state.py` and reads its tables:

```bash
python3 workflow/kinetic_state/build_ti.py --filedb ~/runs/campaign/filedb --atlas ~/runs/campaign/atlas/v1
```

It writes four things:
- `ti_inferred.csv`: per evaluation point, either a Thomson channel or a `core_profiles` grid point;
- `ti_state.csv`: per state key;
- `ti_summary.json`;
- `core_profiles/<shot>.json.gz`: IMAS `core_profiles`, the `pressure_partition_inferred` lineage.

**Method**
- T_i = (p_EFIT − e n_e T_e) / (e Σn_i).
- Σn_i comes from the Z_eff = 2 / C⁶⁺ closure, which gives 5/6 n_e.
- It assumes one common ion temperature.

**What it refuses or flags**
- It is **inferred** and never measured.
- Electron-kinetic states are refused, because their pressure was fitted with Ti = Te assumed.
- Points with p_i ≤ 0, or with p_i below its own σ, are flagged and left empty.
- Grid points the Thomson channels do not bracket are flagged `outside_ts_span`.
- Grid rows are written only when the `core_profiles` grid is the state's own equilibrium slice: the grid is placed by its own flux label (`grid.rho_pol_norm` or `grid.psi`) and its `rho_tor_norm` must agree with the slice's `rho_tor(psi_N)` within `GRID_RHO_TOLERANCE` (0.02). A `sqrt(psi_N)` proxy under `rho_tor_norm`, another equilibrium's coordinate, or a grid with no flux label is refused and `ti_state.csv` carries the reason; the channel rows are unaffected.

**σ(p_EFIT)** is the spread over the shot's other good or admissible magnetics slices within 1 ms, at the same ψ_N. Its floor is 17 % (#874).

## Matched single-state inference archives

The executed notebooks
[`self_consistent_kinetic_inference_vest_39915.ipynb`](../../notebooks/self_consistent_kinetic_inference_vest_39915.ipynb)
and [`self_consistent_kinetic_inference_vest_40326.ipynb`](../../notebooks/self_consistent_kinetic_inference_vest_40326.ipynb)
use compact, offline inputs in [`archive_data`](archive_data). The snapshots
preserve selected equilibrium and core-profile arrays, separately mapped
Thomson points with their 1σ errors, transient C/O charge-state results, and
SHA-256 hashes of the campaign source products. Optional replay of the
39915 charge-state calculation needs the hash-pinned OpenADAS tables locally.

Both notebooks show magnetic-EFIT pressure partition, 2×2 and 4×1
`vaft.plot` panels, and the electron-kinetic-EFIT branch. The 40326 kinetic
EFIT is admissible; the 39915 kinetic EFIT has a negative edge pressure and
is shown as a rejected comparison. `T_i` inferred from independent magnetic
pressure is distinct from the `T_i=T_e` prior used by kinetic EFIT. The
pressure sum `p_e+p_i=p_eq` is a closure identity. Thomson measurements used
as fit input are shown with errors but do not independently validate their
own fit. The notebooks do not write poster assets.

The composite view is the public plot `vaft.omas.plot_kinetic_overview_state`
([#1837](https://github.com/VEST-Tokamak/vaft/issues/1837)), which reads stored
IMAS paths only. [`archive_to_ods.py`](archive_to_ods.py) turns a snapshot into
such an ODS through VAFT's own writers where they exist: the C/O composition goes
through `populate_radial_impurity_profiles` (fed the archived charge states), `T_i`
through `infer_ti_pressure_partition`, and each quantity carries the
`*_fit.parameters` provenance record the plot reads its evidence role from, with
the equilibrium lineage/occurrence on the ion-temperature record and the `Z_eff`
target on the `zeff` record (the contract as amended on #1837). The writer fills
the one point whose charge states are undefined, where the notebooks leave NaN;
the converter checks the written values against the notebook algebra at every
defined point. The snapshots keep the Thomson mapping but not the 2-D flux map, so
the converter places the channels at their archived radii (39915: the 40326 radii)
and builds a synthetic psi map on which they land at their archived `psi_N`. It is
a workflow-side reconstruction, not package code; the notebooks and the tests
import it by putting this directory on `sys.path`.
