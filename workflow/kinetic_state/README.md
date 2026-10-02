# Kinetic state (lane K)

These scripts build the conference atlas's kinetic-state tables from the #1331 Tier A EFIT campaign.

The state key contract is defined in #1454: rows are keyed by `(shot, time_efit_s, efit_lineage)` and every row carries `efit_quality`. The slice-level calculations live in `vaft.validation.kinetic_state`. The Thomson comparison is the sampler that `criteria.py`'s Thomson verdict grades, `vaft.validation.equilibrium.thomson_pressure_samples`.

| script | reads | writes |
|---|---|---|
| `build_state.py` | the campaign FileDB and `tierA_analysis.json` (slice grades) | `state.csv`, `profiles.csv`, `schema/*.schema.json`, `MANIFEST.json` |
| `summarize.py` | `state.csv` and the analysis JSONs | `summary.json`, `figures/pressure_consistency.{png,pdf}` |

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
- **`thomson_consistent`** is the physical-consistency check `1 ≤ R_sum ≤ 2`, reported per magnetics row and never used to select one.
  - The ceiling follows from `p = p_e (1 + f_i T_i/T_e)` with no fast ions, `T_i ≤ T_e` and `f_i = n_i,tot/n_e ≤ 1`.
  - Version 1 used `[1, 3]` and gated `good` rows on it, so atlases built before 2026-10-02 differ.
