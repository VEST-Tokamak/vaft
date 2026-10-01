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
- **`good` rows are gated on the criteria band** `1 ≤ R_sum ≤ 3`, so their magnetics-only `R_sum` lies in that band by construction. Only `admissible` rows show the ungated distribution.
- The band is a workflow criterion, not a physical bound.

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

**σ(p_EFIT)** is the spread over the shot's other good or admissible magnetics slices within 1 ms, at the same ψ_N. Its floor is 17 % (#874).

## Transport readiness (#1428, handed to lane T)

`build_readiness.py` runs after `build_ti.py`:

```bash
python3 workflow/kinetic_state/build_readiness.py --filedb ~/runs/campaign/filedb --atlas ~/runs/campaign/atlas/v1
```

**Outputs**
- `readiness.csv`: one row per state key, `ti_lineage` and target `rho_tor_norm` (0.3, 0.5 and 0.7).
- `readiness_state.csv`: one verdict per state and lineage.

**How a state is judged**
- `ready`, `conditional` or `insufficient` is decided by calling `prepare_gacode_profile` and `prepare_tglf_input`. A refusal from either is quoted.
- Flux-surface geometry is derived from the 2-D flux map on a private copy, because EFIT g-file products carry none.
- Targets are given in `rho_tor_norm` and converted to the TGLF r/a of the converted profile. A `rho_max` cut moves `a`, so the target is never a fixed r/a.

**Overlay: what makes a state `conditional`**
- an assumed Ti/Te (#1414);
- the Z_eff = 2 / C⁶⁺ composition policy;
- an inferred T_i outside the Thomson span.

An inferred T_i that is flagged truncates the grid at the last valid point. Targets beyond that point are `insufficient`.
