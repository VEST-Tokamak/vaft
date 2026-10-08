# Transport atlas (lane T, #1453)

This directory holds the drivers for the Tier A transport atlas. TGLF, NEO and classical model fluxes are keyed by (shot, time_efit_s, efit_lineage, r/a) and are computed on the #1331 slices labelled good or admissible only.

| script | issue | what it does |
|---|---|---|
| `run_tglf.py` | #1428 | Enumerates the states (good/admissible slices that carry a core_profiles slice within 0.5 ms), resolves each through `vaft.process.transport_state`, and runs native TGLF at r/a = 0.30, 0.40, …, 0.80 through the existing runner and execution backend. Projects the solved surfaces with `core_transport_from_tglf`. Writes `states.jsonl`, `run_manifest.json`, and one `state.json` plus the native directories per state. Resumable: a surface already solved under the same run identity is not run again. `--sat-rule` and `--field-model` (`es`, `em-bper`, `em-bper-bpar`) are required and have no default (#1482). Each configuration writes its own `tglf-sat<n>-<field>/` tree. Each ready surface also stores its classical Braginskii baseline (#1435), which needs no solver. |
| `run_neo.py` | #1431 | The same states through the same resolver, so NEO and TGLF share `state_identity`. One NEO profile run per state, with `N_RADIAL` linearly spaced over the requested r/a; the default reproduces the TGLF surfaces, and an unevenly spaced request is refused. Projects through `core_transport_from_neo`. |
| `build_atlas.py` | #1427 | Builds the atlas from one `tglf-sat*/` tree and the NEO tree, without rerunning any solver. Writes `atlas.csv` (one row per (shot, time_efit_s, efit_lineage, r_over_a)), `radial_summary.csv`, `discharge_summary.csv` and `schema.json`. The schema gives each column a unit, an epistemic category and a mathtext symbol. NEO is joined by `state_identity`, and the neoclassical/turbulent/classical partition comes from `transport_partition`. An atlas built from more than one TGLF configuration is refused. |
| `build_sensitivity.py` | #1482 | Reads `tglf-sat<n>-<field>/` trees run on the same states and writes `sensitivity.csv` (one row per state × surface × configuration, with each configuration's full settings), `sensitivity_pairs.csv` (per surface: the max \|Δ\| across SAT rules at a fixed field model, and Δ(EM-BPER, ES) at each SAT rule) and `schema.json`. Δ(A, B) = (A − B)/max(\|A\|, \|B\|, 10⁻³ gB). No configuration is preferred, and the product is separate from the atlas. |
| `plot_atlas.py` | #1427 | A thin CLI over the table renderers `vaft.plot.transport_atlas.transport_atlas_scatter` and `transport_atlas_mode_branch`. It writes the drive-space, response-space and mode-branch maps, plus an ion-drive map that is drawn only when Ti is not a fixed ratio of Te. |

The shared resolver is `vaft.process.transport_state`:
- **State key:** lane K's State key contract v1 (#1454), `(shot, time_efit_s, efit_lineage)`, with `time_efit_s` rounded to 1e-4 s, `efit_lineage ∈ {magnetics, electron_kinetic}`, and an `efit_quality ∈ {good, admissible}` column. Unreconstructible slices are refused, not run.
- **Time pairing:** core_profiles is matched to the equilibrium slice by time, within an explicit tolerance, never by index.
- **Ion temperature:** measured, then a #1426 pressure-partition result when one is passed, then the `vest.yaml` ratio (#1414: Ti = Te ± 0.5, `assumed`). Otherwise the state is insufficient. Composition goes through `prepare_gacode_profile(impurity="C", z_eff=2)`.
  - **The atlas of record uses Ti = Te only** (decision 2026-10-02). The #1426 pressure-partition Ti gives a median Ti/Te ≈ 3.45 on the Tier A surfaces, which is not credible for ohmic VEST.
  - The consuming code is kept but off. A core_profiles product whose ion temperature is labelled `origin=inferred` is refused (`inferred_ti_not_enabled`) unless the drivers get `--use-inferred-ti`, alongside `--core-profiles-dir` (lane K's per-shot products). Such a temperature is never taken as measured.
  - When enabled, a surface runs only if its solver input does not depend on the gap fill (`ti_not_inferred_here` otherwise).
- **Geometry:** EFIT products carry no `r_inboard`/`r_outboard`, so these are derived from the 2-D flux map. Some products also lack shape profiles (#1458); for those, elongation and triangularity come from closed contours inside the boundary outline. `provenance.shape.kind` says which applies.

Runs go on tdst (`lowpri-short`, GACODE `TDST_GNU`). The branch worktree sits on `PYTHONPATH` in front of the non-editable `vaft` env, and the wrapper asserts `vaft.__file__`. The inputs are a read-only subset of the campaign FileDB copied from vestserver. Results are copied back to `vestserver:~/runs/campaign/atlas/transport/`, never into the production FileDB. Pass `--mem-mb`: without `--mem`, tdst's default reserves the whole node for a one-core job.

## Linear check against CGYRO (Lane Y, #1484)

Lane Y ran linear CGYRO on the five #1482 sensitivity states at r/a 0.6/0.7/0.8, k_yρ_s 0.1–6, ES and EM. Both codes read the same resolved state (`state_identity`) and the same local input; TGLF was rerun single-k_y on it. The product is `vestserver:~/runs/campaign/atlas/gyrokinetic_linear/` (`linear.csv`, `linear_summary.csv`, `schema.json`). Only `cgyro_status == "converged"` is a growth rate; `max_time` means marginal and `decayed` means stable.

**What it measures: the accuracy of TGLF's linear growth rate. It does not rank SAT rules.**
- The SAT rules do **not** share one linear model. TGLF's `USE_PRESETS` (`tglf_startup.f90`) gives SAT0/1 `XNU_MODEL=2`, while SAT2/3 get `XNU_MODEL=3`, `WDIA_TRAPPED=1.0` and `UNITS=CGYRO`, so their eigenvalues differ (Lane Y's correction, #1484, 2026-10-03).
- The pointwise numbers below compare CGYRO with the **SAT0/1 linear model**.
- Ranking the rules needs nonlinear CGYRO fluxes. That is Lane Y's next step, on 39916 and 39915 at r/a 0.8, where the SAT spread is largest.

**Result (2026-10-03)**

| quantity | value |
|---|---|
| CGYRO points (360) | 82 converged, 258 max_time, 14 decayed, 6 failed |
| both codes unstable (converged CGYRO) | 77 points |
| TGLF γ / CGYRO γ (SAT0/1 linear model) | median 1.9, IQR 1.5–3.0 |
| median ratio by field model | ES 1.9 (38 points), EM 1.9 (39 points) |
| same branch (sign of ω) | 75/77 |
| CGYRO damped, TGLF unstable | 5 points, all 40330 |
| strongly driven surfaces (39915/39916, r/a ≥ 0.7) | same branch, γ_max ratio 1.4–2.4 (SAT0/1 model); TGLF peaks at k_yρ_s 1.0, CGYRO at 0.6–0.8 |
| γ_max ratio, SAT2/3 linear model (5 driven surfaces, ES) | 1.2–1.8, against 1.4–2.4 for SAT0/1 |
| weakly driven states (40330, all of 42962) | no converged CGYRO mode on most surfaces, so the comparison is inconclusive |

Lane Y's summary (#1453/#1484) gives 82 points, a median of 1.8 and 78/82 branch agreement. That count includes the 5 points where CGYRO is damped; the ratio here uses only the points where both codes are unstable.

Consequences for reading the atlas:
- **Linear drive.** Both TGLF linear models over-predict linear growth on VEST; SAT2/3's is the closer. SAT2/3's larger Q therefore comes from the saturation rule, not from a more inflated linear drive.
- **Absolute fluxes.** Q/Q_GB is likely high on the driven surfaces. The SAT rules stay unranked: Lane Y's first nonlinear run (39915 r/a 0.7) failed locality QA and is not a benchmark (#1484, 2026-10-05).
- **Robust quantities.** The mode direction (`omega_dom_sign`, the mode-branch map) and the radial and inter-shot ordering are what the check supports.
- **Weak surfaces.** Near-marginal surfaces stay near-marginal; their f_neo partition is not affected.
- **ES low-k_y caveat.** `converged` (`cgyro_qualified`) records CGYRO's converged exit only and screens no mode: electrostatic runs at k_yρ_s ≤ 0.2 can carry the high-frequency electron (ω_H) branch (|ω| ~ 200 c_s/a, worse with finer θ resolution, absent once A_∥ is kept), so an ES row that converges there is not a drift-wave growth rate and would set γ_max on its surface. In the product above those points ended `max_time`; trust the EM (`em-aperp`) rows at low k_y (#1484, #1656).
- **Conventions.** In `linear_summary.csv`, `gamma_max_ratio` is CGYRO/TGLF, the inverse of the ratio quoted here. The EM field model is named `em-aperp`, which is the atlas's `em-bper`.
