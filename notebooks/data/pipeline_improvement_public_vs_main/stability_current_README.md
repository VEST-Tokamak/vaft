# Tier A stability atlas, v2 (Lane N, #1429)

Built 2026-10-03 by `workflow/stability_atlas/build_atlas.py` (`ATLAS_VERSION = 2`, #1509) from `~/runs/campaign/lane-n/batch/`. The column dictionary, statuses, layers and provenance are in `schema.json`. v1 (n = 1, 2) is kept in `../stability_v1/`.

## Files
| file | rows | key |
|---|---|---|
| `atlas_n.csv` | 900 | (shot, time_efit_s, efit_lineage, n_tor), n = 1..6 |
| `atlas_surfaces.csv` | 6970 | key + (solver ∈ {rdcon, stride}, m), n = 1, 2 |
| `schema.json` | | column dictionary, layers, status values, provenance |

## Scope
- **Slices:** 150 #1331 Tier A slices labelled good or admissible (`statistical_891`). No unreconstructible slice is present.
  - magnetics-only: 15 good + 118 admissible.
  - electron-kinetic: 1 good + 16 admissible. Each takes the label of the magnetics-only slice at exactly its own time.
- **Coverage:** DCON (full and truncated edge, mpsi 256/512) for n = 1..6. RDCON and STRIDE for n = 1, 2 only; their columns are NOT_APPLICABLE and empty for n ≥ 3.
- **Key:** provisional `(shot, time_efit_s, efit_lineage)` + `n_tor`, until lane K publishes its State key contract.
- **Equilibrium:** the EFIT g-file, CHEASE-refined with pipeline 1's `run_chease_refinement.py`.
- **GPEC:** e68d7ac2 stock, except 8 RDCON rows that used a format-only `gal.f` patch where stock crashed (`rdcon_gpec_build`).

## Layers (maintainer decision, #1429 comment of 2026-10-02)
QA annotates how far a result can be trusted. It never redefines a physical quantity or removes a solver result.

1. **Raw.** W_t per n, edge treatment and mpsi; Δ′ per surface at mpsi 256 and 512; `{s}_delta_prime_max_raw` over every surface (with `_m/_psi_n/_q/_position`); `{s}_delta_prime_first_surface`.
2. **QA annotation.**
   - `dcon_{full,trunc}_sign_class` ∈ {robust_stable, robust_unstable, numerically_unresolved, not_checked, unavailable}.
   - Per surface: `two_resolution_consistent` (same sign and ≤ 20 % between mpsi 256 and 512; a heuristic, the median change of #141 mid-radius surfaces), `sign_agreement`, `ordinal_position`, `shear_psi` (d ln q / d ln ψ_N), `psi_n_from_qmin`, `q_minus_qmin`, `psi_n_to_edge`.
   - QA summaries: `*_qa` (consistent surfaces at any position) and `*_qa_interior`.
3. **Universal physical criteria.**
   - Ideal: `ideal_unstable_{full,trunc}_edge` = sign of W_t.
   - Classical tearing drive: `{s}_classical_tearing_drive` = Δ′ > 0 at **any** rational surface, plus its `_qa` form.
   - Local: D_I > 0 (Mercier), D_R > 0 (resistive interchange), C_A < 0 (ballooning).
   - `derived_ideal_unstable_any_n` lists the (n, edge) pairs per slice.
4. **Interpretation** is not produced. Δ′ is an outer-region quantity; a tearing verdict needs an inner layer (RMATCH, #797; #939).

## Read before plotting
- **No magnitude band (#142).** A negative W_t whose sign agrees at mpsi 256 and 512 is ideal-unstable however small. `MARGINAL` is reserved and not produced.
- **Edge treatments.** Full-edge and truncated-edge energies are separate columns and never combined. Full-edge W_t is `psihigh`-sensitive (#141, #792).
- **Status columns.** `{s}_status` keeps its **v1** meaning (RESOLVED = a consistent *interior* surface exists). `{s}_qa_status` is the v2 form (any position). Neither is a tearing verdict.
- **Deprecated for one version** (removed in v3): `{s}_delta_prime_max` (= `_qa_interior`), `{s}_{m,psi_n,q}_at_delta_prime_max`, `{s}_n_resolved_surfaces`, `{s}_n_summary_surfaces`, `{s}_n_positive_delta_prime_summary`, `resolved`, `in_summary`.
- **Never combine** RDCON with STRIDE, DCON with RDCON, or values across n (except the labelled `derived_*` column).

## Headline
**Full-edge DCON sign class (robust_stable / robust_unstable / numerically_unresolved):**

| n | robust_stable | robust_unstable | numerically_unresolved |
|---|---|---|---|
| 1 | 145 | 1 | 4 |
| 2 | 147 | 1 | 2 |
| 3 | 145 | 3 | 2 |
| 4 | 144 | 2 | 4 |
| 5 | 141 | 7 | 2 |
| 6 | 135 | 11 | 4 |

Median full-edge W_t falls with n: 5.17, 2.68, 1.83, 1.59, 1.15, 0.92.

- **Truncated-edge DCON:** robust_stable at every n except n = 6, where 39906@319, 39915@315 and 40324@324 are robust_unstable and one row is numerically_unresolved.
- **Ideal instability:** 16 of 150 slices are ideal-unstable at some (n, edge). Most are high-n, full-edge cases in the 399xx and 403xx series.
- **Classical tearing drive** (Δ′ > 0 at any surface; raw / two-resolution-consistent):

  | solver | n=1 | n=2 |
  |---|---|---|
  | RDCON | 96 / 6 | 92 / 5 |
  | STRIDE | 103 / 5 | 143 / 37 |

- **Local criteria:** C_A < 0 at 12 slices; D_I > 0 at one slice (39915@324).
- **Large Δ′:** |Δ′| > 1e4 occurs mostly on the first surface, at low shear near q_min (median |shear_psi| 0.14 against 3.5 overall). Diagnosed, not discarded.

## Known gaps
- 39915@0.322 s, n=2: RDCON (`ode_unorm`) and STRIDE ("Integration direction not reversed") both fail → SOLVER_FAILURE.
- No equilibrium-descriptor columns (li_3, β_N). Join them from lane K / EFIT descriptors after #1477.
- Solver timeouts at n = 5, 6 were rerun with a 7200 s limit, and all completed.
