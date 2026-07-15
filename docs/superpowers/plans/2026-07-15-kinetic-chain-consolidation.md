# Kinetic chain consolidation into `vaft/code/` (plan)

Date: 2026-07-15
Branch: `feature/kinetic_profile_eq`
Status: draft — awaiting user confirmation of the forks in §6

## Goal

Move the whole VEST kinetic pipeline — **kinetic profile fitting → kinetic-pressure EFIT
→ CHEASE refinement** — out of the `ids_test` scratch scripts and `notebooks/` and into
the `vaft` library (under `vaft/code/`, per user decision), so every stage is an
importable, testable function driven from the repo. Deliver a new end-to-end notebook on
top of the consolidated API. Binaries are resolved via env vars (`$EFIT`, `$CHEASE`) the
way `vaft/code/gpec.py` uses `$GPECHOME`, and every stage degrades gracefully (a
`status="skipped"` result) when its binary/env var is absent.

## Current state (from the 5-agent code map, 2026-07-15)

| stage | library today | production today | gap |
|---|---|---|---|
| profiles | `vaft.process.profile` (mapping/fit/`core_profiles`) + `vaft.formula.fit_profile` — **complete** | `notebooks/build_kinetic_profiles.py::build()` wires them per slice | orchestration trapped in `notebooks/`; **CX σ-weighting fix lost**; Er/wexb physics absent |
| kinetic EFIT | `vaft/code/efit.py` — **magnetic only** (0 pressure-constraint fields) | 6 `ids_test/run_*.py` scripts text-patch a `KPRFIT=1` block into a filedb kfile, run `efit-new2`, sweep `PLASMA` scale | no pressure-point builder, no kfile pressure injector, no scale-sweep wrapper in vaft |
| CHEASE | `vaft/code/chease.py` — pure-python jsk95 re-impl, **unused by production** | `run_chease_fullip_psin.py` shells out to external `eqdsk.py jsk95` (+ `sys.exit()` bug) | adapter missing q95 match, pprime/ffprim edge-zeroing, FFT boundary smooth, `EPSLON=1e-10`, `NIDEAL=11` |

Environment (verified, see `scratchpad/env_facts.md`): EFIT `/home/user1/work/efit-new2/build/efit/efit`;
CHEASE `/home/user1/work/chease_1/chease`; interpreter `/home/user1/miniconda3/envs/vaft/bin/python`
(the docs' `/opt/anaconda3/...` path does **not** exist here). 5/9 target slices converge at
full Ip (48224×3 + 48226@300/301); 48226@299 + 48233×3 fail for eddy/boundary reasons.

## Target layout

1. **`vaft/code/kineticEfit.py`** (NEW — user-named) — the whole kinetic-EFIT orchestration,
   following the Config/Inputs/Result + prepare/run/collect convention of `efit.py`. Keeps the
   magnetic `efit.py` untouched and layers the kinetic-pressure constraint on top of it:
   - `build_kinetic_core_profiles(ods, geq, time_ms, *, te_mode, ne_mode, ti_mode, vtor_mode, ion_index=0) -> ods`
     — orchestrates the existing `vaft.process.profile` chain (TS+CX map→fit→`core_profiles`),
     pure ODS→ODS. (Lifts `build()` steps 2–4.)
   - `kinetic_pressure_points(ods, time_ms, geq=None, *, encoding='raw6', sigma_floor=0.05) -> PressurePoints`
     — `p = e·nₑ·(Tₑ+Tᵢ)` with the SIGPRE error propagation; **both** encodings:
     `'raw6'` (5 TS @ real R + `psi_N=1,p=0` anchor, default), `'raw5'`, `'spline'` (psi-space 129).
   - `inject_pressure_constraint(kfile_text, points, *, encoding) -> str` — KPRFIT=1 / NPRESS /
     RPRESS / PRESSR / SIGPRE / FWTPRE / ZPRESS block before the IN1 `/` (ports build_kfile /
     build_6pt / add_pressure).
   - `scale_plasma(kfile_text, scale) -> str` + `run_kinetic_efit(inputs, config) -> KineticEFITResult`
     — scale-sweep (largest converging Ip), reusing `efit.run_efit`/`collect_efit_outputs`.
   - `KineticEFITConfig/Inputs/Result` (frozen Config; `$EFIT` env resolution; skipped-result on
     missing binary) and `run_kinetic_chain(...)` end-to-end = profiles → kinetic EFIT → CHEASE refine.
2. **`vaft/code/efit.py`** (MINIMAL) — leave magnetic reconstruction as-is; only add `$EFIT` env-var
   executable resolution (shared by `kineticEfit.py`), replacing the bare `ValueError` with a
   skipped result when unresolved. No pressure code here.
3. **`vaft/code/chease.py`** (EXTEND) — close the jsk95 parity gaps (q95 constraint,
   pprime/ffprim edge-zeroing, FFT boundary smoothing, `EPSLON=1e-10`, `NIDEAL=11`); keep the
   adapter's better features (COCOS normalization, limiter restore, comparison metrics); add
   `$CHEASE` env resolution for symmetry with EFIT.
4. **`vaft/process/profile.py`** (FIX) — restore the lost CX σ-weighting: `_leaf_values_and_errors`
   (read `<leaf>.data_error_upper` explicitly), `_sanitize_std` (median-replace 0/NaN/floor σ),
   real-σ weighting in `profile_fitting_charge_exchange`, and `ion.0.temperature_fit.measured_error_upper`.
5. **`vaft/code/__init__.py`** — register `kineticEfit` (3 edit points: submodule set, `__all__`,
   `_EXPORT_MAP`).
6. **`notebooks/_build_kinetic_profiles.py`** → thin CLI over `vaft.code.kineticEfit` (+ `process.profile`).
7. **`notebooks/kinetic_efit_end_to_end.ipynb`** (NEW) — the deliverable notebook: one slice
   gfile → kinetic profiles → kinetic-pressure EFIT → CHEASE refine → compare, with graceful
   skips when a binary is absent.

## Verification strategy (binaries optional)

- **Unit / no-binary**: synthetic ODS + gfile → assert pressure points (`p=e·nₑ·(Tₑ+Tᵢ)`, SIGPRE
  formula), kfile text contains a well-formed KPRFIT block, `status="skipped"` when `$EFIT`/`$CHEASE`
  unset. Restore-fix test: CX fit is genuinely σ-weighted (reproduce the summary's synthetic case).
- **End-to-end (my hands, not fanned out — the scripts share a cwd and race)**: drive one known-good
  slice **48224@300** with the real binaries and check it reproduces the shipped share-bundle
  (|Ip|≈137.6 kA CHEASE, q0≈1.86, p0≈132 Pa).

## §6 Decisions (confirmed by user 2026-07-15)

1. **CHEASE** → extend the pure-python vaft adapter to jsk95 parity (q95 match, pprime/ffprim
   edge-zeroing, FFT boundary smoothing, `EPSLON=1e-10`, `NIDEAL=11`). Must numerically validate
   q95 + output signs against a known-good slice (48224@300) before it replaces the external path.
2. **Pressure encoding** → implement **BOTH** raw-6pt (5 TS @ real R + `psi_N=1,p=0` anchor) **and**
   psi-space spline-129, selectable via an `encoding=` argument. raw-6pt is the default (matches the
   shipped share bundle).
3. **Restore the lost CX σ-weighting fix** → yes.
4. **Er + Hahm-Burrell wexb** → defer to a follow-up.

## Implementation status (2026-07-15) — DONE + validated

All items implemented and validated end-to-end with the real binaries on slice **48224@300**:

- **profiles** — `build_kinetic_core_profiles` rebuilds from raw TS/CX matching the shipped
  bundle: ne0 9.92e18 (vs 1.036e19), Te0 64.8 (vs 65.4), Ti0 14.3 (vs 13.5 — the +0.8 eV
  shift is the restored CX σ-weighting, consistent with the summary). CX fix restored.
- **kinetic EFIT** — `run_kinetic_efit` with the real `efit-new2` binary converges at full Ip
  (scale=1.0), |Ip|=143.2 kA, q0=1.867. raw6 + spline encodings both build.
- **CHEASE** — `refine_equilibrium` (pure-python jsk95 parity) with the real CHEASE binary
  reproduces the external `eqdsk.py jsk95` reference: **|Ip|=137.6 kA, q0=1.855, 513×513**
  (ref 137.6 / 1.856 / 513); q0 spike 8.07→1.855 removed.
- **one-call chain** — `run_kinetic_chain` runs profiles→EFIT→CHEASE in a single call:
  EFIT 143.2 kA/q0 1.867 → CHEASE 137.7 kA/q0 1.870/513.
- **tests** — 26 new no-binary unit tests (test_cx_weighting / test_kineticEfit /
  test_chease_parity) pass; full suite 163 passed / 2 skipped, no regressions.

Two correctness bugs found + fixed during binary validation (unit tests could not catch them):
1. **CHEASE edge-zeroing sign** — `_edge_zero_profiles` tested a literal `pprime > 0`, but the
   adapter feeds `_write_expeq` the COCOS-02-normalized gEQDSK whose pprime sign is flipped, so
   it zeroed the ENTIRE pp/ff profile → CHEASE NaN → SIGSEGV. Fixed to zero reversals relative
   to the bulk (median) sign — robust to the COCOS convention.
2. **EFIT grid-size argument** — EFIT needs `efit <n>` (grid dim); `run_kinetic_efit` now passes
   `KineticEFITConfig.grid_size` (default 129) as the arg.
Plus: `KineticEFITConfig.base_kfile` added so the kinetic EFIT can patch a filedb magnetic kfile
(the profile-only ODS has no magnetics for `efit.generate_kfile`).

Binaries via env vars: `$EFIT=/home/user1/work/efit-new2/build/efit/efit`,
`$CHEASE=/home/user1/work/chease_1/chease`; interpreter `/home/user1/miniconda3/envs/vaft/bin/python`.

Known remaining gap: `_parse_chi2` does not match this EFIT build's log format (chi2 reports None;
convergence is detected from the produced g-file, which is correct). The notebook data
(gfiles/MAT) is not packaged under `vaft/data`; it lives in the ids_test tree.

## Out of scope

- Remote HSDS/MySQL DB writes.
- The share-bundle packaging scripts (`gen_share_*`) — separate follow-up (could later become a
  `vaft` multi-slice-merge helper).
- The 4 non-converging slices (48226@299, 48233×3) — a magnetic/boundary problem, not this refactor.
