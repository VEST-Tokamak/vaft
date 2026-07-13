# Kinetic Profile Chain Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix the review-confirmed defects in the TS fitting chain, then promote the TS+IDS/CES → core_profiles kinetic-profile logic into the VAFT library with a single CLI script, deleting the EFIT comparison/export scripts.

**Architecture:** Diagnostic MAT files → `machine_mapping` builders (thomson_scattering, charge_exchange) → `process/profile.py` equilibrium mapping (psi_N) + fitting (`formula/utils.py fit_profile`) → extended `core_profiles()` writing ne/Te/Ti/Vtor + `pressure_thermal` on the equilibrium grid → OMAS JSON + IMAS export. Spec: `docs/superpowers/specs/2026-07-13-kinetic-profile-chain-design.md`.

**Tech Stack:** Python, numpy/scipy, OMAS, omfit-classes (gfile), uncertainties. Interpreter: `/opt/anaconda3/envs/vaft/bin/python` (conda env `vaft` — PATH `python3` and repo `.venv` lack the stack).

## Global Constraints

- All tests run with `/opt/anaconda3/envs/vaft/bin/python -m pytest`.
- Physics: p_thermal = 1.602176634e-19 · n_e · (T_e + T_i) [Pa], T in eV; n_i ≈ n_e; Ti(H+) = Ti(C3+); all-thermal electrons.
- Mapping coordinate is psi_N (`fluxSurfaces['levels']`); never write psi_N into a `rho_tor_norm` leaf.
- Never call the remote HSDS/MySQL database from tests or the new script.
- Target shots 48224/48226/48233 × 299/300/301 ms; gfiles under `vaft/data/efit_scaled_f/<shot>/`.
- Tests that need omas must `pytest.importorskip("omas")`.

---

### Task 1: Thomson builder — NeTe_{shot}.mat schema, 7 channels, invalid-sigma masking

Fixes findings: builder filename patterns (thomson_scattering.py:45), shot-suffixed 7-channel schema + Rposition (…:199), complex-sigma→0 (…:35), v9 zero/zero unmasked (…:112).

**Files:**
- Modify: `vaft/machine_mapping/thomson_scattering.py`
- Test: `test/test_thomson_mat_suffixed.py` (new)

**Interfaces:**
- Produces: `thomson_scattering(ods, shot, data_root=None, mat_file=None)` now also ingests `NeTe_{shot}.mat` with keys `tsTime_{shot}, Te_{shot}, Ne_{shot}, sigmaTe_{shot}, sigmaNe_{shot}, Rposition_{shot}`; builds one channel per Rposition entry (mm→m), overwriting static positions; invalid sigma (non-finite, ≤0, or complex with nonzero imag) becomes `np.nan` in `*.data_error_upper`.

- [ ] **Step 1: Write failing test** — synthetic suffixed MAT via `scipy.io.savemat` in `tmp_path`: 7 channels × 3 times, one sigmaTe entry `0+5j`, one sigmaNe entry `0.0`. Assert: loads without KeyError; `len(ods['thomson_scattering.channel']) == 7`; `channel.5.position.r == 0.475`, `channel.6.position.r == 0.650`; time in seconds; `data_error_upper` NaN at the two poisoned entries; also assert `thomson_scattering(ods, shot, data_root=tmp_path)` (no mat_file) finds `NeTe_{shot}.mat`.
- [ ] **Step 2: Run test, verify FAIL** (KeyError 'Unsupported Thomson MAT schema' / FileNotFoundError).
- [ ] **Step 3: Implement** — add `NeTe_{shot}.mat` to `_candidate_thomson_paths` (both roots); add `_set_dynamic_from_suffixed(mat_data, ods, shotnumber)`: reads suffixed keys, `Rposition_{shot}`/1000 → per-channel `position.r` (z=0, name `Polychrometer {i+1}`), `_sanitize_sigma(raw)` helper returning float array with NaN where invalid (checks `np.iscomplexobj` + `imag != 0`, `real <= 0`, non-finite) applied to both sigma arrays in ALL three loaders (suffixed, v9, simple); dispatcher tries suffixed keys before raising.
- [ ] **Step 4: Run test, verify PASS**; also run `test/test_thomson_mat_resolve.py` if importable (regression).
- [ ] **Step 5: Commit** `fix(machine_mapping): ingest NeTe_{shot}.mat 7-channel schema, mask invalid TS sigmas`

### Task 2: fit_profile robustness (scale-aware p0, masking, maxfev, linear sort, GP y_std=None)

Fixes findings: utils.py:452 (p0=0.1 silent), :455 (zero sigma silent, maxfev=800), :457 (NaN crash/poison), :372 (unsorted linear), :354 (GP y_std None), :452 (order docstring), :455 (list input).

**Files:**
- Modify: `vaft/formula/utils.py` (`fit_profile`)
- Test: `test/test_fit_profile_robustness.py` (new)

**Interfaces:**
- Produces: `fit_profile(x, y, y_std, x_eval, ...)` unchanged signature; accepts lists; internally masks non-finite x/y and non-finite/≤0 y_std points; raises `ValueError` if <2 valid points; p0 scaled to `max|y|` (log-scale for exponential modes); `maxfev=20000`; curve_fit `RuntimeError` re-raised with mode/scale context; warns via `warnings.warn` if returned coeffs == p0.

- [ ] **Step 1: Write failing tests** — (a) `y = 5e18*(1-x**2)`, polynomial mode, no y_std: assert `fit(0) > 1e18` (not 0.1); (b) one `y_std=0` point: assert fit unaffected vs all-valid fit within 20%; (c) one NaN y point: no exception, finite output; (d) `fitting_function='linear'` with unsorted x: y_eval matches sorted interp; (e) GP mode with `y_std=None`: no TypeError; (f) plain-list inputs work.
- [ ] **Step 2: Run, verify FAIL** (a: fit(0)≈0.1; b: hijack/p0-return; c: crash; d: garbage; e: TypeError; f: AttributeError).
- [ ] **Step 3: Implement** — at fit_profile top: coerce to float ndarrays, flatten x; build validity mask; slice all inputs; `ValueError` on <2 points. Generic branch: `y_scale=max(|y|) or 1.0`; exponential-family (`{'exponential','free_exponential','exp_free','exponential_unconstrained','sqrt_exponential','sqrt_exp'}`) → `p0=zeros(order); p0[0]=log(y_scale+tiny)`; else `p0=full(order,0.1); p0[0]=y_scale`. Add `maxfev=20000` to both generic curve_fit calls; wrap in try/except RuntimeError with informative re-raise; `warnings.warn` when `np.allclose(coeffs,p0)`. linear branch: argsort x. GP branch: `alpha = y_std_gp**2 if y_std_gp is not None else 1e-10`. Docstring: order = number of coefficients (polynomial degree order−1).
- [ ] **Step 4: Run, verify PASS**; regression `pytest test/test_confinement_scaling.py test/test_s_alpha.py` (should be untouched).
- [ ] **Step 5: Commit** `fix(formula): make fit_profile robust to scale, NaN, zero sigma; sort linear; GP y_std=None`

### Task 3: profile_fitting_* — nearest-time tolerance, channel filtering, sigma floor after normalization

Fixes findings: profile.py:129 (zero-sigma hijack, floor-before-normalization), :116 (exact float time match), :185 (unclipped T_e_rho).

**Files:**
- Modify: `vaft/process/profile.py` (`profile_fitting_thomson_scattering`, `profile_fitting_charge_exchange`)
- Test: `test/test_profile_fitting_filtering.py` (new, `pytest.importorskip("omas")`)

**Interfaces:**
- Produces: both fitters gain `time_tolerance_ms=1.0`; nearest-time lookup raising `ValueError` beyond tolerance; channels dropped (with printed count) when value/sigma/rho non-finite or sigma ≤ 0; sigma floor = `1e-3 * max(|y_valid|)` applied AFTER ne normalization; returned `T_e_rho`/`Ti_rho` arrays clipped ≥0 like their functions.

- [ ] **Step 1: Write failing tests** — synthetic ODS with 7 TS channels, one channel `data_error_upper=0` at a 0.1 eV point, one channel NaN rho (outside-LCFS marker): fit at exact time must not collapse (fit(0) within 30% of clean-fit axis value); `time_ms` off by 0.4 ms still resolves; off by 5 ms raises ValueError.
- [ ] **Step 2: Run, verify FAIL** (collapse to ~0; IndexError on off-time).
- [ ] **Step 3: Implement** in both fitters: nearest-index + tolerance; validity mask over channels (finite value, finite sigma>0, finite rho); relative sigma floor post-normalization; drop + print `[INFO] dropped N invalid TS channels at ...`; clip returned eval arrays.
- [ ] **Step 4: Run, verify PASS.**
- [ ] **Step 5: Commit** `fix(process): filter invalid channels and use tolerant time matching in profile fitting`

### Task 4: equilibrium mapping — outside-LCFS channels → NaN

Fixes findings: profile.py:54 and :247 (outside-LCFS points pinned to psi_N≈1).

**Files:**
- Modify: `vaft/process/profile.py` (`equilibrium_mapping_thomson_scattering`, `equilibrium_mapping_charge_exchange`)
- Test: extend `test/test_profile_fitting_filtering.py`

**Interfaces:**
- Produces: both mappers return `np.nan` for measurement points outside the outermost traced flux surface (matplotlib.path containment on the last surface's R,Z contour); inside points unchanged (clipped [0,1]). Task 3's filtering consumes the NaN.

- [ ] **Step 1: Write failing test** — fake `geq` dict with two circular surfaces (levels 0.5, 1.0); point at R outside the outer circle must map to NaN; inside point maps to nearest level.
- [ ] **Step 2: Run, verify FAIL** (outside point returns 1.0/level).
- [ ] **Step 3: Implement** — shared `_outside_boundary(r, z, geq)` helper using `matplotlib.path.Path(np.column_stack([R_b, Z_b])).contains_point`; boundary = surface with max level; on containment failure return NaN before the nearest-surface loop.
- [ ] **Step 4: Run, verify PASS.**
- [ ] **Step 5: Commit** `fix(process): map outside-LCFS diagnostic channels to NaN instead of psi_N~1`

### Task 5: core_profiles() extension — real Ti, Vtor, kinetic pressure, honest grid

Fixes finding profile.py:440 (psi_N mislabeled rho_tor_norm) + implements the spec's core deliverable.

**Files:**
- Modify: `vaft/process/profile.py` (`core_profiles`)
- Test: `test/test_core_profiles_kinetic.py` (new, `pytest.importorskip("omas")`)

**Interfaces:**
- Produces: `core_profiles(ods, time_ms, mapped_rho_position, n_e_function, T_e_function, tol_ms=0.1, T_i_function=None, V_tor_function=None, ti_mapped_rho_position=None, rho_points=100)`.
  - With `equilibrium` in ods at matching time: grid = equilibrium `rho_tor_norm` + `psi`; fits evaluated at `psi_N(grid)`.
  - Without equilibrium: uniform psi_N grid written as `grid.rho_pol_norm = sqrt(psi_N)` (NOT rho_tor_norm).
  - `T_i_function` given → `ion.0.temperature` = Ti fit, `pressure_thermal = e·ne·(Te+Ti)`, Ti fit-metadata block from `charge_exchange` measured points; else legacy Ti=Te.
  - `V_tor_function` given → `ion.0.velocity.toroidal`.
  - Always: H+ metadata (`z_ion=1`, `element.0.{a=1,z_n=1,atoms_n=1}`), `core_profiles.ids_properties.homogeneous_time=1`, `core_profiles.time` kept consistent.

- [ ] **Step 1: Write failing test** — ODS with a 1-slice synthetic equilibrium (`rho_tor_norm`, monotone `psi`) + minimal TS/CES trees; call with Ti/Vtor functions; assert grid.rho_tor_norm equals the equilibrium array, `pressure_thermal == e*ne*(te+ti)` elementwise, velocity.toroidal present, H+ element block present, duplicate-time replacement still works, and no `grid.rho_tor_norm` written in the no-equilibrium fallback (rho_pol_norm instead).
- [ ] **Step 2: Run, verify FAIL** (TypeError: unexpected keyword).
- [ ] **Step 3: Implement** per interface (equilibrium slice matched by `|eq.time − time_ms/1e3| ≤ tol_ms/1e3`, psi_N from eq psi; fallback uniform).
- [ ] **Step 4: Run, verify PASS.**
- [ ] **Step 5: Commit** `feat(process): core_profiles with real Ti/Vtor, kinetic pressure, equilibrium grid`

### Task 6: charge_exchange read_doppler_single time units

Fixes finding charge_exchange.py:76 (time stored in ms).

**Files:**
- Modify: `vaft/machine_mapping/charge_exchange.py:76`
- Test: none practical (needs xlrd fixture); covered by code inspection — single-line change

- [ ] **Step 1: Change** `np.array(df["Time [ms]"])` → `np.array(df["Time [ms]"]) / 1e3` with a comment matching the IDS/CES loaders.
- [ ] **Step 2: Commit** `fix(machine_mapping): read_doppler_single stores charge_exchange.time in seconds`

### Task 7: build_kinetic_profiles.py

**Files:**
- Create: `notebooks/build_kinetic_profiles.py`
- Test: end-to-end run (Task 9)

**Interfaces:**
- Consumes: builders + fitters + extended `core_profiles()` from Tasks 1–5.
- Produces: CLI `--shots 48224 48226 48233 --times 299 300 301 --source auto|ids|ces --outdir notebooks/kinetic_profiles --te-mode polynomial --ne-mode free_exponential --ti-mode polynomial --vtor-mode polynomial`. Per slice: gfile → `eq.to_omas()`; `machine_mapping.thomson_scattering(ods, shot)` (native NeTe_{shot}.mat — no temp-MAT hack); `machine_mapping.charge_exchange(ods, shot, options=source)`; mapping; fitting; `core_profiles(...)` with Ti/Vtor; save `ods_<shot>_<time>ms.json` (+ IMAS export via `vaft.imas` if importable, else warn); `summary.csv` (shot, time, ne0, Te0, Ti0, Vtor0, p_th0, n_grid). No pre-emptive output-dir deletion (finding build_omas:126): write into fresh files, keep existing.

- [ ] **Step 1: Write the script** (structure mirrors deleted build_omas_for_kinetic_efit.py `build()`/`main()`, minus bundle/tar/README, plus `--source auto` = `ids` if `IDS_{shot}.mat` exists else `ces`).
- [ ] **Step 2: Smoke-run one slice** `/opt/anaconda3/envs/vaft/bin/python notebooks/build_kinetic_profiles.py --shots 48224 --times 300` → JSON exists, summary row printed.
- [ ] **Step 3: Commit** `feat(notebooks): add build_kinetic_profiles.py (TS+IDS/CES kinetic profiles to ODS/IMAS)`

### Task 8: Delete EFIT scripts

**Files:**
- Delete: `notebooks/ts_ids_efit_pressure.py`, `notebooks/export_kinetic_pressure_for_efit.py`, `notebooks/generate_scaled_efit_ids_plots.py`, `notebooks/build_omas_for_kinetic_efit.py` (all untracked — plain `rm`)

- [ ] **Step 1: rm the four files**, verify `git status` shows no tracked deletions.
- [ ] **Step 2: No commit needed** (untracked); note in final summary.

### Task 9: End-to-end verification

- [ ] **Step 1:** `/opt/anaconda3/envs/vaft/bin/python notebooks/build_kinetic_profiles.py` (all 9 slices) — expect 9 `[OK]` rows.
- [ ] **Step 2:** Load one output JSON with omas; assert core_profiles grid strictly increasing, ne0 ~1e18–1e19 m⁻³, Te0 ~50–200 eV, Ti0 ~5–30 eV, p_th ≥ 0, velocity.toroidal finite.
- [ ] **Step 3:** Full `pytest test/` in the conda env; compare failures against pre-change baseline (record baseline first).
- [ ] **Step 4: Commit** any test fixture leftovers; final summary.

## Out of plan (reported, not fixed here)

- Notebook `profile_fitting_using_equilibrium_and_kinetic_diagnostics.ipynb` breakages (exist_ts_file, chicken-and-egg time list, estimate_C chain, load() misuse, int() gfile suffix) — the notebook is rewritten wholesale by open PR #37; fixing it now guarantees conflicts. Revisit after PR #37 lands or when the notebook is redone on the new chain.
- `thomson_scattering.py:30` uncertainties-absent silent fallback (env always has uncertainties).
- Deleted-script findings (COCOS psi convention, tar arcname collision, artifact-screen divergence, free_exponential monotonicity doc) — moot after Task 8; the free_exponential monotonicity caveat is documented in the new script's `--ne-mode` help text.
