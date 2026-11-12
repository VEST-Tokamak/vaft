# Consolidation migration plan (Phases B–E of #2147)

This plan is the Phase A deliverable E for #2148. It builds on:

- `inventory.json` and `callgraph.md` (preprocessing);
- `config_audit.md` (gaps G1–G17);
- `magnetics_reconciliation.md` and `magnetics_inventory.json` (sensors).

PR #2135 merged into develop as `75b154e31`. Since then, `apply_fir_filter` is the
FIR application for measured waveforms, with two exceptions: `fir_filter(zero_phase=False)` still calls `lfilter` directly (`vaft/process/signal_processing.py:742`), and `anti_alias_filter` runs its own `filtfilt`. `b_field_pol_probe_field` is the single B-pol chain.
Line numbers refer to `75b154e31`.

Every numerical change is classified in its PR. The difference classes are:
`equivalent`, `metadata_only`, `approved_correction`, `unexplained_regression`,
`reference_inconclusive`, and `not_tested`.

## 1. Risk-ranked findings

The ranking weighs four things: scientific impact, production reach, how
implicit the assumption is, and whether the finding can be verified
independently.

| Rank | Finding | Kind | Reach | Evidence | Disposition |
|---|---|---|---|---|---|
| 1 | Causal FIR (`lfilter`) on B-pol probes (251 taps at 250 kHz), the TF current (26 taps at 25 kHz) and the Ip-reference comparison, giving a **0.5 ms lag** against the undelayed PF currents | confirmed defect | P1 diagnostics, every shot (probes, TF); IMPA via TF | #2147 comment: main = public(t − 0.50 ms) on 63 probes; vacuum residual improves on 5 shots | fixed by **#2135**. The 1,012 stale 43685–46342 override products and every pre-#2135 product need regeneration, which needs owner approval (out of scope here) |
| 2 | 75 sensors defined by **three positional lists**: MD.yaml ↔ the geometry YAML joined by index, plus table.yaml by field. The geometry filename says 2302 while its content declares 2409. | duplicated authority | P1 static, diagnostics, constraints, EFUND | `test_magnetics_geometry_join.py`; #843 | **B1** registry in vest.yaml; EFUND identity kept historical |
| 3 | No stable physical id. The ODS `name`/`identifier` follow the DAQ field, so index 35 and index 47 swap names at 39437/39438. The `MagneticsWiring` docstring claims names do not move. Five table labels disagree with the geometric ladder. | identity ambiguity | P1 static/diagnostics; anything keyed by probe name across eras | `MM/magnetics.py:1713-1722`, `:738-741`; test `test_equilibrium_magnetics_processing.py:584` | **B1** separate `id`; ODS labels unchanged; Q4/Q5 to the owner |
| 4 | Flux loops get **no anti-alias low pass** before the decimating `np.interp` onto the 25 kHz grid (250 kHz acquisition in the native-DAQ era ≥ 46404). Two comments claim the loop voltage is "already low-passed". The same `_interpolate_signal` also carries unfiltered Ip (`MM/magnetics.py:2232`) and diamagnetic flux (`:2468`). | plausible, unverified | P1 diagnostics: 11 loops (mostly ≥ 46404), Ip, diamagnetic flux | `vaft/process/magnetics.py:1001-1003` vs `vest_flux_loop_voltage` (calibration only); `MM/magnetics.py:925-927` | **D** measures the spectra above the target Nyquist on a native-DAQ shot; **B3** fixes the comments; no change without evidence |
| 5 | Declared sample rates (IMPA 25 kHz, magnetics 250 kHz, TF/PF 25 kHz, Ip-ref) are never checked against the acquired time vector | implicit assumption | P1 IMPA (opt-in), diagnostics | `MM/impa.py:214,251`; #432/#425 | **B2** `check_declared_sample_rate`, report-only |
| 6 | Provenance gaps. The diagnostics manifest stores the effective magnetics config only when overridden, with no vest.yaml digest and no vaft version. `vest_processing_provenance` has no stage caller. | reproducibility | every P1 product | VU:1125-1140; `MM/provenance.py:161` | **B2 + C1**: always record effective config, fingerprint, vest.yaml sha256 and vaft version |
| 7 | An explicit `info_file` that is missing falls back silently to the packaged vest.yaml; `_deep_merge` copies shallowly; the policy cache is returned mutable | latent defect | any caller passing `info_file`, any mutating caller | `MM/utils.py:89-115`, `:70-86` | **B2** (first PR in the merge order) |
| 8 | Revision **unknown keys** are accepted. IMPA `shot_era_overrides` has no overlap check, takes the first match, and defaults missing bounds to 0. `magnetics_processing_for_shot` raises a bare `TypeError`. `VestMagneticsProcessingConfig` is unvalidated. | latent defect | all revisions; IMPA; workflow overrides | G5–G8 | **B2** shared revision validator + field classes |
| 9 | Duplicate linear-baseline helpers with **different guards**: `process.magnetics._linear_baseline` (finite-only, #639) vs `signal_processing.subtract_baseline` (`curve_fit`, no guards) vs `MM._linear_baseline_subtract`. `MM._polyfit_baseline` is **dead**. Also: `legacy.smooth` is a verbatim copy; the plot-recipe Butterworth is a copy; Rogowski/diamagnetic `cumulative_trapezoid` bypasses `cumtrapz_compat`; `_firwin_order` duplicates `fir_filter_coefficients`. | maintenance duplication | P1 (probes, loops, Ip) | inventory rows; `_firwin_order` coefficients identical (checked) | **B3**, each slice with an equivalence test (1e-12 or exact) |
| 10 | Stale second authorities in vest.yaml: the dead `0.magnetics.flux_loop`; shot-keyed PF gain blocks (20259/38110/38361/38401) that restate the live revisions; `toroidal_mirnov_reference` restating field 171 with the **opposite sign under the same divide convention**; `vaft/data/legacy/sql_table.txt` as a second label table (it has extra sensors and gives field 25 two names). A hard-coded `VEST_MACHINE_ERAS` lives in Python. | misleading authority (#843 pattern) | field 171: preprocessed Mirnov plots of pre-#825 replicas only; the rest have no runtime effect | G13–G16; `magnetics_reconciliation.md` Q1, Q6 | **B1** removes the placeholder and decides on `sql_table.txt`; **E** removes the PF blocks; follow-up issue for the field-171 alias sign; B2 decides on `VEST_MACHINE_ERAS` |

Findings outside the consolidation, recorded for their owners. These are not
work packages here:

- Pipeline 2 fitted profiles carry zero σ (`vaft/formula/utils.py:1313`).
- σ(T_i) is computed but not stored (VU:2773).
- The ADF11 interpolator extrapolates linearly outside the table (`vaft/formula/atomic.py:130-135`).
- The electron-only core-profile fit takes its coordinate from a hard-coded default, not from vest.yaml.
- The Pipeline 3 data path is hard-coded.

## 2. Work packages

The order follows the plan's merge order: B2 loader/fallback → B1 registry → B2
effective config → B3 slices → C1..C5 → D → E.

### B2a — loader safety (#2159), first

- **Files:** `vaft/machine_mapping/utils.py` (`_resolve_info_file_path`, `_deep_merge`, `_policy_document`).
- **Change:**
  - a missing explicit path raises `VestConfigurationError`;
  - `_deep_merge` deep-copies, and a type mismatch raises;
  - `_policy_document` returns a `MappingProxyType`, or a deepcopy.
- **Tests:** synthetic YAML cases for G1–G3; immutability; cache invalidation by mtime.
- **Risk:** callers that relied on the fallback. A grep finds no production caller
  that passes `info_file`; tests may.
- **Acceptance:**
  - static and diagnostics ODS for 39915/41524/41672/42962 are array-identical (`equivalent`);
  - `test_vest_yaml_boundaries.py` is green (core since #2196; it was non-core when this plan was written).

### B1 — canonical magnetics registry (#2158)

- **Schema:** `vest.yaml` `0.diagnostics.equilibrium_magnetics.sensors` (block-level
  `source`, 75 ordered `records`), as specified in `magnetics_reconciliation.md` §7.
  It **deviates from the plan in two ways**: `source` is given once per block, and an
  `id_basis` field is added.
- **New module:** `vaft/machine_mapping/magnetics_registry.py`:
  - `resolve_equilibrium_magnetics_sensors(shot, info_file=None) -> tuple[SensorRecord, ...]`, with `SensorRecord` a frozen dataclass;
  - validation as in §7;
  - YAML is loaded through one private function, which B2b replaces.
- **Rewire:** the loaders in the consumer map (`magnetics_reconciliation.md` §6).
  - Public names keep returning byte-identical structures: `vest_equilibrium_magnetics_channel_definitions`, `vest_md_channel_definitions`, `magnetics_wiring_for_shot`, `known_magnetics_faults`, `equilibrium_probe_count`.
  - `wiring` and `known_faults` entries are re-targeted by `id`. The legacy `index`+r/z form is accepted through a validated adapter.
  - `utils._load_magnetics_channel_groups` uses the registry `family` and the `magnetics.py` threshold constants.
  - Remove the dead `0.magnetics.flux_loop` (G13).
- **Atomic in the same PR:**
  - replace `test_magnetics_geometry_join.py` with `test_magnetics_registry.py`, checked against **frozen copies** of the three YAMLs in `test/data/legacy_magnetics/` (non-runtime fixtures);
  - fix `test_magnetics_geometry_registry.py:46, 58` (direct YAML read);
  - `test/verify_dist.py:86`;
  - `MANIFEST.in:41` and `pyproject.toml:232` stay (they are globs);
  - docs: `docs/_guide/Machine_mapping.md`, `docs/_guide/Magnetics.md`, `vaft/data/README.md`, the `vest.yaml:20-45` comment;
  - delete the three runtime YAMLs.
- **Provenance:**
  - the static manifest records `magnetics_registry` = vest.yaml + registry fingerprint (VU:278-289);
  - `efund_table_manifest.json` and the reference JSONs are **untouched**;
  - an EFUND projection test proves that `rsi`/`xmp2` (`MM/efund_geometry.py:293-330`) are unchanged, classified `metadata_only`.
- **Field 171:** keep both values; open a follow-up issue on the sign of the legacy alias (Q1).
- **Legacy flux baseline (Q7):** the late-baseline rule targets positional loops
  9–11, i.e. fl_13/fl_14/fl_15. Preserve that behaviour explicitly.
- **`sql_table.txt` (Q6):** derive its magnetics rows from the registry, or document
  it as the raw-DB name table.
- **Decisions the owner must make before merge:**
  - Q4: the 5 position-shaped ids and their ladder candidates;
  - Q2: names without `_Bz`, kept verbatim.
- **Risk:** the stash/wiring hazard. Registry, rewiring and YAML deletion must ship
  together; an intermediate commit with two live authorities is not acceptable.
- **Acceptance:**
  - static ODS for shots 0, 39437, 39438, 39915 and 46404 are array-identical to the Wave-1 baseline;
  - the new test is equal to the frozen fixtures;
  - the `magnetics_inventory.json` field set is reproducible from vest.yaml alone.

### B2b — effective configuration (#2159)

- **New module:** `vaft/machine_mapping/effective_config.py`:
  - `ResolvedVestConfiguration(shot, diagnostic, effective, provenance, fingerprint)`, frozen;
  - `resolve_vest_processing_config(shot, diagnostic, *, overrides=None, info_file=None)`;
  - per-key provenance (`source`, `revision_index`, `revision_bounds`, `overridden`, `defaulted`);
  - `fingerprint` is the sha256 of canonical JSON (sorted keys, float `repr`) of the effective values;
  - `source_sha256` of the file is stored separately.
- **Revision validation (G5, G6):**
  - one shared validator that rejects unknown keys against the base;
  - an IMPA `shot_era_overrides` adapter: `min_shot`/`max_shot` become inclusive bounds, overlap is rejected, and missing bounds are explicit;
  - the YAML schema is unchanged.
- **Field classes (G7, G8, G17):**
  - machine facts (sensors, wiring, `daq_mode`) cannot be overridden;
  - calibration needs an explicit flag;
  - processing and analysis keys can be overridden;
  - an unknown key raises `VestConfigurationError`;
  - `VestMagneticsProcessingConfig.__post_init__` validates rates, taps, and cutoff < Nyquist.
- **Observed acquisition (G9, G10):**
  - `vaft/process/acquisition.py:check_declared_sample_rate(time, declared, rtol)` returns `consistent | deviation | invalid | not_verifiable`;
  - it is called at the IMPA low pass, the magnetics probe chain and the Ip reference;
  - the Nyquist skip in `mirnov_preprocess_signal` becomes an explicit status;
  - IMPA (#432) is report-only, with no change of numbers.
- **Provenance (G11, G12):**
  - `vest_processing_provenance` gains `info_file`, TF/IMPA/time policies and the fingerprint;
  - the diagnostics and IMPA manifests always record the effective config, fingerprint, vest.yaml sha256 and vaft version.
- **G14** (`VEST_MACHINE_ERAS`): either move it into vest.yaml or document it as code-owned. Decide in this PR.
- **Tests:**
  - synthetic YAML for every G-row;
  - the boundaries 39437/8, 46402/3, 46403/4, 45895/6 and 48371/2;
  - stable fingerprint; immutability.
- **Acceptance:** numerical outputs are `equivalent`; manifests change only as `metadata_only`.

### B3 — canonical processing (#2162), one slice per row

| Slice | Change | Equivalence evidence |
|---|---|---|
| B3.1 linear baseline | `signal_processing.subtract_baseline(method="polyfit")` absorbs the `_linear_baseline` guards (finite-only, #639 degenerate error/fallback). `_linear_baseline` and `MM._linear_baseline_subtract` delegate. `MM._polyfit_baseline` is **deleted** (it has no caller). | 1e-12 on synthetic and 39915 probe/loop/Ip |
| B3.2 smoothing | `code/efit/legacy.py:347 smooth` → import `signal_processing.smooth` | rtol 1e-12, not bit-identical (cumsum vs loop rounding: 2–3e-8 absolute on 1e5-scale data); check that no caller passes span > record length, which only the canonical version clips |
| B3.3 integration | the `cumulative_trapezoid` calls in `MM/magnetics.py:1401` (diamagnetic Rogowski) and `:1474, :1505-1506` (diamagnetic flux) → `cumtrapz_compat` | exact |
| B3.4 Mirnov design | `_firwin_order` → `fir_filter_coefficients` | coefficients identical (Phase A check, max \|diff\| 0.0 at 25 k / 250 k / 1 M / 2 MHz) |
| B3.5 PF filter | named `pf_current_filter` in `vaft.process`: `fir_filter_coefficients(window="hamming", numtaps=251)` + `uniform_filter1d`, machine values from vest.yaml. `_legacy_pf_filter` already applies the filter through `apply_fir_filter(short_record="causal")` (`MM/pf_active.py:156`, since #2135), so what remains is naming and the shared coefficient design | coefficient equality + phase test |
| B3.6 plot recipe | `recipes._build_interferometer_spectrogram` Butterworth → `butterworth_highpass(zero_phase=True)` (add it) | exact |
| B3.7 plot SXR | `plot/soft_x_rays._baseline_correct` → `sxr_baseline_correction` **only where the window semantics match** (range mean vs tail with fallback); otherwise keep and document | per call site |
| B3.8 comments | correct the "already low-passed" comments (`vaft/process/magnetics.py:1001`, and `MM/magnetics.py:925`, which also covers Ip and diamagnetic flux) | none needed |
| B3.10 box average | `P1/generate_constraints_ods.py:295 _box_averaged` (cumsum) → a vectorised `box_average` next to `legacy.box_average` | 1e-12 against per-sample `box_average` |
| B3.12 short-run anti-alias | `resample_to_time` → `anti_alias_filter` (`vaft/process/signal_processing.py:1170`) passes a record or finite run shorter than `3·numtaps + 1` samples (`_filtfilt_min_length`, `:1146`) through **unfiltered**, and `resample_to_time` then decimates it, so the short run can alias. The unfiltered samples are reported only through `return_mask` / `return_filter_mask` (#915), plus a `RuntimeWarning`. Langmuir already uses the mask: `MM/langmuir_probes.py:340-356` sets validity −1 on every target sample an unfiltered run touches, which covers the NaN-split runs of 39915. The exposed callers are the ones that take no mask: `MM/barometry.py:81`, `MM/ec_launchers.py:267, 272`, `MM/spectrometer_uv.py:171`, `vaft/omas/process_wrapper.py:1097`, `vaft/process/fluctuation.py:1351`, `vaft/plot/mirnov.py:283, 307`, `vaft/process/signal_processing.py:1729`. Fix: make the mask (or a validity projection like Langmuir's) mandatory for production callers, or refuse short runs. | analytic chirp test on a short run; per-caller check that unfiltered samples are flagged |
| B3.11 SXR baselines | one owner for the four SXR baseline variants (`MM/soft_x_rays.vfit_soft_x_rays_dynamic` index-range mean, `process.soft_x_rays.sxr_baseline_correction` tail mean, `plot.soft_x_rays._baseline_correct` range mean, the rolling detrend in `sxr_electron_temperature`), with explicit window semantics | per call site |
| B3.9 ownership | `docs/architecture/consolidation/api_ownership.json`; deprecations via the existing alias pattern | — |

- **Do not touch** (intentional specializations, see the inventory):
  - TF excursion repair + `smooth(50)`;
  - the FL10 decimate chain;
  - onset `zero_phase_lowpass` / `median_smooth`;
  - barometry/Langmuir `medfilt`;
  - the flux-loop baseline rules;
  - the diamagnetic two-point baseline;
  - the Rogowski `savgol` in the generic `rogowski_coil_ip`.
- **Dependency:** #2135 (merged as `75b154e31`, satisfied).

### C — consumer and workflow migration (#2164), sequential slices

Each slice runs:
- a frozen-baseline compare on 39915, 41524, 41672 and 42962 in an isolated root;
- a Snakemake dry-run DAG diff;
- `test_pipeline1_workflow.py`, which is non-core, so run it explicitly.

| Slice | Content | Depends on |
|---|---|---|
| **C1** | `build_static_ods` / `build_diagnostics_ods` take the registry and the effective config; the Snakefile passes overrides only through the B2 API; manifests carry the fingerprint; stages call `vest_processing_provenance`. **Also fix channel status:** `missing_channels` and `component_status` for magnetics are computed from the configured field codes **absent from the archive** (`vaft/omas/vest_upstream.py:1015-1033`, `1059-1060`), not from processed output. Since #2135, a probe whose record is too short for the zero-phase low pass is published as an empty channel (`vaft/process/magnetics.py:962-972`), but because its field is archived it is **not** listed in `missing_channels`, and the component can still report `success`. Derive the status from the processed product, or add those channels to `signals_left_out`. | B1, B2b |
| **C2** | `kfile.py:578-624` grouping from registry `family`; IMPA / eddy / constraints use the effective config | C1 |
| **C3** | EFIT / k-file / EFUND: prove EFUND input identity unchanged (`mhdin.dat` projection) | C2 |
| **C4** | Pipeline 2: no preprocessing duplicates were found, so the slice is limited to provenance | C1 |
| **C5** | CLI; `workflow/magnetics_geometry/verify_wiring.py` by `id`; `workflow/resistive_zeff/build_zeff.py:239` FL10 by registry id; notebooks | B1 |

### D — independent verification (#2166)

- **Short-record anti-alias finding** (reported by D): `resample_to_time` decimates
  records or finite runs shorter than `3·numtaps + 1` without an anti-alias filter.
  - D saw it on the NaN-split Langmuir runs of 39915. There, `MM/langmuir_probes.py:340-356`
    already marks the affected target samples invalid (#915).
  - The callers that take no filter mask are unprotected (list in B3.12). The fix
    belongs to B3, slice B3.12.
- **Case manifest** with offline replayable shots 39915, 41524, 41672 and 42962.
  Boundary shots 39437/8, 46402–46404, 45895/6 and 48371/2 need vestserver raw
  data: read-only, isolated output roots, replication off.
- **Analytic signal tests:** impulse, step, sinusoid, chirp, short record and NaN,
  checking magnitude **and** phase/group delay.
- **Vacuum benchmark** per sensor, with negative controls: swap two probe ids, flip
  a sign, move a probe.
- **Spectrum check of the unfiltered `np.interp` sources** (finding 4).
- **Donor check for Q7** (positional vs physical loops 9–11), and whether the
  `sql_table.txt`-only fields record (Q6).
- **TF-coupling shift after #2135** (1.01–1.05 → 1.02–1.07), recorded.
- **48224:** the stage-by-stage comparison stays `reference_inconclusive` unless
  there is evidence. #1541 must explicitly rule the 0.5 ms delay in or out.

### E — retirement and CI (#2168)

- **Guards** in `test/core_selection.py`, each with a negative control:
  - registry schema and uniqueness;
  - an AST check that nothing in vaft/ or workflow/ reads the retired YAML names (`build_inventory.py`'s scanner can be reused);
  - no second sensor inventory in YAML or Python;
  - resolver path and override safety;
  - layer contract: process must not import machine_mapping/workflow, and machine_mapping must not design filters outside an allowlist;
  - a deprecated-API guard.
- **Retire** (public exports such as `raw_database_info`/`get_diagnostic_info` go through deprecation first):
  - `MM._polyfit_baseline`;
  - `signal_processing.vest_coil_current_noise_reduction` / `VEST_CoilCurrentNoiseReduction`, `process_signal`, `signal_on_offset`, if no caller remains;
  - the shot-keyed PF gain blocks (G15);
  - the dead getters `raw_database_info`, `get_diagnostic_info`, `get_static_info`;
  - `legacy.correct_flux_loop` / `vest_signal_onoffsetpeak`, if `fl_correct_option` stays 0.
- `tdst_gate.sh` runs the same selection.
- `verify_dist.py` + wheel smoke test.
- Add `test/test_architecture_audit_inventory.py` to the core selection only after
  B1 lands. The JSON stores no line numbers, so plain line shifts do not stale it.
  A PR that adds, removes or rewrites a scanned call, or renames a curated
  function, has to regenerate it. The generator refuses curated rows with no
  scanned implementation, so a stale classification cannot survive a removal.

## 3. Where the evidence contradicts or refines the approved plan

1. **"`code/efit/efund_geometry.py:293`".**
   - The file is `vaft/machine_mapping/efund_geometry.py`.
   - Line 293 is `_flux_loops`; `_probes` follows it.
   - Both read the **static ODS** in index order, not the YAMLs.
   - The "EFUND index = ODS index = YAML order" conclusion holds.
2. **`MM._polyfit_baseline:989` is dead code.** Nothing in vaft/, workflow/ or
   test/ calls it, so B3 deletes it rather than delegating to it.
3. **`utils._load_magnetics_channel_groups` is not production-reachable.** Its only
   path runs through `apply_default_constraint_uncertainties`, which only tests
   call. The B1 rewiring still applies, at low urgency.
4. **Stable `id` = "physical name, e.g. `bpol_H1-01`".**
   - The table.yaml names are **DAQ-field labels that follow the wiring**. The ODS
     identifiers are those labels, and a test pins them.
   - The registry `id` therefore has to be a new, separate field.
   - For 5 sensors (indices 35, 46, 47, 61, 62) the label disagrees with the
     geometric ladder. Label-shaped ids would reuse another position's DAQ label,
     so those ids are **position-shaped**, with a ladder candidate for the owner
     to confirm.
5. **Record shape.** A per-record `source` would repeat three strings 75 times, so
   the schema gives `source` once per block instead. `id_basis` is added.
6. **Field 171 "opposite sign gain": confirmed.**
   - Both copies divide. The plan's reading is right.
   - The runtime reach is narrow: the legacy `:phase_reference` alias, used when
     preprocessing Mirnov plots of replicas mapped before #825 (now closed).
     Field 171 is otherwise published once, as equilibrium probe 36.
7. **"Rogowski savgol" as a production specialization.**
   - The production Rogowski chains (`vest_plasma_rogowski_current`,
     `vest_diamagnetic_rogowski_current`) apply no savgol.
   - `savgol` appears only in the generic `process.magnetics.rogowski_coil_ip`, which
     no Snakefile reaches, and in study scripts.
8. **"`process/magnetics.py:1209` silently skips cutoff ≥ Nyquist".**
   - The code is correct as described. It also skips cutoffs ≤ 0.
   - It sits in `mirnov_preprocess_signal`, which no Snakefile reaches: Pipeline 1
     publishes Mirnov voltages raw.
9. **Not in the plan:**
   - finding 4 (unfiltered `np.interp` sources: loops, Ip, diamagnetic flux);
   - `vaft/data/legacy/sql_table.txt` as a second label authority (Q6);
   - the positional loop rule (Q7);
   - the workflow-local `_box_averaged` and the fourth SXR baseline variant;
   - diagnostics QA writes validity (it is not report-only);
   - G14 (`VEST_MACHINE_ERAS`) and G15 (shot-keyed PF gain blocks);
   - the second direct YAML reader `test/test_magnetics_geometry_registry.py:46, 58`;
   - the misleading `MagneticsWiring` docstring.
10. **Confirmed as stated:**
    - the `MD.yaml` / geometry / `table.yaml` 75-code identity;
    - the `ver_2409.mat` declaration;
    - the 35/47 pre-39438 overrides;
    - the line numbers of the magnetics loaders;
    - the resolver gaps (G1–G12);
    - the `test_magnetics_geometry_join.py` / `verify_dist.py:86` blockers;
    - the specializations to keep.
