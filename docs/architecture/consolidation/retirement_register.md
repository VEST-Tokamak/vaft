# Legacy retirement register (#2168, umbrella #2147)

This register tracks every legacy implementation or configuration asset that the
VEST configuration and preprocessing consolidation may retire. It is the Phase E
(#2168 Scope A) inventory. **Every row now carries its final disposition** (conditional on #2196 and #2199 merging) after
B1 (#2181), B2 (#2177, #2184, #2185), B3 (#2178, #2179, #2180, #2183, #2182), C
(#2187, #2188, integrated by #2189), the Phase D qualification (#2191) and the
Phase E PRs (#2196 guards, #2199 deprecations; both still open when this was written,
so rows resting on them say so). A row that is not closed names the PR or issue that owns what is left.

**Phase D evidence (#2191,** `qualification_report.md`**).** BEFORE 75b154e31
(post-#2135) against AFTER 52576480a (the #2189 head; `vaft/` and `workflow/`
identical to develop f7472664d), on vestserver raw data, 50 shots: every ODS
leaf and every k-file (364 per state) is byte-identical, 266 records (262 passed,
4 blocked on 46403, 0 failed, 0 unexplained), every difference `metadata_only`.
#2191 does **not** cover #2182 (not in its AFTER state; #2182 has its own analytic and
synthetic evidence), EFIT/EFUND execution, pipeline 2, or modules outside the
pipeline-1 stages (its §10). Cases are cited below by their #2191 name.

The mapping of every Phase A finding to its disposition is
[`final_reconciliation.md`](final_reconciliation.md). The guards in
`test/test_architecture_guards.py` (core gate) enforce this register. See
[Enforcement](#enforcement).

## Status vocabulary (#2168 §4)

| Status | Meaning | Action |
|---|---|---|
| `qualified_for_removal` | Canonical replacement and regression evidence established | Remove |
| `compatibility_required` | Public callers still depend on the interface | Keep a thin adapter with explicit deprecation |
| `scientific_specialization` | Distinct justified numerical or physical behaviour | Retain under appropriate ownership |
| `reference_only` | Needed to reproduce historical results | Preserve outside active runtime authority |
| `validation_pending` | Independent evidence insufficient | Defer retirement |
| `scientific_correction_pending` | Behaviour change needs explicit scientific approval | Handle separately |
| `removed` | Retired in the named PR with the recorded evidence (a closed row) | None |
| `pending_pr` | The retirement is implemented in an open PR, named in the row | Merge that PR |

Consumers were found by grepping `vaft/`, `workflow/`, `test/` and `docs/` on
develop (7b1fcf192, re-checked at 75b154e31, f7472664d and aeec6d91a).

## Configuration assets

| # | Asset | Canonical replacement | API | Final status | Evidence | PR / tracking |
|---|---|---|---|---|---|---|
| C1 | `vaft/data/geometry/MD.yaml` (field_code, kind, calibration; 64 probes + 11 loops) | `vest.yaml` `0.diagnostics.equilibrium_magnetics.sensors` registry via `magnetics_registry.resolve_equilibrium_magnetics_sensors` | private data file | `removed`; frozen copy `reference_only` in `test/data/legacy_magnetics/`, out of the wheel (`test/verify_dist.py` forbids it) | #2181: old (75b154e31 export) vs new outputs over 47 shots, every wiring/fault/revision boundary, 3 eras: 0 physics differences, 62 metadata-only (provenance `sensor_registry`, static-input manifest/EFUND header names). #2191 wiring-39437-39438: registry = legacy = ODS name on 50 shots, 0 mismatches | #2181 |
| C2 | `vaft/data/geometry/VEST_MagneticsGeometry_Full_ver_2302.yaml` (r, z; content ver_2409) | same registry | private data file | `removed`; frozen copy `reference_only` | as C1; EFUND namelist text and rsi/zsi/xmp2 arrays 0 differences over 3 eras (#2181) | #2181 |
| C3 | `vaft/data/geometry/table.yaml` (field_code → name, 75 magnetic codes) | registry `name`/`aliases` | private data file | `removed`; frozen copy `reference_only` | as C1; all 75 codes magnetic, checked against `sql_table.txt` | #2181 |
| C4 | `vest.yaml` `0.magnetics.flux_loop` placeholder (G13, Q3) | registry flux-loop records | config block | `removed` | no reader in `vaft/`, `workflow/`, `test/` (#2181) | #2181 |
| C5 | `vest.yaml` shot-keyed PF gain blocks 20259/38110/38361/38401 (G15) | `pf_active.processing.revisions` (production reads only these) | config block | `compatibility_required`: read only by the public getters of L7, so they go together | identical values to the live revisions (config_audit.md G15); no production reader | #2205 |
| C6 | `vaft/data/legacy/sql_table.txt` (raw-database field names, extra fields 16/17/22/23/168) (Q6) | none: a separate contract, not a sensor inventory | private data file | `scientific_specialization` (kept: the raw-DB name table) | #2181 tests it against all 75 registry labels; whether the five extra fields record is open | #2181; Q6 check #2203 |
| C7 | `VEST_MACHINE_ERAS` Python table (G14) | none | code constant | `scientific_specialization` (documented as code-owned) | #2184 | #2184 |

## Configuration loaders and public adapters

| # | Implementation | Canonical replacement | API | Final status | Evidence | PR / tracking |
|---|---|---|---|---|---|---|
| L1 | `machine_mapping/magnetics.py::_load_equilibrium_magnetics_channels` (MD.yaml) | `resolve_equilibrium_magnetics_sensors` | private | `removed` | as C1; guard (a2) | #2181 |
| L2 | `magnetics.py::_load_static_channels` (geometry YAML) | same | private | `removed` (verify_wiring.py moved onto the registry in the same PR) | as C1; guard (a2) | #2181 |
| L3 | `magnetics.py::_load_names_by_code` (table.yaml) | registry names | private | `removed` | as C1; guard (a2) | #2181 |
| L4 | `machine_mapping/utils.py::_load_magnetics_channel_groups` (opened the geometry YAML, restated thresholds) | registry + `magnetics.py` threshold constants | private | `removed` as a YAML reader: the function remains, registry-backed, group membership identical | #2181 | #2181 |
| L5 | `magnetics.py` `vest_md_channel_definitions` (alias of `vest_equilibrium_magnetics_channel_definitions`) | `vest_equilibrium_magnetics_channel_definitions` | **public** alias, same object | `compatibility_required`: warns on every access through the module (`__getattr__`) and on the first through the package; removal release vaft 0.11.0 | not needed (same object); `test_api_layer_boundaries.py` pins warning, release and identity; guard (g), pending in #2196, keeps production off it | #2199 (open); removal #2208 |
| L6 | `magnetics.py` `vfit_md` (alias of `vfit_equilibrium_magnetics`) | `vfit_equilibrium_magnetics` | **public** alias, same object | as L5 | as L5 | #2199 (open); removal #2208 |
| L7 | `utils.py::raw_database_info`, `get_diagnostic_info`, `get_static_info` (the only readers of C5) | `resolve_vest_diagnostic`, `resolve_vest_processing_config` | **public**, no production caller | `compatibility_required`: deprecate first, then delete with C5 | refuse a missing explicit path since #2177 (guard (e), pending in #2196, calls them) | #2205 |
| L8 | `magnetics_processing_for_shot` override path (G7) | `resolve_magnetics_processing` (validated, fingerprinted) | public | `removed` as an unvalidated path: it now delegates to the validated resolver | #2184; #2191 metadata_only; guard (f) pending in #2196 | #2184 |

## Processing duplicates

| # | Implementation | Canonical replacement | API | Final status | Evidence | PR / tracking |
|---|---|---|---|---|---|---|
| P1 | `process/magnetics.py::_linear_baseline` (the #639 guards) | `subtract_baseline(fitting_opt="polyfit")`, which absorbs the guards | private | `qualified_for_removal`: delegates since B3.1, the private alias is what is left | #2178 bit-identical to frozen copies (`test/test_baseline_consolidation.py`); #2191 byte-identical (diagnostics on 49 shots; 46403 fails at both states) | #2178; alias #2206 |
| P2 | `machine_mapping/magnetics.py::_polyfit_baseline` | `subtract_baseline` | private | `removed` (dead code) | no caller (#2174) | #2178 |
| P3 | `machine_mapping/magnetics.py::_linear_baseline_subtract` | `subtract_baseline(fitting_opt="polyfit", nan_policy="propagate")` | private | `qualified_for_removal`: delegates since B3.1 | as P1 | #2178; alias #2206 |
| P4 | `code/efit/legacy.py::smooth` | `process.signal_processing.smooth` | **public** (`vaft.code.efit`) | `scientific_specialization` (kept in B3.2: its NaN locality differs); retires with its only caller, the deprecated H-alpha detector | #2179 | #2179; #2206 |
| P5 | `plot/soft_x_rays.py::_baseline_correct` | `process.soft_x_rays.sxr_offset_baseline` | private | `pending_pr` (kept in B3.7; B3.11 routes the plot, the mapper and `sxr_baseline_correction` through one owner, bit-identical to frozen copies) | `test/test_sxr_baseline_owner.py` on the PR branch | #2197 (B3.11, open) |
| P6 | `plot/backend/recipes.py::_build_interferometer_spectrogram` inline `butter`/`filtfilt` | `butterworth_highpass(zero_phase=True)` | private | `removed` | #2179; guard (i) pending in #2196 | #2179 |
| P7 | `process/magnetics.py::_firwin_order` | `fir_filter_coefficients` | private | `removed` (coefficients identical at 25 k/250 k/1 M/2 MHz; an even `filter_order` now raises) | #2179 | #2179 |
| P8 | `machine_mapping/pf_active.py::_legacy_pf_filter` (now `_pf_filter`) | `vaft.process.signal_processing.pf_current_filter` (Hamming `firwin(251)` + `uniform_filter1d(10)`), called by `_pf_filter` | private | `scientific_specialization`, now under process ownership | #2180 bit-identical (`test/test_pf_current_filter.py`); #2191 pf1-gain-45895-45896 passed | #2180 |
| P9 | `process/signal_processing.py` `VEST_CoilCurrentNoiseReduction`, `vfit_signal_startend`, `vfit_signal_start_end`, `signal_onoffset` | `vest_coil_current_noise_reduction`, `detect_active_window`, `signal_on_offset` | **public**, warn since 0.6.0 | `compatibility_required`: the warnings name vaft 0.11.0 | delegating; guard (g) pending in #2196 | #2199 (open); removal #2208 |
| P10 | `machine_mapping/utils.py::process_signal` | `vaft.process.process_signal` | **public**, warns since 0.6.0 | as P9 | delegating; guard (g) pending in #2196 | #2199 (open); removal #2208 |
| P11 | `code/efit/legacy.py::vfit_signal_startend` (its own body: threshold 0.06, the H-alpha window; not the P9 alias) | none decided | **public** (`vaft.code.efit`) | `validation_pending`: retire with the H-alpha detector or keep as a specialization | guard (g) deliberately does not report it | #2206 |
| P12 | `code/efit/legacy.py::correct_flux_loop`, `vest_signal_onoffsetpeak` | none (option `fl_correct_option` is 0 in config.yaml) | public | `validation_pending`: retire if the option stays 0 | statically reached only through the option | #2206 |
| P13 | `process/signal_processing.py::process_signal`, `signal_on_offset`, `vest_coil_current_noise_reduction` (canonical names, legacy semantics) | `vaft.process.onset.*`, the PF filter composition | public | `validation_pending`: count callers, then decide | inventory rows "E: retire once ..." | #2206 |
| P14 | `machine_mapping/tf.py::vfit_tf_current_detailed` and `magnetics.py::vfit_plasma_current` local 26-tap Hamming `firwin` designs | a process-owned donor design (`fir_filter_coefficients` forces odd lengths) | private | `scientific_specialization` under machine_mapping, allowlisted in guard (i) as debt | applied through `apply_fir_filter` since #2135 | #2196; #2206 |
| P15 | `process/magnetics.py` `VestMagneticsProcessingConfig` legacy window ladder (`late_shot_min` 41660, `transient_shot_min/max` 41446/41451) | vest.yaml era `window_override` (set by every era, checked over 30000-50000) | public dataclass fields | `qualified_for_removal` as policy (inert in production), allowlisted in guard (h) as debt | #2196 review sampled ~2,860 shots: none without `window_override` | #2206 |
| P16 | `resample_to_time` decimating a short run unfiltered (an alias) | `resample_to_time` storing NaN where a short run could not be filtered (B3.12) | public behaviour | `removed` (approved scientific correction) | #2182: analytic chirp and synthetic short-run tests (`test/test_resample_short_run_alias.py`); not covered by #2191 (§10) | #2182 |
| P17 | `machine_mapping/soft_x_rays.py::vfit_soft_x_rays_dynamic` own baseline windows (inventory: duplicate) | `sxr_offset_baseline` | public mapper | `pending_pr` | as P5 | #2197 (B3.11, open) |
| P18 | `process/magnetics.py::_calibrated_signal` (inventory: duplicate of `calibrate_vest_signal` for the linear case) | `calibrate_vest_signal` | private | `validation_pending` (the zero-calibration half is closed: the registry rejects zero/non-finite since #2181) | #2181 | #2206 |

The canonical API ownership matrix that B3.9 adds (`api_ownership.json`) is in
#2198 (open); rows here that name a canonical owner should agree with it when it
merges.

## Tests that encode the old arrangement

| # | Test | Replacement | Final status | Evidence | PR |
|---|---|---|---|---|---|
| T1 | `test/test_magnetics_geometry_join.py` (pinned the MD.yaml/geometry positional join and forbade probes in vest.yaml) | `test/test_magnetics_registry.py` (schema, uniqueness, order, 39437/39438 equivalence, equality against the frozen copies) | `removed`, replaced atomically | #2181 | #2181 |
| T2 | `test/test_consolidation_numerics.py::test_a_short_record_is_decimated_without_aliasing` (#2172, strict xfail: the #2166 short-record alias finding) | `test_a_short_record_is_refused_as_nan_rather_than_aliased` (asserts the NaN result B3.12 stores) | `removed`: the xfail had become stale after #2182 (it still failed, on NaN instead of an alias), and is a positive test now | #2182; this closure PR | #2182, closure PR |

## Enforcement

`test/test_architecture_guards.py` is in `test/core_selection.py`
`CORE_MODULES`, in the "Layer boundaries" group. Both routes to `merge-gate`
therefore run it. Package CI `core-test` and `.github/scripts/tdst_gate.sh`
both run `pytest -m "core and not perf"` after `pip install -e ".[dev]"`, and
`test/test_core_selection.py` checks that the two stay identical. grimp is in
the `dev` extra, so the layer guard runs on both gates.

Each active guard has an allowlist of today's state. Any entry that stops
matching fails the guard as stale. When B1, B2 and B3 land, the PR that retires
a row above **must** delete its allowlist entry in the same change.

| Guard | State | Allowlist today |
|---|---|---|
| (a1) no executable string literal in `vaft/` or `workflow/` names `MD.yaml`, `table.yaml` (both case-insensitive, `.yml` too) or the `VEST_MagneticsGeometry_Full_ver_` family (which also matches an f-string over the version). Implemented as an AST scan: docstrings and bare string statements are ignored, Snakefile/`.smk` string and f-string tokens are included, and dot-directories are skipped. **Not caught:** names assembled from fragments, `with_suffix`, and globs | active | only `workflow/architecture_audit/build_inventory.py` (the Phase A generator, which joins the frozen copies in `test/data/legacy_magnetics/`). The runtime readers were removed in #2181 |
| (a2) no new reference (import, call, attribute, `getattr` string) to `_load_equilibrium_magnetics_channels`, `_load_static_channels` or `_load_names_by_code`, keyed by (file, enclosing function, loader). The owner module counts too, so a new public wrapper is caught | active | empty: #2181 deleted the loaders and rewired every caller, so any reference is a reintroduction |
| (b) grimp layer contract: `vaft.process` does not import `vaft.machine_mapping` or `workflow`; `vaft.formula` does not import `vaft.process`, `vaft.machine_mapping`, `vaft.omas` or `omas` | active | five known `process` → `machine_mapping` edges, all lazy function-level imports: `process.impurity` → `machine_mapping.core_profiles` (VEST impurity preset); `process.kinetic_closure` and `process.transport_state` → `machine_mapping.core_profiles` (T_i record grammar, `policy_for_ods`); `process.local_mhd_criteria` and `process.q_profile_context` → `machine_mapping.utils.path_exists` (a generic ODS helper housed in machine_mapping). `vaft.formula` is clean |
| (c) no YAML under `vaft/` or `workflow/` holds sensor records (`r` and `z` plus one of `field_code`, `index`, `name`), either as a list or as a mapping with at least two such values | active | `vest.yaml` `0.diagnostics.equilibrium_magnetics.sensors.records`, the canonical registry and the single allowed authority (#2181); `processing.wiring.revisions[].probe_overrides`, which name a registry id and carry r/z only as pins the resolver checks. The known-fault entries name an id without r/z and no longer match |
| registry schema and uniqueness, through B1's validator | active (#2181; negative control: duplicate id; per-rule controls in `test/test_magnetics_registry.py`) | n/a |
| (e) an explicit, missing `info_file` raises `VestConfigurationError` naming that path: every public function with an `info_file` parameter in `vaft.machine_mapping` (the only reader of the file), `vaft.omas` and `vaft.process` (pass-throughs) is discovered (no hand-kept list) and called with a missing path; the getters that spell it `file` or `options["info_file"]` (`raw_database_info`, `get_diagnostic_info`, `get_static_info`) are called explicitly. A new resolver with an unknown required parameter fails the guard until it gets a representative argument. Not discovered: a resolver hidden behind a decorator without `functools.wraps` | active once #2196 merges (#2196; negative control: a resolver that falls back, one that raises the wrong exception, one that refuses for an unrelated reason) | n/a |
| (f1) unsafe override rejected (machine facts, unknown keys, unflagged calibration) through both public override entry points, `resolve_vest_processing_config` and `resolve_magnetics_processing`, with positive controls (processing override, flagged calibration, restated machine fact) | active once #2196 merges (#2196) | n/a |
| (f2) no code outside the validated owners calls `_deep_merge` or `dataclasses.replace` or `copy.replace` (resolved through the module's imports, aliases included) with an argument whose name, attribute or string subscript key contains `override`. Keyed by (file, function, override argument) | active once #2196 merges (#2196) | `utils.resolve_shot_revisions_with_provenance` (`override`: revision keys validated against the base, G5); `utils.resolve_diagnostics_time_policies` (`overrides`: analysis windows by design, each window validated); `vest_upstream.resolve_magnetics_processing` (`overrides`: validated before `replace`, then `__post_init__`). The main validated merge in `effective_config` applies the already-validated `changes` and is not a site |
| (g) no internal use of deprecated aliases (L5, L6, P9, P10) in `vaft/` or `workflow/`: Python modules are resolved through their imports (relative ones included), attribute chains, `getattr` strings, bare uses inside the owner module, and star imports from an owner module; in Snakefiles, NAME tokens (`name`, `bound.name`) resolved through the file's import lines, so the canonical `vaft.process.process_signal` is not reported. `code/efit/legacy.py`'s own `vfit_signal_startend` is a different function and is not reported. The other legacy names of `machine_mapping.__init__._LEGACY_REPLACEMENTS` (warned on package-level access) are outside this rule | active once #2196 merges (#2196) | empty |
| (h) `vaft.process` names no `vest.yaml` path in an executable string (a whitespace-free literal ending in the file name), references no `vaft.machine_mapping` / `vaft.data.geometry` module by string (dynamic import, `importlib.resources`), compares no shot number (a name, attribute, string subscript key such as `...pulse`, or `int()` of one) with a literal or tuple of literals of magnitude >= 1000, and binds no shot-boundary name (`shot`, `first_shot`, `late_shot_min`, `shot_max`, ..., defaults included) to such a literal; `MAX_SHOTS` or `shot_window_ms` are not boundaries. `np.interp`, `scipy.signal`, integration, comparisons against a passed-in boundary, and prose error messages stay legal (negative control includes them). Not caught: a relative dynamic import (`import_module("..machine_mapping", ...)`), nested tuples, dict keys and lambda defaults holding shot numbers | active once #2196 merges (#2196) | `process/magnetics.py` `VestMagneticsProcessingConfig.late_shot_min`/`transient_shot_min`/`transient_shot_max` (41660/41446/41451): the legacy window ladder `window_for_shot` uses only when no `window_override` is set; every vest.yaml era sets one (checked over 30000-50000). Debt, move or delete with the ladder (#2206) |
| (i) no digital filter is designed (`scipy.signal.firwin`, `firwin2`, `firls`, `remez`, `kaiserord`, `butter`, `cheby1/2`, `ellip`, `bessel`, `iirfilter`, `iirdesign`, `iirnotch`, `iirpeak`, `savgol_coeffs`; resolved through imports) in `vaft/` outside `vaft/process`, or in `workflow/`. Applying a filter (`lfilter`, `filtfilt`) stays legal everywhere, and so do `decimate` and `savgol_filter`, which design internally but are resampling/smoothing calls (migration_plan.md E: machine_mapping designs no filters outside an allowlist). Not caught: `from scipy.signal import *`, a design function passed uncalled (`partial(signal.butter, ...)`), a file that also binds stdlib `signal`, Snakefile `run:` blocks | active once #2196 merges (#2196) | `tf.py::vfit_tf_current_detailed` and `magnetics.py::vfit_plasma_current`: the donor Hamming `firwin` with vest.yaml values, applied through `apply_fir_filter` (#2135); both 26 taps (an even-length design `fir_filter_coefficients` cannot produce: it forces odd lengths), so they need a named process-owned donor design or an even-length option (#2206) |

Each active guard has a negative-control test that feeds the detector a
synthetic violation under `tmp_path` and asserts the reported key and failure
message: file (and line, except for YAML), rule and the expected-instead hint.
The controls include near misses that must *not* be reported: docstrings,
`cmd.yaml`, `efit_table.yaml`, bare positions and name-only tables. A separate test shows that a stale
allowlist entry fails.

The five `process` → `machine_mapping` edges in (b) are architectural debt.
They predate the consolidation, and the guard only stops the list from
growing. They are outside the #2147 approved scope unless the lead assigns
them.

## Non-core boundary tests

Two test modules that the consolidation relies on were measured for the develop gate; one
joined it in #2196. The other still runs on the `main` gate and on the push to `develop`.
Timings are local wall clock on the shared 10-core MacBook, `-n 0`, at
7b1fcf192:

| Module | Tests | Wall clock | What it protects | Proposal |
|---|---|---|---|---|
| `test/test_vest_yaml_boundaries.py` | 62 | 33.2 s at 7b1fcf192; 1.7 s locally and 0.9 s on vestserver (`-n 0`) after #2177's mtime-keyed parse cache | revision overlap and shot-range boundaries across vest.yaml diagnostics | **joined core in #2196** (the "VEST configuration" group of `core_selection.py`): at about 1-2 s it is cheaper than most boundary checks there |
| `test/test_pipeline1_workflow.py` | 19 | 42.9 s | Snakemake parse and DAG dry-run of Pipeline 1 (shells out to `snakemake`) | **keep out of core.** It costs more than `test_vest_yaml_boundaries.py` and shells out to an external tool, which core excludes by policy (`core_selection.py` docstring). Phase C (#2164) PRs run it explicitly. It needs a writable snakemake cache directory (`~/Library/Caches/snakemake` on macOS): 16 of 19 tests fail without one |

## Change log

- 2026-10-25: closure audit (#2168). Every row carries its final status, evidence (#2191 included) and owning PR or issue; new rows C5-C7, L7-L8, P11-P18, T2; the `pending_pr` status for rows implemented in an open PR. Follow-ups filed: #2200-#2208. Mapping of every Phase A finding: `final_reconciliation.md`.
- 2026-10-25: after #2174 merged, `workflow/architecture_audit/build_inventory.py` (the Phase A generator) is allowlisted in guard (a1) for its three module-level references: it reads the split files as reconciliation evidence, not as runtime configuration. #2158 repoints it at the frozen fixtures in `test/data/legacy_magnetics/`.
- 2026-10-25: rows P1, P2, P7 and P8 updated after #2135 merged (75b154e31). Every guard allowlist still matches exactly.
- 2026-10-25: skeleton and active guards (a)–(c), placeholders (d). PR `architecture/2168-guards-prep`. The module adds about 1 s serial (`-n 0`) to a core gate that took 583 s locally at `-n 2`.
