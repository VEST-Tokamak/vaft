# Legacy retirement register (#2168, umbrella #2147)

This register tracks every legacy implementation or configuration asset that the
VEST configuration and preprocessing consolidation may retire. It is the Phase E
(#2168 Scope A) inventory. B1 (#2158) removed rows C1-C4, L1-L4 and T1 in #2181, with the equivalence
evidence recorded in each row; the other rows have no independent equivalence
evidence yet, so none of them is `qualified_for_removal`. The
Phase D qualification report (#2166) supplies that evidence. Removal then
happens one row at a time, in the PR named in the row.

The guards in `test/test_architecture_guards.py` (core gate) enforce part of
this register today. See [Enforcement](#enforcement).

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

The **Status** column is the current status. **Proposed** is the expected
disposition once the owner phase lands and Phase D supplies evidence. A
proposal is not a decision.

Consumers were found by grepping `vaft/`, `workflow/`, `test/` and `docs/` on
`develop` at 7b1fcf192 (2026-10-25) and re-checked after #2135 (the
FIR-filter consolidation) merged at 75b154e31. #2135 settled several facts:

- `vest_b_field_pol_probe_legacy` is removed (BREAKING, see `vaft/version.py`).
- The PF, IMPA, Mirnov, TF and I_p-reference FIR paths now all apply their
  taps through `process.signal_processing.apply_fir_filter`.

## Configuration assets

| # | Asset | Canonical replacement | Consumers (runtime / other) | API | Equivalence evidence | Status | Proposed | Owner phase | PR |
|---|---|---|---|---|---|---|---|---|---|
| C1 | `vaft/data/geometry/MD.yaml` (field_code, kind, calibration; 64 probes + 11 loops) | `vest.yaml` `0.diagnostics.equilibrium_magnetics.sensors` registry via `magnetics_registry.resolve_equilibrium_magnetics_sensors` (#2158) | runtime: `machine_mapping/magnetics.py::_load_equilibrium_magnetics_channels`; `omas/vest_upstream.py::build_static_ods` (sha256 into static provenance as `magnetics_calibration`). That hash propagates: `machine_mapping/efund_geometry.py::_machine_block` copies `static_inputs`, `code/efit/efund.py::_header_lines` writes it into the mhdin header, and `test/test_efit_packaged_table.py` requires `static_inputs`. Replacing it with a registry fingerprint therefore changes EFUND headers and manifests (`metadata_only`, see the plan). Other: `test/verify_dist.py:86` (required file), `test/test_magnetics_geometry_join.py`, `vaft/data/efit/efund_table_manifest.json` (historical sha256), `docs/_guide/Machine_mapping.md`, `docs/_guide/Magnetics.md`, `vaft/data/README.md`, `vest.yaml` comments | private data file (shipped via `data/geometry/*.yaml`) | #2181: old (75b154e31 export) vs new outputs over 47 shots (0-49999, every wiring/fault/revision boundary, 3 machine eras): 0 physics differences; 62 metadata-only (provenance `sensor_registry` key, static-input manifest/EFUND header names) | `removed` in #2181 (frozen copy kept `reference_only` in `test/data/legacy_magnetics/`, out of the wheel) | `reference_only` (frozen copy to `test/data/legacy_magnetics/`, out of the wheel) | B1 #2158, E #2168 | #2181 |
| C2 | `vaft/data/geometry/VEST_MagneticsGeometry_Full_ver_2302.yaml` (field_code, kind, r, z; content is ver_2409) | same registry (#2158) | runtime: `magnetics.py::_load_static_channels`; `utils.py::_load_magnetics_channel_groups`; `vest_upstream.py::build_static_ods` (sha256 as `magnetics_geometry`; same EFUND header and manifest propagation as C1). Other: `test/test_magnetics_geometry_registry.py`, `test_magnetics_geometry_join.py`, `test_magnetics_toroidal_placement.py` (through the loader), EFUND manifest (historical sha256), docs as for C1 | private data file | #2181: old (75b154e31 export) vs new outputs over 47 shots (0-49999, every wiring/fault/revision boundary, 3 machine eras): 0 physics differences; 62 metadata-only (provenance `sensor_registry` key, static-input manifest/EFUND header names) | `removed` in #2181 (frozen copy kept `reference_only` in `test/data/legacy_magnetics/`, out of the wheel) | `reference_only` | B1 #2158, E #2168 | #2181 |
| C3 | `vaft/data/geometry/table.yaml` (field_code → name, the same 75 codes, no non-magnetic entries) | registry `name`/`aliases` (#2158) | runtime: `magnetics.py::_load_names_by_code`, used by `_populate_flux_loop_static`, `_populate_probe_static` and `raw_daq_sources.py::_wiring_records`. Other: `test_equilibrium_magnetics_processing.py` (docstring), docs | private data file | #2181: old (75b154e31 export) vs new outputs over 47 shots (0-49999, every wiring/fault/revision boundary, 3 machine eras): 0 physics differences; 62 metadata-only (provenance `sensor_registry` key, static-input manifest/EFUND header names) | `removed` in #2181 (frozen copy kept `reference_only` in `test/data/legacy_magnetics/`; all 75 codes magnetic, checked against `sql_table.txt`) | `reference_only` | B1 #2158 | #2181 |
| C4 | `vest.yaml` `0.magnetics.flux_loop` (a single placeholder loop and channel `FL1`) | registry flux-loop records (#2158) | none found: no resolver reads the `0.magnetics` key (grep for `resolve_vest_diagnostic(…"magnetics")`, `["magnetics"]` on config). Phase A (#2148) confirms | config block | #2181: confirmed absence of readers (no resolver reads the top-level `magnetics` key; nothing in `vaft/`, `workflow/` or `test/` used it) | `removed` in #2181 (no reader) | `qualified_for_removal` | B1 #2158 | #2181 |

## Configuration loaders

| # | Implementation | Canonical replacement | Consumers | API | Equivalence evidence | Status | Proposed | Owner phase | PR |
|---|---|---|---|---|---|---|---|---|---|
| L1 | `machine_mapping/magnetics.py::_load_equilibrium_magnetics_channels` (opens MD.yaml) | `resolve_equilibrium_magnetics_sensors` | `vest_equilibrium_magnetics_channel_definitions` (and its alias), `magnetics_wiring_for_shot`; tests `test_magnetics_geometry_join.py` | private | #2181: old (75b154e31 export) vs new outputs over 47 shots (0-49999, every wiring/fault/revision boundary, 3 machine eras): 0 physics differences; 62 metadata-only (provenance `sensor_registry` key, static-input manifest/EFUND header names) | `removed` in #2181 | `qualified_for_removal` | B1 #2158 | #2181 |
| L2 | `magnetics.py::_load_static_channels` (opens the geometry YAML) | same | `_equilibrium_probe_field_codes`, `_pinned_channel_index` and `_pinned_probe_index` (they pin per-era wiring overrides), `_populate_flux_loop_static`, `_populate_probe_static`, `vfit_mirnov_raw_dynamic`; cross-module: `workflow/magnetics_geometry/verify_wiring.py`; tests `test_magnetics_toroidal_placement.py`, `test_magnetics_geometry_join.py` | private | #2181: old (75b154e31 export) vs new outputs over 47 shots (0-49999, every wiring/fault/revision boundary, 3 machine eras): 0 physics differences; 62 metadata-only (provenance `sensor_registry` key, static-input manifest/EFUND header names) | `removed` in #2181 (verify_wiring.py moved onto the registry in the same PR) | `qualified_for_removal` | B1 #2158, C5 #2164 (verify_wiring) | #2181 |
| L3 | `magnetics.py::_load_names_by_code` (opens table.yaml) | registry names | `_populate_flux_loop_static`, `_populate_probe_static`; cross-module: `raw_daq_sources.py::_wiring_records` | private | #2181: old (75b154e31 export) vs new outputs over 47 shots (0-49999, every wiring/fault/revision boundary, 3 machine eras): 0 physics differences; 62 metadata-only (provenance `sensor_registry` key, static-input manifest/EFUND header names) | `removed` in #2181 | `qualified_for_removal` | B1 #2158 | #2181 |
| L4 | `machine_mapping/utils.py::_load_magnetics_channel_groups` (opens the geometry YAML directly, restates the inboard/side/outboard thresholds) | registry plus `magnetics.py` threshold constants | `utils.py::apply_magnetics_uncertainties` | private | #2181: old (75b154e31 export) vs new outputs over 47 shots (0-49999, every wiring/fault/revision boundary, 3 machine eras): 0 physics differences; 62 metadata-only (provenance `sensor_registry` key, static-input manifest/EFUND header names) | `removed` in #2181 (now reads the registry and the `magnetics.py` threshold constants; group membership identical) | `qualified_for_removal` | B1 #2158 | #2181 |
| L5 | `machine_mapping/magnetics.py` `vest_md_channel_definitions` (alias of `vest_equilibrium_magnetics_channel_definitions`) | `vest_equilibrium_magnetics_channel_definitions` | `machine_mapping/__init__.py` (`_EXPORT_MAP`, `__all__`, lazy legacy map); no internal caller | **public** alias, no distinct behaviour | not needed (same object) | `compatibility_required` | `compatibility_required` (deprecation warning plus a removal milestone, #2168 §11) | E #2168 | |
| L6 | `machine_mapping/magnetics.py` `vfit_md` (alias of `vfit_equilibrium_magnetics`) | `vfit_equilibrium_magnetics` | `machine_mapping/__init__.py` (`__all__`, the lazy legacy map); no internal caller | **public** alias, no distinct behaviour | not needed (same object) | `compatibility_required` | `compatibility_required` (deprecation warning plus a removal milestone) | E #2168 | |

## Processing duplicates

| # | Implementation | Canonical replacement | Consumers | API | Equivalence evidence | Status | Proposed | Owner phase | PR |
|---|---|---|---|---|---|---|---|---|---|
| P1 | `process/magnetics.py::_linear_baseline` (carries the #639 guards) | `process.signal_processing.subtract_baseline(method="polyfit")`, which absorbs the guards | `vest_flux_loop_flux_from_voltage`, `b_field_pol_probe_field` (`vest_b_field_pol_probe_legacy` was removed by #2135) | private | none yet (B3 targets 1e-12) | `validation_pending` | `qualified_for_removal` (after delegation) | B3 #2162 | |
| P2 | `machine_mapping/magnetics.py::_polyfit_baseline` | `subtract_baseline` | **no caller** (confirmed after #2135 and by the audit PR #2174): dead code | private | not needed (dead code) | `qualified_for_removal` | delete in B3 | B3 #2162 | |
| P3 | `machine_mapping/magnetics.py::_linear_baseline_subtract` | `subtract_baseline` | `vfit_plasma_current` (I_p and I_p reference) | private | none yet | `validation_pending` | `qualified_for_removal` (after delegation) | B3 #2162 | |
| P4 | `code/efit/legacy.py::smooth` (verbatim copy) | `process.signal_processing.smooth` | `legacy.py::vest_Halpha_tstart_tend`; re-exported from `vaft.code.efit` | **public** (exported by `vaft.code.efit`) | none yet (expected identical) | `validation_pending` | `compatibility_required` (re-export of the canonical function, no own body) | B3 #2162 | |
| P5 | `plot/soft_x_rays.py::_baseline_correct` | `process.soft_x_rays.sxr_baseline_correction` | `plot_soft_x_ray_signal`, `plot_soft_x_ray_spectrogram`, `plot_soft_x_ray_pattern` | private | none yet | `validation_pending` | `qualified_for_removal` | B3 #2162 | |
| P6 | `plot/backend/recipes.py::_build_interferometer_spectrogram`: inline `butter(4, highpass)` and `filtfilt` | `butterworth_highpass(zero_phase=True)` (to be added) or the existing bandpass | interferometer spectrogram recipe (`recipes.py:2144`) | private | none yet | `validation_pending` | `qualified_for_removal` (inline code) | B3 #2162 | |
| P7 | `process/magnetics.py::_firwin_order` (order = odd `fs·1e-3`) | `fir_filter_coefficients`, only if the coefficients are identical | `mirnov_preprocess_signal` (used by `plot/mirnov.py`, `plot/backend/recipes.py`); the taps are applied through `apply_fir_filter` since #2135 | private | audit PR #2174: the order rule gives coefficients identical to `fir_filter_coefficients` | `validation_pending` (until #2174 merges and B3 adds the equality test) | `qualified_for_removal` (replace with a `fir_filter_coefficients` call) | B3 #2162 | |
| P8 | `machine_mapping/pf_active.py::_legacy_pf_filter` (Hamming `firwin(251)`; since #2135 the taps are applied by `apply_fir_filter(short_record="causal")`, which replaced the local `filtfilt`/`lfilter` fallback; then `uniform_filter1d(10)`) | named `pf_current_filter` in `vaft.process` composed from `fir_filter_coefficients` plus `uniform_filter1d`, with machine values from vest.yaml | `pf_active.py::_read_pf_circuits` (every PF reader) | private | application path canonical since #2135; the tap design (`firwin` here) and the smoothing are still local, and coefficient-equality tests are planned | `validation_pending` | `scientific_specialization` (a diagnostic-specific composition, kept under process ownership; #2168 §6 says not to delete it on name alone) | B3 #2162 | |
| P9 | Deprecated aliases in `process/signal_processing.py`: `vfit_signal_start_end`, `vfit_signal_startend`, `signal_onoffset`, `VEST_CoilCurrentNoiseReduction` (each warns and delegates) | `detect_active_window`, `signal_on_offset`, `vest_coil_current_noise_reduction` | `vfit_signal_start_end`: `test_api_layer_boundaries.py`. `vfit_signal_startend`: `code/efit/legacy.py` defines its **own** function of that name, used by `vest_Halpha_tstart_tend` and exported by `vaft.code.efit`. That is a separate body, not the alias. Others: no internal caller | **public**, in `__all__` "for one deprecation cycle" | not needed (delegating) | `compatibility_required` | `compatibility_required` until a removal milestone is set. The `code/efit/legacy.py` duplicate `vfit_signal_startend` needs its own row in B3 | B3 #2162, E #2168 | |
| P10 | `machine_mapping/utils.py::process_signal` (deprecated wrapper) | `vaft.process.process_signal` | no internal caller (`test_mirnov_spectrogram.py` defines an unrelated local function of the same name) | **public**, deprecated | not needed (delegating) | `compatibility_required` | `compatibility_required` (then removal after the deprecation cycle) | E #2168 | |

## Tests that encode the old arrangement

| # | Test | Replacement | Notes | Status | Proposed | Owner phase | PR |
|---|---|---|---|---|---|---|---|
| T1 | `test/test_magnetics_geometry_join.py` (pins the MD.yaml/geometry positional join and forbids probes in vest.yaml) | `test/test_magnetics_registry.py` (schema, uniqueness, order, 39437/39438 equivalence, equality against frozen copies of the old YAMLs) | must change in the **same** PR that moves the inventory into vest.yaml, or that PR cannot pass. Not core | `removed` in #2181 (replaced by `test/test_magnetics_registry.py`) | `qualified_for_removal` (replaced atomically) | B1 #2158 | #2181 |

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
| an explicit, missing `info_file` raises | skipped | n/a |
| unsafe override rejected (machine facts, unknown keys, unflagged calibration) | skipped | n/a |
| no internal use of deprecated APIs (L5, L6, P9, P10) | skipped | n/a |

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

Two test modules that the consolidation relies on are **not** in the develop
gate. Both still run on the `main` gate and on the push to `develop`.
Timings are local wall clock on the shared 10-core MacBook, `-n 0`, at
7b1fcf192:

| Module | Tests | Wall clock | What it protects | Proposal |
|---|---|---|---|---|
| `test/test_vest_yaml_boundaries.py` | 62 | 33.2 s | revision overlap and shot-range boundaries across vest.yaml diagnostics | **keep out of core.** `core_selection.py` records it as 18 % of the gate when it was measured, and it is shot-range data validation, not a boundary contract. Instead, the B2 (#2159) PR runs it explicitly (per the plan's Verification section). A fast subset (resolver round-trip at the boundary shots 39437/8, 46403/4, 45895/6, 48371/2) is a better core candidate once B2's `resolve_vest_processing_config` exists |
| `test/test_pipeline1_workflow.py` | 19 | 42.9 s | Snakemake parse and DAG dry-run of Pipeline 1 (shells out to `snakemake`) | **keep out of core.** It costs more than `test_vest_yaml_boundaries.py` and shells out to an external tool, which core excludes by policy (`core_selection.py` docstring). Phase C (#2164) PRs run it explicitly. It needs a writable snakemake cache directory (`~/Library/Caches/snakemake` on macOS): 16 of 19 tests fail without one |

## Change log

- 2026-10-25: after #2174 merged, `workflow/architecture_audit/build_inventory.py` (the Phase A generator) is allowlisted in guard (a1) for its three module-level references: it reads the split files as reconciliation evidence, not as runtime configuration. #2158 repoints it at the frozen fixtures in `test/data/legacy_magnetics/`.
- 2026-10-25: rows P1, P2, P7 and P8 updated after #2135 merged (75b154e31). Every guard allowlist still matches exactly.
- 2026-10-25: skeleton and active guards (a)–(c), placeholders (d). PR `architecture/2168-guards-prep`. The module adds about 1 s serial (`-n 0`) to a core gate that took 583 s locally at `-n 2`.
