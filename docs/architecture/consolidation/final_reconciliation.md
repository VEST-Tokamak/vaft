# Consolidation final reconciliation (#2168, umbrella #2147)

This document closes the Phase A audit (#2148). It maps **every** Phase A finding to its final
disposition:

| Disposition | Meaning |
|---|---|
| **implemented** | changed on develop; the PR is named |
| **validated** | evidence that behaviour is unchanged or correct; the evidence is named |
| **retained** | kept on purpose, as a specialization, a separate contract or a compatibility adapter; the reason is given |
| **deferred** | open work, named by its tracking issue |
| **pending PR** | implemented in an open PR, not yet merged |

The sources are, all under `docs/architecture/consolidation/`:

- `config_audit.md` (gaps G1–G17);
- `magnetics_reconciliation.md` (open questions Q1–Q7, consumer map §6);
- `migration_plan.md` (risk-ranked findings §1, work packages §2, refinements §3);
- the generated `inventory.json`;
- `migration_matrix.json`.

Row-level retirement status is in `retirement_register.md`.

## Evidence base

| Evidence | What it shows | Where |
|---|---|---|
| Phase D qualification, #2191 | BEFORE 75b154e31 (post-#2135) vs AFTER 52576480a (the #2189 head; `vaft/` and `workflow/` equal develop f7472664d). Vestserver raw data, 50 shots: every ODS leaf and all 364 k-files per state byte-identical; 266 records, 262 passed, 4 blocked (46403), 0 failed, 0 unexplained; every difference `metadata_only` | `qualification_report.md`, `qualification_results.json` |
| Per-PR equivalence | each B/C PR's frozen-copy or capture comparison (bit-identical, exact or `metadata_only`) | the PR bodies and the tests they add |
| #2182 (B3.12) | approved scientific correction: NaN where a short run could not be filtered, with analytic chirp and synthetic short-run tests. Merged after #2191, which therefore does not cover it (#2191 §10) | `test/test_resample_short_run_alias.py`, `test/test_consolidation_numerics.py` |
| Guards | the architecture invariants, enforced on both develop gates | `test/test_architecture_guards.py` (#2170, #2181, #2196) |

#2191 cases cited below:

| Case | Status | Shots |
|---|---|---|
| wiring-39437-39438 | passed | 39417, 39436–39439, 39446 |
| pf1-gain-45895-45896 | passed | 45891, 45895, 45896, 45919, 45964 |
| impa-archived | passed | 35325, 35376, 35385, 39204, 39923 |
| ip-fl10-46402-46403 | blocked | 46403 |
| daq-46403-46404 | blocked | 46403 |
| pf5-gain-48371-48372 | blocked | physically inconclusive |
| discrepancy-48224 | reference_inconclusive | 48224 |
| vacuum-benchmark-plasma-free | blocked | physical qualification of the benchmark not shown |
| ref-39915-chain | not_tested by rule | 39915 |

## Summary

| Source | Items | Implemented | Validated only | Retained | Deferred | Pending PR |
|---|---|---|---|---|---|---|
| config_audit.md G1–G17 | 17 | 14 (G9, G11, G17 in part) | — | 2 (G14; the G16 values) | 2 fully (G15; G16 sign, #2176) + remainders of G9, G11, G17 (#2204) | — |
| magnetics_reconciliation.md Q1–Q7 | 7 | 2 (Q3; Q6 as documentation) | 2 (Q2; Q7 as behaviour) | — | 3 (Q1, Q4, Q5) + the Q6/Q7 checks (#2203) | — |
| migration_plan.md §1 findings | 10 | 7 (1, 2, 3, 5, 6, 7, 8); 9 and 10 in part | — | parts of 9 (B3.2, B3.7) | 1 (finding 4) + remainders of 1, 3, 10 | part of 9 (B3.11) |
| migration_plan.md §2 work packages (rows of §4, by first disposition) | 42 | 26 (several with a deferred remainder) | 5 | 2 (B3.2, B3.7) | 7 | 2 (B3.9, B3.11) |
| inventory.json triaged rows (70 curated + 18 asset consumers) | 88 | see §5 | | | | |
| retirement_register.md rows | 35 | 15 `removed` | — | 9 (5 specializations or kept contracts: C6, C7, P4, P8, P14; 4 aliases dated for 0.11.0: L5, L6, P9, P10) | 9 with an owner (C5, L7 → #2205; P1, P3, P11, P12, P13, P15, P18 → #2206) | 2 `pending_pr` (P5, P17 → #2197) |

Two counting notes. "Implemented partly" counts once under implemented, and its remainder has an
issue. The §5 inventory counts are by classification, not by disposition, because most curated
rows record a *decision* (canonical, specialization) rather than a change.

Follow-up issues filed by this audit, for actionable items that had no tracking issue:

| Issue | Topic |
|---|---|
| #2200 | regenerate pre-#2135 products |
| #2201 | finding 4 spectrum check |
| #2202 | Q4/Q5 sensor identity |
| #2203 | Q6/Q7 registry facts |
| #2204 | B2 remainders: G9 TF/PF, G17 PF gains, G11, langmuir blocks |
| #2205 | G15 blocks and the dead getters |
| #2206 | Phase E code remainders |
| #2207 | triage of the 72 uncurated call sites |
| #2208 | removal promises and the 0.11.0 deletions |

Existing follow-ups used:

| Issue | Topic |
|---|---|
| #2175 | IMPA 26-tap band edge |
| #2176 | field 171 sign |
| #2192 | dead probes reach EFIT |
| #2193 | PF2/PF5 vacuum misfit |
| #2194 | 46403 baseline window |
| #2195 | no-plasma k-files |
| #2164 | C4 pipeline-2 provenance and the preflight |
| #2166 | Phase D coverage limits |

## 1. config_audit.md gaps G1–G17

| Gap | Finding (short) | Disposition | PR / evidence |
|---|---|---|---|
| G1 | explicit missing `info_file` silently falls back | **implemented**: `_resolve_info_file_path` raises `VestConfigurationError` (directories too) | #2177; `test_vest_config_loading.py`; guard (e) calls every public `info_file` function in machine_mapping/omas/process with a missing path (#2196) |
| G2 | `_deep_merge` shallow; type mismatch wins silently | **implemented**: shares nothing with its inputs, mismatch raises with the dotted path | #2177 |
| G3 | `_policy_document` returns the cached document by reference | **implemented**: deep copy, cache keyed by (path, mtime_ns, size) | #2177 |
| G4 | two caching regimes | **implemented** for `resolve_vest_diagnostic` and the effective-config loader, which read through the mtime-keyed `_policy_document`. **Retained**: `impa._vest_config` stays deliberately uncached (its comment: small file, not hot, a cache would serve stale settings). Not a correctness defect, as the audit said | #2177, #2184 |
| G5 | unknown revision keys merged silently | **implemented**: the shared revision validator rejects keys absent from the base, with an explicit `era_only_keys` mechanism for six tables | #2177; guard (f2) allowlists the one revision merge as validated |
| G6 | IMPA `shot_era_overrides`: defaults 0, no overlap check, first match | **implemented** + **validated**: `impa_shot_era`, inclusive bounds, overlap and a missing `max_shot` refused; era identical to the old loop on shots 0–50000 | #2184; #2191 impa-archived passed |
| G7 | unknown override key → bare `TypeError` | **implemented**: `VestConfigurationError`; `daq_mode`/`fast_sample_rate` machine facts; calibrations need `calibration_override_reason` | #2184; guard (f1) (#2196). Deployment note in #2184: old run configs carrying `daq_mode: legacy` now fail |
| G8 | `VestMagneticsProcessingConfig` unvalidated | **implemented**: `__post_init__` (rates, taps, cutoff < Nyquist) | #2184 |
| G9 | declared sample rates never compared with the timebase | **implemented** at the IMPA low pass, the probe chain and the I_p-reference comparison (report-only); **deferred** for TF/PF `fs` | #2185; #2191: 343 acquisition keys, all consistent (not independent on archived IMPA, §7); TF/PF → #2204 |
| G10 | `mirnov_preprocess_signal` silently skips an out-of-range pass | **implemented**: per-stage `report`, RuntimeWarning without one | #2185, combined with #2179 in #2189 |
| G11 | `vest_processing_provenance` has no stage caller | **implemented by an alternative**: every stage manifest carries `vest_configuration_record`. `vest_processing_provenance` (extended by #2184) still has no stage caller → **deferred** (call, keep as a query, or deprecate) | #2184; #2204 |
| G12 | manifest records effective config only when overridden; no vest.yaml digest or vaft version | **implemented** + **validated** | #2184, #2185, #2187; #2191 §4.2: 12,576 + 4,294 keys, `metadata_only` on 49 shots |
| G13 | dead `0.magnetics.flux_loop` placeholder | **implemented** (removed) | #2181 (register C4) |
| G14 | `VEST_MACHINE_ERAS` is a Python table | **retained**: documented as code-owned | #2184 (register C7) |
| G15 | shot-keyed PF gain blocks restate the live revisions | **deferred**: the only readers are three public getters, which need a deprecation first | #2205 (register C5, L7) |
| G16 | field 171: +0.004529 vs −0.004528986 under the same divide convention | **retained** (both values kept, recorded as a vest.yaml comment) / **deferred** (the sign) | #2181; #2176 |
| G17 | per-channel calibrations not validated | **implemented** for the registry records and `probe_overrides` (finite, non-zero; #2181) and the field classes (#2184); **deferred** for PF `coil_gains` (a missing coil defaults to 0.0) and the PyYAML numeric strings | #2181, #2184; #2204 |

## 2. magnetics_reconciliation.md Q1–Q7 and the consumer map

| Q | Question | Disposition | PR / evidence |
|---|---|---|---|
| Q1 | field 171 sign | **deferred** (same as G16) | #2176 |
| Q2 | labels 223–226 without `_Bz` | **validated**: kept verbatim as registry `name`s | #2181; #2191 wiring-39437-39438: registry = legacy = ODS name on 50 shots |
| Q3 | dead placeholder | **implemented** (removed) | #2181 |
| Q4 | five position-shaped ids (35/46/47/61/62) | **deferred** (owner decision). #2191 §5 supports the 35/47 *wiring* assignment experimentally (39436 ×6.3/×14.3, 39446 ×1.7/×2.1); it does not confirm the ids | #2181 keeps `id_basis: position` + `ladder_candidate`; #2202 |
| Q5 | ODS identifier: DAQ label or physical id | **deferred** (public product-contract change). #2191 §9.2 re-flags it | #2202 |
| Q6 | `sql_table.txt`-only sensors (fields 16, 17, 22, 23, 168) | **implemented** as documentation: the raw-DB name table is a separate contract, tested against all 75 registry labels. **Deferred**: whether those fields record | #2181 (register C6); #2203 |
| Q7 | positional late-baseline loops 9–11 = fl_13–fl_15 | **validated** as behaviour: a test pins the positions. **Deferred**: the donor check | #2181 (`test/test_magnetics_registry.py`); #2203 |

Consumer map (§6), every entry migrated:

- the loaders were replaced, guarded by (a1)/(a2) (#2181);
- the public adapters stayed byte-identical (#2181, 47 shots; #2191, 50 shots, 0 mismatches);
- k-file grouping comes from the registry family (C2, #2188; 194/194 k-files identical);
- `verify_wiring.py` works by id (#2181);
- `build_zeff.py` resolves `fl_10` from the registry (#2188);
- `test_magnetics_geometry_join.py` was replaced (#2181);
- `verify_dist.py` and packaging were updated (#2181).

`vaft/database/raw.py` keeps its `sql_table.txt` use by decision (Q6).

## 3. migration_plan.md §1 risk-ranked findings

| Rank | Finding | Disposition | PR / evidence |
|---|---|---|---|
| 1 | causal FIR, 0.5 ms lag (probes, TF, I_p reference) | **implemented** (zero-phase `apply_fir_filter`) and **validated** as the BEFORE state of #2191 (§7: TF-coupling shift recorded, alignment shots stay certified). **Deferred**: regenerating every pre-#2135 product, including the 1,011 #1731-override products 43685–46342 (owner approval) | #2135; #2200 |
| 2 | 75 sensors in three positional lists | **implemented**: the vest.yaml registry; the split files retired to `reference_only` fixtures | #2181; #2191 wiring-39437-39438; guards (a)/(c) |
| 3 | no stable physical id | **implemented**: a separate registry `id`; ODS labels unchanged by design. The misleading `MagneticsWiring` docstring was corrected | #2181; Q4/Q5 → #2202 |
| 4 | flux loops, I_p, diamagnetic flux decimated by `np.interp` with no anti-alias filter | **implemented** for the comment half only (B3.8). **Deferred**: the spectrum measurement, which #2191 did not do (§10) | #2179; #2201 |
| 5 | declared sample rates never checked | as G9 | #2185; TF/PF → #2204 |
| 6 | provenance gaps | as G11/G12 | #2184, #2185, #2187; #2204 |
| 7 | loader safety (G1–G3) | **implemented** | #2177 |
| 8 | revision/IMPA/override/dataclass validation (G5–G8) | **implemented** | #2177, #2184 |
| 9 | duplicate helpers | **implemented** B3.1, B3.3–B3.6, B3.8, B3.10 (§4); B3.2 and B3.7 **retained**; B3.11 **pending PR** | see §4 |
| 10 | stale second authorities | placeholder removed (#2181); `sql_table.txt` documented (#2181); `VEST_MACHINE_ERAS` documented (#2184); field 171 → #2176; PF shot blocks → #2205. New from #2184: the unread `0.diagnostics.langmuir_probe_mid/upper` blocks → #2204 | as listed |

Findings recorded for their owners, outside the consolidation, are **not** consolidation work.
They are listed here so they are not lost:

- zero σ on fitted profiles;
- σ(T_i) not stored;
- ADF11 extrapolation;
- the hard-coded electron-only fit coordinate;
- the Pipeline 3 data path.

None of them was changed by #2147, and none got an issue from this audit, because it is not the
owner.

## 4. migration_plan.md §2 work packages

| Package / slice | Disposition | PR / evidence |
|---|---|---|
| B2a loader safety | **implemented**, `equivalent` (248 shots × all tables, 36,245 leaves identical) | #2177 |
| B1 registry | **implemented**, `metadata_only`, atomic (registry, rewiring, YAML retirement, guards) | #2181; #2191 wiring case. #2158 is still open: "Closes #2158" does not fire on a develop merge, so it needs a manual close |
| B2b effective configuration | **implemented**, `metadata_only` (72,501 leaves identical) | #2184, #2185; remainders → #2204 |
| B3.1 linear baseline | **implemented**, bit-identical; the private delegating aliases remain | #2178; aliases → #2206 |
| B3.2 smoothing (`legacy.smooth`) | **retained** (NaN locality differs); retires with the H-alpha detector | #2179; #2206 |
| B3.3 integration | **implemented**, exact | #2179 |
| B3.4 Mirnov design | **implemented**, coefficients identical | #2179 |
| B3.5 PF filter | **implemented**, bit-identical (`pf_current_filter`) | #2180; #2191 pf1-gain-45895-45896 |
| B3.6 plot-recipe Butterworth | **implemented** (`butterworth_highpass`) | #2179; guard (i) |
| B3.7 plot SXR baseline | **retained** in #2179 (window semantics differ), superseded by B3.11 | #2179; #2197 |
| B3.8 comments | **implemented** | #2179 |
| B3.9 API ownership matrix | **pending PR** | #2198 (open) |
| B3.10 box average | **implemented**, equivalent | #2183; #2191: 364 k-files per state identical |
| B3.11 SXR baselines | **pending PR** (one owner `sxr_offset_baseline`, bit-identical to frozen copies) | #2197 (open) |
| B3.12 short-run anti-alias | **implemented** (approved scientific correction); the #2172 strict xfail became a positive test in this PR | #2182 |
| C1 channel status, manifests, fingerprints | **implemented** (`_record_unusable_magnetics_channels`, approved_correction/`metadata_only`; manifest records) | #2187, #2184, #2181 |
| C2 k-file families | **implemented**, 194/194 k-files identical | #2188 |
| C3 EFUND identity | **validated**: namelist text and rsi/zsi/xmp2 arrays 0 differences over 3 eras, headers `metadata_only`; no EFUND/EFIT run in #2191 because the inputs are byte-identical | #2181; matrix row now `done` |
| C4 pipeline-2 provenance | **deferred**: the pipeline-2 builders and `build_eddy_ods` do not yet write the `vest_configuration` record (matrix rows `not_started`) | #2164 |
| C5 CLI / verify_wiring / build_zeff / notebooks | **implemented** for CLI (through `resolve_magnetics_processing`), verify_wiring and build_zeff. Notebooks: no alias or retired-file use remains (guard (g) does not scan notebooks; grep clean) | #2184, #2181, #2188 |
| C preflight `required_fields` | **deferred**: still workflow-local after #2185 | #2164 |
| D infrastructure (case manifest, analytic tests, negative controls) | **implemented** | #2172, #2191 |
| D qualification | **validated** (no unexplained regression) | #2191 |
| D: spectrum check (finding 4), Q6/Q7 checks | **deferred** | #2201, #2203 |
| D: vacuum benchmark physical qualification, 41524/41672/42962 chains, corrective-kinetic, analytic-preprocessing, H2 fingerprint properties, cross-environment baseline tolerance (#2191 §4.3), benchmark sensitivity (§9.5), EFIT ≠ plasma (§9.7) | **deferred** to Phase D's remaining scope | #2166 (open) |
| D findings outside the consolidation (§9.1, 9.3, 9.4, 9.6) | **deferred** to their own issues | #2192, #2193, #2194, #2195 |
| D: TF-coupling shift after #2135; 48224 | **validated** (recorded); 48224 `reference_inconclusive` (#2191 §8 rules the #2135 delay out as the explanation) | #2191 |
| E guards: registry schema/uniqueness | **implemented** | #2181 |
| E guards: no reader of the retired YAML (AST) | **implemented** | #2170 (a1/a2) |
| E guards: no second sensor inventory (YAML) | **implemented** for YAML. **Retained** as YAML-only: a Python table of r/z records is not scanned, but the layer guard and (h) keep the generic layers free of machine tables | #2170 (c) |
| E guards: resolver path and override safety | **implemented** | #2196 (e), (f1), (f2) |
| E guards: layer contract incl. "machine_mapping designs no filters outside an allowlist" | **implemented** (grimp layer guard; filter-design guard with two allowlisted donor designs) | #2170 (b), #2196 (i); designs → #2206 |
| E guards: deprecated-API usage | **implemented** | #2196 (g) |
| E guards: vaft.process configuration reads (#2168 brief) | **implemented** (one allowlisted legacy window ladder) | #2196 (h); ladder → #2206 |
| E retire `MM._polyfit_baseline` | **implemented** (removed) | #2178 |
| E retire `VEST_CoilCurrentNoiseReduction`, `process_signal`, `signal_on_offset`, `vest_coil_current_noise_reduction` | **implemented** for the deprecated aliases (dated vaft 0.11.0); the canonical-but-legacy names → **deferred** | #2199; #2206, #2208 |
| E retire G15 blocks + dead getters | **deferred** (deprecation first) | #2205 |
| E retire `correct_flux_loop` / `vest_signal_onoffsetpeak` | **deferred** | #2206 |
| E `tdst_gate.sh` parity | **validated**: `tdst_gate.sh` runs `pip install -e ".[dev]"` and `pytest -m "core and not perf"`, the same as Package CI `core-test`, and the same `python -m build` + `test/verify_dist.py` as `package`. `test/test_core_selection.py` (`test_the_cluster_gate_runs_the_same_steps`) pins the parity. The workflows were not edited | `.github/scripts/tdst_gate.sh`, `.github/workflows/package-ci.yml` |
| E `verify_dist.py` + wheel smoke | **validated**: the dist must contain `vaft/machine_mapping/vest.yaml` and `magnetics_registry.py` and must not contain the three retired YAMLs (`verify_dist.py` lists them); run by both gates | #2181; the #2196/#2199 gate runs |
| E `test_architecture_audit_inventory.py` in core | **implemented** | #2174 |
| E `test_vest_yaml_boundaries.py` in core | **implemented**: 33 s before #2177's parse cache, 1–2 s serial now (local and vestserver) | #2196 |

§3 refinements 1–10 were each acted on as above, or were confirmations; §3.9's "not in the plan"
items are G14/G15/Q6/Q7/finding 4. The one addition that has not been rechecked: §3.9 "diagnostics
QA writes validity (it is not report-only)" is now visible in #2191 §9.1, where dead probes pass
QA. It is tracked by #2192.

## 5. inventory.json

Regenerated by this PR (`build_inventory.py --check` clean). 468 rows. Classification counts:

| Classification | Before (develop aeec6d91a) | After this PR | Change |
|---|---|---|---|
| unverified | 382 | 382 | — |
| intentional_specialization | 35 | 37 | +2: the two `sql_table.txt` readers |
| legacy | 26 | 8 | −18 |
| canonical | 23 | 39 | +16 |
| duplicate | 2 | 2 | — |

The 18 asset-consumer rows were stale generator labels. Since #2181, the callers of the public
magnetics adapters consume the vest.yaml registry. 16 are now `canonical` (`resolved_by` #2181); the
two `sql_table.txt` readers are a kept contract (Q6).

Dispositions by group:

- **canonical (39) and intentional_specialization (37):** each is a recorded decision, **retained**
  with the reason in its row. The do-not-touch list of migration_plan.md B3 is among them.
  `resolved_by` names the PR wherever a row changed.
- **legacy (8):**
  - `legacy.smooth`: retained (B3.2), retires with the H-alpha detector;
  - `correct_flux_loop`, `vest_signal_onoffsetpeak`: deferred;
  - `MM._linear_baseline_subtract`, `process.magnetics._linear_baseline`: delegate since #2178, alias retirement deferred;
  - `signal_processing.process_signal`, `signal_on_offset`, `vest_coil_current_noise_reduction`: deferred.

  All eight are owned by #2206 (register P1, P3, P4, P12, P13).
- **duplicate (2):**
  - `MM.soft_x_rays.vfit_soft_x_rays_dynamic`: pending PR #2197 (B3.11);
  - `process.magnetics._calibrated_signal`: the zero-calibration half is implemented (#2181); keep-or-route is deferred to #2206 (register P17, P18).
- **unverified preprocessing call sites (76):**
  - 72 still say "triage in the owning B3/C slice"; 71 are statically reached in production, 44 of them resampling. Pipeline-1 products are byte-identical across the consolidation (#2191), but the call sites were never triaged → **deferred**, #2207.
  - The other four were triaged in Phase A: two plot-time transformations (retained); one study script; and `MM.impa._resample`, whose IMPA low-pass rate check landed in #2185 while `_resample` itself stays with #2175/#432.
- **derived_numerics_not_triaged (306):** outside the measured-signal consolidation by Phase A's scope decision. Not dispositioned here.

Still open in the inventory text: `_interpolate_signal` and `vest_equilibrium_magnetics_detailed`
say "D: measure" → #2201.

## 6. migration_matrix.json

Statuses refreshed by this PR on develop aeec6d91a. The blockers named in the matrix (#2178–#2185,
#2182) have all merged.

| Status | Before | After |
|---|---|---|
| done | 1 | 16 |
| blocked | 10 | 0 |
| in_review | 5 | 0 |
| provenance_pending | 8 | 0 |
| not_started | 1 | 9 |
| no_change_needed | 11 | 11 |
| not_affected | 17 | 17 |
| retired | 1 | 1 |

Each `done` row names its merged PR and, where it applies, its #2191 case. The nine `not_started`
rows (eight C4 provenance builders and the preflight) are #2164's remaining scope.

## 7. Items that need an owner action, not code

- **Close #2158 by hand.** #2181 said "Closes #2158", but closing keywords only fire on the default branch.
- **Merge the pending PRs.** #2197 (B3.11) and #2198 (B3.9) are open. When they merge, register rows P5/P17 become `removed`, and the canonical owners named in the register should be checked against `api_ownership.json`.
- **Approve or decline the regeneration** of pre-#2135 products (#2200).
- **Decide the identity questions** Q4/Q5 (#2202).
