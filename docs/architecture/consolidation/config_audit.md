# VEST configuration-resolution audit

Phase A audit for #2148 (scope C), part of #2147. Develop is at `75b154e31` (after #2135);
line numbers refer to it. `MM` = `vaft/machine_mapping`, `VU` = `vaft/omas/vest_upstream.py`,
`Y` = `MM/vest.yaml`.

## 1. How a value reaches the processing today

`vest.yaml` has three revision mechanisms. They resolve differently.

| Mechanism | Where | Resolver | Bounds | Overlap | Unknown keys | Merge |
|---|---|---|---|---|---|---|
| **A. `revisions` lists** inside a block (`from_shot`/`to_shot`) | `0.diagnostics.<name>.revisions` and nested blocks, e.g. `equilibrium_magnetics.processing.{window,wiring,known_faults}` and `plasma_current.processing.*` | `resolve_shot_revisions[_with_provenance]` (`MM/utils.py:193/212`) via `_match_revision` (`:133`) | inclusive; at least one bound; `from_shot ≤ to_shot` checked | **rejected** whenever two revisions overlap (`:170-183`), and again if a shot matches two (`:185`) | **accepted silently**: any key merges in (`:233-240`) | exactly one revision over the base, **no chaining** (the vest.yaml comment at the `window` block says so); lists are replaced, not extended |
| **B. shot-keyed top-level blocks** (`20259:`, `38110:`, `38361:`, `38401:`) | `Y:1588-1618` (`pf_active.channel.N.gain` only) | `_shot_block` (`MM/utils.py:1045`), used by `resolve_impa_config` (`MM/impa.py:118`), `resolve_langmuir_probe_config` (`MM/langmuir_probes.py:131`) and the dead getters `raw_database_info` / `get_diagnostic_info` / `get_static_info` (`MM/utils.py:1061/1148/1179`) | the exact shot only | n/a | accepted | `_deep_merge(default, shot_block)` |
| **C. IMPA `calibration.shot_era_overrides`** (`min_shot`/`max_shot`) | `Y` `impa.calibration` | three separate loops in `MM/impa.py:169-176, 183-186, 197-204` | inclusive; a missing bound **defaults to 0**, so an entry without `max_shot` never matches | **none**: the first match wins (`break` / `return`) | accepted | per field (channels, bz_channels, gain) |

Shot-independent policy blocks live outside `0:` and are read through
`_policy_document` (`MM/utils.py:76`):
- `diagnostics_time_policies`, read by `resolve_diagnostics_time_policies` (`:413`);
- `plasma_timing`, `discharge_timing` and `plasma_features` (`:734/832/929`).

Workflow overrides enter at two points:
- the `vest.magnetics.processing` block of Pipeline 1 `config.yaml`
  (Snakefile:605), which goes to `magnetics_processing_for_shot` (VU:682);
- `diagnostics.tstart/tend/dt`, which retune only the default analysis window
  (`resolve_diagnostics_time_policies` `analysis_override`).

**Precedence, as resolved today:**
1. explicit workflow override;
2. the matching revision;
3. the block's base values.

Shot-keyed blocks (B) can replace any of these for the resolvers that read them.
Python defaults sit underneath everything:
- dataclass defaults such as `VestMagneticsProcessingConfig`;
- the IMPA `processing.get("sample_rate", 25_000.0)` fallback (`MM/impa.py:214`).

They apply when a key is absent, **with no record that a default was used**.

## 2. Gaps

Every gap lists its location, its effect, and the B2 change that closes it. All
locations were confirmed in the develop source.

| # | Gap | Location | Effect on scientific meaning | B2 remedy |
|---|---|---|---|---|
| G1 | Explicit `info_file` that does not exist **silently falls back** to the packaged vest.yaml | `MM/utils.py:89-103` (final `return package_data_path("vest.yaml")` at :103) | a mistyped campaign config runs with production policy and records nothing | missing explicit path → `VestConfigurationError` |
| G2 | `_deep_merge` copies only the top level (`merged = dict(base)`, `:109`). Untouched nested mappings stay **shared with the source document**. A type mismatch (mapping vs scalar) lets the override win without notice (`:107-108`). | `MM/utils.py:106-115` | combined with G3, a caller that mutates a resolved block mutates the cached document for every later shot | deep copy; documented rules for null/list/type mismatch (mismatch raises) |
| G3 | `_policy_document` returns the `lru_cache`d parsed document **by reference**. The docstring says "callers must not mutate"; nothing enforces it. | `MM/utils.py:70-86` | same as G2 | `MappingProxyType` or deepcopy |
| G4 | Two caching regimes. `resolve_vest_diagnostic` re-parses the whole file on **every call** (`load_yaml`, `:313`), and so does `resolve_impa_config` (`_vest_config`, `MM/impa.py:101-104`, deliberately uncached). The policy resolvers cache by mtime. | `MM/utils.py:313`, `MM/impa.py:101` | not a correctness defect, but two paths can see different files mid-edit | one loader (`effective_config`) with mtime-keyed cache |
| G5 | Revision **unknown keys** are merged silently. A typo such as `probe_baseline_ned` adds a key nothing reads, and the base value stays in force. | `MM/utils.py:233-240` | a shot-era correction can be a silent no-op | reject keys absent from the base (shared revision validator) |
| G6 | IMPA `shot_era_overrides`: `min_shot`/`max_shot` default to 0, there is no overlap check, the first match wins, and the logic is repeated in three loops | `MM/impa.py:169-170, 183-184, 197-198` | overlapping or half-bounded entries resolve by list order | adapter that normalises to inclusive bounds and rejects overlap (YAML schema unchanged) |
| G7 | `magnetics_processing_for_shot`: an unknown override key raises a bare `TypeError` from `dataclasses.replace` | VU:732 | fails, but with an unhelpful message, after the guards at VU:699-731 | unknown key → `VestConfigurationError`; generalise the shadowed/grid guards |
| G8 | `VestMagneticsProcessingConfig` has **no `__post_init__` validation**: rates, taps and cutoffs are unchecked, cutoff ≥ Nyquist included | `vaft/process/magnetics.py:268` | an override can request an impossible filter; `firwin` then raises late or designs something unintended | `__post_init__` validates rates/taps and cutoff < Nyquist |
| G9 | Declared sample rates are **never compared** with `np.diff(time)`: IMPA `sample_rate` (default 25 kHz, `MM/impa.py:214`; the code comment at `:251` names #425), magnetics `fast_sample_rate` 250 kHz (`vaft/process/magnetics.py:281`), the TF/PF `fs` 25 kHz in Y, and the Ip reference comparison `sample_rate` | listed | a filter is designed at the wrong rate when the DAQ rate differs; #432 is the open instance | `vaft/process/acquisition.py:check_declared_sample_rate` returning `consistent \| deviation \| invalid \| not_verifiable`; report-only for IMPA |
| G10 | `mirnov_preprocess_signal` **silently skips** a high or low pass whose cutoff is ≤ 0 or ≥ Nyquist | `vaft/process/magnetics.py:1167-1174` | the record is returned unfiltered with no status | explicit status in provenance |
| G11 | `vest_processing_provenance` (`MM/provenance.py:161`) exists but **no stage calls it**. The static checker finds no caller in `vaft/omas` or `workflow`. | — | per-shot effective policy is recoverable only by re-running the resolver | call from static/diagnostics/IMPA builders (C1) |
| G12 | The diagnostics manifest records `vest_magnetics_processing_effective` **only when an override was passed** (otherwise `None`, VU:1134-1137). It records **no vest.yaml digest and no vaft version**. Only IMPA hashes its resolved block (`_impa_configuration_sha256`, VU:1183). | VU:1125-1140 | a product cannot be tied to the policy that made it after vest.yaml changes; this is why the 1,012 stale #1731 products had to be found by their override text | always record effective config + fingerprint + vest.yaml sha256 + vaft version |
| G13 | **Dead `0.magnetics.flux_loop` placeholder** (r 0.45, z 0, one channel `FL1`) | `Y:5-18` | the only readers are the getters `get_static_info`, `get_diagnostic_info(…, "magnetics")` and `raw_database_info(…, "magnetics")` (`MM/utils.py:1179/1148/1061`), which have no production caller. It looks authoritative, and #843 shows what a stale authoritative-looking table costs. | delete in B1 |
| G14 | Machine eras for static geometry are a **Python table**, `VEST_MACHINE_ERAS`, not vest.yaml | VU:102-106 | a second home for machine history | move to vest.yaml or document as code-owned in B2 |
| G15 | **Shot-keyed PF gain blocks** (mechanism B) restate gains that the live `pf_active.processing.revisions` already own as **ranges**. For example `20259: pf_active.channel.0.gain: -50000` corresponds to the revision `from_shot 20259 / to_shot 38109, coil_gains {0: -5.0e4, …}`. Production reads only the revisions. | `Y:1588-1618`, readers `MM/utils.py:1061/1148/1179` (no production caller) | harmless today; misleading second authority (#843 pattern) | delete in E after confirming no external reader; `raw_database_info` / `get_diagnostic_info` are public exports (and a test uses the latter), so E deprecates them first |
| G16 | `toroidal_mirnov_reference` restates field 171 with gain **+0.004529**; MD.yaml gives **−0.004528986**. Both are applied as **divide**: `_fluctuation_mirnov_gains` (`MM/magnetics.py:489-501`) → `vaft/plot/mirnov.py:152-160` → `mirnov_preprocess_signal(amplifier_gain=…)`, which returns `processed / amplifier_gain`. | `Y` `toroidal_mirnov_reference.channels[3]`; `vaft/data/geometry/MD.yaml` index 36 | genuine sign conflict. Field 171 is no longer published as a phase reference (`MM/magnetics.py:671-705`, #825 closed), but the legacy alias `MagneticFieldProbe_C2-05_Bz:phase_reference` keeps the +0.004529 gain, so preprocessed Mirnov plots of pre-fix replicas get the opposite sign to the equilibrium chain | keep both values; follow-up issue to settle the alias sign (see `magnetics_reconciliation.md` Q1) |
| G17 | Calibration validation covers only a top-level `calibration` (`resolve_vest_diagnostic`, `:325-327`). Per-channel gains (`pf_active.processing.coil_gains`), MD.yaml calibrations and `wiring.probe_overrides[].calibration` are **not validated** (zero or NaN accepted). | `MM/utils.py:325-327`, `MM/magnetics.py:802-832` | a zero calibration divides by zero in `_calibrated_signal` (flux loops, `vaft/process/magnetics.py:374`) and `_calibration_gain` (probes since #2135, `:513`) | registry validation (B1: nonzero, finite) + B2 field classes |

## 3. Recommended contract (input to B2, #2159)

1. **One loader.**
   - `effective_config._load(info_file)`.
   - An explicit path must exist, with no fallback.
   - Cached by `(path, mtime_ns)`.
   - Returns an immutable view.
2. **One revision validator**, shared by mechanisms A and C (C through an adapter):
   - inclusive bounds;
   - overlap rejected;
   - unknown keys rejected against the base keys;
   - mechanism B is kept only for IMPA/Langmuir and flagged in provenance;
   - G15 blocks are deleted.
3. **`ResolvedVestConfiguration(shot, diagnostic, effective, provenance, fingerprint)`**, frozen:
   - per-key provenance: `source`, `revision_index`, `revision_bounds`, `overridden`, `defaulted`;
   - `fingerprint` = sha256 of canonical JSON of the effective values only (sorted keys, float `repr`);
   - `source_sha256` of the file, kept separately.
4. **Field classes.**

   | Class | Examples | Overridable from a workflow? |
   |---|---|---|
   | Machine facts | sensors, wiring, `daq_mode` | No |
   | Calibrations | — | Only with an explicit flag |
   | Processing / analysis parameters | — | Yes |

   An unknown key raises `VestConfigurationError`.
5. **Observed acquisition.** Wherever a declared rate designs a filter, check it
   against the time vector (G9) and record the status in provenance.
6. **Manifests always record:**
   - the effective config and its fingerprint;
   - the vest.yaml sha256;
   - the vaft version.

## 4. Evidence against the plan

- **Field 171 sign (G16).** The plan's "opposite sign gain" is **confirmed** as a
  genuine conflict, because both copies divide. Its runtime reach is narrower than
  "opposite calibration", though: only the legacy `:phase_reference` alias used to
  preprocess Mirnov plots of pre-#825 replicas.
- **G15 is new.** The plan's resolver list has no item for it.
- **G14 is new.** The plan has no item for the `VEST_MACHINE_ERAS` table either.
