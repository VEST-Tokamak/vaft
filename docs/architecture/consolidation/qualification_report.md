# Consolidation qualification report (Phase D, #2166 — D2)

Status of this report: **evidence for one code pair**. It qualifies the PR #2189 head against
develop before the consolidation code, on raw data from the VEST FileDB. It does not cover PR
#2182, which is held for owner approval. It also does not re-qualify the packaged-sample
baselines that other lanes own.

Machine-readable companion: [`qualification_results.json`](qualification_results.json).
Case statuses: [`test/data/consolidation/case_manifest.yaml`](../../../test/data/consolidation/case_manifest.yaml).
The bulk evidence stays on vestserver under `/home/user1/runs/consolidation-d2/` (23 GB, listed in §11).

## 1. Verdict

| Question | Answer | Evidence type |
|---|---|---|
| Did #2189 change any scientific product on the regression portfolio? | **No.** Every ODS leaf of every stage product was byte-identical: static on 50 shots, and diagnostics, eddy and constraints on the 49 shots that ran. All 364 generated k-files were byte-identical too. | historical regression |
| Did it change anything else? | Yes, but only the manifest/provenance records: B2 effective-config and acquisition records, C missing-channel reasons, and the B1 registry input record. The IMPA result also gained acquisition provenance. All of these are `metadata_only`. | structural |
| Unexplained regressions | **None found.** | — |
| Did the new resolver keep the sensor mapping across the wiring/DAQ boundaries? | Yes. On all 50 shots, the registry, the frozen legacy YAMLs (the registry's donor) and the produced ODS agree for all 75 sensors. This is migration fidelity, not independent geometry evidence (#2166 C2). The raw-dump test confirms that the code applied the calibration it was given (3,570/3,570). It identifies the source field code only where the test discriminates (§5). | historical regression + processing self-check |
| Is the 39437/39438 wiring assignment physically right? | Supported. Applying the other era's wiring degrades the vacuum fit of the swapped probes on 39436 (pre, ×6.3/×14.3) and 39446 (post, ×1.7/×2.1). | experimental (vacuum response) |
| Does the vacuum benchmark respond to injected configuration errors? | Sign flips: detected on every evaluable informative case. Signal swaps and the era-wiring swap: mostly. A 0.20 m probe move: **not reliably** (§6.3). | self-consistency of the benchmark |

**Blocker for #2189: none.** Five cases are `blocked` in the manifest. In every one, the case
question cannot be closed with the available data because of pre-existing data or behaviour that
is identical at both code states. That is a hold on the case, not on the #2189 migration: no
output differs between the states, so #2166 G's "hold affected migration" does not apply.
Several pre-existing scientific findings came up along the way (§9); they need owners.

## 2. Code states, inputs, isolation

| | SHA | What |
|---|---|---|
| BEFORE | `75b154e318e81e831afe74c4d4e1084b52f16acc` | develop after #2135 (zero-phase FIR), before #2177/#2181/#2189 |
| AFTER | `52576480adc1248d74991ee547a93b02986fc535` | PR #2189 head (open, not merged at run time). It contains develop `0ae3b2bfe` (#2177 strict config, #2181 registry) plus B2 #2159, B3 #2162 and C #2164 |

- **Worktrees.** `~/work/d2-before` and `~/work/d2-after` were created from private refs
  (`refs/consol/d2-*`). The import shims strip the editable finder, and the parent and every
  child launched with the stage environment assert `vaft.__file__` inside the worktree.
  `~/git/vaft`'s checkout was not touched.
- **Environment.** vestserver, conda `vaft-ml`: python 3.14.7, numpy 2.5.3, scipy 1.18.1,
  omas 0.94.2, BLAS threads 2, `nice -n 15`, at most 8 processes.
- **Raw.** `/srv/vest.filedb/raw/<shot>/vest_<shot>_daq_raw.json.gz` was read-only. The sha256
  of each raw file is in the JSON, per shot. 39915 used its packaged raw (the server copy has a
  different sha256: `4ce6884d…` vs packaged `ac94261a…`).
- **Config.** Server shots used the production
  `workflow/automatic_pipeline_1_routine_data_processing/config.yaml` (`vest: {}`, so the era
  policy decides). Both states consumed the same file, the AFTER copy (sha256 `99960758…`); the
  BEFORE file differs from it only in comments and parses identically.
  39915 used its packaged `pipeline-config.yaml`.
- **Stage chain.** Both states ran the same capture logic: `capture_baseline.py` from the AFTER
  tree (sha256 `5a4c5510…`). The driver `d2_capture.py` loads it once and re-points
  `REPO_ROOT`/`PIPELINE_DIR` at each worktree and `SAMPLES_DIR` at a symlink farm. The stage
  scripts run as the Snakefile invokes them.
- **HSDS.** Never contacted. `HS_ENDPOINT` points to an unroutable address,
  `HOME`/`VAFT_FILEDB_DIR` are redirected into the output root, and `replicate_to_hsds.py` is
  never called. Nothing was written under `/srv`.

## 3. Case table

Availability: raw was present on vestserver for every requested shot (§11 lists them).

| Case | Shots run | Evidence | Regression result | Physical result | Manifest status |
|---|---|---|---|---|---|
| wiring-39437-39438 | 39417, 39436, 39437, 39438, 39439, 39446 | regression, mapping fidelity, vacuum, controls | equivalent / metadata_only | Mapping unchanged (§5). The era-wiring swap is detected on 39436 and 39446, the experimental support for the assignment. Benchmark informative on 39417/39436/39446 | **passed** |
| ip-fl10-46402-46403 | 46396–46403 | regression | equivalent on 46396–46402 | not_tested: no plasma after the boundary in this window, and 46403 fails | **blocked** (46403) |
| daq-46403-46404 | 46403–46410 | regression, identity, acquisition check | equivalent on 46404–46410 | Identity and acquisition consistent on 46404. Dead probes found (§9.1) | **blocked** (46403) |
| pf1-gain-45895-45896 | 45891, 45895, 45896, 45919, 45964 | regression, vacuum, controls | equivalent / metadata_only | Benchmark informative on both sides, including PF1-energised 45964 (§6) | **passed** |
| pf5-gain-48371-48372 | 48366–48376 | regression, vacuum | equivalent / metadata_only | not_tested (inconclusive physical evidence): the PF5-only shots before the boundary are uninformative (§9.3) | **blocked** |
| discrepancy-48224 | 48224 | regression, server k-file comparison | equivalent / metadata_only | reference_inconclusive: today's chain vs the archived server k-file (§8) | **blocked** |
| vacuum-benchmark-plasma-free | 48 cases + 2 rejected | vacuum, controls | per-sensor rows identical | 17 informative cases, only 2 in the 45968+ era. The declared move control misses its threshold on 8/17 (§6.3), so physical qualification of the model is not shown | **blocked** |
| impa-archived | 35325, 35376, 35385, 39204, 39923 | regression | equivalent / metadata_only | Alignment shots stay certified | **passed** |
| ref-39915-chain | 39915 (packaged raw) | regression (cross-check) | equivalent / metadata_only | Controls detected except the Z −0.42 half of the era swap (×0.93) | not_tested (kept: the manifest test forbids a pass on a baseline case) |
| ref-41524/41672/42962-chain, corrective-kinetic, analytic-preprocessing | — | — | not run here | — | not_tested |

Counts over the 266 per-(shot, layer) records in the full aggregate:
- 262 passed: 50 shots × 4 stages minus the 3 blocked stage records, plus 16 k-file and 49
  identity records;
- 4 blocked: the diagnostics, eddy and constraints stages of 46403, and the 46403 raw identity
  test, which has no ODS data to test;
- 0 failed, 0 unexplained_regression.

Of the stage records, 99 are `metadata_only` and 98 are `equivalent` with no manifest change at
all. Constraints products are 2-leaf `EFIT not applicable` markers on vacuum shots.

Case-level counts (14 manifest cases):
- passed: 3 (wiring-39437-39438, pf1-gain-45895-45896, impa-archived);
- blocked: 5 (ip-fl10, daq-46403-46404, pf5-gain, discrepancy-48224, vacuum-benchmark);
- failed: 0;
- not_tested: 6.

Among the blocked cases, the inconclusive evidence is `reference_inconclusive` (an
historical/new difference) only for 48224. The others are `not_tested` on their physical side.

Coverage by machine era (shots captured / with a raw identity test / vacuum cases / informative):
- pre-43017: 11 / 11 / 11 / 7;
- 43017–45967: 11 / 11 / 10 / 8;
- 45968+: 28 / 27 / 27 / 2.

Coverage by diagnostic family: the stage products cover all magnetics (64 probes, 11 flux
loops, Ip, diamagnetic loop), pf_active, tf, barometry, ec_launchers, langmuir_probes and
spectrometer_uv. Only magnetics and PF have physical tests here (§5–§6).

## 4. Historical regression (BEFORE vs AFTER)

### 4.1 Products

`d2_compare.py` hashed every leaf of every stage product and loaded the products of both states
to measure the size of any mismatch. None was found:

| Stage | Shots captured at both states | Leaves per product | Leaves changed | Product bytes |
|---|---|---|---|---|
| static | 50 | 12,914 (pre-43017) / 13,049 | 0 | identical |
| diagnostics | 49 (46403 fails at both) | 4,903–5,313 | 0 | identical |
| eddy | 49 (43026's product is a 3-leaf skip marker: PF6/9/10 not acquired, at both) | 9,517 / 9,667 | 0 | identical |
| constraints | 49 | 2 (not applicable) up to 16,725 | 0 | identical |
| k-files (default + routine presets) | 16 shots with constraint slices, 364 files per state | — | 0 | identical |

### 4.2 Every difference, classified

| Difference | Where | Size | Class | Responsible PR |
|---|---|---|---|---|
| `configuration.vest_configuration.*` added (effective config with per-component fingerprints) | diagnostics manifest, 49 shots | 12,576 keys | metadata_only | #2159 (B2) via #2189 |
| `configuration.vest_magnetics_processing_{effective,fingerprint,provenance}` added | diagnostics manifest, 49 shots | 4,294 keys | metadata_only | #2159 |
| `channel_status.<ch>.acquisition.*` added (declared vs observed rate) | diagnostics manifest, 49 shots | 343 keys | metadata_only | #2159 (acquisition checks) |
| `channel_status.magnetics.missing_channel_reasons` added (`not_archived`) | 39417, 39915, 41445, 41452 | 20 keys | metadata_only | #2164 (C, missing channels) |
| `input.magnetics_calibration` / `input.magnetics_geometry` (MD.yaml, geometry YAML) replaced by `input.magnetics_registry` (`vest.yaml:…sensors`, version 2409) | static manifest, 50 shots | 300 keys (6 per shot) | metadata_only | #2181 (B1, #2158) |
| `result.provenance.acquisition.*` added | IMPA result, 5 shots | 8 leaves | metadata_only | #2159 |
| `DegenerateBaselineWindowError` raised from `vaft.process.signal_processing` instead of `vaft.process.magnetics` | 46403 log | same message | metadata_only | #2162 (B3) |
| k-file manifest text | all | absolute run path only | not a product difference | — |

The geometry/calibration provenance is not lost: the static manifest now names the registry
(`vest.yaml:0.diagnostics.equilibrium_magnetics.sensors`, sha256 `2bd5daea…`) in place of the
two retired YAMLs.

### 4.3 Reproducibility note (H)

The committed packaged baseline `baseline_75b154e31.json` does not reproduce byte for byte on
vestserver at the same code SHA. It was captured with python 3.12.13, numpy 2.4.3 and
omas 0.95.2; this run used 3.14.7, 2.5.3 and 0.94.2. The evidence is
`env_repro_39915.txt`. For 39915:
- 96 diagnostics arrays, 950 eddy arrays and 22 constraints arrays differ;
- their summary statistics differ by at most 4e-13 relative (≤ 3e-12 in constraints);
- the probe constraint scalar group differs by 1.2e-5 relative in its mean;
- the `channel_decisions` / `energisation` strings differ.

This is an environment effect, not a code effect. It means a cross-environment comparison needs
tolerances, while BEFORE/AFTER on one host can and did use exact equality.

## 5. Shot-era sensor identity (legacy-vs-registry fidelity and raw-dump checks)

`d2_identity.py` compares three sources per ODS index for 64 probes and 11 flux loops:
1. the **frozen legacy YAMLs** (`test/data/legacy_magnetics/{MD,VEST_MagneticsGeometry_Full_ver_2302,table}.yaml`)
   plus the shot-era wiring overrides frozen in `pre_2158_outputs.json`, which was generated from
   a git-archive export of 75b154e31 and not from the registry code;
2. the AFTER **registry** (`resolve_equilibrium_magnetics_registry(shot)`);
3. the **produced ODS** of both states.

It then tests the raw dump directly, with no vaft processing. For each candidate field code c,
x_c = −∫raw_c dt is fitted to the ODS waveform y (y ≈ a·x_c + b₀ + b₁t). The field code with the
largest |partial correlation| is the measured source, and sign(a) is the measured polarity.

What kind of evidence this is (#2166 C2/D3):
- **Sources 1–3.** The legacy YAMLs are the registry's donor and the input of the BEFORE ODS.
  Their agreement is **migration fidelity** (historical regression), not independent geometry
  evidence.
- **The raw polarity test.** The ODS waveform is calibration × ∫raw, and in every evaluation
  a/calibration > 0. The test therefore shows that the code applied the calibration it was
  given. It is a **processing self-check**, not evidence that the sign is physically right.
- **The raw source-code test.** This one is independent of the resolver. It discriminates only
  where the best code beats the runner-up by > 0.01 in |r|, which happens on 1,490 of the 3,570
  evaluations. On strongly PF-driven shots neighbouring probes see nearly the same waveform, so
  the test does not discriminate there.
- **Physical evidence for the wiring.** This comes from the vacuum response: the era-wiring-swap
  control of §6.3.

| | Result |
|---|---|
| Shots | 50: every captured shot. 39915 is tested against its server raw, and 46403 static-only (no diagnostics product) |
| Registry vs legacy: position, name, field code, calibration. ODS (both states): position and name | **0 mismatches** on every shot |
| Raw-derived polarity vs calibration sign (self-check) | **3,570 / 3,570** conclusive channel-evaluations agree (|partial r| ≥ 0.5) |
| Raw-derived source field code vs expected | Mostly the expected code. 2 contradictions, both flux loops on weakly driven vacuum shots: #13 on 39438 (r 0.60 vs 0.69 for loop #14's code) and #12 on 39439 (0.82 vs 0.92). Flux loops in vacuum see nearly the same waveform, so this test cannot discriminate between them. Unresolved for those two loops (no BEFORE/AFTER difference) |
| The 39437/39438 wiring pair | Probe 35 (R 0.796, Z +0.06, registry id `bpol_C_z+0.060`) reads field 170 up to 39437 (39417, 39436, 39437) and field 225 from 39438 (39438, 39439, 39446). Probe 47 (Z −0.42) is the reverse. The raw test **discriminates only on the post side**: on 39438/39439 the expected field gives |r| ≥ 0.992 against a runner-up of 0.87–0.98. On the pre side (39417, 39436, 39437) the expected field gives |r| = 1.0000, but the runner-up reaches ≥ 0.99996, so the pre-side assignment rests on the frozen legacy wiring plus the vacuum control (§6.3, 39436) |
| 46403/46404 | The registry is layout 2409 on both sides, and positions, field codes and calibrations are identical across the boundary. 46404 raw identity passes on 72/72 evaluated channels; 46403 has no ODS (§9.4) |

Caveat, unchanged by the consolidation: the ODS `name`/`identifier` of probes 35 and 47 is the
DAQ label of the field code. It therefore swaps across the boundary (`C2-04_Bz` ↔ `C4-05`) while
the physical sensor and its position stay put, and only the registry `id` is positional. A
consumer that keys on the ODS name across 39437/39438 would join the wrong sensors. #2166 C1 asks
that physical identity and mutable acquisition fields stay distinguishable; today the ODS does
not carry the physical id.

## 6. Vacuum benchmark and negative controls

### 6.1 Method

For every captured shot (p1 + p2), the diagnostics and eddy products are joined with
`vaft.database.composition.compose_stage_products`. `vaft.validation.vacuum_benchmark.run_benchmark_case`
then runs with the full channel set, at both code states. A case is called **informative** when:
- the scored correlation median is ≥ 0.95, and
- the scored normalized-residual median is ≤ 0.2.

**Disclosure.** A first run (`vacuum2-run1/`) applied the controls by name to 39436–39439 and
48927, so the 39436 control outcomes had been seen before this precondition was written. The
precondition was then defined, after seeing that the baselines of 39437–39439 carry no
information. It admitted 16 further cases whose control outcomes had not been seen. It is a
global median gate: for example, 48374 is admitted with an inboard median of 0.40, so a family
can be poor on an admitted case. The benchmark itself has no acceptance thresholds (#190).

Plasma-free does not mean vacuum shot: 8 of the 17 informative cases are plasma shots evaluated
on their pre-plasma interval (39446, 39915, 43016, 43639, 43689, 45891, 45964, 48374).

### 6.2 Results (scored normalized-residual median per family, AFTER = BEFORE on every per-sensor row)

| shot | era | informative | scored | corr med | inboard | outboard | side | inb. FL | outb. FL |
|---|---|---|---|---|---|---|---|---|---|
| 39417 | pre-43017 | yes | 51 | 0.999 | 0.033 | 0.076 | — | 0.128 | 0.026 |
| 39436 | pre-43017 | yes | 68 | 0.999 | 0.031 | 0.072 | 0.083 | 0.186 | 0.025 |
| 39437 | pre-43017 | no | 69 | 0.380 | 0.234 | 0.171 | 0.095 | 0.435 | 0.250 |
| 39438 | pre-43017 | no | 33 | 0.030 | 1.833 | 0.634 | 0.642 | 1.102 | 1.258 |
| 39439 | pre-43017 | no | 34 | −0.009 | 2.079 | 0.778 | 0.751 | 0.535 | 0.271 |
| 39446 | pre-43017 | yes | 69 | 0.998 | 0.057 | 0.037 | 0.022 | 0.027 | 0.028 |
| 39915 | pre-43017 | yes | 72 | 0.997 | 0.059 | 0.037 | 0.021 | 0.012 | 0.018 |
| 41445 | pre-43017 | yes | 69 | 0.992 | 0.062 | 0.037 | 0.112 | 0.034 | 0.026 |
| 41452 | pre-43017 | no | 69 | 0.336 | 0.493 | 0.471 | 0.460 | 0.029 | 0.025 |
| 43006 | pre-43017 | yes | 72 | 0.996 | 0.043 | 0.056 | 0.072 | 0.014 | 0.020 |
| 43016 | pre-43017 | yes | 72 | 0.996 | 0.052 | 0.086 | 0.066 | 0.018 | 0.012 |
| 43017 | 43017–45967 | no | 32 | 0.004 | 1.965 | 0.650 | 0.578 | 0.818 | 0.628 |
| 43639 | 43017–45967 | yes | 71 | 0.994 | 0.059 | 0.105 | 0.075 | 0.037 | 0.017 |
| 43684 | 43017–45967 | no | 65 | 0.987 | 0.209 | 0.180 | 0.271 | 0.126 | 0.044 |
| 43685 | 43017–45967 | yes | 67 | 0.998 | 0.044 | 0.053 | 0.022 | 0.011 | 0.019 |
| 43689 | 43017–45967 | yes | 64 | 0.999 | 0.047 | 0.044 | 0.021 | 0.013 | 0.027 |
| 45891 | 43017–45967 | yes | 71 | 0.999 | 0.058 | 0.105 | 0.112 | 0.071 | 0.018 |
| 45895 | 43017–45967 | yes | 71 | 0.995 | 0.066 | 0.150 | 0.209 | 0.078 | 0.026 |
| 45896 | 43017–45967 | yes | 48 | 0.996 | 0.046 | 0.036 | 0.083 | 0.073 | 0.029 |
| 45919 | 43017–45967 | yes | 48 | 0.996 | 0.026 | 0.031 | 0.085 | 0.105 | 0.024 |
| 45964 | 43017–45967 | yes | 67 | 0.999 | 0.067 | 0.119 | 0.146 | 0.123 | 0.016 |
| 46396–46410 (14 cases) | 45968+ | no | 30–46 | −0.16…0.04 | 0.50–3.0 | 0.47–2.1 | 0.43–1.7 | 0.38–3.2 | 0.30–2.2 |
| 48224 | 45968+ | no | 45 | 0.993 | 0.359 | 0.153 | 0.096 | 0.092 | 0.054 |
| 48366–48371 (6 cases) | 45968+ | no | 36–51 | −0.25…0.05 | 1.8–2.6 | 2.1–5.2 | 0.8–8.0 | 0.6–2.1 | 1.2–2.3 |
| 48372 | 45968+ | no | 49 | 0.989 | 1.064 | 0.201 | 7.257 | 0.469 | 0.249 |
| 48373 | 45968+ | no | 52 | 0.984 | 1.682 | 0.117 | 10.743 | 0.515 | 0.349 |
| 48374 | 45968+ | yes | 49 | 0.992 | 0.399 | 0.090 | 0.123 | 0.066 | 0.016 |
| 48375 | 45968+ | no | 43 | 0.991 | 0.347 | 0.139 | 0.236 | 0.065 | 0.049 |
| 48376 | 45968+ | yes | 54 | 0.994 | 0.122 | 0.089 | 0.105 | 0.013 | 0.032 |
| 48927 | 45968+ | no | 52 | 0.989 | 0.372 | 0.942 | 0.106 | 0.067 | 0.135 |

43026 is rejected because its eddy stage has no PF6/9/10 current, and 46403 has no product. Both
fail the same way at both states.

The uninformative cases are mostly shots with essentially no PF drive. Peak |I| is:
- ≤ 0.21 kA on 39438/39439;
- ≤ 0.4 kA on 46396–46410, except 46400 (PF2 1.4 kA);
- 21 A on 43017.

Three exceptions are PF-driven and still uninformative (§9.3):
- 39437: PF6 3.5 kA, PF9/10 1.0 kA, correlation median 0.38;
- 41452: PF1 19 kA, correlation 0.34;
- 48366–48371: PF5-only;
- 46400: PF2 1.4 kA.

Conversely, 45896 is weakly driven (PF1 10 A, PF9/10 77 A) and still passes the precondition.
"Uninformative" is therefore not simply "no PF drive".

In the 45968+ era, the inboard probes fit worse than in the earlier eras (0.12–0.40 vs
0.03–0.07) even on informative shots.

### 6.3 Negative controls (declared before running; `test/_negative_controls.py`)

| Control (declared threshold) | Informative cases where detected | Notes |
|---|---|---|
| sign flip, flux loop #4 (≥ 2×, r exactly negated) | **17/17** (×26–342) | r negated to 1e-9 on every evaluated sign control |
| sign flip, probe C1-04 (≥ 2×, r negated) | **15/15** evaluable (×9.8–30) | not evaluated on 48374/48376 |
| swap H2-04 ↔ C3-04 (≥ 2× each) | 27/32 sensor detections. H2-04 on every evaluable case (×4.5–42; not evaluable on 45919); C3-04 missed on 39417, 43685, 45964 (×1.8–1.9), 48374 (×0.11), 48376 (×0.45) | |
| era-wiring swap probes 35 ↔ 47, i.e. the other side of 39437/39438 (≥ 1.2× each) | 19/24 evaluable sensor detections. Both channels on 39436 (pre-era: ×6.3/×14.3), 39446, 41445, 43006, 43016, 43639, 45895. The Z −0.42 channel is missed on 39915, 43685, 43689, 45891, 45964. Not evaluable on 39417, 45896, 45919, 48374, 48376 | |
| move C3-04 by 0.20 m inward (≥ 1.2×) | **9/17**. Missed (×0.58–1.11) on 39417, 43016, 43639, 43685, 45891, 45964, 48374, 48376 | On 6 cases (39436, 39437, 39446, 41445, 48374, 48376) the move also changes other probes' rows (L-03…L-07), through the array-contradiction neighbour review |

Every control result is identical at BEFORE and AFTER. Evidence type: these controls are a
**self-consistency** check of the benchmark's sensitivity on real plasma-free data (#2166 D3).
They are not independent evidence that the machine model is right.

Wiring-era coverage requested by the brief:
- pre-39438: 39436 (every control detected) and 39417;
- post-39438: 39446 (every control detected), 39915, 41445 and later.

On 39437 itself (uninformative) the controls are not detected, except the C4-05 half of the
era swap (×7.98). On 39438/39439 most sensors are not evaluable.

48927 was run by name, and it is uninformative:
- detected: the H2-04 half of the swap (×40), the FL4 sign flip (×17.5) and the move (×1.52);
- missed: the C3-04 half of the swap (×0.40);
- not evaluable: the C1-04 sign flip and the era swap.

In the current 45968+ era (48374, 48376), sensitivity rests on the FL4 sign flip and the H2-04
half of the swap: C1-04 and the era swap are not evaluable, and C3-04 and the move are missed.

## 7. IMPA (archived extracts)

`process_impa_shot` gives identical numeric leaves and quality verdicts at both states for
35325, 35376, 35385, 39204 and 39923. AFTER adds eight `result.provenance.acquisition.*` leaves
(#2159), and the declared 25 kHz is reported `consistent` on all five. That check is not
independent here: the archived extracts store no timebase (`{type, data}` only), so the time
axis is synthesized from the declared rate. The
alignment shots 35376 and 35385 stay certified (geometry and `tf_coupling` pass), and
35325/39204/39923 stay rejected.

## 8. 48224 at 300 ms: stage-by-stage against the server k-file

The reference is `vaft/data/samples/48224/efit_magnetic_server/k048224.00300`, the server run
fetched on 2026-09-18 for #952. Ours is `k048224.00300`, generated from the D2 constraints product, and it is
identical at BEFORE and AFTER.

The server k-file and today's chain differ in many fields, and most of those differences are
structural. The root cause is **not asserted**.

| Stage / quantity | Server | VAFT (both states) | Observation |
|---|---|---|---|
| Ip (`PLASMA`) | 143.2 kA | 204.5 kA (diagnostics `magnetics.ip` at 300 ms: 204.6 kA; pre-compensation Rogowski 205.9 kA) | +43 %. The FL10 compensation is recorded as disabled from 47117 in today's `ip.method_name`. Not tested further |
| Flux loops used by the server fit (`FWTSI` = indices 4, 5, 6, 10: #9, #10, #11, #15) | 11.05 / 11.03 / 11.25 / 9.85 mWb | 11.01 / 10.87 / 11.28 / 10.02 mWb | agree within 1.7 % |
| Other flux loops | #4 −45.9, #5 −45.3, #6 −32.7, #12 13.9, #13 12.3, #14 −3.6 mWb | #4 −42.9, #5 −42.1, #6 −0.5, #12 18.1, #13 21.0, #14 −30.0 mWb | large differences. The server weighted these out; VAFT now enables 9 loops and drops #6 and #14 |
| Probes (`EXPMP2`) | 27 distinct values for 64 probes, exactly mirror-symmetric, U/L ≈ −0.09999 T; `BITMPI = 64*100` (weighted out) | measured fields (e.g. H1-01 +0.098 T) | The server values are a smooth template, not a measurement record |
| PF (`BRSP`, 26 groups) | 1694, 1691, −2058, −1313 A | 1694, 1691, −2058, −1314 A, plus 59 A on groups 9–10 | agree; VAFT disables the groups for PF circuits it reads as idle (FWTFC 0) |
| DFLUX | −7.34 mWb | +7.32 mWb | sign is the #1196 correction (the archived k-file keeps the pre-#1196 sign); magnitude agrees to 0.3 % |
| Settings | CUTIP 5000, RELIP 0.4, KFFCUR 2 | CUTIP 15000, RELIP 0.32; default preset KFFCUR 1, routine 2 | preset changes since 09-18 |

**Does the #2135 0.5 ms delay correction explain any part?**
- **Probes.** No part of the server fit. The server's probe values are a template and weighted
  out, so the probe low-pass delay could not have shaped that reconstruction, and a time-shift
  scan cannot test the delay against them. The scan of ±1.5 ms against those values selects the
  scan edge; it is meaningless here.
- **Flux loops.** These are unfiltered, so #2135 does not touch them. The scan's best shift is
  −0.15 ms, and the residual is dominated by loops #6 and #12–#14.
- **Ip.** The 61 kA difference is far larger than a 0.5 ms shift could make: dIp/dt near 300 ms
  is about 14 kA/ms, so a 0.5 ms shift moves Ip by about 7 kA. Also, the Ip reference that #2135
  filters is not used from 47117 on.

**Classification: `reference_inconclusive`.** The case stays `blocked` for root cause.

## 9. Findings outside the consolidation (identical BEFORE and AFTER — need owners, not blockers)

1. **Dead raw probes pass the quality layer and reach EFIT.** Several raw records are constant
   −5 V:
   - H3-08 (field 204): 22 shots between 43017 and 46410, though not on 43639 or 45891;
   - C1-01 (87): 45964 and 46396–46402;
   - U-06 (79): 46397–46410, except 46400;
   - L-04 (69): 48927;
   - H3-01 (197): 39436 (quality: fail);
   - C4-04 (224) on every captured shot from 41445 except 48927, already excluded by #977.

   The processed field is then about 1e-11 T. The diagnostics quality layer returns **pass** for
   some of these (U-06 on 46398, 46399, 46401, 46404, 46405, 46406, 46409; C1-01 on
   46396–46398 and 46401; H3-08 on 46400, 46406 and 46410; L-04 on 48927). On 46406, H3-08 and
   U-06 enter the EFIT constraints as ~0 T measurements (constraint weight 0.1, k-file
   `FWTMP2` = 1 in the default preset). The quality metrics look
   at the processed signal (`constant_fraction` 0, `flatlined` false), not at the raw record.
2. **ODS probe name follows the DAQ label, not the sensor** (§5).
3. **PF-driven shots with no vacuum agreement:**
   - 48366–48371 are PF5-only. Their PF5 record shows multi-kA offsets at the record start and
     end (48371: +2.88 kA at both ends, −6.3 kA excursion) and correlation ~0 with the probes.
   - On 48372/48373 the side-probe residuals are 7–11. These are the only shots in the set with
     multi-kA PF2 (3.5 and 4.8 kA); 48374–48376, with 10–64 A of PF2, fit well. 46400 (PF2
     1.4 kA) is uninformative too. This points at the PF2 channel (`coil_field_codes {1: 4}`),
     its gain or its coupling, rather than at PF5.
   - 39437 (PF6 3.5 kA) has correlation 0.38, and 41452 (full PF drive) 0.34.

   These matter for the PF5 calibration boundary (48366–48371), the PF2 channel (48372/48373)
   and PF6 or the era models (39437). The cause is not asserted.
4. **46403 cannot be processed.** The flux-loop baseline window [0.24, 0.26] s lies before the
   0.26–0.36 s record, so `DegenerateBaselineWindowError` is raised. Production has no
   diagnostics product for this shot either. This matches the donor's self-inconsistency at
   46403 that `vest.yaml` notes.
5. **Benchmark sensitivity.**
   - The 0.20 m move control is not a reliable detector.
   - Moving one probe changes other probes' rows through the array-contradiction review, so
     per-sensor attribution leaks. The committed 39915 test asserts that it does not on 39915,
     and it does not there.
   - The C3-04 half of the H2-04/C3-04 swap is missed in every era: on 39417 (pre-43017), 43685
     and 45964 (×1.8–1.9), and on 48374 and 48376 (×0.11 and ×0.45).
6. **No-plasma shots still get k-files.** On 39417, 45919 and 46406 the constraints stage found
   no plasma window (log: "constraint times cover the whole analysis range"), yet 3, 2 and 1
   k-files were produced. 46406 has a single file at 360 ms, with `PLASMA` = −15 kA.
7. **Production EFIT on a shot does not establish a plasma.** 39417 and 45919 were chosen as
   production-EFIT-reconstructed neighbours, but they have no plasma window and PF1 peaks of
   247 A and 10 A. The plasma-side coverage of the 39438 and 45896 boundaries is therefore
   39446, 45891 and 45964.

## 10. Not tested (coverage limits)

- PR #2182 (anti-alias correction). It is held, and nothing here covers it.
- EFIT/EFUND execution and equilibrium quality. The k-file inputs are byte-identical, so a
  reconstruction cannot differ through the consolidation (F4: "unchanged, verified inputs").
  No EFIT was run.
- The Ip polarity/baseline/units check across the FL10 boundary: no plasma shot is near the
  boundary in 46396–46410, and 46403 fails.
- `corrective-kinetic` (Thomson/CES/profiles) and `analytic-preprocessing` were not run by D2.
- The 41524, 41672 and 42962 packaged chains: other lanes, and their baselines stay references.
  41524 was also not run through the vacuum benchmark here.
- CHEASE / stability / GPEC consumers: not run, for the same unchanged-input reason.
- Code that #2189 changes and that may not have been exercised. D2 did not measure call
  coverage; these modules are partly or wholly outside the pipeline-1 stage scripts and
  `generate_kfile.py`:
  - `vaft/code/efit/legacy.py`;
  - `vaft/code/efit/recovery.py`;
  - `vaft/formula/{units,boundaries}.py`;
  - `vaft/code/nice/validate_reference.py`;
  - `workflow/resistive_zeff/build_zeff.py`;
  - `workflow/magnetics_geometry/verify_wiring.py`;
  - `vaft/plot/*`;
  - `vaft/data/{fourier,representation}.py`.
- #2166 H2 configuration-identity properties were not tested: determinism, sensitivity to
  parameters and insensitivity to cosmetic edits. D2 only records the AFTER fingerprints per
  shot in the JSON.
- 39915's raw identity test was fitted against the server raw (sha `4ce6884d…`), while its ODS
  was built from the packaged raw (`ac94261a…`). The packaged raw omits field 179 (recorded
  `not_archived`) and the time keys, so the test needs the server copy. The test only asks which
  raw field reproduces the ODS waveform, and the 74 channels evaluated give |r| ≥ 0.5. That is
  consistent with the two copies holding the same records for those fields, but it does not
  prove it.

## 11. Reproduction and evidence paths (vestserver, `ssh -p 2222 user1@147.46.36.244`)

Raw availability: `/srv/vest.filedb/raw/<shot>/vest_<shot>_daq_raw.json.gz` was present for:
- the portfolio shots 39437/39438, 46402/46403/46404, 45895/45896 (+45929/45960/45964/45965),
  48371/48372, 48224 and 39915/41524/41672/42962;
- the vacuum and neighbour shots 39417, 39436, 39439, 39446, 41445, 41452, 43006, 43016, 43017,
  43026, 43639, 43684, 43685, 43689, 45891, 45919, 46396–46401, 46405–46410, 48366–48370,
  48373–48376 and 48927.

Selection rules (fixed before results):
- Vacuum shots are the shots nearest each era boundary (39438, 41446, 43017, 43685) on the
  2026-10-05 vacuum/BD-failure list (`~/runs/vacuum-move-20261005/vac.txt`), plus 48927, the
  regression shot `vaft.omas.shot_class` names.
- Plasma neighbours are the nearest production-EFIT-reconstructed shots on each side of 39438,
  43017, 43685 and 45896. Near 46404 and 48372 no shot was reconstructed, so the five nearest
  shots on each side were used.

| What | Path (under `/home/user1/runs/consolidation-d2/`) |
|---|---|
| Captures (per shot: products, logs, per-leaf `baseline.json`) | `captures/{p1,p2}-{before,after}/<shot>/` |
| Leaf-by-leaf comparison with magnitudes and manifest diffs | `compare/{p1,p2}_leaf_compare.json` |
| k-files | `kfiles/{p1,p2}-{before,after}/{default,routine}/<shot>/kfile/` |
| Identity checks | `identity/{p1,p2}/<shot>.json`, `identity/summary.json` |
| Vacuum benchmark + controls | `vacuum2/{before,after}/{p1,p2}.json` (first run without the auto precondition: `vacuum2-run1/`) |
| IMPA | `impa/{before,after}.json` |
| 48224 | `48224/compare_{before,after}.json` |
| Full aggregate (266 records) | `qualification_results.full.json` |
| PF peaks / constraint verdicts per shot | `pf_peaks.json` |
| Inputs (symlink farm, configs, committed baseline copy) | `inputs/` |
| §4.3 environment reproducibility check | `env_repro_39915.txt` |
| Superseded runs, kept for the record | `vacuum/` (eddy-only inputs, all 49 rejected: no pf_active grid), `captures/smoke-before/` (driver smoke test) |

Tools are in `/home/user1/work/d2-tools/`, listed with the sha256 prefix of each file:
- `d2_capture.py` `b92fec22`, `d2_run.sh` `ab6ddcd1`
- `d2_compare.py` `29315444`
- `d2_kfile.sh` `71a98ee6`
- `d2_identity.py` `57e05da9`, `d2_identity_all.sh` `991468b4`
- `d2_vacuum.py` `c22167e1`, `d2_vacuum2.sh` `682bc621`
- `d2_impa.py` `4182bb34`
- `d2_48224.py` `b30e3fd8`
- `d2_aggregate.py` `70425861`, `agg_identity.py` `59106dd9`
- `chk_baseline.py` `d94bb463`, `chk_env_repro.py` `5d3fd86f`
- `d2_vacuum.sh` `365a0d2a` (the superseded eddy-only run)
- the frozen `capture_baseline.py` `5a4c5510`

Re-run one shot at one state:

```bash
~/work/d2-tools/d2_run.sh after <name> <shot>          # capture (both states: before|after)
~/work/d2-tools/d2_kfile.sh after <name> <shot>        # k-files
python ~/work/d2-tools/d2_compare.py captures/<name>-before captures/<name>-after --tree ~/work/d2-after --output cmp.json
```
