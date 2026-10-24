# Production call graph: Snakemake stages and the preprocessing they run

Phase A audit for #2148, part of #2147. This file covers the **develop** tree at
`75b154e31`, i.e. after PR #2135 merged. Where #2135 changed a chain, both
behaviours are given:

- **before #2135:** the code at `7b1fcf192`, which produced the existing products;
- **develop:** the code that runs now.

Line numbers refer to develop at `75b154e31`, except in the "before #2135" column,
which cites `7b1fcf192`. Paths are relative to the repository root.

Abbreviations:
- `P1` = `workflow/automatic_pipeline_1_routine_data_processing`
- `P2` = `workflow/automatic_pipeline_2_corrective_data_update`
- `VU` = `vaft/omas/vest_upstream.py`
- `MM` = `vaft/machine_mapping`
- `Y` = `MM/vest.yaml`

The machine-readable counterpart is `inventory.json`. Its
`production_reachability` field is computed statically:

- **Entry modules:** the Snakefile `python <script>.py` commands.
- **Followed:** resolved imports, relative imports and parent-package re-exports.
- **Not followed:** calls through `self`, `getattr`, registries and string
  dispatch. Those are listed under [Unresolved dispatch](#unresolved-dispatch)
  with a way to verify each one.

Regenerate it with:

```bash
python workflow/architecture_audit/build_inventory.py        # write
python workflow/architecture_audit/build_inventory.py --check
```

## Summary of what actually runs

The table shows the filter on each chain before and after #2135. All entries were
checked against the source.

| Chain | Stage | Filter before #2135 (`7b1fcf192`) | Filter on develop (since #2135) | Other numerics |
|---|---|---|---|---|
| B-pol probes, 64 channels | diagnostics | `firwin(251, 2.5 kHz, fs 250 kHz)`, then causal `lfilter`, a 0.5 ms lag (`vaft/process/magnetics.py:594,600`) | zero-phase `apply_fir_filter` inside `b_field_pol_probe_field` (`vaft/process/magnetics.py:1870`; taps from `vest_probe_lowpass_taps`, `:522`) | divide by the MD.yaml calibration; `-cumtrapz_compat`; `_linear_baseline` (deg-1 polyfit over the era's leading samples); `np.interp` onto the era window, then `np.interp` onto the policy grid (`MM/magnetics.py:928`) |
| Flux loops, 11 channels | diagnostics | **none** | none | calibration; `-cumtrapz`; divide by 2π; era baseline (leading samples, a [0.24, 0.26] s window, or legacy index ranges); `np.interp` with no anti-alias filter (`vaft/process/magnetics.py:1001-1003`). The comment there claims "an already low-passed loop voltage", which the code does not do. The same holds for the second `np.interp` onto the policy grid (`MM/magnetics.py:1930` via `_interpolate_signal`), whose comment (`:925-927`) claims every caller is low-passed; unfiltered callers there also include Ip (`:2232`) and diamagnetic flux (`:2468`). |
| Ip (production, `ref = -1`) | diagnostics, IMPA | none | none | `calibrate_vest_signal`; `_linear_baseline_subtract` (polyfit); FL10 reference by Y `plasma_current.processing.reference`, see the note below; sign flip |
| Ip reference comparison (`ref != -1`) | no production caller | `firwin` + causal `lfilter` (`MM/magnetics.py:1215-1225`) | `apply_fir_filter` (`MM/magnetics.py:1227`) | — |
| Diamagnetic Rogowski (`rogowski_coil[1]`) | diagnostics | none | none | `repair_clipped_interval`; `scipy.integrate.cumulative_trapezoid` × `rogo_gain` (`MM/magnetics.py:1401`) |
| Diamagnetic flux | diagnostics | none | none | `repair_clipped_interval`; three `cumulative_trapezoid` passes; two-point hand-written baseline; ×1000 (`MM/magnetics.py:1405-1536`) |
| TF current | diagnostics, IMPA | `firwin(26, 2.5 kHz, fs 25 kHz)`, then causal `lfilter` (`MM/tf.py:181,195`) | zero-phase `apply_fir_filter` (`MM/tf.py:197`) | `calibrate_vest_signal` (×−3e4); mean baseline (1000 samples); `repair_tf_excursions`; `smooth(50)`; `np.interp` |
| PF active, 10 circuits | diagnostics, IMPA (strict `vfit_pf`) | `firwin(hamming 251, 2.5 kHz, fs 25 kHz)` → `filtfilt`, or `lfilter` below `_filtfilt_min_length`; then `uniform_filter1d(10)` (`MM/pf_active.py:146-161`) | same rule via `apply_fir_filter(short_record="causal")` (`MM/pf_active.py:156`) | mean baseline (5000 samples); per-era `coil_gains`; `np.interp` alignment; PF6 `repair_clipped_interval` |
| IMPA Hall/Bz (opt-in) | impa | `firwin` → `filtfilt`, `lfilter` fallback (`vaft/process/impa.py:380-384`) | `apply_fir_filter(short_record="causal")` (`vaft/process/impa.py:381`) | `np.interp` channel alignment (`MM/impa.py:252`); gain; baseline; `lstsq` and polyfit fits |
| Mirnov raw (equilibrium positions + phase reference) | diagnostics | none: native-rate crop only | none | — |
| Barometry | diagnostics | `medfilt` | `medfilt` | `resample_to_time` |
| Langmuir / UV / EC | diagnostics | anti-alias inside `resample_to_time` | same | `calibrate_vest_signal` |
| Diamagnetic plasma window | diagnostics, constraints | `onset.active_window`: `medfilt` or a zero-phase Butterworth, chosen by Y `plasma_timing` | same | — |

FL10 reference modes for production Ip (the Ip row above), selected by
Y `plasma_current.processing.reference`:
- **subtract:** `np.interp` plus its own polyfit baseline.
- **subtract_fl10_windowed** (shots 46403–47116): `signal.decimate`, then `smooth`, then `np.interp` (`MM/magnetics.py:1032-1097`).
- **disabled** (shots ≥ 47117).

`mirnov_preprocess_signal` (`vaft/process/magnetics.py:1094`) is **not** reached by
any Snakefile: fluctuation Mirnov voltages are published raw.

## Pipeline 1 (routine)

Each stage is listed as Snakefile rule → script → VAFT API → configuration →
preprocessing.

Workflow-local numerics: the only production script that computes anything is
`P1/generate_constraints_ods.py`. Its `_box_averaged` (`:295-300`) is a cumsum box
average, i.e. `legacy.box_average` evaluated at every sample.

### `generate_raw_db_dump` (Snakefile:462)

- **Script:** `P1/generate_raw_db_dump.py:81`.
- **API:** `vaft.database.raw.dump_all_raw_signals_for_shot` (`vaft/database/raw.py:1463`).
- **Config:** `raw.mode` and `raw.archive_template`.
- **Numerics:** none. The time base is rebuilt from t0/dt (`raw.py:759,765`).

### `generate_static_ods` (Snakefile:489)

- **Script:** `P1/generate_static_ods.py:36`.
- **API:** `build_static_ods` (VU:194).
- **Era:** chosen by the hard-coded `VEST_MACHINE_ERAS` (VU:102-106). That table is
  outside vest.yaml; see `config_audit.md` G14.
- **Magnetics:** `vfit_magnetics_static(ods)` with shot 0 (`MM/magnetics.py:1814`):
  - probe and loop **positions** come from `VEST_MagneticsGeometry_Full_ver_2302.yaml` (`MM/magnetics.py:853`);
  - **names** come from `table.yaml` (`:858`), keyed by field code.
- **Manifest:** hashes MD.yaml and the geometry YAML under `magnetics_calibration` /
  `magnetics_geometry` (VU:278-289).
- **Numerics:** none.

### `preflight_raw_dumps` (checkpoint, Snakefile:510)

- **Script:** `P1/validate_raw_dumps.py:48`.
- **Checks:** field presence and sample count only. `raw.preflight.required_fields` = [1, 12, 25, 59, 109].

### `select_impa_shots` (checkpoint, Snakefile:539) and `generate_impa_ods` (Snakefile:563)

- **Gate:** both run only when `impa.enable` is set. It is `false` in `config.yaml`.
- **Chain:** `build_impa_ods` (VU:1194) → `resolve_impa_config` (`MM/impa.py:107`) →
  `impa()` (`MM/impa.py:570`) → `process_impa_shot` → `process_impa`
  (`vaft/process/impa.py:1559`).
- **Indirect legacy chains:** `vfit_tf_current`, `vfit_plasma_current` and strict
  `vfit_pf`, i.e. the TF, Ip and PF chains above.
- **Config:**
  - `impa.processing.sample_rate` defaults to 25 kHz (`MM/impa.py:214`). It is
    **never compared** with the acquired time vector (#432; `config_audit.md` G9).
  - The manifest records `_impa_configuration_sha256` (VU:1183) of the resolved
    IMPA block. This is the only stage that hashes its effective configuration.

### `generate_diagnostics_ods` (Snakefile:591): main focus

- **Script:** `P1/generate_diagnostics_ods.py:71`.
- **API:** `build_diagnostics_ods` (VU:754).
- **Time policies:** `resolve_diagnostics_time_policies` (`MM/utils.py:413`) over
  Y `diagnostics_time_policies`:
  - the analysis window is 0.26-0.36 s at dt 4e-5, and covers PF, magnetics, IMPA, UV and Langmuir;
  - TF, barometry and EC power use the full discharge, 0-1 s;
  - config.yaml `diagnostics.tstart/tend/dt` retune only the default (analysis) window.
- **Override path:**
  - Snakefile:605 serialises `config["vest"]["magnetics"]["processing"]`.
    `config.yaml` has `vest: {}`, so this is `{}`.
  - `magnetics_processing_for_shot` (VU:682) returns `None` for an empty override,
    and the era policy from Y applies.
  - A non-empty override goes through `dataclasses.replace(era, **overrides)`
    (VU:732), after the shadowed-key guard (VU:699-713) and the output-grid guard
    (VU:721-731).
  - The Snakefile never passes `time_policies`.
- **Per-component chains:** see the summary table.
- **Configuration readers:**
  - every component calls `resolve_vest_diagnostic(shot, name)` (`MM/utils.py:299`);
  - magnetics also reads `equilibrium_magnetics_processing_config`
    (`MM/magnetics.py:1569`), `magnetics_wiring_for_shot` (`:802`, MD.yaml plus
    `wiring` revisions) and `known_magnetics_faults` (`:834`).
- **QA:** `validate_magnetics_signals` + `project_validity` (VU:1097-1100).
  This is **not** report-only. `project_validity` (`vaft/validation/magnetics.py:1321-1339`)
  writes the channel validity nodes, so EFIT and every other reader see the
  condemned channels through that validity (VU:1095-1096).
- **Affected eras:**
  - the probe causal lag (before #2135) affects every product made before #2135 merged;
  - the flux-loop anti-alias question applies mostly to native-DAQ shots ≥ 46404,
    where loops are acquired at 250 kHz;
  - FL10 windowed compensation covers shots 46403-47116.

### `generate_eddy_ods` (Snakefile:629)

- **API:** `build_eddy_ods` (VU:1462), which reads PF currents from the diagnostics
  product, interpolates Ip onto the PF time base with `np.interp` (VU:1528), and calls
  `compute_eddy_currents` (`vaft/omas/process_wrapper.py:731`) →
  `solve_eddy_currents` (`vaft/process/electromagnetics.py:816`).
- **Numerics in the solver:**
  - sub-stepping with `np.interp` onto `t_fine` (:942);
  - `np.diff` divided by dt (:949);
  - `expm` and `inv(R)`.
- **Config:** `eddy.*` from config.yaml.

### `generate_constraints_ods` (Snakefile:670)

- **Script:** `P1/generate_constraints_ods.py`.
- **Inputs:** `compose_stage_products`.
- **Time selection:** `plasma_timing`, i.e. the onset detectors.
- **`_drop_vacuum_times`:** `box_average` (`vaft/code/efit/legacy.py:67`) of Ip
  against `vest_equilibrium_regime_policy`.
- **`correct_flux_loop`** (`legacy.py:256`): runs only when `fl_correct_option != 0`.
  It is `0` in config.yaml.
- **`decide_efit_channels`** and **`gaussian_probe_recovery`**
  (`vaft/code/efit/recovery.py:140`): both are enabled by config.
- **Constraint assembly:** `vaft.code.efit.kfile.generate_constraints_ods`
  (`vaft/code/efit/kfile.py:423`):
  - `build_efit_coil_currents` with `np.interp`;
  - box-averaged constraints;
  - uncertainty = |relative error × data|;
  - probe and loop grouping by the `INBOARD_PROBE_MAX_R` / `SIDE_PROBE_MIN_ABS_Z` /
    `OUTBOARD_PROBE_MIN_R` thresholds (`kfile.py:578-580, 623-624`);
  - VCURRT from the eddy loop currents.

### `generate_efit_table` (Snakefile:721, era tables only)

- **API:** `generate_era_table` (`vaft/code/efit/efund.py:941`) → `build_static_ods(era)` →
  `efund_geometry_from_static` (`MM/efund_geometry.py:434`).
- **Geometry:** probe and loop r/z come **from the static ODS, in ODS index order**
  (`MM/efund_geometry.py:293-330`), so EFUND order = ODS order = YAML order.
- **Numerics:** none.

### `generate_kfile`, `run_efit_reconstruction`, `generate_efit_ods` (Snakefile:754, 779, 811)

- **K-file:** `generate_kfile` (`kfile.py:834`) applies the preset sigma floor.
  No filtering.
- **Reconstruction:** `run_efit` runs a subprocess.
- **Output:** `collect_efit_outputs` (`vaft/code/efit/magnetic.py:1144`) reads the
  g-files.

### Stage plots, HSDS replication, CHEASE, GPEC, MHD linear

- **Preprocessing:** none that feeds a stored product.
- **Stage plots:** `render_stage_plots` dispatches by plot name
  (`render_stage_plots`, `vaft/database/production_qa.py:629`, then `_render_ods_plot`, `:568`).
- **Replication:** off by default (`hsds.replicate: false`).
- **CHEASE:** boundary and profile resampling with `np.interp`/`interp1d`
  (`vaft/code/chease.py:621-1384`), applied to equilibrium output, not to measured
  signals.

## Pipeline 2 (corrective)

Every `P2/generate_*.py` script is a thin argparse wrapper: one call into `VU`,
then `write_stage_product`. **No workflow-local numerics.**

| Rule (Snakefile line) | API | Preprocessing actually executed | Config |
|---|---|---|---|
| `generate_thomson_ods` (190) | `build_thomson_ods` (VU:1856) → `thomson_scattering` (`MM/thomson_scattering.py:637`) | unit conversion; sigma sanitising; dead-channel sentinels set to NaN | `external.data_root`; channel R hard-coded in `_CHANNEL_META` (`thomson_scattering.py:24`, "DERIVED, NOT AS-BUILT") |
| `generate_ces_ods` (212) | `build_ces_ods` (VU:1907) → `charge_exchange` (`MM/charge_exchange.py:413`) | unit conversion; `abs()` errors | `external.ces_options` |
| `generate_core_profiles_ods` (239) | `build_core_profiles_ods` (VU:1975) → `build_kinetic_core_profiles` (`vaft/code/efit/kinetic.py:823`) or `profile_fitting_thomson_scattering` | ψ mapping with `RegularGridInterpolator` (linear); `curve_fit` via `fit_profile` (`vaft/formula/utils.py:935`); order-lowering retry. Fitted profiles carry **zero** σ (`utils.py:1313`). The electron-only branch omits `coordinate=` and uses the hard-coded default `rho_tor_norm` instead of the vest.yaml value. | vest.yaml `core_profiles` |
| `generate_composition_ods` (275) | `build_composition_ods` (VU:2352) | onset (see below); `np.interp` volume weights; ADF11 log-log `RegularGridInterpolator`, **linear extrapolation** outside the table (`vaft/formula/atomic.py:55,130-135`); `expm` | `composition.*`; vest.yaml `impurity_model`, `ti_te_ratio` |
| `generate_ti_inferred_ods` (317) | `build_ti_inferred_ods` (VU:2623) | `np.interp`; `nanstd` σ(p); linear error propagation. σ(T_i) is computed but **not stored** (VU:2773). | `ti_inference.*` |
| `generate_resistive_zeff_ods` (354) | `build_resistive_zeff_ods` (VU:2959) → `compute_resistive_zeff_ods` (`vaft/omas/resistive_zeff.py:327`) | `time_derivative` (`vaft/process/numerical.py:20`); `cumulative_trapezoid`; optional `smooth_local_polynomial` (off by default); `minimize_scalar`. Reads no flux loops. | `resistive_zeff.*` |
| `generate_{electron,kinetic}_efit_ods` (403-443) | `build_kinetic_efit_ods` (VU:3366) → `run_kinetic_chain` (`kinetic.py:1151`) | `np.interp` ψ_N(R); weighted `np.polyfit` T_i; `np.interp` onto a 129-point grid | `kinetic.*`, `efit_preset` by name |
| plots / replicate (454-531) | Pipeline 1 scripts | none | — |

`P2/legacy/` (`update_*.py`) is imported by nothing in production. The only
mentions of it are comments and docstrings in `vaft/database/utils.py` (`:120`, `:310`, `:417-419`, `:467`).

`workflow/resistive_zeff/build_zeff.py` is a study script, not part of Pipeline 2.
It finds FL10 through the literal name `"Flux Loop - #10"` (`build_zeff.py:239`).
That only works because `MM/magnetics.py:1677` copies `table.yaml` names into
`magnetics.flux_loop.*.name`: it is an implicit string coupling to `table.yaml`,
so it is a C5 consumer.

## Pipeline 3 (summary)

- **Scripts:** seven `gen_*.py` scripts, called by relative path.
- **Data path:** `BASE_PATH` is hard-coded to `/srv/vest.filedb/public`.
- **Numerics:** volume-averaged `trapz` and `np.interp` inside
  `vaft/database/_summary.py`.
- **Scope:** no preprocessing of measured signals.

## Unresolved dispatch

Each item names how to verify it at run time.

1. **Shot revisions.** Every Y parameter is shot-dependent through `revisions`
   (`MM/utils.py:193/212`).
   - The diagnostics manifest records the effective magnetics config **only when an
     override is passed** (VU:1134-1137).
   - To verify, call `resolve_vest_diagnostic(shot, name, with_provenance=True)`, or
     `vaft.machine_mapping.vest_processing_provenance(shot)`. No stage calls the
     latter.
2. **`plasma_timing` detector rules** (`_signature_rule`, `MM/utils.py:635`) choose
   between median, Butterworth and threshold detectors. Verify with
   `resolve_plasma_timing_policy()` and a traced run on a sample shot.
3. **`cumtrapz_compat`** (`vaft/compat.py:65-71`) chooses `cumulative_trapezoid` or
   `cumtrapz` through `getattr`. Both are numerically identical.
4. **Name-dispatched calls:**
   - stage-plot renderers: `STAGE_VALIDATION_PLOTS` → `vaft.omas.plotting.render`;
   - GPEC modules: `_runner_module`;
   - the EFIT preset: `efit_preset(name)`.
   Verify by listing the registries.
5. **Stage set:** `STAGES = scoped(stage_scope(config["stages"]))` and
   `CORRECTIVE_LINEAGE` decide which P2 rules exist. Verify with
   `snakemake -n --list-target-rules` under the deployed config.
6. **Config-gated branches** that are statically reached but off by default:
   - `correct_flux_loop` / `vest_signal_onoffsetpeak` (`fl_correct_option = 0`);
   - IMPA (`impa.enable = false`);
   - HSDS replication.
   The inventory lists them as reached. The off-by-default status is recorded in
   `production_reachability_note`.

Static reachability **over-approximates** in two ways:
- a called class reaches all of its methods;
- config-gated branches count as reached.

It **under-approximates** wherever dispatch goes through the mechanisms in items 1-5.
