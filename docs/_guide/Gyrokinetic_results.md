---
title: Gyrokinetic scientific results
author: VEST team
date: 2026-10-14 22:00
category: guide
layout: post
permalink: /reference/gyrokinetic-results/
guide:
  architecture: Audited IMAS primary quantities, retained native sources and reusable scientific reductions with separate comparison evidence.
  prerequisites: Existing TGLF/CGYRO output or the compact offline nonlinear reference fixture.
  expected: Preserve model identity and reproduce qualified spectral, radial and saved-theta 2D results without running a solver.
related:
  api: [process, code, mapping, plot]
---

This implements the 2D milestone of #2020, #2025, #2026, #2028, #2030 and #2019. A stored number, an executed solver, a selected average, a stationary screen and demonstrated convergence are different pieces of evidence.

## Data contract and audit

`vaft.code.gacode.output_audit.output_audit()` enumerates the actual native container fields. `mapping_transformations()` exposes the existing leaf-level TGLF/CGYRO mapping audit, preserving its physical conversion evidence. The reference workflow exports both as `output_audit.json`.

- **A: standardized primary.** Only the audited subset maps to IMAS after documented species, coordinate, sign and normalization conversions.
- **B: reproducibly derived.** Retain inputs, method and auxiliary information. No invented IMAS leaf is needed for every diagnostic.
- **C: native-required.** Preserve complex histories, cross phases, QL weights, model internals and source metadata that cannot be recovered from standardized averages.

Qualification is orthogonal to A/B/C. A mathematically correct conversion can still describe an unconverged or physically unqualified calculation. The table below is generated from the same inventory as the JSON export; detailed destination, dependencies, transformation and evidence remain in that export.

| Solver | Quantity | Storage | Native axes |
|---|---|---|---|
| TGLF | directory | C | metadata |
| TGLF | gbflux | A | species,quantity |
| TGLF | ky_spectrum | A | ky |
| TGLF | eigenvalue_spectrum | A | ky,mode |
| TGLF | sum_flux_spectrum | A | species,field,ky,quantity |
| TGLF | field_spectrum | C | ky,mode,(vector,phi,Apar,Bpar) |
| TGLF | intensity_spectrum | C | species,ky,mode,(density,temperature,parallel_velocity,parallel_energy) |
| TGLF | density_spectrum | C | ky,species; sqrt(mode-summed intensity) |
| TGLF | temperature_spectrum | C | ky,species; sqrt(mode-summed intensity) |
| TGLF | nete_crossphase_spectrum | C | ky,mode |
| TGLF | nsts_crossphase_spectrum | C | species,ky,mode |
| TGLF | ql_flux_spectrum | C | species,field,mode,ky,(particle,energy,toroidal_stress,parallel_stress,exchange) |
| TGLF | ql_intensity_spectrum | C | species,ky,mode,moment |
| TGLF | width_spectrum | C | ky |
| TGLF | spectral_shift_spectrum | C | ky |
| TGLF | ave_p0_spectrum | C | ky |
| TGLF | saturation_parameters | C | named scalar parameters |
| TGLF | spectral_species | C | 1-based native species indices |
| TGLF | grid | C | metadata |
| TGLF | precision | C | metadata |
| TGLF | version | C | metadata |
| TGLF | errors | C | metadata |
| TGLF | files | C | metadata |
| CGYRO | directory | C | metadata |
| CGYRO | grid | A | p,theta,energy,xi,ky |
| CGYRO | equilibrium | A | species,geometry |
| CGYRO | time | C | raw print record or scalar provenance |
| CGYRO | time_error | C | raw print record or scalar provenance |
| CGYRO | frequency | A | ky,time |
| CGYRO | growth_rate | A | ky,time |
| CGYRO | ion_direction | C | raw print record or scalar provenance |
| CGYRO | ballooning | A | extended theta |
| CGYRO | flux | A | species,moment,field,ky,time |
| CGYRO | info | C | metadata |
| CGYRO | errors | C | metadata |
| CGYRO | exit_message | C | metadata |
| CGYRO | version | C | metadata |
| CGYRO | files | C | metadata |
| CGYRO | record_index | B | raw print-record indices |
| CGYRO | complex_field_history | C | radial slot,saved theta,ky,raw time |
| shared | zonal_fraction | B | method dependent |
| shared | zonal_shear_proxy | B | method dependent |
| shared | window_statistics | B | method dependent |
| shared | spatial_field | B | method dependent |

## Multi-model identity and time

`core_transport_from_tglf` and `core_transport_from_cgyro` retain their positional inputs and add optional `state_identity`, `realization_id` and `configuration`. Supply the state identity produced by the existing transport-state resolver and an explicit realization identity when intentionally grouping study members. `configuration` supplements native settings; contradictions with audited effective settings are refused.

`identifier.index=6` denotes the anomalous category. It never identifies a unique scientific realization. Selection uses solver, effective shared configuration and realization identity. Surface-dependent inputs/hashes are recorded per radius, not used as radial collection keys. All contributing surfaces must agree on species identity/order and shared controls. Native input controls, requested inputs and the effective configuration are distinct records. For b493397, the hard-coded `tglf_startup.f90` presets are applied: SAT2/3 change XNU_MODEL, WDIA_TRAPPED and potentially UNITS. These are not pure saturation changes.

Distinct settings append distinct model entries. Reapplying the same realization/state/time replaces only that slice; other times survive. A different state at the same time, incompatible configuration under one realization ID, or a time index already occupied by another time raises before the original ODS changes. Legacy entries without identity are preserved and never selected for overwrite. Without an explicit state ID, the mapper hashes retained local physical inputs and marks the evidence as inferred, rather than claiming experimental lineage. Full configuration cannot be certified if effective solver metadata are absent; an unverified source is isolated.

Compact versioned JSON is encoded in the standard `code.parameters` XML envelope. Existing code metadata tags are preserved. The reader handles both XML strings and OMAS-decoded dictionaries. The envelope separates collection settings from per-time plasma state and per-radius local run. A local `gyrokinetics_local` product represents one simulation; replacing it with a conflicting state, model, solver or experimental time is refused transactionally.

The per-input status records distinguish measured, inferred, assumed, solver-default and unavailable values. The physical evidence is retained separately from a numeric effective input. An explicitly supplied zero without measurement evidence is an assumption, not a measured zero. Nonzero flow/shear conversions that have not been audited remain native and are reported unsupported rather than mapped as zero.

Experimental time is in seconds. CGYRO evolution and averaging windows are in `a/c_s`. Each radial source keeps its own averaging interval, sample count, observed bounds and window statistics. TGLF quasilinear transport is not presented as a nonlinear time average.

## Native fields and Fourier reconstruction

`vaft.code.gacode.cgyro.read_field_history` returns a memory-mapped `FieldHistory`. It supports field selection, saved-theta selection, a simulation-time window and stride. Striding retains the final selected record. `history.read()` returns `(kx, ky, time)` at one saved theta, without loading unrelated fluxes or every field frame. Output precision comes from `out.cgyro.equilibrium`.

The ordinary CGYRO layout has a special first radial slot. It stays in `history.raw`, but is excluded by default from reconstruction coefficients. The reader reuses `CgyroOutputs`' first-strictly-increasing restart-record selection. Incomplete binary tail bytes and unpaired trailing records are reported; ambiguous/malformed time tables are rejected. `locality._kxky_phi` remains a compatibility entry point to the same parser.

`vaft.process.gyrokinetics.reconstruct_field` follows the pinned upstream `plot_fluct.py` half-sum:

```text
Re sum[p,n>=0] w[n] phi[p,n] exp(+i p x_angle - i n y_angle)
w[0] = 1/2; w[n>0] = 1
```

The positive display box lengths and orientation follow upstream `plot_fluct`, while signed native box length is retained in metadata. The algorithm refuses undersampled/nonuniform grids. This is a periodic local plane at an actually saved theta. The #39915 fixture has **THETA_PLOT=1**: it cannot establish full parallel structure, twist-and-shift continuity across parallel boundaries, flux-surface averaging, or a 3D global field. A `ky=0` component at that theta is labelled as such; it is not asserted to be a pure flux-surface-averaged flow.

Only phi history is retained in the verified EM reference. Apar/Bpar histories raise an explicit unavailable-field error. An EM transport run does not imply it wrote every spatial field.

## Reusable science and comparison

`zonal_turbulence_intensity(...).ratio` remains `I_Z/I_T`. The new `.fraction` and `zonal_fraction(...)` return `I_Z/(I_Z+I_T)`; an all-zero field is undefined. Weights and one-sided Fourier multiplicities are explicit. Unit-weight potential intensity is a proxy, not physical energy. `zonal_shear_proxy` remains a spectral proxy, not a rate in inverse seconds.

`spectral_integral` distinguishes signed bin contributions from densities. Density integration requires audited edges. `rebin_contributions` preserves signed totals using piecewise-uniform density within source bins and identical source/destination coverage; it never guesses quadrature from ky centres. `field_transport_contributions` reports signed fractions and cancellation; fractions are not clipped to [0,1]. `saturated_intensity` reconstructs TGLF moment intensity from QL intensity weights and saturated phi intensity; it agrees with both retained SAT0 and SAT2 native fixtures. No transport-flux reconstruction is offered without its independent quadrature information.

`reduce_time_series` and `time_window_average` integrate only finite adjacent observed samples. The default gap limit is three median spacings, a numerical convenience that callers can override. Boundaries are clipped within observed linear segments; there is no extrapolation or integration across rejected gaps. Returned coverage, sample counts and block coverage expose missing information. Standard error requires complete blocks with at least two observed samples each and is explicitly conditional on block independence. No confidence interval or numerical convergence is inferred. Relative drift is unavailable near a zero mean; the 0.2 drift threshold is only a heuristic screen. The native CGYRO flux mapper and nonlinear workflow use the same averaging policy.

`radial_correlation_from_spectrum` is array-only. The existing locality reader retains its all-native-slot, unweighted, theta/time-averaged observable for compatibility. A missing 1/e crossing remains a lower bound. Neither correlation/box adequacy nor a stationarity screen demonstrates resolution convergence.

`vaft.process.gyrokinetic_comparison.assess_comparison` returns separate display, state, model and qualification evidence. Missing metadata never produces matched evidence. Comparison intent is `matched_baseline`, `controlled_sensitivity` or `exploratory_overlay`; controlled changes are explicitly named `axis.key`. Renderers consume this assessment instead of judging equivalence themselves. Registered growth/frequency/flux plots retain structured reasons in view-model metadata and short annotations on figures. Numerical arrays remain unchanged by these presentation checks.

Radial plots accept `time_tolerance` in seconds and preserve requested/selected times and their differences. Without a tolerance, nearest-time selection is marked unverified. Outside-tolerance and ambiguous time selection are refused. Missing ion channels cannot silently become an apparently complete ion sum. Solver/SAT/QL-NL/configuration labels distinguish multiple model entries.

## Reproduce existing results

From a repository checkout, with VAFT's ordinary dependencies and the optional `video` extra:

```bash
python -m workflow.gyrokinetic.build_result_reference --out /tmp/vaft-gk-reference
```

The default is the compact **real** #39915 EM fixture, preserving 21 contiguous native records from `100–104 a/c_s`. This validates I/O, reconstruction, plotting and encoding offline. It does **not** validate a saturated mean or converged transport.

The command writes static total/zonal/nonzonal fields, a flux-window figure, MP4/WebM with provenance sidecars, `reference.json`, and `output_audit.json`. `--skip-video` needs no encoder. Scientific timestamps remain separate from playback FPS. Fixed contour levels span the selected sequence. Prepared diagnostics expose only samples at or before each frame and break across gaps.

For the full retained server snapshot:

```bash
python -m workflow.gyrokinetic.build_result_reference \
  --run /path/to/copied/39915-em --window 100 1037 --stride 100 \
  --out /tmp/vaft-gk-full
python -m workflow.gyrokinetic.build_multimodel_reference \
  --runs /path/to/copied/sat --linear /path/to/copied/linear \
  --nonlinear /path/to/copied/39915-em --out /tmp/vaft-gk-models
```

The SAT copy contains `sat0` through `sat3`, each with its original `state.json` and native radial directories. SI scales are restored from the recorded normalization; the recorded radial coordinate correspondence is reused. The example checks remapped electron energy flux against the source report, verifies CGYRO input species/geometry against that same state, saves all five models into one ODS, and checks identity after serialization. The linear option remaps the retained ky runs and SAT spectra, retaining all contributing run identities in the merged scan.

Capture a smaller native fixture with `python -m workflow.gyrokinetic.capture_reference SOURCE DEST --origin SOURCE_DESCRIPTION`. The manifest records source/fixture SHA256, byte counts, capture time, raw-record bounds and extraction policy. The original server files are never modified. Restart checkpoints are excluded. `workflow.gyrokinetic.verify_field_oracle` reproduces the oracle using hash-pinned upstream `data.py` and `plot_fluct.py` sources. The supplied fixture's oracle records exact raw-array agreement with upstream pygacode and maximum spatial reconstruction error `5.42e-16` against its actual FFT implementation at t=100.

The full reference contains restart-related time gaps. The new reduction excludes those gaps and can therefore withhold a block standard error that a bridge-across-gap average would report. The legacy all-slot zonal fraction differs from the special-slot-excluded saved-theta fraction by approximately `4.1e-6` over the inspected interval; the reference report records this definition change.

## Ownership and remaining work

Keep native parsing in `vaft.code.gacode`, numerical reconstruction/reductions in `vaft.process`, and presentation in the existing plot framework. No independent turbulence or workflow framework is introduced.

Mapper relocation is a separate compatibility-preserving series: first extract shared time/identity/XML helpers; then migrate GPEC together with the DCON/RDCON/STRIDE ownership review; then TGLF/CGYRO/NEO projections. Keep legacy import shims, update workflow/upstream entry points and verify scientific conversions, multi-time products, IMAS round trips and GPEC derived plots at each step.

The inspected import graph has `gpec_ideal` depending on shared helpers in `mhd_linear`, and both importing the GPEC package for native result types/readers. Moving only `gpec_ideal` into the eagerly exported GPEC package would risk a reader/mapper import cycle. Put solver projection exports behind lazy imports and move shared helpers before moving those modules. NEO and TGLF upstream call sites include `vaft.omas.vest_upstream`, `vaft.omas.neoclassical`, `workflow/transport_atlas/run_neo.py`, `run_tglf.py`, and `install/check_gacode.py`; CGYRO and registered plot examples import the current gyrokinetics mapper. Each needs a compatibility test before old aliases can be retired. No relocation is required to use the 2D result path.

The 3D #2019 milestone remains open. It must use #1286's shared mesh/field representation and needs sufficient saved parallel information and validated geometry/metric conventions. Local periodic copies will remain explicitly local, rather than being presented as global simulations.

## CGYRO elapsed time and discharge time (#2040)

CGYRO's native coordinate is **normalized gyrokinetic elapsed time**, `t c_s/a`.
The default plot label is `Gyrokinetic elapsed time [a/c_s]`. This is a physical
solver evolution coordinate measured from the solver origin. The discharge
state timestamp identifies the frozen background used to construct the input;
it does not advance when the solver evolves.

For verified per-run scales, `t_elapsed [s] = t_native * a/c_s`, with
`c_s = sqrt(T_e,ref/m_D)`. Use the actual reference length, electron temperature
and GACODE deuterium mass convention, even for hydrogen or mixed-ion plasmas.
`CgyroTimeNormalization` reuses the mass from the existing local-profile
normalization. `from_equilibrium` reads `a_meters` [m], `temp_norm` [keV],
`mass_norm` [10^-27 kg], and `vth_norm` [m/s]. It validates positive finite scales,
the mass convention and the recorded velocity against the temperature. It keeps
the recorded velocity, allowing GACODE's rounded energy constant (relative
consistency tolerance 10^-4). If the velocity alone is absent, it derives it from
the other verified quantities. Zero scales, including dimensionless
`PROFILE_MODEL=1` output, cannot establish SI conversion.

```python
from vaft.code.gacode.cgyro import (
    collect_cgyro_outputs, CgyroTimeNormalization,
    gyrokinetic_elapsed_time_s, gyrokinetic_normalized_time,
)

native = collect_cgyro_outputs(run_directory)
# For output containing actual SI reference scales:
norm = native.time_normalization
seconds = gyrokinetic_elapsed_time_s(native.time, norm)
restored_native_time = gyrokinetic_normalized_time(seconds, norm)
# For dimensionless local runs, explicitly use the retained corresponding input:
norm = CgyroTimeNormalization.from_normalisation(local_input.normalisation)
```

A retained local normalization must belong to that run. Native positive scales
and retained local scales that disagree are recorded as unavailable, rather than
silently replaced. Public `CgyroOutputs.time`, `CGYROConfig.max_time`, `delta_t`
and averaging-window parameters retain their native units and existing behavior.
Semantic accessors `gyrokinetic_normalized_time`, `time_normalization` and
`gyrokinetic_elapsed_time_s` supplement them; unavailable conversion raises an
error. Restart alignment, missing intervals and selection occur on the original
native coordinate.

`plot_flux_trace`, the predator–prey trace, phase-space and lag views, and
`plot_gyrokinetics_field_potential` accept `time_unit="native"` (default), `"s"`
or `"ms"`, together with `time_normalization=norm` for SI views. The field view
can also restore verified normalization already retained in its metadata.
Inputs, fits, peak times, lags and averaging windows stay native; plotting
transforms their coordinates together. A shared figure uses one temporal domain
for its field timestamp, diagnostic traces and animation driver. Playback FPS is
independent of either time coordinate. Custom native time labels remain supported.
These conversions are independent of the generic `xunit` display option.

CGYRO-to-GKDB time remains a different normalization:
`t_GKDB = t_native * sqrt(2)/RMAJ`, with `RMAJ=R0/a`. Its dimensional unit is
`R0/(sqrt(2)*c_s)`, so conversion through GKDB recovers the same SI elapsed time.
Experimental IMAS time leaves remain discharge time. Compact `code.parameters`
provenance stores per-run native window/unit, SI window when available, reference
scales, scale source, unavailable reasons and the conversion factor. It does not
introduce IDS leaves or store a full native history inside IMAS.

### Worked #39915 example

The existing EM nonlinear run at `r/a=0.70` is dimensionless and writes zero SI
scales. The companion `test/data/gacode/cgyro_39915_time_normalization.json` retains
scales from the corresponding SAT0 state: `a=0.28950719222017623 m`,
`T_e=1.7729478239492605e-17 J` and `c_s=72819.09630504964 m/s`. The fixture records
the source-state hash and verifies the converted local geometry and species
charge, mass, density, temperature and gradient inputs against CGYRO at
`rtol=1e-10`. Absolute SI scales are retained-state evidence, not reconstructed
from dimensionless output. The normalization file preserves the native input
hash and state identity; the reproduction workflow rejects a mismatched run.

Here `1 a/c_s = 3.975704271409646 microseconds`. The compact native interval
`100–104 a/c_s` represents `0.397570427–0.413473244 ms` from the solver origin.
The full run's `1037 a/c_s` corresponds to `4.122805329 ms`. The background state
is associated with **discharge time 0.317 s**. These values do not demonstrate
that the solver predicts the experimental plasma at 0.321122805 s. Any explicitly
requested addition to discharge time would be a hypothetical alignment requiring
that label; the default API never performs that addition.

Reproduce flux/field/zonal diagnostics and synchronized MP4/WebM offline:

```bash
python -m workflow.gyrokinetic.build_result_reference \
  --out /tmp/vaft-cgyro-time-ms --time-unit ms \
  --time-normalization test/data/gacode/cgyro_39915_time_normalization.json \
  --stride 2
```

Synthetic tests verify native/SI round trips, keV/J units, mass and missing-data
rejection, and the GKDB factor. Real fixture tests verify matching elapsed-time
coordinates across flux, field, zonal diagnostics, fits/peaks and averaging
windows. These checks do not establish physical stationarity, numerical
convergence, or discharge-trajectory equivalence.

The common physics-normalized discharge-time interface in #2033 is not yet
present on this implementation base. This adapter exposes explicit SI scales and
solver-relative coordinates for that future bridge; it does not duplicate fixed
reference or cumulative normalization algorithms. No discharge-relative alignment
is inferred.
