# Tutorial 02 Startup Workflow Redesign

## Goal

Restructure `tutorial/02_startup_scenario_and_vacuum_fields.ipynb` into a progressive startup-analysis workflow that starts from observed plasma response, establishes a single breakdown time, connects actuators to circuit/vacuum-field physics, validates the vacuum model against magnetics, and ends with interactive single-shot and comparative multi-shot analysis.

## Design principles

1. The notebook teaches public VAFT APIs rather than notebook-local plotting machinery.
2. A single `t_breakdown` is found early and reused everywhere; hard-coded breakdown times are removed.
3. The default packaged example remains usable offline, but the same workflow must accept `vaft.database.load(shot)` and shot lists/ODC sources where data are available.
4. Missing diagnostics are handled as capability gaps, not fabricated signals.
5. VEST-specific PF roles are described explicitly as VEST configuration details, not universal tokamak naming conventions.
6. Reduced startup quantities (`Bz`, `Vloop`, decay index, Lloyd/Townsend metrics) are presented as proxies/scalings, not replacements for equilibrium or transport analysis.

## EC assumption

For this tutorial stage, define the explicit teaching constant

```python
EC_FREQUENCY_HZ = 2.45e9
```

and use it only for electron-cyclotron resonance geometry. The resonance field is

\[
B_{ECR} = \frac{2\pi m_e f_{EC}}{e},
\]

and with the VEST toroidal-field scaling `B_T(R) * R = constant`, the midplane resonant radius is obtained from the measured/mapped TF field reference.

Do not fabricate an EC power trace or populate `ec_launchers` until the VEST EC mapper in issue #165 is implemented and historically validated. The hard-coded 2.45 GHz assumption must be visible in both code and explanatory markdown.

## Notebook flow

### Part I — One discharge

1. Load packaged shot by default and show how to replace it with `vaft.database.load(shot)`.
2. Plot response signals individually: Ip, H-alpha, all available impurity spectral lines, diamagnetic flux, inboard-midplane loop voltage, and an inboard-midplane Bz probe.
3. Show a fast-camera animation early for visual intuition when calibrated frames are available.
4. Determine `t_breakdown` immediately after the response overview and reuse it everywhere.
5. Build a six-row onset overview with one common breakdown marker.
6. Move actuators before the detailed circuit model: TF current and `B_t`, PF current and ampere-turns, pressure in Torr and Pa, and the 2.45 GHz EC resonance assumption.
7. Introduce active PF, passive vessel loops, and illustrative plasma filaments geometrically before deriving mutual-inductance/circuit equations.
8. Explain the axisymmetric circular-filament Green-function relation and how it produces mutual-inductance matrices.
9. Solve vacuum vessel eddy currents.
10. Explain the five uses of the same Green-function machinery: eddy solve, synthetic magnetics, PF/vessel/plasma field maps, pre-breakdown startup metrics, and post-breakdown reduced proxies.
11. Validate measured magnetics against coil-only and coil+eddy synthetic signals, then show plasma residuals.
12. Explain the two Rogowski-derived chains without calling calibrated sensor current a raw DAQ voltage: raw DAQ -> calibrated physical sensor current -> processed `magnetics.ip` / `magnetics.diamagnetic_flux`.
13. Plot `Bz(t)`, `Vloop(t)`, and decay index from coil onset through Ip peak with `t_breakdown` marked.
14. Add an interactive `Z=0` radial startup cut for `Vloop(R)`, `Bz(R)`, and Lloyd condition/margin.
15. Extend the interactive 2-D vacuum-field maps with a 2.45 GHz ECR-radius overlay.
16. Near `t_breakdown`, project startup field-line traces seeded at `(R_ref, 0)` and `(R_ECR, 0)` onto a calibrated fast-camera frame when supported.
17. Retain and reorganize useful null, connection-length, Townsend/Lloyd, and burn-through material around the common timing/state variables.

### Part II — Comparative startup analysis

Load multiple shots using `vaft.database.load([...])` and/or ODC. Compute per-shot breakdown timing and common startup features. Compare both absolute time and breakdown-aligned time. A compact summary should expose at least breakdown time, Ip peak, prefill pressure, representative actuator settings, and startup proxy values near breakdown when available.

### Exercise

Students inspect the VEST experiment log and VAFT database, choose an interesting date/shot group, compare at least 2–3 discharges, and write a one-page startup report linking actuator settings, vacuum-field conditions, breakdown evidence, and limitations.

## API boundary

Prefer existing canonical `vaft.omas.plot_*` and compute functions. Only add reusable helpers where the notebook would otherwise duplicate substantial composition logic. Candidate helpers are:

- a six-row startup response overview;
- a common startup-proxy time view (`Vloop`, `Bz`, decay index);
- an interactive radial startup cut.

Do not create one-off public APIs merely to shorten a single notebook cell.

## Physics wording constraints

- PF1 is described as the VEST transformer/Ohmic-drive coil; `Vloop` is electrical/Ohmic drive, not identical to dissipated Ohmic power during current ramp-up.
- `Bz` at the startup reference point is a reduced radial-force-balance proxy, not a full equilibrium solution.
- The vertical-field scaling

\[
B_v \simeq -\frac{\mu_0 I_p}{4\pi R_0}\left[\ln\frac{8R_0}{a}+\frac{l_i}{2}+\beta_p-\frac{3}{2}\right]
\]

is introduced only as a reduced interpretation model.
- Decay index is a simplified vertical-stability proxy; conducting-wall effects and full equilibrium/stability analysis are outside the toy model.
- For PF waveform control, distinguish VEST capacitor-bank circuit evolution from programmable current-regulated/H-bridge feedforward waveforms plus feedback correction.

## Success criteria

- No hard-coded `0.3307` or equivalent breakdown-time literals remain in analysis cells.
- The notebook still runs on the packaged example without network/database access.
- Arbitrary loaded shots can reuse the non-camera workflow when required IDS paths exist.
- The 2.45 GHz assumption is explicit, centralized, and used for resonance geometry only.
- New reusable helpers, if added, have focused unit tests and do not duplicate existing public plotting functions.
- Existing startup-physics work associated with issues #230/#783 is reused rather than reimplemented inconsistently.
