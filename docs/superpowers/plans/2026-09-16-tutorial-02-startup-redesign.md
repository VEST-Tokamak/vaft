# Tutorial 02 Startup Redesign Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rebuild Tutorial 02 as a single-shot-to-comparative startup workflow with early breakdown timing, richer diagnostics/actuators, Green-function/circuit interpretation, vacuum-model validation, 2.45 GHz ECR overlays, and multi-shot comparison.

**Architecture:** Keep the notebook as a composition layer over canonical `vaft.omas` APIs. Add only narrowly reusable plotting/analysis helpers where several cells would otherwise duplicate composition logic. Preserve offline execution on packaged shot 39915 and gate optional camera/database content by capability.

**Tech Stack:** Python, OMAS/IMAS ODS/ODC, NumPy, Matplotlib, VAFT plotting/process APIs, nbformat/nbclient tests.

**Spec:** `docs/superpowers/specs/2026-09-16-tutorial-02-startup-redesign-design.md`

## Global Constraints

- `EC_FREQUENCY_HZ = 2.45e9` is an explicit tutorial assumption used only for resonance geometry.
- Do not fabricate `ec_launchers` power data before issue #165 is implemented and validated.
- Remove hard-coded breakdown-analysis times, including `0.3307`; use one early `t_breakdown` variable.
- Preserve offline packaged-shot execution.
- Prefer canonical `vaft.omas.plot_*` and compute functions over notebook-local reimplementation.
- Distinguish raw DAQ, calibrated Rogowski sensor current, and processed Ip/diamagnetic flux.
- Treat `Vloop`, reference-point `Bz`, decay index, Lloyd/Townsend quantities as reduced proxies/scalings.

---

### Task 1: Add executable regression coverage for Tutorial 02

**Files:**
- Create: `test/test_tutorial_session_02.py`
- Reference: `test/test_tutorial_session_01.py`
- Exercise: `tutorial/02_startup_scenario_and_vacuum_fields.ipynb`

**Interfaces:**
- Consumes: packaged `vaft.omas.sample_ods()` and notebook execution environment.
- Produces: an offline notebook execution test plus structural assertions for the timing/EC invariants.

- [ ] **Step 1: Copy the Session 01 execution-test pattern and target Tutorial 02**

Use `nbformat` + `NotebookClient`, `MPLBACKEND=inline`, and the same tutorial-mode environment handling as Session 01. The test must execute the notebook from first cell to last without network/database access.

- [ ] **Step 2: Add notebook-source invariants**

Parse code cells and assert:

```python
code = "\n".join(
    "".join(cell.get("source", [])) if isinstance(cell.get("source"), list) else cell.get("source", "")
    for cell in notebook.cells
    if cell.cell_type == "code"
)
assert "EC_FREQUENCY_HZ = 2.45e9" in code
assert "0.3307" not in code
assert "t_breakdown =" in code
```

- [ ] **Step 3: Run the new test and record the current failure**

Run:

```bash
pytest test/test_tutorial_session_02.py -q
```

Expected before notebook edits: FAIL on missing EC constant and/or hard-coded `0.3307`.

- [ ] **Step 4: Commit the failing regression test**

```bash
git add test/test_tutorial_session_02.py
git commit -m "test: cover Tutorial 02 startup workflow"
```

---

### Task 2: Restructure loading, early response signals, camera, and breakdown timing

**Files:**
- Modify: `tutorial/02_startup_scenario_and_vacuum_fields.ipynb`

**Interfaces:**
- Consumes: `vaft.omas.sample_ods()`, canonical plot APIs, `vaft.omas.find_breakdown_onset`.
- Produces: `ods`, `t_breakdown`, early response plots, and common timing variables reused by later tasks.

- [ ] **Step 1: Rewrite the load cell to preserve offline default while documenting database replacement**

Default executable path:

```python
ods = vaft.omas.sample_ods()
```

Immediately below, add a non-executed/commented alternative:

```python
# shot = 39915
# ods = vaft.database.load(shot)
```

Explain that downstream cells operate on either source when required IDS paths exist.

- [ ] **Step 2: Expand the response section**

Show individually, in this order:

```python
vaft.omas.plot_plasma_current_time(ods)
vaft.omas.plot_spectrometer_uv_time_intensity(ods, emission="H_alpha")
vaft.omas.plot_spectrometer_uv_time_intensity(ods, emission=[...all available impurity labels...], layout="subplots")
vaft.omas.plot_diamagnetic_flux_time(ods)
vaft.omas.plot_flux_loop_time_voltage(ods, selection="inboard_midplane")
vaft.omas.plot_b_field_probe_time_field(ods, ...inboard-midplane Bz selection...)
```

Do not hard-code a single CIII line when additional mapped impurity lines are present.

- [ ] **Step 3: Move the fast-camera visual overview here**

Use the existing visible-camera animation/frame API when packaged calibrated frames are available. Guard the cell so shots lacking calibrated camera data skip with a clear explanatory message rather than fail.

- [ ] **Step 4: Compute breakdown timing immediately after the response overview**

```python
t_breakdown = float(vaft.omas.find_breakdown_onset(ods))
```

Also derive reusable coil-onset and Ip-peak times from mapped data/available timing helpers.

- [ ] **Step 5: Build the six-row onset overview using existing composable plots where possible**

Create one `fig, axes = plt.subplots(6, 1, sharex=True, ...)`; render the six response families into supplied axes with `show=False`, and draw the same `ax.axvline(t_breakdown, ...)` on every row. If the impurity renderer needs multiple axes, use one compact combined impurity panel rather than expanding the overview beyond six rows.

- [ ] **Step 6: Execute the notebook test**

```bash
pytest test/test_tutorial_session_02.py -q
```

Fix only failures introduced by this task.

- [ ] **Step 7: Commit**

```bash
git add tutorial/02_startup_scenario_and_vacuum_fields.ipynb
git commit -m "tutorial: move breakdown timing into startup response"
```

---

### Task 3: Move and enrich the actuator section

**Files:**
- Modify: `tutorial/02_startup_scenario_and_vacuum_fields.ipynb`

**Interfaces:**
- Consumes: `ods`, `t_breakdown`.
- Produces: `EC_FREQUENCY_HZ`, `B_ECR`, startup reference radius, and actuator interpretation used later.

- [ ] **Step 1: Add explicit tutorial constants**

```python
from scipy.constants import e, m_e

EC_FREQUENCY_HZ = 2.45e9
R_STARTUP_REF = 0.4
B_ECR = 2.0 * np.pi * m_e * EC_FREQUENCY_HZ / e
```

If the project avoids SciPy constants in tutorials, use existing VAFT constants or define documented SI constants locally; do not add a new dependency solely for this cell.

- [ ] **Step 2: Plot TF current and TF field**

Show `plot_tf_coil_time_current(ods)`, then a `B_t` representation at/reference to `R_STARTUP_REF`, and explain `B_T \propto 1/R`.

- [ ] **Step 3: Plot PF current and ampere-turns**

Show PF current in layout/subplots and then the current-turn product. Explain the VEST-specific startup roles of PF1, PF5, PF6, and PF9/10 without presenting the numbering as universal.

- [ ] **Step 4: Add VEST capacitor-bank versus programmable-current-control explanation**

State that the VEST waveform is largely determined by charging voltage, switching timing, and L/R/C circuit dynamics; contrast this with programmable feedforward waveform tracking and feedback correction in current-regulated/H-bridge systems.

- [ ] **Step 5: Plot pressure in Torr and Pa**

Use the canonical barometry renderer/unit option or the named conversion constant; do not add another rounded Pa/Torr literal.

- [ ] **Step 6: Replace the old EC mapping-gap prose**

Explain that EC power mapping remains issue #165, but this tutorial adopts a visible `2.45 GHz` resonance-frequency assumption for geometry only. Compute/display `B_ECR` numerically.

- [ ] **Step 7: Run Tutorial 02 test and commit**

```bash
pytest test/test_tutorial_session_02.py -q
git add tutorial/02_startup_scenario_and_vacuum_fields.ipynb
git commit -m "tutorial: expand VEST startup actuators"
```

---

### Task 4: Put geometry and Green-function physics before the vessel circuit solve

**Files:**
- Modify: `tutorial/02_startup_scenario_and_vacuum_fields.ipynb`
- Reference: `vaft/formula/green.py`
- Reference: geometry plotting APIs under `vaft/omas/plotting.py` / backend recipes.

**Interfaces:**
- Consumes: machine/PF/passive geometry in `ods`.
- Produces: pedagogical link `geometry -> Green response -> mutual inductance -> coupled circuits`.

- [ ] **Step 1: Add a poloidal geometry cell before the eddy equation**

Compose active PF coils and passive vessel structures; add illustrative plasma-filament markers only if an existing public plasma-filament geometry renderer provides them. Do not inject fake filament data into the ODS just for the plot.

- [ ] **Step 2: Explain the circular-filament Green response**

Introduce

```text
psi(R,Z) = G_psi(R,Z; R',Z') I
M_ij = G_psi(R_i,Z_i; R_j,Z_j)
```

and state that off-diagonal mutual terms follow from the axisymmetric Green relation while self terms require the conductor/self-inductance model.

- [ ] **Step 3: Rewrite the vessel equation with unambiguous notation**

Use

```text
L_vv dI_v/dt + R_v I_v = -M_va dI_a/dt - M_vp dI_p/dt
```

then specialize to the vacuum solve by omitting plasma drive. Avoid using `M` and `L` names in a way that conflicts with standard self/mutual-inductance notation.

- [ ] **Step 4: Keep the existing `compute_eddy_currents(ods, [], [])` execution path**

Explain that empty plasma filaments select the vacuum response.

- [ ] **Step 5: Run Tutorial 02 test and commit**

```bash
pytest test/test_tutorial_session_02.py -q
git add tutorial/02_startup_scenario_and_vacuum_fields.ipynb
git commit -m "tutorial: connect startup geometry to eddy circuit physics"
```

---

### Task 5: Reorder vacuum-model validation and Rogowski processing explanation

**Files:**
- Modify: `tutorial/02_startup_scenario_and_vacuum_fields.ipynb`
- Reference: mapped Rogowski processing code/tests associated with issue #215.

**Interfaces:**
- Consumes: solved vessel currents and mapped `magnetics` IDS.
- Produces: validation sequence and sensor-to-derived-quantity interpretation.

- [ ] **Step 1: Place vacuum magnetics validation immediately after the eddy solve**

Show the full B-pol probe and flux-loop views with measured versus synthetic coil+eddy contribution, using the existing vacuum overview renderer or its canonical components.

- [ ] **Step 2: Follow with the plasma-residual view**

Use the existing residual renderer after, not before, the vacuum-model check.

- [ ] **Step 3: Add Rogowski processing-chain cells**

Inspect/display the two relevant `magnetics.rogowski_coil` sensor-current rows and explain:

```text
DAQ source -> calibrated physical Rogowski sensor current -> processing/compensation -> derived quantity
```

For Ip, mention baseline/reference/FL10 compensation and sign convention only where supported by the mapped implementation. For diamagnetic flux, describe its dedicated processing chain. Explicitly state that `rogowski_coil.current` is calibrated sensor current, not raw DAQ voltage.

- [ ] **Step 4: Add the conceptual five-use overview**

Connect Green-function/circuit machinery to eddy solve, synthetic magnetics, equilibrium field maps, pre-breakdown startup conditions, and post-breakdown reduced proxies.

- [ ] **Step 5: Run test and commit**

```bash
pytest test/test_tutorial_session_02.py -q
git add tutorial/02_startup_scenario_and_vacuum_fields.ipynb
git commit -m "tutorial: validate startup vacuum model against magnetics"
```

---

### Task 6: Replace hard-coded startup snapshots with common timing/proxy histories

**Files:**
- Modify: `tutorial/02_startup_scenario_and_vacuum_fields.ipynb`

**Interfaces:**
- Consumes: `t_breakdown`, coil onset, Ip peak, vacuum-field compute APIs.
- Produces: common startup interval and time histories for `Vloop`, `Bz`, decay index.

- [ ] **Step 1: Delete every analysis use of `0.3307`**

Use `t_breakdown` for snapshot quantities unless the text explicitly discusses another event derived from data.

- [ ] **Step 2: Define the common plotting interval**

Set x-limits from coil-current onset through Ip peak using data-derived times.

- [ ] **Step 3: Plot common startup proxies**

Render `Vloop(t)`, `Bz(t)` at `R_STARTUP_REF, Z=0`, and decay-index history with the same x-limits and `t_breakdown` vertical marker.

- [ ] **Step 4: Add reduced-physics interpretation**

Include the vertical-field scaling from the spec and clarify that `Vloop` is transformer/Ohmic electrical drive rather than direct dissipated Ohmic power during ramp-up.

- [ ] **Step 5: Run test and commit**

```bash
pytest test/test_tutorial_session_02.py -q
git add tutorial/02_startup_scenario_and_vacuum_fields.ipynb
git commit -m "tutorial: align startup proxies to detected breakdown"
```

---

### Task 7: Add reusable ECR geometry helper if no equivalent already exists

**Files:**
- Prefer Modify/Create under the existing formula/startup module that owns scalar startup formulas.
- Test: add to the corresponding formula test file.
- Modify exports only if required by established VAFT conventions.

**Interfaces:**
- Produces a pure scalar helper equivalent to:

```python
def electron_cyclotron_resonance_field(frequency_hz: float, harmonic: int = 1) -> float:
    ...
```

and, only if it clearly belongs in the same module,

```python
def toroidal_resonance_radius(b_t_ref: float, r_ref: float, b_res: float) -> float:
    ...
```

- [ ] **Step 1: Search existing formula APIs first**

If both relations already exist publicly, use them and skip code creation; do not duplicate.

- [ ] **Step 2: Otherwise write failing scalar tests**

For `2.45e9 Hz`, assert the fundamental ECR field is approximately `0.0875 T` within a physically reasonable numerical tolerance. Test harmonic validation (`harmonic >= 1`).

- [ ] **Step 3: Implement the minimal pure-SI formula**

Use named physical constants from the project/dependency stack, with a docstring specifying frequency in Hz and field in T.

- [ ] **Step 4: Run focused formula tests and commit**

```bash
pytest <corresponding-formula-test-file> -q
git add <formula files> <test files>
git commit -m "feat: add electron cyclotron resonance geometry formula"
```

---

### Task 8: Add interactive radial and 2-D startup views with ECR overlay

**Files:**
- Modify: `tutorial/02_startup_scenario_and_vacuum_fields.ipynb`
- Modify plotting/backend code only if an existing renderer cannot accept a simple overlay/reference radius.
- Add focused plotting tests only for any new public option/helper.

**Interfaces:**
- Consumes: `t_breakdown`, `R_STARTUP_REF`, `EC_FREQUENCY_HZ`, `B_ECR`, TF reference field, vacuum map compute APIs.
- Produces: `R_ECR` and interactive radial/2-D interpretation.

- [ ] **Step 1: Compute `R_ECR` from TF field reference**

Use `B_T(R) R = constant`; do not infer EC power or launcher geometry.

- [ ] **Step 2: Add the `Z=0` radial interactive section**

At minimum show `Vloop(R)`, `Bz(R)`, and Lloyd condition/margin for a selectable time centered initially at `t_breakdown`.

- [ ] **Step 3: Extend/use the existing interactive 2-D vacuum map**

Overlay a vertical/reference locus at `R_ECR` with a legend annotation `2.45 GHz ECR` while preserving the existing flux/Bp/Ephi/Lloyd/decay-index/null-related views.

- [ ] **Step 4: Add tests only if plotting API surface changed**

Verify that the overlay accepts a finite reference radius, appears on all appropriate R-Z axes, and does not modify computed field arrays.

- [ ] **Step 5: Run focused plotting tests plus Tutorial 02 execution and commit**

```bash
pytest <focused-plot-tests> test/test_tutorial_session_02.py -q
git add <changed files>
git commit -m "tutorial: add interactive startup and ECR field views"
```

---

### Task 9: Align camera field-line interpretation to breakdown and ECR geometry

**Files:**
- Modify: `tutorial/02_startup_scenario_and_vacuum_fields.ipynb`
- Potentially modify: camera/field-line process wrapper only if it cannot consume vacuum-field traces.
- Test: `test/test_field_line_tracing.py` or a focused new camera-overlay test.

**Interfaces:**
- Consumes: `t_breakdown`, `R_STARTUP_REF`, `R_ECR`, calibrated camera mapping.
- Produces: onset-near camera image with two labeled startup field-line seeds when supported.

- [ ] **Step 1: Inspect whether the existing camera field-line overlay is equilibrium-only**

If it already accepts explicit world-space trajectories, compute vacuum trajectories separately and pass them through. If it is hard-wired to equilibrium, add the smallest explicit-trajectory/vacuum-trace path rather than overloading equilibrium semantics.

- [ ] **Step 2: Add focused tests before modifying the helper**

Cover projection of two trajectories and preservation of existing equilibrium overlay behavior.

- [ ] **Step 3: Update Tutorial 02 camera section**

Choose the available frame nearest `t_breakdown`; overlay trajectories seeded at `(R_STARTUP_REF, 0)` and `(R_ECR, 0)`. If the selected shot lacks calibrated camera geometry, print/markdown-skip clearly.

- [ ] **Step 4: Run tests and commit**

```bash
pytest test/test_field_line_tracing.py test/test_tutorial_session_02.py -q
git add <changed files>
git commit -m "tutorial: align startup field lines with fast camera"
```

---

### Task 10: Add comparative multi-shot section without duplicating Part I

**Files:**
- Modify: `tutorial/02_startup_scenario_and_vacuum_fields.ipynb`

**Interfaces:**
- Consumes: `vaft.database.load([shot, ...])`, existing ODS/ODC-compatible plot adapters, timing helpers.
- Produces: a concise comparative workflow and exercise.

- [ ] **Step 1: Add a non-offline-executed database example**

```python
# shots = [39915, 39916, 39917]
# ods_list = vaft.database.load(shots)
```

Keep network/database-dependent cells disabled/commented or tutorial-mode guarded so packaged CI remains offline.

- [ ] **Step 2: Show breakdown-aligned comparison**

Use existing time-convention/alignment API when available rather than manually shifting every signal array.

- [ ] **Step 3: Build a compact per-shot summary**

Compute/display available values for breakdown time, Ip peak, prefill pressure, representative actuator values, `Vloop`, `Bz`, decay index, and Lloyd/null metrics at/near breakdown. Missing diagnostics should appear as unavailable, not silently as zero.

- [ ] **Step 4: Add the final one-page report exercise**

Require students to choose a real experiment date and at least 2–3 shots, compare actuator/vacuum/breakdown behavior, and state model/data limitations.

- [ ] **Step 5: Run Tutorial 02 execution and commit**

```bash
pytest test/test_tutorial_session_02.py -q
git add tutorial/02_startup_scenario_and_vacuum_fields.ipynb
git commit -m "tutorial: add comparative VEST startup analysis"
```

---

### Task 11: Final verification and cleanup

**Files:**
- All files changed by Tasks 1–10.

**Interfaces:**
- Produces: a clean, executable Tutorial 02 redesign ready for review.

- [ ] **Step 1: Search invariants**

```bash
git grep -n "0.3307" -- tutorial/02_startup_scenario_and_vacuum_fields.ipynb
```

Expected: no match.

Check that `2.45e9` occurs in one authoritative tutorial/formula location plus explanatory notebook text, not as multiple conflicting magic numbers.

- [ ] **Step 2: Run focused tests**

```bash
pytest test/test_tutorial_session_02.py test/test_field_line_tracing.py test/test_omas_signal_time.py -q
```

Add any formula/plot test files changed in Tasks 7–9.

- [ ] **Step 3: Execute the notebook once from a clean kernel**

Confirm no stale outputs or hidden state are required.

- [ ] **Step 4: Review the notebook pedagogically**

Verify the narrative order is:

```text
observe response -> detect breakdown -> inspect actuators -> geometry/Green/circuit -> validate magnetics -> interpret startup proxies -> interactive radial/2-D fields -> camera alignment -> compare shots -> exercise
```

- [ ] **Step 5: Final commit if cleanup changed files**

```bash
git add <cleanup files>
git commit -m "chore: finalize Tutorial 02 startup workflow"
```
