# Direct fixed-boundary PF-current fit (#1608, stage 1)

`fit_free_boundary_coils(eq, machine, ...)` takes a canonical
`EquilibriumData` and ODS `pf_active` element geometry. It needs no OFT or
TokaMaker runtime. The result keeps the plasma and coil contributions to full
weber flux and tesla field at each fitted boundary/X-point sample. Static
passive vessel currents are zero in this free-space model.

The target is converted to COCOS 11. `J_phi = -2π(R p' + FF'/(μ0 R))` is
integrated over fractional LCFS grid cells, producing ring currents in A.
The COCOS 11 target has `Br=+∂ψ/∂Z/(2πR)` and `Bz=-∂ψ/∂R/(2πR)`;
the ring Green flux has the opposite derivative orientation. Returned flux
components use the Green orientation, with an arbitrary additive gauge, while
the fitted physical fields follow the target orientation. The exact kernels
used by `compute_point_response_matrices` evaluate plasma self-field and
grouped signed-turn PF responses. Finite coil rectangles use a 3×3 area
quadrature of the same kernels. Circuit variables are physical A; no
filament-turn current is exposed as a circuit current.

The objective uses the difference in full-weber flux from the first LCFS
sample, avoiding an arbitrary flux offset. `flux_normal` also fits `B·n=0`
away from X-points, which instead get `Br=Bz=0`. The default weights divide
flux rows by `|psi_boundary-psi_axis|` and field rows by peak plasma boundary
field. `regularization` is a dimensionless Tikhonov coefficient on currents
divided by `current_scale_A` (default `|Ip|/ncoil`); `current_penalties` scales
individual terms. `current_bounds` maps circuit names to `(min_A,max_A)`;
omitted bounds are unbounded. The result includes physical residuals, weighted
matrix singular values/rank/condition number, regularization norm, active
bounds, integrated `Ip`, optimizer status and acceptance status. Acceptance
checks both RMS and maximum relative flux residuals (defaults 2% and 4%),
field residuals, and an integrated plasma current within 5% of declared `Ip`.
It also requires finite caller-supplied bounds for every PF circuit: without
machine current ratings, a good numerical field fit is `bounds_unverified`,
not certified as physically realizable. A regularized rank-deficient fit may
be accepted under bounds for its field fit; rank and condition remain explicit.

## Reproduce the small cases

From the repository root:

```bash
python -m pytest test/test_equilibrium_coil_fit.py -q
PYTHONPATH=. python validation/fixed_free_1608/direct_fit.py
```

The script uses the packaged VEST `pf_active` geometry, a 33-point radial
target grid, 40 LCFS samples, `flux_normal`, and `regularization=1e-5`.
The JSON output contains fitted currents and all primary diagnostics. The
following values were measured on 2026-10-04; small platform differences in
contour extraction and numerical quadrature are expected.

| Family | Topology | Integrated/target Ip | RMS/max flux/span | RMS Bn (T) | Max saddle component (T) | Condition |
|---|---|---:|---:|---:|---:|---:|
| Solov'ev/CF | limited | 0.9929 | 0.0038 / 0.0081 | 0.0016 | — | 1.13e8 |
| Solov'ev/CF | lower single null | 0.9911 | 0.0440 / 0.1135 | 0.0047 | 0.0030 | 9.31e7 |
| Solov'ev/CF | double null | 0.9863 | 0.0064 / 0.0153 | 0.0021 | 0.0038 | 1.28e8 |
| Guazzotto–Freidberg Part 1 | limited | 0.9996 | 0.0014 / 0.0028 | 0.0001 | — | 1.67e10 |
| Guazzotto–Freidberg Part 1 | lower single null | 0.9996 | 0.0105 / 0.0230 | 0.0004 | 0.0002 | 8.31e9 |
| Guazzotto–Freidberg Part 1 | double null | 0.9996 | 0.0013 / 0.0031 | 0.0001 | 0.0001 | 7.50e9 |

The Solov'ev lower-single-null case exceeds the default flux thresholds with
this coil geometry. None of these reference runs supplies hardware current
bounds, so their physical current feasibility remains unverified. The large
condition numbers show that individual currents are sensitive even where the
boundary fit is good.
Physical field and later free-boundary closure comparisons are the validation
targets; matching another inverse method's current vector is not required.

The direct fit does not run a Grad-Shafranov closure. Later stages add
profile transfer and free-boundary forward solves. Guazzotto
pressure/bootstrap surface currents and toroidal
flow are rejected because this volume-current model cannot represent them.

## Independent TokaMaker fixed-boundary path (stage 2)

`vaft.code.tokamaker.fit_free_boundary_coils_vfixed(eq, machine, workdir, ...)`
builds a plasma-only mesh from the target LCFS, sets
`settings.free_boundary=False`, solves, and fits `get_vfixed()` samples.
`fit_vfixed_samples(points, flux, machine, flux_scale_Wb=...)` exposes the
OFT-free linear inverse step for existing solver samples. Both functions use
the same exact PF response as the direct path, but the required external field
comes from OFT alone. No direct plasma filaments or direct external-field
target enter the native route. Samples are copied before the OFT solver is
reset and saved to a new work directory.

OFT's `get_vfixed` is an external FEM vacuum flux in Wb/rad. The PF fitter
converts it with `+2*pi` to VAFT's full-weber finite-coil response. OFT's
mathematical `eval_green` has the opposite sign to the assembled FEM coil
response; an independent 1 A finite-winding probe agrees with `+2*pi` to
within 0.019%. Flux fitting
removes the additive gauge. The result preserves the required external flux,
fitted external flux, RMS/max residuals, SVD, rank, condition, regularization
cost, bound activity, and the native fixed-solve statistics.

The stage-2 benchmark used the adapter's existing **power-law source
shape**, with the target LCFS and Ip. `fixed_profile_mode="power_law"` records
that distinction. Its vacuum field is an independent solver result, but its
current vector is not yet a comparison of identical analytic profiles. Stage
3 below adds the target `pprime`/`ffprime` shape. Boundary-field and free-boundary
closure comparisons follow that transfer; `accepted` here describes only the
linear sample fit with complete finite caller-supplied bounds.
The stage-2 fixed solve explicitly requires finite positive Ip after COCOS-11
conversion, matching the installed OFT target API. Other signs fail before
runtime initialization or creation of the output directory. Signed-current
orientation handling remains explicitly restricted by the native target API.

The limited-family resolution comparison ran on vestserver, in the isolated
directory `/home/user1/scratch/vaft-1608-vfixed.w1EylN`, using the existing
`/home/user1/miniconda3/envs/vaft/bin/python` OFT runtime. No solver installation
was changed. Run from the source root on a server:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 PYTHONPATH=. python \
  validation/fixed_free_1608/vfixed_fit.py \
  --workdir /tmp/vaft-1608-vfixed-new --resolutions .05 .035
```

The output directory must not already exist. Mesh order was 2,
regularization `1e-5`, and physical current bounds were not supplied.

| Family | mesh dx (m) | boundary samples | fixed/target Ip | RMS/max flux/span | Condition |
|---|---:|---:|---:|---:|---:|
| Solov'ev/CF limited | .050 | 40 | .999935 | .008773 / .019920 | 1.91e8 |
| Solov'ev/CF limited | .035 | 58 | .999883 | .008710 / .019927 | 1.92e8 |
| Guazzotto–Freidberg Part 1 limited | .050 | 21 | .999676 | .001885 / .004335 | 6.04e10 |
| Guazzotto–Freidberg Part 1 limited | .035 | 31 | .999672 | .001252 / .002677 | 6.50e10 |

All four fits report `bounds_unverified`. The largest unbounded Solov'ev
current is 12.8 MA; these fits demonstrate sample residuals and numerical
sensitivity, not attainable VEST currents. Two mesh sizes establish a small
smoke comparison, not asymptotic convergence or final acceptance tolerances.
Null-family solver coverage, matching analytic profiles, and closure remain
in the later validation matrix.

## Canonical and explicit source profiles (stage 3)

`TokaMakerConfig.profile_mode` now supports `power_law` (unchanged default),
`equilibrium`, and `explicit`. Forward input preparation uses
`profile_equilibrium=eq` for canonical source shapes, Ip and vacuum F0;
explicit scalar config overrides retain precedence. The native fixed bridge
uses its target equilibrium when `profile_mode="equilibrium"`:

```python
from vaft.code.tokamaker import TokaMakerConfig, fit_free_boundary_coils_vfixed
fit = fit_free_boundary_coils_vfixed(
    eq, machine, "/tmp/new-fixed-profile-run",
    config=TokaMakerConfig(profile_mode="equilibrium", dx_plasma=.035),
    regularization=1e-5,
)
```

`equilibrium_to_tokamaker_profiles(eq)` converts declared COCOS to 11, sorts
profile coordinates into axis-to-edge `psi_n`, and multiplies both derivatives
by `-2*pi` for native per-radian units. It does not perform another coordinate
reversal: OFT's wrapper already reverses `psi_n` internally. The canonical
profile derivative integral determines relative axis pressure; stored edge
pressure is preserved as metadata. OFT sets the internal edge pressure to
zero. Tables represent source shape; global Ip and relative axis pressure
set amplitudes. `fixed_boundary.json` and the forward result sidecar record
both requested and solver-realized native derivatives, rather than assuming
that source amplitudes survived the nonlinear solve unchanged.

Explicit tables contain `psi_n`, native per-radian `pprime` and `ffprime`, and
`axis_pressure_Pa` in `config.profile_tables`. Both endpoints 0 and 1 are
required, with finite, strictly increasing coordinates and matching arrays.
For zero FFprime the pressure source has only one adjustable amplitude;
pressure fixes it, and Ip is an independent output. For zero pprime the
pressure normalization is zero. An incompatible `R0` or `Ip_ratio` target is
rejected because OFT would give it precedence over pressure normalization.
Negative/nonfinite canonical Ip fails explicitly, matching the native target
API. COCOS 1/2/7/11/12/17, reversed table ordering, per-radian/full-weber
conversion, negative-current rejection and Bt-sign independence of FFprime
are tested. Tabulated normalization is currently static; evolution rejects
it instead of silently changing its per-step current constraints.

Four repeated solves ran in isolated vestserver directory
`/home/user1/scratch/vaft-1608-profiles.dEcsUZ` using the existing OFT runtime:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 PYTHONPATH=. python \
  validation/fixed_free_1608/vfixed_fit.py --workdir /tmp/new-profile-matrix \
  --resolutions .05 .035 --profile-mode equilibrium
```

| Family | dx (m) | fixed/target Ip | pprime relative error | FFprime relative error | RMS/max fit flux/span |
|---|---:|---:|---:|---:|---:|
| Solov'ev limited | .050 | 1.000000 | .00840 | .01492 | .001888 / .004048 |
| Solov'ev limited | .035 | 1.000000 | .00754 | .01192 | .001677 / .003410 |
| Guazzotto Part 1 limited | .050 | .993389 | .00849 | zero exactly | .002230 / .005702 |
| Guazzotto Part 1 limited | .035 | .995138 | .00162 | zero exactly | .001285 / .002814 |

These coarse runs quantify the source discrepancy introduced by table and
mesh discretization. They are not free-boundary closure results. All fits
remain `bounds_unverified`; physical boundary/global comparison and final
convergence tolerances are established in the remaining stages.

## Stage 4: true fixed-current free-boundary closure

`vaft.code.tokamaker.fixed_to_free(target, machine, new_workdir,
fit_backend="direct" | "tokamaker_vfixed", config=..., fit_options=...)`
retains the inverse fit, prescribed physical currents, realized native currents,
forward convergence/error and an independent `compare_equilibria` report in
`closure.json`. The default uses canonical source tables. No isoflux, saddle,
VSC or coil optimization is enabled during this forward solve. A converged
solver does not upgrade an unaccepted inverse fit to hardware feasibility or
certify the shape error. Coil halves use the adapter's bounding rectangles;
the inverse uses individual finite winding rectangles.

The initial current density is sampled from the target volume sources on native
nodes, then `vac_solve(rhs_source=Jphi)` assembles and solves that source on the
native mesh. This is a starting state, not a fixed boundary constraint; the
nonlinear solve subsequently determines its own LCFS. No direct plasma Green
integration enters native fitting or initialization. `init_psi(curr_source=...)`
is inappropriate here: that argument is an already assembled FE load vector.

Two convention corrections emerged from native closure:

* **`get_vfixed()` conversion is +2π**, rather than the -2π used in stage 2.
  The native FEM flux and VAFT ring flux have the same orientation. The native
  mathematical `eval_green` kernel instead has the opposite sign. An isolated
  1 A winding check gives FEM flux `7.4490813e-8 Wb/rad` at `(0.4,0)`,
  exact finite-winding flux `4.6812407e-7 Wb`, and `eval_green=-7.4507006e-8`.
  The +2π FEM conversion agrees within 0.019%. Stage-2/3 residual magnitudes
  remain valid, but their native-route current vectors have the wrong sign and
  are superseded by this correction.
* Closure exports **native COCOS 7**, then converts the parsed record to COCOS
  11 for comparison. OFT's COCOS-2 exporter flips flux and its derivatives while
  retaining the COCOS-7 Ip/F signs; treating that output as a full coordinate
  conversion gives reversed canonical Ip. The closure bridge consequently uses
  COCOS 7 even when the general forward config defaults to COCOS 2.

Reproduce the isolated server benchmark:

```bash
PYTHONPATH=. python validation/fixed_free_1608/closure.py --workdir new_closure --dx .04
PYTHONPATH=. python validation/fixed_free_1608/vacuum_response.py \
  --mesh new_closure/solovev_direct/free/vest_gs_mesh_<hash>.h5
```

The synthetic machine has twelve independent 1-turn 20 mm windings, complete
±200 kA bounds and regularization `1e-5`. Its rectangular limiter touches the
inboard target LCFS, and differs from the LCFS elsewhere. This is a numerical
reference geometry, not VEST hardware qualification. Runs used the existing
server OFT v26.9 runtime in isolated `vaft-1608-closure.roju1O/densityseed`.

| Family / route | Fit RMS / max relative flux | LCFS RMS / max [mm] | Axis [mm] | Outcome |
| --- | --- | --- | --- | --- |
| Solov'ev / direct | 0.008284 / 0.023189 | 2.942 / 6.146 | 1.933 | converged, limited |
| Solov'ev / native | 0.006930 / 0.018140 | 2.406 / 6.160 | 2.683 | converged, limited |
| Guazzotto–Freidberg ν=0.5 / direct | see `summary.json` | unavailable | unavailable | target normalization failed |
| Guazzotto–Freidberg ν=0.5 / native | see `summary.json` | unavailable | unavailable | target normalization failed |

Solov'ev realized currents differ by less than `4e-12 A`; canonical Ip matches
100 kA. Shared-definition beta_p, virial li, thermal energy and volume are
reported with their definitions. Target q95 is unavailable because the analytic
record lacks a q table; the report preserves that reason instead of inventing a
comparison. These are coarse reference results, not final acceptance tolerances.
Guazzotto pure-pressure ν=1 also failed native closure at dx=0.04 and 0.02; fixed
boundary solves and linear fits succeed. Its free-boundary normalization remains
a validation requirement for the subsequent refinement/final-case stages.

## Stage 5: optional native LCFS / X-point refinement

```python
result = fixed_to_free(target, machine, new_workdir, fit_backend="direct",
                       refine_shape=True,
                       fit_options={"current_bounds": bounds, "regularization": 1e-5},
                       refinement_options={"boundary_samples": 64,
                                           "regularization": 1e-5,
                                           "current_scale_A": 1000.})
```

The initial linear fit and fixed-current forward result remain in `fit`,
`forward`, `initial_currents_A`, `realized_currents_A`, `comparison` and `status`.
A separate solve in `refined/` applies installed native `set_isoflux` and
`set_saddles`, with arc-length LCFS samples and explicit active X-points.
Explicit `x_points=[(R,Z),...]` overrides saddle detection; `[]` disables saddles.
`set_coil_bounds` uses physical A and requires complete finite bounds containing
all starting currents. `set_coil_reg` penalizes `(I-I_initial)/current_scale_A`;
its native OFT weights are distinct from the initial linear-fitter penalty.

`refined`, `refined_currents_A`, `refined_comparison`, `refinement_status` and
`refinement_diagnostics` record the subsequent optimizer outcome, current changes,
bound activity and scaled current deviation. A successful refined solve does not
replace a failed initial closure or upgrade the initial fit's hardware status.
Refinement can start from a failed initial closure's target-volume-current seed;
when the initial closure converges, it starts from its solved volume sources.

Reproduce on a server:

```bash
PYTHONPATH=. python validation/fixed_free_1608/closure.py \
  --workdir new_refinement --dx .04 --refine-shape
```

Existing server OFT v26.9 outputs are isolated in
`vaft-1608-refine.jxMufS/results`. Same synthetic coils, bounds and source profiles
as stage 4, with 64 isoflux samples and native regularization weight `1e-5`.

| Family / route | Initial LCFS RMS / max [mm] | Refined LCFS RMS / max [mm] | Refined axis [mm] | Max current change [A] |
| --- | --- | --- | --- | --- |
| Solov'ev / direct | 2.942 / 6.146 | 0.874 / 2.142 | 0.159 | 18297.5 |
| Solov'ev / native | 2.406 / 6.160 | 0.871 / 2.132 | 0.135 | 1189.6 |
| Guazzotto–Freidberg ν=.5 / direct | solve failed | 0.0778 / 0.3324 | 0.189 | 28368.9 |
| Guazzotto–Freidberg ν=.5 / native | solve failed | 0.0778 / 0.3324 | 0.189 | 17942.7 |

All four refined cases converged with currents within bounds. Refined canonical
Ip equals 100 kA for Solov'ev and differs from the Guazzotto target 14254.4548 A
by +0.0013 / -0.0144 A for direct/native. This demonstrates recovery by a
separate current optimization; it does not resolve or hide the Guazzotto initial
fixed-current normalization failure. Diverted saddle validation, native closure
with final currents frozen, pure-pressure/current-pedestal cases, q95 and final
convergence-based acceptance remain stage-6 requirements.

## Stage 6: frozen-current verification and analytic matrix

`fixed_to_free(..., refine_shape=True, verify_refinement=True)` performs a
second **unconstrained** native free-boundary solve in the same finite-element
state after refinement. It removes all isoflux and saddle constraints, freezes
the measured optimized PF currents, and saves a separate
`refined/verified_free/` equilibrium. The initial fit, initial fixed-current
closure, and constrained refinement retain their own status and files. This
second solve tests whether the final coils and source profiles sustain the
surface without shape-control constraints. Restarting from a gridded g-file can
select another nonlinear branch, so verification deliberately retains the
native FE state. A failed initial closure remains failed even when refinement
and frozen verification succeed.

The public API notebook
[`notebooks/analytic_fixed_to_free_boundary.ipynb`](../../notebooks/analytic_fixed_to_free_boundary.ipynb)
uses `EquilibriumData`, ODS machine geometry, and `fixed_to_free`; it contains
no inverse-solver implementation. To reproduce the full native matrix in an
isolated directory with an installed OFT runtime, run from the repository root:

```bash
PYTHONPATH=. python validation/fixed_free_1608/matrix.py \
  --workdir /path/to/new/analytic-matrix \
  --families solovev guazzotto_freidberg guazzotto_pedestal \
  --topologies limited lower_single_null double_null --dx .04
PYTHONPATH=. python validation/fixed_free_1608/matrix.py \
  --workdir /path/to/new/pure-pressure \
  --families guazzotto_freidberg --topologies limited --nu 1.0 --dx .04
```

The measured 20-case result is in [`measured_matrix.json`](measured_matrix.json):
18 family/topology/route combinations and two pure-pressure routes, each a
condensed per-case record (`fit`, `initial`, `refined`, `verified` blocks)
on which `matrix.acceptance_failures` re-evaluates the same gates as on a
run's `summary.json`. It was
produced on the isolated vestserver OFT v26.9 runtime with a 65×65 target,
`dx_plasma=.04 m`, 12 independent one-turn rectangular PF coils, ±200 kA
per-coil bounds, initial and refinement regularization `1e-5`, 64 boundary
samples, and saddle isoflux-row weight 100. The current pedestal is 0.1 in
the Guazzotto model's dimensionless parameter; pressure/bootstrap surface
currents and toroidal flow are rejected explicitly. These are synthetic coils,
not VEST hardware ratings.

| Target | Direct fit / native fit | Initial direct / native | Frozen LCFS maximum direct / native [mm] | Frozen native topology |
| --- | --- | --- | ---: | --- |
| Solov'ev limited | accepted / accepted | converged / converged | 2.29 / 2.26 | limited |
| Solov'ev lower single null | residual or conditioning / accepted | converged / converged | 10.45 / 10.44 | lower single null |
| Solov'ev double null | accepted / accepted | failed / failed | 5.68 / 5.29 | double null |
| Guazzotto Part 1 limited | accepted / accepted | failed / failed | 0.38 / 0.38 | limited |
| Guazzotto Part 1 lower single null | accepted / accepted | failed / failed | 1.33 / 1.33 | lower single null |
| Guazzotto Part 1 double null | accepted / accepted | failed / failed | 2.00 / 1.99 | double null |
| Guazzotto current pedestal limited | accepted / accepted | failed / failed | 0.38 / 0.38 | limited |
| Guazzotto current pedestal lower single null | accepted / accepted | failed / failed | 3.53 / 3.53 | lower single null |
| Guazzotto current pedestal double null | accepted / accepted | failed / failed | 2.51 / 2.51 | double null |
| Guazzotto pure pressure limited | accepted / accepted | failed / failed | 0.31 / 0.31 | limited |

Every refined and subsequent frozen-current solve in this matrix converged.
The native FE saddle check requires flux agreement within 0.01% of the active
boundary, one-to-one matching to the target active X-points within 10 mm, and
the expected count and upper/lower placement. A mismatch is marked ambiguous
rather than promoted to double null. It checks X-point candidates, not a
separatrix trace. The checked native classification matches all diverted targets;
gridded 129×129 g-files can miss a narrow X-point, so both gridded and native
topology diagnostics are retained. The maximum native X-point displacement is
0.186 mm for Solov'ev, 0.011 mm for Guazzotto Part 1, and 0.033 mm for the
current pedestal. The independent 95% contour field integral supplies a
target `q95` even when the analytic record has no q table. Maximum absolute
frozen differences across the 18 principal cases are 0.043 in q95, 1.36 mm
for the axis, 0.0063 in beta_p, 0.0085 in virial li, and 9.25 A in Ip.
Energy, volume, fit condition and current changes are summarized in the JSON;
pressure integral and the full comparisons are in the individual manifests.

These coarse-grid **benchmark gates**, rather than universal hardware
guarantees, are justified by the measured discretization study:

| Gate | Limit | Measured worst case |
| --- | ---: | ---: |
| Frozen solve and native topology | converged and exact topology | 20/20 |
| LCFS maximum distance | 12 mm | 10.45 mm |
| Axis displacement | 2 mm | 1.36 mm |
| Native active X displacement | 0.5 mm | 0.186 mm |
| Absolute q95 field-integral difference | 0.06 | 0.043 |
| Absolute Ip difference | 15 A | 9.25 A |

The matrix command evaluates these gates per case in `summary.json` and exits
nonzero if any generated target, native solve, metric, X-point match or gate
fails. It retains all case records even after a failure. The initial inverse
fit acceptance and initial fixed-current closure are reported independently;
they are not silently counted as final frozen-current gate failures or passes.

For the direct plasma-field integration at target grids 33, 65 and 97,
Solov'ev RMS relative flux decreases `0.00458 → 0.00343 → 0.00211` and
Guazzotto RMS relative flux `0.000702 → 0.000685 → 0.000681`;
Solov'ev Ip relative error is `0.00711, 0.00293, 0.00298` and Guazzotto
`0.000440, 0.000119, 0.000055`. The nonmonotone Solov'ev Ip and normal-field
errors preclude an asymptotic convergence claim. A separate native limited
dx=.025 m check gave Solov'ev direct LCFS maximum 2.51 mm versus 2.29 mm at
dx=.04 m, and Guazzotto 0.24 versus 0.38 mm. The above gates cover these
observed variations with margin; finer meshes or other machines require new
convergence checks.

The Solov'ev lower-single-null direct linear fit is **not accepted** by its
default inverse residual criterion: RMS relative flux 0.01745, maximum
0.05697 (default maximum 0.04), despite the later shape-refined physical
closure. It remains an explicit inverse-fit limitation of this synthetic coil
set. Likewise, failed initial Guazzotto fixed-current closures are not counted
as initial successes. The final frozen-current result establishes the
coil-driven equilibrium reached after optimization, not acceptance of those
earlier stages.
