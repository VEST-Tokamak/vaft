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

OFT's `get_vfixed` uses its `eval_green` orientation in Wb/rad. The PF fitter
converts it with `-2*pi` to VAFT's full-weber Green orientation. An optional
installed-OFT test checks this against OFT's independent `eval_green`, with
relative agreement better than `2e-8` at three separated points. Flux fitting
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
