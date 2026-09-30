---
title: Equilibrium representations
author: VEST team
date: 2026-09-28 09:00
category: guide
layout: post
permalink: /reference/equilibrium-representations/
guide:
  architecture: One canonical EquilibriumData, and five kinds of thing derived from it (coordinate, projection, representation, derived quantity, solver convention), each with its VAFT entry points.
  prerequisites: An equilibrium record; the packaged 39915 sample is enough.
  expected: For every branch of the equilibrium hierarchy, which category it belongs to, which public API implements it, what provenance it carries, and what is still missing.
related:
  notebooks: [equilibrium-representations]
  api: [process, formula, plot, diagram, code]
---

A tokamak equilibrium reaches an analysis as $\psi(R,Z)$, a $q(\psi)$ profile, a
$\rho_{tor,N}$ grid, a set of Miller numbers, Fourier boundary coefficients, a PEST grid, camera
pixels, a field-line trajectory, an $m/n$ helical perturbation or a local transport input. These
forms are related, but they are not interchangeable. This page names the five kinds of object
VAFT derives from an equilibrium, maps the hierarchy of issue #1201 onto the functions that
implement it, and lists what is not yet implemented.

Every branch starts from one canonical record, `vaft.data.EquilibriumData`, built by
`vaft.process.as_equilibrium` from an ODS, a GEQDSK or an analytic generator. The record keeps the
source's convention and does not convert it on construction. What follows is always

```text
canonical state  +  explicit transformation  =  derived object (with provenance)
```

and never one object that holds every coordinate, projection and approximation at once.

## The five categories

| Category | Answers | VAFT examples | Is not |
| --- | --- | --- | --- |
| coordinate | *where* a quantity is parameterized | $\psi_N$, $\rho_{pol,N}$, $\rho_{tor,N}$, PEST $\theta^*$ | $\rho_{tor,N} = \sqrt{\psi_N}$ |
| projection | *where* the same object is viewed | R–Z plane, top view, 3-D embedding, camera pixels | a camera overlay as a 2-D equilibrium |
| representation | *how* the state is encoded or approximated | Miller surface, `FourierSurface`, Solov'ev fit, prescribed island | Miller as an arbitrary LCFS |
| derived quantity | *what physics* follows from the state | $q$, shear, $a/L_T$, rational surfaces, shaping | $a/L_T$ as a radial coordinate |
| solver convention | *how an external code* stores it | COCOS index, Wb vs Wb/rad, DCON/GPEC Jacobian, TGLF units | a Fourier contour angle as a magnetic angle |

### Coordinate

A coordinate labels positions in the plasma. A quantity is a function *of* it, and changing the
coordinate changes the abscissa but not the physics. Radial coordinates label flux surfaces.
Poloidal coordinates label points on one surface. Two coordinates can share the range $[0,1]$ or
$[0,2\pi)$ and still label different points. The definition, not the range or the name, makes a
coordinate.

* `vaft.process.derive_radial_coordinates(eq)` returns `psi_n`, `rho_pol_n` and `rho_tor_n` as
  `DerivedValue` records. Each record gives its definition, for example
  `sqrt(int_axis^psi q dpsi / int_axis^boundary q dpsi)` for $\rho_{tor,N}$. When the record cannot
  form $\rho_{tor,N}$, it gives a reason and does not substitute $\sqrt{\psi_N}$.
* `vaft.process.MappedPositions` from `equilibrium_mapping_thomson_scattering` and
  `equilibrium_mapping_charge_exchange` gives each kinetic channel in every radial coordinate the
  equilibrium supports ([Equilibrium and kinetic profiles]({{ '/workflows/equilibrium-kinetic-profiles/' | relative_url }})).
* `vaft.process.straight_field_line_map` returns a `StraightFieldLineMap` that evaluates the
  PEST angle $\theta^*$ and $\psi_N$ at any $(R,Z)$ and tabulates `theta` and `theta_star` on a
  surface.
* `vaft.formula.generalized_straight_field_line_angle` gives the PEST, Boozer, Hamada and
  equal-arc angles of one surface. `vaft.formula.sfl_toroidal_angle_shift` gives the matching
  toroidal shift $\nu$. The family is drawn by `vaft.diagram.sfl_coordinate_taxonomy`
  ([Scientific diagrams]({{ '/reference/diagrams/' | relative_url }}#straight-field-line-coordinates)).

**Is not:** $\rho_{tor,N}$ is not $\sqrt{\psi_N}$, and the geometric angle
$\theta = \mathrm{atan2}(Z-Z_a, R-R_a)$ is not the PEST $\theta^*$. The packaged 39915 ODS stores
$\sqrt{\psi_N}$ under `profiles_1d.rho_tor_norm`, so the stored name does not tell you the
definition (see [Gaps](#gaps)).

### Projection

A projection shows the same physical object in another space: the poloidal plane, the machine
$X$–$Y$ plane, a 3-D scene or a detector. It creates no new equilibrium, fits nothing and loses no
physics that the viewing geometry does not remove. It can add calibration: a camera projection
depends on a calibrated camera model as well as on the equilibrium.

* The R–Z flux map, drawn by the `equilibrium_field_psi` plot recipe from a `vaft.plot.Field2D`
  model.
* The top view, `vaft.plot.equilibrium_geometry_topview`.
* The camera view: `vaft.omas.camera_projection_for` returns a calibrated
  `vaft.process.camera_geometry.CameraProjection`, and `vaft.omas.compute_camera_visible_efit_overlay`
  projects the LCFS and flux surfaces into pixels without writing anything back.
* The 3-D axisymmetric embedding: `vaft.process.camera_geometry.sweep_toroidal` sweeps a contour
  toroidally, and `vaft.plot.Geometry3DLayers` / `vaft.plot.render_geometry_3d_layers` draw it.

**Is not:** a camera overlay is not a 2-D equilibrium, and a toroidal sweep of an axisymmetric
surface is not a 3-D equilibrium.

### Representation

A representation encodes the state, or part of it, in another form, usually with fewer degrees
of freedom. It is lossy, so it has to report how well it reproduces what it replaced. A
representation fitted to the state describes it only within its fit validity.

* `vaft.process.fit_miller_surface` returns a `MillerFitResult` with `rms_error`,
  `normalized_rms_error`, `max_error`, `hausdorff_distance`, `accepted` and `reason`. It refuses
  surfaces near an active X-point.
* `vaft.process.fit_fourier_surface` / `evaluate_fourier_surface` / `fit_fourier_surface_sequence`
  work with `vaft.data.FourierSurface`, which records its curve parameter in `angle_convention`
  (`"arc_length"`).
* The analytic ladder: `vaft.process.solovev_to_equilibrium` and `solovev_example(topology=...)`,
  `vaft.process.guazzotto_freidberg_to_equilibrium` for Guazzotto–Freidberg parts 1 and 2, and
  the inverse fit `vaft.process.fit_solovev`. All of them return or consume `EquilibriumData`, so
  an analytic state follows the same downstream paths as a reconstruction.
* A prescribed island: `vaft.process.MagneticIslandSpec` and `magnetic_island_topology`.

**Is not:** a Miller shape is not an arbitrary LCFS. A fit that works on the nearly up-down
symmetric VEST limited boundary says nothing about beans, racetracks or diverted shapes
(`notebooks/edge_and_boundary_representation.ipynb`).

### Derived quantity

A derived quantity is physics computed from the state: a number or profile that the equilibrium
implies, not a new label for positions. It is plotted *against* a coordinate. Its value can
depend on the coordinate and length scale used to define it, and then that choice is part of the
quantity.

* $q$, magnetic shear and $\alpha$ on a Miller surface (`MillerSurface.q`, `.magnetic_shear`,
  `.alpha`), and $q$ recomputed from the PEST map as $F\oint dl/(R|\nabla\psi|)$.
* Shaping observables from `vaft.process.contour_shaping_observables`, and the boundary topology
  record from `vaft.process.derive_boundary_representation` (`Topology`, X-points, strike points,
  gaps).
* Normalized gradients: `vaft.formula.normalized_gradient_scale_length(x, y, a)` and
  `SyntheticKineticProfiles.a_over_L(channel, coordinate)`.

**Is not:** $a/L_T$ is not a radial coordinate. On one $T_e$ profile at $\rho_{tor,N}=0.6$, the
reference notebook gets 3.0, 2.95, 2.07 and 2.95 from four gradient conventions. A gradient is
defined only once its derivative coordinate and its reference length are stated.

### Solver convention

A solver convention is how a code, a file format or an IMAS version stores a state: the COCOS
index, the sign and direction choices it implies, the flux unit (Wb or Wb/rad), the poloidal-angle
Jacobian of a solver mesh, and a code's normalizations. It changes numbers without changing
physics. Quantities from two sources are comparable only after their conventions are converted to
a common one.

* `vaft.data.EquilibriumConvention` (`cocos`, `psi_per_radian`, sign fields, `source`, and
  `contradicted` when the data disagree with the declaration). `vaft.process.convert_cocos(eq, target)`
  converts it explicitly. [Process reference: cocos]({{ '/reference/process/cocos/' | relative_url }}).
* `vaft.process.straight_field_line_tables` reads the angle tables of a DCON/GPEC mesh and
  requires `jacobian=` to be named. `lab_to_straight_field_line` applies them.
* `TGLFNormalisation` in `vaft.code.gacode.tglf.inputs` gives TGLF's $a$, $c_s$, $\rho_s$ and
  $B_\mathrm{unit}$ from explicit definitions, checked against `locpargen`.

**Is not:** a Fourier curve parameter is not a magnetic poloidal angle. `FourierSurface` uses an
arc-length angle, which differs from both the geometric and the PEST angle on the same surface.
Reading a solver's angle (`straight_field_line_tables`) is also different from constructing one
from the equilibrium (`straight_field_line_map`).

## The hierarchy in VAFT

The branches of #1201 §2, their category, and where each starts in the code. **Status** follows the
acceptance audit posted on #1201: *met* means a public API plus a test or documentation demonstrates
it.

| Branch | Category | Canonical entry points | Status |
| --- | --- | --- | --- |
| Radial coordinates | coordinate | `vaft.process.derive_radial_coordinates`, `vaft.process.MappedPositions`, `vaft.process.psi_to_radial` (midplane radii) | partly met |
| Poloidal / magnetic coordinates | coordinate | `vaft.process.straight_field_line_map` (native PEST), `vaft.process.straight_field_line_tables` (read DCON/GPEC), `vaft.formula.generalized_straight_field_line_angle`, `vaft.formula.sfl_toroidal_angle_shift` | partly met |
| Physical-space views | projection | `Profile1D` and `Field2D` in `vaft.plot`, `vaft.plot.equilibrium_geometry_topview`, `vaft.plot.Geometry3DLayers` | partly met |
| Diagnostic projections | projection | `vaft.omas.camera_projection_for`, `vaft.omas.compute_camera_visible_efit_overlay` | partly met |
| Compact geometric representations | representation | `vaft.process.fit_miller_surface`, `vaft.process.fit_fourier_surface`, `vaft.process.evaluate_fourier_surface`, `vaft.data.MXHChebyshevRepresentation` | Miller and Fourier met |
| Analytic hierarchy | representation | `vaft.process.solovev_to_equilibrium`, `vaft.process.solovev_example`, `vaft.process.guazzotto_freidberg_to_equilibrium`, `vaft.process.fit_solovev` | partly met |
| Kinetic / gradient representations | derived quantity | `vaft.formula.normalized_gradient_scale_length`, `vaft.data.SyntheticKineticProfiles`, `vaft.data.GradientProfile` | not met |
| Boundary / topology | derived quantity | `vaft.process.derive_boundary_representation`, `vaft.data.BoundaryRepresentation`, `vaft.data.Topology`, `vaft.process.contour_shaping_observables` | shape vs topology met |
| Perturbed / helical | representation | `vaft.process.MagneticIslandSpec`, `vaft.process.magnetic_island_topology`, `GpecCylindricalOutput` in `vaft.code.gpec` | partly met |
| Conventions (all branches) | solver convention | `vaft.data.EquilibriumConvention`, `vaft.process.convert_cocos`, `vaft.process.make_equilibrium_field_interpolator` | partly met |

The per-function pages are under [Process reference]({{ '/reference/process/' | relative_url }}),
[Formula reference]({{ '/reference/formula/' | relative_url }}) and the
[Plot reference]({{ '/reference/plot/' | relative_url }}).

## The provenance thread

Every object derived from an equilibrium should point back to the same state. Five things identify
that state, and each has a place in the code:

| Thread | Where it lives |
| --- | --- |
| source | `EquilibriumData.metadata`, `EquilibriumConvention.source`; `DerivationProvenance.source_type` and `.source_fields` |
| time | `EquilibriumData.time`; `DerivationProvenance.source_time` |
| COCOS and flux unit | `EquilibriumData.convention.cocos` and `.psi_per_radian`; `DerivationProvenance.convention` |
| method | `DerivationProvenance.method`, `.interpolation`, `.fit_range`, `.radial_coordinate` |
| residual | `DerivedValue.quality`; `MillerFitResult` / `FourierFitResult` errors; `SolovevFit.metrics`; camera `reproj_mean_px` |

Same shot and time does not mean the same equilibrium. The package also ships a standalone 39915
g-file at 319 ms, which is a different EFIT run stored per radian. The source record, not the
timestamp, identifies the state.

The check below runs on the packaged sample. It asserts the COCOS instead of trusting it, then
confirms that each derived object still names the same time and convention.

```python
index = int(np.argmin(np.abs(np.asarray(ods["equilibrium.time"]) - 0.319)))
eq = vaft.process.as_equilibrium(ods, time_index=index, convention=11)
assert eq.convention.cocos == 11 and eq.convention.psi_per_radian is False   # psi in Wb

coords = vaft.process.derive_radial_coordinates(eq)
for name, item in coords.items():
    p = item.provenance
    assert item.available, item.reason
    assert abs(p.source_time - eq.time) < 5e-4 and p.convention.cocos == eq.convention.cocos
    print(f"{name:10s} {item.definition:55s} [{p.method}]")

miller = vaft.process.fit_miller_surface(eq.lcfs)
fourier = vaft.process.fit_fourier_surface(eq.lcfs, modes=6)
print("Miller  nRMS", round(miller.normalized_rms_error, 4), "accepted", miller.accepted)
print("Fourier nRMS", round(fourier.normalized_rms_error, 4), "angle", fourier.surface.angle_convention)
print("Miller provenance carries a convention:", miller.provenance.convention is not None)  # False: #604

eq_rad = vaft.process.convert_cocos(eq, 1)                  # same state, psi per radian
print("flux-span ratio", (eq.psi_boundary - eq.psi_axis) / (eq_rad.psi_boundary - eq_rad.psi_axis))  # 2*pi
```

The last lines show where the thread breaks today. A Miller fit of a bare contour carries no time
or convention (#604), so the caller has to keep them.

The whole chain is `notebooks/equilibrium_representation_reference.ipynb`. It takes the 39915
slice at 0.319 s through the four radial coordinates, the R–Z map, a PEST grid with a $q$
residual, Miller and Fourier fits, $a/L_T$ in four conventions, rational surfaces, a prescribed
island with a field-line check, a 3-D embedding and a camera projection, and checks source, time,
COCOS and flux unit at each step. The GPEC response step is missing because no GPEC output is
packaged.

## Gaps

These are the items the #1201 audit grades *not met* or *partly met*, with the issue that owns
each. "Unowned" means no issue other than #1201 tracks it.

**Not met**

* Gradient semantics. No `gradient_coordinate` / `reference_length` contract, and no $1/L$,
  $a/L$, $R/L$ quantity family (#551; #563 was closed as its duplicate).

**Partly met**

* Radial coordinates:
  * The process layer names coordinates `psi_n` / `rho_*_n`, while profiles and plots use
    `*_norm`.
  * The packaged 39915 ODS stores $\sqrt{\psi_N}$ under `profiles_1d.rho_tor_norm`. The plot
    recipe detects this with `vaft.data._derived.is_rho_pol_proxy`, but a direct reader of the
    leaf does not (unowned).
* Geometric radii. `r_minor` exists only in `vaft.plot` as $(R_{out}-R_{in})/2$, and
  `vaft/plot/onedim.py` uses $r - r_{axis}$. `r_minor_norm` and `r_center` have no definition
  (#479).
* Flux coordinates. The native map is PEST only. There is no direct inverse
  $(\psi,\theta^*)\to(R,Z)$, no Jacobian output, no native Boozer or Hamada map (only the
  per-surface formulas), and no closure or Jacobian residual on the map (#472).
* Angle identity on the read path:
  * `straight_field_line_tables` returns bare `(lab, sfl)` tuples.
  * `vaft/machine_mapping/mhd_linear.py` defaults an unrecorded Jacobian to `"hamada"` (unowned).
* On-surface view. There is no first-class $f(\theta \mid \psi)$ view; the notebook builds one
  from `StraightFieldLineMap.surface()` (#1201 phase E).
* Projections:
  * There is no projection abstraction and no general 3-D embedding. `sweep_toroidal` is a camera
    helper that works in centimetres and calls the toroidal angle `theta`.
  * The camera overlay reads raw ODS paths instead of going through `as_equilibrium`.
  * The camera's toroidal frame is not reconciled with the port frame (#746).
* Compact representations:
  * `MillerSurface` mixes geometry with $q$, shear and $\alpha$.
  * `BoundaryRepresentation.fourier_coefficients` and `MXHChebyshevRepresentation` do not record
    their angle convention (#942).
* Flow equilibria. `EquilibriumData` has no governing-equation field. A Guazzotto–Freidberg flow
  case is kept distinct only because exporting it raises (#1149).
* Code presets. Only TGLF resolves its normalization. CGYRO, GS2, GENE and GKW are absent (#551).
* Advanced divertors. Snowflake, X-divertor and Super-X exist only as a TES input (#950).
* Islands and perturbations (unowned; #886 covers the prescribed island):
  * The solver-derived island separatrix is reconstructed only inside the plot backend.
  * No island record says whether it is prescribed or solver-derived.
  * There is no public real-space or diagnostic-space projection of a complex GPEC perturbation.
* Conventions:
  * A default-constructed record falls back to Wb/rad (#603).
  * Miller provenance carries no convention (#604).
* Discovery. No API lists the available coordinates, projections and representations with their
  required source data (#1201 phase H).

Two audit findings have been fixed since the audit:

* The kinetic plot recipes drew $\sqrt{\psi_N}$ under a $\psi_N$ label (#1314).
* `make_equilibrium_field_interpolator` assumed Wb/rad when no COCOS was given. It now reads the
  convention from the record (#1313).

## See also

* [Equilibrium and kinetic profiles]({{ '/workflows/equilibrium-kinetic-profiles/' | relative_url }}):
  radial mapping of kinetic channels.
* [Geometric approximations]({{ '/reference/geometric-approximations/' | relative_url }}): slab,
  cylinder and torus as model geometries, which are a different axis from the representations here.
* [Scientific diagrams]({{ '/reference/diagrams/' | relative_url }}): the tokamak-geometry and
  straight-field-line sections.
* Notebooks: `equilibrium_representation_reference.ipynb`, `edge_and_boundary_representation.ipynb`,
  `compact_equilibrium_representation.ipynb`, `local_miller_equilibrium_fitting.ipynb`,
  `analytic_solovev_equilibrium.ipynb`, `analytic_island_model_and_synthetic_response_model.ipynb`.
