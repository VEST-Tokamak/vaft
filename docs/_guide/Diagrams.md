---
title: Scientific diagrams
author: VEST team
date: 2026-09-17 09:00
category: guide
layout: post
permalink: /reference/diagrams/
guide:
  architecture: Explanatory schematics drawn from vaft.formula physics and rendered to self-contained SVG.
  prerequisites: Nothing to build a diagram or read its TikZ source; latex and dvisvgm to render a new SVG.
  expected: One physical model drawn consistently in every projection, and committed SVG assets whose freshness CI checks.
related:
  api: [diagram, formula, plot]
---

`vaft.diagram` draws **concepts**: topology, coordinate systems, workflows. It does not draw data. The
three layers split the work:

| Layer | Owns | Example |
| --- | --- | --- |
| `vaft.formula` | the physics: every equation a diagram depends on | `helical_phase`, `island_pendulum_hamiltonian` |
| `vaft.diagram` | explanatory geometry: sampling, projection, camera, labels | `magnetic_island(projection="3d")` |
| `vaft.plot` | data and numerical results from an ODS | `equilibrium_2d_profiles` |

A diagram never restates an equation. It calls the formula function, so the picture and the
[formula reference]({{ site.baseurl }}/reference/formula/) cannot drift apart.

This page explains the diagrams family by family. The complete list -- every builder, every committed
SVG and the exact call that draws it -- is the generated
[diagram gallery]({{ site.baseurl }}/reference/diagram/).

## Magnetic island

```python
import vaft

d = vaft.diagram.magnetic_island(
    m=3, n=2, width=0.16, phase=0.0, projection="poloidal",
    r_s=0.55, elongation=1.7, triangularity=0.4,   # D-shaped plasma; defaults give a circle
)
d.tikz       # the LaTeX/TikZ source; needs no TeX installation
# With latex and dvisvgm installed:
# d            -> displays inline in Jupyter (SVG)
# d.save("island.svg")
```

| Projection | Poloidal section at $\phi=0$ | Top view $(R,\phi)$ | 3-D view of the O and X helices |
| --- | --- | --- | --- |
| | ![poloidal]({{ '/assets/diagrams/magnetic_island_poloidal.svg' | relative_url }}) | ![top]({{ '/assets/diagrams/magnetic_island_top.svg' | relative_url }}) | ![3d]({{ '/assets/diagrams/magnetic_island_3d.svg' | relative_url }}) |

All three views are drawn from one model, and the convention is the same in each:

| Quantity | Definition |
| --- | --- |
| helical phase | $\xi = m\theta^* - n\phi - \phi_0$, where $\theta^*$ is the straight-field-line (PEST) angle of each surface, running from the outboard midplane towards the top, and $\phi$ runs counter-clockwise seen from above |
| straight-field-line angle | $\theta^* = 2\pi\int \mathcal{J}/R^2\,d\theta / \oint \mathcal{J}/R^2\,d\theta$ (`vaft.formula.straight_field_line_angle`), taken from the surface geometry alone. O- and X-points are evenly spaced in $\theta^*$, so they spread out on the low-field side even on circular surfaces |
| flux function | $\mathcal{H}(x,\xi) = \tfrac12 x^2 - (w/4)^2\cos\xi$, with $x = r - r_s$ |
| O-points / X-points | $\xi = 0$ (minimum of $\mathcal{H}$) / $\xi = \pi$ (saddle): $m$ of each per poloidal section, alternating |
| separatrix | $\mathcal{H} = (w/4)^2$, so $x_\mathrm{sep} = \tfrac{w}{2}\lvert\cos(\xi/2)\rvert$ |
| `width` | the **full** radial width at the O-point, in units of the minor radius |
| geometry | Miller shaping from `vaft.formula.miller_surface`: $R = R_0 + r\cos(\theta + \arcsin\delta(r)\,\sin\theta)$ and $Z = \kappa r\sin\theta$, with $\delta(r) = \delta\,r$ so the surfaces stay nested and become circular towards the axis. Lengths are in units of the minor radius $a$ |
| caveats | The surfaces are prescribed, with no Shafranov shift, so $\theta^*$ is exact for these surfaces rather than for a Grad-Shafranov equilibrium. `width` is a width in the flux label $r$ and is a physical distance only on the outboard midplane |

The top view suppresses $Z$, so crossings of the projected O and X loci there are not reconnection
points. The separatrix and $w$ are only visible in the poloidal section.

## Stability and operational-space diagrams

Textbook 2-D charts: two axes, the boundaries that divide the plane, and one label per region. A
boundary that physics defines is computed by `vaft.formula`. Only the peeling–ballooning boundary,
which has no closed form, is a schematic, and the figure says so.

```python
vaft.diagram.peeling_ballooning()
vaft.diagram.s_alpha_ballooning(s_max=1.5, alpha_max=3.5)
vaft.diagram.hugill(elongation=1.0, q_limit=2.0)          # no size parameter: R, a, B cancel
vaft.diagram.troyon(beta_N_max=2.8, aspect_ratio=3.0, elongation=1.7)
```

| | |
| --- | --- |
| ![peeling-ballooning]({{ '/assets/diagrams/peeling_ballooning.svg' | relative_url }}) | ![s-alpha]({{ '/assets/diagrams/s_alpha_ballooning.svg' | relative_url }}) |
| ![Hugill]({{ '/assets/diagrams/hugill.svg' | relative_url }}) | ![Troyon]({{ '/assets/diagrams/troyon.svg' | relative_url }}) |

| Diagram | Question it answers | Axes | Boundaries |
| --- | --- | --- | --- |
| Peeling–ballooning | Which edge instability limits the pedestal? | $\alpha_\mathrm{max}$, $J_{B,\mathrm{max}}$ (arbitrary units) | **Schematic.** Two linear margins joined by a smooth maximum. The ★, where the peeling and ballooning limits meet (typical ELM onset), is computed where the two margins are equal |
| $s$–$\alpha$ | How does shear set the ballooning limit, and where is second stability? | $\alpha$, $s$ | The first and second stability boundaries come from `s_alpha_marginal_alpha`, which applies Newcomb's criterion to the Connor–Hastie–Taylor equation. The dashed line is the $0.6\,s$ approximation of `ballooning_stability_criterion`. Not resolved below $s \approx 0.05$ |
| Hugill | Where are the density and low-$q$ disruption limits? | $\bar n_e R/B_T$, $1/q_\mathrm{cyl}$ | The Greenwald line comes from `greenwald_density` and `q_cyl_from_B_R_epsilon_kappa_I`. Its slope depends only on $\kappa_a$: $50\kappa_a/\pi$. The low-$q$ limit is $q_\mathrm{cyl} = q_\mathrm{limit}$ |
| Troyon | How much pressure can the current hold? | $I_p/(aB_T)$, $\beta_T$ | The beta limit is the line on which `beta_N_from_beta_a_B0_Ip` equals $\beta_{N,\max}$. The low-$q$ cutoff comes from `q_cyl_from_B_R_epsilon_kappa_I` |

These charts show the *boundaries* of an operating space. For how measured discharges are projected
onto the same axes, see #944 (operational-space projections) and #636 (Hugill and Greenwald
analysis).

## Single-particle motion

Gyration and guiding-centre drifts, drawn in the island family's style and 3-D camera. Every orbit
is **integrated from the Lorentz force** by `vaft.formula.boris_orbit`, and every drift arrow is the
drift formula of `vaft.formula.particle`. The tests require the integrated guiding centre to move at
the formula's velocity, so the figure cannot show a drift that the orbit does not make.

```python
vaft.diagram.exb_drift(mass_ratio=4.0)
vaft.diagram.curvature_drift()
vaft.diagram.magnetization_current()
vaft.diagram.toroidal_drift(aspect_ratio=2.2)
```

| | |
| --- | --- |
| ![E x B drift]({{ '/assets/diagrams/exb_drift.svg' | relative_url }}) | ![curvature drift]({{ '/assets/diagrams/curvature_drift.svg' | relative_url }}) |
| ![magnetization current]({{ '/assets/diagrams/magnetization_current.svg' | relative_url }}) | ![toroidal drift]({{ '/assets/diagrams/toroidal_drift.svg' | relative_url }}) |

| Diagram | Shows | Arrows from |
| --- | --- | --- |
| E×B drift | An ion and an electron at equal energy in crossed uniform fields. The orbits differ in size, but the drift is the same, so no current flows | `exb_drift_velocity` |
| Curvature and ∇B drift | An ion spirals along a field line of $B_0R_0/R\,\hat\phi$ and drifts along $+z$ | `grad_b_drift_velocity` + `curvature_drift_velocity` |
| Magnetization current | Gyro-currents cancel inside a region. At its edge the diamagnetic $\mathbf{J}_M = \nabla\times\mathbf{M}$ survives | the binned current of the integrated orbits |
| Toroidal drift | ∇B and curvature drifts separate charge. The resulting vertical $\mathbf{E}$ drives an outward $\mathbf{E}\times\mathbf{B}$, so a purely toroidal field cannot confine | the drift formulas at the drawn cross-section |

Every view shows the relevant equations in a box. They are read from the `$$…$$` definition in each
formula's docstring, so the figure shows exactly what the formula documents and implements. There is
no second copy of any equation.

Each has further projections of the same computed orbits, chosen with `projection=`:

| Diagram | Projections (default first) | What the extra views add |
| --- | --- | --- |
| `exb_drift` | `perpendicular`, `3d` | Helices along $\mathbf{B}$ drifting sideways. The parallel velocity does not change the perpendicular motion, so both views show the same orbits |
| `curvature_drift` | `3d`, `poloidal`, `top` | `poloidal` looks along the field line, where the gyration circle climbs at the drift velocity. `top` shows the curved line and the inward $\nabla B$ |
| `magnetization_current` | `perpendicular`, `3d` | Helical columns along $\mathbf{B}$ with the edge current looping around them |
| `toroidal_drift` | `3d`, `poloidal`, `top` | `poloidal` is the textbook $(R, z)$ cross-section. `top` shows the circular field lines, with the vertical drifts pointing out of the page |

The units are normalised ($|q| = 1$, $m_e = 1$, fields of order one). The ion-to-electron mass ratio is reduced (4 by
default) so that both orbits are visible; the figures state this.

## Tearing physics

The ideas upstream of an island, **one concept per diagram**, so each can be used on its own in a
page, notebook or slide. None takes an equilibrium, shot or solver output: the curves are schematic.

```python
vaft.diagram.rational_surface(m=2, n=1)
vaft.diagram.delta_prime(sign="positive")   # "positive", "zero" or "negative"
vaft.diagram.tearing_layer_matching()
```

| | | |
| --- | --- | --- |
| ![rational surface]({{ '/assets/diagrams/rational_surface.svg' | relative_url }}) | ![Delta prime]({{ '/assets/diagrams/delta_prime.svg' | relative_url }}) | ![layer matching]({{ '/assets/diagrams/tearing_layer_matching.svg' | relative_url }}) |

| Diagram | Question | What is schematic |
| --- | --- | --- |
| `rational_surface` | Where does a perturbation resonate with the field-line pitch, $q(r_s) = m/n$? | The monotonic $q(r)$; it is not an equilibrium profile |
| `delta_prime` | What does the tearing stability index measure? | The outer solutions: quadratics that vanish on the axis and at the edge. Their slopes at $r_s$ go through `vaft.formula.delta_prime_from_outer_derivatives`, and only the sign of $\Delta'$ is meaningful |
| `tearing_layer_matching` | Why are the ideal outer regions and the non-ideal inner layer solved separately? | The layer width and the layer solution, which only joins the outer solutions in value and slope |

The index drawn here is the definition, not a stability result: no outer equation is solved. Solver
values of $\Delta'$ (RDCON, STRIDE) belong to `vaft.plot`, and they are not the RDCON $D_R$ or the
Modified Rutherford terms. The object these lead to is `magnetic_island`.

## 3-D perturbation harmonics

How a linear 3-D perturbation is written as complex Fourier harmonics, and what the complex numbers
mean. Each diagram is one concept and needs no equilibrium, shot or GPEC output.

```python
vaft.diagram.normal_field_component()
vaft.diagram.complex_harmonic(amplitude=1.0, phase=1.05)
vaft.diagram.toroidal_harmonic_phase(n=1)
vaft.diagram.harmonic_real_space_projection(m=2, n=1, phase=1.05)
vaft.diagram.complex_field_superposition(case="screening")   # "amplification", "phase_shift"
```

| | |
| --- | --- |
| ![normal component]({{ '/assets/diagrams/normal_field_component.svg' | relative_url }}) | ![complex harmonic]({{ '/assets/diagrams/complex_harmonic.svg' | relative_url }}) |
| ![toroidal phase]({{ '/assets/diagrams/toroidal_harmonic_phase.svg' | relative_url }}) | ![superposition]({{ '/assets/diagrams/complex_field_superposition.svg' | relative_url }}) |

![real-space projection]({{ '/assets/diagrams/harmonic_real_space_projection.svg' | relative_url }})

| Diagram | Concept |
| --- | --- |
| `normal_field_component` | $\delta B_n = \delta\mathbf B\cdot\hat{\mathbf n}$ is the part of a perturbation that crosses a magnetic surface. It is geometric only, with no code-specific normalisation |
| `complex_harmonic` | One harmonic is $\hat b = b_R + i\,b_I = A e^{i\alpha}$. $b_R$ and $b_I$ are the cosine and sine quadratures of one pattern, not two fields |
| `toroidal_harmonic_phase` | Moving the toroidal origin by $\Delta\phi$ turns $\hat b$ by $-n\Delta\phi$. $\lvert\hat b\rvert$ is invariant; $b_R$ and $b_I$ are not |
| `harmonic_real_space_projection` | The physical field is real, $\delta b = \mathrm{Re}[\hat b\,e^{i(m\theta - n\phi)}]$ (`vaft.formula.helical_harmonic`): stripes of slope $n/m$ on the unwrapped $(\phi, \theta)$ plane |
| `complex_field_superposition` | External and plasma-response fields add as complex numbers, so screening, amplification and phase shift all come from one vector sum |

The phase convention is `helical_phase`'s $\xi = m\theta - n\phi$, with both mode numbers positive and
the helicity in the minus sign. `vaft.code.gpec` stores each complex quantity as a real/imaginary pair
(`i = 0` real, `i = 1` imaginary) and rebuilds it as `real + 1j * imag`, deciding no convention. For
GPEC's spectral outputs that pair is the $(b_R, b_I)$ of `complex_harmonic`, and it becomes a field only
through the real-space reconstruction. Two sources differ:

* the `*_fun` quantities (`b_n_fun`, `xi_n_fun`) are already real-space in $\theta$, and GPEC writes
  them as $(\mathrm{Re}, -h\,\mathrm{Im})$ with its helicity $h$;
* `vaft.process.toroidal_mode_decomposition` returns the conjugate, $\hat b = 2\,\overline{C_n}$.

`helical_harmonic`'s Convention section states both. Amplitudes, phases and responses in these
figures are schematic.

## Collision processes

A classification of the interactions in a fusion plasma, built from the concept-diagram primitives.
Coulomb collisions between charged particles relax the distribution. Atomic processes change charge
states and bound electrons. Nuclear reactions change nuclei.

```python
vaft.diagram.collision_processes()
```

![collision processes]({{ '/assets/diagrams/collision_processes.svg' | relative_url }})

The diagram contains no numbers except the D–T alpha energy, which is `vaft.formula.constants.E_ALPHA`. Collision frequencies, the
Coulomb logarithm and collisionality regimes are left to formula-backed diagrams (#1111).

## Geometric approximations

How slab, cylindrical and toroidal models relate, keeping geometry and ordering on separate axes. The
physics is in `vaft.formula.geometry`. [Geometric approximations]({{ '/reference/geometric-approximations/' | relative_url }})
explains each representation.

```python
vaft.diagram.geometry_ordering_map()
vaft.diagram.field_line_geometry(geometry="toroidal")   # "cylindrical", "slab"
vaft.diagram.mode_number_mapping(m=2, n=1)
```

| Diagram | Concept |
| --- | --- |
| `geometry_ordering_map` | Geometries are columns and orderings are bands. Each reduction arrow names what it keeps or drops |
| `field_line_geometry` | The same $q$ field line on a torus and on the cylinder straightened at $R_0$, and the tilt of the sheared-slab field lines growing with $x$ |
| `mode_number_mapping` | The cylinder's $k_\parallel(r)$ crosses zero at $q(r_s) = m/n$. The local slab of `local_slab_from_cylinder` is its tangent there |

## Tokamak geometry and flux coordinates

The parent geometry that the cylindrical and slab reductions start from. Surfaces are
`miller_surface`, the shift is `shafranov_shift_from_r_a_R0_beta_p_li`, the field is
`vacuum_toroidal_field`, and $\theta^*$ is `straight_field_line_angle`.

```python
vaft.diagram.tokamak_torus(projection="3d")        # "poloidal"
vaft.diagram.flux_surfaces(shape="circular")       # "shifted"
vaft.diagram.shaping_family()
vaft.diagram.hfs_lfs_field()
vaft.diagram.safety_factor_winding(q=3)
vaft.diagram.flux_coordinates()
vaft.diagram.poloidal_angle_comparison()
vaft.diagram.unwrapped_flux_surface(q=2.5)
vaft.diagram.field_line_pitch(q=1.0)
```

| | |
| --- | --- |
| ![torus]({{ '/assets/diagrams/tokamak_torus_3d.svg' | relative_url }}) | ![cross-section]({{ '/assets/diagrams/tokamak_torus_poloidal.svg' | relative_url }}) |
| ![concentric]({{ '/assets/diagrams/flux_surfaces_circular.svg' | relative_url }}) | ![Shafranov shift]({{ '/assets/diagrams/flux_surfaces_shifted.svg' | relative_url }}) |
| ![HFS/LFS]({{ '/assets/diagrams/hfs_lfs_field.svg' | relative_url }}) | ![safety factor]({{ '/assets/diagrams/safety_factor_winding.svg' | relative_url }}) |
| ![flux coordinates]({{ '/assets/diagrams/flux_coordinates.svg' | relative_url }}) | ![theta vs theta*]({{ '/assets/diagrams/poloidal_angle_comparison.svg' | relative_url }}) |

![field-line pitch]({{ '/assets/diagrams/field_line_pitch.svg' | relative_url }})

![shaping]({{ '/assets/diagrams/shaping_family.svg' | relative_url }})

![unwrapped surface]({{ '/assets/diagrams/unwrapped_flux_surface.svg' | relative_url }})

| Diagram | Concept |
| --- | --- |
| `tokamak_torus` | $R_0$, $a$, $\phi$ (counter-clockwise from above) and $\theta$ (from the outboard midplane), with $R = R_0 + r\cos\theta$ |
| `flux_surfaces` | Concentric surfaces, then the Shafranov shift: $\Delta(r)$ is zero at the edge and largest on axis, so the magnetic axis sits outside the geometric axis |
| `shaping_family` | Circular, $\kappa$, $\delta$, and both. Positive triangularity pulls the top in to $R_0 - \delta r$ |
| `hfs_lfs_field` | $B_\phi = B_0R_0/R$ is stronger on the inboard (high-field) side |
| `field_line_pitch` | $\mathbf B = B_\phi\hat{\boldsymbol\phi} + B_\theta\hat{\boldsymbol\theta}$ at a point of a field line, with $B_\theta/B_\phi = r/(qR)$ so that $\mathbf B$ lies along the line |
| `safety_factor_winding` | $q$ toroidal turns per poloidal turn, counted at one cross-section |
| `flux_coordinates` | $(\psi, \theta, \phi)$, with $+\phi$ into the page when $R$ is to the right and $Z$ is up |
| `poloidal_angle_comparison` | On a D shape, equal steps of $\theta^*$ are not rays of the geometric angle |
| `unwrapped_flux_surface` | A field line is straight, $d\phi/d\theta^* = q$, in straight-field-line coordinates, and not in the parametrisation angle $\theta$ |

The field line on a torus, its cylindrical and slab reductions, and the geometry/ordering map are in
the geometric-approximations section above. The toroidal → cylindrical → slab bridge is
`geometry_ordering_map` together with `field_line_geometry`.

## Toroidicity and TF ripple

From the $1/R$ mirror to ripple-induced fast-particle transport. The formulas are in
[`vaft.formula.ripple`]({{ '/reference/formula/ripple/' | relative_url }}), and each one names its diagram
under *See Also*. $B_\phi \propto 1/R$ itself is `hfs_lfs_field`, in the tokamak-geometry section.

```python
vaft.diagram.trapped_and_passing_orbits()
vaft.diagram.toroidal_field_ripple(n_tf=16)
vaft.diagram.ripple_well_formation()
vaft.diagram.stochastic_ripple_orbit()
```

| | |
| --- | --- |
| ![trapped and passing]({{ '/assets/diagrams/trapped_and_passing_orbits.svg' | relative_url }}) | ![TF ripple]({{ '/assets/diagrams/toroidal_field_ripple.svg' | relative_url }}) |
| ![ripple wells]({{ '/assets/diagrams/ripple_well_formation.svg' | relative_url }}) | ![stochastic tips]({{ '/assets/diagrams/stochastic_ripple_orbit.svg' | relative_url }}) |

| Diagram | Concept | Formula |
| --- | --- | --- |
| `trapped_and_passing_orbits` | $\mu$ and energy conservation in $B \propto 1/R$: small pitches bounce as bananas, large ones pass | `parallel_speed_from_mu`, `vacuum_toroidal_field` |
| `toroidal_field_ripple` | $N_\mathrm{TF}$ coils corrugate $B(\phi)$: maximal under a coil, minimal between | `toroidal_ripple_field`, `ripple_amplitude` |
| `ripple_well_formation` | Along a field line the ripple makes local wells where $\alpha^* \lesssim 1$, near the midplanes (to first order in $\epsilon$) | `ripple_well_parameter` |
| `stochastic_ripple_orbit` | Ripple kicks at banana tips decorrelate above $\delta_\mathrm{GWB}$. Drawn as the standard map with $K \sim \delta/\delta_\mathrm{GWB}$ | `gwb_stochastic_threshold`, `gwb_stochasticity_parameter` |

These are regime indicators, not a loss calculation. Orbit following (ASCOT, NUBEAM) is the
quantitative check. Low-$n$ error fields, NTV and locking are separate topics.

## Guiding-centre invariants and toroidal symmetry

This is the global view of the orbit physics that `curvature_drift` and `toroidal_drift` show locally.
The grad-B and curvature drifts say why a guiding centre moves at each instant. Conservation of
$P_\phi$ constrains the whole orbit in an axisymmetric field. These are the same physics seen two
ways, not competing explanations.

```python
vaft.diagram.guiding_center_invariants()
vaft.diagram.canonical_toroidal_momentum(phase=0.45)
vaft.diagram.toroidal_symmetry_breaking()
```

| | |
| --- | --- |
| ![invariants]({{ '/assets/diagrams/guiding_center_invariants.svg' | relative_url }}) | ![P_phi]({{ '/assets/diagrams/canonical_toroidal_momentum.svg' | relative_url }}) |

![symmetry breaking]({{ '/assets/diagrams/toroidal_symmetry_breaking.svg' | relative_url }})

| Diagram | Concept | Formula |
| --- | --- | --- |
| `guiding_center_invariants` | Gyration, bounce and toroidal drift, with invariants $\mu$, $J_\parallel = \oint p_\parallel\,dl$ and $P_\phi$, valid for $\Omega_c \gg \omega_b \gg \omega_d$ | `magnetic_moment` |
| `canonical_toroidal_momentum` | A banana built from $\mu$ and $P_\phi$ conservation. From the bounce tip, $\Delta(q\psi)$ and $\Delta(mv_\parallel Rb_\phi)$ cancel, so $\psi$ moves with $v_\parallel$: this is the orbit width. `phase` is the state a future animation steps | `guiding_center_toroidal_momentum`, `parallel_speed_from_mu` |
| `toroidal_symmetry_breaking` | A 3-D field changes $P_\phi$. Away from resonance the change oscillates; where $\Delta\omega_\mathrm{BH} = 0$ it is secular | `bounce_harmonic_detuning` |

**Conventions.**
* $P_\phi = mRv_\phi + qRA_\phi$ (`canonical_toroidal_momentum`) uses physical components and the
  IMAS $\phi$.
* The guiding-centre form uses $\psi = RA_\phi$ in **Wb per radian**, largest on the axis for a current
  along $+\phi$. `psi_per_radian_from_cocos` converts a stored flux: $-\psi/2\pi$ for COCOS 11
  (IMAS DD3) and $+\psi/2\pi$ for COCOS 17 (DD4). Used as stored, the flux has the wrong sign or a
  $2\pi$ error.
* $J_\parallel$ is documented but deliberately not a numerical helper: its bounce interval and
  orientation depend on the orbit.
* The drift of one orbit's $P_\phi$ is not NTV. Torque and transport are #1111's, and need the kinetic
  response of the whole distribution.

## Straight-field-line coordinates

Straight field lines are a condition, not a coordinate system. The freedom the condition leaves is what
PEST, Boozer, Hamada and equal-arc fix in different ways. All four are members of one generalised
family, $\mathcal{J} \propto R^{p_R}/(B_p^{p_{Bp}}B^{p_B})$ (`vaft.formula.generalized_straight_field_line_angle`,
as in DCON/GPEC). The unwrapped picture of a straight field line is `unwrapped_flux_surface`.
Clebsch, field-aligned and ballooning coordinates are #1075's.

```python
vaft.diagram.sfl_coordinate_grids()
vaft.diagram.sfl_coordinate_taxonomy()
vaft.diagram.sfl_fourier_convergence()
```

![grids]({{ '/assets/diagrams/sfl_coordinate_grids.svg' | relative_url }})

| | |
| --- | --- |
| ![taxonomy]({{ '/assets/diagrams/sfl_coordinate_taxonomy.svg' | relative_url }}) | ![spectra]({{ '/assets/diagrams/sfl_fourier_convergence.svg' | relative_url }}) |

| Coordinate | $(p_{Bp}, p_B, p_R)$ | Geometric $\phi$ kept | Also simplified | Typical use |
| --- | --- | --- | --- | --- |
| PEST | $(0, 0, 2)$ | yes | the toroidal angle | classical MHD stability |
| Boozer | $(0, 2, 0)$ | no ($\zeta = \phi + \nu$) | $\mathcal{J} \propto B^{-2}$, the $B$ spectrum | orbits, neoclassical, 3-D |
| Hamada | $(0, 0, 0)$ | no | flux-function Jacobian; current lines straight too | MHD stability |
| equal-arc | $(1, 0, 0)$ | no ($\zeta = \phi + \nu$) | uniform poloidal sampling | numerical representation |

* **Same equilibrium, different grids.** All four panels of `sfl_coordinate_grids` share one
  equilibrium: $R_0/a = 1.7$, $\kappa = 2$, $\delta = 0.45$. The surfaces are identical and only the
  angle changes. A test checks this.
* **Different Fourier costs.** In `sfl_fourier_convergence`, one outboard-localised, axisymmetric
  perturbation needs anywhere from 9 to 26 poloidal harmonics, depending on the angle it is expanded
  in. Moved inboard, the ranking reverses: which angle is compact depends on where the structure sits.
  For $n \ne 0$, the toroidal shift $\nu$ of every angle except PEST couples harmonics further. The
  figure leaves that out.
* **Only PEST keeps $\phi$.** Keeping the geometric $\phi$ forces $\mathcal{J} \propto R^2$, so every
  other member has $\zeta = \phi + \nu$.
* **Boozer ≈ PEST here.** The two nearly coincide at this $B_p \ll B_\phi$, where $B^2 \propto R^{-2}$.
* **COCOS is not a coordinate choice.** It fixes signs and orientations across every node of the
  taxonomy, independently of which angle is chosen.

## Clebsch labels and the ballooning representation

These figures go from straight-field-line coordinates to the local, field-aligned and ballooning
descriptions that high-$n$ stability and turbulence models use. The formulas are
`field_line_label`, `s_alpha_curvature_drive` and `s_alpha_ballooning_solution` in
`vaft.formula.stability`, on the circular $s$–$\alpha$ model.

```python
vaft.diagram.clebsch_field_line_label(q=2.5)
vaft.diagram.ballooning_curvature_drive()
vaft.diagram.ballooning_newcomb_test()
vaft.diagram.ballooning_harmonic_envelope(n=20)
vaft.diagram.ballooning_workflow()
```

| | |
| --- | --- |
| ![Clebsch]({{ '/assets/diagrams/clebsch_field_line_label.svg' | relative_url }}) | ![curvature]({{ '/assets/diagrams/ballooning_curvature_drive.svg' | relative_url }}) |
| ![Newcomb test]({{ '/assets/diagrams/ballooning_newcomb_test.svg' | relative_url }}) | ![harmonics]({{ '/assets/diagrams/ballooning_harmonic_envelope.svg' | relative_url }}) |

![workflow]({{ '/assets/diagrams/ballooning_workflow.svg' | relative_url }})

| Diagram | Concept |
| --- | --- |
| `clebsch_field_line_label` | A field line is where $\psi$ = const meets $\alpha = \phi - q\theta$ = const. In `helical_phase`'s left-handed $(\psi, \theta, \phi)$, with $\psi$ rising outward and $\mathbf B$ along $+\phi$, $\mathbf B \propto \nabla\psi\times\nabla\alpha$. Right-handed coordinates give the Connor–Hastie–Taylor form $\nabla\alpha\times\nabla\psi$ |
| `ballooning_curvature_drive` | Normal curvature $\cos\theta > 0$ (bad, outboard, shaded) once per $2\pi$ period of the covering space, and the total drive $K = \cos\theta + \Lambda\sin\theta$. Its outer lobes come from the geodesic term, which grows with the local shear. Also shown with $\theta_0$ |
| `ballooning_newcomb_test` | Newcomb's test on the marginal solution $F(\theta)$: it stays positive (stable), crosses zero (unstable), or is positive again beyond $\alpha_2$ (second stability). These are not localised eigenfunctions |
| `ballooning_harmonic_envelope` | $a_m = \hat F(m - nq)$ for a stated model envelope: many coupled harmonics around $nq$. Their sum oscillates at $m \approx nq$ under the envelope, localised outboard |
| `ballooning_workflow` | Straight-field-line coordinates → field-line label → field-aligned → ballooning / flux tube, and the infinite-$n$ path. A finite-$n$ global calculation keeps what that path drops |

The straight-field-line coordinates they start from are in the section above (`sfl_coordinate_taxonomy`).
The resulting $(s, \alpha)$ stability diagram is `s_alpha_ballooning`.

## Slab resonant layers: tearing and twisting parity

How a global harmonic becomes a local layer response. The mapping $(m, n) \to (k_y, k_z)$ and
$q = m/n \Leftrightarrow k_\parallel = 0$ is `mode_number_mapping`. The flux is
`vaft.formula.slab_perturbed_flux`; its tearing form is the island pendulum of `magnetic_island`.

```python
vaft.diagram.slab_parity(parity="tearing")   # "twisting"
vaft.diagram.slab_parity_comparison()
vaft.diagram.poloidal_harmonic_coupling(m=3)
vaft.diagram.resonant_layer_matching()
```

![parity]({{ '/assets/diagrams/slab_parity_comparison.svg' | relative_url }})

| | |
| --- | --- |
| ![coupling]({{ '/assets/diagrams/poloidal_harmonic_coupling.svg' | relative_url }}) | ![matching]({{ '/assets/diagrams/resonant_layer_matching.svg' | relative_url }}) |

| Diagram | Concept |
| --- | --- |
| `slab_parity` | Contours of $\Psi_T = B_s'x^2/2 + \psi_0\cos k_yy$: an island of width $4\sqrt{\psi_0/B_s'}$, O- and X-points, and $\delta B_x(0) \ne 0$. Contours of $\Psi_W = B_s'x^2/2 + \psi_1x\cos k_yy$, drawn with its $O(\psi_1^2)$ completion: no normal field at the layer ($k_\parallel = 0$) and no reconnection, and every surface, the rational one included, displaced together by $\xi = -\psi_1\cos k_yy/B_s'$ |
| `slab_parity_comparison` | Both side by side, with the parity of $\tilde\psi$ and $\tilde\phi$, $\delta B_x(0)$ and the topology |
| `poloidal_harmonic_coupling` | $\cos\theta$ (toroidicity) couples $m \to m \pm 1$ and $\cos 2\theta$ (elongation) couples $m \to m \pm 2$, at fixed $n$. The harmonic index $m$ is not the parity |
| `resonant_layer_matching` | Every rational surface of one $n$ has a T and a W channel. The outer region couples them all into one $2N\times2N$ matrix (RDCON/STRIDE), and each layer is solved on its own (SLAYER) |

## Cylindrical geometry: profiles, mode shapes and matching

The screw-pinch picture between the torus and the slab. The profile is Wesson's peaked current
$j \propto (1 - x^2)^\nu$, given by `peaked_current_safety_factor` together with
`cylindrical_poloidal_field` (Ampère). The screw-pinch field line is
`field_line_geometry("cylindrical")`, and the cylinder-versus-torus harmonic contrast is
`poloidal_harmonic_coupling`.

```python
vaft.diagram.current_to_q_profile(nu=1.0, q_a=3.5)
vaft.diagram.cylindrical_rational_surfaces(n=1)
vaft.diagram.cylindrical_mode_morphology()
vaft.diagram.internal_external_kink()
vaft.diagram.plasma_vacuum_wall(m=2)
vaft.diagram.cylindrical_tearing_outer(m=2, n=1)
```

![profiles]({{ '/assets/diagrams/current_to_q_profile.svg' | relative_url }})

| | |
| --- | --- |
| ![rational surfaces]({{ '/assets/diagrams/cylindrical_rational_surfaces.svg' | relative_url }}) | ![mode shapes]({{ '/assets/diagrams/cylindrical_mode_morphology.svg' | relative_url }}) |
| ![kinks]({{ '/assets/diagrams/internal_external_kink.svg' | relative_url }}) | ![wall]({{ '/assets/diagrams/plasma_vacuum_wall.svg' | relative_url }}) |

![tearing outer]({{ '/assets/diagrams/cylindrical_tearing_outer.svg' | relative_url }})

| Diagram | Concept |
| --- | --- |
| `current_to_q_profile` | $j \to B_\theta \to q$. More peaked current means lower $q_0 = q_a/(\nu+1)$ and stronger shear |
| `cylindrical_rational_surfaces` | For one $n$, one surface $q(r_s) = m/n$ for each $m$ between $q_0$ and $q_a$ |
| `cylindrical_mode_morphology` | $m = 0$ sausage, $m = 1$ kink (a rigid shift), $m = 2, 3$ helical distortions |
| `internal_external_kink` | The $m = 1$ internal kink lives inside $q = 1$. An external kink reaches the boundary. Kruskal–Shafranov is a heuristic |
| `plasma_vacuum_wall` | One harmonic matched across plasma, vacuum ($Ar^m + Br^{-m}$) and an ideal wall |
| `cylindrical_tearing_outer` | The outer solutions at $r_s$ and their $\Delta'$, for the outer, ideal problem. The inner layer at $r_s$ is `slab_parity` |

From the cylinder to the slab: `mode_number_mapping` expands $k_\parallel(r)$ about $r_s$ into the local
slab of `local_slab_from_cylinder`, and `resonant_layer_matching` couples several such layers.
The screw-pinch field line itself is `field_line_geometry("cylindrical")`, and the cylinder-vs-torus harmonic
picture (independent $m$ vs toroidally coupled $m, m\pm1$) is `poloidal_harmonic_coupling`.

## Field configurations, reconnection and MHD waves

The canonical slab configurations, the topology of reconnection, and the linear ideal-MHD waves.
All are drawn in the slab frame of `vaft.formula.geometry`: $x$ is the sheet normal (radial), $y$ the
reconnecting (binormal) direction, and $z$ the current and guide-field direction. The geometry itself
(slab, sheared slab, cylinder, torus) belongs to [Geometric approximations]({{ '/reference/geometric-approximations/' | relative_url }}).
The formulas are `harris_sheet_field`, `harris_sheet_current_density` and `x_point_flux` in `geometry`,
and `shear_alfven_frequency` and `magnetosonic_phase_speeds` in `stability`.

```python
vaft.diagram.slab_field_configuration(kind="sheared")   # "uniform", "reversed", "guide"
vaft.diagram.current_sheet(guide_field=False)
vaft.diagram.harris_sheet()
vaft.diagram.x_point()
vaft.diagram.magnetic_reconnection()
vaft.diagram.island_formation()
vaft.diagram.shear_alfven_wave()
vaft.diagram.fast_magnetosonic_wave()
vaft.diagram.mhd_wave_family()
```

| | |
| --- | --- |
| ![sheared]({{ '/assets/diagrams/slab_field_configuration_sheared.svg' | relative_url }}) | ![reversed]({{ '/assets/diagrams/slab_field_configuration_reversed.svg' | relative_url }}) |
| ![sheet]({{ '/assets/diagrams/current_sheet.svg' | relative_url }}) | ![harris]({{ '/assets/diagrams/harris_sheet.svg' | relative_url }}) |
| ![x-point]({{ '/assets/diagrams/x_point.svg' | relative_url }}) | ![reconnection]({{ '/assets/diagrams/magnetic_reconnection.svg' | relative_url }}) |
| ![shear Alfven]({{ '/assets/diagrams/shear_alfven_wave.svg' | relative_url }}) | ![fast]({{ '/assets/diagrams/fast_magnetosonic_wave.svg' | relative_url }}) |

![island formation]({{ '/assets/diagrams/island_formation.svg' | relative_url }})

![wave family]({{ '/assets/diagrams/mhd_wave_family.svg' | relative_url }})

| Diagram | Concept |
| --- | --- |
| `slab_field_configuration` | The field on stacked $x$ = const sheets. Uniform; sheared, where the direction rotates and $\lvert\mathbf B\rvert = B_0$ to first order in $x/L_s$; reversed, where $B_y(-x) = -B_y(x)$ with a null at $x = 0$; and reversed with a guide field $B_g$, which rotates with no null. Shear and reversal are different things |
| `current_sheet` | The reversing Harris field seen along the current, with lines at equal flux spacing (spacing $\propto 1/\lvert B_y\rvert$), the sheet of thickness $2a$, its normal, and $\otimes J_z$. With `guide_field=True`, $B_g\hat{\mathbf z}$ removes the null and leaves $J_z$ unchanged |
| `harris_sheet` | $B_y = B_0\tanh(x/a)$ and $J_z = (B_0/\mu_0a)\,\mathrm{sech}^2(x/a)$ on one chart: the reversal and the localized current are the same layer |
| `x_point` | The current-free null $\psi = B'(x^2 - y^2)/2$: four branches and two separatrices at right angles. Geometry only |
| `magnetic_reconnection` | Model-neutral reconnection: inflow, outflow jets, diffusion region, upstream and reconnected field lines about a stretched X-point. No Sweet–Parker, Petschek, Hall or kinetic assumption |
| `island_formation` | `slab_perturbed_flux` with growing $\psi_0$: straight sheared lines, then X- and O-points, then an island of width $w = 4\sqrt{\psi_0/B'}$. The growth is `delta_prime`, `slab_parity` and `tearing_layer_matching` |
| `shear_alfven_wave` | Field-line bending with equally spaced lines, so $\lvert\mathbf B\rvert$ is unchanged to first order. $\delta\mathbf v_\perp$ and $\delta\mathbf B_\perp = -(B_0/v_A)\delta\mathbf v_\perp$ lie normal to the $\mathbf k$-$\mathbf B_0$ plane, and $\omega = \lvert k_\parallel\rvert v_A$ |
| `fast_magnetosonic_wave` | At $\mathbf k \perp \mathbf B_0$ the lines bunch and spread, $\delta B_z = -B_0\partial_x\xi_x$, and $v_f = (v_A^2 + c_s^2)^{1/2}$. It is the compressional Alfvén wave only in the limit $c_s \ll v_A$ |
| `mhd_wave_family` | The Friedrichs diagram of fast, shear-Alfvén and slow phase speeds against the angle to $\mathbf B_0$ ($v_s \le v_A\lvert\cos\theta\rvert \le v_f$), with each branch's restoring force, compressibility and polarization |

From these to the tokamak: the current sheet and the island lead to the tearing layer (`slab_parity`),
then to the cylindrical and toroidal tearing mode (`delta_prime`, `resonant_layer_matching`). Here
$q(r_s) = m/n$ globally is $k_\parallel = 0$ locally (`mode_number_mapping`). The uniform-slab shear
Alfvén wave leads to the Alfvén continuum, where $v_A(r)$ and $k_\parallel(r)$ vary. Toroidal coupling
(`poloidal_harmonic_coupling`) then opens the gaps of the TAE and EAE. Those are not computed here.

## The Grad–Shafranov problem: regions, boundaries and problem classes

The axisymmetric field is $\mathbf B = R^{-1}\nabla\psi\times\hat{\boldsymbol\phi} + F(\psi)R^{-1}\hat{\boldsymbol\phi}$,
with $\psi$ the poloidal flux per radian and $F = RB_\phi$ the poloidal-current function. Its components are
`radial_magnetic_field_from_psi` and `vertical_magnetic_field_from_psi` (COCOS-aware); in vacuum $F$ is constant, and
$B_\phi = F/R$ is `vacuum_toroidal_field`. One flux function $\psi(R, Z)$ is solved across the plasma, vacuum and coil regions, each with its own
source. `vaft.formula.grad_shafranov_source` gives the Ampère form $\Delta^*\psi = -\mu_0RJ_\phi$,
which holds in every region. `toroidal_current_density_from_p_prime_ff_prime` gives the plasma current
that force balance allows, $J_\phi = Rp' + FF'/(\mu_0R)$. On a grid, the operator is
`vaft.process.equilibrium.grad_shafranov_operator`. The flux maps are a toy: prescribed ring currents
and three coils superposed through `green_psi_exact`. The topology (axis, X-point, limiter contact, LCFS)
is then found from the total flux, as a free-boundary code finds it. A production solver would also make
the plasma current consistent with $p'$ and $FF'$.

```python
vaft.diagram.grad_shafranov_domain_decomposition()
vaft.diagram.fixed_vs_free_boundary_equilibrium()
vaft.diagram.limiter_and_diverted_topologies()
vaft.diagram.equilibrium_problem_taxonomy()
vaft.diagram.poloidal_flux_source_decomposition()
```

![domains]({{ '/assets/diagrams/grad_shafranov_domain_decomposition.svg' | relative_url }})

![fixed vs free]({{ '/assets/diagrams/fixed_vs_free_boundary_equilibrium.svg' | relative_url }})

![topologies]({{ '/assets/diagrams/limiter_and_diverted_topologies.svg' | relative_url }})

![taxonomy]({{ '/assets/diagrams/equilibrium_problem_taxonomy.svg' | relative_url }})

![sources]({{ '/assets/diagrams/poloidal_flux_source_decomposition.svg' | relative_url }})

| Diagram | Concept |
| --- | --- |
| `grad_shafranov_domain_decomposition` | Same $\psi$, different $J_\phi$. The plasma source comes from force balance. The vacuum is homogeneous, $\Delta^*\psi = 0$: Laplace-type, but not $\nabla^2$. A coil carries its prescribed current. All of this sits inside the computational boundary |
| `fixed_vs_free_boundary_equilibrium` | The boundary is an input (LCFS and $\psi_b$ given, only the inside solved) or it is part of the solution (coils and sources given, $\psi$ everywhere, LCFS read from the topology) |
| `limiter_and_diverted_topologies` | Limited: the LCFS is the surface through the limiter tip. Diverted: the separatrix through the X-point ($\nabla\psi = 0$), with SOL and private flux. The boundary is the larger of $\psi_\mathrm{lim}$ and $\psi_X$ |
| `equilibrium_problem_taxonomy` | Forward/inverse and fixed/free are separate axes. CHEASE is forward and fixed, TokaMaker forward and free, EFIT inverse and free. Free boundary and inverse are not synonyms |
| `poloidal_flux_source_decomposition` | $\psi_\mathrm{plasma} + \psi_\mathrm{coil} = \psi_\mathrm{total}$. Only the sum has the X-point and the LCFS. $\psi_\mathrm{passive}$ (eddy currents) is a further term, not drawn |

## Equilibrium-aware phenomena: kink displacement and sawtooth

This layer sits between the reference diagrams and result plotting. The geometry comes from an
equilibrium (any `vaft.data.equilibrium.EquilibriumData`); the physical state is a prescribed, documented
model. The default equilibrium is the exact Solov'ev (Cerfon–Freidberg) equilibrium of
`vaft.process.equilibrium.solovev_example` with $A = 0$, where $q$ rises from about 0.8 to 3. Surfaces,
normals and the PEST angle $\theta^*$ come from `straight_field_line_map`, and $q(\rho)$ from
`calculate_q_profile_from_psi`, with $\rho = \sqrt{\psi_N}$. The mode phase uses $\theta^*$. The drawing
uses the real $(R, Z)$ surfaces.

```python
vaft.diagram.kink_mode(equilibrium=None, m=1, n=1, amplitude=0.06, radial_profile="internal")
vaft.diagram.kink_mode(m=2, n=1, radial_profile="global", harmonics={2: 1.0, 3: 0.3})
vaft.diagram.sawtooth(stage="precursor")   # "reconnection", "post_crash"
```

| | |
| --- | --- |
| ![1/1 internal]({{ '/assets/diagrams/kink_mode_1_1_internal.svg' | relative_url }}) | ![2/1 global]({{ '/assets/diagrams/kink_mode_2_1_global.svg' | relative_url }}) |

![precursor]({{ '/assets/diagrams/sawtooth_precursor.svg' | relative_url }})

| | |
| --- | --- |
| ![reconnection]({{ '/assets/diagrams/sawtooth_reconnection.svg' | relative_url }}) | ![post crash]({{ '/assets/diagrams/sawtooth_post_crash.svg' | relative_url }}) |

| Diagram | Model class | Concept |
| --- | --- | --- |
| `kink_mode` | synthetic parameterization | Each surface moves along its normal by $\xi_n = A\,a\,F(\rho)\,\mathrm{Re}\sum c_m e^{i(m\theta^* - n\phi)}$, with a named envelope. `internal` is a top hat inside $q = m/n$, `global` is $\rho^{m-1}$, and `edge` is $\rho^{4m}$. Under flux freezing this is `flux_perturbation_from_normal_displacement`, $\delta\psi = -\xi_n\lvert\nabla\psi\rvert$. It is not an eigenfunction. The cylindrical reference view is `internal_external_kink` |
| `sawtooth` | reduced model | `precursor`: the 1/1 internal kink inside $q = 1$, with nested topology kept. `reconnection`: a hot core of radius $\rho_1(1-f)$ pushed against an outer separatrix, with the X-point where they touch and the 1/1 island in the crescent between. `post_crash`: nested surfaces again, with the region inside the Kadomtsev mixing radius flattened at conserved $\int T\rho\,d\rho$ |

Two reduced Hamiltonian models extend the vocabulary of `magnetic_island` to several resonances, and to the
X-point.

```python
vaft.diagram.stochastic_layer(regime="touching")   # "isolated", "overlapping"; or overlap=, perturbations=
vaft.diagram.separatrix_lobes(perturbation=0.02, m=8, n=4)
```

| | |
| --- | --- |
| ![isolated]({{ '/assets/diagrams/stochastic_layer_isolated.svg' | relative_url }}) | ![overlapping]({{ '/assets/diagrams/stochastic_layer_overlapping.svg' | relative_url }}) |

![touching]({{ '/assets/diagrams/stochastic_layer_touching.svg' | relative_url }})

![lobes]({{ '/assets/diagrams/separatrix_lobes.svg' | relative_url }})

| Diagram | Model class | Concept |
| --- | --- | --- |
| `stochastic_layer` | reduced Hamiltonian | A Poincaré section of $H = \int\iota\,d\psi_N - \sum_k\epsilon_k\cos(m_k\theta^* - n_k\phi)$ on the equilibrium's $q$ (default 3/2 and 2/1). Each resonance alone is the pendulum of `island_pendulum_hamiltonian`, width $4\sqrt{\epsilon/\lvert\iota'\rvert}$. The pair overlap $\sigma$ (`vaft.process.perturbation.chirikov`) is 0.5, 1 or 1.6. The inset shows where the section sits |
| `separatrix_lobes` | reduced Hamiltonian | The single-null Solov'ev equilibrium plus a prescribed $\delta\psi \propto (r/r_X)^m\cos(m\vartheta - n\phi)$. The field-line map over $2\pi/n$ has a hyperbolic fixed point (Newton, multipliers $\lambda$ and $1/\lambda$). Its unstable and stable manifolds split from the unperturbed separatrix and cross each other, which makes lobes, and one strike point on the target becomes several |

In `stochastic_layer`, $x = \psi_N$ stands in for the toroidal-flux action, so $\epsilon$ is a model amplitude and area in the section is not flux. `separatrix_lobes` draws the single-null Solov'ev equilibrium only, for now. Neither diagram is a GPEC, MARS or vacuum-field trace: those belong to result plotting.

The mixing radius comes from `vaft.formula.kadomtsev_mixing_radius`: the 1/1 helical flux
$\psi_* \propto \int r(1/q - 1)\,dr$ returns to its axis value there. It equals $\sqrt2\,r_1$ when
$1/q - 1$ is parabolic. Complete (Kadomtsev) reconnection is the $f \to 1$ limit, not a claim about every
crash.

## Disruption physics: quench sequence, runaways and energy paths

The chain from loss of confinement to a runaway plateau, each link a relation in the new
`vaft.formula.disruption` category or an existing one:
- `thermal_quench_temperature`, `current_quench_current` and `inductive_parallel_electric_field`;
- `connor_hastie_critical_field`, `dreicer_field`, `runaway_critical_momentum` and
  `relativistic_collision_time`;
- `dreicer_generation_rate`, `avalanche_growth_rate`, `avalanche_efolds_from_current_drop` and
  `runaway_current_from_density`;
- the Spitzer resistivity, and the `startup` plasma resistance, inductance and L/R time.

Detecting a disruption in data stays in `vaft.process.transients`. Simulating one belongs to kinetic or
integrated codes.

```python
vaft.diagram.disruption_timeline()
vaft.diagram.disruption_causal_chain()
vaft.diagram.runaway_generation()
vaft.diagram.disruption_energy_pathways()
```

![timeline]({{ '/assets/diagrams/disruption_timeline.svg' | relative_url }})

![causal chain]({{ '/assets/diagrams/disruption_causal_chain.svg' | relative_url }})

| | |
| --- | --- |
| ![runaway generation]({{ '/assets/diagrams/runaway_generation.svg' | relative_url }}) | ![energy]({{ '/assets/diagrams/disruption_energy_pathways.svg' | relative_url }}) |

| Diagram | Concept |
| --- | --- |
| `disruption_timeline` | A 0-D reference model built from the formulas. A prescribed thermal quench raises the Spitzer $\eta$. The L/R current quench then induces $E_\parallel \approx 10^3E_c$ ($\approx 2\,\%$ of $E_D$). A Dreicer seed of a few kA is multiplied about 25-fold by the avalanche (at 1 MA, only a few e-folds) into a runaway plateau. Magnitudes are illustrative: no universal waveform |
| `disruption_causal_chain` | The same sequence as cause and effect, each arrow labelled by its formula |
| `runaway_generation` | The avalanche rate (per runaway) and the Dreicer rate (per electron) against $E/E_c$. The normalisations differ, so the two magnitudes are not compared. Nothing runs away below $E_c$, and Dreicer is drawn only within its asymptotic range, $E \le 0.1E_D$. Hot-tail seeding is not drawn |
| `disruption_energy_pathways` | Thermal energy leaves by conduction and radiation. Magnetic energy $\tfrac12L_pI_p^2$ goes to ohmic heating, the vessel and coils, runaway kinetic energy and halo currents (#1042). The existing `stored_energy_from_p_V`, `virial_thermal_energy` and `magnetic_energy_from_li_B_pa_V_p` compute the two pools |

## Spectroscopy and ionization

Concept diagrams in the vocabulary of `vaft.spectroscopy`. `parse_emission_term` and `parse_line_label` are
the same parsers `emission=` uses in `vaft.plot`, so a term that selects a trace selects the same diagram.
Metadata is progressive, and nothing is fabricated:
- level 0 is the semantic identity (stage, charge, element);
- level 1 is what the data declare (the wavelength in an IMAS `processed_line` label such as `OI_7770`);
- hydrogenic lines add Bohr-model levels and Rydberg vacuum wavelengths with the isotope's reduced mass
  (`hydrogenic_energy_level` and `hydrogenic_transition_wavelength` in `vaft.formula.atomic`). For one-electron
  systems this model is the authoritative source; each model records it under `source`;
- many-electron levels and photon emissivities would need OPEN-ADAS ADF04 and ADF15. Those are extension
  points and are not loaded; ADF11 stays in `vaft.formula.atomic`.

```python
vaft.diagram.spectroscopy_ionization_stages("C III")   # "C2+", "carbon", "CIII_1909" too
vaft.diagram.spectroscopy_transitions("H-alpha")        # "OI_7770": declared wavelength only
vaft.diagram.spectroscopy_energy_levels("D-alpha")
vaft.diagram.spectroscopy_spectrum()                    # the labels VEST's spectrometer declares
```

![stages]({{ '/assets/diagrams/spectroscopy_ionization_stages.svg' | relative_url }})

| | |
| --- | --- |
| ![H-alpha]({{ '/assets/diagrams/spectroscopy_transitions_h_alpha.svg' | relative_url }}) | ![O I]({{ '/assets/diagrams/spectroscopy_transitions_oi_7770.svg' | relative_url }}) |

![levels]({{ '/assets/diagrams/spectroscopy_energy_levels.svg' | relative_url }})

![spectrum]({{ '/assets/diagrams/spectroscopy_spectrum.svg' | relative_url }})

| Diagram | Concept |
| --- | --- |
| `spectroscopy_ionization_stages` | Every stage of an element, with the named one outlined. Stage $s$ is charge $s - 1$, and D and T are hydrogen with a mass number. Semantic only |
| `spectroscopy_transitions` | A hydrogen series member gets Bohr-model levels and its vacuum wavelength (an unspecified isotope is taken as protium, and the title says so). Fully stripped ions are refused, since they have no lines. Any other line gets unnamed levels, and a wavelength only if its label declares one |
| `spectroscopy_energy_levels` | The hydrogenic ladder with the Lyman, Balmer and Paschen series. Hydrogenic only: other species need ADF04 |
| `spectroscopy_spectrum` | Declared lines, each at its label's wavelength (air above 200 nm by convention). Computed hydrogenic lines are dashed and in vacuum. Lines with no wavelength are listed, not placed |

## Cold-plasma waves: dispersion, cutoffs, resonances and the CMA diagram

Each diagram is drawn from the cold-plasma equations in `vaft.formula.waves`:
- `plasma_frequency`;
- `stix_parameters`, giving $R, L, S, D, P$ with the signed cyclotron frequency, so $\Omega_e < 0$;
- `cold_plasma_refractive_index_squared`, the two roots of $An^4 - Bn^2 + C = 0$;
- `perpendicular_refractive_index_squared`, giving $n_O^2 = P$ and $n_X^2 = RL/S$;
- `cma_coordinates`, giving $X = \omega_{pe}^2/\omega^2$ and $Y = |\Omega_e|/\omega$;
- `propagation_regime`, which classifies propagating, evanescent, cutoff and resonance.

Boundaries are zeros or poles of the Stix parameters, located by bracketing the formulas; no closed form
is typed into a drawing. The $\pm$ roots are algebraic branches, not mode names, so O/X and R/L are named
only where the mode is tracked: at $\theta = \pi/2$ and $\theta = 0$. Electrons only, ions immobile:
the electron-cyclotron range. Warm-plasma effects, damping, ray tracing and full-wave solutions are out of
scope.

```python
vaft.diagram.o_mode_cutoff()                                  # n_O^2 = P, cutoff at omega_pe
vaft.diagram.x_mode_dispersion(omega_pe_over_omega_ce=1.2)    # L, R cutoffs; upper-hybrid resonance
vaft.diagram.cma_diagram()                                    # P, R, L, S = 0 and Y = 1 in (X, Y)
vaft.diagram.profile_propagation()                            # layers along an example midplane
```

| | |
| --- | --- |
| ![O mode]({{ '/assets/diagrams/o_mode_cutoff.svg' | relative_url }}) | ![X mode]({{ '/assets/diagrams/x_mode_dispersion.svg' | relative_url }}) |
| ![CMA]({{ '/assets/diagrams/cma_diagram.svg' | relative_url }}) | ![profile]({{ '/assets/diagrams/profile_propagation.svg' | relative_url }}) |

| Diagram | Concept |
| --- | --- |
| `o_mode_cutoff` | Evanescent below $\omega_{pe}$, propagating above; the cutoff $P = 0$ does not depend on $B$ |
| `x_mode_dispersion` | Evanescent below $\omega_L$, propagating to the upper-hybrid pole, evanescent to $\omega_R$, then propagating. Poles are masked |
| `cma_diagram` | Cutoffs (solid) and resonances (dashed) of a cold electron plasma in the CMA plane |
| `profile_propagation` | $n_O^2$ and $n_X^2$ along $R$ for an example tokamak (not a device) at the on-axis electron cyclotron frequency: O cutoffs, L and R cutoffs, the upper-hybrid layer behind the R cutoff, and the ECR |

## Using the committed assets

The reference SVGs live in `docs/assets/diagrams/` and are the artifacts to embed anywhere:

* Documentation pages: {% raw %}`![island]({{ '/assets/diagrams/magnetic_island_poloidal.svg' | relative_url }})`{% endraw %}.
* The README, notebooks and slides: link or copy the SVG. It is self-contained, with glyphs stored as
  paths and no external fonts or images.

They do not ship in the wheel. An installed package builds any diagram from its packaged TikZ
template.

## Regenerating and checking

```bash
python -m vaft.diagram.build            # re-render diagrams whose TikZ source changed
python -m vaft.diagram.build --check    # verify the committed assets; needs no TeX
```

`manifest.json`, stored next to the SVGs, records the SHA-256 of the generated TikZ document each SVG
was rendered from, together with the SHA-256 of the SVG itself. `--check` rebuilds the TikZ in pure
Python and fails in four cases: an asset is stale, an asset was edited by hand, an asset is
missing, or an asset is orphaned. CI runs it on every pull request. Freshness is judged on the source
and not on SVG bytes, because two `dvisvgm` releases write different but equally correct SVG for the
same picture. Rendering needs `latex` and `dvisvgm`, which TeX Live and MacTeX provide. The generated
`.tex`, PDF, PNG and LaTeX by-products are never committed.
