---
title: Characteristic scales and orderings
author: VEST team
date: 2026-10-11 12:00
category: guide
layout: post
permalink: /reference/characteristic-scales/
guide:
  architecture: The conceptual bridge from a plasma state to its characteristic frequencies, times and lengths, from ratios of those scales to asymptotic orderings, and from orderings to reduced models, applicability and resolvability - with the VAFT function that computes each scale today.
  prerequisites: None. The Asymptotic orderings and Plasma models pages go deeper on the orderings and on the models.
  expected: Which kind of quantity a number is (scale, ordering, regime, similarity coordinate, applicability verdict or resolution), which comparison scale an ordering must name, and where each one is computed.
related:
  api: [formula, process, validation, diagram]
---

This page answers one question: **how does a plasma state become a statement about which model may describe it?**
VAFT spreads the answer over several layers. This page names the steps and the vocabulary that keeps them apart,
and links to where each step is implemented.

```text
physical plasma state                    n, T, B, q, a, R, I_p(t), ...
        ↓
dimensional characteristic scales        Ω_ci, ρ_i, d_i, λ_mfp, L_T, τ_A, τ_R, ...
        ↓
ratios of scales                         ρ_i/L_T, d_i/a, λ_e/qR, τ_evol/τ_A, ...
        ↓
asymptotic orderings                     ρ_i/L_T ≪ 1, S ≫ 1, ...
        ↓
reduced physical models                  ideal MHD, drift kinetics, gyrokinetics, ...
        ↓
state-specific applicability             SUPPORTED / OUTSIDE / UNASSESSED / NOT_APPLICABLE
        ↓
diagnostic / numerical resolvability     is the structure resolved at all?
```

Each arrow is a separate step, and none of them is implied by the one before.

| step | owned by | page |
| --- | --- | --- |
| scales | `vaft.formula` (domain modules; a canonical scale layer is not yet implemented, Open: #1064) | this page, [Formula reference]({{ '/reference/formula/' | relative_url }}) |
| ratios and orderings | `vaft.formula.ordering`, `vaft.validation.orderings` | [Asymptotic orderings]({{ '/reference/asymptotic-orderings/' | relative_url }}) |
| models | `vaft.code.formalism`, `vaft.validation.model_contracts` | [Plasma models]({{ '/reference/plasma-models/' | relative_url }}) |
| applicability | `vaft.validation.applicability`, `vaft.process.ordering_state` | [Credibility and applicability]({{ '/reference/credibility-applicability/' | relative_url }}) |
| resolvability | not implemented | [below](#physical-scales-are-not-resolution) |

## Four different kinds of statement

| kind | what it is | example | what it does **not** say |
| --- | --- | --- | --- |
| **characteristic scale** | a dimensional quantity of the state | $\rho_i$, $d_i$, $\lambda_{\rm mfp}$, $L_T$, $\tau_A$, $\tau_R$, $\Omega_{ci}$ | anything about an approximation |
| **ordering parameter** | a dimensionless ratio of two named scales | $\rho_i/L_T$, $d_i/a$, $\lambda_e/qR$, $\omega/\Omega_{ci}$, $\tau_A/\tau_R$ | which model is valid |
| **physical regime** | a qualitative name for where one or more orderings sit | banana / plateau / Pfirsch–Schlüter; ideal / resistive; strongly / weakly magnetized | an exact boundary, unless a theory defines one |
| **model applicability** | whether the assumptions of one formula, approximation or solver are supported by one state | `evaluate_contract(contract("ideal_single_fluid_mhd"), state)` | that the model's *output* is correct |

A characteristic scale on its own establishes nothing. $\rho_i = 3$ mm is neither small nor large until it
is divided by something, and the answer changes with the divisor ([§ Name the comparison scale](#name-the-comparison-scale)).
A regime name is a reading of orderings. The banana regime is $\nu_* \ll 1$, and the plateau regime has no
ordering of its own (Asymptotic orderings, "Ordering contracts"). The neoclassical and NTV regime diagrams
(`ntv_precession_regimes` and the collisionality-regime figures in
[Scientific diagrams]({{ '/reference/diagrams/' | relative_url }})) draw regimes the same way: as readings of
ratios, with boundaries only where the theory gives one [7]. Applicability is a verdict on a state and
belongs to the validation layer. It is never a property of the ordering's definition.

## What exists today

The page distinguishes three states, and says which applies everywhere it names an interface:

- **implemented on develop.** A function name in `code` is importable from `develop`.
- **planned under an open issue.** Named with its issue; no interface is promised.
- **open PR only.** None at the time of writing. Interfaces in open PRs are not documented here as stable.

The main gap is a canonical scale layer (`vaft.formula.scales`, Open: #1064): it does not exist yet. Today
each scale lives in the domain module that first needed it, listed below. Once that layer exists, the tables
below should point at it and the domain functions stay as convention-bearing variants.

## Scale taxonomy, by physical role

Scales are grouped by what they mean physically, not by the module that computes them. All functions are in
`vaft.formula` unless the path says otherwise. Units are SI, temperatures in eV.

### Frequencies and rates

| scale | meaning | VAFT function | returns |
| --- | --- | --- | --- |
| $\omega_{ps}$ | plasma frequency of species $s$ [1] | `waves.plasma_frequency(n, q, m)` | rad/s |
| $\Omega_{cs}$ | gyrofrequency, signed with the charge | `particle.gyrofrequency(q, m, B)` | rad/s |
| $\tau_e$, $\tau_i$ | Braginskii collision times [2] | `ordering.braginskii_electron_collision_time`, `ordering.braginskii_ion_collision_time` | s |
| $\tau_e$, $\tau_i$ (several ion species) | Braginskii, multi-species | `kinetic.electron_collision_time_from_T_e_n_species`, `kinetic.ion_collision_time_from_T_i_n_species` | s |
| $\omega_t$ | transit frequency $v/qR_0$ | `neoclassical.transit_frequency(v, q, R0)` | rad/s |
| $\omega_b$ | deeply trapped bounce frequency | `neoclassical.deeply_trapped_bounce_frequency(v_perp, q, R0, epsilon)` | rad/s |
| $\nu_{\rm eff}$ | effective trapped-particle collision frequency $\nu/\epsilon$ | `neoclassical.trapped_particle_effective_collision_frequency` | 1/s |
| $\omega_A$ | shear-Alfvén frequency $k_\parallel v_A$ | `stability.shear_alfven_frequency(k_parallel, v_A)` | rad/s |

**Not implemented:**
- a direct NRL $\nu_{ei}$ or $\nu_{ee}$ frequency function. Use the inverse of a collision time, and say so.
- magnetic-drift precession $\omega_D$ and $E\times B$ precession $\omega_E$ as *computed* quantities. They are
  inputs [rad/s] to `ntv.ntv_precession_frequency` and `particle.bounce_harmonic_detuning`. The drift
  *velocities* (`particle.exb_drift_velocity`, `grad_b_drift_velocity`, `curvature_drift_velocity`) are
  velocities, not frequencies. `vaft.code.transp.profiles.exb_frequency` reads $\omega_E$ from TRANSP output.

**Angular frequency, ordinary frequency and inverse time are different quantities.**

| quantity | unit | relation |
| --- | --- | --- |
| angular frequency $\omega$ | rad/s | $\omega = 2\pi f$ |
| ordinary frequency $f$ | Hz | what a spectrum or a DAQ rate is quoted in |
| inverse time $1/\tau$ | 1/s | a rate; carries no phase |

Every angular-frequency function above returns rad/s; the collision times are in s and $\nu_{\rm eff}$ in 1/s. At $B = 0.1$ T, $\Omega_{ce} \approx 1.76\times10^{10}$ rad/s
is $f_{ce} \approx 2.8$ GHz. Comparing $\Omega_{ce}$ with a 2.8 GHz bandwidth without the $2\pi$ is an
error of 6.3. A period is $2\pi/\omega$, not $1/\omega$: $1/\Omega$ is the time to turn one radian, and that
is what `timescale_hierarchy` plots as the gyration time.

### Speeds

| scale | VAFT function |
| --- | --- |
| thermal speed $v_t = \sqrt{T/m}$ (NRL convention, no factor 2) | `ordering.thermal_speed(T, m)` |
| ion sound speed $c_s$ | `sol.ion_sound_speed(T_e, T_i, m_i, Z, gamma_i)` |
| Alfvén speed $v_A$ | `stability.v_alfven_from_B_n_mi(B, n, m_i)` |

### Microscopic lengths

| scale | physical meaning | VAFT function |
| --- | --- | --- |
| $\lambda_D$ | Debye screening | `ordering.debye_length(n_e, T_e)` (electron) |
| $\rho_e$, $\rho_i$ | gyro-orbit, finite-Larmor-radius scale | `particle.larmor_radius(q, m, v_perp, B)` |
| $\rho_s$ | sound gyroradius, the drift-wave scale | `ordering.sound_gyroradius(T_e, m_i, B, Z)` |
| $d_e$ | electron inertia scale | `ordering.inertial_length(n, m, q)` with the electron mass |
| $d_i$ | ion inertia, Hall scale | `ordering.inertial_length(n, m, q)` with the ion mass |
| $\lambda_{\rm mfp}$ | mean free path $v_t\tau$ | `ordering.mean_free_path(v_t, tau)` |

These are not universally ordered by magnitude. With $v_t=\sqrt{T/m}$, $d_i/\rho_i = \sqrt{2/\beta_i}$: in a
low-$\beta$ tokamak $d_i$ exceeds $\rho_i$, by a factor that depends on the state, and the two meet only at
$\beta_i\sim1$. $d_e/\rho_e$ is $\sqrt{2/\beta_e}$ in the same way. $\lambda_D$ is the smallest in a fusion
plasma, but not by a fixed factor.

### Orbit and parallel-motion scales

| scale | VAFT function | assumption |
| --- | --- | --- |
| banana width $\Delta_b \approx q\rho/\sqrt\epsilon$ [6, 7] | `neoclassical.banana_width(rho, q, epsilon)` | large aspect ratio, deeply trapped |
| transit and bounce frequency | above | circular large-aspect-ratio flux surfaces |
| connection length $qR$ | no analytic function; field-line maps in `vaft.process.equilibrium.connection_length_map` and `vaft.process.field_line_topology.connection_length_weight` | — |

The analytic orbit scales carry the large-aspect-ratio ordering $\epsilon \ll 1$. In VEST $\epsilon \approx 0.7$
(the median of the VEST ordering atlas, `notebooks/vest_ordering_and_model_applicability.ipynb`), so they are order-of-magnitude estimates there, not results.

### Fluctuation and layer lengths

| scale | meaning | where it comes from |
| --- | --- | --- |
| $1/k_\perp$, $1/k_\parallel$ | perpendicular and parallel wavelength of a perturbation | the caller: mode numbers, a $k_y$ grid, a measured spectrum |
| $\delta_{\rm layer}$ | width of a resonant (tearing, resistive) layer | the caller, or a scaling estimate such as $a S^{-2/5}$ [4] |

Neither is a property of the equilibrium state alone, so VAFT has no function that returns them. The
mode-scope and layer-scope ordering quantities take them as inputs and are `UNASSESSED` without them.

### Profile and equilibrium scales

$$L_n=-\left(\frac{d\ln n}{dr}\right)^{-1},\qquad L_T=-\left(\frac{d\ln T}{dr}\right)^{-1},\qquad
L_p=-\left(\frac{d\ln p}{dr}\right)^{-1}.$$

These need three choices before they are numbers, and VAFT keeps them apart:

| concept | what it fixes | VAFT name |
| --- | --- | --- |
| **profile coordinate** | the grid the profile is given on ($\rho_{\rm tor,N}$, $\psi_N$, $R$, ...) | `profile_coordinate` |
| **gradient coordinate** | what $r$ in $d/dr$ is ($r_{\rm minor}$, $\rho$, $\psi$, ...) | `gradient_coordinate` |
| **reference length** | what a gradient is normalized by: $a/L$, $R/L$, or none | `reference_length` (`a_minor`, `R_major_axis`, `R_major_surface`, `L_ref`, `"none"`) |

`vaft.process.profile_gradients.profile_gradient` takes all three explicitly, and its `convention` presets
(`tglf`, `cgyro`, `gs2`, `gkw`, `gene`) fix them the way each code does. The table of combinations is in
[Equilibrium representations]({{ '/reference/equilibrium-representations/' | relative_url }}), "Profile
gradients". The pure kernel is `utils.normalized_gradient_scale_length(x, y, a)`, which returns $a/L_y$.
There is no bare `L_n`/`L_T` function on purpose: a gradient length without its coordinate is ambiguous.

Other equilibrium scales: minor radius $a$ and major radius $R_0$ with
`equilibrium.inverse_aspect_ratio_from_a_R`; the magnetic shear length `geometry.shear_length_from_q_R0_s`
(shear itself from `equilibrium.shear_from_r_q`).

### Macroscopic times

| scale | construction | VAFT function |
| --- | --- | --- |
| $\tau_A = L/v_A$ | crossing time | `ordering.alfven_time(L, v_A)` |
| $\tau_s = L/c_s$ | crossing time | compose from `sol.ion_sound_speed` |
| $\tau_R = \mu_0L^2/\eta$ | diffusion time | `ordering.resistive_diffusion_time(L, eta)`, with $\eta$ from `equilibrium.spitzer_resistivity_from_T_e_Z_eff_ln_Lambda` |
| $S = \tau_R/\tau_A$ | ratio | `ordering.lundquist_number(L, v_A, eta)`; `vaft.process.perturbation.lundquist_number(tau_resistive, tau_alfven)` |
| $\tau_E = W/P_{\rm loss}$ | confinement time | `equilibrium.confinement_time_from_P_loss_W_th`; scalings in `equilibrium`; series in `vaft.process.confinement` |
| $\tau_w = \mu_0\sigma d b$ | wall diffusion time | `vde.thin_wall_time`, `vde.wall_mode_decay_time`; `vaft.process.wall_modes.global_time_constants` |
| $\tau_{\rm evol} = |I_p/\dot I_p|$ | evolution time | `ordering.evolution_time(X, dX_dt)` with $X=I_p$ |
| $L/R$ time | circuit time | `startup.lr_time_from_L_R` |
| $\tau_D = L^2/D$ | diffusion time | no function yet; the diffusivities $D$ and $\chi$ are in `vaft.formula.transport` |

These are different physical constructions. One state can have several "times" with the same name, and they
must not be swapped:

| construction | example | note |
| --- | --- | --- |
| oscillation period | $2\pi/\omega$ | includes the $2\pi$ |
| inverse rate | $1/\nu$, $1/\Omega$ | one radian or one e-fold, not one cycle |
| crossing time | $\tau_A$, $\tau_s$ | depends on the length it crosses |
| diffusion time | $\tau_R$, $\tau_D$ | $\propto L^2$; the slowest cylindrical mode of $\tau_R$ decays $j_{01}^2\approx5.8$ times faster |
| confinement time | $\tau_E$ | a ratio of an inventory to a loss, not a microscopic rate |
| growth or damping time | $1/\gamma$ | a property of a mode, not of the state alone |
| plasma age, pulse length | $\tau_{\rm age}$, $\tau_{\rm pulse}$ | a property of the discharge history |
| measurement cadence | EFIT slice spacing | not a physical scale at all ([below](#physical-scales-are-not-resolution)) |

## Scale classes are not a magnitude ordering

A schematic ladder is useful for orientation:

```text
lengths:  Debye → gyro / inertial → orbit / turbulence → profile gradient → minor radius → major radius
times:    gyro motion → orbit / collisions → Alfvénic → mode / wall response → transport → current diffusion → discharge evolution
```

**These are scale classes, not a universal magnitude ordering.** The actual order is a property of one state
and must be computed from it. `vaft.diagram.timescale_hierarchy` does exactly that: it evaluates each time
from a formula kernel for one illustrative low-field state (only the pulse length is an input) and places it by value, so the hierarchy is a
result, not an assumption (Asymptotic orderings, "A worked hierarchy").

The ladder fails in practice:
- In a low-field device the electron transit frequency can reach the ion gyrofrequency. In VEST,
  $\omega_{t,e}\approx5\times10^6$ against $\Omega_{ci}\approx10^7$ rad/s ([Plasma models]({{ '/reference/plasma-models/' | relative_url }}) §7).
- A short pulse compresses the slow end. In the VEST ordering atlas the median $\tau_{\rm age}/\tau_R$ is 0.57: current
  diffusion is not slower than the discharge, so "discharge evolution" is not the last rung.

## Name the comparison scale

**`L` is part of an ordering's definition. It is never universal.** The same numerator over different
denominators answers different questions:

| ratio | question | VAFT registry name |
| --- | --- | --- |
| $d_i/a$ | is the Hall term negligible on the scale of the device? | `ion_skin_depth_over_a` (global) |
| $d_i/L_T$ | is it negligible on the scale of the temperature profile? | not registered |
| $d_i/\delta_{\rm layer}$ | is it negligible inside a resonant layer? | `ion_skin_depth_over_layer` (layer) |
| $\rho_i/a$ | where is the device in similarity space? | not an ordering: $\rho_*$ ([next section](#similarity-coordinates-are-not-orderings)) |
| $\rho_i/L_{T_i}$ | is the FLR correction to the background small? | `rho_i_over_LTi` (profile) |
| $k_\perp\rho_i$ | is a fluctuation larger than a gyro-orbit? | `k_perp_rho_i` (mode) |

Every ordering quantity in `vaft.validation.orderings.ORDERING_QUANTITIES` records its comparison scale
explicitly. The issue's proposed vocabulary maps onto those fields:

| concept | where VAFT records it |
| --- | --- |
| scale value | the kernel's return value (`OrderingQuantity.kernels` names the kernels) |
| scale kind | `OrderingQuantity.definition`, e.g. `"d_i / a"` |
| reference scale | `OrderingQuantity.scale`, e.g. `"a"`, `"q R"`, `"delta_layer"`, `"1 / k_perp"` |
| scope | `OrderingQuantity.scope`: `global`, `profile`, `mode`, `local`, `time_history` |
| group | `OrderingQuantity.group`: `foundational`, `global`, `profile`, `perturbation`, `layer`, `time_history` |
| species | in the name: `electron_...`, `ion_...`, `rho_s` (electron temperature, ion mass) |
| radial location | a column of the point table built by `vaft.process.ordering_state.profile_ordering_quantities` |
| mode / event | the caller's input (mode numbers, $k_y$ grid, layer width); missing gives `UNASSESSED` |
| provenance | the atlas manifest (`manifest.json` of `workflow/ordering_atlas/build_states.py`) and the `ordering_check` run provenance of each adapter |

The kernels in `vaft.formula.ordering` take the length as an explicit argument for the same reason. A global
length cannot be silently swapped for a layer length.

## Similarity coordinates are not orderings

Some dimensionless numbers place a state in **similarity space**. Others are the **small or large parameters**
a derivation expands in. Some symbols appear in both lists, but the role decides what a value means.

| role | quantities | answers | used by |
| --- | --- | --- | --- |
| similarity coordinate [11, 12] | $\rho_*$, $\beta$, $\nu_*$, $q$, $\epsilon$, $\kappa$ | is this state comparable with another machine's? | confinement scalings, operational space |
| asymptotic ordering | $\rho_i/L_T$, $d_i/a$, $\lambda/qR$, $k_\perp\rho_i$, $S$ | may this reduced model be used here? | ordering contracts, applicability |

Concretely:

```text
rho_i / a      → machine similarity / finite-size coordinate     (ρ*: equilibrium.rho_star_from_M_T_B_R_epsilon, on a = R_geo ε)
rho_i / L_T    → local FLR / background-scale ordering            (rho_i_over_LTi)
k_perp rho_i   → fluctuation-scale ordering                       (k_perp_rho_i)
```

The same function can serve both roles. $\epsilon = a/R_0$ is a shape coordinate in a confinement fit and the
expansion parameter of the large-aspect-ratio ordering. Sauter's $\nu_{*e}$ is a regime coordinate in a
neoclassical fit and the ordering of the banana-regime contract. The number is identical; the claim is not.

Similarity coordinates in VAFT:
- $\rho_*$: `equilibrium.rho_star_from_M_T_B_R_epsilon` (Verdoolaege form, on $a$),
  `equilibrium.normalized_larmor_radius_from_M_T_a_Bt` ($\rho_i/a$);
- $\nu_*$: `neoclassical.electron_collisionality_sauter`, `neoclassical.ion_collisionality_sauter`,
  `equilibrium.nu_star_from_n_T_B_R_epsilon_kappa_I` (Verdoolaege form);
- $\beta$, $\beta_N$, $q$: `equilibrium.beta_t_from_n_T_B`, `stability.beta_N_from_beta_a_B0_Ip`,
  `equilibrium.q_cyl_from_B_R_epsilon_kappa_I`.

Different $\nu_*$ definitions [8, 12] differ by order-unity factors. Never compare two without naming both.
`vaft.process.confinement.dimensionless_confinement_indices` turns fitted engineering exponents into
$\rho_*$, $\beta$, $\nu_*$, $q$ exponents; its docstring says they are not measured dimensionless dependences.

## Geometry is not an ordering

Slab, cylindrical and toroidal describe **geometry** [3]. Local or global, $a/R\ll1$, $\rho/L\ll1$,
$k_\parallel/k_\perp\ll1$ and $\omega/\Omega\ll1$ describe **scale separation**. They are two axes, not one
accuracy ladder. "Large aspect ratio" is an ordering, not a geometry, and a local slab may keep coefficients
inherited from a realistic toroidal equilibrium. The full treatment, with `geometry_ordering_map`, is
[Geometric approximations]({{ '/reference/geometric-approximations/' | relative_url }}), "Two axes, not one
hierarchy".

## From orderings to models

Each plasma description is defined by the orderings it adopts and the ones it relaxes. It is not a rung on one
fidelity ladder. [Plasma models]({{ '/reference/plasma-models/' | relative_url }}) is the reference:
- §1 explains why a single "fidelity" axis fails;
- §2 derives the fluid hierarchy (ideal, resistive, Hall/two-fluid, reduced MHD [5]), drift kinetics,
  gyrokinetics, particle orbits and kinetic-MHD hybrids;
- §6 says what "kinetic" means in each VAFT name;
- §7 is the compact table of what each family removes and which ordering justifies it.

The comparison axes are:

```text
ρ/L    d_i/L    ω/Ω_i    k⊥ρ_i    k∥/k⊥    collisionality    resistivity    locality    distribution-function dynamics
```

The ordering contracts encode them per model (`vaft.validation.orderings.CONTRACTS`, drawn as
`ordering_contract_map`). The adapter modes say which contracts each VAFT code mode consumes,
and in which **role** (`governing`, `outer_solution`, `layer_solution`, `fluid_part`, `kinetic_response`,
`closure`, `inherited`, `transport_coupling`). That is `vaft.validation.model_contracts.ADAPTER_MODES`, drawn as
`adapter_contract_map` ([Asymptotic orderings]({{ '/reference/asymptotic-orderings/' | relative_url }}),
"Which codes consume which contracts").

### Drift kinetics and gyrokinetics are not one ladder

A guiding-centre or drift-kinetic description assumes fields that vary slowly compared with gyromotion,
$\rho/L\ll1$ and $\omega/\Omega\ll1$, on every scale it keeps [6]. A gyrokinetic description keeps the low
frequency ordering but allows perpendicular structure at the gyroradius, $k_\perp\rho_i\sim O(1)$, at the cost
of ordering the fluctuation amplitude and $k_\parallel/k_\perp$ small [9, 14]. So:

```text
drift kinetic   ≠  a lower-fidelity gyrokinetic model
gyrokinetic     ≠  "more kinetic" MHD
```

They keep different scale information for different problems. Drift kinetics is the natural description of
equilibrium-scale kinetic response (neoclassical transport, NTV, kinetic stabilization). Gyrokinetics is the
description of $\rho$-scale turbulence, with transport on a separate, slower scale [10]. Plasma models §2.2–§2.3; a cross-formalism gyrokinetic modelling framework is Open: #1353.

### Kinetic GPEC is not on the gyrokinetic branch

Kinetic DCON/GPEC and PENTRC are not on the same branch as CGYRO. Both evolve a gyro-averaged kinetic equation
under $\omega/\Omega\ll1$, and gyrokinetics reduces to drift kinetics at long wavelength, but they answer
different questions:

```text
MHD perturbed equilibrium
  + kinetic orbit / distribution response
  + bounce / transit / drift resonances
        ↓
kinetic correction to the global plasma response            (DCON kinetic, GPEC kinetic, PENTRC) [15-17]

gyrokinetic microturbulence
        ↓
turbulent transport                                          (CGYRO, TGLF)
```

They differ on every axis that matters: physical formalism (bounce-harmonic drift kinetics against $\delta f$
gyrokinetics), spatial domain (global against flux-tube local), frequency ordering (precession and bounce
resonances against drift-wave frequencies) and the problem solved (response or stability against turbulent
flux). `vaft.code.formalism.PlasmaFormalism` records these as separate fields for exactly this reason (Plasma
models §3, §4, §11).

## From orderings to applicability

An applicability verdict is assembled from **separate assumptions, each with its own evidence**. The
`ideal_single_fluid_mhd` contract needs, among others,

$$S\gg1,\qquad d_i/a\ll1,\qquad \rho_i/L_{T_i}\ll1,\qquad \omega/\Omega_{ci}\ll1,$$

and these fail independently. Slow evolution, $\tau_{\rm evol}/\tau_A\gg1$, is **not** one of them: ideal
MHD describes Alfvénic dynamics, $\omega\tau_A\sim1$. Slow evolution is the separate
`quasi_static_equilibrium` contract, which an equilibrium sequence consumes. VAFT never reduces them to `ideal_mhd_valid = True`.
`vaft.validation.applicability.evaluate_contract` returns a margin and a status for every assumption, names
the limiting one, and reports a missing quantity as `UNASSESSED`. Local gyrokinetic applicability likewise
involves $\rho_s/L_T$, $\rho_s/L_n$ and $\rho_s$ over the geometric scale, plus fluctuation orderings where a
spectrum is available.

This page defines no verdicts. The evaluation, its statuses and its margins are
[Credibility and applicability]({{ '/reference/credibility-applicability/' | relative_url }}). The state
tables are built by `vaft.process.ordering_state`. The VEST population study is the notebook
`vest_ordering_and_model_applicability.ipynb`.

## One model, two scales

**A model can be asymptotically appropriate at one scale and inadequate at another.** Tearing is the standard
example. Take the VEST ordering-atlas medians: $d_i/a\approx0.32$ and $S\approx1\times10^5$.
- Globally, $d_i/a\approx0.3$ is below one. The Hall term is not negligible, but single-fluid MHD is not
  obviously wrong on the device scale.
- The resistive layer is much thinner than $a$. Its width scales as $\delta/a\propto S^{-2/5}$ for the
  Furth–Killeen–Rosenbluth tearing layer [4] and $S^{-1/3}$ for the resistive internal kink [13], so $\delta/a\sim0.01$–$0.02$
  here, up to factors of mode number and shear.
- Inside the layer, $d_i/\delta\sim15$–$30$. The two-fluid ordering fails by more than a decade. A
  single-fluid resistive layer model is outside its contract, even though the device-scale ordering looked
  acceptable.

The same holds for resistivity: $S_{\rm global}\gg1$ does not make resistivity negligible in the layer,
because the layer is where it is not. This is why VAFT registers layer quantities
(`ion_skin_depth_over_layer`, `electron_skin_depth_over_layer`, `rho_s_over_layer`) separately from the
global ones, and why the RDCON/RMATCH adapter mode consumes `resistive_mhd`, which orders $d_i/\delta$ and
$\rho_s/\delta$, in the `layer_solution` role, next to `ideal_single_fluid_mhd` in the `outer_solution`
role.

## Physical scales are not resolution

Whether a structure is **resolved** is a different question from how big it is. A diagnostic cadence or a
grid spacing is never a characteristic scale of the plasma, and this page does not define one.

| resolution | temporal examples | spatial examples |
| --- | --- | --- |
| diagnostic | DAQ sample period, Mirnov bandwidth, fast-camera frame interval, Thomson repetition period | chord spacing, optical resolution, tomographic resolution |
| reconstruction | EFIT cadence | equilibrium grid spacing |
| numerical | time step, output cadence | finite-element mesh, gyrokinetic radial and perpendicular resolution |
| control | control-system update period | — |

Comparing the two belongs to an analysis layer above both, for example "is the EFIT cadence shorter than
$\tau_{\rm evol}$?" or "does the Thomson chord spacing resolve $L_{T_e}$?". **Not implemented.** Only the
time-grid description exists today (`vaft.process.signal_processing.describe_time_grid`). Nothing compares a
cadence or a grid with a physical scale yet. When it is built, it is evidence on the `numerical` or
`transformation` axis of the credibility vocabulary, not a new kind of scale.

## Diagram contracts

Four diagrams make the chain visible. Two exist. All of them follow the rule of this page: a magnitude order
is drawn only when it comes from values.

| diagram | kind | contract | state |
| --- | --- | --- | --- |
| `characteristic_scale_map` | conceptual, state-independent | primitive state → frequencies, speeds, lengths → times → dimensionless ratios → model orderings; no numerical ranking | planned |
| `length_scale_ladder` | state-dependent, or labelled illustrative | $\lambda_D$, $\rho_e$, $d_e$, $\rho_i$/$\rho_s$, $d_i$, $\lambda_{\rm mfp}$, orbit width, gradient length, $a$, $R$; placed by value, never in a hard-coded order | **partly exists** as `spatial_scale_orderings` ($\rho_e$, $\rho_i$, $\rho_s$, $d_i$, banana width, gradient length, $a$, $R_0$, $qR_0$); $\lambda_D$, $d_e$ and $\lambda_{\rm mfp}$ still planned |
| `time_scale_ladder` | state-dependent, or labelled illustrative | gyration, collision, transit/bounce, Alfvén, growth, wall, confinement, transport, resistive diffusion, age/pulse; placed by value | **exists** as `timescale_hierarchy` (illustrative state, eleven times including sound crossing, ion transit and bounce, placed by value) |
| `model_ordering_map` | multidimensional relationship map | fluid ↔ kinetic closure, ideal ↔ dissipative, single- ↔ two-fluid, long wavelength ↔ finite $k_\perp\rho$, local ↔ global, equilibrium ↔ fluctuation scale; no single fidelity axis | **exists** in parts: `ordering_contract_map`, `adapter_contract_map`, `reduced_representation_hierarchy`, `geometry_ordering_map` |

Diagrams live in `vaft.diagram` and are conceptual. Ladders of measured states belong to `vaft.plot`, with the
values from `vaft.process`. The two planned diagrams wait for the canonical scale layer (Open: #1064), so that they draw canonical scale
functions rather than define their own.

## Who owns what

```text
vaft.formula.scales           generic dimensional scale primitives                          planned, Open: #1064
vaft.formula.<domain>         convention-bearing scales (neoclassical, wall, equilibrium, ...)   implemented
vaft.formula.ordering         ordering kernels with explicit lengths                          implemented
vaft.process                  state- and profile-aware construction, coordinate and time alignment
                              (profile_gradients, ordering_state)                            implemented
vaft.validation               ordering registry, contracts, applicability, credibility       implemented
vaft.diagram                  conceptual scale and ordering diagrams                          partly implemented
vaft.plot                     scale ladders and ordering spaces of real states                ordering spaces implemented (vaft.plot.ordering_space)
```

This page consumes those layers. It defines no formula: every expression above is either the definition a
linked kernel implements, or a scaling quoted with its source.

## References

1. J. D. Huba, *NRL Plasma Formulary*, Naval Research Laboratory (2019). Plasma and gyrofrequencies, Debye
   length, gyroradii, inertial lengths, collision rates.
2. S. I. Braginskii, in *Reviews of Plasma Physics*, Vol. 1, Consultants Bureau (1965), p. 205. Collision
   times and the collisional fluid ordering.
3. J. P. Freidberg, *Ideal MHD*, Cambridge University Press (2014). The ideal-MHD ordering and the
   large-aspect-ratio expansion.
4. H. P. Furth, J. Killeen and M. N. Rosenbluth, Phys. Fluids 6 (1963) 459. Resistive layer width and its
   Lundquist-number scaling.
5. H. R. Strauss, Phys. Fluids 19 (1976) 134. Reduced MHD.
6. R. D. Hazeltine and J. D. Meiss, *Plasma Confinement*, Dover (2003). Drift-kinetic ordering and orbit
   scales.
7. F. L. Hinton and R. D. Hazeltine, Rev. Mod. Phys. 48 (1976) 239. Neoclassical regimes and collisionality.
8. O. Sauter, C. Angioni and Y. R. Lin-Liu, Phys. Plasmas 6 (1999) 2834. The $\nu_*$ definitions VAFT uses.
9. E. A. Frieman and L. Chen, Phys. Fluids 25 (1982) 502. The nonlinear gyrokinetic ordering.
10. I. G. Abel et al., Rep. Prog. Phys. 76 (2013) 116201. Multiscale gyrokinetics and the separation of
    turbulence and transport scales.
11. B. B. Kadomtsev, Sov. J. Plasma Phys. 1 (1975) 295. Dimensionless similarity.
12. G. Verdoolaege et al., Nucl. Fusion 61 (2021) 076006. The $\rho_*$ and $\nu_*$ forms of the ITPA20
    scaling.
13. B. Coppi, R. Galvão, R. Pellat, M. N. Rosenbluth and P. H. Rutherford, Sov. J. Plasma Phys. 2 (1976) 533.
    The resistive internal kink and its $S^{-1/3}$ layer.
14. J. A. Krommes, Annu. Rev. Fluid Mech. 44 (2012) 175. The gyrokinetic description of microturbulence.
15. M. D. Kruskal and C. R. Oberman, Phys. Fluids 1 (1958) 275. The kinetic energy principle.
16. J.-K. Park, A. H. Boozer and A. H. Glasser, Phys. Plasmas 14 (2007) 052110. Ideal perturbed equilibrium.
17. N. C. Logan, J.-K. Park, K. Kim, Z. Wang and J. W. Berkery, Phys. Plasmas 20 (2013) 122507. NTV from the
    perturbed equilibrium (PENT).
