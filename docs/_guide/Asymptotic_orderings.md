---
title: Asymptotic orderings
author: VEST team
date: 2026-10-05 16:00
category: guide
layout: post
permalink: /reference/asymptotic-orderings/
guide:
  architecture: The asymptotic ordering parameters reduced plasma models expand in - Lundquist number, inertial lengths, gyroradii, Knudsen numbers, magnetization, evolution times - as explicit-length kernels in vaft.formula.ordering, kept separate from performance and similarity quantities.
  prerequisites: None.
  expected: What each ordering parameter tests, which length or species it must name, which model assumption it bears on, and a worked timescale hierarchy.
related:
  api: [formula, diagram]
---

Some dimensionless numbers say how well a plasma performs, or where it sits in similarity space
($\beta_N$, $\rho_*$, $\nu_*$). Others are the small or large parameters a model's derivation expands
in. A Lundquist number tests whether resistive diffusion is slow compared with Alfvénic dynamics.
$d_i/L$, $\rho_i/L$, a Knudsen number or $\Omega\tau$ each test a different reduction. This page covers
the second kind: **asymptotic ordering parameters** (#1627). The question they answer is which reduced
models the scale separation of a given state justifies.

The kernels are in [`vaft.formula.ordering`]({{ '/reference/formula/ordering/' | relative_url }}). Their
inputs come from VAFT's Alfvén speed, Spitzer resistivity, gyrofrequency and Larmor radius formulas, and
the inertial length calls the plasma frequency. The
characteristic length, the resistivity and the collision time are always explicit arguments, so a global
scale cannot be silently swapped for a layer scale.

## The parameters and what they test

| Quantity | Kernel | Tests | Must name |
| --- | --- | --- | --- |
| $\tau_A = L/v_A$, $\tau_R = \mu_0L^2/\eta$ | `alfven_time`, `resistive_diffusion_time` | the two ends of the MHD hierarchy | $L$, $\eta$ model, $Z_{eff}$, $\ln\Lambda$ |
| $S = \tau_R/\tau_A$ | `lundquist_number` | resistivity perturbatively small **at that scale** | $L$: global $a$ or a layer width |
| $R_m = \mu_0VL/\eta$ | `magnetic_reynolds_number` | flux freezing in a flow | a measured or modelled $V$ |
| $d_i/L$, $d_e/L$ | `inertial_length` | Hall and electron-inertia corrections | species; global $a$ vs layer $\delta$ |
| $\rho_s/L_T$, $\rho_i/L$ | `sound_gyroradius`, `particle.larmor_radius` | finite-Larmor-radius corrections | the local gradient length |
| $Kn = \lambda/L$ | `thermal_speed`, `braginskii_*_collision_time`, `mean_free_path`, `knudsen_number` | local collisional fluid closure | species; $L_T$, $L_n$ or $L_\parallel$ |
| $\chi = \lvert\Omega\rvert\tau$ | `magnetization` | strongly magnetized transport | species |
| $\tau_{evol}/\tau_A$ | `evolution_time` | the plasma evolves through a sequence of equilibria | the state variable ($I_p$, $W$, axis position) |

Some distinctions must not be blurred:
- **A large global $S$ does not make resistivity irrelevant in a tearing layer.** The layer's own $S$, and
  $d_i/\delta$, $\rho_s/\delta$, decide that.
- **Global $\rho_*$ is a similarity coordinate. Local $\rho_i/L_T$ is an FLR ordering.** They are not
  interchangeable.
- **$\tau_{evol}/\tau_A \gg 1$ is physics.** It says nothing about whether an EFIT run converged, and
  convergence says nothing about it.
- **$Kn$ and $\Omega\tau$ are independent.** A plasma can be strongly magnetized and collisionless along
  the field at the same time.

## A worked hierarchy

![timescale hierarchy]({{ '/assets/diagrams/timescale_hierarchy.svg' | relative_url }})

`timescale_hierarchy` evaluates the hierarchy of one illustrative low-field spherical-tokamak state rather
than assuming it. Every time comes from a formula kernel; only the pulse length is an input.
- **Clearly ordered:** $S$ and $\tau_{evol}/\tau_A$, both about $10^4$.
- **Of order one, so not ordered:**
  - $\tau_{pulse}/\tau_R \approx 0.7$. Resistive relaxation has to be checked, not assumed either way.
    $\tau_R$ carries no geometric factor, and the slowest cylindrical mode decays $j_{01}^2 \approx 5.8$
    times faster.
  - $d_i/a \approx 0.3$, so Hall corrections are not globally negligible.

These numbers come from an illustrative state, not a measured discharge. Answering the same questions
across the VEST database is the ordering atlas of #1629.

## Ordering contracts

Whether an ordering holds depends on the model that assumes it. Ideal single-fluid MHD needs $S \gg 1$,
$d_i/L \ll 1$, $\rho_i/L \ll 1$ and $\tau_{evol}/\tau_A \gg 1$ all at once, not $S$ alone. A local collisional
fluid needs $Kn \ll 1$, and a strongly magnetized one needs $\Omega\tau \gg 1$.

These contracts will be registered as `ApproximationContract` objects of `vaft.validation.applicability`
(#1639, PR #1695). They report a continuous ordering margin per assumption ($\mp\log_{10}x$), not a
valid/invalid flag. They follow in a second #1627 PR once that module is on develop.

## References

1. S. I. Braginskii, in *Reviews of Plasma Physics*, Vol. 1, Consultants Bureau (1965), p. 205.
2. J. D. Huba, *NRL Plasma Formulary*, Naval Research Laboratory (2019).
3. D. Biskamp, *Magnetic Reconnection in Plasmas*, Cambridge University Press (2000).
4. H. R. Strauss, Phys. Fluids 19 (1976) 134.
5. F. L. Hinton and R. D. Hazeltine, Rev. Mod. Phys. 48 (1976) 239.
