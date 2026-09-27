---
title: Geometric approximations
author: VEST team
date: 2026-09-26 11:00
category: guide
layout: post
permalink: /reference/geometric-approximations/
guide:
  architecture: The slab, cylindrical and toroidal reductions plasma theory uses, with geometry and ordering kept on separate axes; formulas in vaft.formula.geometry.
  prerequisites: None.
  expected: For each approximation its coordinates, field, ordering, canonical formulas, what it keeps and drops, and how it connects to the next one.
related:
  api: [formula, diagram]
---

Analytic tokamak theory seldom works in the real torus. It works in a slab, a cylinder or an expanded
torus, and it carries a few coefficients over from the real equilibrium. This page sets out those
reductions using the literature's names. The formulas are in
[`vaft.formula.geometry`]({{ '/reference/formula/geometry/' | relative_url }}).

## Two axes, not one hierarchy

*Geometry* says what space the model lives in: slab, cylindrical or toroidal. *Ordering* says which
scales are assumed small: local ($|x| \ll r$), large aspect ratio ($\epsilon = a/R_0 \ll 1$) or neither
(global). The two are independent. A slab may carry $q$, $\hat s$, $R_0$ or a $1/R_0$ curvature
inherited from a torus and still be a slab. Large aspect ratio is an ordering, not a geometry: some
large-aspect-ratio theories stay toroidal.

![geometry and ordering]({{ '/assets/diagrams/geometry_ordering_map.svg' | relative_url }})

| Representation | Geometry | Typical ordering | Retained | Neglected or parameterized |
| --- | --- | --- | --- | --- |
| straight slab | slab | local | local gradients, wave propagation | curvature, magnetic shear |
| sheared slab | slab | local | local gradients, magnetic shear, resonance | global geometry; curvature unless added |
| curved slab | slab | local | shear plus a prescribed curvature, e.g. $1/R_0$ | global toroidal geometry |
| cylindrical tokamak | cylindrical | large aspect ratio, leading order | radial structure, current profile, $q(r)$, rational surfaces | toroidicity, shaping |
| large-aspect-ratio torus | toroidal | $\epsilon \ll 1$ | selected $O(\epsilon)$ toroidal corrections | higher orders in $\epsilon$ |
| general axisymmetric torus | toroidal | global | shaping, curvature, toroidal coupling | nothing geometric |

## Conventions used throughout

* Perturbations are $e^{i(m\theta - n\phi)}$, as in `vaft.formula.helical_phase`. Both mode numbers
  are positive and the helicity is in the minus sign.
* $q$ is a magnitude, whatever sign a COCOS gives it. $\hat s$ keeps the sign of $d|q|/dr$:
  reversed shear ($\hat s < 0$) flips the sign of $L_s$.
* A torus is straightened at $R_0$ with $z = R_0\phi$ and $B_z \leftrightarrow B_\phi$. Along that $z$,
  the harmonic has wavenumber $-n/R_0$.
* A sheared slab is $\mathbf B = B_0(\hat{\mathbf z} + (x/L_s)\hat{\mathbf y})$, with $z$ **along the field
  at $x = 0$** and $y$ the binormal in the surface. This field-aligned $z$ is not the straightened-torus
  $z = R_0\phi$.
* The local slab of a surface $r_0$ is that frame, with $x = r - r_0$. There the harmonic has:
  * $k_y = m/r_0$;
  * $k_z = (m - nq_0)/(q_0R_0)$, which is zero on a rational surface;
  * $L_s$, which is **signed**: positive shear gives $L_s = -q_0R_0/\hat s < 0$.

  `local_slab_from_cylinder` returns $(k_y, L_s, k_z)$ in the order `sheared_slab_parallel_wavenumber`
  takes them. `shear_length_from_q_R0_s` returns the magnitude $qR_0/|\hat s|$.

## Straight slab

* **Coordinates:** $(x, y, z)$: radial, binormal, along $\mathbf B$.
* **Field:** $\mathbf B = B_0\hat{\mathbf z}$, uniform.
* **Ordering:** local. Gradients are taken over a region small against every equilibrium scale.
* **Formula:** $\tilde f \propto e^{i(k_xx + k_yy + k_zz)}$, and $k_\parallel = \mathbf k\cdot\mathbf B/B = k_z$
  (`slab_parallel_wavenumber`).
* **Keeps:** wave propagation and local density or temperature gradients.
* **Drops:** magnetic curvature and magnetic shear, unless they are added separately.
* **Used for:** drift waves, and local dispersion relations before shear or curvature enter.

## Sheared slab

* **Field:** $\mathbf B = B_0(\hat{\mathbf z} + (x/L_s)\hat{\mathbf y})$ (`sheared_slab_field`), or equivalently
  $B_y(x) \simeq B_y'x$ near a resonant surface.
* **Shear:** $\hat s = (r/q)\,dq/dr$ (`vaft.formula.shear_from_r_q`). Under the local, large-aspect-ratio
  ordering, $|L_s| = qR_0/|\hat s|$ (`shear_length_from_q_R0_s`).
* **Formula:** $k_\parallel(x) \simeq k_z + k_y x/L_s$ (`sheared_slab_parallel_wavenumber`). The
  perturbation is field-aligned on one surface and $|k_\parallel|$ grows linearly away from it.
* **Keeps:** the resonance and its localisation.
* **Drops:** global geometry, and curvature unless it is added.
* **Used for:** the inner layer of tearing modes (FKR, Rutherford), drift waves with shear, and the
  local limits of gyrokinetics.

## Curved slab

Curved slab is a *family* of local models, not one model. VAFT fixes no closure here: the curvature
coefficient is whatever a particular model inherits from its parent torus.

* **Coordinates and field:** those of the straight or sheared slab.
* **Curvature:** toroidal curvature is added as a parameter, either an effective curvature
  $\boldsymbol\kappa = \mathbf b\cdot\nabla\mathbf b$ (typically $|\kappa| \simeq 1/R_0$ pointing to the major
  axis), or an effective gravity $g \sim v_{th}^2/R_0$.
* **Ordering:** local. The curvature is a constant coefficient over the region.
* **Keeps:** the interchange drive (the pressure gradient against the curvature) and the $\nabla B$ and
  curvature drifts.
* **Drops:** the poloidal variation of curvature (good and bad curvature sides), the geometry of the
  real flux surfaces, and toroidal mode coupling.
* **Next more realistic model:** a local toroidal (flux-tube) model, where $\kappa$ varies along the
  field line.
* **Used for:** interchange and Rayleigh–Taylor analogues, resistive-g modes, and reduced edge and SOL
  turbulence.

## Cylinder, screw pinch, cylindrical tokamak

* **Coordinates:** $(r, \theta, z)$.
* **Field:** $\mathbf B = B_\theta(r)\hat{\boldsymbol\theta} + B_z(r)\hat{\mathbf z}$.
* **Safety factor:** $q(r) = rB_z/(R_0B_\theta)$ (`cylindrical_safety_factor_from_r_B`).
* **Resonance:** $k_\parallel = (m - nq)/(qR_0)$ (`cylindrical_parallel_wavenumber`), so
  $k_\parallel = 0 \Leftrightarrow q(r_s) = m/n$.
* **The three are related but not synonyms:**
  * A *straight cylinder* is only the coordinate system.
  * A *screw pinch* is a cylindrical equilibrium with both $B_\theta(r)$ and $B_z(r)$, so its field lines
    are helices. Pure $\theta$- and $z$-pinches are its limits.
  * A *cylindrical (straight) tokamak* is a screw pinch that is periodic in $z$ with length $2\pi R_0$
    and has $B_z \gg B_\theta$, both taken from a torus.
* **Keeps:** radial structure, the current profile, $q(r)$ and rational surfaces.
* **Drops:** $1/R$ variation of $B_\phi$, toroidal curvature, inboard/outboard asymmetry, the Shafranov
  shift, toroidal coupling of harmonics, and shaping. The helical field lines are still curved, which is
  what drives Suydam interchange.
* **Used for:** Newcomb's equation, the internal kink, the outer tearing problem and $\Delta'$.

## Large-aspect-ratio ordering

For a circular torus $R = R_0 + r\cos\theta$, so

$$\frac{1}{R} = \frac{1}{R_0}\left[1 - \frac{r}{R_0}\cos\theta + O(\epsilon^2)\right],\qquad
B_\phi = B_0\frac{R_0}{R} = B_0\left[1 - \frac{r}{R_0}\cos\theta + O(\epsilon^2)\right].$$

The second expression is `vaft.formula.vacuum_toroidal_field`.

**Worked $O(\epsilon)$ example: toroidal coupling.** Any factor of $1/R$ multiplying a harmonic
$e^{im\theta}$ creates sidebands:

$$\frac{e^{im\theta}}{R} = \frac{1}{R_0}\left[e^{im\theta} - \frac{\epsilon}{2}\left(e^{i(m+1)\theta}
+ e^{i(m-1)\theta}\right)\right] + O(\epsilon^2),\qquad \epsilon = r/R_0 .$$

So in a torus an $m/n$ perturbation drags along $(m \pm 1)/n$ components of relative size
$\epsilon/2$. In a cylinder it cannot.

The expansion is used in two different ways:

1. **Large-aspect-ratio toroidal model.** It keeps selected $O(\epsilon)$ terms:
   * curvature;
   * inboard/outboard asymmetry;
   * coupling of $m$ to $m \pm 1$.

   It is still toroidal. It keeps the $O(\epsilon)$ curvature and $1/R$ terms and drops $O(\epsilon^2)$.
   The Shafranov shift enters at $O(\epsilon(\beta_p + l_i/2))$ relative to $a$. Shaping is outside the
   circular expansion used here, although shaped large-aspect-ratio models exist (for example Miller
   local equilibria). Typical uses are the $s$–$\alpha$ ballooning model, toroidal Alfvén gaps and
   neoclassical trapping.
2. **Cylindrical tokamak limit.** It keeps only $O(1)$: $R \simeq R_0$, and the toroidal direction is
   straightened as $z = R_0\phi$. The cylindrical tokamak is the **leading-order limit** of the
   large-aspect-ratio expansion. It is not the same thing as large aspect ratio.

## Local sheared-slab reduction

Near a surface $r_0$, with $x = r - r_0$ and $|x| \ll r_0$, expand $q \simeq q_0 + q_0'x$ and take the
field-aligned frame: $z$ along $\mathbf B(r_0)$, $y$ the binormal. The harmonic $e^{i(m\theta - n\phi)}$
has there

$$k_y = \frac{m}{r_0},\qquad L_s = -\frac{q_0R_0}{\hat s},\qquad k_z = \frac{m - nq_0}{q_0R_0}$$

(`local_slab_from_cylinder`). Along the straightened torus the same harmonic has $-n/R_0$; along the
field it has $k_z$. On a rational surface $q_0 = m/n$, so the resonance $q(r_s) = m/n$ becomes
$k_\parallel(0) = k_z = 0$ and the shear becomes $k_\parallel' = k_y/L_s = -k_y\hat s/(q_0R_0)$. The test
suite checks that this slab reproduces the cylinder's $k_\parallel$ to first order in $x$.

![mode-number mapping]({{ '/assets/diagrams/mode_number_mapping.svg' | relative_url }})

A local slab does **not** need a cylinder as an intermediate step. It can be expanded from a toroidal
equilibrium directly. What defines it is locality relative to the equilibrium scale, not large
aspect ratio. Turbulence and interchange models often keep toroidal coefficients in a slab, such as
$|L_s| \simeq qR_0/|\hat s|$ and, in curved slabs, $\kappa \simeq 1/R_0$.

## The same field line in each geometry

| Toroidal | Cylindrical | Sheared slab |
| --- | --- | --- |
| ![torus]({{ '/assets/diagrams/field_line_geometry_toroidal.svg' | relative_url }}) | ![cylinder]({{ '/assets/diagrams/field_line_geometry_cylindrical.svg' | relative_url }}) | ![slab]({{ '/assets/diagrams/field_line_geometry_slab.svg' | relative_url }}) |

## Where these lead

```text
cylinder / screw pinch  -> Newcomb equation, internal kink, outer tearing (Delta')
cylinder -> sheared slab -> resistive inner layer: FKR, Rutherford, modified Rutherford
straight / sheared / curved slab -> drift waves, interchange, reduced edge turbulence, local gyrokinetics
```

The tearing diagrams (`rational_surface`, `delta_prime`, `tearing_layer_matching`) start from the
cylinder and sheared slab described here. See [Scientific diagrams]({{ '/reference/diagrams/' | relative_url }}).

## References

1. J. Wesson, *Tokamaks*, 4th ed., Oxford University Press (2011), Ch. 3, 6 and 8.
2. J. P. Freidberg, *Ideal MHD*, Cambridge University Press (2014), Ch. 9 and 11.
3. H. P. Furth, J. Killeen and M. N. Rosenbluth, Phys. Fluids 6 (1963) 459.
4. W. Horton, Rev. Mod. Phys. 71 (1999) 735.
