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

## Magnetic island

```python
import vaft

d = vaft.diagram.magnetic_island(
    m=3, n=2, width=0.16, phase=0.0, projection="poloidal",
    r_s=0.55, elongation=1.7, triangularity=0.4,   # D-shaped plasma; defaults give a circle
)
d            # displays inline in Jupyter (SVG)
d.save("island.svg")
d.tikz       # the LaTeX/TikZ source; needs no TeX installation
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
| geometry | Miller shaping: $R = R_0 + r\cos(\theta + \arcsin\delta(r)\,\sin\theta)$ and $Z = \kappa r\sin\theta$, with $\delta(r) = \delta\,r$ so the surfaces stay nested and become circular towards the axis. Lengths are in units of the minor radius $a$ |
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
vaft.diagram.hugill(elongation=1.0, q_limit=2.0)
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
