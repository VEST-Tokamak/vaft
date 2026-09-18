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
