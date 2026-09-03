---
title: Formula reference
author: VEST team
date: 2026-09-02 10:00
category: guide
layout: post
permalink: /reference/formula/
guide:
  architecture: Index of the generated per-function reference for the vaft.formula physics layer.
  prerequisites: None.
  expected: Which submodule holds a formula, what convention it uses, and where its definition is documented.
related:
  notebooks: [confinement-scaling]
  api: [formula, process]
  data_sources: [sample-ods]
---

`vaft.formula` is documented from a single source of truth: every public function carries a
standardized docstring (summary, definition, units, convention, assumptions, validity, limitations,
numerical notes, references), and these pages are generated from those docstrings by
`python -m vaft.formula.catalog`.  Nothing here is written by hand, so the site and
`vaft.formula.describe("<name>")` always agree.

```python
import vaft.formula as F

print(F.describe("greenwald_density"))          # one formula, rendered
F.search("Sauter")                              # every formula whose text mentions it
F.list_formulas(category="stability")           # imports only that submodule
```

The discovery layer is loaded on first use only; `import vaft.formula.stability` never touches it.

## Categories

<table class="formula-table">
  <thead><tr><th>Category</th><th>Module</th><th>Functions</th><th>Contents</th></tr></thead>
  <tbody>
  {% for category in site.data.formula_catalog.categories %}<tr>
    <td>{% if category.count > 0 %}<a href="{{ site.baseurl }}/reference/formula/{{ category.name }}/">{{ category.name }}</a>{% else %}{{ category.name }}{% endif %}</td>
    <td><code>{{ category.module }}</code></td><td>{{ category.count }}</td><td>{{ category.title | escape }}</td>
  </tr>
  {% endfor %}</tbody>
</table>

## Conventions that change the number

Every function whose result depends on a sign, normalisation, COCOS or unit choice carries a
**Convention** section and is marked *convention-sensitive* on its page; empirical fits open their
**Validity** section with *Empirical fit.* and name the dataset or publication.  The recurring traps:

- **Poloidal flux, Wb versus Wb/rad.** The equilibrium helpers default to flux per radian
  (COCOS 1-8, EFIT g-files, VFIT); the IMAS Data Dictionary and the Green's functions use full weber
  (COCOS 11-18).  `poloidal_field_factor(cocos)` carries both the $2\pi$ and the orientation sign;
  `vaft.data.eqdsk.ods_psi_to_wb_per_radian_factor` settles which family an ODS holds.
- **Engineering units.** The confinement scalings, $n_G$, $\beta_N$ and the Verdoolaege dimensionless
  parameters are defined in MA, MW, $10^{19}$ m$^{-3}$ and percent; each page states what is converted
  internally and what is not.
- **Several definitions of one quantity.** Three $\nu_*$ and three $\rho_*$ definitions coexist
  (tracked in [#353](https://github.com/VEST-Tokamak/vaft/issues/353)); each page names its own.

{% assign constants = site.data.formula_catalog.categories | where: "name", "constants" | first %}
## Constants

{{ constants.overview | markdownify }}

<table class="formula-table">
  <thead><tr><th>Name</th><th>Meaning</th><th>Unit</th></tr></thead>
  <tbody>
  {% for row in constants.notation %}<tr><td><code>{{ row.symbol | escape }}</code></td><td>{{ row.description | escape }}</td><td>{{ row.unit | escape }}</td></tr>
  {% endfor %}</tbody>
</table>

{% for category in site.data.formula_catalog.categories %}{% if category.count > 0 %}
## {{ category.name | capitalize }}

{{ category.title }} &mdash; [{{ category.count }} functions]({{ site.baseurl }}/reference/formula/{{ category.name }}/).

{% assign entries = site.data.formula_catalog.formulas | where: "category", category.name %}
<ul class="formula-index">
{% for f in entries %}  <li><a href="{{ site.baseurl }}/reference/formula/{{ category.name }}/#{{ f.name }}"><code>{{ f.name }}</code></a>{% if f.empirical %} <em>(empirical)</em>{% endif %}{% if f.convention_sensitive %} <em>(convention)</em>{% endif %} &mdash; {{ f.summary | markdownify | remove: "<p>" | remove: "</p>" }}</li>
{% endfor %}</ul>
{% endif %}{% endfor %}

## Refreshing this snapshot

From a checkout of the `develop` branch, run:

```bash
python -m vaft.formula.catalog --output /path/to/vaft-gh/_data/formula_catalog.yml
```

The snapshot records the SHA-256 of every `vaft/formula/*.py` source file; documentation
validation compares them when `VAFT_REGISTRY_SOURCE` points to the corresponding source checkout.
The same text is available offline as `vaft.formula.describe("<name>")`,
`vaft.formula.search("<text>")` and `vaft.formula.list_formulas(category="<category>")`.
