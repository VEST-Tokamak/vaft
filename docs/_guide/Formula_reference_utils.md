---
title: "Formula reference: utils"
author: VEST team
date: 2026-09-02 10:10
category: guide
layout: post
permalink: /reference/formula/utils/
guide:
  architecture: Generated per-function reference for vaft.formula.utils, read from the standardized docstrings (issue 248).
  prerequisites: None.
  expected: Definition, units, conventions, validity, limitations and literature references for every public function of the submodule.
related:
  api: [formula]
---

{% assign category = site.data.formula_catalog.categories | where: "name", "utils" | first %}
{% assign entries = site.data.formula_catalog.formulas | where: "category", "utils" %}

This page is generated from the docstrings of
[`vaft/formula/utils.py`](https://github.com/VEST-Tokamak/vaft/blob/develop/vaft/formula/utils.py):
{{ entries.size }} public functions.  The category overview and notation come from the module
docstring; every entry below is what `vaft.formula.describe("utils.<name>")` prints.
Back to the [formula reference index]({{ site.baseurl }}/reference/formula/).

## Overview

{{ category.overview }}

{% if category.notation.size > 0 %}<table class="formula-table">
  <thead><tr><th>Symbol</th><th>Meaning</th><th>Unit</th></tr></thead>
  <tbody>
  {% for row in category.notation %}<tr><td>{{ row.symbol | escape }}</td><td>{{ row.description | escape }}</td><td>{{ row.unit | escape }}</td></tr>
  {% endfor %}</tbody>
</table>

{% endif %}{% if category.conventions != "" %}{{ category.conventions }}

{% endif %}## Functions

<ul class="formula-index">
{% for f in entries %}  <li><a href="#{{ f.name }}"><code>{{ f.name }}</code></a> &mdash; {{ f.summary | markdownify | remove: "<p>" | remove: "</p>" }}</li>
{% endfor %}</ul>

{% for f in entries %}
### `{{ f.name }}` {#{{ f.name }}}

<p class="formula-signature"><code>{{ f.name }}{{ f.signature }}</code>{% if f.aliases.size > 0 %} &mdash; aliases {% for alias in f.aliases %}<code>{{ alias }}</code>{% unless forloop.last %}, {% endunless %}{% endfor %}{% endif %}</p>

{% if f.empirical or f.convention_sensitive or f.deprecated or f.shadowed_by %}<p>{% if f.empirical %}<strong>Empirical fit.</strong> {% endif %}{% if f.convention_sensitive %}<strong>Convention-sensitive.</strong> {% endif %}{% if f.deprecated %}<strong>Deprecated.</strong> {% endif %}{% if f.shadowed_by %}<em><code>vaft.formula.{{ f.name }}</code> resolves to the <code>{{ f.shadowed_by }}</code> copy; reach this one as <code>vaft.formula.{{ f.category }}.{{ f.name }}</code>.</em>{% endif %}</p>

{% endif %}{{ f.summary }}

{% if f.description != "" %}{{ f.description }}

{% endif %}{% if f.parameters.size > 0 %}<table class="formula-table">
  <thead><tr><th>Parameter</th><th>Type</th><th>Unit</th><th>Description</th></tr></thead>
  <tbody>
  {% for p in f.parameters %}<tr><td><code>{{ p.name }}</code></td><td>{{ p.type }}</td><td>{{ p.unit }}</td><td>{{ p.description | markdownify }}</td></tr>
  {% endfor %}</tbody>
</table>

{% endif %}{% if f.returns.size > 0 %}<table class="formula-table">
  <thead><tr><th>Returns</th><th>Type</th><th>Unit</th><th>Description</th></tr></thead>
  <tbody>
  {% for r in f.returns %}<tr><td>{% if r.name %}<code>{{ r.name }}</code>{% endif %}</td><td>{{ r.type }}</td><td>{{ r.unit }}</td><td>{{ r.description | markdownify }}</td></tr>
  {% endfor %}</tbody>
</table>

{% endif %}{% for s in f.sections %}<p><strong>{{ s.title }}.</strong></p>

{{ s.text }}

{% endfor %}{% if f.references.size > 0 %}<p><strong>References.</strong></p>

<ol class="formula-references">
{% for ref in f.references %}  <li>{{ ref.text | markdownify | remove: "<p>" | remove: "</p>" }}</li>
{% endfor %}</ol>

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
