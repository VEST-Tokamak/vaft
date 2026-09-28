---
title: Diagram gallery
author: VEST team
date: 2026-09-27 10:05
category: guide
layout: post
permalink: /reference/diagram/
guide:
  architecture: Generated gallery of every canonical vaft.diagram builder and its committed, TikZ-derived SVG assets.
  prerequisites: None to browse; latex and dvisvgm to render a diagram yourself.
  expected: What each diagram looks like, the exact call that draws it, and which vaft.formula functions it is computed from.
related:
  api: [diagram, formula]
---
{%- assign catalog = site.data.diagram_catalog -%}
{%- if catalog.provenance.commit -%}{%- assign source_ref = catalog.provenance.commit -%}{%- elsif site.track == "development" -%}{%- assign source_ref = "develop" -%}{%- else -%}{%- assign source_ref = "main" -%}{%- endif -%}

<p class="ref-intro">Generated from <code>vaft.diagram.build.CANONICAL</code> and
<code>docs/assets/diagrams/manifest.json</code> by <code>python -m vaft.diagram.docs_catalog</code>:
<strong>{{ catalog.builders.size }}</strong> builders, <strong>{{ catalog.assets.size }}</strong> committed SVGs.
Nothing on this page is written by hand.</p>

Each picture is the committed SVG rendered from the builder's TikZ source; the caption is the exact
call that produced it, so `vaft.diagram.<call>.save("x.svg")` reproduces it. What the diagrams mean,
and the conventions they share, is explained on [Scientific diagrams]({{ site.baseurl }}/reference/diagrams/).
`python -m vaft.diagram.build --check` verifies that every SVG here still matches its source.

## Index

<div class="ref-index" data-ref-index>
<input class="ref-filter" type="search" placeholder="Filter {{ catalog.builders.size }} diagrams by name, family or formula" aria-label="Filter diagrams" data-ref-filter>
<div class="ref-table-wrap"><table class="ref-table ref-index-table">
  {% for family in catalog.families %}<tbody>
  <tr class="ref-group"><th colspan="2"><a href="#family-{{ family.name }}">{{ family.title }}</a></th></tr>
  {% for name in family.builders %}{% assign b = catalog.builders | where: "name", name | first %}<tr data-ref-row="{{ b.name }} {{ family.name }} {% for f in b.formula %}{{ f.name }} {% endfor %}"><td><a href="#{{ b.name }}"><code>{{ b.name }}</code></a></td><td>{{ b.summary | markdownify | remove: "<p>" | remove: "</p>" }}</td></tr>
  {% endfor %}</tbody>{% endfor %}
</table></div>
<p class="ref-filter-empty" hidden>No diagram matches.</p>
</div>

{% for family in catalog.families %}
## {{ family.title }} {#family-{{ family.name }}}

<p class="ref-note">Module <code>{{ family.module }}</code></p>

<div class="ref-entries" markdown="block">
{% for name in family.builders %}{% assign b = catalog.builders | where: "name", name | first %}
{% include reference/diagram-entry.html b=b ref=source_ref %}
{% endfor %}
</div>
{% endfor %}
