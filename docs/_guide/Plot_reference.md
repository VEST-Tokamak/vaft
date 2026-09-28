---
title: Plot reference
author: VEST team
date: 2026-09-27 10:00
category: guide
layout: post
permalink: /reference/plot/
guide:
  architecture: Generated index of every plot the vaft.plot registry holds, organised subject, then view.
  prerequisites: None.
  expected: Which plots exist for a subject, which adapter draws each from an ODS, and which IDS paths it needs.
related:
  notebooks: [plotting-sample]
  api: [plot]
  data_sources: [sample-ods]
---
{%- assign catalog = site.data.plot_catalog -%}
{%- if catalog.provenance.commit -%}{%- assign source_ref = catalog.provenance.commit -%}{%- elsif site.track == "development" -%}{%- assign source_ref = "develop" -%}{%- else -%}{%- assign source_ref = "main" -%}{%- endif -%}
{%- assign kinds = catalog.subjects | map: "kind" | uniq -%}

<p class="ref-intro">Generated from <code>vaft.plot.registry</code> by <code>python -m vaft.plot.docs_catalog</code>:
<strong>{{ catalog.plots.size }}</strong> plots over <strong>{{ catalog.subjects.size }}</strong> subjects.
Each entry is a record of <code>vaft.plot.available_plots()</code>; nothing on this page is written by hand.</p>

A plot's identity is **subject / view / quantity**: the subject is what it shows physically, the view
is the kind of figure (`time`, `profile`, `field`, ...), and the quantity picks one of several when a
subject has more than one. From data, draw a plot with its `vaft.omas.plot_<name>` adapter; the
renderer `vaft.plot.<name>` takes the typed view model instead. How the shared keywords behave is
explained on [Experimental interpretation]({{ site.baseurl }}/workflows/experimental-interpretation/)
and in the [`vaft.plot` API]({{ site.baseurl }}/reference/api/).

```python
import vaft

ods = vaft.omas.sample_ods()
print(vaft.omas.available_plots(ods))              # what this input can draw
vaft.omas.plot_plasma_current_time(ods, yunit="kA")
```

## Index

<div class="ref-index" data-ref-index>
<input class="ref-filter" type="search" placeholder="Filter {{ catalog.plots.size }} plots by name, subject, view or IDS" aria-label="Filter plots" data-ref-filter>
<div class="ref-table-wrap"><table class="ref-table ref-index-table">
  <thead><tr><th>Plot</th><th>View</th><th>Quantity</th><th>Description</th></tr></thead>
  {% for subject in catalog.subjects %}{% assign plots = catalog.plots | where: "subject", subject.name %}<tbody>
  <tr class="ref-group"><th colspan="4"><a href="#subject-{{ subject.name }}">{{ subject.name }}</a>{% if subject.aliases.size > 0 %} <span class="ref-type">[{{ subject.aliases | join: ", " }}]</span>{% endif %}</th></tr>
  {% for p in plots %}<tr data-ref-row="{{ p.name }} {{ p.subject }} {{ subject.aliases | join: ' ' }} {{ p.view }} {{ p.ids | join: ' ' }}"><td><a href="#{{ p.name }}"><code>{{ p.name }}</code></a></td><td>{{ p.view }}</td><td>{{ p.quantity }}</td><td>{{ p.description | escape }}{% if p.status != "canonical" %} <span class="ref-flag ref-flag-deprecated">{{ p.status | capitalize }}</span>{% endif %}</td></tr>
  {% endfor %}</tbody>{% endfor %}
</table></div>
<p class="ref-filter-empty" hidden>No plot matches.</p>
</div>

{% for kind in kinds %}
## {{ kind | capitalize }} subjects

{% assign subjects = catalog.subjects | where: "kind", kind %}{% for subject in subjects %}
### {{ subject.name }}{% if subject.aliases.size > 0 %} [{{ subject.aliases | join: ", " }}]{% endif %} {#subject-{{ subject.name }}}

{% assign plots = catalog.plots | where: "subject", subject.name %}<div class="ref-entries" markdown="block">
{% for p in plots %}
{% include reference/plot-entry.html p=p ref=source_ref %}
{% endfor %}
</div>
{% endfor %}{% endfor %}
