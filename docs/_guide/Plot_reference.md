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
{%- comment -%}Source links are pinned to the commit the catalog was generated from, never to a branch (#1069).{%- endcomment -%}{%- assign source_ref = catalog.provenance.commit | default: "" -%}
{%- assign kinds = catalog.subjects | map: "kind" | uniq -%}

<p class="ref-intro">Generated from <code>vaft.plot.registry</code> by <code>python -m vaft.plot.docs_catalog</code>:
<strong>{{ catalog.plots.size }}</strong> registered plots over <strong>{{ catalog.subjects.size }}</strong> subjects,
plus <strong>{{ catalog.entry_points.size }}</strong> other plotting functions.
Each entry is a record of <code>vaft.plot.available_plots()</code>; nothing on this page is written by hand.</p>

A plot's identity is **subject / view / quantity**: the subject is what it shows physically, the view
is the kind of figure (`time`, `profile`, `field`, ...), and the quantity picks one of several when a
subject has more than one. From data, draw a plot with its `vaft.omas.plot_<name>` adapter; the
renderer `vaft.plot.<name>` takes the typed view model instead. How the shared keywords behave is
explained on [Experimental interpretation]({{ site.baseurl }}/workflows/experimental-interpretation/)
and in the [`vaft.plot` API]({{ site.baseurl }}/reference/api/).

Each picture is drawn from one of the packaged sample shots by the plot's own adapter, with
`python -m vaft.plot.docs_thumbnails`, and committed; the caption names the shot. A plot no packaged
sample can draw shows why instead, and a picture drawn before its renderer or sample last changed is
marked *stale* until it is re-rendered.

```python
import vaft

ods = vaft.omas.sample_ods()
print(vaft.omas.available_plots(ods))              # what this input can draw
vaft.omas.plot_plasma_current_time(ods, yunit="kA")
```

### One plot, three presentations

A plot that has a slice control (`time_slice`, a camera's `frame_index`, the PF programme's
`time_index`) can be presented three ways. They share every plot keyword; only the choice of state
differs: one index, a slider over all of them, or a sequence:

```python
vaft.omas.plot_camera_visible_image(ods, frame_index=120)                    # one state: a figure
vaft.omas.plot_camera_visible_image(ods, interactive=True)                   # a slider over the states
movie = vaft.omas.plot_camera_visible_image(ods, time_range=(0.3095, 0.3105), animation=True, fps=10)
movie.save("camera.mp4")                                                     # or .webm, .gif
```

`animation=True` animates the same slice control the slider moves. It draws exactly the selected
states, one frame each, with no interpolation. The whole sequence uses one colour scale.

The **scientific coordinate is not the playback time**:

- `fps=` or `duration=` sets only how fast the frames are shown.
- Each frame's physical time is written to `movie.metadata` and to the `camera.mp4.json` sidecar.

The **output suffix picks the writer**, and no backend name appears in the call:

- `.mp4`/`.webm` need the optional `vaft[video]` extra (PyAV).
- `.gif` needs nothing beyond Matplotlib.

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

{% assign plots = catalog.plots | where: "subject", subject.name %}{% include reference/plot-gallery.html plots=plots %}

<div class="ref-entries" markdown="block">
{% for p in plots %}
{% include reference/plot-entry.html p=p ref=source_ref %}
{% endfor %}
</div>
{% endfor %}{% endfor %}

## Other plotting functions

Plotting functions `vaft.plot` offers outside the registry: ad-hoc and analytic figures that take a
result rather than an ODS (`support`), and the cross-shot statistics kept as they are until they get a
canonical home (`legacy`). They have no subject / view identity and no `vaft.omas` adapter.

<div class="ref-entries" markdown="block">
{% for e in catalog.entry_points %}
<section class="ref-entry" id="{{ e.name }}" data-catalog="plot-function" markdown="block">
<header class="ref-head">
<h4 class="ref-name no_toc"><a href="#{{ e.name }}"><code>{{ e.name }}</code></a></h4>
<span class="ref-flags"><span class="ref-flag{% if e.status == "legacy" %} ref-flag-deprecated{% endif %}">{{ e.status | capitalize }}</span></span>
{% include reference/source-link.html src=e.source ref=source_ref key=e.name %}
</header>
<pre class="ref-signature"><code>vaft.plot.{{ e.name }}{{ e.signature | escape }}</code></pre>
{% include reference/source-code.html src=e.source ref=source_ref key=e.name %}

{{ e.summary }}

<p class="ref-note">Defined in <code>{{ e.module }}</code></p>
</section>
{% endfor %}
</div>
