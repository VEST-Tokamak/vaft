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

What a plot is *for* is written once, in its renderer's docstring, and read here through
`vaft.plot.documentation(name)`: **Interpretation** says what the figure shows and which questions
it answers, **Options** what the choices that change the representation mean, and **Limitations**
what not to conclude from it alone. A GUI help panel reads the same parsed text. The option
values themselves are listed by `vaft.omas.available_plots(ods)`, not repeated in the prose.
Plots not yet written to this contract show only their one-line description.

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
differs: one index, a slider over all of them, or a sequence. Camera frames are packaged with
shot 40600:

<!-- docs-snippet: skip needs-file (writes camera.mp4, which needs the video extra) -->
```python
ods = vaft.omas.sample_ods(40600)
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

### Moving through a sequence

Every plot with more than one state to show describes how to move through them in one record,
`vaft.omas.available_plots(ods).find(name).sequence`. Its shape is the same for all three
kinds of storage:

| `kind` | `option` (picks one state) | Example | States |
| --- | --- | --- | --- |
| `samples` | `time_index` | PF programme of `vacuum_field`; shared magnetics grid of `*_spatial_*` | every sample index |
| `frames` | `frame_index` | the stored frames of `camera_visible_image*` | every frame index |
| `stored` | `time_slice` | the stored equilibrium reconstructions of a profile or 2-D map | the usable slices only |

The record holds the following:

- `states`: the values `option` accepts.
- `selected`: the state the static call draws, which is where a slider starts.
- `coordinate`/`unit`/`start`/`stop`: what the states mean.

The storage keywords keep their meaning, and navigation is the same for all of them:

- **One selector at a time.** `time=` picks an instant: it snaps to the nearest stored frame,
  sample or slice. The exception is `vacuum_field`, which computes the field at that instant from
  the PF programme. Combining `time=` with an index keyword is refused instead of one silently winning.
- **One state set for everyone.** The `interactive=True` slider and `animation=True` both move
  through exactly `states`.
- **Labels are times, not indices.** Each state's physical value comes from the data, never from the
  index alone, so a frame or slider label is the time that state was recorded.
- **Usability is not validity.** A state that exists but whose data are flagged remains in the
  sequence, and `validity=` decides how its flags are drawn. Only a state that cannot be drawn at
  all, such as an unusable equilibrium slice, is left out.

The **output suffix picks the writer**, and no backend name appears in the call:

- `.mp4`/`.webm` need the optional `vaft[video]` extra (PyAV).
- `.gif` needs nothing beyond Matplotlib.

### Several plots in one figure

A `vaft.plot.FigureComposition` places several canonical plots in one figure: a grid of cells, each
naming a plot, its keywords and the region it covers (`rowspan`/`colspan`), and the axes that move
together. `vaft.omas.compose` (or `vaft.imas.compose`) draws it from data:

```python
from vaft.plot import FigureCell, FigureComposition

ods = vaft.omas.sample_ods()

# m x 1: stacked time traces on one time axis, tick labels on the bottom one only
stack = FigureComposition.stack(
    ["plasma_current_time", "flux_loop_time_voltage", "equilibrium_time_q95"], panel_labels=True,
)
figure, axes = vaft.omas.compose(stack, ods)

# a map over two rows beside two profiles
equilibrium = FigureComposition(
    shape=(2, 2),
    cells=(
        FigureCell("equilibrium_field_psi", row=0, col=0, rowspan=2),
        FigureCell("equilibrium_profile_pressure", row=0, col=1),
        FigureCell("equilibrium_profile_q", row=1, col=1, options={"coordinate": "psi_norm"}),
    ),
)
figure, axes = vaft.omas.compose(equilibrium, ods, format="double_column")
```

Each cell is built by the plot's own recipe, so it shows what `plot_<name>` would. The keywords:

- **`share_x`** links the x axes of the cells in each column. **`share_y`** links the y axes of the
  cells in each row. **`AxisLink("x", (...cell names...))`** links any other group.
- **`title=None`** names the shot. **`panel_labels=True`** marks the cells (a), (b), ...
- **Matplotlib** returns `(Figure, axes)`, one axes per cell, and takes `format=`/`theme=`/`figsize=`.
- **`backend="plotly"`** returns one Plotly figure with the same cells and links.

`composition.to_dict()` is plain JSON and draws the same figure again.

This is not a plot's own `layout=`. That keyword spreads the series of one plot over several axes;
a composition places several plots. A cell therefore holds one panel, and an overview or a
`layout="subplots"` plot is refused there.

### Figure options and reproducible requests

On top of a `format` and `theme`, `figure_options=` sets what a figure asks for explicitly. It works
on any plot and on `compose`, with either backend. The settings it covers:

- **Axes**: title, axis labels, limits and scales.
- **Legend**: shown or hidden, location, columns, frame.
- **Ticks and grid**: grid, minor ticks, tick direction.
- **Type**: font sizes and faces. Sizes scale with the format's proportions.
- **Series**: line and marker scale.
- **Scalar fields**: colour map, colour range, colorbar label.

Every option left out is inherited from the format, theme or plot:

```python
from vaft.plot import DataSource, FigureOptions, PlotRequest

ods = vaft.omas.sample_ods()
vaft.omas.plot_plasma_current_time(
    ods, format="single_column", theme="technical",
    figure_options={"xlim": (0.30, 0.33), "legend": False, "minor_ticks": True},
)

request = PlotRequest(
    DataSource("sample", (39915,)), plot="plasma_current_time",
    format="single_column", figure_options=FigureOptions(xlim=(0.30, 0.33)),
)
figure, axes = request.render()
print(request.to_python())   # the plot call that draws it
print(request.to_cli())      # vaft plot plasma_current_time --sample 39915 --format ...
```

A `PlotRequest` describes a whole figure:

- **Data**: samples, files or database shots.
- **Content**: one plot with its keywords, or a composition.
- **Presentation**: the format, theme, backend and options.

`to_python()`, `to_cli()` and `to_dict()` write only what was set, so a figure reproduced later
follows any improvement to the canonical defaults. `to_python(pylustrator=True)` starts
Pylustrator first, for one-off finishing by hand.

Each theme pairs its text face with a MathText font. Vector files written by `save_figure` keep
their text as text: TrueType embedded in PDF, `<text>` in SVG.

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

What these objects mean scientifically -- which concept a plot draws, which diagnostic measures it, which Data Dictionary path represents it -- is generated in the [scientific ontology explorer]({{ site.baseurl }}/reference/ontology/).
