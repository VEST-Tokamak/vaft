---
title: Animation gallery
author: VEST team
date: 2026-10-14 23:00:00 +0900
category: guide
layout: post
permalink: /reference/animation/
related:
  api: [diagram]
---

Scientific animations from the same generated catalog as the [Diagram gallery]({{ site.baseurl }}/reference/diagram/).
The canonical SVG is shown first. Activate an example to load its stored scientific states; playback speed changes presentation timing, never the scientific coordinate. No interpolation or browser physics solver is used.

E×B time and length are **normalized**, with a reduced ion/electron mass ratio for illustration. Its two species are sampled at shared stored times. These are illustrative reference trajectories, not experimental data.

<div data-animation-gallery>
<div class="va-gallery-filters">
<label>Search <input type="search" data-animation-search aria-label="Search animations"></label>
<label>Family <select data-animation-family><option value="">All families</option>{% for family in site.data.diagram_catalog.families %}{% assign animated = false %}{% for builder in site.data.diagram_catalog.builders %}{% if builder.family == family.name and builder.animations.size > 0 %}{% assign animated = true %}{% endif %}{% endfor %}{% if animated %}<option value="{{ family.name }}">{{ family.title }}</option>{% endif %}{% endfor %}</select></label>
<label>Driver <select data-animation-driver><option value="">All drivers</option><option value="phase">Phase</option><option value="physical_time">Physical time</option><option value="parameter">Parameter</option><option value="process_stage">Process stage</option><option value="iteration">Iteration</option></select></label>
<label>Capability <select data-animation-capability><option value="">All capabilities</option><option value="interactive">Interactive replay</option></select></label>
</div>
{% for builder in site.data.diagram_catalog.builders %}{% for animation in builder.animations %}
<section id="{{ animation.id }}" data-catalog="animation" data-animation-card data-family="{{ builder.family }}" data-driver="{{ animation.driver.kind }}" data-capability="interactive" data-search="{{ builder.name }} {{ builder.summary | escape }} {{ animation.title | escape }} {{ builder.family }}">
<h2>{{ animation.title }}</h2>
{% include reference/animation-player.html animation=animation %}
</section>
{% endfor %}{% endfor %}
<p data-animation-empty hidden>No animation matches these filters.</p>
</div>
