---
title: External scientific codes
author: VEST team
date: 2026-10-04 12:00
category: guide
layout: post
permalink: /reference/external-codes/
exclude_from_search: true
guide:
  architecture: The external solvers VAFT integrates, generated from the vaft._ecosystem registry, the adapters' own {CODE}HOME constants and the installers that exist (python -m vaft._ecosystem_catalog, issue 1648).
  prerequisites: None. The installation guide and install/README.md hold the platform-specific build instructions.
  expected: For each code, its scientific role, how VAFT runs it, who installs it, where VAFT finds it, how to check it, its upstream and literature, and what standardized result VAFT maps it into.
related:
  api: [code]
---

{%- assign e = site.data.ecosystem -%}
{%- assign commit = e.provenance.commit | default: "" -%}
An external scientific code is an **independently maintained implementation**: its upstream
project owns the solver, and VAFT owns only the integration boundary around it -- input
preparation, execution, collection, provenance, validation and mapping. None of these codes is a
Python dependency of VAFT, and installing VAFT installs none of them. Which Python packages VAFT
itself needs is a different question, answered on the
[software dependencies]({{ site.baseurl }}/reference/software-dependencies/) page.

![How an external code becomes a reproducible VAFT capability]({{ site.baseurl }}/assets/diagrams/external_code_integration.svg)

Every code below passes through the same lifecycle. A native result is never treated as IMAS
output: it is kept, and a standardized result exists only where a mapping does. Where a code runs
in the production pipelines, its rules are linked into the
[pipeline lineage explorer]({{ site.baseurl }}/reference/pipeline-graph/); where its adapter
imports other VAFT modules, the [dependency explorer]({{ site.baseurl }}/reference/dependency-graph/)
shows them.

## Overview

<div class="ref-index" data-ref-index>
<input class="ref-filter" type="search" placeholder="Filter by code, role, installation or platform" aria-label="Filter codes" data-ref-filter>
<div class="ref-table-wrap"><table class="ref-table">
<thead><tr><th>Code</th><th>Scientific role</th><th>Integration</th><th>Installation</th><th>Adapter</th><th>Platforms</th></tr></thead>
<tbody>
{%- for c in e.codes %}
<tr data-ref-row="{{ c.name }} {{ c.roles | join: ' ' }} {{ c.mode }} {{ c.installation }} {{ c.platforms | join: ' ' }} {{ c.maturity }}">
<td><a href="#code-{{ c.id }}">{{ c.name }}</a>{% if c.maturity != "supported" %} <span class="ref-flag">{{ c.maturity | replace: "_", "-" }}</span>{% endif %}</td>
<td>{{ c.roles | join: ", " }}</td>
<td>{{ c.mode_label }}</td>
<td>{{ c.installation_label }}</td>
<td><code>{{ c.adapter }}</code></td>
<td>{% if c.platforms.size > 0 %}{{ c.platforms | join: ", " }}{% elsif c.installation == "reader_only" %}wherever its results are{% else %}as the site installs it{% endif %}</td>
</tr>
{%- endfor %}
</tbody></table></div>
<p class="ref-filter-empty" hidden>No code matches.</p>
</div>

## What VAFT actually installs for you

| Code | VAFT Python install | VAFT source-build helper | User or site installation | Checker |
| --- | --- | --- | --- | --- |
{% for c in e.codes %}| {{ c.name }} | {% if c.installation == "python_package" %}yes, `vaft[{{ c.extra }}]`{% else %}no{% endif %} | {% if c.installation == "vaft_managed_source_build" %}yes, from source you supply{% else %}no{% endif %} | {% if c.installation == "site_managed" %}yes{% elsif c.installation == "reader_only" %}not run by VAFT{% else %}no{% endif %} | {% if c.checker != "" %}`{{ c.checker }}`{% else %}none{% endif %} |
{% endfor %}

"VAFT source-build helper" means an installer under `install/` configures, builds and checks a
source tree; it never downloads EFIT, which you obtain yourself under its users agreement, and
`install/README.md` says for each code where its source comes from (some installers fetch build
dependencies, such as NUBEAM's NTCC libraries or Homebrew packages). The installers that keep a
build record write `vaft-external-install.json` beside the executables -- source path and
revision, dirty-tree state, build command, toolchain and the installed files -- and the checker
reads it back; each code's entry below says whether its installers do.

## Per-code reference
{% for c in e.codes %}
<section class="ref-entry" id="code-{{ c.id }}" data-catalog="external-code" markdown="block">

### {{ c.name }}

{% if c.note != "" %}<p class="ref-note">{{ c.note }}</p>{% endif %}

| Aspect | |
| --- | --- |
| Scientific role | {{ c.roles | join: ", " }} |
| VAFT adapter | `{{ c.adapter }}`{% if commit != "" %} ([source](https://github.com/VEST-Tokamak/vaft/blob/{{ commit }}/{{ c.adapter_source }})){% endif %} |
| How VAFT runs it | {{ c.mode_label }}; {{ c.execution | join: ", " }} |
| Installation | {{ c.installation_label }}{% if c.access == "registration" %}; licensed, obtained by registration under a users agreement{% elsif c.access == "not_open_source" %}; not open source{% elsif c.access == "not_stated" %}; this repository records no distribution terms{% endif %} |
| Configuration | {% if c.home_variable != "" %}`{{ c.home_variable }}`{% elsif c.extra != "" %}`pip install "vaft[{{ c.extra }}]"`{% else %}none (reads result files){% endif %} |
| Build record | {% for p in c.provenance %}`vaft-external-install.json` from `{{ p }}`{% unless forloop.last %}; {% endunless %}{% else %}{% if c.installers.size > 0 %}not written by these installers{% else %}none{% endif %}{% endfor %} |
| Installers | {% for i in c.installers %}`{{ i }}`{% unless forloop.last %}, {% endunless %}{% else %}none in this repository{% endfor %} |
| Checker | {% if c.checker != "" %}`python {{ c.checker }}`{% else %}none{% endif %} |
| Platforms | {{ c.platforms | join: ", " | default: "as the site installs it" }} |
| Native result | {{ c.native }} |
| Standardized result | {% for s in c.standardized %}`{{ s.ids | join: "`, `" }}` via {% if s.via == "stage" %}[the `{{ s.target }}` stage]({{ site.baseurl }}{{ s.url }}){% else %}`{{ s.target }}`{% endif %}{% unless forloop.last %}; {% endunless %}{% else %}none: VAFT keeps the native result only{% endfor %} |
| Production pipeline | {% for w in c.workflow %}[`{{ w.rule }}`]({{ site.baseurl }}{{ w.url }}){% unless forloop.last %}, {% endunless %}{% else %}not run by a production pipeline{% endfor %} |
| Install instructions | {% if c.install_section != "" %}[install/README.md](https://github.com/VEST-Tokamak/vaft/blob/{% if commit != "" %}{{ commit }}{% else %}develop{% endif %}/install/README.md#{{ c.install_section }}){% else %}not documented in install/README.md{% endif %} |
{% for l in c.links %}| {{ l.role | replace: "_", " " | capitalize }} | {% if l.url != "" %}[{{ l.title }}]({{ l.url }}){% else %}{{ l.title }}{% endif %}{% if l.doi != "" %} (doi:{{ l.doi }}){% endif %} |
{% endfor %}
</section>
{% endfor %}
