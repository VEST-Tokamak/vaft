---
title: Software dependencies
author: VEST team
date: 2026-10-04 12:00
category: guide
layout: post
permalink: /reference/software-dependencies/
exclude_from_search: true
guide:
  architecture: The Python packages VAFT needs, grouped by the capability each provides, generated from pyproject.toml and the vaft._ecosystem registry (python -m vaft._ecosystem_catalog, issue 1648).
  prerequisites: None.
  expected: Which capabilities are part of the core, which an optional extra enables, and which only development needs, with each requirement exactly as pyproject.toml states it.
related:
  api: []
---

{%- assign e = site.data.ecosystem -%}
A software dependency is a Python package VAFT imports to implement part of VAFT itself. External
scientific solvers such as EFIT or CHEASE are a different relation -- independent implementations
VAFT integrates through `vaft.code` -- and are described on the
[external scientific codes]({{ site.baseurl }}/reference/external-codes/) page.

![Which software capabilities are core and which are optional]({{ site.baseurl }}/assets/diagrams/software_dependency_ecosystem.svg)

**Optional is not experimental, and required is not more important.** A capability is optional
when the core works without it; a mandatory storage library is less scientifically specialized
than an optional physics backend. Version constraints have one owner, `pyproject.toml`, and are
shown here exactly as it states them. Which modules import which package is shown, from the
source, by the [dependency explorer]({{ site.baseurl }}/reference/dependency-graph/).

## Required core

`pip install vaft` installs every package below.

{% for cap in e.capabilities %}{% if cap.scope == "runtime" %}{% assign rows = e.dependencies | where: "capability", cap.id %}
### {{ cap.title }}

{{ cap.summary | capitalize }}.

| Package | Requirement | Used for |
| --- | --- | --- |
{% for d in rows %}| `{{ d.name }}` | `{{ d.requirement }}` | {{ d.purpose }} |
{% endfor %}
{% endif %}{% endfor %}

## Optional capabilities and development

Each extra adds a capability the core does not need. `dev` and `architecture` are for working on
VAFT itself.

| Extra | Install | Capability | What it enables | Requirements |
| --- | --- | --- | --- | --- |
{% for x in e.extras %}{% assign cap = e.capabilities | where: "id", x.capability | first %}| `{{ x.name }}` | `{{ x.install }}` | {{ cap.title }} | {{ x.purpose }} | {% for r in x.requirements %}`{{ r }}`{% unless forloop.last %}, {% endunless %}{% endfor %} |
{% endfor %}

Check what the current environment can actually use with `python install/check_vaft_environment.py`.
