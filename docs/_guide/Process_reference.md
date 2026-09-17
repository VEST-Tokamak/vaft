---
title: Process reference
author: VEST team
date: 2026-09-03 10:00
category: guide
layout: post
permalink: /reference/process/
guide:
  architecture: Index of the generated per-function reference for the vaft.process computation layer.
  prerequisites: None.
  expected: Which submodule holds a routine, whether it is machine-independent or VEST policy, what it does to its input, and where its method and defaults came from.
related:
  notebooks: [confinement-scaling]
  api: [process, formula, mapping]
  data_sources: [sample-ods]
---

`vaft.process` is the **computation layer** of VAFT.  Almost every function in it takes plain NumPy
arrays and scalars and returns arrays, tuples or dataclasses; it does not read or write ODS.  The
ODS-aware layer lives in `vaft.omas` (mainly `vaft.omas.process_wrapper`), which pulls geometry and
signals out of an ODS, calls into `vaft.process`, and writes the results back.

> Rule of thumb: **`vaft.process` is the math, `vaft.omas.compute_*` is the API you usually call.**

Every public processing function is documented under one contract, and these pages are generated
from those docstrings by `python -m vaft.process.catalog`.  Where a formula page answers *what is
this quantity and when is it valid*, a process page answers *how is this input turned into this
output*: the parameters and returns with their units, the processing steps in order, the defaults
that change the result and what kind of value each is, the conventions assumed, the machine scope,
the limitations, and the provenance of the method.  Nothing here is written by hand, so the site
and `vaft.process.describe("<name>")` always agree.

```python
import vaft.process as P

print(P.describe("repair_clipped_interval"))     # one routine, rendered
P.search("vest.yaml")                             # every routine whose text mentions it
P.list_processes(category="signal_processing")   # imports only that submodule
```

The discovery layer is loaded on first use only; `import vaft.process.signal_processing` never
touches it.

## Categories

The layer is being brought under the contract one submodule at a time
([#252](https://github.com/VEST-Tokamak/vaft/issues/252)).  A category gets a reference page when
every function in it conforms; until then its count is shown and its page is absent.

<table class="formula-table">
  <thead><tr><th>Category</th><th>Module</th><th>Functions</th><th>Under contract</th><th>Contents</th></tr></thead>
  <tbody>
  {% for category in site.data.process_catalog.categories %}<tr>
    <td>{% if category.conforming %}<a href="{{ site.baseurl }}/reference/process/{{ category.name }}/">{{ category.name }}</a>{% else %}{{ category.name }}{% endif %}</td>
    <td><code>{{ category.module }}</code></td>
    <td>{{ category.count }}</td>
    <td>{% if category.conforming %}yes{% else %}{{ category.documented }} of {{ category.count }}{% endif %}</td>
    <td>{{ category.title | escape }}</td>
  </tr>
  {% endfor %}</tbody>
</table>

## Machine-independent or VEST policy

Every function's **Applicability** section opens with one of two sentences, and each page badges
it.  *Machine-independent* means the caller supplies every machine-specific number -- an
acquisition rail, a chord length, a processing window -- and the routine infers nothing about the
hardware; the VEST values are read from `vest.yaml` by `vaft.machine_mapping`, never here.
*VEST-specific* means the routine embeds a VEST constant, a VEST column name or a VEST
acquisition-era rule, and its page says which and for what shot range.

## Where a default came from

A default that changes the result is classified on its page as one of: physical constant,
literature value, diagnostic calibration, empirical estimate, validated-workflow default,
machine-specific setting, acquisition-era policy, legacy compatibility value, or numerical
convenience.  A legacy compatibility value is one that arrived with a ported routine and whose
derivation is not recorded; the page says so rather than inventing one.

## Refreshing this snapshot

From a checkout of the `develop` branch, run:

```bash
python -m vaft.process.catalog --output docs/_data/process_catalog.yml
```

The snapshot records the SHA-256 of every `vaft/process/*.py` source file, the private
`_equilibrium_parametric.py` included; documentation validation compares them when
`VAFT_REGISTRY_SOURCE` points to the corresponding source checkout.  The same text is available
offline as `vaft.process.describe("<name>")`, `vaft.process.search("<text>")` and
`vaft.process.list_processes(category="<category>")`.
