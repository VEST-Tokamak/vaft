---
title: VAFT pipeline lineage explorer
author: VEST team
date: 2026-10-04 12:00
category: guide
layout: post
permalink: /reference/pipeline-graph/
exclude_from_search: true
guide:
  architecture: The implemented Snakemake production pipelines, generated from Snakemake's own graphs, PipelinePaths and STAGE_REPLICATION (python -m vaft._pipeline_graph, issue 1647).
  prerequisites: None. Automated pipelines explains how to run them; Database and data sources explains the HSDS sources.
  expected: Which rule runs after which, which file each produces and consumes, which upstream products a stage consults without scheduling them, and what each stage publishes where.
related:
  api: [database]
---

{%- assign g = site.data.pipeline_graph -%}
This page shows **what the implemented production pipelines actually do**, generated from the
documented commit
{% if g.provenance.commit %}(<a href="https://github.com/VEST-Tokamak/vaft/tree/{{ g.provenance.commit }}"><code>{{ g.provenance.commit | slice: 0, 7 }}</code></a>){% endif %}.
It is the production-lineage view of the
[Scientific architecture]({{ site.baseurl }}/reference/scientific-architecture/), and complements the conceptual workflow diagrams and the
[Automated pipelines]({{ site.baseurl }}/workflows/automated-pipelines/) guide; it does not
replace either, and it shows no live run state.

Each fact has one owner, and the explorer only reads it:

| What | Owner | View |
| --- | --- | --- |
| Which rule runs after which | Snakemake {% if g.engine.version %}{{ g.engine.version }}{% endif %} (`--rulegraph`, `--dag`) | Rules, Resolved job DAG |
| Which file each rule produces and consumes | Snakemake `--filegraph`, named by `PipelinePaths` | Artifacts |
| What each stage owns and where it is published | `vaft.database.sources.STAGE_REPLICATION` and the HSDS source catalog | HSDS publication |
| What a rule consults without scheduling it | `SCIENTIFIC_REFERENCES` in the workflow's `paths.py` | red dashed edges |

The four views are different semantics and are never merged into one edge type:

- **Execution** edges are Snakemake's own: a job must finish before the next is scheduled.
- **Scientific references** are upstream products a rule *consults* but deliberately does not
  schedule on. Pipeline 2 reads pipeline 1's magnetic EFIT and its constraints through `params`,
  so a shot pipeline 1 never reconstructed is recorded rather than failed. These edges are declared
  once, in `paths.SCIENTIFIC_REFERENCES`, which the Snakefile builds those params from.
- **Validation** products (stage plots and their manifests) are evidence about a stage product,
  never inputs to it.
- **Publication** is stage-wise: each stage replicates only the IDS it owns, to the source
  `STAGE_REPLICATION` names, and writes a replication record. The record is publication evidence,
  not a scientific product. A shot that has `diagnostics` and `eddy` but no `equilibrium` is a
  correctly published partial state, and dashed (optional or sparse) stages and sources may be
  absent for a shot without that being a failure.

The resolved job DAG is for one documented shot per pipeline, with every optional branch the
configuration has switched on, not the production shot list. Pipeline 1's `rule all` reaches its
per-shot products through checkpoints, which a dry run cannot see past, so its graph is built from
the `configured_products` target: the same products with the checkpoints taken as passed.
Generating the graph is a dry run in a temporary directory. It contacts no database or HSDS
server, runs no solver, and needs no credentials.

{% include graph/viewer.html adapter="pipeline" src="/assets/graph/pipeline-graph.json" label="VAFT production pipeline lineage graph" placeholder="Search a rule, file, stage or HSDS source" layout="dagre" %}

{% if g %}<p class="vg-meta">{% for p in g.pipelines %}{{ p.title }}: {{ p.rules }} rules and {{ p.jobs }} jobs for shot {{ p.shot }}{% unless forloop.last %}; {% endunless %}{% endfor %}.
{{ g.references | size }} declared scientific references.</p>{% endif %}

<noscript><p class="vg-noscript">The explorer is interactive and needs JavaScript. The pipelines
themselves are described in <a href="{{ site.baseurl }}/workflows/automated-pipelines/">Automated pipelines</a>.</p></noscript>

The source dependency graph of the library itself is a different relation; see the
[dependency explorer]({{ site.baseurl }}/reference/dependency-graph/). What the stages' products
*mean* -- which concept an IDS represents, which diagnostic measures it -- is the
[scientific ontology explorer]({{ site.baseurl }}/reference/ontology/). Regenerate this snapshot
locally with:

```bash
python -m vaft._pipeline_graph --output docs/_data/pipeline_graph.yml
```
