---
title: VAFT scientific ontology explorer
author: VEST team
date: 2026-10-04 12:00
category: guide
layout: post
permalink: /reference/ontology/
exclude_from_search: true
guide:
  architecture: What VAFT's objects mean scientifically, generated from the plot taxonomy and registry, the Data Dictionary bridge, the diagnostic registry, the validation registry, the COCOS registry and the external-code catalog (python -m vaft._ontology_graph, issue 1702).
  prerequisites: None.
  expected: For a concept such as plasma current, which diagnostics measure it, which Data Dictionary paths represent it, which plots draw it, which checks assess it, and which conventions apply.
related:
  api: []
---

{%- assign o = site.data.ontology_graph -%}
This is the third generated graph of VAFT. The
[dependency explorer]({{ site.baseurl }}/reference/dependency-graph/) shows which module imports
which; the [pipeline lineage explorer]({{ site.baseurl }}/reference/pipeline-graph/) shows what
produced what. This one shows **what things mean**: plasma current, the diagnostics that measure
it, the Data Dictionary path that represents it, the plots that draw it and the checks that assess
it. The three never share edges, and an import or a pipeline dependency is never read as a
scientific relation. How the three views relate is mapped on
[Scientific architecture]({{ site.baseurl }}/reference/scientific-architecture/); which research modes and
physics domains its concepts serve is set out on [Research domains]({{ site.baseurl }}/reference/research-domains/).

Nothing here is written by hand. Every node and edge comes from a registry that already owns the
fact and says which (*From* in the detail panel):

| Source | Contributes |
| --- | --- |
| `vaft.plot.taxonomy` | the vocabulary: concepts, their kinds, strict aliases and families |
| `vaft.plot.registry` and `vaft.plot.backend.dd` | which concept each plot draws, the Data Dictionary paths it reads, with units, coordinates and lifecycle |
| `vaft.machine_mapping.registry` | each VEST diagnostic, its IDS, what it measures and derives, and its mapping function |
| `vaft.validation.registry` | named checks, what they check and the function that computes them |
| `vaft.data.cocos` | the COCOS convention of each code and data format |
| `vaft._ecosystem` | external codes, their adapters, the IDS their results are mapped into, and the measured Data Dictionary paths a reconstruction code is *constrained by* (EFIT's k-file inputs) |
| `vaft.formula._taxonomy` (#1626) | which quantity a formula reduces to which, through the vocabulary concept each reduction quantity declares; each formula's `Reduction` section (input and output representation, reduction kind, locality, physical role) as facets |
| `Semantics` sections of formula and process docstrings | the vocabulary quantities a function consumes and produces, where nothing else connects it; an unknown term fails generation |
| `vaft.validation.orderings` (#1627) | each physical model's approximation contract, the ordering quantities it assumes small or large, and the formula kernels that compute them |
| `vaft.formula.boundaries` | each published or derived operational boundary or design guideline, the quantity it *bounds* and the quantities it takes, through the vocabulary concept each boundary quantity declares, with the machine class, ranges, assumptions and references its source established it on |
| `DATA_ROUTINES` of `vaft._ontology_graph` | data-facing routines that select inputs from an ODS: the Data Dictionary paths each *reads*, the formulas it *applies*, and its `Semantics` section, such as `vaft.omas.edge_q.edge_q_estimate` |

The reduction graphs name quantities by their own keys (`s_hat`, `j_phi_field`, `p_profile`). A key becomes
an edge only through the vocabulary concept its quantity declares (`s_hat` is `magnetic_shear`; `j_phi`
and `j_phi_field` are both `j_tor` in two representations); a composite such as `q_features` declares
none and is listed as unresolved, so the remaining gap is visible rather than assumed.

Not consumed yet: of the machine-readable applicability contracts (#1628), the operational-boundary
registry of `vaft.formula.boundaries` is read (above); contracts declared beside formulas, codes and
workflows do not exist yet. A boundary quantity reaches a concept only through the concept it declares:
a magnitude such as \|I_p\| in MA declares none, so it is not linked to the signed `plasma_current`
and the boundary lists it as *unlinked*. Learned-model metadata (#669) lives in an external model checkout rather than in the
package, so it cannot be read offline.

**Identity is strict.** `ip` and `I_p` are registered aliases of `plasma_current` and resolve to
it; `beta_n`, `beta_p` and `beta_t` are three concepts in one family, not synonyms. A proxy *approximates* a quantity and stays a
separate concept: `estimated_q95`, from an empirical scaling, approximates `q95` of a reconstructed
equilibrium, and is never resolved to it. `f` and `fpol` are aliases of `poloidal_current_function` (F = R B_φ), which the q-profile process consumes with `psi`. A term that
no registry resolves is never turned into a new concept by guessing: it is listed below as
unresolved, with where it came from, so a gap in the vocabulary is visible. Ids are namespaced, so
an alias may share its bare spelling with a node of another kind (`tf` is an alias of the toroidal
field coil and the name of the `tf` IDS) and still resolve to exactly one subject; only an alias
that identifies two different subjects is dropped and listed as unresolved. A concept is linked to
a Data Dictionary path (*represented by*) only where that is unambiguous: a plot of that single
quantity that reads exactly one quantity path, as stored. A plot that rescales the path draws a
different quantity: `b_t` divides `tf/b_field_tor_vacuum_r` (R0 B0, in T m) by `tf/r0`, so `b_t` is
*derived from* `b_t_vacuum_r`, which that path represents. A path that stores a quantity as it is but
that no single-quantity plot reads alone is declared in `DECLARED_REPRESENTATIONS` and checked against
the Data Dictionary's unit and definition: `equilibrium/vacuum_toroidal_field/b0`, the vacuum field at
R0 in T, represents `b_t`.

Start from the compact **Concepts** view, search for a concept or an alias (`ip`, `ne`, `q95`,
`thomson`, `resistive_mhd`), and switch views to see its representations, implementations or assessment.

{% include graph/viewer.html adapter="ontology" src="/assets/graph/ontology-graph.json" label="VAFT scientific ontology graph" placeholder="Search a concept, alias, diagnostic, IDS, Data Dictionary path or check" %}

{% if o %}<p class="vg-meta">{{ o.nodes | size }} nodes and {{ o.edges | size }} typed relations in this snapshot; {{ o.unresolved | size }} unresolved terms.</p>{% endif %}

<noscript><p class="vg-noscript">The explorer is interactive and needs JavaScript. The registries
it reads are documented in the <a href="{{ site.baseurl }}/reference/plot/">plot reference</a>,
the <a href="{{ site.baseurl }}/reference/vest-diagnostics/">VEST diagnostics</a> page and the
<a href="{{ site.baseurl }}/reference/api/">API reference</a>.</p></noscript>

## Relations

| Relation | Meaning |
| --- | --- |
{% for r in o.relations %}| `{{ r[0] }}` | {{ r[1] }} |
{% endfor %}

## Unresolved terms

{% if o.unresolved.size > 0 %}These terms appear in a registry but resolve to no single concept of the vocabulary: nothing
defines them, or an alias identifies two different subjects. They are kept here, not merged by
similarity, until the vocabulary or the registry is changed on purpose.

<details><summary>{{ o.unresolved | size }} unresolved terms</summary>

| Term | From | Where |
| --- | --- | --- |
{% for u in o.unresolved %}| `{{ u.term }}` | `{{ u.origin }}` | {{ u.contexts | join: "; " }} |
{% endfor %}
</details>{% else %}None.{% endif %}

Regenerate the snapshot locally with:

```bash
python -m vaft._ontology_graph --output docs/_data/ontology_graph.yml
```
