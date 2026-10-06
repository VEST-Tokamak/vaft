---
title: Reduced representations
author: VEST team
date: 2026-10-05 14:00
category: guide
layout: post
permalink: /reference/reduced-representations/
guide:
  architecture: A controlled taxonomy of how VAFT formulas reduce plasma information - spatial representation, reduction kind, locality and physical role - declared per formula in a Reduction docstring section, generated into tables and drawn by vaft.diagram.
  prerequisites: None.
  expected: Which formulas turn fields into profiles, profiles into scalars and dimensional into dimensionless quantities, grouped four ways, with the current/q, pressure/energy, kinetic-profile and similarity families as graphs.
related:
  api: [formula, diagram]
---

VAFT holds many reduced descriptions of one plasma state: $q(\rho) \to q_{95}$, $B_p \to l_i$,
$p(\rho) \to \beta$, and $T, B, a \to \rho_*$. Each output may be a scalar, but they are different
operations. The first extracts a feature, the second is a quadratic integral, and the third and fourth
are normalisations. This page classifies them on four independent axes:

| Axis | Values | Question |
| --- | --- | --- |
| spatial representation | `field_3d`, `field_2d`, `profile_1d`, `scalar_0d`, `flux_surface_quantity`, `boundary_quantity`, `time_series` | what space the input and the output live in |
| reduction kind | `integral`, `moment`, `quadratic_integral`, `weighted_average`, `projection`, `feature_extraction`, `extremum`, `differential`, `normalization`, `dimensionless_normalization`, `similarity_transform`, `closure`, `empirical_scaling` | which mathematical operation it is |
| locality | `point_local`, `flux_surface_local`, `edge`, `global` | where the result is defined |
| physical role | `state_coordinate`, `profile_descriptor`, `global_descriptor`, `regime_coordinate`, `similarity_coordinate`, `stability_coordinate`, `closure_input`, `closure_output` | why the reduced quantity is useful |

**Dimensionless is not 0-D.** A local $\nu^*(\rho)$ is a `dimensionless_normalization` with a
`profile_1d` output, and a global $\beta$ is one with a `scalar_0d` output. Likewise, $l_i$ is a quadratic
integral of $B_p$, not a moment of $j_\phi$.

![reduced representations]({{ '/assets/diagrams/reduced_representation_hierarchy.svg' | relative_url }})

## Declaring it

The formula's docstring is the single source of truth. A formula declares its place in the taxonomy with
a `Reduction` section:

```text
Reduction
---------
input: field_2d
output: scalar_0d
kind: quadratic_integral
locality: global
role: global_descriptor
```

The vocabulary lives in `vaft/formula/_taxonomy.py`. The catalog validates every value against it:
- a missing, repeated or unknown key is an error;
- so is a value outside the vocabulary;
- so is a `dimensionless_normalization`, `similarity_transform` or `similarity_coordinate` whose return
  unit is not `-`.

`vaft.formula.describe(name).reduction` exposes it. `list_formulas(reduction_kind=..., locality=...,
role=..., output_representation=...)` filters on it. The generated catalog snapshot carries it as
`reduction`, which is where the tables below come from. Generic numerical helpers carry no `Reduction`
section, because the taxonomy says nothing useful about them.

## Families

Each graph is the relation metadata in `vaft.formula._taxonomy.REDUCTION_FAMILIES`, laid out by
`vaft.diagram.reduction_graph`. When an edge names a formula, its reduction kind is read from that formula's
catalog entry, so the figure cannot disagree with the docstrings. A dashed edge is a step VAFT performs
elsewhere, such as a flux-surface average or a profile feature, and states its own kind.

### Current and $q$

![current and q]({{ '/assets/diagrams/reduction_graph_current_q.svg' | relative_url }})

### Pressure and energy

![pressure and energy]({{ '/assets/diagrams/reduction_graph_pressure_energy.svg' | relative_url }})

### Kinetic profiles

![kinetic profiles]({{ '/assets/diagrams/reduction_graph_kinetic_profiles.svg' | relative_url }})

### Dimensionless similarity

The $\nu_*$ and $\rho_*$ here are the Verdoolaege engineering forms. The other definitions VAFT carries
are tracked in #353 and are not declared equivalent.

![dimensionless similarity]({{ '/assets/diagrams/reduction_graph_dimensionless_similarity.svg' | relative_url }})

## Generated tables

### By spatial mapping

{% include reference/reduction-table.html by="mapping" %}

### By reduction kind

{% include reference/reduction-table.html by="kind" %}

### By locality

{% include reference/reduction-table.html by="locality" %}

### By physical role

{% include reference/reduction-table.html by="role" %}
