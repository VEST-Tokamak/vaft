---
title: Scientific architecture
author: VEST team
date: 2026-10-06 09:00
category: guide
layout: post
permalink: /reference/scientific-architecture/
guide:
  architecture: "The top-level map of how VAFT is organized as scientific software, connecting the contracts that the other reference pages define (issue #1803). It defines no new abstraction."
  prerequisites: None. Each section links to the page that owns its details.
  expected: Where a new API or scientific operation belongs, where side effects and mutable state are expected, which architecture view answers which question, and which concepts exist today versus are planned.
related:
  api: [formula, process, code, database, validation, mapping, plot, diagram]
---

**VAFT uses a data-centric, layered scientific computing architecture.** Scientific kernels favor
explicit transformations over hidden mutable state. Side effects are concentrated at the boundaries:
machine mapping, persistence, external-code execution, orchestration and the user interface.
Configuration and production orchestration are declarative where that helps. Typed data objects and
protocols provide the contracts between layers. VAFT favors composition over inheritance, and
reusable scientific operations over deep object hierarchies.

This page is a map, not a specification. It introduces no runtime abstraction. Every rule below is
already defined, implemented or tracked elsewhere, and each section links to the page that owns it.
When this page and an owning page disagree, the owning page wins, and the disagreement is a
documentation bug.

## Three views of one system

VAFT has three architecture views. Each answers a different question, and they must not be read
as one graph.

| View | Question it answers | Where |
| --- | --- | --- |
| **Normative scientific architecture** | How *should* scientific responsibilities be organized? | [Computational layers]({{ '/reference/computational-layers/' | relative_url }}), [Credibility and applicability]({{ '/reference/credibility-applicability/' | relative_url }}) and the ownership diagram below |
| **Observed source architecture** | How does the Python source tree *actually* import itself today? | [Dependency explorer]({{ '/reference/dependency-graph/' | relative_url }}), generated from imports |
| **Production execution and data lineage** | How do production rules, artifacts, scientific references and publications flow? | [Pipeline lineage explorer]({{ '/reference/pipeline-graph/' | relative_url }}), generated from a Snakemake dry run and the stage-replication catalog |

The dependency explorer is generated from imports. It is not a scientific dependency graph or an
architectural verdict: an import that crosses a layer shows where code *is*, and the normative view
says where it *should* be. The pipeline lineage explorer keeps execution, scientific-reference, validation-evidence
and publication relations apart, and shows artifact lineage as a view of its own.
Where VAFT's software comes
from is in [Software dependencies]({{ '/reference/software-dependencies/' | relative_url }}) and
[External scientific codes]({{ '/reference/external-codes/' | relative_url }}).

## Scientific ownership

The normative high-level model is the scientific-ownership and maturation architecture (#1645),
drawn by `vaft.diagram.scientific_ownership_architecture()`:

![Scientific ownership and maturation]({{ '/assets/diagrams/scientific_ownership_architecture.svg' | relative_url }})

In summary:

- **Research and Study** give the scientific context. Research groups Studies by membership, and
  neither executes anything.
- **A workflow or notebook** composes computation for one analysis and incubates new logic.
- **Reusable computation** produces results and evidence. It comprises data and the database, Formula,
  Process, Code and learned models.
- **Validation** interprets that evidence. It is not part of the computation.
- **An optional use policy** (accept, review, warn, reject, rerun, fallback) decides what a workflow
  does about a validation outcome. It belongs to the workflow, not to `vaft.validation`.

[Computational layers]({{ '/reference/computational-layers/' | relative_url }}) is the authoritative
definition of Formula, Process, Code and the optional Actor, and the zoomed view of the computation band.
The figure's full description is on [Diagrams]({{ '/reference/diagrams/' | relative_url }}).

## Where new functionality belongs

The placement rule follows meaning, not implementation technology or prestige.

| What it is | Owner |
| --- | --- |
| An explicit mathematical relation, or a compact analytic kernel | `vaft.formula` |
| A reusable VAFT-native scientific computation, including reduced, surrogate and approximate models | `vaft.process` |
| An adapter to an independent scientific program or solver | `vaft.code` |
| A scientific representation or portable value model | `vaft.data`, or the domain that owns the representation |
| Mapping machine-specific acquisition, configuration or raw data to the standard | `vaft.machine_mapping` |
| Assessing evidence, credibility or applicability | `vaft.validation` |
| Retrieval, persistence and publication | `vaft.database` |
| Adapting results for presentation and rendering them | `vaft.plot` |
| Study-specific composition and orchestration | the workflow or notebook that needs it |

Ask the questions in this order:

1. Is it an explicit mathematical relation? → **Formula**.
2. Is it a reusable computation implemented inside VAFT? → **Process**.
3. Does it wrap an independent scientific solver? → **Code**.
4. Is it scientific state or a portable representation? → **Data**, or the owning domain's representation.
5. Does it map machine-specific acquisition or configuration? → **Machine mapping**.
6. Does it assess evidence or scientific applicability? → **Validation**.
7. Does it retrieve, persist or publish data? → **Database**.
8. Does it adapt results for presentation or render them? → **Plot**.
9. Is it orchestration for one study? → keep it in the **workflow or notebook**.

Logic written in a workflow graduates to its semantic owner once it is reused or copied elsewhere,
needs its own tests, defines a stable operation, or enters routine production (#1642, drawn under #1645). The
[workflow graduation audit]({{ '/reference/credibility-applicability/' | relative_url }}#workflow-graduation-audit)
lists the candidates found so far.

Some facts are **not** reasons for an abstraction:

- Several codes belonging to one broad physics domain do not justify an Actor. An Actor needs the
  *same* scientific operation and a contract that makes comparison meaningful
  ([When an Actor is justified]({{ '/reference/computational-layers/' | relative_url }}#when-an-actor-is-justified)).
- A function taking an `ODS` does not make it belong to OMAS. It belongs to whatever it computes.
- Logic first written in a workflow is not owned by that workflow permanently.

## Programming model

VAFT is neither a purely functional system nor an inheritance-heavy object framework. Its tendency is
a **transformation-oriented scientific core with imperative boundaries**.

| Concern | Style |
| --- | --- |
| Primary execution | imperative, procedural scientific Python |
| Scientific kernels | function-oriented transformations: input state in, result out |
| Configuration | declarative where practical (YAML machine registries, typed config objects) |
| Production workflows | declarative DAG orchestration ([Snakemake pipelines]({{ '/workflows/automated-pipelines/' | relative_url }})) |
| Data and contracts | typed dataclasses, enums, protocols and immutable value objects |
| GUI | event-driven mutable state where the interface needs it ([Browser GUI]({{ '/workflows/gui/' | relative_url }})) |

**Comparatively explicit computational regions** include `vaft.formula`, most `vaft.process`
operations, typed equilibrium representations, plot view models and validation value objects.

**Side effects are expected** in `vaft.machine_mapping` (reading raw machine data), `vaft.database`
(persistence and publication), `vaft.code` (writing inputs and launching external programs),
production orchestration and the GUI. Side effects inside `vaft.process` are the exception rather than
the rule.

Assessment follows the same rule from the other side. A process returns its result, and assessment is
a separate optional call over that result
([the rule Credibility and applicability protects]({{ '/reference/credibility-applicability/' | relative_url }}#the-rule-this-page-protects)).

### Selective object orientation

VAFT has many classes without being primarily object-oriented. Each class has one of a few roles:

| Role | Examples |
| --- | --- |
| Value objects | `vaft.data.equilibrium.EquilibriumData`, `vaft.validation.model.ValidationReport`, the frozen plot view models in `vaft.plot.models` |
| Configuration objects | `EFITConfig` (`vaft.code.efit`), `TGLFConfig` (`vaft.code.gacode.tglf`) |
| Result objects | `vaft.code.base.CodeResult` and solver-specific result types |
| Contracts | `typing.Protocol` interfaces such as `ExecutionBackend` (`vaft.code.execution`) and `PathAccessor` (`vaft.ods_access`) |
| Stateful UI objects | GUI components |

Use an object when identity, structured state, validation, configuration or an interface contract
benefits from it. Do not turn scientific operations into mutable domain-object hierarchies for
uniformity. Prefer composition over inheritance.

### Typed boundaries and intermediate representations

A growing pattern is an explicit typed representation between an interoperability container and an
algorithm:

```text
ODS / IDS               → EquilibriumData → equilibrium algorithms
scientific profile data → GACODEProfile   → TGLF / NEO / CGYRO adapters
ODS / IDS / database    → plot view model → renderer
```

This does not ban `ODS` or IDS inputs. A complex domain algorithm should not depend on an opaque
universal container when its scientific inputs can be stated explicitly.
[Equilibrium representations]({{ '/reference/equilibrium-representations/' | relative_url }}) describes
the equilibrium case.

## Scientific data semantics

VAFT is **data-centric** in the sense of a standardized scientific state: state is standardized,
explicit transformations act on it, and the result or evidence goes on to validation, persistence or
presentation. It is not Data-Oriented Design in the memory-layout or cache-locality sense, and memory
layout is not its organizing principle.

**VAFT organizes scientific state according to IMAS semantics. Concrete in-memory and
interoperability representations may evolve independently of that semantic contract.**

| | Representation |
| --- | --- |
| Current implementation | Many scientific APIs consume or produce an OMAS `ODS` ([Fusion data structure and IMAS concepts]({{ '/reference/imas-concepts/' | relative_url }})) |
| Target architecture (#1127–#1133) | `DD` (a logical Data Entry, possibly holding several semantic IDS instances) and `DDView` (one selected scientific state) become the ordinary VAFT data model; `ODS`/`ODC` and native IMAS interfaces become interoperability projections |

`ODS` is therefore today's representation, not the permanent centre of the architecture.
The DD core is developed outside `develop` (#1127), and the documentation moves from ODS-first to
DD/DDView-first usage under #1133.

Machine differences are absorbed at the mapping boundary
([Data access and IMAS]({{ '/workflows/data-access-imas/' | relative_url }})). Everything above it sees
standardized state:

![Machine-agnostic architecture]({{ '/assets/diagrams/machine_agnostic_architecture.svg' | relative_url }})

## Composition and orchestration

| Level | What composes |
| --- | --- |
| Library operation | a direct Formula, Process or Code call |
| Workflow or notebook | scientific operations composed for one analysis |
| Production pipeline | external DAG orchestration over those operations ([Automated pipelines]({{ '/workflows/automated-pipelines/' | relative_url }})) |
| Actor | an *optional* interoperability contract for equivalent operations, off the normal execution path |

**There is no general VAFT `Workflow` object model, and none is implied here.** The repository's
`workflow/` tree holds concrete scripts, studies and Snakemake pipelines. Production orchestration is
DAG-based without making `Workflow` a public domain abstraction.

## Implementation status

A box in a diagram does not mean a public Python class exists. This table says which concepts do.

| Concept | Status |
| --- | --- |
| Formula, Process, Code adapters | implemented |
| Data and database | implemented |
| Validation | implemented, still expanding (#1639) |
| Declarative production pipelines | implemented |
| Dependency explorer, pipeline lineage explorer | implemented on `develop` |
| Learned-model infrastructure | partial, evolving (#669) |
| Actor | terminology defined; no protocol or registry yet (#1077) |
| `DD` / `DDView` / `DDCollection` | target architecture, under development outside `develop` (#1127–#1133) |
| Study | planned (#1165) |
| Research | planned (#1170) |
| Common sensitivity, linearization and uncertainty contracts | evolving (#1642) |

## Non-goals

These follow existing decisions and add none. VAFT is not meant to become:

- an inheritance-driven ontology of all fusion physics;
- one mutable Shot object that owns every possible operation;
- a mandatory Actor layer around every solver;
- a general workflow engine inside the Python API;
- a system where validation and acceptance policy are indistinguishable;
- a system where storage or database layers own scientific interpretation;
- a system where every scientific routine operates directly on one universal container;
- a second scientific hierarchy organized around ML implementation technology.

## Related reference

- [Computational layers]({{ '/reference/computational-layers/' | relative_url }}): Formula, Process, Code and Actor.
- [Credibility and applicability]({{ '/reference/credibility-applicability/' | relative_url }}): computation, evidence, validation and applicability.
- [Plasma models, orderings, and scales]({{ '/reference/plasma-models/' | relative_url }}) and
  [Asymptotic orderings]({{ '/reference/asymptotic-orderings/' | relative_url }}): what physical model a
  computation represents. This page says only where it lives.
- [Fusion data structure and IMAS concepts]({{ '/reference/imas-concepts/' | relative_url }}): the IMAS data model.
- [Database and data sources]({{ '/reference/database-data-sources/' | relative_url }}): retrieval and persistence.
- [Diagrams]({{ '/reference/diagrams/' | relative_url }}): every canonical architecture and concept figure.
