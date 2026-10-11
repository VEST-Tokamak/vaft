---
title: Research domains
author: VEST team
date: 2026-10-10 09:00
category: guide
layout: post
permalink: /reference/research-domains/
guide:
  architecture: "The conceptual map of what VAFT connects: research modes, physics domains, research activities and scientific states, and how they meet through standardized representations (issue #1872). It defines no runtime abstraction and no new vocabulary."
  prerequisites: None. The Scientific architecture page says how the software is organized; this page says what research it serves.
  expected: "Which word to use for which classification, why experiment, theory, modelling and AI/ML are not physics domains, and how one physics question (equilibrium) is answered by several research modes through one shared representation."
related:
  api: [diagram, imas]
---

VAFT's tagline is *Connecting Nuclear Fusion Knowledge Across Domains for Integrated Tokamak Research*. In
that sentence, **domains** has a defined scope: the **research modes** that produce and use knowledge
(experiment, theory, modelling, AI/ML) *and* the **physics domains** that knowledge is about (equilibrium,
stability, transport, ...). VAFT connects them through standardized scientific states. It replaces neither
the modes nor the specialized codes of each domain.

This page fixes the vocabulary. It introduces no runtime class, registry or ontology: every term below is
already used by a VAFT page, diagram or registry, and each section links to the one that owns it.

## Six classifications that overlap

These are different questions about the same work, not one hierarchy. A single analysis has a place on
every axis at once.

| Classification | Answers | Examples | Owned by |
| --- | --- | --- | --- |
| **Research modes** | How is knowledge produced, evaluated and applied? | Experiment, Theory, Modelling & Simulation, Data-driven methods (AI/ML) | the common-model diagrams below |
| **Physics domains** | What is it about? | Equilibrium, MHD stability, transport and turbulence, heating, plasma initiation | no registry: the broad domains of the [Computational layers]({{ '/reference/computational-layers/' | relative_url }}) granularity rule and `integrated_scientific_framework(domain=...)` |
| **Research activities** | What is being done? | Planning, measurement, processing, reconstruction, modelling, validation, synthesis | the [fusion research ecosystem](#research-activities-and-scientific-states) |
| **Computational layers** | Where does the computation live in VAFT? | Formula, Process, Code | [Computational layers]({{ '/reference/computational-layers/' | relative_url }}) |
| **Scientific representations** | How is the result stored and exchanged? | IMAS IDS, OMAS `ODS`, the planned `DD`/`DDView` | [Fusion data structure and IMAS concepts]({{ '/reference/imas-concepts/' | relative_url }}) |
| **Applicability domains** | Under which conditions is a model or result valid? | Parameter ranges, plasma regimes, ordering assumptions | [Credibility and applicability]({{ '/reference/credibility-applicability/' | relative_url }}), [Plasma models, orderings, and scales]({{ '/reference/plasma-models/' | relative_url }}) |

Two consequences matter in practice:

- **A research mode is not a physics domain.** Experiment and modelling both study equilibrium, and
  equilibrium is studied by all four modes. "Theory" is never a subject area here, and "equilibrium" is never a
  method.
- **A physics domain is not a computational layer.** Equilibrium work spans Formulas (virial closures,
  `vaft.formula.virial`), Processes (Solov'ev equilibria, flux-surface geometry), and Codes (EFIT, CHEASE,
  TokaMaker). Several codes in one physics domain do not
  make a shared abstraction by themselves ([When an Actor is justified]({{ '/reference/computational-layers/' | relative_url }}#when-an-actor-is-justified)).

### Which word to use

- **Research modes** for experiment, theory, modelling and simulation, and data-driven methods (AI/ML).
- **Physics domains** for subject areas and physical processes.
- **Research domains** only with its scope stated, as this page states it for the tagline: the modes and the
  physics domains together.
- **Research activities** for what researchers do; **research roles** for who does it (a person may combine roles).
`domain` keeps its established technical meanings, and nothing is renamed. Among others:

- `PlotSpec.domain` is the IDS a plot reads.
- `integrated_scientific_framework(domain="equilibrium")` selects a physics domain.
- The two *application domains* of `machine_agnostic_architecture` are existing fusion experiments and future
  devices and reactor concepts: where the science is used.
- An applicability domain is a validity range.
- The spatial and topology domains of [Plasma models, orderings, and scales]({{ '/reference/plasma-models/' | relative_url }})
  say where in the plasma a model applies.

Each is unambiguous in its own context.

## Research modes and physics domains intersect

Each cell names a VAFT component that exists today. A dash means VAFT has none, and *backend only* means
VAFT runs such models but ships no trained weights. The rows are physics domains and the columns are research
modes, so one row is one physics question answered several ways.

| Physics domain | Experiment | Theory | Modelling & simulation | Data-driven (AI/ML) |
| --- | --- | --- | --- | --- |
| Equilibrium | magnetics mapping (`vaft.machine_mapping`) and EFIT reconstruction (`vaft.code.efit`) | Solov'ev and Guazzotto–Freidberg equilibria (`vaft.process.equilibrium`), virial closures (`vaft.formula.virial`) | CHEASE, TokaMaker (`vaft.code`) | — |
| MHD stability | spectral analysis of measured fluctuations (`vaft.process.fluctuation`) | stability formulas (`vaft.formula.stability`) | DCON, RDCON, STRIDE, GPEC (`vaft.code.gpec`), with post-processing in `vaft.process.mhd_stability` | backend only (`vaft.process.ml`) |
| Transport and turbulence | Thomson and charge-exchange profile fits (`vaft.process.profile`) | neoclassical and turbulence–zonal-flow formulas (`vaft.formula.neoclassical`, `vaft.formula.turbulence`) | TGLF, NEO, CGYRO (`vaft.code.gacode`) | backend only: TGLF neural-network surrogates (`vaft.code.gacode.tglf.surrogate`) |
| Heating and fast ions | — | — | NUBEAM (`vaft.code.nubeam`), GENRAY (`vaft.code.genray`) | — |
| Plasma initiation | signal-onset and active-window detection (`vaft.process.onset`) | Townsend and start-up formulas (`vaft.formula.startup`) | vacuum-field and eddy-current models (`vaft.process.electromagnetics`) | — |

The diagrams draw the same intersection for one domain. The four modes sit around one common data model, and
with `domain="equilibrium"` each mode's representative equilibrium route appears:

![Integrated scientific framework: equilibrium]({{ '/assets/diagrams/integrated_scientific_framework_equilibrium.svg' | relative_url }})

## Research activities and scientific states

Research activities are what the modes do, and scientific states are what passes between them. The ecosystem
figure (#1643) arranges research roles, six activity groups and the shared states:

- **Activities**: Ask & Plan; Produce & Observe; Process & Infer; Model & Predict; Test & Synthesize;
  Preserve & Transfer.
- **States**: planned, measured, processed, reconstructed, simulated, predicted, and qualified.

Experiment and simulation are parallel producers of states that differ in kind. A measured state is never a
reconstructed one, and a derived number is never a measurement. Validation and synthesis turn states into
*qualified* states and feed new questions back into planning.

![The fusion-research ecosystem]({{ '/assets/diagrams/fusion_research_ecosystem.svg' | relative_url }})

## Scientific integration through VAFT

Integration is the connection between modes and domains. It is not a mode or a domain itself. Two layers do
the work and stay distinct:

- **IMAS, the Common Data Model**, defines what a scientific state *means* and how it is represented: the
  equilibrium IDS, the magnetics IDS, and so on. With a common model, each mode needs one adapter instead of
  one per partner.
- **VAFT, the integrated scientific framework**, connects, runs, compares and reproduces research through that
  model. It provides the machine mappings, the domain computation (Formula, Process, Code), workflows,
  provenance, validation and analysis.

The ontology explorer adds a third, separate thing: the semantic relations among concepts, diagnostics,
representations, codes and checks. It describes the vocabulary. It does not own the data or the computation.

![With a common model]({{ '/assets/diagrams/experiment_modeling_theory_data_network.svg' | relative_url }})
![Integrated scientific framework]({{ '/assets/diagrams/integrated_scientific_framework.svg' | relative_url }})

## Worked example: equilibrium research

One physics domain, four research modes, one representation. Each step names its mode and its activity:

1. **Experiment: measurement.** Magnetics (flux loops, B-probes, Rogowski coils, diamagnetic loop) are mapped from VEST's
   native data into the IMAS `magnetics` IDS ([Experimental interpretation]({{ '/workflows/experimental-interpretation/' | relative_url }})).
2. **Experiment: reconstruction.** EFIT turns those measurements into a reconstructed equilibrium in the IMAS
   `equilibrium` IDS: a *reconstructed* state, not a measured one.
3. **Modelling: simulation.** CHEASE refines the equilibrium, and TokaMaker solves free-boundary equilibria from coil
   currents. Both read and write the same `equilibrium` IDS ([Equilibrium and kinetic profiles]({{ '/workflows/equilibrium-kinetic-profiles/' | relative_url }})).
4. **Theory: analytic reference.** Solov'ev and Guazzotto–Freidberg solutions and the virial closures give
   closed-form references that the numerical routes are checked against.
5. **Data-driven methods: inference.** A learned model can fill the same representation. VAFT has the backbone
   (`vaft.process.ml`), but no equilibrium model ships today.
6. **Any mode: comparison and validation.** Because every route lands in one representation, the routes can be
   compared directly. That comparison covers shape, $q_{95}$, $\beta$, $\ell_i$ and the operational space,
   and validation then qualifies it ([Credibility and applicability]({{ '/reference/credibility-applicability/' | relative_url }})).

The physics domain stayed the same throughout; what changed was the research mode, the activity and the
state. In the ontology explorer, search `equilibrium` to see the same subject linked to its diagnostics, IDS,
plots and checks.

## Relationship to the architecture

| Question | Page |
| --- | --- |
| How is VAFT organized as software, and where does new code go? | [Scientific architecture]({{ '/reference/scientific-architecture/' | relative_url }}) |
| What are Formula, Process and Code? | [Computational layers]({{ '/reference/computational-layers/' | relative_url }}) |
| What do the concepts mean, and how are they related? | [Scientific ontology explorer]({{ '/reference/ontology/' | relative_url }}) |
| How is scientific data represented? | [Fusion data structure and IMAS concepts]({{ '/reference/imas-concepts/' | relative_url }}) |
| Which physical model does a computation represent? | [Plasma models, orderings, and scales]({{ '/reference/plasma-models/' | relative_url }}) |
| Every canonical figure | [Diagrams]({{ '/reference/diagrams/' | relative_url }}) |

**Not on this page:** a `ResearchDomain` class, a registry of domains or modes, or any change to an API name.
If a runtime need appears, it gets its own issue.
