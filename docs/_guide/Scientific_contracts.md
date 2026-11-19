---
title: Scientific contracts
author: VEST team
date: 2026-10-25 09:00
category: guide
layout: post
permalink: /reference/scientific-contracts/
guide:
  architecture: "Where VAFT's scientific contracts are defined: quantity identity, units, COCOS and machine polarity, Fourier conventions, applicability, validation evidence and provenance. Records who owns each, how it is enforced, and the gaps the scientific-contracts interoperability work closes."
  prerequisites: The Scientific architecture and Credibility and applicability pages.
  expected: Which module is authoritative for a definition, a transformation and a verdict; which conventions are machine-readable and which are prose only; and why VAFT extends its existing taxonomy instead of adding a quantity registry.
related:
  api: [formula, process, validation, mapping, code]
---

VAFT already defines most of its scientific contracts. They live in the modules that own the
science, though, and nothing connects them. This page is the Phase A audit of the Scientific Contracts
initiative (provenance: umbrella #2160). It inventories the contracts, names an owner for each one, traces
five representative paths end to end, and records the minimal work Phase B has to do. Phase B has
since added `vaft.formula.units`, `vaft.data.fourier` and `vaft.data.representation`; the tables
below record the state at the audited commit.

It audits `develop` at `7b1fcf192`. Symbols are cited by name so that line drift does not break the
references. When this page and an owning module disagree, the module wins, and the disagreement is a
documentation bug.

## Four questions kept apart

The audit keeps four questions separate because the code already does, and merging them was the
main risk the initiative identified:

| Question | Answered by | Never answered by |
| --- | --- | --- |
| Do two values mean the same physical thing? | quantity identity: `vaft.plot.taxonomy`, `vaft.formula.boundaries` | matching units or dimensions |
| Is this representation the one it claims to be? | convention checks: `vaft.validation.cocos`, `vaft.validation.perturbation_phase` | a declaration alone |
| Do the model's assumptions hold here? | `vaft.validation.applicability` with `vaft.validation.orderings` | solver convergence |
| Is the result supported by evidence? | `vaft.validation.credibility`, six axes (E/T/I/A/N/V) | any single score |

## Inventory and coverage

Coverage labels describe the *software* (existing, partial, missing, unresolved). They are not
physical applicability verdicts. Enforcement is one of: **doc** (prose only), **structural** (checked
when docs or catalogs are generated), **runtime** (checked when called) or **independent** (checked
against data the result did not use).

### Units, dimensions and normalization

| Concept | Owner | Implementation | Representation | Enforcement | Coverage | Gap |
| --- | --- | --- | --- | --- | --- | --- |
| Units of formula and process parameters and returns | `vaft.formula`, `vaft.process` | `vaft._docstring._UNIT_TAG`, `ParamDoc.unit`, `ReturnDoc.unit` | free-text `[unit]` tag | structural: tag present (`test_every_parameter_and_return_carries_a_unit_tag`) | partial | Tags are never parsed, so `[MA]`, `[1e19 m^-3]` and `[varies]` are equally opaque. **Definition not machine-readable.** |
| Dimensionless reductions | `vaft.formula._taxonomy` | `Reduction`, `formula.catalog._reduction` | frozen dataclass of vocabulary strings + `[-]` return | structural | existing | Only check that relates a declared kind to a unit. |
| Physical dimensions for scaling laws | `vaft.formula.dimensional` | `BASE_DIMENSIONS`, `PHYSICAL_DIMENSIONS`, `dimensionless_groups` | exponent tuples keyed by local symbols | runtime (algebra) | existing | No unit strings and no scale factors, and not linked to taxonomy names. |
| Engineering-unit scaling (A↔MA, W↔MW, fraction↔%, m⁻³↔10¹⁹ m⁻³) | none | hand-written at each call site (`_synthetic_kinetic_profiles` `/1e6`, `plot.history` column sniffing, `pr08_mhd_state._Direct`, `gacode.inputs.DENSITY_SCALE`, `efit.kfile` `unit_scale`) | inline literals | none | missing | **Missing interoperability infrastructure.** Each call site carries its own conversion. |
| Unit of a boundary quantity | `vaft.formula.boundaries` | `BoundaryQuantity.unit`, `same_quantity` | string, compared literally | runtime (window edges must match) | partial | `line_average_density [1e19 m^-3]` and `[1e20 m^-3]` count as different quantities when they are one quantity in two units. |
| Poloidal flux Wb vs Wb/rad | `vaft.data.cocos` (exponent), `vaft.data.eqdsk` (detection) | `CocosSpec.psi_per_radian`, `ods_psi_to_wb_per_radian_factor`, `flux_exponent_tier` | COCOS index + probes | runtime | existing | Convention-dependent, so not a unit conversion. The terminal `1/(2π)` default in `ods_psi_to_wb_per_radian_factor` is a documented default, not a detection, and the detected tier is not persisted. |

### Quantity identity

| Concept | Owner | Implementation | Enforcement | Coverage | Gap |
| --- | --- | --- | --- | --- | --- |
| Canonical quantity names, strict aliases | `vaft.plot.taxonomy` | `_QUANTITY_ALIAS_PAIRS`, `_UNALIASED_QUANTITIES`, `resolve_quantity` | structural (`test_quantity_aliases_resolve_deterministically`, `test_every_strict_alias_resolves_to_exactly_one_canonical_concept`) | existing | none |
| Families are not synonyms | `vaft.plot.taxonomy` | `QuantityFamily` (`beta = beta_n, beta_p, beta_t`) | structural (`test_family_membership_is_not_synonymy`) | existing | none |
| Formula outputs to concepts | `vaft.formula._taxonomy` | `Quantity(concept=…)` | structural | existing | Records representation (0-D/1-D/2-D), not unit. |
| Consumed and produced quantities | `vaft._semantics` (shape), `vaft._ontology_graph` (resolution) | `Semantics`, `parse_semantics` | structural: unknown term fails generation | partial | Used by 9 docstrings. A term carries no unit, so `plasma_current` means A in `beta_normal_from_beta_volume` and MA in `greenwald_density`. |
| Same-name-different-meaning (density, li, q, P) | `vaft.formula.boundaries` | `BoundaryQuantity.name`, `same_quantity` | runtime | existing | Distinctions such as `line_average_density` vs `edge_density`, or li(1)/li(3), exist **only here**, and the names never refer to the taxonomy (`normalized_beta` vs `beta_n`, `toroidal_field` vs `b_t`, `edge_safety_factor_95` vs `q95`). **Duplicate/ambiguous ownership.** |
| Density averaging (line, volume, separatrix, edge) | none in the taxonomy | `electron_density` is one subject (`ne`, `n_e`) | none | missing | `process.langmuir` produces a local probe density under the same concept as a line-averaged one. **Missing scientific definition.** |
| Vacuum vs plasma-response field | none | `vacuum` is a *model* subject, `b_t` is unqualified | none | missing | **Missing scientific definition.** |
| ψ vs ψ_N | none | `psi` only | none | missing | **Missing scientific definition.** |
| Unknown terms | `vaft._ontology_graph` | `resolve_term`, `graph.miss`, `snapshot["unresolved"]` | structural (`test_unknown_terms_are_audited_never_canonicalized`) | existing | none |

**Decision: no new Quantity Registry.** The taxonomy already provides canonical identity, strict
aliases, families and an audit of unknown terms. What is missing is a handful of canonical
distinctions plus a link from `BoundaryQuantity` to the taxonomy. Phase B adds those to the existing
vocabulary.

### Coordinates, orientation and COCOS

| Concept | Owner | Implementation | Representation | Enforcement | Coverage | Gap |
| --- | --- | --- | --- | --- | --- | --- |
| COCOS definitions | `vaft.data.cocos` | `CocosSpec` (from `omas.define_cocos`), `expected_sign`, `bp_factor` | frozen dataclass | runtime | existing | none |
| External-code conventions | `vaft.data.cocos` | `CodeConvention`, `register_convention`, `convention_for` | registry: chease 2, flare 3, gacode 2, vfit 1 (`confirmed=False`), imas/omas 11, geqdsk/efit/tes/gpec per file | structural | partial | No `dcon` entry. Evidence is a `confirmed` boolean. |
| Convention evidence for one equilibrium | `vaft.data.equilibrium` | `EquilibriumConvention` (`cocos`, `identified`, `contradicted`, `ambiguous`, `source`, `uncarried`) | dataclass + free-text `source` | runtime | partial | Declared, inferred, contradicted and unresolved are spread across fields and free text. There is no evidence-level vocabulary. |
| Identification from data | `vaft.process.cocos` | `identify_convention`, `identify_flux_exponent`, `identify_flux_exponent_from_q` | narrowing to candidates; abstains | runtime (self-consistency: Ampère, contour q) | existing | none |
| Consistency check | `vaft.validation.cocos` | `validate_cocos` → `ValidationReport` | issue codes | runtime | partial | `j_phi` and `phi_tor` are always unverifiable. |
| Transformation | `vaft.process` (`convert_cocos`), `vaft.omas.general` (`equilibrium_psi_to_weber`, `_transform_cocos_leaves`) | omas `cocos_transform` factors | functions | runtime | partial | `convert_cocos` calls itself "the only COCOS transform", but CHEASE, FLARE, GACODE and VFIT each carry their own path. Records are ad hoc: a CHEASE `chease_cocos_transform.json` manifest, the free-text `convention.source` set by `convert_cocos`, GACODE `provenance["cocos"]`, a TokaMaker sidecar field. There is no shared record shape. **Lost evidence/provenance.** |
| Machine-native polarity (VEST) | `vaft.machine_mapping.conventions` | `DischargeSignContract`, `SignTransformation(confirmed=False)`, `equilibrium_orientation_is_resolved()` | dataclass + boolean | doc + structural | unresolved | **Open: #288** (the VEST sign chain is not verified end to end). A COCOS relabel cannot repair it. **Missing scientific evidence, not a software defect.** |
| External-code boundary | `vaft.code.*` | CHEASE forces signs (`_desired_signs_for_cocos`, `CHEASE_ORIENTATION`), FLARE refuses ambiguity (`FlareEquilibriumScales`) | per adapter | runtime | partial | GPEC/DCON (`copy_gfile_for_gpec`), EFIT and TES never check COCOS. **Incomplete transformation boundary**, a follow-up outside Phase B. |

### Fourier, complex harmonics and phase

The kernels already in the package differ, which is legitimate because each matches the code it
talks to. Until this audit they were recorded only in prose.

| Function | Kernel or synthesis, with normalization | Reconstruction |
| --- | --- | --- |
| `vaft.process.coils_non_axisymmetric.toroidal_mode_decomposition` | `C_n = (1/K) Σ_k v_k e^{-inφ_k}` | `v = 2 Re(C_n e^{+inφ})`; `A cos(nφ+δ)` gives `(A/2) e^{+iδ}` |
| `vaft.process.perturbation.toroidal_phase_audit` (GPEC, FLARE) | `C = 2 mean_φ(B e^{+inφ})` | `B = Re(C e^{-inφ})`; the conjugate of the row above, times two |
| `vaft.formula.stability.helical_harmonic` | synthesis only: `Re[b̂ e^{i(mθ − nφ)}]` | `b̂ = 2 conj(C_n)` |
| `vaft.process.perturbation.composite_drive_at_q` | synthesis only: `Σ Re(z_n e^{inu})`, `u = qθ − ζ`, not doubled | `peak`, `quadrature` and `linear_bound` are gauge invariant; `peak_angle` is not |

| Concept | Owner | Representation | Enforcement | Coverage | Gap |
| --- | --- | --- | --- | --- | --- |
| Kernel sign, normalization, sidedness | the function above | docstring prose | doc | partial | **Definition not machine-readable.** No descriptor can be compared. |
| Toroidal angle orientation | `vaft.machine_mapping.conventions` (`VEST_PORT_CLOCK_TO_PHI_SIGN`), IMAS counter-clockwise | constant + prose | structural | existing | none |
| GPEC helicity rule (real space vs spectral output) | `vaft.machine_mapping.conventions.VEST_GPEC_COIL_DIRECTIONS` | prose | doc | unresolved | The VEST coil-word conflict is open. |
| Phase-convention audit | `vaft.process.perturbation.ToroidalPhaseAudit`, `vaft.code.gpec._phase_audit.audit_coil_field_phase` | dataclass with gauge-invariant `separation_ratio` | independent (Biot–Savart reference) | existing | none |
| Phase verdict | `vaft.validation.perturbation_phase.phase_convention_verdict` | `PhaseConventionVerdict`: `PASS` or `INDETERMINATE`, never `FAIL` | runtime | partial | **Missing integration**: no production caller. |
| Phase origin `φ0 = arg Φ_res − π/2` | `vaft.formula.stability.helical_phase` | prose; measured at helicity −1 only | doc | partial | Recorded, not verified for both helicities. |

### Applicability, validation and provenance

| Concept | Owner | Implementation | Coverage | Gap |
| --- | --- | --- | --- | --- |
| Validation statuses | `vaft.validation.model` | `ValidationStatus`: `PASS`, `WARN`, `FAIL`, `INDETERMINATE`, `NOT_AVAILABLE` | existing | none |
| Named checks | `vaft.validation.registry` | `CheckSpec`, `CHECKS` (22, all equilibrium) | partial | No transformation check exists. |
| Credibility axes | `vaft.validation.credibility` | `AXES`, `Evidence`, `EVIDENCE_ROLES`, `compose` | existing | **The `transformation` (T) axis has no producer.** No category or check maps to it. |
| Approximation contracts | `vaft.validation.applicability`, `vaft.validation.orderings` | `ApproximationContract`, `evaluate_contract`, `as_evidence`, 15 `CONTRACTS` | existing | none for this initiative |
| Adapter-to-contract map | `vaft.validation.model_contracts` | `ADAPTER_MODES`, `ordering_provenance`, `safe_ordering_provenance` | partial | Recorded by EFIT, GPEC, PENTRC and CGYRO. Not by CHEASE. |
| Data validity | `vaft.validation.validity`, `vaft.validation.imas` | `ValidityRecord`, `write_validity` (own assessment only) | existing | none |
| Digest lineage | `vaft.database.provenance` | `ChainLink`, `verify_chain`, `NOT_RECORDED` | existing | none |
| Scientific transformation history | none | GACODE per-field `{kind, source/reason}` and `provenance["cocos"]`; free-form notes elsewhere | partial | Per-field shapes exist in three places (GACODE inputs `{kind, source/reason}`, `pr08_mhd_state` `source_unit`/`transformation`, `gacode.output_audit` `native_units`/`transformation`) but share no schema. **Lost evidence/provenance**: no common record and no link from `Evidence` to what was transformed. |

## Ownership

An owner is named separately for the **definition**, the **transformation** and the **verdict**,
because these are different responsibilities.

| Concern | Definition | Transformation | Verdict |
| --- | --- | --- | --- |
| Canonical quantity identity | `vaft.plot.taxonomy` (extended, not duplicated) | — | `vaft._ontology_graph` audit (structural) |
| Boundary-specific identity | `vaft.formula.boundaries`, linked to the taxonomy | — | `same_quantity` |
| Physical dimensions | `vaft.formula.dimensional` | — | — |
| Unit scale factors | `vaft.formula` (new `units` module, Phase B) | explicit conversion call | — |
| COCOS | `vaft.data.cocos` | `vaft.process` `convert_cocos`, `vaft.omas.general` | `vaft.validation.cocos` |
| Machine polarity | `vaft.machine_mapping.conventions` | `SignTransformation` (unconfirmed) | none yet (**Open: #288**) |
| Fourier conventions | `vaft.data` (new `fourier` descriptor, Phase B), next to `vaft.data.cocos` | explicit harmonic conversion | `vaft.validation.perturbation_phase` |
| Representation and compatibility | `vaft.data` (new `representation` module, Phase B) | references the providers above | — (compatibility is not a verdict) |
| Applicability | `vaft.validation.orderings` | — | `vaft.validation.applicability` |
| Evidence and credibility | `vaft.validation.credibility` | — | `compose` per axis, no total |
| Generated views | `vaft.formula.catalog`, `vaft.process.catalog`, `vaft._ontology_graph` | — | — |

**Out of scope**, with owners recorded: the unresolved VEST polarity (Open: #288); collisionality and
gyroradius definition consolidation (Open: #353); expansion of ordering and applicability contracts
(Open: #1627, #1628);
consolidating the COCOS transform paths, and COCOS checks at the GPEC, EFIT and TES boundaries
(follow-ups to be filed after Phase B).

## Five traces

Each step is marked **reliable** (machine-checked), **implicit** (correct, but resting on caller
knowledge or prose), **verifiable** (an independent check exists and can be run), or **gap** (Phase B
work).

### A. COCOS through an equilibrium chain

1. A g-file or ODS enters. `equilibrium_psi_to_weber` reads the declared COCOS, or probes Wb against
   Wb/rad through `flux_exponent_tier` (slope, Ampère, contour q), and **refuses** when no tier
   decides. **Verifiable.** The separate reader-side `ods_psi_to_wb_per_radian_factor` runs the same
   probes but ends in a stated default of 1/(2π), which is **implicit**.
2. `_resolve_convention` builds `EquilibriumConvention`: identification always runs and the
   declaration wins, with `contradicted` set when they disagree. **Reliable**, but the evidence level
   lives in booleans and the `source` string. **Gap**: no shared evidence vocabulary.
3. `convert_cocos` applies omas factors to ψ, F, q, p′, FF′, I_p and B0, and refuses an ambiguous
   source. **Reliable.** What it keeps is a free-text `source` ("converted from COCOS N") plus the
   carried `identified` candidates, not a structured record. **Gap** for Phase C.
4. CHEASE input forces signs to COCOS 2 with `CHEASE_ORIENTATION`. **Reliable** (byte parity test).
5. The GPEC g-file copy (`copy_gfile_for_gpec`) passes the file through without a COCOS check.
   **Gap**, filed as a follow-up and kept out of this initiative.
6. VEST machine polarity (`IP_SIGN_VEST_TO_IMAS.confirmed=False`) never enters steps 1–5.
   **Unresolved (Open: #288).** Correct COCOS handling of the stored data does not confirm it.

### B. Greenwald density in engineering units

1. `greenwald_density(I_p, a)` takes `I_p [MA]` and returns `[1e19 m^-3]`. Its Convention section
   notes that the literature quotes 10²⁰. **Implicit**: the unit is a docstring tag.
2. Callers convert by hand: `_synthetic_kinetic_profiles` writes `1e19 * greenwald_density(abs(ip)/1e6, …)`,
   and `plot.history` sniffs column names (`Ip_MA`, `ne_19m3`). **Implicit.**
3. `greenwald_fraction(n_e, n_G)` expects a line-averaged density in the same unit, stated only in
   prose. **Implicit.**
4. Nothing can detect a caller passing amperes. The result would be 10⁶ too large, finite and
   silent. **Gap**: no parsed unit, no opt-in assessment.
5. Normalized beta follows the same pattern three times: `beta_N_from_beta_a_B0_Ip` (% and MA),
   `beta_normal_from_beta_tor` (fraction and A, converted inside), `beta_normal_from_beta_volume`
   (Menard ⟨β_N⟩, a different beta in the same unit). **Gap**: the taxonomy has a single `beta_n`.

### C. Complex toroidal response

1. Sector currents `I_k = A cos(nφ_k + δ)` come from `phased_sector_currents`. **Reliable.**
2. `toroidal_mode_decomposition` gives `C_n = (A/2) e^{+iδ}` with kernel `e^{-inφ}` and `1/K`.
   **Reliable** (`test_cosine_pattern_gives_half_amplitude_and_positive_phase`), but the convention is
   prose. **Gap.**
3. GPEC writes fields under `Re(C e^{-inφ})`. `toroidal_phase_audit` compares the stored coefficient
   and its conjugate against a Biot–Savart reference. **Verifiable**, with a gauge-invariant
   separation ratio.
4. `phase_convention_verdict` returns `PASS` or `INDETERMINATE`. **Reliable** when called, but nothing
   calls it. **Gap** for Phase C.
5. The VEST coil-direction words are an open conflict (`VEST_GPEC_COIL_DIRECTIONS`). **Unresolved.**

### D. Model applicability

1. A state (dict, DataFrame row or object) holds ordering quantities named as in
   `ORDERING_QUANTITIES`. **Reliable.**
2. `evaluate_contract(CONTRACTS["gyrokinetic_delta_f"], state)` returns per-assumption margins
   (`log10` of threshold over value) and `SUPPORTED`, `OUTSIDE`, `UNASSESSED` or `NOT_APPLICABLE`.
   A missing input gives `UNASSESSED`, never `SUPPORTED`. **Reliable.**
3. `as_evidence` places the result on the A axis. `compose` keeps it apart from N. **Reliable.**
4. Adapters record `ordering_provenance` in run provenance as advisory data. This is explicit and
   never automatic. **Reliable.** No gap for this initiative beyond attaching the evaluation to a
   result context (Phase C).

### E. Two densities with the same unit

1. `formula.boundaries` separates `line_average_density` from `edge_density`, whose definition says
   it is "not the line-averaged or separatrix density". **Reliable** inside boundaries. No separatrix
   density is defined anywhere as a quantity of its own.
2. The taxonomy has only the subject `electron_density` (`ne`, `n_e`); it is not a canonical quantity. Both densities would resolve to that subject.
   **Gap.**
3. The ontology never ingests boundaries, so the distinction is invisible there. **Gap** for Phase D.
4. Nothing anywhere treats these as interchangeable, but nothing would stop it. **Gap**: no
   compatibility check that refuses it.

## Phase B specification

Each item names its gap, owner, behaviour, tests and constraints. Everything is opt-in: no formula
signature, return type or import cost changes.

1. **Unit scale** (`vaft.formula.units`, new).
   - Gap: unparsed unit tags and about 15 hand-written conversions.
   - Behaviour: `parse_unit` covers only the tags the audit found in use (A, kA, MA, W, kW, MW, T, G,
     J, eV, keV, m, m⁻³, 10¹⁹ m⁻³, 10²⁰ m⁻³, s, ms, Wb, `-`, %). `conversion_factor` refuses a
     dimension mismatch and refuses Wb/rad, pointing to the COCOS provider instead. An unknown tag
     raises; nothing is guessed.
   - Tests: the scale factors; refusals; an unknown tag; no numpy-free import cost.
2. **Identity distinctions** (`vaft.plot.taxonomy`, `vaft.formula.boundaries`).
   - Gap: density averaging, ψ vs ψ_N, vacuum vs response field; two disjoint identity vocabularies.
   - Behaviour: add canonical quantities for those distinctions without merging aliases. Add an
     optional `BoundaryQuantity.concept` resolved through `resolve_quantity`. `same_quantity`
     keeps its meaning. A separate predicate reports when the same quantity is held in two
     convertible units.
   - Tests: no alias collisions; every boundary concept resolves; the 10¹⁹ and 10²⁰ line densities are
     convertible but `same_quantity` is still false.
3. **Fourier descriptor** (`vaft.data.fourier`, new).
   - Gap: kernels are prose only.
   - Behaviour: a frozen `FourierConvention` (kernel sign, normalization, sidedness, angle orientation,
     phase origin) with named constants for the four kernels above, each citing its function. An
     explicit coefficient conversion is allowed only for coefficients of a real field with every
     field declared. A differing phase origin or orientation is never bridged.
   - Tests: the sector-current → decomposition → GPEC-convention round trip reproduces the
     `toroidal_phase_audit` coefficient and the real-space field; missing metadata raises.
4. **Representation and compatibility** (`vaft.data.representation`, new).
   - Gap: no common way to ask whether two representations can be combined.
   - Behaviour: a `Representation` (quantity, unit, optional COCOS and Fourier convention, evidence
     level: declared, inferred, verified or unresolved) and `assess_compatibility` with outcomes
     `COMPATIBLE`, `REQUIRES_TRANSFORM`, `INCOMPATIBLE` and `UNRESOLVED`, structured reasons, and a
     reference to the transformation provider. These outcomes are not validation statuses.
     Assessment never touches arrays; executing a transform is a separate call.
     `EquilibriumConvention` maps to a representation, with `contradicted` or `ambiguous` giving
     `UNRESOLVED`.
   - Tests: Cases A, B, C and E; a family match is `INCOMPATIBLE`; missing units or COCOS give
     `UNRESOLVED`; inputs are not mutated.
5. **Opt-in boundary** (`vaft.formula.catalog`).
   - Behaviour: derive a parameter's `Representation` from its existing unit tag and `Semantics`
     term, so a caller can assess before calling `greenwald_density` or a β_N function.
   - Tests: the Greenwald `I_p` parameter is MA; a value in A is `REQUIRES_TRANSFORM` with factor
     10⁻⁶; a value with no unit is `UNRESOLVED`.

Phase C (Open: #2165) then adds a transformation record, the first producers of `transformation` (T)
evidence, and an opt-in result context. Phase D (Open: #2167) exposes all of it through the generated
catalogs and ontology.

## Related reference

- [Scientific architecture]({{ '/reference/scientific-architecture/' | relative_url }})
- [Credibility and applicability]({{ '/reference/credibility-applicability/' | relative_url }})
- [Asymptotic orderings]({{ '/reference/asymptotic-orderings/' | relative_url }})
- [Scientific ontology]({{ '/reference/ontology/' | relative_url }})
