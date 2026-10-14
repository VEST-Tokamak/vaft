# #2047: declared ML data uses and scientific evidence

Core contract: `ml.import_data_roles(dataset, assignments, permissions,
role_to_partition=..., provenance=...)` returns a fingerprinted dataset and a
fixed split. All supplied assignments, including groups absent from the current
source, are retained and hashed. No second hashing algorithm regenerates shots.
Execution fractions are placeholders for fixed memberships, not a claim about
scientific sampling fractions (record those in provenance).

The owner defines role permissions for `train`, `preprocess`, `calibrate`,
`select`, `evaluate`, `infer`. Training records fit groups, enabled normalization
and validation-loss checkpoint selection for iterative backends. Closed-form
regression/reconstruction backends record validation evaluation; score backends
that ignore validation create no use receipt for it. Pre-existing fitted transformations
are checked but explicitly remain an external-owner declaration. Evaluation,
calibration and dataset inference return usage receipts. Model selection or
preprocessing outside these entrypoints must call `record_data_use` explicitly;
this cannot inspect arbitrary Python or retrospectively prove training history.
Legacy datasets without a policy retain their behavior. Raw-array inference has
no dataset-role contract. Policies are JSON copies and bound by a digest;
changing roles/permissions requires a fresh import and fingerprint.

`DatasetSpec.metadata.data_use_policy.provenance` is owner-supplied evidence:
record split version/salt/realized assignment, source and target origin,
label-generation method/revision, units/coordinates, preprocessing fit groups
and hashes, geometry/basis dependencies where relevant, and protocol/scorer
versions. Unknown evidence stays unknown. PEP610 Git installs can provide the
installed VAFT SHA; ordinary wheels remain unknown, never the consumer repo SHA.

No MAST policy, NeurIPS scorer or geometry enters VAFT. Metrics remain numerical
results; scientific evidence and criteria belong to existing validation APIs.
The next bridge is bounded by explicit model/dataset/split/protocol/criterion
identities and the existing credibility axes; it does not promote models.

## #669 acceptance audit (2026-10-15)

| Requirement | Status and evidence |
| --- | --- |
| Dataset/result contracts, independent-unit split, backend neutrality | Implemented: `_types.py`, `dataset.py`, structured contract; existing contract/mask tests, #2027/#2039. |
| Optional Torch, minimal architecture, deterministic CPU reference | Implemented: backend registry/shared engine; Torch/structured tests and bounded GPU #2016 receipts. TensorFlow SXR is a separate migration/domain workstream. |
| Train-only normalization, artifact/config/data provenance | Implemented computationally; label/use history remains partial until this opt-in contract and domain declarations are supplied. |
| Structured inference and NeurIPS physical adapter | Implemented and independently qualified for six families (#223/#227); fixed target shapes and research-owned consistency/subdivision boundaries. This does not complete IRE/SXR scientific validation. |
| Artifact hashes, model manifests, bundles, exact resolver and aliases | Implemented: `artifact.py`, `bundle.py`, `resolver.py` and regression tests. Consume registry identities; no duplicate registry. |
| First validated IRE artifact published in vaft-nn | Unimplemented: registry `models/` currently has only README/.gitkeep; #973 remains open. Resolver tests do not establish production publication. |
| Scientific ownership, applicability vs training support, separate uncertainties | Architectural direction established by #669 refresh and existing validation/credibility/applicability. ML-specific evidence/domain implementations partial. |
| Target origin, parent model, domain restrictions, calibration/scientific report binding | Partial: existing free metadata/model cards and research declarations; strict reusable usage/evidence binding is #2047. |
| Numerical calibration | Scalar threshold calculation implemented. Structured scientific calibration is explicitly unqualified; no new promise in #2047. |
| Physically meaningful perturbations, robustness, OOD/abstention, uncertainty and temporal causality | Domain-specific/unimplemented in this generic integration; not satisfied by smoke tests or a finite forward result. |
| Retraining candidates and manual promotion/publication | Broad lifecycle work remains open. Do not infer production acceptance from successful training/CI. Automatic promotion and general retraining are deferred here. |
| Synthetic-to-experimental and cross-machine scientific generalization | Role/origin declarations exist; numerical adapter compatibility does not establish scientific holdout accuracy. MAST remains label-free per research policy. |

#669 stays open. #973/#991/#1001/#1882 remain deferred for this project. A full
immutable deployment example linking research preprocessing/geometry + network
+ evidence through canonical vaft-nn is subsequent work, not a challenge release.

## Scientific evidence bridge

`vaft.validation.ml.MetricCriterion` requires explicit bounds, axis, units,
provider, source and criterion revision. Declare it before qualification; no
bounds are inferred from scores. `build_ml_report` binds supplied numerical
EvaluationResult to model identity **and the contract hash captured at evaluation
time**, dataset, realized split, population, protocol and criteria. The contract
includes shapes/names, weights, normalization, configuration, calibration and
declared/observed history, covering in-memory models whose state-only identity
would miss altered normalization.

`ml_report_evidence(..., expected_report_sha256=trusted_reference)` verifies the
sidecar and original metric values before creating existing credibility Evidence.
Keep the expected hash outside the untrusted report. It provides integrity, not
a signature/authentication. A report's own editable checksum is not its trust
anchor. Store the reference in a trusted run/publication record; mutating a
bound model manifest to insert a report hash creates a circular dependency.

Fitted/calibration/selection populations route to inference evidence. Independent
validation needs an opt-in role receipt, recorded training uses, disjoint groups
and a role reserved from train/preprocess/calibrate/select. Prior external fits
remain owner declarations; this is not proof of unknown external history. Missing
metrics or evaluation populations are not_available; produced invalid metrics
are indeterminate. Evidence stays per-axis, with protocol/population scope;
there is no overall acceptance score, implicit physical applicability or
production promotion. Domain providers still compute physics and official scores.
