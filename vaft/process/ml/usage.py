"""Opt-in declared data roles and observed entrypoint uses (#2047).

No project/machine policy is built in. The owning workflow supplies roles and
permissions; this records declared history, not proof of arbitrary prior fits.
"""
from __future__ import annotations

from dataclasses import replace
import hashlib
import json

from ._types import ModelContractError, Split, SplitSpec, canonical_json

__all__ = ["import_data_roles", "record_data_use"]
_OPERATIONS = frozenset({"train", "preprocess", "calibrate", "select", "evaluate", "infer"})


def _hash(value):
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


def _policy(dataset):
    policy = dataset.spec.metadata.get("data_use_policy")
    if policy is None:
        return None
    # JSON copy: callers cannot mutate a retained reference while checking it.
    policy = json.loads(canonical_json(policy))
    if _hash(policy) != dataset.spec.metadata.get("data_use_policy_sha256"):
        raise ModelContractError("data-use policy digest does not match declaration")
    if policy.get("schema_version") != 1:
        raise ModelContractError("unsupported data-use policy schema")
    roles, permissions = policy["group_roles"], policy["allowed_operations"]
    if not roles or any(not isinstance(k, str) or not isinstance(v, str) or not v for k, v in roles.items()):
        raise ModelContractError("data roles need named groups and nonempty roles")
    if set(roles.values()) - set(permissions):
        raise ModelContractError("every data role needs explicit permissions")
    for operations in permissions.values():
        if not isinstance(operations, list) or set(operations) - _OPERATIONS:
            raise ModelContractError("unknown data-use operation")
    if set(roles.values()) - set(policy["role_to_partition"]):
        raise ModelContractError("every data role needs a partition mapping")
    if set(dataset.group_labels) - set(roles):
        raise ModelContractError("dataset contains undeclared groups")
    return policy


def import_data_roles(dataset, assignments, allowed_operations, *, role_to_partition=None, provenance=None):
    """Import exact group roles without recomputing a scientific split.

    Parameters
    ----------
    dataset : DatasetArtifact or StructuredDatasetArtifact
        Existing stable, fingerprinted samples [-].
    assignments : mapping of str to str
        Actual independent group-to-role assignments, including absent groups [-].
    allowed_operations : mapping of str to sequence of str
        Explicit role permissions: train, preprocess, calibrate, select,
        evaluate or infer [-].
    role_to_partition : mapping or None, optional
        Explicit mapping to execution partitions; None preserves role names [-].
    provenance : mapping or None, optional
        Split/protocol revisions, origins, label methods and dependencies [-].

    Returns
    -------
    tuple
        Re-fingerprinted dataset and exact Split, without random reassignment [-].

    Processing steps
    ----------------
    1. Copy the declared assignments, permissions, mapping and provenance.
    2. Validate role coverage and record the full assignment digest.
    3. Recompute dataset identity and construct a fixed split for its groups.

    Convention
    ----------
    Roles are owner-defined; neither machine restrictions nor salts are inferred.

    Applicability
    -------------
    Machine-independent.

    Limitations
    -----------
    Re-fingerprinting streams structured samples. Sources must remain stable.
    Supplied historical provenance is a declaration, not retrospective proof.

    Provenance
    ----------
    .. [vaft2047] VEST-Tokamak/vaft#2047, exact assignment and usage provenance.
    """
    roles = {str(k): str(v) for k, v in assignments.items()}
    mapping = dict(role_to_partition) if role_to_partition is not None else {v: v for v in roles.values()}
    policy = {"schema_version": 1, "group_roles": roles,
        "allowed_operations": {str(k): sorted(set(v)) for k, v in allowed_operations.items()},
        "role_to_partition": mapping, "assignment_sha256": _hash(roles),
        "provenance": dict(provenance or {}), "history_basis": "owner_declaration"}
    spec = replace(dataset.spec, metadata={**dataset.spec.metadata, "data_use_policy": policy, "data_use_policy_sha256": _hash(policy)})
    data = replace(dataset, spec=spec)
    _policy(data)
    from .structured import StructuredDatasetArtifact, build_structured_dataset
    if isinstance(data, StructuredDatasetArtifact):
        data = build_structured_dataset(data.source, data.groups, input_specs=data.input_specs,
            target_specs=data.target_specs, spec=spec, sample_meta=data.sample_meta, available=data.available)
    else:
        from .dataset import dataset_fingerprint
        data = replace(data, fingerprint=dataset_fingerprint(data))
    fixed = {g: mapping[roles[g]] for g in sorted(set(data.group_labels))}
    if not fixed or any(not isinstance(v, str) or not v for v in fixed.values()):
        raise ModelContractError("execution partitions must be nonempty strings")
    names = sorted(set(fixed.values()))
    split = Split(spec=SplitSpec(fractions={p: 1 / len(names) for p in names}, fixed=fixed), assignment=fixed)
    return data, split


def record_data_use(dataset, split, operation, *, groups=None):
    """Check a declared use and return its provenance receipt.

    Parameters
    ----------
    dataset : DatasetArtifact or StructuredDatasetArtifact
        Dataset with an optional data_use_policy [-].
    split : Split
        Realized execution assignment to verify against the declaration [-].
    operation : str
        Requested train, preprocess, calibrate, select, evaluate or infer [-].
    groups : iterable of str or None, optional
        Groups actually consumed; None selects every dataset group [-].

    Returns
    -------
    dict or None
        Usage receipt; None means no policy was supplied, not validated history [-].

    Processing steps
    ----------------
    1. Verify the dataset's execution assignment matches its declared mapping.
    2. Check explicit permissions for every selected group.
    3. Return dataset, policy and assignment identities with observed group roles.

    Convention
    ----------
    Calling this reports a requested entrypoint use; it cannot inspect external
    training history or intercept scientific operations outside this API.

    Applicability
    -------------
    Machine-independent.

    Provenance
    ----------
    .. [vaft2047] VEST-Tokamak/vaft#2047, opt-in operation-role checks.
    """
    policy = _policy(dataset)
    if policy is None:
        return None
    if operation not in _OPERATIONS:
        raise ModelContractError("unknown data-use operation")
    roles = policy["group_roles"]
    if _hash(roles) != policy["assignment_sha256"]:
        raise ModelContractError("data assignment digest does not match declaration")
    for g in set(dataset.group_labels):
        if split.assignment.get(g) != policy["role_to_partition"][roles[g]]:
            raise ModelContractError("split differs from declared data roles")
    selected = sorted(set(dataset.group_labels if groups is None else map(str, groups)))
    if set(selected) - set(dataset.group_labels):
        raise ModelContractError("requested use contains groups outside the dataset")
    for g in selected:
        if operation not in policy["allowed_operations"][roles[g]]:
            raise ModelContractError(f"{operation} is forbidden for role {roles[g]!r}, group {g!r}")
    return {"operation": operation, "groups": selected, "group_roles": {g: roles[g] for g in selected},
        "dataset_fingerprint": dataset.fingerprint, "assignment_sha256": policy["assignment_sha256"],
        "policy_sha256": _hash(policy), "history_basis": "declared_use_at_entrypoint"}


def _fit_uses(dataset, split, config, train, validation, *, iterative=True, supervised=True, validation_used=True):
    import numpy as np
    groups = lambda indices: sorted(set(dataset.group_labels[np.asarray(indices, dtype=int)]))
    records = [record_data_use(dataset, split, "train", groups=groups(train))]
    normalize = config.normalize_inputs or (config.normalize_targets and supervised)
    if hasattr(dataset, "input_specs"):
        normalize = (config.normalize_inputs and any(v.normalize and np.dtype(v.dtype).kind == "f" for v in dataset.input_specs.values())) or (
            config.normalize_targets and any(v.normalize and np.dtype(v.dtype).kind == "f" for v in dataset.target_specs.values()))
    if normalize:
        records.append(record_data_use(dataset, split, "preprocess", groups=groups(train)))
    if validation_used and len(validation):
        records.append(record_data_use(dataset, split, "select" if iterative else "evaluate", groups=groups(validation)))
    fitted = dataset.spec.metadata.get("fitted_transform_groups", ())
    if fitted:
        receipt = record_data_use(dataset, split, "preprocess", groups=fitted)
        if receipt is not None:
            receipt["history_basis"] = "external_fit_owner_declaration"
            records.append(receipt)
    return [r for r in records if r is not None]


def _inference_use(dataset):
    policy = _policy(dataset)
    if policy is None:
        return None
    fixed = {g: policy["role_to_partition"][policy["group_roles"][g]] for g in set(dataset.group_labels)}
    names = sorted(set(fixed.values()))
    split = Split(spec=SplitSpec(fractions={p: 1 / len(names) for p in names}) if names else SplitSpec(), assignment=fixed)
    return record_data_use(dataset, split, "infer", groups=set(dataset.group_labels[dataset.available_mask]))
