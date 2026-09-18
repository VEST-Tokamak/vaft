"""Dataset construction, fingerprinting, group splitting and windowing (#669).

The split is defined over the *independent scientific unit* -- a shot, a
simulation case -- and is carried as a group-to-partition assignment, so it
applies unchanged to every windowed or augmented dataset derived afterwards.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Mapping

import numpy as np

from ._types import (
    DatasetArtifact,
    DatasetSpec,
    FeatureDataset,
    FeatureSpec,
    ModelContractError,
    Split,
    SplitSpec,
    canonical_json,
)

__all__ = [
    "build_dataset",
    "dataset_fingerprint",
    "load_dataset",
    "save_dataset",
    "split_groups",
    "window_dataset",
]


def _hash_array(digest, label: str, array: np.ndarray) -> None:
    array = np.ascontiguousarray(array)
    if array.dtype == object:
        raise ModelContractError(f"{label} has dtype object; fingerprints need a fixed-width dtype")
    digest.update(f"{label}|{array.dtype.str}|{array.shape}|".encode())
    digest.update(array.tobytes())


def dataset_fingerprint(dataset):
    """SHA-256 content fingerprint of a dataset.

    Two datasets share a fingerprint only when their spec, feature schema,
    inputs, targets, groups and per-sample metadata are identical in value,
    dtype and shape.  A model records this fingerprint as part of its identity,
    so "which data trained it" is answerable from the artifact alone.

    Parameters
    ----------
    dataset : DatasetArtifact
        The dataset to fingerprint; its own ``fingerprint`` field is ignored [-].

    Returns
    -------
    str
        64-character lowercase hex digest [-].

    Processing steps
    ----------------
    1. Hash the canonical JSON of the spec (and the feature spec, if any).
    2. Hash inputs, targets, string group labels, the availability mask (when
       one is set) and each ``sample_meta`` array in sorted key order, each
       prefixed by its name, dtype and shape.

    Applicability
    -------------
    Machine-independent.
    """
    digest = hashlib.sha256()
    digest.update(canonical_json(dataset.spec.to_dict()).encode())
    if isinstance(dataset, FeatureDataset):
        digest.update(canonical_json(dataset.feature_spec.to_dict()).encode())
    _hash_array(digest, "inputs", np.asarray(dataset.inputs))
    if dataset.targets is not None:
        _hash_array(digest, "targets", np.asarray(dataset.targets))
    _hash_array(digest, "groups", dataset.group_labels.astype(str))
    if dataset.available is not None:
        _hash_array(digest, "available", np.asarray(dataset.available))
    for key in sorted(dataset.sample_meta):
        _hash_array(digest, f"meta:{key}", np.asarray(dataset.sample_meta[key]))
    return digest.hexdigest()


def build_dataset(inputs, groups, *, spec, targets=None, sample_meta=None, feature_spec=None, available=None):
    """Assemble a fingerprinted dataset from arrays.

    The generic entry point for every ML task: an engineered-feature table
    (pass ``feature_spec``), a waveform or image tensor, or paired
    input-target supervision (pass ``targets``).

    Parameters
    ----------
    inputs : array_like
        Samples along the first axis, shape ``(n, *input_shape)`` [any].
    groups : array_like
        Independent unit of each sample (shot number, simulation case), shape
        ``(n,)`` [-].
    spec : DatasetSpec
        Name, task, group key and column names of the dataset [-].
    targets : array_like or None, optional
        Paired targets, shape ``(n, *target_shape)``; ``None`` for an
        unsupervised dataset [any].
    sample_meta : mapping of str to array_like or None, optional
        Per-sample arrays that travel with every subset, such as window times [any].
    feature_spec : FeatureSpec or None, optional
        Column schema of a 2-D engineered-feature table; returns a
        :class:`FeatureDataset` when given [-].
    available : array_like of bool or None, optional
        Per-sample availability: ``False`` where the inputs do not exist (a
        diagnostic absent for that shot).  Such samples may hold non-finite
        inputs; they are stored as zeros and excluded from training,
        calibration and evaluation, and :func:`predict` reports them as
        unavailable.  ``None`` means every sample is available [-].

    Returns
    -------
    DatasetArtifact
        The dataset with its ``fingerprint`` filled in [-].

    Raises
    ------
    ModelContractError
        On mismatched lengths, a non-finite input in an available sample, or a
        feature table whose width differs from ``feature_spec``.

    Applicability
    -------------
    Machine-independent.

    Limitations
    -----------
    Available inputs must be finite: imputation is a scientific choice
    (train-partition medians, a flag, a drop) that belongs to the caller.
    Unavailability is per sample, not per feature: a two-stage model whose
    second stage lacks a diagnostic builds that stage's dataset with its own
    ``available``.
    """
    if not isinstance(spec, DatasetSpec):
        raise ModelContractError("spec must be a DatasetSpec")
    x = np.asarray(inputs)
    if x.ndim < 2:
        raise ModelContractError(f"inputs need a sample axis and at least one feature axis; got {x.shape}")
    mask = None
    if available is not None:
        mask = np.asarray(available)
        if mask.dtype != bool or mask.shape != (len(x),):
            raise ModelContractError(f"available must be a boolean array of shape ({len(x)},)")
        x = x.astype(np.result_type(x.dtype, np.float64), copy=True)
        x[~mask] = 0.0
    if not np.all(np.isfinite(x)):
        raise ModelContractError(
            "inputs contain non-finite values in available samples; impute or drop them, "
            "or mark the samples unavailable"
        )
    y = None if targets is None else np.asarray(targets)
    if y is not None and y.ndim == 1:
        y = y[:, None]
    g = np.asarray(groups)
    if g.ndim != 1:
        raise ModelContractError(f"groups must be one-dimensional; got {g.shape}")
    meta = {str(k): np.asarray(v) for k, v in dict(sample_meta or {}).items()}
    fields = dict(spec=spec, inputs=x, groups=g, targets=y, sample_meta=meta, available=mask)
    if feature_spec is not None:
        if not isinstance(feature_spec, FeatureSpec):
            raise ModelContractError("feature_spec must be a FeatureSpec")
        dataset = FeatureDataset(feature_spec=feature_spec, **fields)
    else:
        dataset = DatasetArtifact(**fields)
    return dataset.replace(fingerprint=dataset_fingerprint(dataset))


def split_groups(groups, spec=None):
    """Assign independent groups to train/validation/test partitions.

    Splits the *groups* -- never the samples -- so correlated samples of one
    shot or one simulation case can never sit on both sides of a holdout.  Do
    this on the record-level data, before any windowing or augmentation; the
    returned :class:`Split` then applies unchanged to everything derived.

    Parameters
    ----------
    groups : array_like or DatasetArtifact
        Group label of every sample, or a dataset whose groups are used [-].
    spec : SplitSpec or None, optional
        Partition fractions (of groups), seed and pinned groups; ``None`` uses
        70/15/15 with seed 0 [-].

    Returns
    -------
    Split
        Group-to-partition assignment [-].

    Raises
    ------
    ModelContractError
        When a pinned group is absent, or when there are fewer groups than
        partitions with a positive fraction.

    Processing steps
    ----------------
    1. Take the unique groups as strings, sorted, so the result depends on the
       set of groups and the seed only -- not on sample order.
    2. Assign the groups ``spec.fixed`` pins.
    3. With ``method="permutation"``, permute the remaining groups with
       ``numpy.random.default_rng(seed)`` and cut the permutation at the
       cumulative fractions, rounding so every partition with a positive
       fraction receives at least one group.  With ``method="hash"``, map
       each group to ``u = int(sha256("seed:group")) / 2**256`` in ``[0, 1)``
       and place it in the partition whose cumulative-fraction interval holds
       ``u``; no partition is guaranteed a group.

    Defaults
    --------
    The 70/15/15 fractions are numerical convenience, the same shares the VEST
    IRE prototype used [ire].

    Applicability
    -------------
    Machine-independent.

    Provenance
    ----------
    .. [kaufman2012] S. Kaufman, S. Rosset, C. Perlich, O. Stitelman, "Leakage
       in data mining: formulation, detection, and avoidance", ACM TKDD 6, 15
       (2012).
    .. [ire] VEST IRE two-stage anomaly-detection notebook (v4), shot-level split;
       reviewed in issue #669 section 15A.
    """
    spec = spec or SplitSpec()
    labels = groups.group_labels if isinstance(groups, DatasetArtifact) else np.asarray([str(g) for g in np.asarray(groups)])
    unique = sorted(set(labels.tolist()))
    missing = sorted(set(spec.fixed) - set(unique))
    if missing:
        raise ModelContractError(f"SplitSpec.fixed pins groups not present: {missing}")
    assignment = dict(spec.fixed)
    free = [g for g in unique if g not in assignment]
    if spec.method == "hash":
        edges = np.cumsum(list(spec.fractions.values()))
        names = list(spec.fractions)
        for g in free:
            u = int.from_bytes(hashlib.sha256(f"{spec.seed}:{g}".encode()).digest(), "big") / 2.0**256
            assignment[g] = names[min(int(np.searchsorted(edges, u, side="right")), len(names) - 1)]
        return Split(spec=spec, assignment=assignment)
    order = np.random.default_rng(spec.seed).permutation(len(free))
    free = [free[i] for i in order]
    positive = [p for p, f in spec.fractions.items() if f > 0]
    already = {p: sum(1 for v in assignment.values() if v == p) for p in spec.fractions}
    need = [p for p in positive if already[p] == 0]
    if len(free) < len(need):
        raise ModelContractError(
            f"{len(unique)} groups cannot fill {len(positive)} partitions with a positive fraction"
        )
    total = len(unique)
    target = {p: spec.fractions[p] * total for p in spec.fractions}
    counts = {p: max(0, int(round(target[p])) - already[p]) for p in spec.fractions}
    for p in need:
        counts[p] = max(counts[p], 1)
    # Reconcile rounding: take from / give to the partitions furthest from their share.
    while sum(counts.values()) > len(free):
        p = max((q for q in counts if counts[q] > (1 if q in need else 0)),
                key=lambda q: counts[q] + already[q] - target[q])
        counts[p] -= 1
    while sum(counts.values()) < len(free):
        p = min(positive, key=lambda q: counts[q] + already[q] - target[q])
        counts[p] += 1
    cursor = 0
    for p in spec.fractions:
        for g in free[cursor:cursor + counts[p]]:
            assignment[g] = p
        cursor += counts[p]
    return Split(spec=spec, assignment=assignment)


def _runs(labels: np.ndarray) -> list[tuple[int, int]]:
    """Maximal contiguous runs of one group label, as ``[start, stop)``."""
    if len(labels) == 0:
        return []
    change = np.flatnonzero(labels[1:] != labels[:-1]) + 1
    starts = np.r_[0, change]
    stops = np.r_[change, len(labels)]
    return list(zip(starts.tolist(), stops.tolist()))


def window_dataset(dataset, window, step, *, target_mode=None):
    """Cut sliding windows that never cross a group boundary.

    Turns a record-level dataset -- samples in time order, one row per time
    point -- into a window dataset for a sequence model or a feature extractor.

    Parameters
    ----------
    dataset : DatasetArtifact
        Record-level samples in time order within each group [-].
    window : int
        Samples per window [-].
    step : int
        Samples between consecutive window starts [-].
    target_mode : {"window", "last", "none"} or None, optional
        What each window's target is: the target rows over the window, the
        target at its last sample, or none; ``None`` is ``"window"`` for a
        dataset with targets and ``"none"`` otherwise [-].

    Returns
    -------
    DatasetArtifact
        Inputs of shape ``(n_windows, window, *input_shape)``; groups repeated
        per window; ``sample_meta`` carries each window's ``window_start`` and
        ``window_stop`` source indices and, for every per-sample array of the
        source, its value at the window's first sample [-].

    Raises
    ------
    ModelContractError
        When ``window`` or ``step`` is not a positive integer, or a target mode
        asks for targets the dataset does not have.

    Processing steps
    ----------------
    1. Find the maximal contiguous runs of one group label.
    2. Within each run, start a window every ``step`` samples while a full
       ``window`` fits; a shorter tail is dropped.
    3. Recompute the fingerprint of the result.

    Input semantics
    ---------------
    One row per sample time, grouped and time-ordered within each group.

    Output semantics
    ----------------
    One row per window.  The group of every window is the group of all of its
    samples, so a :class:`Split` made on the source applies unchanged.  A
    window is available only when every sample in it is.

    Applicability
    -------------
    Machine-independent.

    Provenance
    ----------
    .. [kaufman2012] S. Kaufman et al., "Leakage in data mining", ACM TKDD 6,
       15 (2012): windows from one unit on both sides of a holdout leak.
    """
    if int(window) != window or int(step) != step or window < 1 or step < 1:
        raise ModelContractError(f"window and step must be positive integers; got {window}, {step}")
    window, step = int(window), int(step)
    if target_mode is None:
        target_mode = "none" if dataset.targets is None else "window"
    if target_mode not in ("window", "last", "none"):
        raise ModelContractError(f"unknown target_mode {target_mode!r}")
    if target_mode != "none" and dataset.targets is None:
        raise ModelContractError(f"target_mode={target_mode!r} but the dataset has no targets")
    starts = [
        s
        for start, stop in _runs(dataset.group_labels)
        for s in range(start, stop - window + 1, step)
    ]
    idx = np.asarray(starts, dtype=int)
    take = idx[:, None] + np.arange(window)[None, :] if len(idx) else np.zeros((0, window), dtype=int)
    inputs = dataset.inputs[take] if len(idx) else np.zeros((0, window, *dataset.input_shape), dataset.inputs.dtype)
    if target_mode == "window":
        targets = dataset.targets[take]
    elif target_mode == "last":
        targets = dataset.targets[idx + window - 1]
    else:
        targets = None
    meta: Mapping[str, np.ndarray] = {key: np.asarray(v)[idx] for key, v in dataset.sample_meta.items()}
    meta = {**meta, "window_start": idx, "window_stop": idx + window}
    available = None
    if dataset.available is not None:
        available = dataset.available[take].all(axis=1) if len(idx) else np.zeros(0, dtype=bool)
    windowed = DatasetArtifact(
        spec=dataset.spec,
        inputs=inputs,
        groups=np.asarray(dataset.groups)[idx],
        targets=targets,
        sample_meta=meta,
        available=available,
    )
    return windowed.replace(fingerprint=dataset_fingerprint(windowed))


_DATASET_ARRAYS = "arrays.npz"
_DATASET_RECORD = "dataset.json"


def save_dataset(dataset, directory):
    """Materialise a dataset on disk so training can run elsewhere.

    Building a corpus-scale dataset (reading thousands of shots) and training
    on it are separate steps, often on separate machines; this is the hand-off.

    Parameters
    ----------
    dataset : DatasetArtifact
        Fingerprinted dataset from :func:`build_dataset` [-].
    directory : str or path-like
        Target directory; must not exist or be empty [-].

    Returns
    -------
    str
        The dataset fingerprint, which :func:`load_dataset` re-checks [-].

    Raises
    ------
    ModelContractError
        When the dataset has no fingerprint, the directory is not empty, or an
        array has dtype ``object``.

    Applicability
    -------------
    Machine-independent.
    """
    if not dataset.fingerprint:
        raise ModelContractError("dataset has no fingerprint; build it with build_dataset")
    directory = Path(directory)
    if directory.exists() and any(directory.iterdir()):
        raise ModelContractError(f"{directory} is not empty")
    directory.mkdir(parents=True, exist_ok=True)
    arrays = {"inputs": np.asarray(dataset.inputs), "groups": dataset.group_labels.astype(str)}
    if dataset.targets is not None:
        arrays["targets"] = np.asarray(dataset.targets)
    if dataset.available is not None:
        arrays["available"] = np.asarray(dataset.available)
    for key, value in dataset.sample_meta.items():
        arrays[f"meta:{key}"] = np.asarray(value)
    for key, value in arrays.items():
        if value.dtype == object:
            raise ModelContractError(f"{key} has dtype object and cannot be stored without pickle")
    np.savez(directory / _DATASET_ARRAYS, **arrays)
    record = {
        "fingerprint": dataset.fingerprint,
        "spec": dataset.spec.to_dict(),
        "feature_spec": dataset.feature_spec.to_dict() if isinstance(dataset, FeatureDataset) else None,
        "group_dtype": str(np.asarray(dataset.groups).dtype),
    }
    (directory / _DATASET_RECORD).write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return dataset.fingerprint


def load_dataset(directory):
    """Load a dataset written by :func:`save_dataset`, verifying its fingerprint.

    Parameters
    ----------
    directory : str or path-like
        Directory holding ``arrays.npz`` and ``dataset.json`` [-].

    Returns
    -------
    DatasetArtifact
        The dataset, a :class:`FeatureDataset` when one was saved [-].

    Raises
    ------
    ModelContractError
        When the files are missing or the recomputed fingerprint differs from
        the recorded one -- the arrays changed after they were saved.

    Applicability
    -------------
    Machine-independent.
    """
    directory = Path(directory)
    record_path = directory / _DATASET_RECORD
    if not record_path.is_file() or not (directory / _DATASET_ARRAYS).is_file():
        raise ModelContractError(f"{directory} does not hold a saved dataset")
    record = json.loads(record_path.read_text(encoding="utf-8"))
    with np.load(directory / _DATASET_ARRAYS, allow_pickle=False) as payload:
        arrays = {key: payload[key] for key in payload.files}
    spec_record = record["spec"]
    spec = DatasetSpec(
        name=spec_record["name"],
        task=spec_record["task"],
        group_key=spec_record.get("group_key", "group"),
        input_names=tuple(spec_record.get("input_names", ())),
        target_names=tuple(spec_record.get("target_names", ())),
        description=spec_record.get("description", ""),
        metadata=spec_record.get("metadata", {}),
    )
    groups = arrays["groups"]
    if np.dtype(record.get("group_dtype", groups.dtype)).kind in "iu":
        groups = groups.astype(record["group_dtype"])
    fields = dict(
        spec=spec,
        inputs=arrays["inputs"],
        groups=groups,
        targets=arrays.get("targets"),
        sample_meta={k[len("meta:"):]: v for k, v in arrays.items() if k.startswith("meta:")},
        available=arrays.get("available"),
    )
    fs = record.get("feature_spec")
    if fs is not None:
        dataset = FeatureDataset(
            feature_spec=FeatureSpec(
                feature_names=tuple(fs["feature_names"]),
                blocks={k: tuple(v) for k, v in fs.get("blocks", {}).items()},
                version=fs.get("version", ""),
            ),
            **fields,
        )
    else:
        dataset = DatasetArtifact(**fields)
    fingerprint = dataset_fingerprint(dataset)
    if fingerprint != record["fingerprint"]:
        raise ModelContractError(
            f"dataset in {directory} has fingerprint {fingerprint}, but was saved as {record['fingerprint']}"
        )
    return dataset.replace(fingerprint=fingerprint)
