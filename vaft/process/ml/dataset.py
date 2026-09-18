"""Dataset construction, fingerprinting, group splitting and windowing (#669).

The split is defined over the *independent scientific unit* -- a shot, a
simulation case -- and is carried as a group-to-partition assignment, so it
applies unchanged to every windowed or augmented dataset derived afterwards.
"""

from __future__ import annotations

import hashlib
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

__all__ = ["build_dataset", "dataset_fingerprint", "split_groups", "window_dataset"]


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
    2. Hash inputs, targets, string group labels and each ``sample_meta``
       array in sorted key order, each prefixed by its name, dtype and shape.

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
    for key in sorted(dataset.sample_meta):
        _hash_array(digest, f"meta:{key}", np.asarray(dataset.sample_meta[key]))
    return digest.hexdigest()


def build_dataset(inputs, groups, *, spec, targets=None, sample_meta=None, feature_spec=None):
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

    Returns
    -------
    DatasetArtifact
        The dataset with its ``fingerprint`` filled in [-].

    Raises
    ------
    ModelContractError
        On mismatched lengths, a non-finite input, or a feature table whose
        width differs from ``feature_spec``.

    Applicability
    -------------
    Machine-independent.

    Limitations
    -----------
    Inputs must be finite: imputation is a scientific choice (train-partition
    medians, a flag, a drop) that belongs to the caller before this point.
    """
    if not isinstance(spec, DatasetSpec):
        raise ModelContractError("spec must be a DatasetSpec")
    x = np.asarray(inputs)
    if x.ndim < 2:
        raise ModelContractError(f"inputs need a sample axis and at least one feature axis; got {x.shape}")
    if not np.all(np.isfinite(x)):
        raise ModelContractError("inputs contain non-finite values; impute or drop them first")
    y = None if targets is None else np.asarray(targets)
    if y is not None and y.ndim == 1:
        y = y[:, None]
    g = np.asarray(groups)
    if g.ndim != 1:
        raise ModelContractError(f"groups must be one-dimensional; got {g.shape}")
    meta = {str(k): np.asarray(v) for k, v in dict(sample_meta or {}).items()}
    fields = dict(spec=spec, inputs=x, groups=g, targets=y, sample_meta=meta)
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
    3. Permute the remaining groups with ``numpy.random.default_rng(seed)`` and
       cut the permutation at the cumulative fractions, rounding so every
       partition with a positive fraction receives at least one group.

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
    samples, so a :class:`Split` made on the source applies unchanged.

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
    windowed = DatasetArtifact(
        spec=dataset.spec,
        inputs=inputs,
        groups=np.asarray(dataset.groups)[idx],
        targets=targets,
        sample_meta=meta,
    )
    return windowed.replace(fingerprint=dataset_fingerprint(windowed))
