"""Named, mask-aware scientific samples with bounded-memory source access.

Framework-free contracts; a source is a picklable sequence implementing
``__len__`` and ``__getitem__``. File-backed sources need not load the corpus.
Only the Torch backend consumes this contract. Ragged axes are leading sample
axes; ``True`` always means valid, including generated padding masks.
"""
from __future__ import annotations

__all__ = ['TensorSpec', 'StructuredSample', 'StructuredDatasetArtifact', 'build_structured_dataset']

from dataclasses import dataclass, field
import hashlib
import re
from types import MappingProxyType
from typing import Mapping

import numpy as np

from ._types import DatasetSpec, ModelContractError, canonical_json


@dataclass(frozen=True)
class TensorSpec:
    """One sample tensor: shape (leading None for a set), dtype, unit and mask.

    ``mask`` names a boolean input tensor. Targets use ``target_masks`` in the
    sample instead. Normalization pools samples and the ragged axis, preserving
    fixed element/channel axes. Boolean/integer tensors are never normalized.
    """
    shape: tuple
    dtype: str = "float32"
    unit: str = "[-]"
    mask: str | None = None
    normalize: bool = True

    def __post_init__(self):
        shape = tuple(self.shape)
        if any(s is None for s in shape[1:]) or any(s is not None and (not isinstance(s, int) or s < 1) for s in shape):
            raise ModelContractError("only the leading tensor axis may be ragged (None)")
        dtype = np.dtype(self.dtype)
        if dtype.kind not in "bifu":
            raise ModelContractError("structured tensors require real numeric or boolean dtypes")
        object.__setattr__(self, "shape", shape)
        object.__setattr__(self, "dtype", dtype.name)

    def to_dict(self):
        return {"shape": list(self.shape), "dtype": self.dtype, "unit": self.unit,
                "mask": self.mask, "normalize": self.normalize}


@dataclass(frozen=True)
class StructuredSample:
    """Arrays for one sample. Target masks use True for supervised elements."""
    inputs: Mapping
    targets: Mapping = field(default_factory=dict)
    target_masks: Mapping = field(default_factory=dict)


@dataclass(frozen=True)
class _Subset:
    source: object
    indices: tuple

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        return self.source[self.indices[index]]


def _restore_dataset(kwargs):
    kwargs["spec"] = DatasetSpec(**kwargs["spec"])
    return StructuredDatasetArtifact(**kwargs)


@dataclass(frozen=True, kw_only=True)
class StructuredDatasetArtifact:
    """Lazy samples and portable schema, grouped using the existing Split.

    Sources must return StructuredSample and have stable content. Build with
    ``build_structured_dataset`` to validate and fingerprint content without
    retaining tensors. Metadata includes machine/time/sample identities when
    provided by the scientific adapter; these are never inferred by VAFT.
    """
    spec: DatasetSpec
    source: object
    input_specs: Mapping[str, TensorSpec]
    target_specs: Mapping[str, TensorSpec]
    groups: np.ndarray
    sample_meta: Mapping = field(default_factory=dict)
    available: np.ndarray | None = None
    fingerprint: str = ""

    def __post_init__(self):
        n = len(self.source)
        if not n or len(self.groups) != n:
            raise ModelContractError("structured source needs samples and one group per sample")
        for schemas in (self.input_specs, self.target_specs):
            for name, spec in schemas.items():
                if "__" in name or not re.fullmatch(r"[a-zA-Z][a-zA-Z0-9_]*", name) or not isinstance(spec, TensorSpec):
                    raise ModelContractError("tensor names must be identifiers and schemas TensorSpec")
        if not self.input_specs or not self.target_specs:
            raise ModelContractError("structured supervised data needs input and target schemas")
        if any(None in s.shape for s in self.target_specs.values()):
            raise ModelContractError("structured output shapes must be fixed; use target masks for missing supervision")
        for s in self.input_specs.values():
            if s.mask and (s.mask not in self.input_specs or np.dtype(self.input_specs[s.mask].dtype).kind != "b"):
                raise ModelContractError("input mask must name a declared boolean input")
        if self.available is not None and (np.asarray(self.available).dtype != bool or np.asarray(self.available).shape != (n,)):
            raise ModelContractError("available must be a boolean array with one entry per sample")
        for name, values in self.sample_meta.items():
            if len(values) != n:
                raise ModelContractError(f"sample_meta {name!r} does not match source length")
        for name in ("input_specs", "target_specs", "sample_meta"):
            object.__setattr__(self, name, MappingProxyType(dict(getattr(self, name))))
        object.__setattr__(self, "groups", np.asarray(self.groups))

    def __reduce__(self):
        # MappingProxyType is intentionally immutable but not pickleable; workers
        # reconstruct these framework-free records under spawn on all platforms.
        return _restore_dataset, ({"spec": self.spec.to_dict(), "source": self.source,
            "input_specs": dict(self.input_specs), "target_specs": dict(self.target_specs),
            "groups": self.groups, "sample_meta": dict(self.sample_meta),
            "available": self.available, "fingerprint": self.fingerprint},)

    @property
    def n_samples(self):
        return len(self.source)

    @property
    def group_labels(self):
        return np.asarray([str(g) for g in self.groups])

    @property
    def available_mask(self):
        return np.ones(self.n_samples, bool) if self.available is None else np.asarray(self.available)

    def subset(self, indices):
        idx = np.asarray(indices, dtype=int)
        return StructuredDatasetArtifact(spec=self.spec, source=_Subset(self.source, tuple(idx.tolist())),
            input_specs=self.input_specs, target_specs=self.target_specs, groups=self.groups[idx],
            sample_meta={k: np.asarray(v)[idx] for k, v in self.sample_meta.items()},
            available=None if self.available is None else np.asarray(self.available)[idx])

    def schema(self):
        return {"inputs": {k: v.to_dict() for k, v in self.input_specs.items()},
                "targets": {k: v.to_dict() for k, v in self.target_specs.items()},
                "mask_convention": "True means valid; <input>__valid is generated for ragged padding"}

    def sample(self, index):
        sample = self.source[int(index)]
        if not isinstance(sample, StructuredSample):
            raise ModelContractError("source must return StructuredSample")
        if set(sample.inputs) != set(self.input_specs) or set(sample.targets) != set(self.target_specs):
            raise ModelContractError("sample tensor names differ from declared schema")
        if set(sample.target_masks) - set(self.target_specs):
            raise ModelContractError("target mask names must reference declared targets")
        arrays = {}
        for role, schemas, values in (("inputs", self.input_specs, sample.inputs), ("targets", self.target_specs, sample.targets)):
            checked = {}
            for name, s in schemas.items():
                a = np.asarray(values[name], dtype=s.dtype)
                if a.ndim != len(s.shape) or any(want is not None and got != want for got, want in zip(a.shape, s.shape)):
                    raise ModelContractError(f"{role}.{name} has shape {a.shape}, expected {s.shape}")
                checked[name] = a
            arrays[role] = checked
        masks = {}
        for name, a in arrays["targets"].items():
            masks[name] = valid_mask(sample.target_masks.get(name), a.shape)
            if not np.isfinite(a[masks[name]]).all():
                raise ModelContractError(f"valid target {name!r} contains nonfinite values")
        for name, a in arrays["inputs"].items():
            s = self.input_specs[name]
            valid = valid_mask(arrays["inputs"].get(s.mask) if s.mask else None, a.shape)
            if (self.available is None or bool(self.available[int(index)])) and not np.isfinite(a[valid]).all():
                raise ModelContractError(f"valid input {name!r} contains nonfinite values")
        return StructuredSample(arrays["inputs"], arrays["targets"], masks)


def valid_mask(mask, shape):
    if mask is None:
        return np.ones(shape, bool)
    a = np.asarray(mask)
    if a.dtype != bool:
        raise ModelContractError("validity masks must be boolean")
    # A token mask (N,) expands over its feature axes.
    if a.shape == tuple(shape[:a.ndim]):
        a = a.reshape((*a.shape, *((1,) * (len(shape) - a.ndim))))
    try:
        return np.broadcast_to(a, shape)
    except ValueError as exc:
        raise ModelContractError(f"mask {a.shape} cannot cover tensor {shape}") from exc


def build_structured_dataset(source, groups, *, input_specs, target_specs, spec, sample_meta=None, available=None):
    """Validate and fingerprint named scientific samples without retaining a corpus.

    Parameters
    ----------
    source : sequence
        Stable provider of StructuredSample objects; lazy file access is allowed [-].
    groups : array_like
        Independent scientific group identity for every sample [-].
    input_specs : mapping
        Named input TensorSpec records with shape, dtype, units and masks [-].
    target_specs : mapping
        Named fixed-shape target TensorSpec records [-].
    spec : DatasetSpec
        Task, group convention and source-role/preprocessing provenance [-].
    sample_meta : mapping or None, optional
        Per-sample machine, time or identity arrays [any].
    available : array_like or None, optional
        Boolean sample-level input availability; True means available [-].

    Returns
    -------
    StructuredDatasetArtifact
        Validated source and schema with a content SHA-256 fingerprint [-].

    Processing steps
    ----------------
    1. Validate names, shapes, mask broadcasts and finite valid elements.
    2. Stream metadata and every input, target and validity array into SHA-256.
    3. Retain the stable source and fingerprint, without retaining sampled tensors.

    Convention
    ----------
    True means valid. Missing observations differ from physical zero. Only a
    leading input axis may be variable; target availability uses separate masks.

    Applicability
    -------------
    Machine-independent.

    Limitations
    -----------
    A source must have stable contents. DataLoader workers require a picklable
    source. Fixed output shapes are required; this is not an IMAS data model.

    Provenance
    ----------
    .. [vaft2016] VEST-Tokamak/vaft#2016, structured observation and provenance
       extension of the existing #669 grouped ML lifecycle.
    """
    data = StructuredDatasetArtifact(source=source, groups=groups, input_specs=input_specs,
        target_specs=target_specs, spec=spec, sample_meta=sample_meta or {}, available=available)
    digest = hashlib.sha256(canonical_json({"spec": spec.to_dict(), "schema": data.schema(),
        "groups": data.group_labels.tolist(), "metadata": dict(data.sample_meta),
        "available": data.available_mask}).encode())
    for i in range(data.n_samples):
        sample = data.sample(i)
        for values in (sample.inputs, sample.targets, sample.target_masks):
            for name, a in sorted(values.items()):
                digest.update(canonical_json([name, a.shape, a.dtype.str]).encode())
                digest.update(np.ascontiguousarray(a).tobytes())
    from dataclasses import replace
    return replace(data, fingerprint=digest.hexdigest())


def fit_moments(data, indices, config):
    """Streaming, masked train-only moments; ragged observations pool tokens."""
    stats = {}
    for role, schemas in (("input", data.input_specs), ("target", data.target_specs)):
        for name, s in schemas.items():
            fixed = tuple(v for v in s.shape if v is not None)
            mean, m2, count = np.zeros(fixed), np.zeros(fixed), np.zeros(fixed)
            enabled = s.normalize and np.dtype(s.dtype).kind == "f" and getattr(config, f"normalize_{role}s")
            if not enabled:
                stats[f"{role}.{name}.mean"] = np.zeros(fixed)
                stats[f"{role}.{name}.std"] = np.ones(fixed)
                continue
            for i in indices:
                sample = data.sample(i)
                a = np.asarray((sample.inputs if role == "input" else sample.targets)[name], dtype=float)
                mask = valid_mask(sample.inputs.get(s.mask) if role == "input" and s.mask else
                                  sample.target_masks[name] if role == "target" else None, a.shape)
                rows = a if None in s.shape else a[None]
                masks = mask if None in s.shape else mask[None]
                for row, valid in zip(rows, masks):
                    new_count = count + valid
                    delta = np.where(valid, row, mean) - mean
                    new_mean = mean + np.divide(delta, new_count, out=np.zeros_like(mean), where=new_count > 0)
                    m2 += delta * (np.where(valid, row, new_mean) - new_mean)
                    mean, count = new_mean, new_count
            if enabled and (count == 0).any():
                raise ModelContractError(f"no training observations for normalized {role}.{name}")
            std = np.sqrt(np.divide(m2, count, out=np.zeros_like(m2), where=count > 0))
            std = np.where(std > np.finfo(float).eps, std, 1.0)
            stats[f"{role}.{name}.mean"] = mean if enabled else np.zeros(fixed)
            stats[f"{role}.{name}.std"] = std if enabled else np.ones(fixed)
    return stats
