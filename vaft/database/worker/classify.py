"""Classify a settled shot before any work is scheduled for it (issue #58).

The classifier is the worker's only authority on *which stages apply* to a
shot.  It sees what SQL recorded and answers with a label, a reason, and the
stages worth requesting:

- ``applicable_stages=None`` -- the whole routine pipeline; its own checkpoints
  decide the rest.
- a tuple of stage names -- only those.  Today only ``("raw",)`` is honoured:
  the raw dump is archived and nothing downstream is attempted.
- ``()`` -- nothing to run; the shot is recorded as excluded.

The four-way physics classification of #57 (``daq_missing``, ``vacuum_only``,
``breakdown_or_burnthrough_failed``, ``success``) needs signals the raw
inventory alone cannot give.  It plugs in here through ``classifier:`` in the
worker configuration; until then :class:`RawFieldClassifier` decides only what
the inventory can decide.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from importlib import import_module
from typing import Any, Mapping, Protocol, Sequence


#: The stage set that archives the raw dump and stops.
RAW_ONLY: tuple[str, ...] = ("raw",)

DAQ_MISSING = "daq_missing"
RAW_INCOMPLETE = "raw_incomplete"
UNCLASSIFIED = "unclassified"

#: The pipeline's own default (Snakefile ``REQUIRED_RAW_FIELDS``).
DEFAULT_REQUIRED_FIELDS = (1, 12, 25, 59, 109)


@dataclass(frozen=True)
class ShotObservation:
    """What SQL says about one settled shot."""

    shot: int
    record_datetime: datetime | str | None
    field_codes: frozenset[int]


@dataclass(frozen=True)
class Classification:
    label: str
    reason: str
    applicable_stages: tuple[str, ...] | None

    @property
    def runnable(self) -> bool:
        return self.applicable_stages != ()


class Classifier(Protocol):
    def classify(self, observation: ShotObservation) -> Classification: ...


class RawFieldClassifier:
    """Decide from the SQL field inventory alone.

    Mirrors the pipeline's raw preflight (``validate_raw_dumps.py``) so the
    worker does not ask Snakemake for products the preflight would refuse.
    The preflight stays authoritative -- it also checks sample counts, which
    only the dump shows -- and the harvest reads its verdict afterwards.
    """

    def __init__(self, required_fields: Sequence[int] = DEFAULT_REQUIRED_FIELDS):
        self.required_fields = tuple(int(code) for code in required_fields)

    @classmethod
    def from_pipeline_config(cls, pipeline_config: Mapping[str, Any]) -> "RawFieldClassifier":
        preflight = (pipeline_config.get("raw") or {}).get("preflight") or {}
        return cls(preflight.get("required_fields", DEFAULT_REQUIRED_FIELDS))

    def classify(self, observation: ShotObservation) -> Classification:
        if not observation.field_codes:
            return Classification(
                DAQ_MISSING, "SQL holds no waveform fields for this shot", ()
            )
        missing = sorted(set(self.required_fields) - set(observation.field_codes))
        if missing:
            return Classification(
                RAW_INCOMPLETE,
                "required raw field code(s) missing in SQL: " + ", ".join(map(str, missing)),
                RAW_ONLY,
            )
        return Classification(
            UNCLASSIFIED,
            "required raw fields present; the physics class is decided downstream (#57)",
            None,
        )


def load_classifier(spec: str | None, pipeline_config: Mapping[str, Any]) -> Classifier:
    """Resolve ``module:callable`` to a classifier, or the raw-field default.

    The callable receives the pipeline configuration and returns an object
    with a ``classify(observation)`` method.
    """
    if not spec:
        return RawFieldClassifier.from_pipeline_config(pipeline_config)
    module_name, _, attribute = str(spec).partition(":")
    if not module_name or not attribute:
        raise ValueError(f"classifier must be 'module:callable', got {spec!r}")
    factory = getattr(import_module(module_name), attribute)
    classifier = factory(pipeline_config)
    if not callable(getattr(classifier, "classify", None)):
        raise TypeError(f"{spec} did not return an object with a classify() method")
    return classifier


__all__ = [
    "DAQ_MISSING",
    "RAW_INCOMPLETE",
    "RAW_ONLY",
    "UNCLASSIFIED",
    "Classification",
    "Classifier",
    "RawFieldClassifier",
    "ShotObservation",
    "load_classifier",
]
