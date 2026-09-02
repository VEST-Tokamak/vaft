"""The named HSDS dataset sources VAFT publishes each analysis lineage to.

VAFT once stored every IMAS product under one namespace, so two valid
representations of the same IDS at the same occurrence overwrote each other.
Named sources give each lineage its own namespace: the EFIT baseline in
``main``, its CHEASE refinement and the linear-MHD results that follow from it
in ``chease-mhd-stability``, and so on.

This module is the only place a namespace name is written down.  Every public
entry point resolves its ``source`` argument through :func:`resolve`, so the
default, the deprecated aliases, the grammar and the read-only rule for the
legacy ``public`` namespace are stated once.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import os
import re
from typing import Any
import warnings


__all__ = [
    "DEFAULT_SOURCE",
    "LEGACY_SOURCE",
    "EXTRA_SOURCES_VARIABLE",
    "HSDSSource",
    "HSDSSourceError",
    "UnknownSourceError",
    "ReadOnlySourceError",
    "MissingSourceError",
    "CATALOG",
    "known_sources",
    "describe",
    "is_writable",
    "resolve",
    "source_for_stage",
    "StageReplication",
    "STAGE_REPLICATION",
    "STAGE_SOURCE",
    "replication_for_stage",
    "replicable_stages",
]


#: Namespace used whenever a caller does not name one.
DEFAULT_SOURCE = "main"

#: Read-only namespace produced by the pre-VAFT pipeline.
LEGACY_SOURCE = "public"

#: Comma-separated namespaces to accept in addition to :data:`CATALOG`.
EXTRA_SOURCES_VARIABLE = "VAFT_HSDS_EXTRA_SOURCES"

# HSDS folder names are also written as dotted domains by the legacy h5pyd
# convention, so a dot here would be ambiguous.  Restricting the grammar to
# lowercase, digits and hyphens keeps '/main/39915/equilibrium.h5' the only
# reading of a resolved name.
_NAME = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")


class HSDSSourceError(ValueError):
    """Base error for an unusable HSDS source name."""


class UnknownSourceError(HSDSSourceError):
    """Raised when a name is not in the catalog and was not opted into."""


class ReadOnlySourceError(HSDSSourceError):
    """Raised when a write is requested against a read-only source."""


class MissingSourceError(HSDSSourceError):
    """Raised when a named source has no folder on the HSDS deployment."""

    def __init__(self, source: str, detail: str = "") -> None:
        message = (
            f"HSDS source {source!r} does not exist on this deployment, or is not "
            f"readable with your credentials. The uploader does not create a "
            f"top-level folder; an HSDS administrator must run:\n"
            f"    hstouch -u <admin> -o <owner> /{source}/"
        )
        if detail:
            message += f"\nDetail: {detail}"
        super().__init__(message)
        self.source = source


@dataclass(frozen=True)
class HSDSSource:
    """One named HSDS namespace and the lineage it stores."""

    name: str
    purpose: str
    writable: bool = True


_CATALOG: dict[str, HSDSSource] = {
    source.name: source
    for source in (
        HSDSSource(
            LEGACY_SOURCE,
            "Legacy source produced by the previous pipeline. Read-only; "
            "never migrated, rewritten or deleted.",
            writable=False,
        ),
        HSDSSource(
            DEFAULT_SOURCE,
            "Default source for the VAFT-native pipeline; stores the VAFT EFIT baseline.",
        ),
        HSDSSource(
            "chease-mhd-stability",
            "CHEASE-refined equilibrium plus DCON/RDCON/GPEC linear-MHD stability results.",
        ),
        HSDSSource("vfit-element", "VFIT element-fitting equilibrium."),
        HSDSSource("vfit-gse", "VFIT Grad-Shafranov-equilibrium fitting result."),
        HSDSSource(
            "electron-efit",
            "Kinetic EFIT derived from Thomson scattering with an assumed Ti/Te ratio.",
        ),
        HSDSSource(
            "kinetic-efit",
            "Kinetic EFIT for shots with Thomson scattering and CES/ion-Doppler spectroscopy.",
        ),
    )
}

#: The catalog from issue #56, keyed by namespace name.
CATALOG: Mapping[str, HSDSSource] = _CATALOG


def _extra_sources(environment: Mapping[str, str] | None = None) -> dict[str, HSDSSource]:
    """Return opted-in namespaces from the environment, validated like the rest."""
    environment = os.environ if environment is None else environment
    raw = environment.get(EXTRA_SOURCES_VARIABLE, "")
    extra: dict[str, HSDSSource] = {}
    for name in (part.strip() for part in raw.split(",")):
        if not name or name in _CATALOG:
            continue
        if not _NAME.fullmatch(name):
            raise HSDSSourceError(
                f"{EXTRA_SOURCES_VARIABLE} entry {name!r} is not a bare HSDS namespace; "
                "use lowercase letters, digits and single hyphens."
            )
        extra[name] = HSDSSource(name, f"Opted in via {EXTRA_SOURCES_VARIABLE}.")
    return extra


def known_sources(
    *, environment: Mapping[str, str] | None = None
) -> tuple[HSDSSource, ...]:
    """Return every source a call may name right now, catalog order first."""
    return (*_CATALOG.values(), *_extra_sources(environment).values())


def describe(
    name: str, *, environment: Mapping[str, str] | None = None
) -> HSDSSource:
    """Return the catalog entry for one resolved namespace name."""
    return _lookup(_grammar(name, "source"), environment)


def is_writable(name: str, *, environment: Mapping[str, str] | None = None) -> bool:
    """Return whether VAFT may publish into ``name``."""
    return describe(name, environment=environment).writable


def _grammar(value: Any, label: str) -> str:
    if not isinstance(value, str) or not _NAME.fullmatch(value):
        raise HSDSSourceError(
            f"{label} must be a bare HSDS namespace such as {DEFAULT_SOURCE!r} — "
            "lowercase letters, digits and single hyphens only; the hdf5:// "
            f"protocol and filesystem paths are not accepted. Got {value!r}."
        )
    return value


def _lookup(
    name: str, environment: Mapping[str, str] | None
) -> HSDSSource:
    # Parse the environment first even for a catalog name: an unparseable
    # VAFT_HSDS_EXTRA_SOURCES is a configuration error the caller should hear
    # about, not a namespace that silently goes missing later.
    extra = _extra_sources(environment)
    source = _CATALOG.get(name) or extra.get(name)
    if source is None:
        available = ", ".join(entry.name for entry in known_sources(environment=environment))
        raise UnknownSourceError(
            f"Unknown HSDS source {name!r}; available sources: {available}. "
            f"Set {EXTRA_SOURCES_VARIABLE} to opt into an experiment namespace."
        )
    return source


def resolve(
    source: str | None = None,
    *,
    directory: str | None = None,
    target: str | None = None,
    default: str | None = DEFAULT_SOURCE,
    writable: bool = False,
    label: str = "source",
    stacklevel: int = 3,
    environment: Mapping[str, str] | None = None,
) -> str:
    """Resolve one namespace from ``source`` and its deprecated aliases.

    ``directory`` and ``target`` are the historical parameter names.  They are
    still accepted, one at a time, and warn.  ``writable=True`` additionally
    refuses a read-only source, so a publication path cannot fall through to
    ``public``.
    """
    named = [
        (name, value)
        for name, value in (("source", source), ("directory", directory), ("target", target))
        if value is not None
    ]
    if len(named) > 1:
        given = ", ".join(name for name, _ in named)
        raise TypeError(
            f"Pass only one of source, directory or target; got {given}. "
            "'directory' and 'target' are deprecated aliases of 'source'."
        )
    if named:
        alias, value = named[0]
        if alias != "source":
            warnings.warn(
                f"{alias}= is a deprecated alias for source=; pass source={value!r}.",
                DeprecationWarning,
                stacklevel=stacklevel,
            )
    else:
        if default is None:
            raise TypeError(f"{label} is required")
        value = default

    name = _grammar(value, label)
    entry = _lookup(name, environment)
    if writable and not entry.writable:
        raise ReadOnlySourceError(
            f"HSDS source {name!r} is a read-only legacy reference and is never "
            "written, migrated or deleted by VAFT. Publish to "
            f"{DEFAULT_SOURCE!r} or another writable source instead."
        )
    return name


# --------------------------------------------------------------------------- #
# FileDB stage -> HSDS replication
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class StageReplication:
    """How one canonical FileDB OMAS stage reaches a remote backend.

    ``ids`` is the stage's *owned* subtree, which is not the same as everything
    its product happens to contain.  The eddy product, for instance, is built by
    starting from the finalized diagnostics ODS, so it carries ``magnetics`` and
    ``pf_active`` through -- but it computes only ``pf_passive``, and replicating
    the rest would have eddy overwrite what diagnostics wrote.
    """

    source: str | None
    ids: tuple[str, ...] = ()
    occurrence: int = 0
    note: str = ""
    deferred_to: str | None = None

    @property
    def replicable(self) -> bool:
        """Whether this stage has a destination and a wired rule today."""
        return self.source is not None and self.deferred_to is None


#: The one authority for where each stage goes and what of it travels.  A
#: workflow must consume this rather than restate it, so a destination is never
#: decided in two places (issues #94, #163).
STAGE_REPLICATION: Mapping[str, StageReplication] = {
    # Versioned by machine era, not by shot, so it has no per-shot destination.
    # Its geometry already reaches HSDS inside the diagnostics product, which
    # copies pf_active/tf/magnetics out of it.
    "static": StageReplication(
        source=None,
        note="machine-era product with no shot; travels inside diagnostics",
    ),
    "diagnostics": StageReplication(
        source=DEFAULT_SOURCE,
        ids=(
            "magnetics",
            "pf_active",
            "tf",
            "barometry",
            "spectrometer_uv",
            "langmuir_probes",
        ),
    ),
    "eddy": StageReplication(
        source=DEFAULT_SOURCE,
        ids=("pf_passive",),
        note="carries the diagnostics IDS through but computes only pf_passive",
    ),
    "efit": StageReplication(source=DEFAULT_SOURCE, ids=("equilibrium",)),
    # Shares the `equilibrium` IDS with the EFIT baseline; the source split is
    # what keeps the refinement from overwriting the baseline it refines.
    "chease": StageReplication(
        source="chease-mhd-stability", ids=("equilibrium",)
    ),
    # `ntms` carries RDCON/STRIDE's classical Delta-prime, which mhd_linear has
    # no home for.
    "mhd_linear": StageReplication(
        source="chease-mhd-stability", ids=("mhd_linear", "ntms")
    ),
    # Collides with the stability branch on `mhd_linear`, so it is separated by
    # occurrence. Note that lazy HSDS access reads occurrence 0 only, so this
    # product is eager-read for now. Execution and replication remain #95.
    "gpec_ideal": StageReplication(
        source="chease-mhd-stability",
        ids=("mhd_linear", "coils_non_axisymmetric"),
        occurrence=1,
        deferred_to="#95",
    ),
}

#: Destination-only view of :data:`STAGE_REPLICATION`, for callers that only
#: need to know where a stage goes.
STAGE_SOURCE: Mapping[str, str] = {
    stage: entry.source
    for stage, entry in STAGE_REPLICATION.items()
    if entry.source is not None
}


def _stage_key(stage: Any) -> str:
    from .filedb import OMASStage

    try:
        return OMASStage(stage).value
    except (TypeError, ValueError) as exc:
        choices = ", ".join(member.value for member in OMASStage)
        raise HSDSSourceError(
            f"Invalid OMAS stage {stage!r}; expected one of: {choices}"
        ) from exc


def replication_for_stage(stage: Any) -> StageReplication:
    """Return the replication contract for one canonical FileDB OMAS stage."""
    key = _stage_key(stage)
    try:
        return STAGE_REPLICATION[key]
    except KeyError as exc:  # pragma: no cover - guarded by test_database_sources
        raise HSDSSourceError(
            f"OMAS stage {key!r} has no replication mapping; add one to "
            "vaft.database.sources.STAGE_REPLICATION rather than choosing a "
            "destination at the call site."
        ) from exc


def source_for_stage(stage: Any) -> str:
    """Return the HSDS source a canonical FileDB OMAS stage is replicated into."""
    key = _stage_key(stage)
    entry = replication_for_stage(key)
    if entry.source is None:
        raise HSDSSourceError(
            f"OMAS stage {key!r} is not replicated to HSDS"
            + (f" ({entry.note})" if entry.note else "")
        )
    return entry.source


def replicable_stages() -> tuple[str, ...]:
    """Return the stages with a destination and a wired replication rule."""
    return tuple(
        stage for stage, entry in STAGE_REPLICATION.items() if entry.replicable
    )
