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
    "STAGE_SOURCE",
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
# FileDB bridge
# --------------------------------------------------------------------------- #
# Which named source each canonical FileDB OMAS stage is published into.  The
# publication path itself belongs to issue #94; this table is the contract it
# and the incremental-synchronization work (#163) build against, so the mapping
# is stated once here rather than rediscovered per workflow.
STAGE_SOURCE: Mapping[str, str] = {
    "static": DEFAULT_SOURCE,
    "diagnostics": DEFAULT_SOURCE,
    "eddy": DEFAULT_SOURCE,
    "efit": DEFAULT_SOURCE,
    "chease": "chease-mhd-stability",
    "mhd_linear": "chease-mhd-stability",
    "gpec_ideal": "chease-mhd-stability",
}


def source_for_stage(stage: Any) -> str:
    """Return the HSDS source a canonical FileDB OMAS stage publishes into."""
    from .filedb import OMASStage

    try:
        key = OMASStage(stage).value
    except (TypeError, ValueError) as exc:
        choices = ", ".join(member.value for member in OMASStage)
        raise HSDSSourceError(
            f"Invalid OMAS stage {stage!r}; expected one of: {choices}"
        ) from exc
    try:
        return STAGE_SOURCE[key]
    except KeyError as exc:  # pragma: no cover - guarded by test_database_sources
        raise HSDSSourceError(
            f"OMAS stage {key!r} has no HSDS source mapping; add one to "
            "vaft.database.sources.STAGE_SOURCE."
        ) from exc
