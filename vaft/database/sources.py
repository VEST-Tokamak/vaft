"""The named HSDS dataset sources VAFT publishes each analysis lineage to.

VAFT once stored every IMAS product under one namespace, so two valid
representations of the same IDS at the same occurrence overwrote each other.
Named sources give each lineage its own namespace: the EFIT baseline in
``main``, its CHEASE refinement in ``main/chease``, and each stability product
that follows from it in a source of its own.

Names are **hierarchical**, and the hierarchy is the storage layout rather than
a convention laid over it -- ``main/chease/dcon-peeling`` is a real HSDS folder
path. So a derived product says what it derives from by where it lives, and the
logical identity and the physical one cannot drift apart. The shape is
``{family}/{refinement}/{product}``: which reconstruction lineage, which
equilibrium was handed to the solver, and which stability calculation ran.

Nesting is **not** composition. ``load(source="main")`` returns ``main`` and
never acquires ``main/chease``; :func:`children` makes the relationship
discoverable, and discovering it composes nothing. That rule matters more here
than it did when every source was flat, because a hierarchy is exactly the shape
that invites an implicit union.

A source may also be a read-only *projection* of another. ``magnetic-efit``
names the magnetic reconstruction lineage and resolves to ``main``, which is
where that reconstruction is stored. It creates no dataset and can never be
written to -- two writable names for one dataset is the collision this module
exists to prevent.

A source is also how an optional diagnostic stays out of the baseline's
semantics: ``impa`` is *sparse*, holding only the shots whose IMPA product was
intentionally produced, so a shot missing from it says nothing about whether the
baseline shot exists or succeeded (issue #305).  Sources are never unioned on
read; composing two is an explicit call to
:func:`vaft.database.composition.compose`.

This module is the only place a namespace name is written down.  Every public
entry point resolves its ``source`` argument through :func:`resolve`, so the
default, the deprecated aliases, the grammar and the read-only rule for the
legacy ``public`` namespace are stated once.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
import os
import re
from typing import Any
import warnings


__all__ = [
    "DEFAULT_SOURCE",
    "LEGACY_SOURCE",
    "EXTRA_SOURCES_VARIABLE",
    "FAMILY_SOURCES",
    "HSDSSource",
    "HSDSSourceError",
    "UnknownSourceError",
    "ReadOnlySourceError",
    "MissingSourceError",
    "CATALOG",
    "known_sources",
    "children",
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
# convention, so a dot here would be ambiguous.  Restricting each segment to
# lowercase, digits and hyphens keeps '/main/39915/equilibrium.h5' the only
# reading of a resolved name.
_SEGMENT = r"[a-z0-9]+(?:-[a-z0-9]+)*"

#: A name is one or more segments joined by "/".
#:
#: Nesting is how a derived product says what it derives from:
#: ``main/chease/dcon-peeling`` is a real HSDS folder path, so the logical
#: identity and the physical one are the same thing rather than two conventions
#: that have to be kept in step. A single segment is still a valid name, so
#: every existing source reads unchanged.
#:
#: This admits ``public/39915``, which used to fail the grammar. It now fails
#: *lookup* instead -- a better error, because the problem with it was never its
#: shape but that it names a shot inside a source rather than a source.
_NAME = re.compile(rf"^{_SEGMENT}(?:/{_SEGMENT})*$")

#: How deep a source name may go. `family/refinement/product` is three, and
#: nothing in the design needs a fourth; a bound keeps a typo'd loop from
#: creating an unbounded folder tree on the deployment.
_MAX_SEGMENTS = 3


class HSDSSourceError(ValueError):
    """Base error for an unusable HSDS source name."""


class UnknownSourceError(HSDSSourceError):
    """Raised when a name is not in the catalog and was not opted into."""


class ReadOnlySourceError(HSDSSourceError):
    """Raised when a write is requested against a read-only source."""


class MissingSourceError(HSDSSourceError):
    """Raised when a named source has no folder on the HSDS deployment."""

    def __init__(self, source: str, detail: str = "") -> None:
        # HSDS has no mkdir -p: `hstouch` on a nested path whose parent is
        # absent fails, so a hierarchical source needs one command per level,
        # outermost first. Listing them saves an operator discovering that one
        # failure at a time.
        parts = source.split("/")
        levels = ["/".join(parts[: i + 1]) for i in range(len(parts))]
        commands = "\n".join(f"    hstouch -u <admin> -o <owner> /{name}/" for name in levels)
        message = (
            f"HSDS source {source!r} does not exist on this deployment, or is not "
            f"readable with your credentials. The uploader does not create a "
            f"folder; an HSDS administrator must run"
            + (" each of these, in order:\n" if len(levels) > 1 else ":\n")
            + commands
        )
        if detail:
            message += f"\nDetail: {detail}"
        super().__init__(message)
        self.source = source


@dataclass(frozen=True)
class HSDSSource:
    """One named HSDS namespace and the lineage it stores.

    ``sparse`` describes coverage, not access: a sparse source holds only the
    shots for which its product was intentionally produced, so a shot missing
    from it means "no published product", never "the shot is missing" or "the
    baseline failed" (issue #305).  It is documentation the CLI and the docs
    render -- discovery needs no new code, because :func:`exist_shot` already
    lists whatever folders exist and nothing unions two sources.
    """

    name: str
    purpose: str
    writable: bool = True
    sparse: bool = False
    #: The source this one hangs beneath, or ``None`` for a root.
    #:
    #: Derived from the name -- everything before the last ``/`` -- but stated
    #: so the catalog can be checked against itself: a child whose parent is not
    #: in the catalog is a typo that would otherwise only surface as a missing
    #: folder on the deployment.
    parent: str | None = None
    #: A read-only name that resolves to another source.
    #:
    #: ``magnetic-efit`` projects to ``main``: the magnetic reconstruction is
    #: what ``main`` holds, and naming it explicitly lets a caller say which
    #: lineage it wants without the catalog carrying two writable destinations
    #: for one dataset. An alias is never a write destination -- that is the
    #: whole point, since two writable names for one dataset is the collision
    #: the source model exists to prevent.
    projects_to: str | None = None

    def __post_init__(self) -> None:
        if self.projects_to is not None and self.writable:
            raise HSDSSourceError(
                f"Source {self.name!r} projects to {self.projects_to!r} and so "
                "must not be writable: two writable names for one dataset is "
                "the collision named sources exist to prevent."
            )

    @property
    def derived_parent(self) -> str | None:
        """The parent implied by the name, which `parent` must agree with."""
        return self.name.rsplit("/", 1)[0] if "/" in self.name else None

    @property
    def ancestors(self) -> tuple[str, ...]:
        """Every source that must exist before this one can be created.

        Outermost first, so an operator can run them in order: HSDS has no
        mkdir -p, and `hstouch` on a nested path whose parent is absent fails.
        """
        parts = self.name.split("/")
        return tuple("/".join(parts[: i + 1]) for i in range(len(parts) - 1))


#: The refinement every stability product in the catalog is built on today.
#: Stated once so adding a second refinement is one edit rather than a sweep.
_CATALOG_REFINEMENT = "chease"

#: The stability products each equilibrium family publishes.
#:
#: DCON's two edge treatments are separate products because a run yields one of
#: them and can never yield both (`dcon/dcon.F:262-279`), which is what makes
#: them separate *sources* rather than two occurrences in one.
_CATALOG_PRODUCTS = ("dcon-peeling", "dcon-kink", "rdcon", "stride")

#: Which root source holds each equilibrium family's reconstruction.
FAMILY_SOURCES: Mapping[str, str] = {
    "magnetic": DEFAULT_SOURCE,
    "electron": "electron-efit",
    "kinetic": "kinetic-efit",
}


def _lineage_sources() -> tuple["HSDSSource", ...]:
    """The refinement and stability sources under each family root.

    Generated rather than written out: the shape is
    `{family}/{refinement}/{product}` for every combination, and twelve
    hand-written entries would drift the moment a product is added. The roots
    themselves are declared above, because each has its own purpose text.
    """
    built: list[HSDSSource] = []
    for family, root in FAMILY_SOURCES.items():
        refinement = f"{root}/{_CATALOG_REFINEMENT}"
        built.append(
            HSDSSource(
                refinement,
                f"CHEASE-refined equilibrium for the {family} reconstruction "
                f"lineage. Holds the refinement alone -- the reconstruction it "
                f"refines stays in {root!r}, and composing the two is explicit.",
                parent=root,
            )
        )
        for product in _CATALOG_PRODUCTS:
            built.append(
                HSDSSource(
                    f"{refinement}/{product}",
                    f"{product} linear-MHD stability on the {family}/"
                    f"{_CATALOG_REFINEMENT} equilibrium. Holds its own "
                    f"mhd_linear and ntms plus upstream identity hashes, not a "
                    f"copy of the equilibrium or the baseline diagnostics.",
                    parent=refinement,
                )
            )
    return tuple(built)


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
            "CHEASE-refined equilibrium plus DCON/RDCON/GPEC linear-MHD stability "
            "results, in one combined namespace. Superseded by the per-product "
            "sources below and no longer written to: it stays readable so an "
            "existing deployment keeps resolving, and is migrated and deleted by "
            "the gated step in #94. Not an alias -- redirecting it onto the "
            "hierarchy would be implicit composition of several products behind "
            "one name.",
            writable=False,
        ),
        # `main` holds the magnetic reconstruction, so `magnetic-efit` is a name
        # for what is already there rather than a second copy of it. Read-only
        # and projecting: it lets a caller ask for a lineage by name without the
        # catalog carrying two writable destinations for one dataset.
        HSDSSource(
            "magnetic-efit",
            "The magnetic-EFIT lineage. A read-only alias projecting to 'main', "
            "which is where that reconstruction is stored; never a write "
            "destination.",
            writable=False,
            projects_to=DEFAULT_SOURCE,
        ),
        *_lineage_sources(),
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
        # NEO needs a measured ion temperature, which is exactly what separates the
        # kinetic lineage from the electron-only one, so the product hangs beneath
        # the reconstruction that can feed it rather than sitting at the root.
        HSDSSource(
            "kinetic-efit/neoclassical",
            "Neoclassical transport from NEO on the kinetic-EFIT lineage: the "
            "bootstrap current and the particle and energy fluxes, plus the "
            "identity of the state they were computed from -- not a copy of that "
            "state's profiles, which stay in the source that owns them.",
            parent="kinetic-efit",
            sparse=True,
        ),
        HSDSSource(
            "impa",
            "Insertable magnetic probe array. Sparse: only shots whose IMPA "
            "product was intentionally produced. Absence means no published "
            "IMPA product, nothing more.",
            sparse=True,
        ),
        HSDSSource(
            "camera-visible-fluctuation",
            "High-frame-rate (>= 50 kfps) FAST-camera acquisitions. Same IDS as "
            "the routine camera product in the baseline source, so it is kept "
            "apart by lineage rather than by occurrence. Sparse, and disjoint "
            "from routine camera: a shot is one regime or the other.",
            sparse=True,
        ),
    )
}

def _check_catalog(catalog: Mapping[str, "HSDSSource"]) -> None:
    """Every nested source's parent must exist, and agree with its name.

    A child whose parent is missing is a typo that would otherwise surface only
    as a failed `hstouch` on a deployment, days later and far from the edit.
    """
    for name, entry in catalog.items():
        if not _NAME.fullmatch(name):
            raise HSDSSourceError(f"Catalog name {name!r} is not a valid namespace")
        if entry.parent != entry.derived_parent:
            raise HSDSSourceError(
                f"Source {name!r} declares parent {entry.parent!r} but its name "
                f"implies {entry.derived_parent!r}; the two must agree."
            )
        for ancestor in entry.ancestors:
            if ancestor not in catalog:
                raise HSDSSourceError(
                    f"Source {name!r} hangs beneath {ancestor!r}, which is not "
                    "in the catalog."
                )
        if entry.projects_to is not None and entry.projects_to not in catalog:
            raise HSDSSourceError(
                f"Source {name!r} projects to {entry.projects_to!r}, which is "
                "not in the catalog."
            )


_check_catalog(_CATALOG)

#: The catalog from issue #56, keyed by namespace name.
CATALOG: Mapping[str, HSDSSource] = _CATALOG


def children(
    name: str, *, environment: Mapping[str, str] | None = None
) -> tuple[HSDSSource, ...]:
    """The sources that hang directly beneath `name`, in catalog order.

    Read from the catalog, not from the deployment. The catalog is what says
    which sources exist; a folder on a server is what says which have been
    provisioned, and conflating the two would make "does this source exist"
    depend on whether an administrator had got to it yet.

    Composition stays explicit either way: knowing that `main/chease` hangs
    beneath `main` does not make `load(source="main")` return it. Nothing is
    ever unioned on read.
    """
    parent = _grammar(name, "source")
    _lookup(parent, environment)
    return tuple(
        entry
        for entry in known_sources(environment=environment)
        if entry.parent == parent
    )


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
                f"{EXTRA_SOURCES_VARIABLE} entry {name!r} is not a valid HSDS "
                "namespace; use lowercase letters, digits and single hyphens per "
                "segment, with segments joined by '/'."
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
            f"{label} must be an HSDS namespace such as {DEFAULT_SOURCE!r} or "
            f"'{DEFAULT_SOURCE}/chease/dcon-peeling' — lowercase letters, digits "
            "and single hyphens per segment, segments joined by '/'; the "
            f"hdf5:// protocol and filesystem paths are not accepted. Got {value!r}."
        )
    if value.count("/") + 1 > _MAX_SEGMENTS:
        raise HSDSSourceError(
            f"{label} {value!r} is {value.count('/') + 1} segments deep; the "
            f"grammar is at most {_MAX_SEGMENTS} "
            "(family / refinement / product)."
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
    if entry.projects_to is not None:
        # Resolved before the writable check, so a write to an alias is refused
        # by the alias's own read-only rule rather than by whatever the target
        # happens to allow -- `magnetic-efit` must never be a write destination
        # even though `main` is.
        if writable:
            raise ReadOnlySourceError(
                f"{label} {name!r} is a read-only projection of "
                f"{entry.projects_to!r}; write to {entry.projects_to!r} directly. "
                "Two writable names for one dataset is the collision named "
                "sources exist to prevent."
            )
        return _lookup(entry.projects_to, environment).name
    if writable and not entry.writable:
        # The source's own purpose text, not one blanket reason: `public` is a
        # legacy reference nothing may rewrite, while `chease-mhd-stability` is
        # superseded and awaiting migration. Telling an operator the wrong one
        # sends them looking in the wrong place.
        raise ReadOnlySourceError(
            f"HSDS source {name!r} is read-only and is never written to by "
            f"VAFT. {entry.purpose} Publish to {DEFAULT_SOURCE!r} or another "
            "writable source instead."
        )
    return name


# --------------------------------------------------------------------------- #
# FileDB stage -> HSDS replication
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class StageReplication:
    """How one canonical FileDB OMAS stage reaches a remote backend.

    ``ids`` is the stage's *owned* subtree: the IDS it computes, and therefore
    both what it publishes and what its local product holds.  A stage that
    solves against IDS another stage owns -- eddy needs ``magnetics`` and
    ``pf_active`` to compute ``pf_passive`` -- reads them, but does not store or
    republish them, because two stages writing the same IDS makes which one is
    authoritative a question the registry cannot answer.  Readers that need more
    than one stage's subtree union them explicitly, in
    :func:`vaft.database.composition.compose_stage_products`.

    ``optional`` marks a stage the baseline does not depend on: a product that
    is not eligible to publish is *recorded* as skipped rather than raised, so
    an absent or rejected optional diagnostic cannot change the exit state of a
    pipeline whose required stages all succeeded (issue #305).
    """

    source: str | None
    ids: tuple[str, ...] = ()
    occurrence: int = 0
    note: str = ""
    deferred_to: str | None = None
    optional: bool = False
    #: Which pipeline builds the product this stage replicates. Routine stages
    #: are produced by pipeline 1 and have a Snakemake rule there; corrective
    #: ones arrive with late external data through pipeline 2 and have none.
    #: Stated rather than inferred from which Snakefile happens to mention a
    #: stage, so a missing rule stays a test failure instead of an ambiguity.
    produced_by: str = "routine"

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
            "ec_launchers",
        ),
    ),
    "eddy": StageReplication(
        source=DEFAULT_SOURCE,
        ids=("pf_passive",),
        note=(
            "solves against the diagnostics IDS but owns only pf_passive; the "
            "product is exactly this projection"
        ),
    ),
    "efit": StageReplication(source=DEFAULT_SOURCE, ids=("equilibrium",)),
    # Owns `magnetics` too, but in its own source: the split is what keeps an
    # optional, campaign-dependent diagnostic out of the baseline product it
    # would otherwise be appended to (issue #305).
    "impa": StageReplication(
        source="impa",
        ids=("magnetics",),
        optional=True,
        note="sparse optional diagnostic; an ineligible product is recorded, not raised",
    ),
    # Externally produced profile diagnostics. They reach `main` beside the
    # baseline they describe, but they are optional in the #305 sense: a shot
    # without a Thomson or CES upload is the normal case, not a failure, so an
    # ineligible product is recorded as skipped rather than raised.
    "thomson": StageReplication(
        source=DEFAULT_SOURCE,
        ids=("thomson_scattering",),
        optional=True,
        note="externally uploaded profile diagnostic; absent on most shots",
        produced_by="corrective",
    ),
    "ces": StageReplication(
        source=DEFAULT_SOURCE,
        ids=("charge_exchange",),
        optional=True,
        note="externally uploaded ion diagnostic; absent on most shots",
        produced_by="corrective",
    ),
    # The mapped profiles, built from thomson (+ ces) against an equilibrium.
    # Electron-only slices carry no total pressure, so this product stays free
    # of any assumed-Ti pressure -- that assumption belongs to electron_efit.
    "core_profiles": StageReplication(
        source=DEFAULT_SOURCE,
        ids=("core_profiles",),
        optional=True,
        note="requires thomson; ion channels only when ces is present",
        produced_by="corrective",
    ),
    # Both share the `equilibrium` IDS with the EFIT baseline and with each
    # other, so all three are separated by source rather than occurrence. Which
    # of the two runs for a shot is decided by whether CES data exist, not by
    # configuration: the ion measurement is what makes a reconstruction fully
    # kinetic rather than electron-only with an assumed Ti/Te ratio.
    "electron_efit": StageReplication(
        source="electron-efit",
        ids=("equilibrium",),
        optional=True,
        note="thomson without ces; pressure uses the assumed Ti/Te ratio",
        produced_by="corrective",
    ),
    "kinetic_efit": StageReplication(
        source="kinetic-efit",
        ids=("equilibrium",),
        optional=True,
        note="thomson with ces; pressure uses the measured ion temperature",
        produced_by="corrective",
    ),
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
    # Both of these are sparse -- a few hundred shots against an archive of
    # thousands -- but sparseness alone is not why IMPA sits in its own source.
    # IMPA is separated because it re-owns `magnetics`, which the baseline
    # diagnostics stage already publishes. These two own IDS no other stage
    # claims, so they can live in the baseline source without a collision, and
    # a shot that never had a soft X-ray or camera acquisition is simply a shot
    # without one (issue #599).
    # Re-owns `core_profiles`, which the corrective profile stage already
    # publishes into the baseline, so it is kept apart by source exactly as the
    # two kinetic EFIT lineages are kept apart from the magnetic baseline. It is
    # corrective rather than routine on purpose: #550 phase 8 requires promotion
    # to stay opt-in, so there is no pipeline-1 rule demanding it for every shot,
    # and on a cold ohmic VEST discharge the bootstrap fraction a campaign-wide
    # sweep would compute is a percent or two of Ip.
    "neoclassical": StageReplication(
        source="kinetic-efit/neoclassical",
        ids=("core_profiles", "core_transport"),
        optional=True,
        produced_by="corrective",
        note=(
            "opt-in solver product; requires a state that passes "
            "vaft.validation.neoclassical.input_readiness"
        ),
    ),
    "soft_x_rays": StageReplication(
        source=DEFAULT_SOURCE,
        ids=("soft_x_rays",),
        optional=True,
        produced_by="corrective",
        note="sparse optional diagnostic; an ineligible product is recorded, not raised",
    ),
    "camera_visible": StageReplication(
        source=DEFAULT_SOURCE,
        ids=("camera_visible",),
        optional=True,
        produced_by="corrective",
        note="sparse optional diagnostic; an ineligible product is recorded, not raised",
    ),
    # Same instrument, same IDS, different acquisition regime, so it collides
    # with routine camera on `camera_visible`. Separated by source rather than
    # by occurrence, because lazy HSDS access reads occurrence 0 only and an
    # occurrence split would make this product unreadable through the path the
    # fluctuation analysis in #161 would use. Verified when the archive was
    # consolidated: no shot appears in both regimes, so the split costs no
    # migration.
    "camera_visible_fluctuation": StageReplication(
        source="camera-visible-fluctuation",
        ids=("camera_visible",),
        optional=True,
        produced_by="corrective",
        note="high-frame-rate camera lineage; kept out of the baseline camera product",
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


#: Which stages publish beneath a family root rather than into it.
#:
#: Keyed on the stage, valued by the path beneath the family's root source. The
#: refinement and the stability product are substituted in, so the destination
#: is derived from the same lineage that named the local product -- one
#: statement of where a thing goes, not two that can disagree.
_LINEAGE_DESTINATIONS = {
    "chease": "{refinement}",
    "mhd_linear": "{refinement}/{product}",
    "gpec_ideal": "{refinement}/{product}",
}


def replication_for_stage(
    stage: Any,
    *,
    family: str = "magnetic",
    refinement: str = "chease",
    product: str | None = None,
) -> StageReplication:
    """Return the replication contract for one canonical FileDB OMAS stage.

    For a stage whose destination depends on the analysis lineage, the source is
    computed rather than looked up: `mhd_linear` for `magnetic/chease/rdcon`
    goes to `main/chease/rdcon`. A table would have to carry one row per
    combination and would drift from the FileDB grammar that produced the local
    product.

    `family` and `refinement` default to the only lineage pipeline 1 runs today,
    so existing callers keep working. `product` does not: there are five of them
    and guessing would send one solver's result to another's source, which is
    the collision the per-product sources exist to prevent.
    """
    key = _stage_key(stage)
    try:
        entry = STAGE_REPLICATION[key]
    except KeyError as exc:  # pragma: no cover - guarded by test_database_sources
        raise HSDSSourceError(
            f"OMAS stage {key!r} has no replication mapping; add one to "
            "vaft.database.sources.STAGE_REPLICATION rather than choosing a "
            "destination at the call site."
        ) from exc

    template = _LINEAGE_DESTINATIONS.get(key)
    if template is None or entry.source is None or entry.deferred_to is not None:
        # A deferred stage has no live destination to compute, and demanding the
        # lineage for one would bury the message that says why it is deferred
        # under a complaint about a missing argument.
        return entry

    root = FAMILY_SOURCES.get(family)
    if root is None:
        raise HSDSSourceError(
            f"Unknown equilibrium family {family!r}; expected one of: "
            + ", ".join(sorted(FAMILY_SOURCES))
        )
    if "{product}" in template and product is None:
        raise HSDSSourceError(
            f"The {key!r} stage is one solve's result, so replicating it needs "
            "product= (for example 'dcon-peeling'); a combined destination is "
            "what the per-product sources replaced."
        )
    suffix = template.format(refinement=refinement, product=product)
    return replace(entry, source=f"{root}/{suffix}")


def source_for_stage(
    stage: Any,
    *,
    family: str = "magnetic",
    refinement: str = "chease",
    product: str | None = None,
) -> str:
    """Return the HSDS source a canonical FileDB OMAS stage is replicated into.

    Takes the same lineage as :func:`replication_for_stage`, because for the
    stages that are one solve's result the destination *is* a function of it.
    """
    key = _stage_key(stage)
    entry = replication_for_stage(
        key, family=family, refinement=refinement, product=product
    )
    if entry.source is None:
        raise HSDSSourceError(
            f"OMAS stage {key!r} is not replicated to HSDS"
            + (f" ({entry.note})" if entry.note else "")
        )
    return entry.source


def replicable_stages(*, produced_by: str | None = None) -> tuple[str, ...]:
    """Return the stages with a destination and a wired replication rule.

    ``produced_by`` narrows the result to one pipeline's products. A workflow
    asserting that it has a rule for every stage it replicates must pass its
    own producer, or it will demand rules for stages another pipeline builds.
    """
    return tuple(
        stage
        for stage, entry in STAGE_REPLICATION.items()
        if entry.replicable and (produced_by is None or entry.produced_by == produced_by)
    )
