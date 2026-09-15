"""Canonical FileDB paths and read-only legacy-layout auditing APIs.

The canonical resolver is deliberately pure: resolving a path never creates a
directory or touches an existing artifact.  The old shot-first layout is
available only through explicitly named read-only APIs.

The grammar (#77)::

    raw/{shot}/
    legacy/{diagnostic}/{shot}/
    omas/static/{machine_version}/
    omas/{stage}/{shot}/                                  diagnostics, eddy, impa, ...
    omas/{stage}/{family}/{shot}/                         efit, chease, mhd_linear, gpec_ideal
    efit/{family}/{shot}/
    chease/{family}/{shot}/
    gpec/{family}/{refinement}/{product}/{shot}/n={n}/
    pipeline/{product}/

with an artifact class (``input``/``output``/``log``/``plot``/``config``/
``work``/``metadata``) as the leaf.

Three dimensions carry the analysis lineage, so a product can never be filed
under a lineage it does not belong to:

``family``
    The equilibrium reconstruction a product descends from -- ``magnetic``,
    ``electron`` or ``kinetic``.
``refinement``
    The equilibrium actually handed to the solver: ``chease``, or ``none``.
``product``
    Which stability calculation ran. DCON's two edge treatments are separate
    products rather than two views of one run, because a run yields one of them
    and can never yield both (``dcon/dcon.F:262-279``).

They are required wherever they apply and refused wherever they do not. Refusal
matters as much as the requirement: a caller who passes ``family`` to a domain
that ignores it believes the product is filed under that lineage when it is not.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from dataclasses import asdict, dataclass, replace
from enum import Enum
import hashlib
import os
from pathlib import Path
import re
from typing import Any


class FileDBError(ValueError):
    """Base error for invalid FileDB configuration or path requests."""


class FileDBConfigError(FileDBError):
    """Raised when the FileDB storage root cannot be configured."""


class FileDBPathError(FileDBError):
    """Raised when a path request is outside the canonical grammar."""


class FileDBDomain(str, Enum):
    RAW = "raw"
    LEGACY = "legacy"
    OMAS = "omas"
    EFIT = "efit"
    CHEASE = "chease"
    GPEC = "gpec"
    PIPELINE = "pipeline"


class OMASStage(str, Enum):
    STATIC = "static"
    DIAGNOSTICS = "diagnostics"
    IMPA = "impa"
    THOMSON = "thomson"
    CES = "ces"
    EDDY = "eddy"
    EFIT = "efit"
    CORE_PROFILES = "core_profiles"
    ELECTRON_EFIT = "electron_efit"
    KINETIC_EFIT = "kinetic_efit"
    CHEASE = "chease"
    MHD_LINEAR = "mhd_linear"
    GPEC_IDEAL = "gpec_ideal"
    # NEO's neoclassical transport on a qualified kinetic state (#550 phase 8).
    # Owns core_transport outright and re-owns core_profiles, which is why it has
    # its own source rather than a place in the baseline.
    NEOCLASSICAL = "neoclassical"
    SOFT_X_RAYS = "soft_x_rays"
    CAMERA_VISIBLE = "camera_visible"
    # The >= 50 kfps acquisitions are the same instrument and the same IDS as
    # routine camera, so they share a mapping -- but two stages cannot own one
    # IDS in one source without the second replacing the first, so they are
    # separated here and given distinct destinations in STAGE_REPLICATION.
    CAMERA_VISIBLE_FLUCTUATION = "camera_visible_fluctuation"


class GPECCode(str, Enum):
    """Legacy stability code names.

    Retained for reading the pre-canonical tree, where a DCON run recorded no
    edge treatment. New products use :class:`StabilityProduct`.
    """

    DCON = "dcon"
    RDCON = "rdcon"
    STRIDE = "stride"
    IDEAL_GPEC = "ideal-gpec"


class EquilibriumFamily(str, Enum):
    """The reconstruction lineage an equilibrium came from.

    Downstream products inherit it, so a kinetic-EFIT refinement and its
    stability results never share a leaf with the magnetic baseline's.
    """

    MAGNETIC = "magnetic"
    ELECTRON = "electron"
    KINETIC = "kinetic"


class Refinement(str, Enum):
    """The equilibrium actually handed to the stability solver."""

    CHEASE = "chease"
    NONE = "none"


class StabilityProduct(str, Enum):
    """One stability calculation.

    DCON's two edge treatments are separate products rather than two views of
    one run: with ``psiedge < psilim`` DCON overwrites its own controls and
    re-integrates (``dcon/dcon.F:262-279``), so a scanned run describes the
    truncated solution and the full-edge one is discarded. A run yields one of
    them and can never yield both.

    ``DCON_LEGACY`` exists only for the pre-canonical tree, whose runs recorded
    no edge treatment and so cannot be attributed to either branch.
    """

    DCON_PEELING = "dcon-peeling"
    DCON_KINK = "dcon-kink"
    RDCON = "rdcon"
    STRIDE = "stride"
    IDEAL_GPEC = "ideal-gpec"
    DCON_LEGACY = "dcon"


#: The lineage a pre-canonical tree is attributed to on migration.
#:
#: The legacy tree recorded neither the reconstruction family nor DCON's edge
#: treatment, so neither can be recovered from a path. Everything in it was
#: produced by the magnetic-EFIT -> CHEASE route, which is what makes the first
#: two safe to assert; the third is deliberately *not* guessed into
#: ``dcon-peeling`` or ``dcon-kink``, because a run that recorded no edge
#: treatment cannot be attributed to either branch.
LEGACY_FAMILY = "magnetic"
LEGACY_REFINEMENT = "chease"
LEGACY_DCON_PRODUCT = "dcon"


#: OMAS stages whose product belongs to one equilibrium family.
_FAMILY_STAGES = frozenset({
    OMASStage.EFIT.value,
    OMASStage.CHEASE.value,
    OMASStage.MHD_LINEAR.value,
    OMASStage.GPEC_IDEAL.value,
})

#: OMAS stages that additionally belong to one refinement and one product.
#:
#: This is where the product dimension has to live: the ``toroidal_mode`` AOS is
#: a dense ``(time, n_tor)`` grid, so two products writing one stage product
#: overwrite each other at the same ``(time_slice, position)``. One product owns
#: one internally coherent ``mhd_linear``; inside it, ``time`` and ``n_tor``
#: remain the only physical axes.
_PRODUCT_STAGES: frozenset[str] = frozenset(
    {OMASStage.MHD_LINEAR.value, OMASStage.GPEC_IDEAL.value}
)


def stage_lineage(
    stage: str,
    *,
    family: str,
    refinement: str | None = None,
    product: str | None = None,
) -> dict[str, str]:
    """The lineage arguments an OMAS `stage`'s path takes, and only those.

    :meth:`FileDB.resolve` refuses a dimension its domain does not carry, which
    is what stops a caller from believing a product is filed under a lineage it
    is not. That guard only works if callers pass exactly the dimensions that
    apply, so the selection is made here once rather than being restated at
    every call site.

    ``refinement`` and ``product`` are required for a stage in
    :data:`_PRODUCT_STAGES` and refused for any other. They have no default
    there: a stage product that is one solve's result has no sensible fallback,
    and guessing would file one solver's result under another's identity.
    """
    if stage not in _FAMILY_STAGES:
        return {}
    if stage not in _PRODUCT_STAGES:
        return {"family": family}
    if refinement is None or product is None:
        raise FileDBPathError(
            f"The {stage!r} stage product belongs to one refinement and one "
            "stability product, so it cannot be resolved without naming them: "
            "pass refinement= (for example 'chease') and product= (for example "
            "'dcon-peeling')."
        )
    return {"family": family, "refinement": refinement, "product": product}


class ArtifactClass(str, Enum):
    INPUT = "input"
    OUTPUT = "output"
    LOG = "log"
    PLOT = "plot"
    CONFIG = "config"
    WORK = "work"
    METADATA = "metadata"


#: Extension every finalized OMAS stage product carries. One constant so a
#: future move to compressed products is a single edit rather than a search.
OMAS_PRODUCT_SUFFIX = ".json"

#: Container a stage's finalized product is written in, where it is not the
#: default. The externally ingested diagnostics are bulk numeric IDS -- one
#: soft X-ray shot is 128 channels by ~39k samples, one camera shot a stack of
#: megapixel frames -- and OMAS's JSON writer spends roughly six bytes of text
#: per float. HDF5 also carries a dtype and a compression filter, which JSON
#: cannot, and is what HSDS stores natively (#599).
OMAS_PRODUCT_SUFFIXES: dict[str, str] = {
    "soft_x_rays": ".h5",
    "camera_visible": ".h5",
    "camera_visible_fluctuation": ".h5",
}

#: Stage manifest: what ran, which components succeeded, why any were skipped.
OMAS_MANIFEST_NAME = "manifest.json"

#: Remote-replication record, kept apart from the manifest on purpose.
OMAS_REPLICATION_NAME = "replication.json"

_SAFE_COMPONENT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_LEGACY_MODE_COMPONENT = re.compile(r"^n{1,2}=[1-9]\d*$")
_ENV_REFERENCE = re.compile(
    r"\$(?:\{(?P<braced>[A-Za-z_][A-Za-z0-9_]*)\}|"
    r"(?P<bare>[A-Za-z_][A-Za-z0-9_]*))"
)
_LEGACY_AREAS = {
    "diagnostics",
    "omas",
    "efit",
    "chease",
    "linear_stability",
    "logs",
}
_DEFAULT_EXPECTED_PRODUCTS = {
    "raw_dump": "diagnostics/vest_{shot}_daq_raw.json.gz",
    "diagnostics_ods": "omas/{shot}_diagnostics.json",
    "eddy_ods": "omas/{shot}_eddy.json",
    "efit_ods": "omas/{shot}_efit.json",
    "chease_ods": "omas/{shot}_chease.json",
}


def _choices(enum: type[Enum]) -> str:
    return ", ".join(member.value for member in enum)


def _enum_value(value: Any, enum: type[Enum], label: str) -> str:
    try:
        return enum(value).value
    except (TypeError, ValueError) as exc:
        raise FileDBPathError(
            f"Invalid {label} {value!r}; expected one of: {_choices(enum)}"
        ) from exc


def _positive_integer(value: Any, label: str) -> int:
    if isinstance(value, bool):
        raise FileDBPathError(f"{label} must be a positive integer, not a boolean")
    try:
        number = int(value)
    except (TypeError, ValueError) as exc:
        raise FileDBPathError(
            f"{label} must be a positive integer; got {value!r}"
        ) from exc
    if str(value).strip() != str(number) or number <= 0:
        raise FileDBPathError(f"{label} must be a positive integer; got {value!r}")
    return number


def _component(value: Any, label: str) -> str:
    if not isinstance(value, str) or not _SAFE_COMPONENT.fullmatch(value):
        raise FileDBPathError(
            f"{label} must be one safe path component containing only letters, "
            f"numbers, '.', '_' or '-'; got {value!r}"
        )
    if value in {".", ".."}:
        raise FileDBPathError(f"{label} cannot be {value!r}")
    return value


def _legacy_stability_component(value: Any) -> str:
    if isinstance(value, str) and _LEGACY_MODE_COMPONENT.fullmatch(value):
        return value
    if isinstance(value, str) and "=" in value:
        raise FileDBPathError(
            "legacy stability mode components must be n=<positive integer> or "
            f"nn=<positive integer>; got {value!r}"
        )
    return _component(value, "legacy stability path component")


def _lineage_absent(domain: str, **values: Any) -> None:
    """Reject every lineage dimension a domain does not carry.

    Silence would be worse than an error here: a caller who passes ``family`` to
    a domain that ignores it believes the product is filed under that lineage
    when it is not.
    """
    for label, value in values.items():
        _absent(value, label, domain)


def _absent(value: Any, label: str, domain: str) -> None:
    if value is not None:
        raise FileDBPathError(f"{label} is not valid for FileDB domain {domain!r}")


def _expand_environment(value: str, environment: Mapping[str, str]) -> str:
    missing: set[str] = set()

    def replace(match: re.Match[str]) -> str:
        name = match.group("braced") or match.group("bare")
        if name not in environment:
            missing.add(name)
            return match.group(0)
        return environment[name]

    expanded = _ENV_REFERENCE.sub(replace, value)
    if missing:
        names = ", ".join(sorted(missing))
        raise FileDBConfigError(
            f"FileDB root references missing environment variable(s): {names}. "
            "Set VAFT_FILEDB_DIR or provide filedb.root explicitly."
        )
    return expanded


@dataclass(frozen=True)
class LegacyResolution:
    """An explicitly read-only path into the former shot-first layout."""

    path: Path
    exists: bool
    read_only: bool = True
    layout: str = "legacy-shot-first"

    def __fspath__(self) -> str:
        return os.fspath(self.path)

    def __str__(self) -> str:
        return str(self.path)


class FileDB:
    """Resolve canonical OMAS-first FileDB paths without filesystem writes."""

    canonical_environment_variable = "VAFT_FILEDB_DIR"

    def __init__(self, root: str | os.PathLike[str]) -> None:
        raw_root = os.fspath(root)
        if not raw_root or not raw_root.strip():
            raise FileDBConfigError("FileDB root must be a non-empty filesystem path")
        self.root = Path(raw_root).expanduser()

    @classmethod
    def from_config(
        cls,
        config: Mapping[str, Any] | None = None,
        *,
        environment: Mapping[str, str] | None = None,
    ) -> "FileDB":
        """Resolve root precedence: explicit config, canonical environment."""

        config = {} if config is None else config
        environment = os.environ if environment is None else environment
        filedb = config.get("filedb", {})
        if filedb is None:
            filedb = {}
        if not isinstance(filedb, Mapping):
            raise FileDBConfigError("filedb configuration must be a mapping")
        configured = filedb.get("root", config.get("filedb_root"))
        if configured is None:
            configured = environment.get(cls.canonical_environment_variable)
        if configured is None:
            raise FileDBConfigError(
                "FileDB root is not configured. Set filedb.root or "
                f"{cls.canonical_environment_variable}."
            )
        if not isinstance(configured, (str, os.PathLike)):
            raise FileDBConfigError("FileDB root must be a filesystem path")
        expanded = _expand_environment(os.fspath(configured), environment)
        return cls(expanded)

    def resolve(
        self,
        domain: str | FileDBDomain,
        *,
        subdomain: str | OMASStage | None = None,
        shot: int | str | None = None,
        machine_version: str | None = None,
        code: str | GPECCode | None = None,
        mode: int | str | None = None,
        family: str | EquilibriumFamily | None = None,
        refinement: str | Refinement | None = None,
        product: str | StabilityProduct | None = None,
        artifact: str | ArtifactClass | None = None,
    ) -> Path:
        """Return a path from the canonical grammar without creating it.

        ``family``/``refinement``/``product`` carry the analysis lineage: which
        equilibrium reconstruction a product descends from, which refinement was
        handed to the solver, and which stability calculation ran. They are
        required wherever they apply and rejected wherever they do not, so a
        product can never be filed under a lineage it does not belong to by
        omitting an argument.
        """

        domain_value = _enum_value(domain, FileDBDomain, "domain")
        artifact_value = (
            None
            if artifact is None
            else _enum_value(artifact, ArtifactClass, "artifact class")
        )

        _lineage = {"family": family, "refinement": refinement, "product": product}

        if domain_value == FileDBDomain.RAW.value:
            _lineage_absent(domain_value, **_lineage)
            _absent(subdomain, "subdomain", domain_value)
            _absent(machine_version, "machine_version", domain_value)
            _absent(code, "code", domain_value)
            _absent(mode, "mode", domain_value)
            _absent(artifact_value, "artifact", domain_value)
            path = self.root / domain_value / str(_positive_integer(shot, "shot"))

        elif domain_value == FileDBDomain.LEGACY.value:
            _lineage_absent(domain_value, **_lineage)
            _absent(machine_version, "machine_version", domain_value)
            _absent(code, "code", domain_value)
            _absent(mode, "mode", domain_value)
            diagnostic = _component(subdomain, "legacy diagnostic")
            path = (
                self.root
                / domain_value
                / diagnostic
                / str(_positive_integer(shot, "shot"))
            )

        elif domain_value == FileDBDomain.OMAS.value:
            _absent(code, "code", domain_value)
            _absent(mode, "mode", domain_value)
            stage = _enum_value(subdomain, OMASStage, "OMAS subdomain")
            if stage == OMASStage.STATIC.value:
                _lineage_absent("omas/static", **_lineage)
                _absent(shot, "shot", "omas/static")
                version = _component(machine_version, "machine_version")
                path = self.root / domain_value / stage / version
            else:
                _absent(machine_version, "machine_version", f"omas/{stage}")
                path = self.root / domain_value / stage
                if stage in _FAMILY_STAGES:
                    path = path / _enum_value(
                        family, EquilibriumFamily, "equilibrium family"
                    )
                else:
                    _absent(family, "family", f"omas/{stage}")
                if stage in _PRODUCT_STAGES:
                    path = path / _enum_value(refinement, Refinement, "refinement")
                    path = path / _enum_value(
                        product, StabilityProduct, "stability product"
                    )
                else:
                    _lineage_absent(
                        f"omas/{stage}", refinement=refinement, product=product
                    )
                path = path / str(_positive_integer(shot, "shot"))

        elif domain_value in {FileDBDomain.EFIT.value, FileDBDomain.CHEASE.value}:
            _absent(subdomain, "subdomain", domain_value)
            _absent(machine_version, "machine_version", domain_value)
            _absent(code, "code", domain_value)
            _absent(mode, "mode", domain_value)
            _lineage_absent(domain_value, refinement=refinement, product=product)
            path = (
                self.root
                / domain_value
                / _enum_value(family, EquilibriumFamily, "equilibrium family")
                / str(_positive_integer(shot, "shot"))
            )

        elif domain_value == FileDBDomain.GPEC.value:
            _absent(subdomain, "subdomain", domain_value)
            _absent(machine_version, "machine_version", domain_value)
            _absent(code, "code", domain_value)
            path = (
                self.root
                / domain_value
                / _enum_value(family, EquilibriumFamily, "equilibrium family")
                / _enum_value(refinement, Refinement, "refinement")
                / _enum_value(product, StabilityProduct, "stability product")
                / str(_positive_integer(shot, "shot"))
                / f"n={_positive_integer(mode, 'toroidal mode')}"
            )

        else:
            _absent(shot, "shot", domain_value)
            _absent(machine_version, "machine_version", domain_value)
            _absent(code, "code", domain_value)
            _absent(mode, "mode", domain_value)
            _lineage_absent(domain_value, **_lineage)
            pipeline_product = _component(subdomain, "pipeline product")
            path = self.root / domain_value / pipeline_product

        return path if artifact_value is None else path / artifact_value

    def raw(
        self, shot: int | str
    ) -> Path:
        """Return the flat per-shot raw archive directory.

        Raw DAQ products are intentionally colocated directly under
        ``raw/{shot}``: waveform dump and its manifest are a single archival
        unit, rather than separate output and metadata artifacts.
        """
        return self.resolve("raw", shot=shot)

    def legacy(
        self,
        diagnostic: str,
        shot: int | str,
        *,
        artifact: str | ArtifactClass | None = None,
    ) -> Path:
        return self.resolve(
            "legacy", subdomain=diagnostic, shot=shot, artifact=artifact
        )

    def omas(
        self,
        stage: str | OMASStage,
        *,
        shot: int | str | None = None,
        machine_version: str | None = None,
        family: str | EquilibriumFamily | None = None,
        refinement: str | Refinement | None = None,
        product: str | StabilityProduct | None = None,
        artifact: str | ArtifactClass | None = None,
    ) -> Path:
        return self.resolve(
            "omas",
            subdomain=stage,
            shot=shot,
            machine_version=machine_version,
            family=family,
            refinement=refinement,
            product=product,
            artifact=artifact,
        )

    def efit(
        self,
        shot: int | str,
        *,
        family: str | EquilibriumFamily,
        artifact: str | ArtifactClass | None = None,
    ) -> Path:
        return self.resolve("efit", shot=shot, family=family, artifact=artifact)

    def chease(
        self,
        shot: int | str,
        *,
        family: str | EquilibriumFamily,
        artifact: str | ArtifactClass | None = None,
    ) -> Path:
        return self.resolve("chease", shot=shot, family=family, artifact=artifact)

    def gpec(
        self,
        product: str | StabilityProduct,
        shot: int | str,
        mode: int | str,
        *,
        family: str | EquilibriumFamily,
        refinement: str | Refinement,
        artifact: str | ArtifactClass | None = None,
    ) -> Path:
        """One stability cell: `(family, refinement, product, shot, n_tor)`.

        ``family`` and ``refinement`` are required and have no default. A
        default would let a caller file an electron-EFIT result under the
        magnetic lineage by omitting an argument, which is the collision this
        grammar exists to prevent.
        """
        return self.resolve(
            "gpec",
            product=product,
            shot=shot,
            mode=mode,
            family=family,
            refinement=refinement,
            artifact=artifact,
        )

    def omas_product(
        self,
        stage: str | OMASStage,
        *,
        shot: int | str | None = None,
        machine_version: str | None = None,
        family: str | EquilibriumFamily | None = None,
        refinement: str | Refinement | None = None,
        product: str | StabilityProduct | None = None,
    ) -> Path:
        """Return the finalized ODS file one OMAS stage writes.

        The stage name is the file name, so a caller never has to know whether
        this stage spells its product ``efit.json`` or ``{shot}_efit.json.gz``.
        The container comes from :data:`OMAS_PRODUCT_SUFFIXES`, so a stage that
        stores bulk numeric data as HDF5 resolves here like any other rather
        than being addressed by a hand-built path.
        """
        name = _enum_value(stage, OMASStage, "OMAS subdomain")
        directory = self.omas(
            stage,
            shot=shot,
            machine_version=machine_version,
            family=family,
            refinement=refinement,
            product=product,
            artifact="output"
        )
        suffix = OMAS_PRODUCT_SUFFIXES.get(name, OMAS_PRODUCT_SUFFIX)
        return directory / f"{name}{suffix}"

    def omas_manifest(
        self,
        stage: str | OMASStage,
        *,
        shot: int | str | None = None,
        machine_version: str | None = None,
        family: str | EquilibriumFamily | None = None,
        refinement: str | Refinement | None = None,
        product: str | StabilityProduct | None = None,
    ) -> Path:
        """Return the stage manifest recording how the product was produced."""
        directory = self.omas(
            stage,
            shot=shot,
            machine_version=machine_version,
            family=family,
            refinement=refinement,
            product=product,
            artifact="metadata"
        )
        return directory / OMAS_MANIFEST_NAME

    def omas_replication_record(
        self,
        stage: str | OMASStage,
        *,
        shot: int | str | None = None,
        machine_version: str | None = None,
        family: str | EquilibriumFamily | None = None,
        refinement: str | Refinement | None = None,
        product: str | StabilityProduct | None = None,
    ) -> Path:
        """Return the record of whether this product reached a remote backend.

        Kept beside the manifest and deliberately separate from it: a finalized
        local product says nothing about whether it was replicated anywhere.
        """
        directory = self.omas(
            stage,
            shot=shot,
            machine_version=machine_version,
            family=family,
            refinement=refinement,
            product=product,
            artifact="metadata"
        )
        return directory / OMAS_REPLICATION_NAME

    def pipeline(
        self, product: str, *, artifact: str | ArtifactClass | None = None
    ) -> Path:
        """Return a canonical, batch-scoped pipeline product directory."""
        return self.resolve("pipeline", subdomain=product, artifact=artifact)

    def resolve_legacy_readonly(
        self,
        shot: int | str,
        area: str,
        *relative: str,
        require_exists: bool = False,
    ) -> LegacyResolution:
        """Resolve the former ``{shot}/{area}`` hierarchy without writes."""

        shot_value = _positive_integer(shot, "shot")
        if area not in _LEGACY_AREAS:
            allowed = ", ".join(sorted(_LEGACY_AREAS))
            raise FileDBPathError(
                f"Invalid legacy area {area!r}; expected one of: {allowed}"
            )
        if area == "linear_stability":
            parts = tuple(_legacy_stability_component(part) for part in relative)
        else:
            parts = tuple(
                _component(part, "legacy relative path component") for part in relative
            )
        path = self.root / str(shot_value) / area
        if parts:
            path = path.joinpath(*parts)
        exists = path.exists() or path.is_symlink()
        if require_exists and not exists:
            raise FileNotFoundError(f"Legacy FileDB artifact does not exist: {path}")
        return LegacyResolution(path=path, exists=exists)


@dataclass(frozen=True)
class LegacyAuditEntry:
    source: str
    proposed_target: str | None
    status: str
    reason: str | None = None
    size: int | None = None
    sha256: str | None = None


@dataclass(frozen=True)
class LegacyCollision:
    proposed_target: str
    sources: tuple[str, ...]
    existing_target: bool = False
    existing_target_kind: str | None = None


@dataclass(frozen=True)
class LegacyDuplicate:
    sha256: str
    size: int
    sources: tuple[str, ...]


@dataclass(frozen=True)
class LegacyMissingProduct:
    shot: int
    product: str
    expected_source: str


@dataclass(frozen=True)
class LegacyAuditReport:
    legacy_root: str
    target_root: str
    entries: tuple[LegacyAuditEntry, ...]
    collisions: tuple[LegacyCollision, ...]
    duplicates: tuple[LegacyDuplicate, ...]
    symlinks: tuple[str, ...]
    missing_products: tuple[LegacyMissingProduct, ...]

    @property
    def unmapped(self) -> tuple[LegacyAuditEntry, ...]:
        return tuple(entry for entry in self.entries if entry.status == "unmapped")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "dry_run": True,
            "legacy_root": self.legacy_root,
            "target_root": self.target_root,
            "summary": {
                "files": len(self.entries),
                "mapped": sum(entry.status == "mapped" for entry in self.entries),
                "unmapped": len(self.unmapped),
                "symlinks": len(self.symlinks),
                "collisions": len(self.collisions),
                "duplicate_groups": len(self.duplicates),
                "missing_products": len(self.missing_products),
            },
            "entries": [asdict(entry) for entry in self.entries],
            "collisions": [asdict(item) for item in self.collisions],
            "duplicates": [asdict(item) for item in self.duplicates],
            "symlinks": list(self.symlinks),
            "missing_products": [asdict(item) for item in self.missing_products],
        }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _existing_path_kind(path: Path) -> str | None:
    """Return the occupied destination kind, including broken symlinks."""

    if path.is_symlink():
        return "symlink"
    if path.is_file():
        return "file"
    if path.is_dir():
        return "directory"
    if path.exists():
        return "other"
    return None


def _legacy_code_and_mode(
    parts: tuple[str, ...],
) -> tuple[str, int, tuple[str, ...]] | None:
    code_index = next(
        (
            index
            for index, part in enumerate(parts)
            if part in {item.value for item in GPECCode}
        ),
        None,
    )
    if code_index is None:
        return None
    mode_index = next(
        (
            index
            for index, part in enumerate(parts)
            if _LEGACY_MODE_COMPONENT.fullmatch(part)
        ),
        None,
    )
    if mode_index is None:
        return None
    remaining = tuple(
        part
        for index, part in enumerate(parts)
        if index not in {code_index, mode_index}
    )
    return parts[code_index], int(parts[mode_index].split("=", 1)[1]), remaining


def _propose_mapping(relative: Path, target: FileDB) -> tuple[Path | None, str | None]:
    parts = relative.parts
    if len(parts) < 3 or not parts[0].isdigit():
        return None, "not a recognized shot-first artifact"
    shot = int(parts[0])
    area = parts[1]
    remainder = tuple(parts[2:])
    filename = remainder[-1]

    if area == "diagnostics":
        if filename == f"vest_{shot}_daq_raw.json.gz":
            return target.raw(shot) / filename, None
        return target.legacy("diagnostics", shot, artifact="input").joinpath(
            *remainder
        ), None

    if area == "omas" and len(remainder) == 1:
        suffixes = {
            f"{shot}_diagnostics.json": ("diagnostics", "output"),
            f"{shot}_eddy.json": ("eddy", "output"),
            f"{shot}_constraints.json": ("efit", "work"),
            f"{shot}_efit.json": ("efit", "output"),
            f"{shot}_chease.json": ("chease", "output"),
        }
        stage_artifact = suffixes.get(filename)
        if stage_artifact is None:
            return None, "legacy OMAS product has no canonical stage mapping"
        stage, artifact = stage_artifact
        return target.omas(
            stage,
            shot=shot,
            artifact=artifact,
            family=LEGACY_FAMILY if stage in _FAMILY_STAGES else None,
        ) / filename, None

    if area in {"efit", "chease"}:
        first = remainder[0].lower()
        artifact = (
            "input"
            if first in {"input", "kfile"}
            else "output"
            if first in {"output", "gfile", "afile", "mfile"}
            else "log"
            if first in {"log", "logs"}
            else "plot"
            if first in {"plot", "plots"}
            else "config"
            if first in {"config", "configuration"}
            else "work"
        )
        base = (
            target.efit(shot, family=LEGACY_FAMILY, artifact=artifact)
            if area == "efit"
            else target.chease(shot, family=LEGACY_FAMILY, artifact=artifact)
        )
        tail = (
            remainder[1:]
            if first
            in {
                "input",
                "output",
                "log",
                "logs",
                "plot",
                "plots",
                "config",
                "configuration",
                "kfile",
                "gfile",
                "afile",
                "mfile",
            }
            else remainder
        )
        return base.joinpath(*tail), None

    if area == "linear_stability":
        parsed = _legacy_code_and_mode(remainder)
        if parsed is None:
            return None, "stability artifact has no recognizable code and mode"
        code, mode, remaining = parsed
        # Every legacy code name is also a `StabilityProduct` value, so the code
        # carries over unchanged -- including DCON, which lands on
        # `LEGACY_DCON_PRODUCT` because the two spellings coincide by design.
        # That coincidence is the point: the tree this came from recorded no
        # edge treatment, so the cell must stay on the legacy product rather
        # than being guessed into `dcon-peeling` or `dcon-kink`, which would
        # invent provenance the migration cannot verify. The assertion is here
        # so that renaming either spelling fails loudly instead of silently
        # attributing legacy runs to an edge branch.
        assert LEGACY_DCON_PRODUCT == GPECCode.DCON.value == StabilityProduct.DCON_LEGACY.value
        return target.gpec(
            code,
            shot,
            mode,
            family=LEGACY_FAMILY,
            refinement=LEGACY_REFINEMENT,
            artifact="work",
        ).joinpath(*remaining), None

    return None, f"legacy area {area!r} has no canonical mapping"


def audit_legacy_filedb(
    legacy_root: str | os.PathLike[str],
    *,
    target_root: str | os.PathLike[str] | None = None,
    expected_products: Mapping[str, str] | None = None,
) -> LegacyAuditReport:
    """Inventory and propose mappings without modifying either FileDB root."""

    source_root = Path(legacy_root).expanduser()
    if not source_root.is_dir():
        raise FileNotFoundError(f"Legacy FileDB root is not a directory: {source_root}")
    target = FileDB(source_root if target_root is None else target_root)
    nodes = sorted(source_root.rglob("*"), key=lambda path: path.as_posix())
    symlinks = tuple(
        path.relative_to(source_root).as_posix() for path in nodes if path.is_symlink()
    )
    files = [path for path in nodes if not path.is_symlink() and path.is_file()]

    entries: list[LegacyAuditEntry] = []
    target_sources: dict[str, list[str]] = defaultdict(list)
    duplicate_sources: dict[tuple[int, str], list[str]] = defaultdict(list)
    for path in files:
        relative = path.relative_to(source_root)
        source = relative.as_posix()
        proposed, reason = _propose_mapping(relative, target)
        size = path.stat().st_size
        checksum = _sha256(path)
        duplicate_sources[(size, checksum)].append(source)
        proposed_text = None if proposed is None else str(proposed)
        status = "unmapped" if proposed is None else "mapped"
        entries.append(
            LegacyAuditEntry(
                source=source,
                proposed_target=proposed_text,
                status=status,
                reason=reason,
                size=size,
                sha256=checksum,
            )
        )
        if proposed_text is not None:
            target_sources[proposed_text].append(source)

    collisions: list[LegacyCollision] = []
    for target_path, sources in sorted(target_sources.items()):
        existing_target_kind = _existing_path_kind(Path(target_path))
        if len(sources) > 1 or existing_target_kind is not None:
            collisions.append(
                LegacyCollision(
                    proposed_target=target_path,
                    sources=tuple(sorted(sources)),
                    existing_target=existing_target_kind is not None,
                    existing_target_kind=existing_target_kind,
                )
            )
    duplicates = tuple(
        LegacyDuplicate(checksum, size, tuple(sorted(sources)))
        for (size, checksum), sources in sorted(duplicate_sources.items())
        if len(sources) > 1
    )

    products = dict(
        _DEFAULT_EXPECTED_PRODUCTS if expected_products is None else expected_products
    )
    shots = sorted(
        int(path.name)
        for path in source_root.iterdir()
        if path.is_dir() and not path.is_symlink() and path.name.isdigit()
    )
    missing: list[LegacyMissingProduct] = []
    for shot in shots:
        for product, template in sorted(products.items()):
            expected = source_root / str(shot) / template.format(shot=shot)
            if not expected.is_file():
                missing.append(
                    LegacyMissingProduct(
                        shot=shot,
                        product=product,
                        expected_source=expected.relative_to(source_root).as_posix(),
                    )
                )

    return LegacyAuditReport(
        legacy_root=str(source_root),
        target_root=str(target.root),
        entries=tuple(entries),
        collisions=tuple(collisions),
        duplicates=duplicates,
        symlinks=symlinks,
        missing_products=tuple(missing),
    )


# ---------------------------------------------------------------------------
# Relocating a canonical tree onto the lineage-bearing grammar (#77 / #527).
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GrammarRelocation:
    """One subtree that moves, and the rule that says where."""

    source: str
    target: str
    rule: str


@dataclass(frozen=True)
class GrammarRelocationReport:
    """What a canonical root needs before it resolves under the new grammar.

    Reports *subtrees*, not files: `efit/39915` moving to `efit/magnetic/39915`
    is a single rename of a directory, whatever it contains. That is what makes
    this a relocation rather than a rewrite -- no file is opened, read, hashed or
    written, so nothing in the tree can be corrupted by it.
    """

    root: str
    applied: bool
    relocations: tuple[GrammarRelocation, ...]
    already_canonical: tuple[str, ...]
    collisions: tuple[str, ...]
    unrecognized: tuple[str, ...]

    @property
    def safe_to_apply(self) -> bool:
        """Whether applying this plan would move every subtree it found.

        A collision means a target is already occupied, which this refuses to
        resolve: the two subtrees were written by different runs, and which one
        is authoritative is not a question a path rewriter can answer.
        """
        return not self.collisions

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "dry_run": not self.applied,
            "root": self.root,
            "summary": {
                "relocations": len(self.relocations),
                "already_canonical": len(self.already_canonical),
                "collisions": len(self.collisions),
                "unrecognized": len(self.unrecognized),
                "safe_to_apply": self.safe_to_apply,
            },
            "relocations": [asdict(item) for item in self.relocations],
            "already_canonical": list(self.already_canonical),
            "collisions": list(self.collisions),
            "unrecognized": list(self.unrecognized),
        }


def _is_shot_directory(name: str) -> bool:
    """Whether `name` is a shot number rather than a lineage segment.

    This is the whole discriminator between the two grammars, and it is exact:
    a shot is `_positive_integer`, and no `EquilibriumFamily`, `Refinement` or
    `StabilityProduct` value is all digits. So `efit/39915` can only be the old
    shape and `efit/magnetic` can only be the new one -- there is no input on
    which the two are ambiguous, which is what makes re-running this safe.
    """
    return name.isdigit() and not name.startswith("0")


def _plan_grammar_relocation(root: Path) -> GrammarRelocationReport:
    relocations: list[GrammarRelocation] = []
    already: list[str] = []
    unrecognized: list[str] = []

    families = {item.value for item in EquilibriumFamily}
    legacy_codes = {item.value for item in GPECCode}
    products = {item.value for item in StabilityProduct}

    def record(node: Path, target: Path, rule: str) -> None:
        relocations.append(
            GrammarRelocation(
                source=node.relative_to(root).as_posix(),
                target=target.relative_to(root).as_posix(),
                rule=rule,
            )
        )

    # efit/{shot} and chease/{shot} gain the family that produced them.
    for domain in (FileDBDomain.EFIT.value, FileDBDomain.CHEASE.value):
        base = root / domain
        if not base.is_dir():
            continue
        for node in sorted(base.iterdir()):
            if not node.is_dir():
                unrecognized.append(node.relative_to(root).as_posix())
            elif node.name in families:
                already.append(node.relative_to(root).as_posix())
            elif _is_shot_directory(node.name):
                record(node, base / LEGACY_FAMILY / node.name, f"{domain}/{{shot}}")
            else:
                unrecognized.append(node.relative_to(root).as_posix())

    # gpec/{code} becomes gpec/{family}/{refinement}/{product}: the whole code
    # subtree moves at once, so every (shot, mode) cell under it travels in one
    # rename rather than one per cell.
    gpec = root / FileDBDomain.GPEC.value
    if gpec.is_dir():
        for node in sorted(gpec.iterdir()):
            if not node.is_dir():
                unrecognized.append(node.relative_to(root).as_posix())
            elif node.name in families:
                already.append(node.relative_to(root).as_posix())
            elif node.name in legacy_codes or node.name in products:
                # DCON keeps the legacy product spelling. The tree records no
                # edge treatment, so attributing it to `dcon-peeling` or
                # `dcon-kink` would invent provenance the move cannot verify.
                record(
                    node,
                    gpec / LEGACY_FAMILY / LEGACY_REFINEMENT / node.name,
                    "gpec/{code}",
                )
            else:
                unrecognized.append(node.relative_to(root).as_posix())

    # The OMAS stages that belong to one family gain it too.
    omas = root / FileDBDomain.OMAS.value
    if omas.is_dir():
        for stage_dir in sorted(omas.iterdir()):
            if not stage_dir.is_dir() or stage_dir.name not in _FAMILY_STAGES:
                continue
            for node in sorted(stage_dir.iterdir()):
                if not node.is_dir():
                    unrecognized.append(node.relative_to(root).as_posix())
                elif node.name in families:
                    already.append(node.relative_to(root).as_posix())
                elif _is_shot_directory(node.name):
                    record(
                        node,
                        stage_dir / LEGACY_FAMILY / node.name,
                        f"omas/{stage_dir.name}/{{shot}}",
                    )
                else:
                    unrecognized.append(node.relative_to(root).as_posix())

    collisions = tuple(
        item.source for item in relocations if (root / item.target).exists()
    )
    return GrammarRelocationReport(
        root=str(root),
        applied=False,
        relocations=tuple(relocations),
        already_canonical=tuple(already),
        collisions=collisions,
        unrecognized=tuple(unrecognized),
    )


def audit_filedb_grammar(
    root: str | os.PathLike[str],
) -> GrammarRelocationReport:
    """Plan a canonical root's move onto the lineage-bearing grammar, changing nothing.

    A tree written before the grammar carried `family`/`refinement`/`product`
    resolves to nothing afterwards: `FileDB.efit(shot, family=...)` looks under
    `efit/magnetic/{shot}`, and the data is at `efit/{shot}`. This says what has
    to move.

    Re-running on an already-moved tree reports every subtree as
    `already_canonical` and plans nothing, because a shot directory and a family
    directory can never be confused -- see :func:`_is_shot_directory`.
    """
    source = Path(root).expanduser()
    if not source.is_dir():
        raise FileNotFoundError(f"FileDB root is not a directory: {source}")
    return _plan_grammar_relocation(source)


def relocate_filedb_grammar(
    root: str | os.PathLike[str],
    *,
    apply: bool = False,
) -> GrammarRelocationReport:
    """Move a canonical root onto the lineage-bearing grammar.

    Dry run unless ``apply=True``, and refused outright when the plan collides
    with something already at a target. Two subtrees claiming one destination
    were written by different runs, and which is authoritative is not a question
    a path rewriter can answer -- so it stops and says which, rather than
    merging or overwriting.

    Directories are renamed, never copied and never opened: no file in the tree
    is read or written by this, so an interrupted run leaves every subtree either
    wholly moved or wholly where it was.
    """
    report = audit_filedb_grammar(root)
    if not apply:
        return report
    if not report.safe_to_apply:
        raise FileDBPathError(
            "Refusing to relocate: "
            f"{len(report.collisions)} subtree(s) already have something at their "
            f"destination ({', '.join(report.collisions[:5])}"
            f"{', ...' if len(report.collisions) > 5 else ''}). "
            "Resolve which copy is authoritative before moving."
        )

    base = Path(root).expanduser()
    for item in report.relocations:
        target = base / item.target
        target.parent.mkdir(parents=True, exist_ok=True)
        (base / item.source).rename(target)
    return replace(report, applied=True)


__all__ = [
    "ArtifactClass",
    "EquilibriumFamily",
    "FileDB",
    "FileDBConfigError",
    "FileDBDomain",
    "FileDBError",
    "FileDBPathError",
    "GPECCode",
    "GrammarRelocation",
    "GrammarRelocationReport",
    "LEGACY_DCON_PRODUCT",
    "LEGACY_FAMILY",
    "LEGACY_REFINEMENT",
    "LegacyAuditEntry",
    "LegacyAuditReport",
    "LegacyCollision",
    "LegacyDuplicate",
    "LegacyMissingProduct",
    "LegacyResolution",
    "OMASStage",
    "Refinement",
    "StabilityProduct",
    "audit_filedb_grammar",
    "audit_legacy_filedb",
    "relocate_filedb_grammar",
    "stage_lineage",
]
