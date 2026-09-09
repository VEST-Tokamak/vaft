"""Native EFIT constraint-identifiability workflow for issue #664.

This module deliberately separates study orchestration from the EFIT adapter.
The adapter owns native response-matrix I/O; this workflow owns immutable case
identities, the Stage-1 validation gate, conditional-information summaries,
and the warm-start continuation plan used after that gate passes.

The pure helpers are also useful while reviewing a completed run::

    python workflow/efit_identifiability/identifiability_study.py plan \
        --output /tmp/efit-identifiability-plan --kfile-root /path/to/663/kfiles \
        --executable /path/to/efit

No production EFIT default is changed by this workflow.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import re
import subprocess
import tempfile
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping, Sequence
import warnings

import numpy as np


SCHEMA_VERSION = 1
ISSUE = 664
STAGE1_CASE_MANIFEST = "case_manifest.json"
STUDY_MANIFEST = "identifiability_manifest.json"
RESULTS_JSON = "identifiability_results.json"
RESULTS_MARKDOWN = "identifiability_results.md"
STAGE2_PLAN = "stage2_plan.json"
RESTART_PROOF = "restart_proof.json"
REPOSITORY = Path(__file__).resolve().parents[2]
PROFILE_STUDY = REPOSITORY / "workflow" / "efit_profile_models" / "profile_model_study.py"
CONSTRAINT_STUDY = (
    REPOSITORY
    / "workflow"
    / "efit_constraint_information"
    / "constraint_information_study.py"
)

FAMILY_ORDER = (
    "plasma_current",
    "pf_current",
    "pf_relation",
    "flux_loop",
    "bpol_probe",
    "diamagnetic_flux",
)
CORE_FAMILIES = ("plasma_current", "pf_current", "pf_relation")
INCREMENTAL_FAMILY_SETS: Mapping[str, tuple[str, ...]] = {
    "core": CORE_FAMILIES,
    "core_plus_flux_loops": (*CORE_FAMILIES, "flux_loop"),
    "core_plus_bpol_probes": (*CORE_FAMILIES, "bpol_probe"),
    "core_plus_magnetics": (*CORE_FAMILIES, "flux_loop", "bpol_probe"),
    "core_plus_diamagnetic": (*CORE_FAMILIES, "diamagnetic_flux"),
    "full": FAMILY_ORDER,
}
MFILE_DIAGNOSTIC_CHI2_VARIABLES: Mapping[str, str] = {
    "flux_loop": "saisil",
    "bpol_probe": "saimpi",
    "diamagnetic_flux": "chidflux",
    "pf_current": "chifcc",
}


@dataclass(frozen=True, order=True)
class ReferenceSlice:
    """One mandatory slice and its role in the nonlinear validation."""

    shot: int
    time_ms: int
    role: str

    @property
    def key(self) -> str:
        return f"{self.shot}:{self.time_ms}"

    @property
    def time_s(self) -> float:
        return self.time_ms / 1000.0


REFERENCE_SLICES = (
    ReferenceSlice(41672, 331, "stable_negative_control"),
    ReferenceSlice(41672, 342, "high_condition_stress"),
    ReferenceSlice(41672, 347, "beta_li_outlier"),
    ReferenceSlice(39915, 319, "confirmation"),
    ReferenceSlice(41524, 332, "confirmation"),
)


@dataclass(frozen=True)
class GateThresholds:
    """Numerical acceptance thresholds for native response exports."""

    solution_scaled_error: float = 1.0e-8
    residual_relative_error: float = 1.0e-10
    singular_value_relative_error: float = 1.0e-10
    condition_number_relative_error: float = 5.0e-6
    curvature_sum_relative_error: float = 1.0e-10
    mode_share_sum_error: float = 1.0e-10
    ip_vcurrt_relative_error: float = 5.0e-8
    # Flux-loop and probe rows are persisted as float32 in the m-file.  The
    # absolute floor is intentional: diamagnetic and PF-current totals are
    # commonly 1e-15--1e-22, where a harmless final-write rounding difference
    # has a large relative value.
    mfile_family_chi2_relative_error: float = 5.0e-8
    mfile_family_chi2_absolute_error: float = 1.0e-12
    equality_residual_factor: float = 100.0


@dataclass(frozen=True)
class Stage1Validation:
    """Machine-readable validation result for one reference slice."""

    reference: ReferenceSlice
    outcome: str
    writer_complete: bool
    main_export_present: bool
    external_current_export_present: bool
    parameter_count: int
    pf_parameter_count: int
    pprime_parameter_count: int
    ffprime_parameter_count: int
    exact_constraint_count: int
    solution_scaled_error: float | None
    residual_relative_error: float | None
    singular_value_relative_error: float | None
    condition_number_relative_error: float | None
    curvature_sum_relative_error: float | None
    mode_share_sum_error: float | None
    ip_vcurrt_relative_error: float | None
    mfile_family_chi2_max_relative_error: float | None
    mfile_family_chi2_max_absolute_error: float | None
    mfile_family_chi2_passed: bool
    equality_residual: float | None = None
    equality_residual_bound: float | None = None
    executable_sha256_matches: bool = True
    diagnostic_errors: tuple[str, ...] = ()


@dataclass(frozen=True)
class Stage1Gate:
    passed: bool
    reasons: tuple[str, ...]
    checked_slices: tuple[str, ...]


@dataclass(frozen=True)
class CaseSpec:
    """All scientific and execution inputs needed to identify one run."""

    stage: int
    reference: ReferenceSlice
    kind: str
    kfile: Path
    executable: Path
    scientific_sha256: str
    table_identity: Mapping[str, Any]
    analysis_identity: Mapping[str, Any]
    kfile_semantic_audit: Mapping[str, Any] = field(default_factory=dict)
    build_provenance: Mapping[str, Any] = field(default_factory=dict)
    objective_scale: float = 1.0
    direction_file: Path | None = None
    restart_parent: Path | None = None
    parent_case_id: str | None = None
    chain: str | None = None
    target: float | None = None
    require_linearization: bool = True


@dataclass(frozen=True)
class TargetMode:
    """A deduplicated weak/borderline mode selected for continuation."""

    mode_index: int
    selector: str
    singular_value: float
    threshold_class: str
    retained: bool
    profile_participation: float
    pf_participation: float
    diamagnetic_share: float


@dataclass(frozen=True)
class ContinuationPoint:
    """One ordered point in a directional or objective-strength chain."""

    chain: str
    order: int
    control: str
    requested_value: float
    parent_order: int | None
    cold_start: bool
    parent_chain: str | None = None
    mode_index: int | None = None
    alpha: float | None = None
    log10_scale: float | None = None
    refinement_level: int = 0


@dataclass(frozen=True)
class TransitionMetrics:
    lcfs_rms_mm: float = 0.0
    area_relative: float = 0.0
    volume_relative: float = 0.0
    beta_p_relative: float = 0.0
    li_relative: float = 0.0
    profile_relative_rms: Mapping[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class TransitionClassification:
    displacement: float
    response_onset: bool
    material_branch: bool
    rank_switching: bool
    failed: bool


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if hasattr(value, "__dataclass_fields__"):
        return _json_safe(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in sorted(value.items())}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json_atomic(path: Path, payload: Mapping[str, Any]) -> Path:
    """Atomically replace *path* with canonical, finite JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(
        _json_safe(dict(payload)),
        indent=2,
        sort_keys=True,
        allow_nan=False,
    ) + "\n"
    descriptor, temporary = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(serialized)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise
    return path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        _json_safe(payload),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def case_identity(spec: CaseSpec) -> tuple[str, dict[str, Any]]:
    """Return a content address and the exact identity payload.

    Parent restart bytes are included, not merely the parent case name.  This
    prevents a resumed continuation from silently reusing a changed state.
    """
    for path, label in ((spec.kfile, "k-file"), (spec.executable, "executable")):
        if not Path(path).is_file():
            raise FileNotFoundError(f"{label} does not exist: {path}")
    optional_hashes: dict[str, str | None] = {
        "direction_sha256": None,
        "restart_parent_sha256": None,
    }
    for path, key in (
        (spec.direction_file, "direction_sha256"),
        (spec.restart_parent, "restart_parent_sha256"),
    ):
        if path is not None:
            if not Path(path).is_file():
                raise FileNotFoundError(f"identity input does not exist: {path}")
            optional_hashes[key] = sha256_file(Path(path))
    identity = {
        "schema_version": SCHEMA_VERSION,
        "stage": spec.stage,
        "shot": spec.reference.shot,
        "time_ms": spec.reference.time_ms,
        "kind": spec.kind,
        "objective_scale": spec.objective_scale,
        "scientific_sha256": spec.scientific_sha256,
        "kfile_sha256": sha256_file(spec.kfile),
        "executable_sha256": sha256_file(spec.executable),
        "table_identity": spec.table_identity,
        "analysis_identity": spec.analysis_identity,
        "kfile_semantic_audit": spec.kfile_semantic_audit,
        "build_provenance": spec.build_provenance,
        "parent_case_id": spec.parent_case_id,
        "chain": spec.chain,
        "target": spec.target,
        "require_linearization": spec.require_linearization,
        **optional_hashes,
    }
    digest = _canonical_sha(identity)
    slug = re.sub(r"[^a-zA-Z0-9_.-]", "-", spec.kind).strip("-")
    case_id = (
        f"s{spec.reference.shot}_t{spec.reference.time_ms:05d}_"
        f"{slug}_{digest[:12]}"
    )
    return case_id, identity


def resolve_reference_kfile(root: Path, reference: ReferenceSlice) -> Path:
    """Resolve exactly one frozen #663 k-file for a reference slice."""
    root = Path(root)
    expected = f"k0{reference.shot}.{reference.time_ms:05d}"
    all_candidates = sorted(path for path in root.rglob(expected) if path.is_file())
    # A complete #663 tree contains the same basename under every variant.
    # Select only its explicit baseline subtree; a curated flat root remains
    # supported, but never guess among scientific variants.
    baseline_candidates = [
        path for path in all_candidates if "baseline" in path.relative_to(root).parts
    ]
    candidates = baseline_candidates or all_candidates
    if not candidates:
        raise FileNotFoundError(f"missing frozen reference k-file {expected} under {root}")
    hashes = {sha256_file(path) for path in candidates}
    if len(hashes) != 1:
        joined = ", ".join(str(path) for path in candidates)
        raise ValueError(f"ambiguous non-identical copies of {expected}: {joined}")
    return candidates[0]


def _table_path_for_shot(
    table_identity: Mapping[str, Any], shot: int
) -> Path:
    """Resolve the path that a provenance record binds to one shot."""
    sources = table_identity.get("sources_by_shot")
    source: Any = None
    if isinstance(sources, Mapping):
        source = sources.get(str(shot), sources.get(shot))
    if source is None:
        for key in ("path", "source", "table_dir"):
            if table_identity.get(key) is not None:
                source = table_identity[key]
                break
    if isinstance(source, Mapping):
        for key in ("path", "source", "table_dir"):
            if source.get(key) is not None:
                source = source[key]
                break
    if not isinstance(source, (str, os.PathLike)):
        raise ValueError(
            f"table_identity.sources_by_shot has no path for shot {shot}"
        )
    return Path(source).expanduser().resolve()


def _as_finite_array(value: Any, *, name: str) -> np.ndarray:
    values = np.asarray(value if isinstance(value, (list, tuple)) else [value], dtype=float)
    values = values.reshape(-1)
    if values.size == 0 or not np.all(np.isfinite(values)):
        raise ValueError(f"{name} must contain finite values")
    return values


def _fortran_number_list(text: str) -> list[float]:
    """Parse the numeric/repeat subset emitted by VAFT's namelist writer."""
    values: list[float] = []
    for raw in re.split(r"[\s,]+", text.strip()):
        if not raw:
            continue
        token = raw.split("!", 1)[0]
        if not token:
            continue
        repeat = re.fullmatch(r"(?P<count>\d+)\*(?P<value>.+)", token)
        count = int(repeat.group("count")) if repeat is not None else 1
        number = repeat.group("value") if repeat is not None else token
        value = float(number.replace("d", "e").replace("D", "E"))
        values.extend([value] * count)
    return values


def _ccoils_from_kfile(text: str, *, nrow: int, ncol: int) -> np.ndarray:
    """Read VAFT's explicit ``CCOILS(1,j)`` column assignments losslessly."""
    pattern = re.compile(
        r"(?ims)^\s*CCOILS\(\s*1\s*,\s*(?P<column>\d+)\s*\)\s*=\s*"
        r"(?P<values>.*?)(?=^\s*(?:CCOILS\s*\(|KCCOILS\s*=))"
    )
    matrix = np.full((nrow, ncol), np.nan, dtype=float)
    seen: set[int] = set()
    for match in pattern.finditer(text):
        column = int(match.group("column")) - 1
        if column < 0 or column >= ncol or column in seen:
            raise ValueError("CCOILS contains an invalid or duplicate column assignment")
        values = _fortran_number_list(match.group("values"))
        if len(values) != nrow:
            raise ValueError(
                f"CCOILS column {column + 1} has {len(values)} values, expected {nrow}"
            )
        matrix[:, column] = values
        seen.add(column)
    if seen != set(range(ncol)):
        missing = sorted(set(range(1, ncol + 1)) - {value + 1 for value in seen})
        raise ValueError(f"CCOILS is missing columns {missing}")
    return matrix


def audit_frozen_kfile(
    path: Path,
    *,
    reference: ReferenceSlice,
    scientific: Any,
    table_identity: Mapping[str, Any],
) -> dict[str, Any]:
    """Prove a frozen k-file implements the declared #664 baseline.

    A supplied one-item ``EFITInputs.kfiles`` bypasses VAFT's scientific
    writer.  Its configuration object is therefore provenance only unless we
    inspect the namelist itself.  This audit prevents a same-named profile,
    restart, reweighting, or alternate-table k-file from being mislabeled as
    the fixed ``(2,2)`` zero-edge baseline.
    """
    import f90nml

    path = Path(path)
    text = path.read_text(encoding="utf-8")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        namelist = f90nml.read(path)
    if "in1" not in namelist or "inwant" not in namelist:
        raise ValueError(f"{path}: expected both &IN1 and &INWANT namelists")
    in1 = namelist["in1"]
    inwant = namelist["inwant"]
    expected_scalars: dict[str, float | int] = {
        "kppcur": scientific.profile.kppcur,
        "kffcur": scientific.profile.kffcur,
        "kppfnc": scientific.profile.kppfnc,
        "kfffnc": scientific.profile.kfffnc,
        "pcurbd": scientific.profile.pcurbd,
        "fcurbd": scientific.profile.fcurbd,
        "fwtbp": scientific.profile.fwtbp,
        "relip": scientific.initialization.seed_rzero,
        "rzero": scientific.initialization.rzero,
        "aelip": scientific.initialization.minor_radius,
        "eelip": scientific.initialization.elongation,
        "cutip": scientific.initialization.current_threshold,
        "icinit": scientific.initialization.icinit,
        "relax": scientific.numerics.relaxation,
        "error": scientific.numerics.error_tolerance,
        "serror": scientific.numerics.measurement_error_floor,
        "mxiter": -scientific.numerics.max_iterations,
        "errmin": scientific.numerics.error_minimum,
        "saicon": scientific.numerics.chi_squared_target,
        "iconvr": scientific.numerics.convergence_mode,
        "nxiter": scientific.numerics.inner_iterations,
        "ivesel": 1,
        "ifitvs": 0,
        "ishot": reference.shot,
        "itime": reference.time_ms,
    }
    errors: list[str] = []
    observed: dict[str, Any] = {}
    for key, expected in expected_scalars.items():
        value = in1.get(key)
        observed[key.upper()] = value
        if value is None or expected is None:
            errors.append(f"{key.upper()} is missing")
            continue
        if isinstance(expected, int):
            matches = not isinstance(value, bool) and int(value) == expected
        else:
            try:
                matches = math.isclose(
                    float(value), float(expected), rel_tol=1.0e-12, abs_tol=1.0e-14
                )
            except (TypeError, ValueError):
                matches = False
        if not matches:
            errors.append(f"{key.upper()}={value!r}, expected {expected!r}")

    # Generated baseline scalars occur exactly once.  Reject duplicates even
    # if a namelist parser's last-value rule happens to recover the expected
    # value, because that is not the frozen scientific input.
    for key in expected_scalars:
        occurrences = len(
            re.findall(rf"(?im)^\s*{re.escape(key)}\s*=", text)
        )
        if occurrences != 1:
            errors.append(f"{key.upper()} occurs {occurrences} times, expected once")

    expected_table = _table_path_for_shot(table_identity, reference.shot)
    table_value = in1.get("table_dir")
    if not isinstance(table_value, str):
        errors.append("TABLE_DIR is missing")
        observed_table = None
    else:
        observed_table = str(Path(table_value).expanduser().resolve())
        if Path(observed_table) != expected_table:
            errors.append(
                f"TABLE_DIR={observed_table!r}, expected {str(expected_table)!r}"
            )

    active_families: dict[str, bool] = {}
    family_weight_keys = {
        "plasma_current": "fwtcur",
        "pf_current": "fwtfc",
        "flux_loop": "fwtsi",
        "bpol_probe": "fwtmp2",
        "diamagnetic_flux": "fwtdlc",
    }
    for family, key in family_weight_keys.items():
        try:
            weights = _as_finite_array(in1.get(key), name=key.upper())
            nonzero = weights[np.abs(weights) > 0.0]
            active_families[family] = bool(nonzero.size)
            if not nonzero.size:
                errors.append(f"{family} has no active submitted FWT row")
            elif not np.allclose(nonzero, 1.0, rtol=0.0, atol=1.0e-14):
                errors.append(
                    f"{key.upper()} has non-unit active objective scales: "
                    f"{sorted(set(float(value) for value in nonzero))}"
                )
        except (TypeError, ValueError) as exc:
            active_families[family] = False
            errors.append(str(exc))

    relation_count = inwant.get("kccoils")
    expected_relations = len(scientific.constraints.coil_constraint_targets)
    relation_active = relation_count == expected_relations and expected_relations > 0
    active_families["pf_relation"] = relation_active
    if not relation_active:
        errors.append(
            f"KCCOILS={relation_count!r}, expected {expected_relations} soft PF rows"
        )
    observed_nccoil = inwant.get("nccoil")
    if observed_nccoil != scientific.constraints.nccoil:
        errors.append(
            f"NCCOIL={observed_nccoil!r}, expected {scientific.constraints.nccoil}"
        )
    expected_targets = np.asarray(
        scientific.constraints.coil_constraint_targets, dtype=float
    )
    try:
        observed_targets = _as_finite_array(inwant.get("xcoils"), name="XCOILS")
        if observed_targets.shape != expected_targets.shape or not np.allclose(
            observed_targets, expected_targets, rtol=0.0, atol=1.0e-14
        ):
            errors.append(
                f"XCOILS differs from the fixed {expected_targets.size}-target vector"
            )
    except (TypeError, ValueError) as exc:
        observed_targets = np.asarray([], dtype=float)
        errors.append(str(exc))
    expected_matrix = np.asarray(
        scientific.constraints.coil_constraint_matrix, dtype=float
    )
    try:
        observed_matrix = _ccoils_from_kfile(
            text,
            nrow=expected_matrix.shape[0],
            ncol=expected_matrix.shape[1],
        )
        if not np.allclose(
            observed_matrix, expected_matrix, rtol=0.0, atol=1.0e-14
        ):
            errors.append("CCOILS differs from the fixed 16x12 soft-relation matrix")
    except ValueError as exc:
        observed_matrix = np.empty((0, 0), dtype=float)
        errors.append(str(exc))

    return {
        "passed": not errors,
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "reference": reference.key,
        "scientific_sha256": scientific.sha256,
        "expected_table_dir": str(expected_table),
        "observed_table_dir": observed_table,
        "observed_scalars": observed,
        "active_families": active_families,
        "pf_relations": {
            "kccoils": relation_count,
            "nccoil": observed_nccoil,
            "ccoils": observed_matrix.tolist(),
            "xcoils": observed_targets.tolist(),
        },
        "errors": errors,
    }


def require_frozen_kfile_audit(audit: Mapping[str, Any]) -> None:
    """Raise before execution when a k-file semantic audit did not pass."""
    if not audit.get("passed"):
        errors = "; ".join(str(value) for value in audit.get("errors", ()))
        raise ValueError(
            f"{audit.get('reference', 'unknown slice')}: frozen k-file semantic "
            f"audit failed: {errors or 'unspecified mismatch'}"
        )


def stage1_gate(
    validations: Iterable[Stage1Validation],
    *,
    thresholds: GateThresholds = GateThresholds(),
    expected_references: Sequence[ReferenceSlice] = REFERENCE_SLICES,
) -> Stage1Gate:
    """Require every native-export invariant before Stage 2 may start."""
    items = tuple(validations)
    records: dict[str, Stage1Validation] = {}
    duplicate_keys: list[str] = []
    for item in items:
        key = item.reference.key
        if key in records:
            duplicate_keys.append(key)
        else:
            records[key] = item
    reasons: list[str] = []
    if duplicate_keys:
        reasons.append(
            "duplicate validation slices: " + ", ".join(sorted(set(duplicate_keys)))
        )
    expected = {item.key: item for item in expected_references}
    missing = sorted(set(expected) - set(records))
    extra = sorted(set(records) - set(expected))
    if missing:
        reasons.append(f"missing validation slices: {', '.join(missing)}")
    if extra:
        reasons.append(f"unexpected validation slices: {', '.join(extra)}")

    def require_limit(
        record: Stage1Validation,
        field_name: str,
        limit: float,
        *,
        optional: bool = False,
    ) -> None:
        value = getattr(record, field_name)
        if value is None:
            if not optional:
                reasons.append(f"{record.reference.key}: missing {field_name}")
        elif not math.isfinite(value) or value > limit:
            reasons.append(
                f"{record.reference.key}: {field_name}={value:.6g} exceeds {limit:.6g}"
            )

    for key in sorted(set(records) & set(expected)):
        item = records[key]
        if item.outcome != "accepted":
            reasons.append(f"{key}: equilibrium outcome is {item.outcome!r}")
        for flag in ("writer_complete", "main_export_present", "external_current_export_present"):
            if not getattr(item, flag):
                reasons.append(f"{key}: {flag} is false")
        if not item.executable_sha256_matches:
            reasons.append(f"{key}: sidecar executable SHA-256 does not match case identity")
        counts = (
            item.parameter_count,
            item.pf_parameter_count,
            item.pprime_parameter_count,
            item.ffprime_parameter_count,
        )
        if counts != (20, 16, 2, 2):
            reasons.append(f"{key}: parameter roles are {counts}, expected (20, 16, 2, 2)")
        if item.exact_constraint_count != 0:
            reasons.append(
                f"{key}: baseline exact-constraint count is "
                f"{item.exact_constraint_count}, expected 0"
            )
        require_limit(item, "solution_scaled_error", thresholds.solution_scaled_error)
        require_limit(item, "residual_relative_error", thresholds.residual_relative_error)
        require_limit(
            item,
            "singular_value_relative_error",
            thresholds.singular_value_relative_error,
        )
        require_limit(
            item,
            "condition_number_relative_error",
            thresholds.condition_number_relative_error,
        )
        require_limit(
            item,
            "curvature_sum_relative_error",
            thresholds.curvature_sum_relative_error,
        )
        require_limit(item, "mode_share_sum_error", thresholds.mode_share_sum_error)
        require_limit(
            item,
            "ip_vcurrt_relative_error",
            thresholds.ip_vcurrt_relative_error,
        )
        if not item.mfile_family_chi2_passed:
            reasons.append(
                f"{key}: native diagnostic chi-squared does not reproduce the "
                "m-file family totals within the relative/absolute numerical floor"
            )
        if item.exact_constraint_count:
            if item.equality_residual is None or item.equality_residual_bound is None:
                reasons.append(f"{key}: missing equality residual validation")
            elif item.equality_residual > item.equality_residual_bound:
                reasons.append(
                    f"{key}: equality residual {item.equality_residual:.6g} exceeds "
                    f"{item.equality_residual_bound:.6g}"
                )
        reasons.extend(f"{key}: {message}" for message in item.diagnostic_errors)
    return Stage1Gate(
        passed=not reasons,
        reasons=tuple(reasons),
        checked_slices=tuple(sorted(set(records) & set(expected))),
    )


def _relative_norm(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    denominator = max(float(np.linalg.norm(right)), np.finfo(float).tiny)
    return float(np.linalg.norm(left - right)) / denominator


def validate_native_problem(
    problem: Any,
    report: Any,
    *,
    reference: ReferenceSlice,
    outcome: str,
    diagnostic_errors: Sequence[str] = (),
    expected_executable_sha256: str | None = None,
    mfile_family_chi2_audit: Mapping[str, Any] | None = None,
) -> Stage1Validation:
    """Build the Stage-1 gate record from a parsed native sidecar.

    The current baseline has no true equality rows, so its solver solution is
    independently reproduced with the exported retained singular mask.  A
    future equality-constrained fixture still receives the KKT feasibility
    check, but is not accepted as the frozen VEST baseline.
    """
    block = problem.main
    matrix = np.asarray(block.solver_a, dtype=float)
    rhs = np.asarray(block.solver_rhs, dtype=float)
    solution = np.asarray(block.solver_solution, dtype=float)
    calculated_residual = matrix @ solution - rhs
    exported_residual = np.asarray(block.row_solver_weighted_residual, dtype=float)
    finite_residual = np.isfinite(exported_residual)
    residual_error = None
    if np.any(finite_residual):
        residual_error = _relative_norm(
            calculated_residual[finite_residual], exported_residual[finite_residual]
        )

    computed_singular = np.linalg.svd(matrix, compute_uv=False)
    native_singular = np.asarray(block.singular_values, dtype=float)
    finite_native = native_singular[np.isfinite(native_singular)]
    compare_count = min(computed_singular.size, finite_native.size)
    singular_error = None
    if compare_count:
        singular_error = _relative_norm(
            computed_singular[:compare_count], finite_native[:compare_count]
        )

    solution_error = None
    # Native C is the matrix actually supplied to DGGLSE and therefore acts
    # on the pre-rescale solver vector. ``block.exact_c`` is its derived
    # physical-coordinate form used by the information analysis.
    exact_c = np.asarray(block.solver_exact_c, dtype=float)
    exact_d = np.asarray(block.solver_exact_d, dtype=float)
    if exact_c.shape[0] == 0:
        u, singular, vh = np.linalg.svd(matrix, full_matrices=False)
        retained = np.asarray(block.retained_mask, dtype=bool)[: singular.size]
        coefficients = np.zeros_like(singular)
        coefficients[retained] = (u.T @ rhs)[retained] / singular[retained]
        reproduced = vh.T @ coefficients
        solution_error = _relative_norm(reproduced, solution)

    condition_error = None
    if computed_singular.size and block.condno is not None:
        calculated_condition = (
            math.inf
            if computed_singular[-1] == 0.0
            else float(computed_singular[0] / computed_singular[-1])
        )
        condition_error = abs(calculated_condition - float(block.condno)) / max(
            abs(float(block.condno)), np.finfo(float).tiny
        )

    physical_a = np.asarray(block.weighted_a, dtype=float)
    total_curvature = physical_a.T @ physical_a
    family_curvature = np.zeros_like(total_curvature)
    row_families = np.asarray(block.row_family, dtype=object)
    for family in set(block.row_family):
        selected = physical_a[row_families == family]
        family_curvature += selected.T @ selected
    curvature_error = _relative_norm(family_curvature, total_curvature)

    share_errors = []
    for mode in report.modes:
        if mode.singular_value > 0.0:
            share_errors.append(abs(sum(mode.family_shares.values()) - 1.0))
    share_error = max(share_errors, default=0.0)

    ip_vcurrt_error = None
    accounting = getattr(report, "ip_accounting", None)
    if accounting is not None:
        predicted = getattr(accounting, "predicted_vcurrt_chi2", None)
        reported = getattr(accounting, "reported_chipasma", None)
        if predicted is not None and reported is not None:
            ip_vcurrt_error = abs(float(predicted) - float(reported)) / max(
                abs(float(reported)), np.finfo(float).tiny
            )

    equality_residual = equality_bound = None
    if exact_c.shape[0]:
        equality_residual = float(np.linalg.norm(exact_c @ solution - exact_d))
        equality_bound = float(
            100.0
            * np.finfo(float).eps
            * (np.linalg.norm(exact_c) * np.linalg.norm(solution) + np.linalg.norm(exact_d))
        )

    roles = tuple(str(value) for value in block.parameter_role)
    sidecar_executable_sha256 = getattr(problem, "executable_sha256", None)
    executable_matches = (
        True
        if expected_executable_sha256 is None
        else sidecar_executable_sha256 == expected_executable_sha256
    )
    chi2_audit = dict(mfile_family_chi2_audit or {})
    return Stage1Validation(
        reference=reference,
        outcome=outcome,
        writer_complete=getattr(problem, "schema_version", None) == 1,
        main_export_present=block is not None,
        external_current_export_present=problem.external_current is not None,
        parameter_count=block.ncol,
        pf_parameter_count=roles.count("pf_current"),
        pprime_parameter_count=roles.count("pprime"),
        ffprime_parameter_count=roles.count("ffprime"),
        exact_constraint_count=block.nexact,
        solution_scaled_error=solution_error,
        residual_relative_error=residual_error,
        singular_value_relative_error=singular_error,
        condition_number_relative_error=condition_error,
        curvature_sum_relative_error=curvature_error,
        mode_share_sum_error=share_error,
        ip_vcurrt_relative_error=ip_vcurrt_error,
        mfile_family_chi2_max_relative_error=chi2_audit.get(
            "maximum_relative_error"
        ),
        mfile_family_chi2_max_absolute_error=chi2_audit.get(
            "maximum_absolute_error"
        ),
        mfile_family_chi2_passed=bool(chi2_audit.get("passed", False)),
        equality_residual=equality_residual,
        equality_residual_bound=equality_bound,
        executable_sha256_matches=executable_matches,
        diagnostic_errors=tuple(str(value) for value in diagnostic_errors),
    )


def _validation_from_dict(payload: Mapping[str, Any]) -> Stage1Validation:
    values = dict(payload)
    values["reference"] = ReferenceSlice(**values["reference"])
    values["diagnostic_errors"] = tuple(values.get("diagnostic_errors", ()))
    values.setdefault("mfile_family_chi2_max_relative_error", None)
    values.setdefault("mfile_family_chi2_max_absolute_error", None)
    values.setdefault("mfile_family_chi2_passed", False)
    return Stage1Validation(**values)


def _artifact_records(paths: Iterable[Path]) -> list[dict[str, str]]:
    unique = sorted({Path(path).resolve() for path in paths if Path(path).is_file()})
    return [{"path": str(path), "sha256": sha256_file(path)} for path in unique]


def resumable_case_manifest(
    path: Path, identity: Mapping[str, Any]
) -> Mapping[str, Any] | None:
    """Return a completed case only when identity and every artifact still match."""
    path = Path(path)
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return None
    if (
        payload.get("schema_version") != SCHEMA_VERSION
        or payload.get("status") != "succeeded"
        or payload.get("identity") != _json_safe(identity)
    ):
        return None
    artifacts = payload.get("artifacts", ())
    if not artifacts:
        return None
    for record in artifacts:
        artifact = Path(record.get("path", ""))
        if not artifact.is_file() or sha256_file(artifact) != record.get("sha256"):
            return None
    return payload


def fixed_scientific_config() -> Any:
    """Reconstruct the frozen #579/#663 scientific baseline locally."""
    from vaft.code.efit import (
        EFITConstraintConfig,
        EFITInitializationConfig,
        EFITNumericsConfig,
        EFITProfileConfig,
        EFITScientificConfig,
    )

    return EFITScientificConfig(
        profile=EFITProfileConfig(kppcur=2, kffcur=2, pcurbd=1, fcurbd=1, fwtbp=0),
        initialization=EFITInitializationConfig(ellipse_rzero=0.32, icinit=2),
        numerics=EFITNumericsConfig(
            relaxation=1.0,
            error_tolerance=1.0e-5,
            measurement_error_floor=5.0e-4,
            max_iterations=100,
            error_minimum=1.0e-2,
            chi_squared_target=80.0,
            convergence_mode=2,
            inner_iterations=1,
        ),
        constraints=EFITConstraintConfig(),
    )


def build_provenance_record(
    executable: Path, supplied: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    """Combine VAFT's binary identity with the build's explicit audit record."""
    executable = Path(executable).expanduser().resolve()
    if not executable.is_file():
        raise FileNotFoundError(f"EFIT executable does not exist: {executable}")
    stat = executable.stat()
    record = dict(supplied or {})
    record["executable"] = {
        "role": "efit",
        "path": str(executable),
        "sha256": sha256_file(executable),
        "size": int(stat.st_size),
        "mtime": datetime.fromtimestamp(stat.st_mtime, timezone.utc)
        .replace(microsecond=0)
        .isoformat(),
    }
    return record


def _validated_provenance_artifact(record: Any, *, label: str) -> str:
    if not isinstance(record, Mapping):
        raise ValueError(f"{label} artifact identity is missing")
    path = Path(str(record.get("path", ""))).expanduser()
    expected = str(record.get("sha256", ""))
    if re.fullmatch(r"[0-9a-f]{64}", expected) is None:
        raise ValueError(f"{label} artifact SHA-256 is invalid")
    if not path.is_file() or sha256_file(path) != expected:
        raise ValueError(f"{label} artifact is absent or its SHA-256 changed")
    return expected


def _canonical_provenance_value(value: Any) -> Any:
    if isinstance(value, bytes):
        return {"bytes_hex": value.hex()}
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return _canonical_provenance_value(value.tolist())
    if isinstance(value, np.generic):
        return _canonical_provenance_value(value.item())
    if isinstance(value, Mapping):
        return {
            str(key): _canonical_provenance_value(item)
            for key, item in sorted(value.items())
        }
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_canonical_provenance_value(item) for item in value]
    return value


def _scientific_output_identity(
    path: Path, kind: str, *, allowed_metadata: set[str]
) -> tuple[str, dict[str, Any]]:
    """Derive canonical scientific bytes and the metadata actually removed."""
    path = Path(path)
    removed: dict[str, Any] = {}
    if kind == "gfile":
        from vaft.data import read_geqdsk

        payload = dict(read_geqdsk(path).mapping)
        case = str(payload.get("CASE", ""))
        if "run_date" in allowed_metadata:
            canonical = re.sub(r"\b\d{2}/\d{2}/\d{4}\b", "<run_date>", case)
            if canonical != case:
                removed["run_date"] = case
            payload["CASE"] = canonical
    elif kind == "afile":
        from vaft.data import read_aeqdsk

        parsed = read_aeqdsk(path)
        payload = {
            key: value
            for key, value in parsed.__dict__.items()
            if key != "source"
        }
        if "run_date" in allowed_metadata:
            first_line = path.read_text(encoding="ascii", errors="strict").splitlines()[0]
            removed["run_date"] = first_line
    elif kind == "mfile":
        from vaft.data import read_meqdsk

        parsed = read_meqdsk(path)
        attributes = dict(parsed.attributes)
        for key in tuple(attributes):
            lowered = key.lower()
            field_name = (
                "efithash"
                if lowered == "efithash"
                else (
                    "source_revision"
                    if lowered in {"commit", "git_commit", "source_revision"}
                    else ("run_date" if "date" in lowered else None)
                )
            )
            if field_name in allowed_metadata:
                removed[f"{field_name}:global:{key}"] = attributes.pop(key)
        variables = {}
        for name, variable in sorted(parsed.variables.items()):
            variable_attributes = dict(variable.attributes)
            for key in tuple(variable_attributes):
                lowered = key.lower()
                field_name = (
                    "efithash"
                    if lowered == "efithash"
                    else (
                        "source_revision"
                        if lowered in {"commit", "git_commit", "source_revision"}
                        else ("run_date" if "date" in lowered else None)
                    )
                )
                if field_name in allowed_metadata:
                    removed[f"{field_name}:variable:{name}:{key}"] = (
                        variable_attributes.pop(key)
                    )
            variables[name] = {
                "dimensions": variable.dimensions,
                "attributes": variable_attributes,
                "data": variable.data,
            }
        payload = {
            "dimensions": parsed.dimensions,
            "attributes": attributes,
            "variables": variables,
        }
    else:
        raise ValueError(f"unknown EFIT output kind {kind!r}")
    canonical = _canonical_provenance_value(payload)
    return _canonical_sha({"scientific_output": canonical}), removed


def _validate_same_metadata_regression(
    proof: Any, build_record: Mapping[str, Any]
) -> None:
    """Validate the pre-commit raw-byte no-op instrumentation proof."""
    if not isinstance(proof, Mapping):
        raise ValueError("same-metadata diagnostics-off regression proof is missing")
    if proof.get("reference_slice_count") != 5 or proof.get("outcomes_unchanged") is not True:
        raise ValueError("same-metadata regression must cover five unchanged outcomes")
    for name in ("build_type", "compiler", "compiler_version", "cmake_options"):
        if proof.get(name) != build_record.get(name):
            raise ValueError(f"same-metadata regression does not bind {name}")
    for name in ("control_source_state_sha256", "instrumented_source_state_sha256"):
        if re.fullmatch(r"[0-9a-f]{64}", str(proof.get(name, ""))) is None:
            raise ValueError(f"same-metadata regression has invalid {name}")
    if proof.get("control_source_revision") != "4d10ed592f8c9d295d393d0cf331f2d8f6be3034":
        raise ValueError("same-metadata control is not the exact 4d10ed5 source")
    if proof.get("instrumented_source_base_revision") != proof.get(
        "control_source_revision"
    ):
        raise ValueError("same-metadata instrumentation is not based on the control source")
    if proof["control_source_state_sha256"] == proof["instrumented_source_state_sha256"]:
        raise ValueError("same-metadata instrumentation source state is not distinct")
    control_executable_hash = _validated_provenance_artifact(
        proof.get("control_executable"), label="same-metadata control executable"
    )
    instrumented_executable_hash = _validated_provenance_artifact(
        proof.get("instrumented_executable"),
        label="same-metadata instrumented executable",
    )
    if (
        control_executable_hash == instrumented_executable_hash
        or Path(proof["control_executable"]["path"]).resolve()
        == Path(proof["instrumented_executable"]["path"]).resolve()
    ):
        raise ValueError("same-metadata control and instrumented executables are not distinct")
    pairs = proof.get("output_pairs")
    if not isinstance(pairs, Sequence) or isinstance(pairs, (str, bytes)):
        raise ValueError("same-metadata regression output pairs are missing")
    expected = {
        (reference.key, kind)
        for reference in REFERENCE_SLICES
        for kind in ("gfile", "afile", "mfile")
    }
    observed: set[tuple[str, str]] = set()
    for index, item in enumerate(pairs):
        if not isinstance(item, Mapping):
            raise ValueError("same-metadata output-pair record is malformed")
        key = (str(item.get("reference")), str(item.get("kind")))
        if key in observed:
            raise ValueError(f"same-metadata regression duplicates {key[0]}/{key[1]}")
        observed.add(key)
        control_hash = _validated_provenance_artifact(
            item.get("control"), label=f"same-metadata pair {index} control"
        )
        instrumented_hash = _validated_provenance_artifact(
            item.get("instrumented"), label=f"same-metadata pair {index} instrumented"
        )
        if item.get("bit_identical") is not True or control_hash != instrumented_hash:
            raise ValueError(
                f"same-metadata regression {key[0]}/{key[1]} is not byte-identical"
            )
        if Path(item["control"]["path"]).resolve() == Path(
            item["instrumented"]["path"]
        ).resolve():
            raise ValueError("same-metadata output pair is a self-comparison")
        expected_runs = {
            "control_run": (
                control_executable_hash,
                proof["control_source_state_sha256"],
            ),
            "instrumented_run": (
                instrumented_executable_hash,
                proof["instrumented_source_state_sha256"],
            ),
        }
        for run_name, (expected_executable, expected_source_state) in expected_runs.items():
            run = item.get(run_name)
            if not isinstance(run, Mapping) or (
                run.get("executable_sha256") != expected_executable
                or run.get("source_state_sha256") != expected_source_state
            ):
                raise ValueError(
                    f"same-metadata {key[0]}/{key[1]} does not bind {run_name}"
                )
    if observed != expected:
        raise ValueError("same-metadata regression does not contain all 15 output pairs")


def _validate_final_clean_regression(
    proof: Any, build_record: Mapping[str, Any]
) -> None:
    """Validate final clean output after narrowly defined metadata canonicalization."""
    if not isinstance(proof, Mapping):
        raise ValueError("final-clean commit validation is missing")
    if (
        proof.get("reference_slice_count") != 5
        or proof.get("outcomes_unchanged") is not True
        or proof.get("printed_physics_equal") is not True
    ):
        raise ValueError("final-clean validation must cover five unchanged physics outcomes")
    if str(proof.get("source_revision", "")).lower() != str(
        build_record.get("source_revision", "")
    ).lower():
        raise ValueError("final-clean validation source revision differs")
    executable_hash = _validated_provenance_artifact(
        proof.get("instrumented_executable"), label="final-clean executable"
    )
    if executable_hash != build_record.get("executable", {}).get("sha256"):
        raise ValueError("final-clean validation executable differs from the study binary")
    allowed_universe = {"run_date", "source_revision", "efithash"}
    allowed = proof.get("allowed_metadata_differences")
    if (
        not isinstance(allowed, Sequence)
        or isinstance(allowed, (str, bytes))
        or len(allowed) != len(set(allowed))
        or not set(allowed).issubset(allowed_universe)
    ):
        raise ValueError("final-clean metadata-difference allowlist is invalid")

    pairs = proof.get("output_pairs")
    if not isinstance(pairs, Sequence) or isinstance(pairs, (str, bytes)):
        raise ValueError("final-clean output pairs are missing")
    expected = {
        (reference.key, kind)
        for reference in REFERENCE_SLICES
        for kind in ("gfile", "afile", "mfile")
    }
    observed: set[tuple[str, str]] = set()
    used_differences: set[str] = set()
    for index, item in enumerate(pairs):
        if not isinstance(item, Mapping):
            raise ValueError("final-clean output-pair record is malformed")
        key = (str(item.get("reference")), str(item.get("kind")))
        if key in observed:
            raise ValueError(f"final-clean validation duplicates {key[0]}/{key[1]}")
        observed.add(key)
        control_record = item.get("control")
        instrumented_record = item.get("instrumented")
        _validated_provenance_artifact(
            control_record, label=f"final-clean pair {index} control"
        )
        _validated_provenance_artifact(
            instrumented_record, label=f"final-clean pair {index} instrumented"
        )
        normalized_control, control_removed = _scientific_output_identity(
            Path(control_record["path"]), key[1], allowed_metadata=set(allowed)
        )
        if Path(control_record["path"]).resolve() == Path(
            instrumented_record["path"]
        ).resolve():
            raise ValueError("final-clean output pair is a self-comparison")
        normalized_instrumented, instrumented_removed = _scientific_output_identity(
            Path(instrumented_record["path"]), key[1], allowed_metadata=set(allowed)
        )
        if normalized_control != normalized_instrumented:
            raise ValueError(
                f"final-clean {key[0]}/{key[1]} scientific payload differs"
            )
        if item.get("normalized_control_sha256") not in (None, normalized_control):
            raise ValueError("final-clean declared control canonical SHA-256 is false")
        if item.get("normalized_instrumented_sha256") not in (
            None,
            normalized_instrumented,
        ):
            raise ValueError("final-clean declared instrumented canonical SHA-256 is false")
        actual_fields = {
            location.split(":", 1)[0]
            for location in set(control_removed) | set(instrumented_removed)
            if control_removed.get(location) != instrumented_removed.get(location)
        }
        differences = item.get("metadata_differences", ())
        if not isinstance(differences, Sequence) or isinstance(differences, (str, bytes)):
            raise ValueError("final-clean machine-readable metadata diff is malformed")
        declared_fields: set[str] = set()
        for difference in differences:
            if not isinstance(difference, Mapping):
                raise ValueError("final-clean metadata-difference record is malformed")
            field_name = str(difference.get("field", ""))
            if field_name not in set(allowed):
                raise ValueError(
                    f"final-clean output contains unapproved difference {field_name!r}"
                )
            declared_fields.add(field_name)
        if declared_fields != actual_fields:
            raise ValueError(
                f"final-clean {key[0]}/{key[1]} metadata diff is not derived from artifacts"
            )
        used_differences.update(actual_fields)
    if observed != expected:
        raise ValueError("final-clean validation does not contain all 15 output pairs")
    if used_differences != set(allowed):
        raise ValueError("final-clean metadata allowlist is not exactly supported by its diff")

    sidecars = proof.get("sidecars")
    if not isinstance(sidecars, Sequence) or isinstance(sidecars, (str, bytes)):
        raise ValueError("final-clean sidecar identities are missing")
    sidecar_references: set[str] = set()
    for index, item in enumerate(sidecars):
        if not isinstance(item, Mapping):
            raise ValueError("final-clean sidecar identity is malformed")
        reference = str(item.get("reference"))
        if reference in sidecar_references:
            raise ValueError(f"final-clean sidecar duplicates {reference}")
        sidecar_references.add(reference)
        _validated_provenance_artifact(
            item, label=f"final-clean sidecar {index}"
        )
        if (
            str(item.get("source_revision", "")).lower()
            != str(build_record["source_revision"]).lower()
            or item.get("executable_sha256")
            != build_record["executable"]["sha256"]
        ):
            raise ValueError(f"final-clean sidecar {reference} has stale native identity")
    if sidecar_references != {reference.key for reference in REFERENCE_SLICES}:
        raise ValueError("final-clean validation does not bind all five native sidecars")


def _validate_archived_issue_663_regression(proof: Any) -> None:
    """Derive the archived-#663 scientific equivalence from bound output files."""
    if not isinstance(proof, Mapping):
        raise ValueError("archived #663 comparison is missing")
    if (
        proof.get("reference_slice_count") != 5
        or proof.get("printed_precision_equal") is not True
        or proof.get("outcomes_unchanged") is not True
    ):
        raise ValueError("archived #663 comparison does not cover five matching outcomes")
    pairs = proof.get("output_pairs")
    if not isinstance(pairs, Sequence) or isinstance(pairs, (str, bytes)):
        raise ValueError("archived #663 output pairs are missing")
    expected = {
        (reference.key, kind)
        for reference in REFERENCE_SLICES
        for kind in ("gfile", "afile", "mfile")
    }
    observed: set[tuple[str, str]] = set()
    allowed = {"run_date", "source_revision", "efithash"}
    for index, item in enumerate(pairs):
        if not isinstance(item, Mapping):
            raise ValueError("archived #663 output-pair record is malformed")
        key = (str(item.get("reference")), str(item.get("kind")))
        if key in observed:
            raise ValueError(f"archived #663 output pair duplicates {key[0]}/{key[1]}")
        observed.add(key)
        archived = item.get("archived")
        control = item.get("control")
        _validated_provenance_artifact(
            archived, label=f"archived #663 pair {index} archived"
        )
        _validated_provenance_artifact(
            control, label=f"archived #663 pair {index} control"
        )
        archived_identity, _ = _scientific_output_identity(
            Path(archived["path"]), key[1], allowed_metadata=allowed
        )
        if Path(archived["path"]).resolve() == Path(control["path"]).resolve():
            raise ValueError("archived #663 output pair is a self-comparison")
        control_identity, _ = _scientific_output_identity(
            Path(control["path"]), key[1], allowed_metadata=allowed
        )
        if archived_identity != control_identity:
            raise ValueError(
                f"archived #663 {key[0]}/{key[1]} scientific payload differs"
            )
    if observed != expected:
        raise ValueError("archived #663 comparison does not bind all 15 output pairs")


def validate_build_provenance(record: Mapping[str, Any]) -> None:
    """Fail execution when the clean build cannot be independently identified."""
    required = {
        "source_base_revision",
        "source_base_is_ancestor",
        "source_revision",
        "source_dirty",
        "build_type",
        "compiler",
        "compiler_version",
        "cmake_options",
        "install_manifest",
        "install_manifest_sha256",
        "ctest",
        "control_comparison",
        "executable",
    }
    missing = sorted(required - set(record))
    if missing:
        raise ValueError(f"build provenance is missing: {', '.join(missing)}")
    if record["source_dirty"] is not False:
        raise ValueError("the #664 native study requires a clean EFIT source worktree")
    if str(record["source_base_revision"]).lower() != (
        "4d10ed592f8c9d295d393d0cf331f2d8f6be3034"
    ):
        raise ValueError("control provenance must identify EFIT base commit 4d10ed5")
    if record["source_base_is_ancestor"] is not True:
        raise ValueError("EFIT commit 4d10ed5 must be an ancestor of the instrumented source")
    install_manifest_sha256 = _validated_provenance_artifact(
        record["install_manifest"], label="EFIT install manifest"
    )
    if install_manifest_sha256 != record["install_manifest_sha256"]:
        raise ValueError("install_manifest_sha256 does not match the bound artifact")
    source_revision = str(record["source_revision"]).lower()
    if re.fullmatch(r"(?:[0-9a-f]{40}|[0-9a-f]{64})", source_revision) is None:
        raise ValueError("instrumented EFIT source revision must be a full commit hash")
    ctest = record["ctest"]
    if not isinstance(ctest, Mapping):
        raise ValueError("control provenance must record both control and instrumented EFIT CTests")
    control = ctest.get("control")
    instrumented = ctest.get("instrumented")
    if not isinstance(control, Mapping) or not isinstance(instrumented, Mapping):
        raise ValueError("control provenance must record both control and instrumented EFIT CTests")
    try:
        control_passed = int(control["passed"])
        control_total = int(control["total"])
        instrumented_passed = int(instrumented["passed"])
        instrumented_total = int(instrumented["total"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("EFIT CTest counts are incomplete") from exc
    control_failures = control_total - control_passed
    instrumented_failures = instrumented_total - instrumented_passed
    recorded_additional = ctest.get("additional_failures")
    if (
        control_total != instrumented_total
        or control_total < 44
        or control_passed < 42
        or instrumented_passed < 42
        or instrumented_failures > control_failures
        or recorded_additional not in (None, instrumented_failures - control_failures)
    ):
        raise ValueError("instrumentation must introduce no additional EFIT CTest failures")
    comparison = record["control_comparison"]
    if not isinstance(comparison, Mapping):
        raise ValueError("five-slice control comparison must record unchanged physics")
    issue_663 = comparison.get("issue_663_baseline")
    same_metadata = comparison.get("same_metadata_default_off_regression")
    final_clean = comparison.get("final_clean_commit_validation")
    _validate_archived_issue_663_regression(issue_663)
    _validate_same_metadata_regression(same_metadata, record)
    _validate_final_clean_regression(final_clean, record)


def _profile_support() -> Any:
    name = "efit_identifiability_profile_support"
    if name in sys.modules:
        return sys.modules[name]
    specification = importlib.util.spec_from_file_location(name, PROFILE_STUDY)
    if specification is None or specification.loader is None:
        raise RuntimeError(f"cannot load profile comparison support from {PROFILE_STUDY}")
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    specification.loader.exec_module(module)
    return module


def _constraint_support() -> Any:
    name = "efit_identifiability_constraint_support"
    if name in sys.modules:
        return sys.modules[name]
    specification = importlib.util.spec_from_file_location(name, CONSTRAINT_STUDY)
    if specification is None or specification.loader is None:
        raise RuntimeError(f"cannot load constraint-study support from {CONSTRAINT_STUDY}")
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    specification.loader.exec_module(module)
    return module


def equilibrium_snapshot(gfile: Path, afile: Path, retained_mask: Sequence[bool]) -> dict[str, Any]:
    """Read the geometry, global values, profiles, and rank mask for one case."""
    support = _profile_support()
    return {
        "gfile": support._gfile_record(Path(gfile)),
        "afile": support._afile_record(Path(afile)),
        "retained_mask": [bool(value) for value in retained_mask],
        "gfile_sha256": sha256_file(Path(gfile)),
        "afile_sha256": sha256_file(Path(afile)),
    }


def native_row_records(problem: Any, reference: ReferenceSlice) -> list[dict[str, Any]]:
    """Flatten native row metadata without losing physical/solver semantics."""
    records: list[dict[str, Any]] = []
    for block_name in ("main", "external_current"):
        block = getattr(problem, block_name, None)
        if block is None:
            continue
        calculated = np.asarray(block.solver_residual, dtype=float)
        exported = np.asarray(block.row_solver_weighted_residual, dtype=float)
        solver_residual = np.where(np.isfinite(exported), exported, calculated)
        for index in range(block.nrow):
            uncertainty = float(block.row_uncertainty[index])
            physical_residual = float(block.row_physical_residual[index])
            statistical = bool(block.row_statistical[index])
            diagnostic_chi2 = (
                (physical_residual / uncertainty) ** 2
                if statistical
                and math.isfinite(physical_residual)
                and math.isfinite(uncertainty)
                and uncertainty > 0.0
                else None
            )
            records.append(
                {
                    "reference": reference.key,
                    "block": block_name,
                    "index": index,
                    "family": str(block.row_family[index]),
                    "kind": str(block.row_kind[index]),
                    "channel": int(block.row_channel[index]),
                    "statistical": statistical,
                    "measurement": float(block.row_measurement[index]),
                    "reconstruction": float(block.row_reconstruction[index]),
                    "physical_residual": physical_residual,
                    "uncertainty": uncertainty,
                    "submitted_fwt": float(block.row_submitted_fwt[index]),
                    "processed_weight": float(block.row_processed_weight[index]),
                    "solver_weighted_residual": float(solver_residual[index]),
                    "diagnostic_chi2": diagnostic_chi2,
                    "solver_objective": float(solver_residual[index] ** 2),
                }
            )
    return _json_safe(records)


def compare_mfile_family_chi2(
    rows: Sequence[Mapping[str, Any]],
    mfile_sums: Mapping[str, float],
    *,
    thresholds: GateThresholds = GateThresholds(),
) -> dict[str, Any]:
    """Compare native statistical rows with EFIT's persisted m-file totals.

    The Ip solve row is deliberately absent: ``chipasma`` includes fixed
    ``VCURRT`` and is validated by the separate accounting identity.  Soft PF
    relation rows are likewise absent because they have neither a statistical
    uncertainty nor an m-file diagnostic-chi-squared array.
    """
    family_records: dict[str, dict[str, Any]] = {}
    for family in MFILE_DIAGNOSTIC_CHI2_VARIABLES:
        native = float(
            np.sum(
                [
                    float(row["diagnostic_chi2"])
                    for row in rows
                    if row.get("block") == "main"
                    and row.get("family") == family
                    and bool(row.get("statistical"))
                    and row.get("diagnostic_chi2") is not None
                ],
                dtype=np.float64,
            )
        )
        persisted = float(mfile_sums[family])
        absolute_error = abs(native - persisted)
        relative_error = absolute_error / max(
            abs(persisted), np.finfo(float).tiny
        )
        passed = bool(
            absolute_error <= thresholds.mfile_family_chi2_absolute_error
            or relative_error <= thresholds.mfile_family_chi2_relative_error
        )
        family_records[family] = {
            "native_diagnostic_chi2": native,
            "mfile_diagnostic_chi2": persisted,
            "absolute_error": absolute_error,
            "relative_error": relative_error,
            "passed": passed,
        }
    return {
        "families": family_records,
        "excluded": {
            "plasma_current": (
                "validated by the separate Ip/VCURRT accounting identity; "
                "chipasma is not the minimized Ip solve row"
            ),
            "pf_relation": "soft non-statistical rows have no m-file chi-squared array",
        },
        "relative_tolerance": thresholds.mfile_family_chi2_relative_error,
        "absolute_tolerance": thresholds.mfile_family_chi2_absolute_error,
        "maximum_relative_error": max(
            (item["relative_error"] for item in family_records.values()),
            default=0.0,
        ),
        "maximum_absolute_error": max(
            (item["absolute_error"] for item in family_records.values()),
            default=0.0,
        ),
        "passed": all(item["passed"] for item in family_records.values()),
    }


def audit_mfile_family_chi2(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
    *,
    thresholds: GateThresholds = GateThresholds(),
) -> dict[str, Any]:
    """Read one EFIT m-file and gate all comparable family chi-squared sums."""
    from vaft.data import read_meqdsk

    data = read_meqdsk(path)
    sums: dict[str, float] = {}
    missing: list[str] = []
    for family, variable_name in MFILE_DIAGNOSTIC_CHI2_VARIABLES.items():
        if variable_name not in data:
            missing.append(f"{family}:{variable_name}")
            continue
        values = np.asarray(data[variable_name].data, dtype=np.float64)
        sums[family] = float(np.nansum(values))
    if missing:
        raise ValueError(
            "m-file is missing diagnostic chi-squared arrays: " + ", ".join(missing)
        )
    audit = compare_mfile_family_chi2(rows, sums, thresholds=thresholds)
    audit["path"] = str(Path(path).resolve())
    audit["sha256"] = sha256_file(Path(path))
    return audit


def summarize_native_rows(
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Summarize residual, weight, and objective activity by block/family."""
    groups: dict[tuple[str, str, str], list[Mapping[str, Any]]] = {}
    for row in rows:
        key = (str(row["reference"]), str(row["block"]), str(row["family"]))
        groups.setdefault(key, []).append(row)
    summaries: list[dict[str, Any]] = []
    for (reference, block, family), selected in sorted(groups.items()):
        residual = np.asarray(
            [row.get("physical_residual", np.nan) for row in selected], dtype=float
        )
        finite_residual = residual[np.isfinite(residual)]
        processed = np.asarray(
            [row.get("processed_weight", np.nan) for row in selected], dtype=float
        )
        uncertainty = np.asarray(
            [row.get("uncertainty", np.nan) for row in selected], dtype=float
        )
        submitted = np.asarray(
            [row.get("submitted_fwt", np.nan) for row in selected], dtype=float
        )
        active = np.isfinite(processed) & (processed != 0.0)
        diagnostic = np.asarray(
            [row.get("diagnostic_chi2", np.nan) for row in selected], dtype=float
        )
        objective = np.asarray(
            [row.get("solver_objective", np.nan) for row in selected], dtype=float
        )

        def spread(values: np.ndarray) -> dict[str, float | None]:
            finite = values[np.isfinite(values)]
            return {
                "min": float(np.min(finite)) if finite.size else None,
                "median": float(np.median(finite)) if finite.size else None,
                "max": float(np.max(finite)) if finite.size else None,
            }

        summaries.append(
            {
                "reference": reference,
                "block": block,
                "family": family,
                "row_count": len(selected),
                "active_channel_count": int(np.count_nonzero(active)),
                "residual_bias": (
                    float(np.mean(finite_residual)) if finite_residual.size else None
                ),
                "residual_rms": (
                    float(np.sqrt(np.mean(finite_residual**2)))
                    if finite_residual.size
                    else None
                ),
                "median_absolute_residual": (
                    float(np.median(np.abs(finite_residual)))
                    if finite_residual.size
                    else None
                ),
                "maximum_absolute_residual": (
                    float(np.max(np.abs(finite_residual)))
                    if finite_residual.size
                    else None
                ),
                "diagnostic_chi2": (
                    float(np.nansum(diagnostic))
                    if np.any(np.isfinite(diagnostic))
                    else None
                ),
                "solver_objective": float(np.nansum(objective)),
                "uncertainty": spread(uncertainty),
                "submitted_fwt": spread(submitted),
                "processed_weight": spread(processed),
            }
        )
    return summaries


def compare_snapshots(
    candidate: Mapping[str, Any], baseline: Mapping[str, Any]
) -> tuple[TransitionMetrics, bool]:
    """Return displacement inputs, failing closed on incomplete physics output."""
    support = _profile_support()
    try:
        left_g, right_g = candidate["gfile"], baseline["gfile"]
        left_a = candidate["afile"]["scalars"]
        right_a = baseline["afile"]["scalars"]
    except (KeyError, TypeError) as exc:
        raise ValueError("snapshot is missing g-file/a-file physics data") from exc

    for label, boundary_record in (
        ("candidate", left_g.get("boundary")),
        ("baseline", right_g.get("boundary")),
    ):
        if not isinstance(boundary_record, Mapping):
            raise ValueError(f"{label} snapshot is missing its LCFS boundary")
        try:
            r = np.asarray(boundary_record["r"], dtype=float).reshape(-1)
            z = np.asarray(boundary_record["z"], dtype=float).reshape(-1)
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"{label} snapshot has an invalid LCFS boundary") from exc
        if r.size < 3 or r.size != z.size or not np.all(np.isfinite(r)) or not np.all(np.isfinite(z)):
            raise ValueError(
                f"{label} snapshot LCFS must contain at least three finite R/Z pairs"
            )
    boundary = support._curve_distance(left_g["boundary"], right_g["boundary"])
    if (
        boundary is None
        or not math.isfinite(float(boundary.get("rms_m", math.nan)))
    ):
        raise ValueError("snapshot LCFS comparison is unavailable or nonfinite")

    def relative(name: str) -> float:
        try:
            before, after = float(right_a[name]), float(left_a[name])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"snapshot scalar {name!r} is missing or invalid") from exc
        if not math.isfinite(before) or not math.isfinite(after):
            raise ValueError(f"snapshot scalar {name!r} must be finite")
        if before == 0.0:
            return 0.0 if after == 0.0 else math.inf
        return (after - before) / abs(before)

    profiles = {}
    for name in ("pressure", "pprime", "ffprime", "jphi_reference_r", "q"):
        try:
            left_profile = np.asarray(left_g["profiles"][name], dtype=float).reshape(-1)
            right_profile = np.asarray(right_g["profiles"][name], dtype=float).reshape(-1)
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"snapshot profile {name!r} is missing or invalid") from exc
        start = 7 if name == "q" else 0
        count = min(left_profile.size, right_profile.size) - start
        if (
            count <= 0
            or not np.all(np.isfinite(left_profile[start : start + count]))
            or not np.all(np.isfinite(right_profile[start : start + count]))
        ):
            raise ValueError(f"snapshot profile {name!r} has no finite common support")
        value = support._relative_rms(
            left_profile,
            right_profile,
            start=start,
        )
        if value is None or not math.isfinite(float(value)):
            raise ValueError(f"snapshot profile {name!r} has no finite relative RMS")
        profiles[name] = float(value)
    left_mask = np.asarray(candidate.get("retained_mask", ()), dtype=bool)
    right_mask = np.asarray(baseline.get("retained_mask", ()), dtype=bool)
    rank_switch = bool(
        left_mask.shape != right_mask.shape
        or (left_mask.size and np.any(left_mask != right_mask))
    )
    return (
        TransitionMetrics(
            lcfs_rms_mm=float(boundary["rms_m"]) * 1000.0,
            area_relative=relative("area"),
            volume_relative=relative("volume"),
            beta_p_relative=relative("betap"),
            li_relative=relative("li"),
            profile_relative_rms=profiles,
        ),
        rank_switch,
    )


def _result_outcome(result: Any) -> tuple[str, Any | None]:
    statuses = tuple(result.slice_statuses)
    status = statuses[0] if len(statuses) == 1 else None
    if status is not None and status.usable:
        return "accepted", status
    if status is not None:
        return status.overall_status, status
    return ("runtime_failed" if not result.ok else "missing_slice_status"), None


def _native_restart_audit(stdout: str) -> dict[str, Any]:
    """Parse EFIT's V2 restart-state provenance from captured stdout."""
    state_matches = re.findall(
        r"EFIT_RESTART_STATE\s+physical_state_loaded=(\d+)\s+"
        r"signature_match=(\d+)\s+minite_iteration_offset=(-?\d+)",
        stdout or "",
    )
    fresh_matches = re.findall(
        r"EFIT_RESTART_HISTORY\s+fresh_response_accepted=1\s+"
        r"response_iteration=(-?\d+)",
        stdout or "",
    )
    objective_matches = re.findall(
        r"EFIT_RESTART_HISTORY\s+objective_reused=1", stdout or ""
    )
    if len(state_matches) > 1:
        raise ValueError("native output contains multiple restart-state records")
    state = state_matches[0] if state_matches else None
    return {
        "state_record_present": state is not None,
        "physical_state_loaded": bool(int(state[0])) if state else False,
        "signature_match": bool(int(state[1])) if state else None,
        "minite_iteration_offset": int(state[2]) if state else None,
        "objective_history_reused": bool(objective_matches),
        "objective_history_record_count": len(objective_matches),
        "fresh_response_accepted": bool(fresh_matches),
        "fresh_response_iterations": [int(value) for value in fresh_matches],
    }


def execute_nonlinear_case(
    spec: CaseSpec,
    *,
    output: Path,
    scientific: Any,
    snapshot_retained_mask: Sequence[bool] | None = None,
) -> dict[str, Any]:
    """Execute one hashed restart/direction case with no silent cold fallback."""
    from vaft.code.efit import (
        EFITConfig,
        EFITInputs,
        IdentifiabilityConfig,
        analyze_efit_identifiability,
        run_efit,
    )

    if scientific.sha256 != spec.scientific_sha256:
        raise ValueError("case scientific hash does not match its resolved configuration")
    case_id, identity = case_identity(spec)
    workdir = Path(output) / "stage2" / "cases" / case_id
    manifest = workdir / STAGE1_CASE_MANIFEST
    resumed = resumable_case_manifest(manifest, identity)
    if resumed is not None:
        return dict(resumed)
    workdir.mkdir(parents=True, exist_ok=True)
    pending = {
        "schema_version": SCHEMA_VERSION,
        "issue": ISSUE,
        "case_id": case_id,
        "status": "running",
        "stage": 2,
        "reference": asdict(spec.reference),
        "kind": spec.kind,
        "workdir": str(workdir),
        "identity": identity,
        "started_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    write_json_atomic(manifest, pending)
    try:
        config = EFITConfig(
            executable=str(spec.executable),
            workdir=workdir,
            shot=spec.reference.shot,
            times=(spec.reference.time_s,),
            args=("129",),
            profile=scientific.profile,
            initialization=scientific.initialization,
            numerics=scientific.numerics,
            constraints=scientific.constraints,
            export_linearization=spec.require_linearization,
            direction_constraint=spec.direction_file,
            restart_from=spec.restart_parent,
            write_restart=True,
            provenance={
                "study": ISSUE,
                "stage": 2,
                "case_identity": case_id,
                "parent_case_id": spec.parent_case_id,
                "chain": spec.chain,
                "target": spec.target,
                "build": identity["build_provenance"],
                "table": identity["table_identity"],
                "input": {
                    "kfile": str(spec.kfile.resolve()),
                    "kfile_sha256": identity["kfile_sha256"],
                    "kfile_semantic_audit": identity["kfile_semantic_audit"],
                    "restart_sha256": identity["restart_parent_sha256"],
                    "direction_sha256": identity["direction_sha256"],
                },
            },
        )
        inputs = EFITInputs(
            workdir=workdir,
            kfiles=(spec.kfile,),
            files=(spec.kfile,),
            configuration={"source": "content_addressed_stage2_case"},
        )
        result = run_efit(inputs, config)
        restart_audit = _native_restart_audit(result.stdout)
        outcome, slice_status = _result_outcome(result)
        errors = list(result.diagnostic_errors)
        if spec.restart_parent is None:
            if restart_audit["physical_state_loaded"] or restart_audit[
                "objective_history_reused"
            ]:
                errors.append("cold case unexpectedly reports restored restart state")
        elif spec.kind == "restart_proof_unchanged_restart":
            if not (
                restart_audit["physical_state_loaded"]
                and restart_audit["signature_match"] is True
                and restart_audit["objective_history_reused"]
            ):
                errors.append(
                    "unchanged restart did not prove physical load, signature match, "
                    "and objective-history reuse"
                )
        elif outcome == "accepted" and not (
            restart_audit["physical_state_loaded"]
            and restart_audit["signature_match"] is False
            and not restart_audit["objective_history_reused"]
            and restart_audit["fresh_response_accepted"]
        ):
            errors.append(
                "changed-control warm case did not prove physical restart load, "
                "history reset, and a fresh accepted response"
            )
        problem = result.linearizations[0] if len(result.linearizations) == 1 else None
        if problem is None and spec.require_linearization:
            errors.append(
                f"expected exactly one native sidecar, found {len(result.linearizations)}"
            )
        elif problem is None and len(result.linearizations) > 1:
            errors.append(
                f"expected at most one optional native sidecar, found {len(result.linearizations)}"
            )
        snapshot = analysis = external_analysis = parameter_state = None
        exact_constraint_count = None
        linear_solve_validation = None
        reduced_linear_solve_validation = None
        solver_method = None
        linearization_attributes = None
        sidecar_restart_audit = None
        native_rows: list[dict[str, Any]] = []
        if problem is not None:
            analysis = analyze_efit_identifiability(problem).to_dict()
            parameter_state = np.asarray(problem.final_brsp, dtype=float).tolist()
            exact_constraint_count = problem.main.nexact
            linear_solve_validation = asdict(problem.main.validation)
            solver_method = problem.main.solver_method
            if problem.main.reduced_solve is not None:
                reduced = problem.main.reduced_solve
                reduced_linear_solve_validation = {
                    "status": reduced.status,
                    "condno": reduced.condno,
                    "discarded_projection_norm": reduced.discarded_projection_norm,
                    "transform_relative_error": reduced.transform_relative_error,
                    "reproduction_relative_error": reduced.reproduction_relative_error,
                    "basis_kind": np.asarray(reduced.basis_kind, dtype=int).tolist(),
                }
            linearization_attributes = _json_safe(problem.attributes)
            restart_attribute_names = {
                "physical_state_loaded": "restart_physical_state_loaded",
                "signature_match": "restart_control_signature_match",
                "objective_history_reused": "restart_objective_history_reused",
                "fresh_response_accepted": "restart_new_response_accepted",
                "minite_iteration_offset": "restart_minite_iteration_offset",
            }
            if not all(name in problem.attributes for name in restart_attribute_names.values()):
                errors.append("native sidecar is missing restart provenance attributes")
            else:
                sidecar_restart_audit = {
                    key: (
                        int(problem.attributes[name])
                        if key == "minite_iteration_offset"
                        else bool(int(problem.attributes[name]))
                    )
                    for key, name in restart_attribute_names.items()
                }
                expected_sidecar_restart = {
                    "physical_state_loaded": spec.restart_parent is not None,
                    "signature_match": False,
                    "objective_history_reused": False,
                    "fresh_response_accepted": True,
                    "minite_iteration_offset": (
                        sidecar_restart_audit["minite_iteration_offset"]
                        if spec.restart_parent is not None
                        else 0
                    ),
                }
                if spec.kind == "restart_proof_unchanged_restart":
                    expected_sidecar_restart.update(
                        {"signature_match": True, "objective_history_reused": True}
                    )
                if sidecar_restart_audit != expected_sidecar_restart:
                    errors.append(
                        "native sidecar restart provenance differs from the case type"
                    )
                if spec.restart_parent is not None and sidecar_restart_audit[
                    "minite_iteration_offset"
                ] <= 0:
                    errors.append("warm sidecar has no positive restart iteration offset")
                for key in (
                    "physical_state_loaded",
                    "signature_match",
                    "objective_history_reused",
                    "fresh_response_accepted",
                ):
                    stdout_value = restart_audit.get(key)
                    # The native stdout restart records are intentionally
                    # emitted only when a restart file is consumed.  A cold
                    # solve still records its accepted fresh response in the
                    # sidecar, so absence of the restart-only stdout marker is
                    # not a contradiction for cold repeat-noise cases.
                    stdout_reports_key = (
                        spec.restart_parent is not None
                        or key != "fresh_response_accepted"
                    )
                    if (
                        stdout_reports_key
                        and
                        (key != "signature_match" or restart_audit["state_record_present"])
                        and stdout_value is not None
                        and stdout_value != sidecar_restart_audit[key]
                    ):
                        errors.append(
                            f"stdout/sidecar restart provenance differs for {key}"
                        )
            native_rows = native_row_records(problem, spec.reference)
            if problem.external_current is not None:
                external_analysis = analyze_efit_identifiability(
                    problem, IdentifiabilityConfig(block="external_current")
                ).to_dict()
        if len(result.gfiles) == 1 and len(result.afiles) == 1:
            retained_mask = (
                problem.main.retained_mask
                if problem is not None
                else snapshot_retained_mask
            )
            if retained_mask is None:
                errors.append(
                    "snapshot without a sidecar requires the baseline retained mask"
                )
            else:
                snapshot = equilibrium_snapshot(
                    result.gfiles[0], result.afiles[0], retained_mask
                )
        else:
            errors.append("expected exactly one g-file and one a-file")
        if result.restart_file is None:
            errors.append("requested restart file was not produced")
        artifacts = _artifact_records(
            (
                *result.gfiles,
                *result.afiles,
                *result.mfiles,
                *result.logs,
                *result.linearization_files,
                *((result.restart_file,) if result.restart_file is not None else ()),
            )
        )
        succeeded = outcome == "accepted" and snapshot is not None and not errors
        payload = {
            **pending,
            "status": "succeeded" if succeeded else "failed",
            "finished_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "outcome": outcome,
            "slice_status": slice_status.to_dict() if slice_status is not None else None,
            "diagnostic_errors": errors,
            "analysis": analysis,
            "external_current_analysis": external_analysis,
            "native_rows": native_rows,
            "family_residual_summary": summarize_native_rows(native_rows),
            "snapshot": snapshot,
            "parameter_state": parameter_state,
            "exact_constraint_count": exact_constraint_count,
            "linear_solve_validation": linear_solve_validation,
            "reduced_linear_solve_validation": reduced_linear_solve_validation,
            "solver_method": solver_method,
            "linearization_attributes": linearization_attributes,
            "restart_control_sha256": result.configuration.get("execution", {}).get(
                "restart_control_sha256"
            ),
            "native_restart_audit": restart_audit,
            "sidecar_restart_audit": sidecar_restart_audit,
            "artifacts": artifacts,
            "linearization_file": (
                str(result.linearization_files[0])
                if len(result.linearization_files) == 1
                else None
            ),
            "restart_file": str(result.restart_file) if result.restart_file else None,
        }
    except Exception as exc:
        payload = {
            **pending,
            "status": "failed",
            "finished_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "outcome": "exception",
            "diagnostic_errors": [f"{type(exc).__name__}: {exc}"],
            "analysis": None,
            "external_current_analysis": None,
            "native_rows": [],
            "family_residual_summary": [],
            "snapshot": None,
            "parameter_state": None,
            "exact_constraint_count": None,
            "linear_solve_validation": None,
            "reduced_linear_solve_validation": None,
            "solver_method": None,
            "linearization_attributes": None,
            "restart_control_sha256": None,
            "native_restart_audit": None,
            "sidecar_restart_audit": None,
            "artifacts": [],
            "linearization_file": None,
            "restart_file": None,
        }
    write_json_atomic(manifest, payload)
    return payload


def fixed_column_scaling(
    matrix: np.ndarray, *, return_unobservable: bool = False
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Return deterministic full-baseline column equilibration.

    Structurally zero columns keep unit scale and are returned in an explicit
    unobservable mask when requested; inventing a scale from other columns
    would conceal the lack of information.
    """
    matrix = np.asarray(matrix, dtype=float)
    if matrix.ndim != 2 or matrix.shape[1] == 0:
        raise ValueError("matrix must be a non-empty two-dimensional array")
    norms = np.linalg.norm(matrix, axis=0)
    if not np.isfinite(norms).all():
        raise ValueError("matrix column norms must be finite")
    unobservable = norms == 0.0
    scaling = np.ones_like(norms)
    np.divide(1.0, norms, out=scaling, where=~unobservable)
    return (scaling, unobservable) if return_unobservable else scaling


def null_space(matrix: np.ndarray, *, rtol: float | None = None) -> np.ndarray:
    """Return an orthonormal basis for the null space without forming C^T C."""
    matrix = np.asarray(matrix, dtype=float)
    if matrix.ndim != 2:
        raise ValueError("constraint matrix must be two-dimensional")
    if matrix.shape[0] == 0:
        return np.eye(matrix.shape[1])
    _, singular, vh = np.linalg.svd(matrix, full_matrices=True)
    cutoff = (
        np.finfo(float).eps * max(matrix.shape) * (singular[0] if singular.size else 0.0)
        if rtol is None
        else float(rtol) * (singular[0] if singular.size else 1.0)
    )
    rank = int(np.count_nonzero(singular > cutoff))
    return vh[rank:].T.copy()


def mode_family_shares(
    matrix: np.ndarray,
    row_families: Sequence[str],
    scaling: np.ndarray,
    null_basis: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """SVD of ASZ plus normalized per-mode family contributions."""
    matrix = np.asarray(matrix, dtype=float)
    scaling = np.asarray(scaling, dtype=float).reshape(-1)
    null_basis = np.asarray(null_basis, dtype=float)
    if matrix.shape[0] != len(row_families):
        raise ValueError("row_families length does not match matrix rows")
    if matrix.shape[1] != scaling.size or null_basis.shape[0] != scaling.size:
        raise ValueError("column scaling/null basis dimensions do not match matrix")
    projected = (matrix * scaling[np.newaxis, :]) @ null_basis
    _, singular, vh = np.linalg.svd(projected, full_matrices=False)
    shares: dict[str, np.ndarray] = {}
    denominator = np.square(singular)
    for family in dict.fromkeys(row_families):
        mask = np.asarray([value == family for value in row_families])
        action = projected[mask] @ vh.T
        numerator = np.sum(np.square(action), axis=0)
        shares[family] = np.divide(
            numerator,
            denominator,
            out=np.zeros_like(numerator),
            where=denominator > 0.0,
        )
    return singular, vh, shares


def singular_class(singular_value: float, tau: float) -> str:
    if singular_value > 10.0 * tau:
        return "resolved"
    if singular_value < 0.1 * tau:
        return "unresolved"
    return "borderline"


def classify_family_information(metrics: Mapping[str, Any]) -> tuple[str, ...]:
    """Apply #664's independent/reinforcing/inactive labels deterministically."""
    rank_gains = tuple(int(value) for value in metrics.get("rank_gain_by_cutoff", ()))
    weak_action = float(metrics.get("weak_subspace_action_fraction", 0.0))
    lift = float(metrics.get("weak_singular_value_lift", 1.0))
    trace_reduction = float(metrics.get("curvature_inverse_trace_reduction", 0.0))
    independent = bool(rank_gains) and all(value >= 1 for value in rank_gains) and weak_action >= 0.1
    labels: list[str] = []
    if independent:
        labels.append("independent information")
    elif lift >= 10.0 or trace_reduction >= 0.20:
        labels.append("reinforcing")
    elif rank_gains and any(value >= 1 for value in rank_gains):
        labels.append("cutoff-sensitive/inconclusive")
    else:
        labels.append("inactive/redundant")
    if (
        float(metrics.get("baseline_solver_objective_share", 1.0)) < 0.01
        and bool(metrics.get("nonlinear_material_response", False))
    ):
        labels.append("overwhelmed")
    if bool(metrics.get("pf_structural_response", False)):
        labels.append("structural anchor")
    if bool(metrics.get("auxiliary_accounting_explains_reported_chi2", False)):
        labels.append("accounting-confounded")
    return tuple(labels)


def merge_nonlinear_family_evidence(
    payload: Mapping[str, Any],
    *,
    continuation_records: Sequence[Mapping[str, Any]] = (),
    prior_evidence: Mapping[str, Mapping[str, Mapping[str, Any]]] | None = None,
) -> dict[str, Any]:
    """Combine local information labels with provenance-qualified nonlinear evidence."""
    merged = dict(payload)
    evidence = prior_evidence or {}
    material_diamagnetic: set[str] = set()
    for item in continuation_records:
        if item.get("control") != "diamagnetic_flux_scale":
            continue
        if (
            bool(item.get("material_branch"))
            or float(item.get("displacement") or 0.0) >= 1.0
        ):
            material_diamagnetic.add(str(item.get("reference")))

    family_results = []
    for original in payload.get("family_results", ()):
        record = dict(original)
        reference = str(record.get("reference"))
        family = str(record.get("family"))
        prior = dict(evidence.get(reference, {}).get(family, {}))
        existing = {str(value).replace("_", " ") for value in record.get("classification", ())}
        prior_labels = {
            str(value).replace("_", " ") for value in prior.get("classification", ())
        }
        record["nonlinear_material_response"] = bool(
            (family == "diamagnetic_flux" and reference in material_diamagnetic)
            or prior.get("nonlinear_material_response")
            or "overwhelmed" in prior_labels
        )
        record["pf_structural_response"] = bool(
            prior.get("structural_effect")
            or "structural anchor" in prior_labels
            or "structural anchor" in existing
        )
        record["auxiliary_accounting_explains_reported_chi2"] = bool(
            prior.get("accounting_confounded")
            or "accounting-confounded" in prior_labels
            or "accounting confounded" in prior_labels
            or "accounting-confounded" in existing
            or "accounting confounded" in existing
        )
        if prior:
            record["nonlinear_evidence"] = prior
        record["classification"] = list(classify_family_information(record))
        family_results.append(record)
    merged["family_results"] = family_results
    return merged


def select_target_modes(
    *,
    singular_values: Sequence[float],
    parameter_vectors: np.ndarray,
    parameter_roles: Sequence[str],
    diamagnetic_shares: Sequence[float],
    tau: float,
    retained_mask: Sequence[bool] | None = None,
) -> tuple[TargetMode, ...]:
    """Pick profile, PF, and diamagnetic weak modes without duplicates."""
    singular = np.asarray(singular_values, dtype=float).reshape(-1)
    vectors = np.asarray(parameter_vectors, dtype=float)
    diamagnetic = np.asarray(diamagnetic_shares, dtype=float).reshape(-1)
    if vectors.shape != (singular.size, len(parameter_roles)):
        raise ValueError("parameter_vectors must have shape (modes, parameters)")
    if diamagnetic.size != singular.size:
        raise ValueError("diamagnetic_shares must have one value per mode")
    retained = (
        np.asarray(retained_mask, dtype=bool).reshape(-1)
        if retained_mask is not None
        else singular > tau
    )
    if retained.size != singular.size:
        raise ValueError("retained_mask must have one value per mode")
    weak = [index for index, value in enumerate(singular) if value <= 10.0 * tau]
    if not weak:
        raise ValueError("no weak or borderline modes are available for continuation")
    roles = np.asarray([str(role).lower() for role in parameter_roles], dtype=object)
    profile_mask = np.isin(roles, ("pprime", "p_prime", "ffprime", "ff_prime"))
    pf_mask = np.isin(roles, ("pf", "pf_current", "coil_current"))
    squared = np.square(vectors)
    norm = np.sum(squared, axis=1)
    profile = np.divide(
        np.sum(squared[:, profile_mask], axis=1), norm, out=np.zeros_like(norm), where=norm > 0
    )
    pf = np.divide(
        np.sum(squared[:, pf_mask], axis=1), norm, out=np.zeros_like(norm), where=norm > 0
    )
    scores = {
        "profile": profile,
        "pf_current": pf,
        "diamagnetic_flux": diamagnetic,
    }
    selected: list[TargetMode] = []
    used: set[int] = set()
    for selector in ("profile", "pf_current", "diamagnetic_flux"):
        candidates = sorted(
            weak,
            key=lambda index: (-float(scores[selector][index]), singular[index], index),
        )
        chosen = next((index for index in candidates if index not in used), None)
        if chosen is None:
            continue
        used.add(chosen)
        selected.append(
            TargetMode(
                mode_index=chosen,
                selector=selector,
                singular_value=float(singular[chosen]),
                threshold_class=singular_class(float(singular[chosen]), tau),
                retained=bool(retained[chosen]),
                profile_participation=float(profile[chosen]),
                pf_participation=float(pf[chosen]),
                diamagnetic_share=float(diamagnetic[chosen]),
            )
        )
    return tuple(selected)


def select_target_modes_from_report(report: Mapping[str, Any]) -> tuple[TargetMode, ...]:
    """Select continuation modes from ``IdentifiabilityReport.to_dict()``."""
    modes = tuple(report.get("modes", ()))
    candidates = [item for item in modes if item.get("state") in {"borderline", "unresolved"}]
    if not candidates:
        raise ValueError("report contains no weak or borderline modes")

    def score(item: Mapping[str, Any], selector: str) -> float:
        if selector == "profile":
            roles = item.get("parameter_role_shares", {})
            return float(roles.get("pprime", 0.0)) + float(roles.get("ffprime", 0.0))
        if selector == "pf_current":
            return float(item.get("parameter_role_shares", {}).get("pf_current", 0.0))
        return float(item.get("family_shares", {}).get("diamagnetic_flux", 0.0))

    selected: list[TargetMode] = []
    used: set[int] = set()
    nominal = float(report["nominal_cutoff"])
    for selector in ("profile", "pf_current", "diamagnetic_flux"):
        ordered = sorted(
            candidates,
            key=lambda item: (
                -score(item, selector),
                float(item["singular_value"]),
                int(item["index"]),
            ),
        )
        chosen = next((item for item in ordered if int(item["index"]) not in used), None)
        if chosen is None:
            continue
        index = int(chosen["index"])
        used.add(index)
        roles = chosen.get("parameter_role_shares", {})
        selected.append(
            TargetMode(
                mode_index=index,
                selector=selector,
                singular_value=float(chosen["singular_value"]),
                threshold_class=str(chosen["state"]),
                retained=float(chosen["singular_value"]) > nominal,
                profile_participation=(
                    float(roles.get("pprime", 0.0))
                    + float(roles.get("ffprime", 0.0))
                ),
                pf_participation=float(roles.get("pf_current", 0.0)),
                diamagnetic_share=float(
                    chosen.get("family_shares", {}).get("diamagnetic_flux", 0.0)
                ),
            )
        )
    return tuple(selected)


def diamagnetic_log10_schedule() -> tuple[float, ...]:
    """The required inclusive 0.25-dex path from 1 through 10,000."""
    return tuple(index / 4.0 for index in range(17))


def diamagnetic_scales() -> tuple[float, ...]:
    return tuple(10.0**value for value in diamagnetic_log10_schedule())


def directional_targets(mode: TargetMode) -> tuple[float, ...]:
    """Return independent negative/positive targets for one mode."""
    amplitudes = (0.125, 0.25, 0.5, 1.0, 2.0, 4.0)
    factor = 1.0 / mode.singular_value if mode.threshold_class == "resolved" else 1.0
    return tuple(-value * factor for value in amplitudes) + tuple(
        value * factor for value in amplitudes
    )


def directional_continuation_plan(mode: TargetMode) -> tuple[ContinuationPoint, ...]:
    """Create two independent warm-start chains rooted at the baseline."""
    points: list[ContinuationPoint] = []
    amplitudes = (0.125, 0.25, 0.5, 1.0, 2.0, 4.0)
    factor = 1.0 / mode.singular_value if mode.threshold_class == "resolved" else 1.0
    for sign, name in ((-1.0, "negative"), (1.0, "positive")):
        chain = f"mode_{mode.mode_index}_{name}"
        for order, amplitude in enumerate(amplitudes):
            points.append(
                ContinuationPoint(
                    chain=chain,
                    order=order,
                    control="direction",
                    requested_value=sign * amplitude * factor,
                    parent_order=order - 1 if order else None,
                    cold_start=False,
                    parent_chain=chain if order else "baseline",
                    mode_index=mode.mode_index,
                    alpha=sign * amplitude,
                )
            )
    return tuple(points)


def diamagnetic_continuation_plan() -> tuple[ContinuationPoint, ...]:
    """Create forward and reverse warm-start paths with explicit ancestry."""
    logs = diamagnetic_log10_schedule()
    points = []
    for order, exponent in enumerate(logs):
        points.append(
            ContinuationPoint(
                chain="diamagnetic_increasing",
                order=order,
                control="diamagnetic_flux_scale",
                requested_value=10.0**exponent,
                log10_scale=exponent,
                parent_order=order - 1 if order else None,
                cold_start=order == 0,
                parent_chain="diamagnetic_increasing" if order else None,
            )
        )
    for order, exponent in enumerate(reversed(logs)):
        points.append(
            ContinuationPoint(
                chain="diamagnetic_decreasing",
                order=order,
                control="diamagnetic_flux_scale",
                requested_value=10.0**exponent,
                log10_scale=exponent,
                parent_order=order - 1 if order else len(logs) - 1,
                # The reverse path starts from the increasing chain's verified
                # high endpoint.  The case identity later includes that exact
                # restart file's hash.
                cold_start=False,
                parent_chain=(
                    "diamagnetic_decreasing" if order else "diamagnetic_increasing"
                ),
            )
        )
    return tuple(points)


def bisection_points(
    lower_log10: float,
    upper_log10: float,
    *,
    maximum_steps: int = 16,
    target_width: float = 0.025,
) -> tuple[float, ...]:
    """Return deterministic midpoint proposals; outcomes choose each bracket."""
    if not lower_log10 < upper_log10:
        raise ValueError("bisection bounds must be increasing")
    if maximum_steps < 0 or target_width <= 0.0:
        raise ValueError("invalid bisection controls")
    points = []
    low, high = float(lower_log10), float(upper_log10)
    for _ in range(maximum_steps):
        if high - low <= target_width:
            break
        midpoint = 0.5 * (low + high)
        points.append(midpoint)
        # This is a proposal sequence, not an outcome-dependent solver.  The
        # caller replaces either bracket after each observed midpoint.
        high = midpoint
    return tuple(points)


_NAMELIST_REAL = re.compile(
    r"(?im)^(?P<prefix>\s*FWTDLC\s*=\s*)"
    r"(?P<value>[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eEdD][-+]?\d+)?)"
    r"(?P<suffix>\s*(?:,)?\s*)$"
)


def derive_diamagnetic_kfile(source: Path, destination: Path, scale: float) -> dict[str, Any]:
    """Copy a frozen k-file while changing exactly the FWTDLC scalar."""
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError("diamagnetic scale must be finite and greater than zero")
    source = Path(source)
    destination = Path(destination)
    original = source.read_text(encoding="utf-8")
    matches = tuple(_NAMELIST_REAL.finditer(original))
    if len(matches) != 1:
        raise ValueError(f"expected exactly one FWTDLC scalar in {source}, found {len(matches)}")
    match = matches[0]
    replacement = f"{match.group('prefix')}{scale:.17g}{match.group('suffix')}"
    derived = original[: match.start()] + replacement + original[match.end() :]
    # Prove that stripping the one allowed value returns identical source text.
    derived_matches = tuple(_NAMELIST_REAL.finditer(derived))
    if len(derived_matches) != 1:
        raise AssertionError("derived k-file no longer has exactly one FWTDLC scalar")
    derived_match = derived_matches[0]
    source_template = original[: match.start("value")] + "<FWTDLC>" + original[match.end("value") :]
    derived_template = (
        derived[: derived_match.start("value")]
        + "<FWTDLC>"
        + derived[derived_match.end("value") :]
    )
    if source_template != derived_template:
        raise AssertionError("derived k-file changed content outside FWTDLC")
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(dir=destination.parent, suffix=".tmp")
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(derived)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    except Exception:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise
    return {
        "source": str(source.resolve()),
        "source_sha256": sha256_file(source),
        "derived": str(destination.resolve()),
        "derived_sha256": sha256_file(destination),
        "field": "FWTDLC",
        "source_value": float(match.group("value").replace("d", "e").replace("D", "E")),
        "derived_value": float(scale),
    }


def scientific_with_diamagnetic_scale(scale: float) -> Any:
    """Return the frozen baseline with only submitted FWTDLC strength changed."""
    baseline = fixed_scientific_config()
    scales = dict(baseline.constraints.objective_scales)
    scales["diamagnetic_flux"] = float(scale)
    constraints = replace(baseline.constraints, group_weights={}, objective_scales=scales)
    return replace(baseline, constraints=constraints)


def parameter_order_sha256(block: Any) -> str:
    """Hash native parameter order, roles, indices, and units canonically."""
    payload = [
        {
            "name": str(name),
            "role": str(role),
            "index": int(index),
            "units": str(units),
        }
        for name, role, index, units in zip(
            block.parameter_name,
            block.parameter_role,
            block.parameter_index,
            block.parameter_units,
        )
    ]
    return _canonical_sha({"parameters": payload})


def write_direction_control(
    path: Path,
    *,
    problem: Any,
    report: Any,
    mode_index: int,
    target: float,
) -> dict[str, Any]:
    """Write ``efit_direction_constraint_v1`` as classic NetCDF.

    ``physical_c`` is redundant by design. Both writer and EFIT validate
    ``physical_c == mode / column_scale`` before the equality row is used.
    """
    if not math.isfinite(target):
        raise ValueError("direction target must be finite")
    block = problem.main
    modes = {int(mode.index): mode for mode in report.modes}
    if mode_index not in modes:
        raise ValueError(f"identifiability report has no mode {mode_index}")
    mode_info = modes[mode_index]
    theta0 = np.asarray(problem.final_brsp, dtype=np.float64)
    scale = np.asarray(report.parameter_scale, dtype=np.float64)
    mode = np.asarray(mode_info.scaled_parameter_direction, dtype=np.float64)
    if theta0.shape != (block.ncol,) or scale.shape != (block.ncol,) or mode.shape != (block.ncol,):
        raise ValueError("direction arrays do not match native parameter count")
    if not np.all(np.isfinite(theta0)) or not np.all(np.isfinite(mode)):
        raise ValueError("direction baseline/mode contains non-finite values")
    if not np.all(np.isfinite(scale)) or np.any(scale <= 0.0):
        raise ValueError("direction column_scale must be finite and positive")
    physical_c = mode / scale
    source_sha = str(problem.sha256)
    if not re.fullmatch(r"[0-9a-f]{64}", source_sha):
        raise ValueError("source sidecar SHA-256 must be 64 lowercase hexadecimal characters")
    order_sha = parameter_order_sha256(block)
    names = tuple(str(value) for value in block.parameter_name)
    encoded_names = [value.encode("ascii", errors="strict") for value in names]
    name_width = max((len(value) for value in encoded_names), default=1)

    from scipy.io import netcdf_file

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(dir=path.parent, suffix=".nc.tmp")
    os.close(descriptor)
    try:
        with netcdf_file(temporary, "w", version=1) as dataset:
            dataset.schema_id = "efit_direction_constraint_v1"
            dataset.schema_version = np.int32(1)
            dataset.writer_complete = np.int32(1)
            dataset.source_sidecar_sha256 = source_sha
            dataset.parameter_order_sha256 = order_sha
            dataset.mode_index = np.int32(mode_index)
            dataset.mode_state = str(mode_info.state)
            dataset.createDimension("parameter", block.ncol)
            dataset.createDimension("name_strlen", name_width)
            for name, values in (
                ("theta0", theta0),
                ("column_scale", scale),
                ("mode", mode),
                ("physical_c", physical_c),
            ):
                variable = dataset.createVariable(name, "d", ("parameter",))
                variable[:] = values
            target_variable = dataset.createVariable("target", "d", ())
            # scipy.io.netcdf_variable.assignValue indexes 0-D arrays with
            # ``[:]`` and fails on current NumPy; ellipsis is scalar-safe.
            target_variable[...] = np.float64(target)
            name_variable = dataset.createVariable(
                "parameter_name", "c", ("parameter", "name_strlen")
            )
            characters = np.full((block.ncol, name_width), b" ", dtype="S1")
            for index, value in enumerate(encoded_names):
                characters[index, : len(value)] = np.frombuffer(value, dtype="S1")
            name_variable[:] = characters
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise
    return {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "schema_id": "efit_direction_constraint_v1",
        "schema_version": 1,
        "source_sidecar_sha256": source_sha,
        "parameter_order_sha256": order_sha,
        "mode_index": mode_index,
        "mode_state": str(mode_info.state),
        "target": float(target),
    }


def nonlinear_displacement(metrics: TransitionMetrics) -> float:
    profile = np.asarray(
        [abs(float(value)) for value in metrics.profile_relative_rms.values()],
        dtype=float,
    )
    profile_rms = float(np.sqrt(np.mean(profile**2))) if profile.size else 0.0
    return max(
        abs(metrics.lcfs_rms_mm) / 5.0,
        abs(metrics.area_relative) / 0.02,
        abs(metrics.volume_relative) / 0.02,
        abs(metrics.beta_p_relative) / 0.02,
        abs(metrics.li_relative) / 0.02,
        profile_rms / 0.05,
    )


def weak_subspace_projection(
    parameter_delta: Sequence[float],
    parameter_scale: Sequence[float],
    modes: Sequence[Mapping[str, Any]],
) -> tuple[float, np.ndarray]:
    """Project a physical parameter displacement into baseline weak modes.

    The projection is performed in the fixed, dimensionless baseline column
    coordinates.  The returned fraction is a norm fraction (not a squared
    power fraction), matching the Stage-2 80% displacement criterion.
    """
    delta = np.asarray(parameter_delta, dtype=float)
    scale = np.asarray(parameter_scale, dtype=float)
    if delta.ndim != 1 or scale.shape != delta.shape:
        raise ValueError("parameter delta and scale must be matching vectors")
    if not np.all(np.isfinite(delta)) or not np.all(np.isfinite(scale)):
        raise ValueError("parameter delta and scale must be finite")
    if np.any(scale <= 0.0):
        raise ValueError("parameter scale must be positive")
    directions = [
        np.asarray(item["scaled_parameter_direction"], dtype=float)
        for item in modes
        if item.get("state") in {"borderline", "unresolved"}
    ]
    if not directions:
        raise ValueError("baseline report contains no weak/borderline directions")
    basis = np.column_stack(directions)
    if basis.shape[0] != delta.size or not np.all(np.isfinite(basis)):
        raise ValueError("weak-mode vectors do not match the parameter order")
    left, singular, _ = np.linalg.svd(basis, full_matrices=False)
    tolerance = np.finfo(float).eps * max(basis.shape) * singular[0]
    orthonormal = left[:, singular > tolerance]
    scaled_delta = delta / scale
    projection = orthonormal @ (orthonormal.T @ scaled_delta)
    denominator = float(np.linalg.norm(scaled_delta))
    fraction = 1.0 if denominator == 0.0 else float(np.linalg.norm(projection) / denominator)
    return min(max(fraction, 0.0), 1.0), projection


def assess_347_diamagnetic_response(
    baseline: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Evaluate the mandatory weak-mode explanation for 41672/347 ms."""
    analysis = baseline.get("analysis") or {}
    theta0 = np.asarray(baseline.get("parameter_state", ()), dtype=float)
    scale = np.asarray(analysis.get("parameter_scale", ()), dtype=float)
    modes = analysis.get("modes", ())
    if theta0.size == 0 or scale.shape != theta0.shape or not modes:
        return {
            "reference": "41672:347",
            "status": "incomplete",
            "classification": "insufficient baseline parameter information",
            "passed": False,
        }
    baseline_mask = tuple(bool(value) for value in baseline["snapshot"]["retained_mask"])
    eligible = sorted(
        (
            item
            for item in records
            if item.get("reference") == "41672:347"
            and item.get("control") == "diamagnetic_flux_scale"
            and str(item.get("chain", "")).startswith("diamagnetic_increasing")
            and item.get("status") == "succeeded"
            and item.get("parameter_state") is not None
            and float(item.get("log10_scale", 0.0)) > 0.0
        ),
        key=lambda item: float(item["log10_scale"]),
    )
    points: list[dict[str, Any]] = []
    projected_vectors: dict[str, np.ndarray] = {}
    for item in eligible:
        state = np.asarray(item["parameter_state"], dtype=float)
        if state.shape != theta0.shape:
            continue
        fraction, projection = weak_subspace_projection(state - theta0, scale, modes)
        case_id = str(item["case_id"])
        projected_vectors[case_id] = projection
        mask = tuple(bool(value) for value in item["snapshot"]["retained_mask"])
        points.append(
            {
                "case_id": case_id,
                "log10_scale": float(item["log10_scale"]),
                "weak_subspace_projection_fraction": fraction,
                "retained_mask_matches_baseline": mask == baseline_mask,
                "beta_p_relative": float(item.get("beta_p_relative", 0.0)),
                "li_relative": float(item.get("li_relative", 0.0)),
                "displacement": float(item.get("displacement", 0.0)),
                "response_onset": bool(item.get("response_onset")),
                "material_branch": bool(item.get("material_branch")),
                "chain": str(item.get("chain", "")),
            }
        )
    same_branch = [
        point
        for point in points
        if point["retained_mask_matches_baseline"] and not point["material_branch"]
    ]
    anchors = [
        point
        for point in same_branch
        if np.linalg.norm(projected_vectors[point["case_id"]])
        > np.finfo(float).eps
    ][:2]
    exact_targets: dict[float, dict[str, Any]] = {}
    for required in (3.0, 4.0):
        matches = [
            point
            for point in points
            if point["chain"] == "diamagnetic_increasing"
            and math.isclose(point["log10_scale"], required, abs_tol=1.0e-10)
        ]
        if matches:
            exact_targets[required] = matches[-1]
    same_branch_targets = [
        point for point in same_branch if point["log10_scale"] >= 3.0
    ]
    required_target_coverage = {
        "x1000": 3.0 in exact_targets,
        "x10000": 4.0 in exact_targets,
    }
    if (
        len(anchors) < 2
        or not all(required_target_coverage.values())
        or not same_branch_targets
    ):
        failures = [
            item
            for item in records
            if item.get("reference") == "41672:347"
            and item.get("control") == "diamagnetic_flux_scale"
            and str(item.get("chain", "")).startswith("diamagnetic_increasing")
            and _qualified_failure(item)
        ]
        if failures:
            first_failure = min(
                failures, key=lambda item: float(item.get("log10_scale", math.inf))
            )
            return {
                "reference": "41672:347",
                "status": "evaluated",
                "classification": "nonlinear/unreliable",
                "passed": False,
                "reason": (
                    "the warm-started path terminates at a repeat-qualified "
                    "deterministic outcome boundary before two same-branch anchors "
                    "or the x1000/x10000 targets are available"
                ),
                "points": points,
                "anchor_case_ids": [item["case_id"] for item in anchors],
                "required_target_coverage": required_target_coverage,
                "same_branch_high_target_case_ids": [
                    item["case_id"] for item in same_branch_targets
                ],
                "qualified_failure_case_id": first_failure["case_id"],
                "qualified_failure_log10_scale": float(
                    first_failure.get("log10_scale", math.nan)
                ),
                "rank_switching": False,
            }
        return {
            "reference": "41672:347",
            "status": "incomplete",
            "classification": "insufficient same-branch continuation points",
            "passed": False,
            "points": points,
            "anchor_case_ids": [item["case_id"] for item in anchors],
            "required_target_coverage": required_target_coverage,
            "same_branch_high_target_case_ids": [
                item["case_id"] for item in same_branch_targets
            ],
        }

    tangent = projected_vectors[anchors[0]["case_id"]]
    tangent = tangent / np.linalg.norm(tangent)

    def coordinate(point: Mapping[str, Any]) -> float:
        return float(np.dot(tangent, projected_vectors[str(point["case_id"])]))

    anchor_coordinates = np.asarray([coordinate(item) for item in anchors])
    denominator = float(anchor_coordinates @ anchor_coordinates)
    prediction: dict[str, list[dict[str, Any]]] = {}
    for response_name in ("beta_p_relative", "li_relative"):
        observed_anchor = np.asarray(
            [float(item[response_name]) for item in anchors]
        )
        slope = (
            float(anchor_coordinates @ observed_anchor) / denominator
            if denominator > np.finfo(float).tiny
            else math.nan
        )
        predictions = []
        for target in same_branch_targets:
            observed = float(target[response_name])
            predicted = slope * coordinate(target)
            relative_error = (
                abs(predicted - observed) / abs(observed)
                if math.isfinite(predicted) and observed != 0.0
                else (0.0 if predicted == observed else math.inf)
            )
            sign_matches = bool(
                (observed == 0.0 and predicted == 0.0)
                or observed * predicted > 0.0
            )
            predictions.append(
                {
                    "target_case_id": target["case_id"],
                    "target_log10_scale": target["log10_scale"],
                    "observed": observed,
                    "predicted": predicted,
                    "relative_magnitude_error": relative_error,
                    "sign_matches": sign_matches,
                    "passed": sign_matches and relative_error <= 0.25,
                }
            )
        prediction[response_name] = predictions

    target_projection = min(
        point["weak_subspace_projection_fraction"] for point in same_branch_targets
    )
    # Rank switching is a trajectory event, so inspect every successful
    # increasing-path and refinement point rather than only the two endpoints.
    rank_switching = any(
        not point["retained_mask_matches_baseline"] for point in points
    )
    projection_passed = target_projection >= 0.80
    prediction_passed = all(
        item["passed"] for records_for_response in prediction.values()
        for item in records_for_response
    )
    coverage_passed = all(required_target_coverage.values())
    passed = (
        coverage_passed
        and projection_passed
        and prediction_passed
        and not rank_switching
    )
    classification = (
        "truncation/rank switching"
        if rank_switching
        else ("weak-mode explained" if passed else "nonlinear/unreliable")
    )
    return {
        "reference": "41672:347",
        "status": "evaluated",
        "classification": classification,
        "passed": passed,
        "weak_subspace_projection_minimum": 0.80,
        "minimum_target_projection_fraction": target_projection,
        "projection_passed": projection_passed,
        "beta_li_prediction_relative_error_maximum": 0.25,
        "prediction_passed": prediction_passed,
        "rank_switching": rank_switching,
        "required_target_coverage": required_target_coverage,
        "same_branch_high_target_case_ids": [
            item["case_id"] for item in same_branch_targets
        ],
        "rank_mask_scanned_case_ids": [item["case_id"] for item in points],
        "anchor_case_ids": [item["case_id"] for item in anchors],
        "prediction": prediction,
        "points": points,
    }


def classify_transition(
    metrics: TransitionMetrics,
    *,
    repeat_noise: float,
    interval_log10: float | None = None,
    reverse_difference: float | None = None,
    previous_retained_mask: Sequence[bool] | None = None,
    retained_mask: Sequence[bool] | None = None,
    converged: bool = True,
    bisection_exhausted: bool = False,
    deterministic_failure_repeats: int = 0,
) -> TransitionClassification:
    displacement = nonlinear_displacement(metrics)
    onset = displacement >= 0.1 and displacement >= 10.0 * max(repeat_noise, 0.0)
    refined = interval_log10 is None or interval_log10 <= 0.025
    branch = converged and refined and (
        displacement >= 1.0
        or (reverse_difference is not None and reverse_difference >= 1.0)
    )
    rank_switching = False
    if previous_retained_mask is not None and retained_mask is not None:
        left = np.asarray(previous_retained_mask, dtype=bool)
        right = np.asarray(retained_mask, dtype=bool)
        if left.shape != right.shape:
            raise ValueError("retained masks must have the same shape")
        rank_switching = bool(np.any(left != right))
    failed = bool(
        not converged and bisection_exhausted and deterministic_failure_repeats >= 2
    )
    return TransitionClassification(displacement, onset, branch, rank_switching, failed)


def build_plan_manifest(
    *,
    output: Path,
    kfile_root: Path,
    executable: Path,
    scientific_sha256: str,
    table_identity: Mapping[str, Any],
    build_provenance: Mapping[str, Any] | None = None,
    analysis_identity: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Materialize an immutable Stage-1 plan without running EFIT."""
    scientific = fixed_scientific_config()
    if scientific.sha256 != scientific_sha256:
        raise ValueError(
            "declared scientific hash does not match the frozen (2,2) zero-edge "
            f"configuration: {scientific_sha256} != {scientific.sha256}"
        )
    analysis = {
        "cutoff_multipliers": [0.1, 1.0, 10.0],
        "core_families": list(CORE_FAMILIES),
        "incremental_family_sets": INCREMENTAL_FAMILY_SETS,
        "analysis_algorithm": "native_response_identifiability_v1",
        "workflow_sha256": sha256_file(Path(__file__).resolve()),
        "linearization_parser_sha256": sha256_file(
            REPOSITORY / "vaft" / "code" / "efit" / "linearization.py"
        ),
    }
    if analysis_identity:
        analysis.update(_json_safe(analysis_identity))
        # Code identities are execution facts, not caller-supplied labels.
        analysis["workflow_sha256"] = sha256_file(Path(__file__).resolve())
        analysis["linearization_parser_sha256"] = sha256_file(
            REPOSITORY / "vaft" / "code" / "efit" / "linearization.py"
        )
    build_record = build_provenance_record(executable, build_provenance)
    cases = []
    for reference in REFERENCE_SLICES:
        kfile = resolve_reference_kfile(kfile_root, reference)
        kfile_audit = audit_frozen_kfile(
            kfile,
            reference=reference,
            scientific=scientific,
            table_identity=table_identity,
        )
        require_frozen_kfile_audit(kfile_audit)
        spec = CaseSpec(
            stage=1,
            reference=reference,
            kind="native_baseline",
            kfile=kfile,
            executable=executable,
            scientific_sha256=scientific_sha256,
            table_identity=table_identity,
            analysis_identity=analysis,
            kfile_semantic_audit=kfile_audit,
            build_provenance=build_record,
        )
        case_id, identity = case_identity(spec)
        case_record = {
            "case_id": case_id,
            "status": "pending",
            "reference": asdict(reference),
            "workdir": str(Path(output) / "stage1" / case_id),
            "kfile": str(kfile),
            "kfile_semantic_audit": kfile_audit,
            "identity": identity,
        }
        previous = resumable_case_manifest(
            Path(case_record["workdir"]) / STAGE1_CASE_MANIFEST, identity
        )
        if previous is not None:
            case_record["status"] = "succeeded"
            case_record["resumed"] = True
        cases.append(case_record)
        if previous is None:
            write_json_atomic(
                Path(case_record["workdir"]) / STAGE1_CASE_MANIFEST,
                {
                    "schema_version": SCHEMA_VERSION,
                    "issue": ISSUE,
                    **case_record,
                },
            )
    payload = {
        "schema_version": SCHEMA_VERSION,
        "issue": ISSUE,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "stage1_gate": {"passed": False, "reason": "not yet evaluated"},
        "stage2_authorized": False,
        "build_provenance": build_record,
        "table_identity": table_identity,
        "cases": cases,
        "diamagnetic_log10_schedule": list(diamagnetic_log10_schedule()),
        "direction_alphas": [-4, -2, -1, -0.5, -0.25, -0.125, 0.125, 0.25, 0.5, 1, 2, 4],
    }
    write_json_atomic(Path(output) / STUDY_MANIFEST, payload)
    return payload


def _reference_from_record(record: Mapping[str, Any]) -> ReferenceSlice:
    return ReferenceSlice(**record["reference"])


def _stage1_case_result(
    record: Mapping[str, Any],
    *,
    executable: Path,
    scientific: Any,
) -> dict[str, Any]:
    """Run and analyze exactly one k-file, retaining failed outcomes."""
    from vaft.code.efit import (
        EFITConfig,
        EFITInputs,
        IdentifiabilityConfig,
        analyze_efit_identifiability,
        run_efit,
    )

    reference = _reference_from_record(record)
    workdir = Path(record["workdir"])
    workdir.mkdir(parents=True, exist_ok=True)
    manifest_path = workdir / STAGE1_CASE_MANIFEST
    identity = record["identity"]
    previous = resumable_case_manifest(manifest_path, identity)
    if previous is not None:
        return dict(previous)

    pending = {
        "schema_version": SCHEMA_VERSION,
        "issue": ISSUE,
        **dict(record),
        "status": "running",
        "started_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    write_json_atomic(manifest_path, pending)
    try:
        config = EFITConfig(
            executable=str(executable),
            workdir=workdir,
            shot=reference.shot,
            times=(reference.time_s,),
            args=("129",),
            profile=scientific.profile,
            initialization=scientific.initialization,
            numerics=scientific.numerics,
            constraints=scientific.constraints,
            export_linearization=True,
            write_restart=True,
            provenance={
                "study": ISSUE,
                "reference_role": reference.role,
                "single_frozen_kfile": True,
                "case_identity": record["case_id"],
                "build": identity["build_provenance"],
                "table": identity["table_identity"],
                "input": {
                    "kfile": str(Path(record["kfile"]).resolve()),
                    "kfile_sha256": identity["kfile_sha256"],
                    "kfile_semantic_audit": identity["kfile_semantic_audit"],
                },
            },
        )
        kfile = Path(record["kfile"])
        inputs = EFITInputs(
            workdir=workdir,
            kfiles=(kfile,),
            files=(kfile,),
            configuration={"source": "frozen_issue_663_kfile"},
        )
        result = run_efit(inputs, config)
        statuses = tuple(result.slice_statuses)
        slice_status = statuses[0] if len(statuses) == 1 else None
        outcome = (
            "accepted"
            if slice_status is not None and slice_status.usable
            else (
                slice_status.overall_status
                if slice_status is not None
                else ("runtime_failed" if not result.ok else "missing_slice_status")
            )
        )
        problems = tuple(result.linearizations)
        case_errors = list(result.diagnostic_errors)
        restart_signature = result.configuration.get("execution", {}).get(
            "restart_control_sha256"
        )
        restart_available = bool(
            result.restart_file is not None and Path(result.restart_file).is_file()
        )
        if not restart_available:
            case_errors.append("write_restart was requested but no esave.dat was produced")
        if not isinstance(restart_signature, str) or re.fullmatch(
            r"[0-9a-f]{64}", restart_signature
        ) is None:
            case_errors.append("native restart-control SHA-256 is missing or invalid")
        if len(problems) != 1:
            case_errors.append(
                f"expected exactly one native sidecar for one k-file, found {len(problems)}"
            )
        problem = problems[0] if len(problems) == 1 else None
        analysis = external_current_analysis = validation = snapshot = parameter_state = None
        mfile_family_chi2_audit = None
        native_rows: list[dict[str, Any]] = []
        if problem is not None:
            if problem.shot != reference.shot or abs(problem.time_ms - reference.time_ms) > 0.5:
                case_errors.append(
                    "native sidecar shot/time does not match the requested reference"
                )
            report = analyze_efit_identifiability(
                problem,
                IdentifiabilityConfig(
                    cutoff_multipliers=(0.1, 1.0, 10.0),
                    accounting_confounded=frozenset(("plasma_current",)),
                ),
            )
            analysis = report.to_dict()
            parameter_state = np.asarray(problem.final_brsp, dtype=float).tolist()
            native_rows = native_row_records(problem, reference)
            if problem.external_current is not None:
                external_current_analysis = analyze_efit_identifiability(
                    problem,
                    IdentifiabilityConfig(
                        block="external_current",
                        cutoff_multipliers=(0.1, 1.0, 10.0),
                    ),
                ).to_dict()
            if len(result.gfiles) == 1 and len(result.afiles) == 1:
                snapshot = equilibrium_snapshot(
                    result.gfiles[0], result.afiles[0], problem.main.retained_mask
                )
            else:
                case_errors.append(
                    "expected exactly one g-file and one a-file for snapshot comparison"
                )
            if len(result.mfiles) == 1:
                try:
                    mfile_family_chi2_audit = audit_mfile_family_chi2(
                        result.mfiles[0], native_rows
                    )
                except Exception as exc:
                    case_errors.append(
                        "m-file family chi-squared audit failed: "
                        f"{type(exc).__name__}: {exc}"
                    )
            else:
                case_errors.append(
                    "expected exactly one m-file for diagnostic chi-squared validation"
                )
            validation = validate_native_problem(
                problem,
                report,
                reference=reference,
                outcome=outcome,
                diagnostic_errors=case_errors,
                expected_executable_sha256=identity["executable_sha256"],
                mfile_family_chi2_audit=mfile_family_chi2_audit,
            )
        artifacts = _artifact_records(
            (
                *result.gfiles,
                *result.afiles,
                *result.mfiles,
                *result.logs,
                *result.linearization_files,
                *((result.restart_file,) if result.restart_file is not None else ()),
            )
        )
        passed = bool(
            validation is not None
            and snapshot is not None
            and restart_available
            and isinstance(restart_signature, str)
            and re.fullmatch(r"[0-9a-f]{64}", restart_signature) is not None
            and stage1_gate((validation,), expected_references=(reference,)).passed
        )
        payload = {
            **pending,
            "status": "succeeded" if passed else "failed",
            "finished_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "runtime": {
                "returncode": result.returncode,
                "status": result.status,
                "reason": result.reason,
                "slice_status": (
                    slice_status.to_dict() if slice_status is not None else None
                ),
                "diagnostic_errors": case_errors,
            },
            "validation": asdict(validation) if validation is not None else None,
            "analysis": analysis,
            "external_current_analysis": external_current_analysis,
            "native_rows": native_rows,
            "family_residual_summary": summarize_native_rows(native_rows),
            "mfile_family_chi2_audit": mfile_family_chi2_audit,
            "snapshot": snapshot,
            "parameter_state": parameter_state,
            "restart_control_sha256": restart_signature,
            "artifacts": artifacts,
            "linearization_file": (
                str(result.linearization_files[0])
                if len(result.linearization_files) == 1
                else None
            ),
            "restart_file": (
                str(result.restart_file) if result.restart_file is not None else None
            ),
        }
    except Exception as exc:
        payload = {
            **pending,
            "status": "failed",
            "finished_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "runtime": {"error": f"{type(exc).__name__}: {exc}"},
            "validation": None,
            "analysis": None,
            "external_current_analysis": None,
            "native_rows": [],
            "family_residual_summary": [],
            "mfile_family_chi2_audit": None,
            "snapshot": None,
            "parameter_state": None,
            "restart_control_sha256": None,
            "artifacts": [],
        }
    write_json_atomic(manifest_path, payload)
    return payload


def _results_from_stage1(
    case_payloads: Sequence[Mapping[str, Any]], gate: Stage1Gate
) -> dict[str, Any]:
    family_results: list[dict[str, Any]] = []
    spectra: list[dict[str, Any]] = []
    external_spectra: list[dict[str, Any]] = []
    mode_shares: list[dict[str, Any]] = []
    subset_results: list[dict[str, Any]] = []
    ip_accounting: list[dict[str, Any]] = []
    native_rows: list[dict[str, Any]] = []
    family_residual_summary: list[dict[str, Any]] = []
    mfile_family_chi2_audits: list[dict[str, Any]] = []
    for case in case_payloads:
        analysis = case.get("analysis")
        if not analysis:
            continue
        reference = _reference_from_record(case).key
        native_rows.extend(dict(item) for item in case.get("native_rows", ()))
        family_residual_summary.extend(
            dict(item) for item in case.get("family_residual_summary", ())
        )
        if case.get("mfile_family_chi2_audit"):
            mfile_family_chi2_audits.append(
                {
                    "reference": reference,
                    **dict(case["mfile_family_chi2_audit"]),
                }
            )
        spectra.append(
            {
                "reference": reference,
                "singular_values": analysis["singular_values"],
                "tau": analysis["nominal_cutoff"],
                "ranks": analysis["ranks"],
                "nullity": analysis["nullity"],
            }
        )
        external = case.get("external_current_analysis")
        if external:
            external_spectra.append(
                {
                    "reference": reference,
                    "singular_values": external["singular_values"],
                    "tau": external["nominal_cutoff"],
                    "ranks": external["ranks"],
                    "nullity": external["nullity"],
                }
            )
        for mode in analysis.get("modes", ()):
            for family, share in mode.get("family_shares", {}).items():
                mode_shares.append(
                    {
                        "reference": reference,
                        "mode_index": mode["index"],
                        "state": mode["state"],
                        "family": family,
                        "share": share,
                    }
                )
        # Case manifests are written with sorted JSON keys.  Keep the derived
        # projection independent of the in-memory insertion order so the
        # Stage-2 content gate reproduces the Stage-1 hash after reloading.
        subsets = analysis.get("subsets", {})
        for name in sorted(subsets):
            subset = subsets[name]
            subset_results.append(
                {"reference": reference, "name": name, **dict(subset)}
            )
        if analysis.get("ip_accounting") is not None:
            ip_accounting.append(
                {"reference": reference, **dict(analysis["ip_accounting"])}
            )
        non_ip_objective = sum(
            float(item.get("solver_objective") or 0.0)
            for family, item in analysis["families"].items()
            if family != "plasma_current"
        )
        for family, item in analysis["families"].items():
            family_results.append(
                {
                    "reference": reference,
                    "family": family,
                    "classification": [
                        value.replace("_", " ")
                        for value in item.get("classifications", ())
                    ],
                    "rank_gain_by_cutoff": item["rank_gain"],
                    "weak_subspace_action_fraction": item["weak_subspace_fraction"],
                    "weak_singular_value_lift": item["maximum_weak_singular_lift"],
                    "curvature_inverse_trace_reduction": item[
                        "curvature_inverse_trace_reduction"
                    ],
                    "diagnostic_chi2": item["diagnostic_chi2"],
                    "solver_objective": item["solver_objective"],
                    "baseline_solver_objective_share": item[
                        "solver_objective_share"
                    ],
                    "solver_objective_share_excluding_ip": (
                        None
                        if family == "plasma_current" or non_ip_objective <= 0.0
                        else float(item.get("solver_objective") or 0.0)
                        / non_ip_objective
                    ),
                }
            )
    return {
        "schema_version": SCHEMA_VERSION,
        "issue": ISSUE,
        "stage1_gate": asdict(gate),
        "stage1_cases": list(case_payloads),
        "family_results": family_results,
        "spectra": spectra,
        "external_current_spectra": external_spectra,
        "mode_family_shares": mode_shares,
        "subset_results": subset_results,
        "ip_accounting": ip_accounting,
        "native_rows": native_rows,
        "family_residual_summary": family_residual_summary,
        "mfile_family_chi2_audits": mfile_family_chi2_audits,
        "continuation_results": [],
    }


def execute_stage1(
    *,
    output: Path,
    kfile_root: Path,
    executable: Path,
    scientific_sha256: str,
    table_identity: Mapping[str, Any],
    build_provenance: Mapping[str, Any],
) -> dict[str, Any]:
    """Run all five native baselines and atomically record the Stage-1 gate."""
    scientific = fixed_scientific_config()
    if scientific.sha256 != scientific_sha256:
        raise ValueError(
            "declared scientific hash does not match the frozen (2,2) zero-edge "
            f"configuration: {scientific_sha256} != {scientific.sha256}"
        )
    build_record = build_provenance_record(executable, build_provenance)
    validate_build_provenance(build_record)
    if not table_identity:
        raise ValueError("table identity must be a non-empty provenance record")
    plan = build_plan_manifest(
        output=output,
        kfile_root=kfile_root,
        executable=executable,
        scientific_sha256=scientific_sha256,
        table_identity=table_identity,
        build_provenance=build_record,
    )
    case_payloads = [
        _stage1_case_result(record, executable=executable, scientific=scientific)
        for record in plan["cases"]
    ]
    validations = [
        _validation_from_dict(item["validation"])
        for item in case_payloads
        if item.get("validation") is not None
    ]
    gate = stage1_gate(validations)
    plan["cases"] = [
        {
            "case_id": item["case_id"],
            "status": item["status"],
            "reference": item["reference"],
            "workdir": item["workdir"],
            "identity": item["identity"],
        }
        for item in case_payloads
    ]
    results = _results_from_stage1(case_payloads, gate)
    projection_sha256 = _canonical_sha(results)
    plan["stage1_gate"] = asdict(gate)
    plan["stage2_authorized"] = gate.passed
    plan["stage1_projection_sha256"] = projection_sha256
    manifest_path = write_json_atomic(Path(output) / STUDY_MANIFEST, plan)
    results["stage1_projection_sha256"] = projection_sha256
    results["stage1_manifest_sha256"] = sha256_file(manifest_path)
    write_report(results, output)
    return results


def analyze_sidecars(paths: Sequence[Path], output: Path) -> dict[str, Any]:
    """Analyze completed native exports without launching EFIT."""
    from vaft.code.efit import (
        IdentifiabilityConfig,
        analyze_efit_identifiability,
        read_efit_linearization,
    )

    by_reference = {item.key: item for item in REFERENCE_SLICES}
    cases = []
    validations = []
    for path in paths:
        problem = read_efit_linearization(path)
        key = f"{problem.shot}:{int(round(problem.time_ms))}"
        if key not in by_reference:
            raise ValueError(f"sidecar is not one of the five reference slices: {key}")
        report = analyze_efit_identifiability(problem)
        validation = validate_native_problem(
            problem,
            report,
            reference=by_reference[key],
            # A sidecar alone cannot prove that the matching equilibrium and
            # its a-/m-files passed the acceptance envelope.
            outcome="unverified",
        )
        validations.append(validation)
        cases.append(
            {
                "reference": asdict(by_reference[key]),
                "status": "analyzed",
                "linearization_file": str(Path(path).resolve()),
                "validation": asdict(validation),
                "analysis": report.to_dict(),
                "native_rows": native_row_records(problem, by_reference[key]),
                "family_residual_summary": summarize_native_rows(
                    native_row_records(problem, by_reference[key])
                ),
                "external_current_analysis": (
                    analyze_efit_identifiability(
                        problem,
                        IdentifiabilityConfig(block="external_current"),
                    ).to_dict()
                    if problem.external_current is not None
                    else None
                ),
            }
        )
    gate = stage1_gate(validations)
    payload = _results_from_stage1(cases, gate)
    write_report(payload, output)
    return payload


def require_stage2_gate(output: Path) -> Mapping[str, Any]:
    """Load and content-validate all Stage-1 inputs before any Stage-2 work."""
    from vaft.code.efit import read_efit_linearization

    output = Path(output)
    path = Path(output) / STUDY_MANIFEST
    if not path.is_file():
        raise RuntimeError("Stage 2 requires an existing Stage-1 study manifest")
    payload = json.loads(path.read_text(encoding="utf-8"))
    gate = payload.get("stage1_gate", {})
    if not payload.get("stage2_authorized") or not gate.get("passed"):
        reasons = "; ".join(gate.get("reasons", ())) or "Stage-1 gate is not passing"
        raise RuntimeError(f"Stage 2 is blocked: {reasons}")
    if payload.get("schema_version") != SCHEMA_VERSION or payload.get("issue") != ISSUE:
        raise RuntimeError("Stage-1 manifest schema/issue identity is invalid")
    manifest_sha256 = sha256_file(path)
    summaries = payload.get("cases", ())
    if not isinstance(summaries, Sequence) or isinstance(summaries, (str, bytes)):
        raise RuntimeError("Stage-1 manifest cases are malformed")
    if len(summaries) != len(REFERENCE_SLICES):
        raise RuntimeError("Stage-1 manifest must contain exactly five cases")
    expected_references = {item.key for item in REFERENCE_SLICES}
    cases: dict[str, Mapping[str, Any]] = {}
    seen_case_ids: set[str] = set()
    build = payload.get("build_provenance", {})
    source_revision = str(build.get("source_revision", "")).lower()
    executable_sha256 = build.get("executable", {}).get("sha256")
    current_workflow_sha256 = sha256_file(Path(__file__).resolve())
    current_parser_sha256 = sha256_file(
        REPOSITORY / "vaft" / "code" / "efit" / "linearization.py"
    )
    for summary in summaries:
        if not isinstance(summary, Mapping):
            raise RuntimeError("Stage-1 case summary is malformed")
        case_id = str(summary.get("case_id", ""))
        if not case_id or case_id in seen_case_ids:
            raise RuntimeError("Stage-1 case ids are missing or duplicated")
        seen_case_ids.add(case_id)
        case_path = Path(str(summary.get("workdir", ""))) / STAGE1_CASE_MANIFEST
        if not case_path.is_file():
            raise RuntimeError(f"Stage-1 case manifest is missing: {case_id}")
        case = json.loads(case_path.read_text(encoding="utf-8"))
        identity = case.get("identity")
        if not isinstance(identity, Mapping):
            raise RuntimeError(f"{case_id}: content identity is missing")
        if summary.get("identity") != identity:
            raise RuntimeError(f"{case_id}: summary/case identity differs")
        analysis_identity = identity.get("analysis_identity", {})
        if (
            analysis_identity.get("workflow_sha256") != current_workflow_sha256
            or analysis_identity.get("linearization_parser_sha256")
            != current_parser_sha256
            or analysis_identity.get("analysis_algorithm")
            != "native_response_identifiability_v1"
        ):
            raise RuntimeError(f"{case_id}: Stage-1 analysis code identity is stale")
        if case.get("case_id") != case_id or not case_id.endswith(
            _canonical_sha(identity)[:12]
        ):
            raise RuntimeError(f"{case_id}: case id does not match its content identity")
        if resumable_case_manifest(case_path, identity) is None:
            raise RuntimeError(f"{case_id}: case artifacts are absent, changed, or incomplete")
        try:
            reference = _reference_from_record(case)
        except (KeyError, TypeError) as exc:
            raise RuntimeError(f"{case_id}: reference identity is malformed") from exc
        if reference.key in cases:
            raise RuntimeError(f"Stage-1 reference is duplicated: {reference.key}")
        sidecar = Path(str(case.get("linearization_file", "")))
        restart = Path(str(case.get("restart_file", "")))
        artifact_hashes = {
            str(Path(item["path"]).resolve()): item.get("sha256")
            for item in case.get("artifacts", ())
            if isinstance(item, Mapping) and item.get("path")
        }
        for artifact, label in ((sidecar, "sidecar"), (restart, "restart")):
            resolved = str(artifact.resolve())
            if (
                not artifact.is_file()
                or artifact_hashes.get(resolved) != sha256_file(artifact)
            ):
                raise RuntimeError(f"{case_id}: {label} is not content-bound")
        problem = read_efit_linearization(sidecar)
        if (
            problem.source_revision.lower() != source_revision
            or problem.executable_sha256 != executable_sha256
        ):
            raise RuntimeError(f"{case_id}: sidecar source/executable identity is stale")
        cases[reference.key] = case
    if set(cases) != expected_references:
        raise RuntimeError("Stage-1 manifest does not contain the exact reference set")

    results_path = output / RESULTS_JSON
    if not results_path.is_file():
        raise RuntimeError("Stage-1 normalized result is missing")
    results = json.loads(results_path.read_text(encoding="utf-8"))
    if results.get("stage1_manifest_sha256") != manifest_sha256:
        raise RuntimeError("normalized results are not bound to the current Stage-1 manifest")
    validations = []
    for reference in REFERENCE_SLICES:
        validation = cases[reference.key].get("validation")
        if not isinstance(validation, Mapping):
            raise RuntimeError(f"{reference.key}: Stage-1 validation record is missing")
        validations.append(_validation_from_dict(validation))
    derived_gate = stage1_gate(validations)
    if _json_safe(asdict(derived_gate)) != _json_safe(gate):
        raise RuntimeError("Stage-1 gate cannot be reproduced from the case manifests")
    derived_projection = _results_from_stage1(
        [cases[reference.key] for reference in REFERENCE_SLICES], derived_gate
    )
    projection_sha256 = _canonical_sha(derived_projection)
    if (
        payload.get("stage1_projection_sha256") != projection_sha256
        or results.get("stage1_projection_sha256") != projection_sha256
    ):
        raise RuntimeError("Stage-1 result sections are stale or internally altered")
    plan_path = output / STAGE2_PLAN
    if plan_path.is_file():
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
        if plan.get("stage1_manifest_sha256") != manifest_sha256:
            raise RuntimeError("Stage-2 plan is not bound to the current Stage-1 manifest")
        plan_references = {
            f"{item['reference']['shot']}:{item['reference']['time_ms']}": item
            for item in plan.get("references", ())
        }
        if plan_references and set(plan_references) != expected_references:
            raise RuntimeError("Stage-2 plan baseline reference set is stale")
        for reference, item in plan_references.items():
            case = cases[reference]
            if (
                item.get("baseline_case_id") != case["case_id"]
                or item.get("baseline_restart_sha256") != sha256_file(
                    Path(case["restart_file"])
                )
                or item.get("baseline_linearization_sha256") != sha256_file(
                    Path(case["linearization_file"])
                )
            ):
                raise RuntimeError(f"{reference}: Stage-2 baseline binding is stale")
    validated = dict(payload)
    validated["_validated_case_payloads"] = cases
    validated["_manifest_sha256"] = manifest_sha256
    return validated


def _table_file_hashes(record: Mapping[str, Any]) -> dict[str, str]:
    files = record.get("files", {})
    if not isinstance(files, Mapping):
        return {}
    return {
        str(name): str(item["sha256"])
        for name, item in files.items()
        if isinstance(item, Mapping) and item.get("sha256")
    }


def _pf_relation_structural_effect(shot: Mapping[str, Any]) -> bool:
    for name in ("no_pf_relations", "no_pf_penalty_or_relations"):
        item = shot.get("comparisons", {}).get(name)
        if not item:
            continue
        pf_change = (
            item.get("pf_reconstructed_current_relative_rms_change", {}).get("median")
            or 0.0
        )
        lcfs = item.get("geometry", {}).get("lcfs_rms_mm", {}).get("median") or 0.0
        acceptance = abs(float(item.get("acceptance_change_percentage_points", 0.0)))
        if float(pf_change) >= 0.02 or float(lcfs) >= 5.0 or acceptance >= 10.0:
            return True
    return False


def _validate_issue_663_endpoint_artifacts(
    payload: Mapping[str, Any], *, shot: int, variant: str
) -> None:
    records = (
        payload.get("endpoint_artifacts", {})
        .get(str(shot), {})
        .get(variant, ())
    )
    if not records:
        raise ValueError("endpoint artifact manifest is missing")
    prefixes: set[str] = set()
    for item in records:
        if not isinstance(item, Mapping):
            raise ValueError("endpoint artifact record is malformed")
        path = Path(str(item.get("path", ""))).resolve()
        if f"shot_{shot}" not in path.parts or variant not in path.parts:
            raise ValueError("endpoint artifact is outside its shot/variant directory")
        if not path.is_file() or item.get("sha256") != sha256_file(path):
            raise ValueError("endpoint artifact is absent or changed")
        name = path.name.lower()
        if re.match(r"[agmk]0\d+\.\d+", name):
            prefixes.add(name[0])
    if not {"a", "g", "m", "k"}.issubset(prefixes):
        raise ValueError("endpoint artifacts do not bind k/g/a/m scientific files")


def _rederive_issue_663_shot(
    payload: Mapping[str, Any], shot: int
) -> Mapping[str, Any]:
    """Recompute #663 summaries, comparisons, and labels from stored raw slices."""
    support = _constraint_support()
    profile = _profile_support()
    source = payload.get("shots", {}).get(str(shot))
    if not isinstance(source, Mapping):
        raise ValueError("shot block is missing")
    source_variants = source.get("variants", {})
    expected_variants = {
        item.name: item
        for item in support.COMPLETE_VARIANTS
        if item.name in ISSUE_663_REPRODUCTION_VARIANTS
    }
    if set(expected_variants) != set(ISSUE_663_REPRODUCTION_VARIANTS):
        raise ValueError("current #663 workflow does not define the required endpoints")
    block: dict[str, Any] = {"variants": {}, "comparisons": {}}
    for name in ISSUE_663_REPRODUCTION_VARIANTS:
        item = source_variants.get(name)
        if not isinstance(item, Mapping) or not isinstance(item.get("run"), Mapping):
            raise ValueError(f"raw endpoint {name} is missing")
        expected = expected_variants[name]
        expected_specification = {
            **expected.__dict__,
            "scales": dict(expected.scales),
        }
        if item.get("specification") != expected_specification:
            raise ValueError(f"raw endpoint {name} has a different control specification")
        run = item["run"]
        if run.get("status") != "completed" or int(run.get("returncode", -1)) != 0:
            raise ValueError(f"raw endpoint {name} did not complete successfully")
        scientific = support.scientific_for(expected, profile)
        if run.get("scientific_sha256") != scientific.sha256:
            raise ValueError(f"raw endpoint {name} scientific hash differs")
        _validate_issue_663_endpoint_artifacts(payload, shot=shot, variant=name)
        block["variants"][name] = {
            "specification": expected_specification,
            "run": run,
            "summary": support.summarize(run, profile),
        }
    baseline_run = block["variants"]["baseline"]["run"]
    for name in ISSUE_663_REPRODUCTION_VARIANTS:
        if name == "baseline":
            continue
        block["comparisons"][name] = support.comparison(
            block["variants"][name]["run"], baseline_run, profile
        )
    passive_comparison = support.comparison(
        block["variants"]["no_passive_response"]["run"],
        block["variants"]["zero_vcurrt"]["run"],
        profile,
    )
    block["passive_attribution"] = {
        "comparison": passive_comparison,
        "agreement": not support.material_response(passive_comparison),
    }
    block["classification"] = support.classify(block)
    return block


def validate_issue_663_evidence(
    path: Path, stage1: Mapping[str, Any]
) -> dict[str, Any]:
    """Accept #663 evidence only when every scientific provenance hash matches."""
    path = Path(path).resolve()
    prior = json.loads(path.read_text(encoding="utf-8"))
    reasons: list[str] = []
    if prior.get("issue") != 663:
        reasons.append("input is not an issue-#663 result")
    current_scientific = {
        summary.get("identity", {}).get("scientific_sha256")
        for summary in stage1.get("cases", ())
    }
    if current_scientific != {prior.get("fixed_scientific_sha256")}:
        reasons.append("fixed scientific configuration SHA-256 differs")
    reproduction = prior.get("reproduction_provenance", {})
    reproduction_identity = reproduction.get("identity", {})
    expected_workflow_sha256 = sha256_file(CONSTRAINT_STUDY)
    expected_reference_sha256 = sha256_file(
        REPOSITORY / "test" / "data" / "efit_reference_set.json"
    )
    if reproduction_identity.get("workflow_sha256") != expected_workflow_sha256:
        reasons.append("#663 workflow SHA-256 differs or is unbound")
    if reproduction_identity.get("reference_set_sha256") != expected_reference_sha256:
        reasons.append("#663 reference-set SHA-256 differs or is unbound")
    prior_executable = (
        reproduction.get("executable", {}).get("sha256")
        or (prior.get("toolchain", {}).get("efit") or {}).get("sha256")
    )
    current_executable = (
        stage1.get("build_provenance", {}).get("executable", {}).get("sha256")
    )
    if not prior_executable or prior_executable != current_executable:
        reasons.append("EFIT executable SHA-256 differs")
    current_table = stage1.get("table_identity", {})
    prior_table = reproduction.get("table_identity") or prior.get("table", {})
    current_hashes = _table_file_hashes(current_table)
    prior_hashes = _table_file_hashes(prior_table)
    if current_hashes and prior_hashes:
        if current_hashes != prior_hashes:
            reasons.append("EFIT table file hashes differ")
    elif _canonical_sha(current_table) != _canonical_sha(prior_table):
        reasons.append("EFIT table identity differs")

    references: dict[str, Mapping[str, Any]] = {}
    for summary in stage1.get("cases", ()):
        reference = ReferenceSlice(**summary["reference"])
        references[reference.key] = summary
        try:
            prior_kfile = resolve_reference_kfile(path.parent, reference)
        except (OSError, ValueError) as exc:
            reasons.append(f"{reference.key}: cannot resolve #663 baseline k-file ({exc})")
            continue
        expected = summary.get("identity", {}).get("kfile_sha256")
        if sha256_file(prior_kfile) != expected:
            reasons.append(f"{reference.key}: baseline k-file SHA-256 differs")

    expected_references = {item.key for item in REFERENCE_SLICES}
    if set(references) != expected_references:
        reasons.append("Stage-1 manifest does not contain exactly five references")
    rederived_shots: dict[int, Mapping[str, Any]] = {}
    for shot in sorted({item.shot for item in REFERENCE_SLICES}):
        block = prior.get("shots", {}).get(str(shot))
        if not isinstance(block, Mapping):
            reasons.append(f"#663 evidence is missing raw results for shot {shot}")
            continue
        missing_variants = sorted(
            set(ISSUE_663_REPRODUCTION_VARIANTS)
            - set(block.get("variants", {}))
        )
        if missing_variants:
            reasons.append(
                f"#663 evidence for shot {shot} is missing variants: "
                + ", ".join(missing_variants)
            )
        variants = block.get("variants", {})
        for name in ("baseline", "zero_vcurrt", "no_passive_response"):
            produced = (
                variants.get(name, {}).get("summary", {}).get("plasma_produced")
            )
            if produced is None or int(produced) <= 0:
                reasons.append(
                    f"#663 evidence for shot {shot}/{name} has no produced plasma slices"
                )
        passive = block.get("passive_attribution")
        common = (
            passive.get("comparison", {}).get("geometry", {}).get("common_produced")
            if isinstance(passive, Mapping)
            else None
        )
        if (
            not isinstance(passive, Mapping)
            or passive.get("agreement") is not True
            or common is None
            or int(common) <= 0
        ):
            reasons.append(
                f"#663 passive attribution for shot {shot} lacks explicit common-slice agreement"
            )
        try:
            rederived = _rederive_issue_663_shot(prior, shot)
        except Exception as exc:
            reasons.append(
                f"#663 raw endpoint rederivation failed for shot {shot}: "
                f"{type(exc).__name__}: {exc}"
            )
            continue
        rederived_shots[shot] = rederived
        for name in ISSUE_663_REPRODUCTION_VARIANTS:
            if name == "baseline":
                continue
            comparison = rederived["comparisons"][name]
            common = comparison.get("geometry", {}).get("common_produced")
            acceptance_change = comparison.get("acceptance_change_percentage_points")
            population_material = bool(
                comparison.get("material_response") is True
                and acceptance_change is not None
                and math.isfinite(float(acceptance_change))
                and abs(float(acceptance_change)) >= 10.0
            )
            # An endpoint that eliminates all common plasma slices is still
            # valid nonlinear evidence when its independently rederived
            # acceptance population crosses the study's 10-point material
            # threshold.  Requiring a common slice here would discard exactly
            # the collapsed/missing-output outcomes the protocol preserves.
            if (common is None or int(common) <= 0) and not population_material:
                reasons.append(
                    f"#663 raw endpoint {shot}/{name} has neither common produced "
                    "slices nor a material acceptance-population change"
                )
        passive_common = (
            rederived["passive_attribution"]["comparison"]
            .get("geometry", {})
            .get("common_produced")
        )
        if (
            rederived["passive_attribution"]["agreement"] is not True
            or passive_common is None
            or int(passive_common) <= 0
        ):
            reasons.append(
                f"#663 rederived passive attribution fails for shot {shot}"
            )
    evidence: dict[str, dict[str, Any]] = {}
    if not reasons:
        for reference in REFERENCE_SLICES:
            shot = rederived_shots[reference.shot]
            classifications = shot.get("classification", {})
            by_family: dict[str, Any] = {}
            for family, item in classifications.items():
                labels = tuple(str(value) for value in item.get("classification", ()))
                by_family[str(family)] = {
                    "source": "issue_663",
                    "classification": labels,
                    "nonlinear_material_response": "overwhelmed" in labels,
                    "structural_effect": "structural anchor" in labels,
                    "accounting_confounded": "accounting-confounded" in labels,
                    "ablation_material_response": bool(
                        item.get("ablation_material_response")
                    ),
                    "ablation": item.get("ablation"),
                    "high_strength_test": item.get("high_strength_test"),
                }
            relation_structural = _pf_relation_structural_effect(shot)
            by_family["pf_relation"] = {
                "source": "issue_663",
                "classification": (
                    ["structural anchor"] if relation_structural else ["inactive/redundant"]
                ),
                "structural_effect": relation_structural,
                "ablation": "no_pf_relations",
            }
            evidence[reference.key] = by_family
    return {
        "source": str(path),
        "source_sha256": sha256_file(path),
        "reusable": not reasons,
        "reasons": reasons,
        "required_reproductions": (
            []
            if not reasons
            else [
                "no_ip",
                "no_flux_loops",
                "no_bpol_probes",
                "no_diamagnetic_flux",
                "no_pf_penalty",
                "no_pf_relations",
                "no_pf_penalty_or_relations",
                "flux_loops_x100",
                "bpol_probes_x100",
                "diamagnetic_flux_x10000",
            ]
        ),
        "evidence": evidence,
    }


ISSUE_663_REPRODUCTION_VARIANTS = (
    "baseline",
    "no_ip",
    "no_flux_loops",
    "no_bpol_probes",
    "no_diamagnetic_flux",
    "no_pf_penalty",
    "no_pf_relations",
    "no_pf_penalty_or_relations",
    "zero_vcurrt",
    "no_passive_response",
    "flux_loops_x100",
    "bpol_probes_x100",
    "diamagnetic_flux_x10000",
)

ISSUE_663_ABLATION_VARIANTS = (
    "no_ip",
    "no_flux_loops",
    "no_bpol_probes",
    "no_diamagnetic_flux",
    "no_pf_penalty",
    "no_pf_relations",
    "no_pf_penalty_or_relations",
)

ISSUE_663_ABLATION_THRESHOLDS: Mapping[str, float] = {
    "lcfs_rms_mm": 5.0,
    "area_relative": 0.02,
    "volume_relative": 0.02,
    "beta_p_relative": 0.02,
    "li_relative": 0.02,
    "profile_relative_rms": 0.05,
    "acceptance_change_percentage_points": 10.0,
}

_TABLE_DIR_RE = re.compile(
    r"(?im)^\s*TABLE_DIR\s*=\s*['\"](?P<path>[^'\"]+)['\"]"
)


def _reference_table_directories(
    stage1: Mapping[str, Any],
) -> dict[int, Path]:
    """Recover each shot's literal TABLE_DIR from its frozen scientific k-file."""
    result: dict[int, Path] = {}
    for payload in _stage1_case_payloads(stage1).values():
        reference = _reference_from_record(payload)
        text = Path(payload["kfile"]).read_text(encoding="utf-8")
        match = _TABLE_DIR_RE.search(text)
        if match is None:
            raise RuntimeError(f"{reference.key}: frozen k-file has no TABLE_DIR")
        table_dir = Path(match.group("path")).expanduser()
        if not table_dir.is_dir():
            raise RuntimeError(
                f"{reference.key}: frozen TABLE_DIR is unavailable: {table_dir}"
            )
        existing = result.get(reference.shot)
        if existing is not None and existing != table_dir:
            raise RuntimeError(
                f"shot {reference.shot}: reference slices use different TABLE_DIR values"
            )
        result[reference.shot] = table_dir
    if set(result) != {41672, 39915, 41524}:
        raise RuntimeError("could not resolve table directories for all three shots")
    return result


def build_issue_663_reproduction_plan(output: Path) -> dict[str, Any]:
    """Create a content-addressed rerun using the Stage-1 executable/tables."""
    stage1 = require_stage2_gate(output)
    executable_record = stage1["build_provenance"]["executable"]
    executable = Path(executable_record["path"])
    if not executable.is_file() or sha256_file(executable) != executable_record["sha256"]:
        raise RuntimeError("Stage-1 executable is absent or changed")
    table_identity = stage1.get("table_identity", {})
    tables_by_shot = _reference_table_directories(stage1)
    identity = {
        "schema_version": SCHEMA_VERSION,
        "purpose": "reproduce_incompatible_issue_663_endpoints",
        "executable_sha256": executable_record["sha256"],
        "table_identity": table_identity,
        "scientific_sha256": fixed_scientific_config().sha256,
        "workflow_sha256": sha256_file(
            REPOSITORY
            / "workflow"
            / "efit_constraint_information"
            / "constraint_information_study.py"
        ),
        "reference_set_sha256": sha256_file(
            REPOSITORY / "test" / "data" / "efit_reference_set.json"
        ),
        "shots": [41672, 39915, 41524],
        "tables_by_shot": {
            str(shot): str(path) for shot, path in sorted(tables_by_shot.items())
        },
        "variants": list(ISSUE_663_REPRODUCTION_VARIANTS),
    }
    reproduction_id = _canonical_sha(identity)[:16]
    workdir = Path(output) / "stage2" / "issue_663_reproduction" / reproduction_id
    groups: dict[Path, list[int]] = {}
    for shot, tables in tables_by_shot.items():
        groups.setdefault(tables, []).append(shot)
    invocations = []
    workflow = (
        REPOSITORY
        / "workflow"
        / "efit_constraint_information"
        / "constraint_information_study.py"
    )
    for order, (tables, shots) in enumerate(sorted(groups.items(), key=lambda item: str(item[0]))):
        group_output = workdir / f"table_group_{order}"
        invocations.append(
            {
                "shots": sorted(shots),
                "tables": str(tables),
                "output": str(group_output),
                "command": [
                    sys.executable,
                    str(workflow),
                    "--output",
                    str(group_output),
                    "--shots",
                    ",".join(str(shot) for shot in sorted(shots)),
                    "--variants",
                    ",".join(ISSUE_663_REPRODUCTION_VARIANTS),
                    "--tables",
                    str(tables),
                    "--packaged-envelope",
                ],
            }
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "issue": ISSUE,
        "reproduction_id": reproduction_id,
        "identity": identity,
        "workdir": str(workdir),
        "result": str(workdir / "constraint_information.json"),
        "invocations": invocations,
        "environment": {"EFIT": str(executable)},
        "status": "planned",
    }


def execute_issue_663_reproduction(output: Path) -> dict[str, Any]:
    """Reproduce incompatible #663 endpoints and merge their classifications."""
    stage1 = require_stage2_gate(output)
    plan = build_issue_663_reproduction_plan(output)
    workdir = Path(plan["workdir"])
    workdir.mkdir(parents=True, exist_ok=True)
    manifest = workdir / "reproduction_manifest.json"
    result_path = Path(plan["result"])
    if result_path.is_file():
        audit = validate_issue_663_evidence(result_path, stage1)
        if audit["reusable"]:
            merged = merge_issue_663_evidence(output, result_path)
            completed = {
                **plan,
                "status": "succeeded",
                "result_sha256": sha256_file(result_path),
                "evidence_audit": audit,
                "merged_result_sha256": _canonical_sha(merged),
            }
            write_json_atomic(manifest, completed)
            return completed
    write_json_atomic(manifest, {**plan, "status": "running"})
    environment = dict(os.environ)
    environment.pop("EFITHOME", None)
    environment["EFIT"] = plan["environment"]["EFIT"]
    group_payloads = []
    source_plot_artifacts: list[dict[str, Any]] = []
    endpoint_artifacts: dict[str, dict[str, list[dict[str, str]]]] = {}
    returncodes = []
    for order, invocation in enumerate(plan["invocations"]):
        completed_process = subprocess.run(
            invocation["command"],
            cwd=REPOSITORY,
            env=environment,
            capture_output=True,
            text=True,
            check=False,
        )
        returncodes.append(completed_process.returncode)
        (workdir / f"reproduction_{order}.stdout").write_text(
            completed_process.stdout, encoding="utf-8"
        )
        (workdir / f"reproduction_{order}.stderr").write_text(
            completed_process.stderr, encoding="utf-8"
        )
        group_result = Path(invocation["output"]) / "constraint_information.json"
        if completed_process.returncode != 0 or not group_result.is_file():
            failed = {
                **plan,
                "status": "failed",
                "returncodes": returncodes,
                "failed_invocation": order,
            }
            write_json_atomic(manifest, failed)
            raise RuntimeError(
                "#663 endpoint reproduction failed for table group "
                f"{order} with status {completed_process.returncode}"
            )
        group_payload = json.loads(group_result.read_text(encoding="utf-8"))
        group_payloads.append(group_payload)
        for shot in invocation["shots"]:
            by_variant = endpoint_artifacts.setdefault(str(shot), {})
            for variant in ISSUE_663_REPRODUCTION_VARIANTS:
                variant_root = (
                    Path(invocation["output"]) / f"shot_{shot}" / variant
                )
                by_variant[variant] = _artifact_records(
                    path for path in variant_root.rglob("*") if path.is_file()
                )
        for plot in group_payload.get("plots", ()):
            plot_path = Path(plot)
            if not plot_path.is_absolute():
                plot_path = Path(invocation["output"]) / plot_path
            if plot_path.is_file():
                source_plot_artifacts.append(
                    {
                        "table_group": order,
                        "path": str(plot_path.resolve()),
                        "sha256": sha256_file(plot_path),
                    }
                )
    payload = dict(group_payloads[0])
    payload["shots"] = {
        shot: block
        for group in group_payloads
        for shot, block in group.get("shots", {}).items()
    }
    payload["table"] = {
        "policy": "literal frozen k-file TABLE_DIR grouped by shot",
        "sources_by_shot": plan["identity"]["tables_by_shot"],
    }
    payload["plots"] = [item["path"] for item in source_plot_artifacts]
    payload["source_plot_artifacts"] = source_plot_artifacts
    payload["endpoint_artifacts"] = endpoint_artifacts
    payload["reproduction_provenance"] = {
        "parent_issue": ISSUE,
        "reproduction_id": plan["reproduction_id"],
        "executable": stage1["build_provenance"]["executable"],
        "table_identity": stage1["table_identity"],
        "identity": plan["identity"],
    }
    write_json_atomic(result_path, payload)
    audit = validate_issue_663_evidence(result_path, stage1)
    if not audit["reusable"]:
        failed = {**plan, "status": "failed_validation", "evidence_audit": audit}
        write_json_atomic(manifest, failed)
        raise RuntimeError(
            "reproduced #663 endpoints failed provenance validation: "
            + "; ".join(audit["reasons"])
        )
    merged = merge_issue_663_evidence(output, result_path)
    finished = {
        **plan,
        "status": "succeeded",
        "returncodes": returncodes,
        "result_sha256": sha256_file(result_path),
        "evidence_audit": audit,
        "merged_result_sha256": _canonical_sha(merged),
    }
    write_json_atomic(manifest, finished)
    return finished


def _median_statistic(value: Any) -> float | None:
    if not isinstance(value, Mapping):
        return None
    median = value.get("median")
    if median is None:
        return None
    median = float(median)
    return median if math.isfinite(median) else None


def issue_663_ablation_effects(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Normalize #663 nonlinear ablations by the material-response thresholds.

    These records keep the physical medians as well as their threshold-normalized
    values.  The latter make the final heatmap directly auditable: one is the
    prespecified material-response boundary for every column.
    """
    records: list[dict[str, Any]] = []
    for shot in (41672, 39915, 41524):
        comparisons = payload.get("shots", {}).get(str(shot), {}).get(
            "comparisons", {}
        )
        for variant in ISSUE_663_ABLATION_VARIANTS:
            comparison = comparisons.get(variant)
            if not isinstance(comparison, Mapping):
                continue
            geometry = comparison.get("geometry", {})
            relative = geometry.get("absolute_relative_change", {})
            profiles = geometry.get("profile_relative_rms", {})
            profile_values = [
                value
                for name in (
                    "pressure",
                    "pprime",
                    "ffprime",
                    "jphi_reference_r",
                    "q",
                )
                if (value := _median_statistic(profiles.get(name))) is not None
            ]
            physical = {
                "lcfs_rms_mm": _median_statistic(geometry.get("lcfs_rms_mm")),
                "area_relative": _median_statistic(relative.get("area")),
                "volume_relative": _median_statistic(relative.get("volume")),
                "beta_p_relative": _median_statistic(relative.get("betap")),
                "li_relative": _median_statistic(relative.get("li")),
                "profile_relative_rms": (
                    float(np.sqrt(np.mean(np.square(profile_values))))
                    if profile_values
                    else None
                ),
                "acceptance_change_percentage_points": (
                    abs(float(comparison["acceptance_change_percentage_points"]))
                    if comparison.get("acceptance_change_percentage_points") is not None
                    else None
                ),
            }
            normalized = {
                name: (
                    abs(float(value)) / ISSUE_663_ABLATION_THRESHOLDS[name]
                    if value is not None
                    else None
                )
                for name, value in physical.items()
            }
            records.append(
                {
                    "shot": shot,
                    "variant": variant,
                    "common_produced": geometry.get("common_produced"),
                    "material_response": bool(comparison.get("material_response")),
                    "physical_median": physical,
                    "threshold_normalized": normalized,
                }
            )
    return records


def _issue_663_plot_artifacts(
    payload: Mapping[str, Any], source: Path
) -> list[dict[str, Any]]:
    """Content-bind any source #663 plots without trusting relative paths."""
    records = []
    for item in payload.get("source_plot_artifacts", ()):
        if not isinstance(item, Mapping):
            continue
        path = Path(str(item.get("path", "")))
        if path.is_file() and item.get("sha256") == sha256_file(path):
            records.append(dict(item))
    if records:
        return records
    for value in payload.get("plots", ()):
        path = Path(str(value))
        if not path.is_absolute():
            path = source.parent / path
        if path.is_file():
            records.append(
                {"path": str(path.resolve()), "sha256": sha256_file(path)}
            )
    return records


def merge_issue_663_evidence(output: Path, path: Path) -> dict[str, Any]:
    """Merge qualified ablations or fail with an explicit reproduction list."""
    stage1 = require_stage2_gate(output)
    audit = validate_issue_663_evidence(path, stage1)
    write_json_atomic(Path(output) / "issue_663_evidence_audit.json", audit)
    if not audit["reusable"]:
        raise RuntimeError(
            "#663 endpoints cannot be reused: "
            + "; ".join(audit["reasons"])
            + ". Reproduce the endpoints listed in issue_663_evidence_audit.json."
        )
    results_path = Path(output) / RESULTS_JSON
    payload = json.loads(results_path.read_text(encoding="utf-8"))
    prior_payload = json.loads(Path(path).read_text(encoding="utf-8"))
    continuation = payload.get("continuation_results", ())
    merged = merge_nonlinear_family_evidence(
        payload,
        continuation_records=continuation,
        prior_evidence=audit["evidence"],
    )
    merged["issue_663_evidence"] = audit
    merged["constraint_ablation_effects"] = issue_663_ablation_effects(
        prior_payload
    )
    merged["issue_663_source_plot_artifacts"] = _issue_663_plot_artifacts(
        prior_payload, Path(path)
    )
    write_report(merged, output)
    return merged


def build_stage2_plan(output: Path) -> dict[str, Any]:
    """Build the nonlinear plan exclusively from a passing Stage-1 run."""
    stage1 = require_stage2_gate(output)
    reference_cases = _stage1_case_payloads(stage1)
    for reference, payload in reference_cases.items():
        if payload.get("status") != "succeeded" or not payload.get("analysis"):
            raise RuntimeError(f"Stage-1 case is not complete: {payload.get('case_id')}")
        restart = Path(payload.get("restart_file") or "")
        sidecar = Path(payload.get("linearization_file") or "")
        if not restart.is_file() or not sidecar.is_file():
            raise RuntimeError(
                f"Stage-1 continuation artifacts are missing for {payload.get('case_id')}"
            )
    expected = {item.key for item in REFERENCE_SLICES}
    if set(reference_cases) != expected:
        raise RuntimeError("Stage-1 manifest does not contain exactly the five reference slices")

    references: list[dict[str, Any]] = []
    for reference in REFERENCE_SLICES:
        baseline = reference_cases[reference.key]
        modes = select_target_modes_from_report(baseline["analysis"])
        direction_chains = {
            str(mode.mode_index): [
                asdict(point) for point in directional_continuation_plan(mode)
            ]
            for mode in modes
        }
        references.append(
            {
                "reference": asdict(reference),
                "baseline_case_id": baseline["case_id"],
                "baseline_restart": baseline["restart_file"],
                "baseline_restart_sha256": sha256_file(Path(baseline["restart_file"])),
                "baseline_linearization": baseline["linearization_file"],
                "baseline_linearization_sha256": sha256_file(
                    Path(baseline["linearization_file"])
                ),
                "selected_modes": [asdict(mode) for mode in modes],
                "direction_chains": direction_chains,
                "diamagnetic_chains": [
                    asdict(point) for point in diamagnetic_continuation_plan()
                ],
            }
        )
    payload = {
        "schema_version": SCHEMA_VERSION,
        "issue": ISSUE,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "stage1_manifest_sha256": stage1.get("_manifest_sha256")
        or sha256_file(Path(output) / STUDY_MANIFEST),
        "stage1_gate": stage1["stage1_gate"],
        "status": "planned",
        "restart_validation": {
            "reference": "41672:331",
            "required_cases": [
                "cold_repeat_1",
                "cold_repeat_2",
                "unchanged_restart",
            ],
            "maximum_restart_difference_multiple_of_repeat_noise": 10.0,
            "passed": False,
        },
        "failure_policy": {
            "maximum_failure_bisections": 3,
            "deterministic_repeats": 2,
            "branch_interval_log10_maximum": 0.025,
            "never_bridge_failure": True,
        },
        "linearity_gate": {
            "alphas": [0.125, 0.25],
            "derivative_relative_difference_maximum": 0.05,
            "derivative_cosine_minimum": 0.995,
        },
        "outlier_gate": {
            "reference": "41672:347",
            "weak_subspace_projection_minimum": 0.80,
            "beta_li_prediction_relative_error_maximum": 0.25,
            "track_retained_mask": True,
        },
        "references": references,
    }
    write_json_atomic(Path(output) / STAGE2_PLAN, payload)
    return payload


def _stage1_case_payloads(stage1: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    validated = stage1.get("_validated_case_payloads")
    if isinstance(validated, Mapping):
        return {str(key): value for key, value in validated.items()}
    result = {}
    for summary in stage1.get("cases", ()):
        path = Path(summary["workdir"]) / STAGE1_CASE_MANIFEST
        payload = json.loads(path.read_text(encoding="utf-8"))
        result[_reference_from_record(payload).key] = payload
    return result


def execute_restart_proof(output: Path) -> dict[str, Any]:
    """Establish deterministic cold-run noise and validate native restart."""
    stage1 = require_stage2_gate(output)
    plan_path = Path(output) / STAGE2_PLAN
    plan = (
        json.loads(plan_path.read_text(encoding="utf-8"))
        if plan_path.is_file()
        else build_stage2_plan(output)
    )
    cases = _stage1_case_payloads(stage1)
    reference = next(item for item in REFERENCE_SLICES if item.key == "41672:331")
    baseline = cases[reference.key]
    if baseline.get("snapshot") is None:
        raise RuntimeError("restart proof requires the Stage-1 baseline snapshot")
    identity = baseline["identity"]
    executable = Path(identity["build_provenance"]["executable"]["path"])
    if sha256_file(executable) != identity["executable_sha256"]:
        raise RuntimeError("EFIT executable changed after Stage 1")
    kfile = Path(baseline["kfile"])
    scientific = fixed_scientific_config()
    if scientific.sha256 != identity["scientific_sha256"]:
        raise RuntimeError("frozen scientific configuration changed after Stage 1")

    common = {
        "stage": 2,
        "reference": reference,
        "kfile": kfile,
        "executable": executable,
        "scientific_sha256": scientific.sha256,
        "table_identity": identity["table_identity"],
        "analysis_identity": {
            "purpose": "restart_validation",
            "schema": SCHEMA_VERSION,
            "baseline_linearization_sha256": sha256_file(
                Path(baseline["linearization_file"])
            ),
            "baseline_retained_mask": baseline["snapshot"].get("retained_mask"),
        },
        "kfile_semantic_audit": identity.get("kfile_semantic_audit", {}),
        "build_provenance": identity["build_provenance"],
        "require_linearization": False,
    }
    baseline_retained_mask = baseline["snapshot"].get("retained_mask")
    if baseline_retained_mask is None:
        raise RuntimeError("restart proof requires the baseline retained singular mask")
    cold_cases = []
    for index in (1, 2):
        cold_cases.append(
            execute_nonlinear_case(
                CaseSpec(kind=f"restart_proof_cold_repeat_{index}", **common),
                output=output,
                scientific=scientific,
                snapshot_retained_mask=baseline_retained_mask,
            )
        )
    baseline_restart = Path(baseline["restart_file"])
    restart_case = execute_nonlinear_case(
        CaseSpec(
            kind="restart_proof_unchanged_restart",
            restart_parent=baseline_restart,
            parent_case_id=baseline["case_id"],
            chain="restart_proof",
            **common,
        ),
        output=output,
        scientific=scientific,
        snapshot_retained_mask=baseline_retained_mask,
    )
    all_cases = [baseline, *cold_cases, restart_case]
    successful = all(item.get("status") == "succeeded" for item in all_cases)
    restart_control_hashes = {
        item.get("restart_control_sha256") for item in all_cases
    }
    restart_control_binding_passed = bool(
        len(restart_control_hashes) == 1 and None not in restart_control_hashes
    )
    repeat_noise = math.inf
    restart_difference = math.inf
    stage1_cold_displacements: list[float] = []
    if successful:
        repeat_metrics, _ = compare_snapshots(
            cold_cases[0]["snapshot"], cold_cases[1]["snapshot"]
        )
        repeat_noise = nonlinear_displacement(repeat_metrics)
        for cold in cold_cases:
            metrics, _ = compare_snapshots(cold["snapshot"], baseline["snapshot"])
            stage1_cold_displacements.append(nonlinear_displacement(metrics))
        restart_metrics, _ = compare_snapshots(
            restart_case["snapshot"], baseline["snapshot"]
        )
        restart_difference = nonlinear_displacement(restart_metrics)
    passed = bool(
        successful
        and restart_control_binding_passed
        and restart_difference <= 10.0 * repeat_noise
    )
    reasons = []
    if not successful:
        reasons.append("one or more cold/restart cases failed")
    if not restart_control_binding_passed:
        reasons.append(
            "cold/write/restart runs do not share one original-control SHA-256"
        )
    if successful and not passed:
        reasons.append(
            f"restart displacement {restart_difference:.6g} exceeds "
            f"10x repeat noise ({10.0 * repeat_noise:.6g})"
        )
    proof = {
        "reference": reference.key,
        "passed": passed,
        "repeat_noise": repeat_noise,
        "stage1_to_cold_displacements": stage1_cold_displacements,
        "restart_displacement": restart_difference,
        "maximum_restart_difference_multiple_of_repeat_noise": 10.0,
        "restart_control_sha256": (
            next(iter(restart_control_hashes))
            if restart_control_binding_passed
            else None
        ),
        "restart_control_binding_passed": restart_control_binding_passed,
        "reasons": reasons,
        "case_ids": [item["case_id"] for item in (*cold_cases, restart_case)],
        "cold_case_ids": [item["case_id"] for item in cold_cases],
        "restart_case_id": restart_case["case_id"],
        "parent_restart": str(baseline_restart),
        "parent_restart_sha256": sha256_file(baseline_restart),
        "stage1_manifest_sha256": stage1.get("_manifest_sha256")
        or sha256_file(Path(output) / STUDY_MANIFEST),
        "case_manifests": _artifact_records(
            Path(item["workdir"]) / STAGE1_CASE_MANIFEST
            for item in (*cold_cases, restart_case)
        ),
    }
    proof_path = write_json_atomic(Path(output) / RESTART_PROOF, proof)
    plan["restart_validation"] = {
        **proof,
        "artifact": {
            "path": str(proof_path.resolve()),
            "sha256": sha256_file(proof_path),
        },
    }
    plan["status"] = "restart_validated" if passed else "restart_blocked"
    write_json_atomic(plan_path, plan)
    return plan["restart_validation"]


def require_restart_proof(
    output: Path,
    stage1: Mapping[str, Any],
    plan: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Reload and independently reproduce the unchanged-control restart proof."""
    record = plan.get("restart_validation")
    if not isinstance(record, Mapping) or record.get("passed") is not True:
        raise RuntimeError("native restart proof is not passing")
    artifact = record.get("artifact")
    _validated_provenance_artifact(artifact, label="restart proof")
    stored = json.loads(Path(artifact["path"]).read_text(encoding="utf-8"))
    expected_record = {key: value for key, value in record.items() if key != "artifact"}
    if stored != expected_record:
        raise RuntimeError("embedded restart proof differs from its hashed artifact")
    stage1_sha256 = stage1.get("_manifest_sha256") or sha256_file(
        Path(output) / STUDY_MANIFEST
    )
    if stored.get("stage1_manifest_sha256") != stage1_sha256:
        raise RuntimeError("restart proof is bound to a stale Stage-1 manifest")
    baselines = _stage1_case_payloads(stage1)
    baseline = baselines.get("41672:331")
    if baseline is None:
        raise RuntimeError("restart proof baseline is absent")
    baseline_restart = Path(str(baseline.get("restart_file", "")))
    if (
        not baseline_restart.is_file()
        or stored.get("parent_restart_sha256") != sha256_file(baseline_restart)
        or Path(str(stored.get("parent_restart", ""))).resolve()
        != baseline_restart.resolve()
    ):
        raise RuntimeError("restart proof baseline restart is absent or changed")

    manifest_records = stored.get("case_manifests", ())
    manifest_by_case: dict[str, Mapping[str, Any]] = {}
    for item in manifest_records:
        _validated_provenance_artifact(item, label="restart-proof case manifest")
        manifest = json.loads(Path(item["path"]).read_text(encoding="utf-8"))
        identity = manifest.get("identity")
        case_id = str(manifest.get("case_id", ""))
        if (
            not isinstance(identity, Mapping)
            or not case_id.endswith(_canonical_sha(identity)[:12])
            or resumable_case_manifest(Path(item["path"]), identity) is None
            or manifest.get("status") != "succeeded"
            or identity.get("require_linearization") is not False
        ):
            raise RuntimeError(f"restart-proof case {case_id!r} is not resumable")
        compare_snapshots(manifest["snapshot"], manifest["snapshot"])
        manifest_by_case[case_id] = manifest
    cold_ids = tuple(str(value) for value in stored.get("cold_case_ids", ()))
    restart_id = str(stored.get("restart_case_id", ""))
    if (
        len(cold_ids) != 2
        or len(set(cold_ids)) != 2
        or set(manifest_by_case) != {*cold_ids, restart_id}
    ):
        raise RuntimeError("restart proof does not bind exactly two cold and one restart case")
    cold = [manifest_by_case[value] for value in cold_ids]
    restarted = manifest_by_case[restart_id]
    baseline_identity = baseline["identity"]
    for item in cold:
        identity = item["identity"]
        if (
            identity.get("restart_parent_sha256") is not None
            or identity.get("kfile_sha256") != baseline_identity.get("kfile_sha256")
            or item.get("native_restart_audit", {}).get("physical_state_loaded")
            or item.get("native_restart_audit", {}).get("objective_history_reused")
        ):
            raise RuntimeError("restart-proof cold case has stale controls or a warm parent")
    restart_identity = restarted["identity"]
    if (
        restart_identity.get("restart_parent_sha256")
        != sha256_file(baseline_restart)
        or restart_identity.get("parent_case_id") != baseline.get("case_id")
        or restart_identity.get("kfile_sha256") != baseline_identity.get("kfile_sha256")
    ):
        raise RuntimeError("unchanged restart case is not bound to the Stage-1 parent")
    restart_audit = restarted.get("native_restart_audit", {})
    if not (
        restart_audit.get("physical_state_loaded") is True
        and restart_audit.get("signature_match") is True
        and restart_audit.get("objective_history_reused") is True
    ):
        raise RuntimeError("unchanged restart has no native state/history-load proof")
    signatures = {
        baseline.get("restart_control_sha256"),
        *(item.get("restart_control_sha256") for item in cold),
        restarted.get("restart_control_sha256"),
    }
    if len(signatures) != 1 or None in signatures:
        raise RuntimeError("restart proof control signatures differ")
    repeat_metrics, _ = compare_snapshots(cold[0]["snapshot"], cold[1]["snapshot"])
    repeat_noise = nonlinear_displacement(repeat_metrics)
    restart_metrics, _ = compare_snapshots(
        restarted["snapshot"], baseline["snapshot"]
    )
    restart_difference = nonlinear_displacement(restart_metrics)
    if (
        not math.isclose(
            float(stored.get("repeat_noise", math.nan)),
            repeat_noise,
            rel_tol=1.0e-12,
            abs_tol=1.0e-12,
        )
        or not math.isclose(
            float(stored.get("restart_displacement", math.nan)),
            restart_difference,
            rel_tol=1.0e-12,
            abs_tol=1.0e-12,
        )
        or restart_difference > 10.0 * repeat_noise
    ):
        raise RuntimeError("restart proof displacement/noise cannot be reproduced")
    return record


def _update_case_payload(payload: Mapping[str, Any], updates: Mapping[str, Any]) -> dict[str, Any]:
    merged = {**dict(payload), **dict(updates)}
    write_json_atomic(
        Path(merged["workdir"]) / STAGE1_CASE_MANIFEST,
        merged,
    )
    return merged


def _annotate_case_payload(
    payload: Mapping[str, Any], updates: Mapping[str, Any]
) -> dict[str, Any]:
    """Persist a real case annotation while keeping pure runner tests usable."""
    if payload.get("workdir"):
        return _update_case_payload(payload, updates)
    return {**dict(payload), **dict(updates)}


def replace_control_records(
    existing: Sequence[Mapping[str, Any]],
    replacement: Sequence[Mapping[str, Any]],
    *,
    control: str,
) -> list[dict[str, Any]]:
    """Replace one control family without erasing other Stage-2 evidence."""
    return [
        *(dict(item) for item in existing if item.get("control") != control),
        *(dict(item) for item in replacement),
    ]


def _diamagnetic_point(
    *,
    output: Path,
    reference: ReferenceSlice,
    baseline: Mapping[str, Any],
    parent: Mapping[str, Any],
    exponent: float,
    chain: str,
    order: int,
    repeat_noise: float,
    attempt: str = "primary",
    cold_start: bool = False,
) -> dict[str, Any]:
    """Run one warm-started FWTDLC point and attach displacement diagnostics."""
    if parent.get("status") != "succeeded" or not parent.get("restart_file"):
        raise RuntimeError(f"{chain}: parent case is not a successful restart source")
    identity = baseline["identity"]
    source_kfile = Path(baseline["kfile"])
    scale = 10.0**exponent
    slug = str(exponent).replace("-", "m").replace(".", "p")
    derived = (
        Path(output)
        / "stage2"
        / "derived_kfiles"
        / f"s{reference.shot}_t{reference.time_ms:05d}"
        / f"{chain}_log10_{slug}_{attempt}"
        / source_kfile.name
    )
    derivation = derive_diamagnetic_kfile(source_kfile, derived, scale)
    scientific = scientific_with_diamagnetic_scale(scale)
    executable = Path(identity["build_provenance"]["executable"]["path"])
    spec = CaseSpec(
        stage=2,
        reference=reference,
        kind=f"{chain}_log10_{slug}_{attempt}",
        kfile=derived,
        executable=executable,
        scientific_sha256=scientific.sha256,
        table_identity=identity["table_identity"],
        analysis_identity={
            "purpose": "diamagnetic_continuation",
            "baseline_linearization_sha256": sha256_file(
                Path(baseline["linearization_file"])
            ),
            "cutoff_multipliers": [0.1, 1.0, 10.0],
        },
        kfile_semantic_audit={
            "baseline": identity.get("kfile_semantic_audit", {}),
            "allowed_change": "FWTDLC",
            "derivation": derivation,
        },
        build_provenance=identity["build_provenance"],
        objective_scale=scale,
        restart_parent=(None if cold_start else Path(parent["restart_file"])),
        parent_case_id=(None if cold_start else parent["case_id"]),
        chain=chain,
        target=scale,
    )
    payload = execute_nonlinear_case(spec, output=output, scientific=scientific)
    updates: dict[str, Any] = {
        "reference": reference.key,
        "control": "diamagnetic_flux_scale",
        "chain": chain,
        "order": order,
        "attempt": attempt,
        "cold_start": cold_start,
        "log10_scale": exponent,
        "requested_value": scale,
        "kfile_derivation": derivation,
        "converged": payload.get("status") == "succeeded",
        "beta_p": None,
        "li": None,
    }
    if payload.get("status") == "succeeded":
        scalars = payload["snapshot"]["afile"]["scalars"]
        baseline_metrics, _ = compare_snapshots(payload["snapshot"], baseline["snapshot"])
        adjacent_metrics, rank_switch = compare_snapshots(
            payload["snapshot"], parent["snapshot"]
        )
        baseline_d = nonlinear_displacement(baseline_metrics)
        adjacent_d = nonlinear_displacement(adjacent_metrics)
        updates.update(
            {
                **asdict(baseline_metrics),
                "beta_p": float(scalars["betap"]),
                "li": float(scalars["li"]),
                "displacement": baseline_d,
                "adjacent_displacement": adjacent_d,
                "response_onset": (
                    baseline_d >= 0.1 and baseline_d >= 10.0 * repeat_noise
                ),
                "material_branch": False,
                "rank_switching": rank_switch,
                "failed": False,
            }
        )
    else:
        updates.update(
            {
                "displacement": None,
                "adjacent_displacement": None,
                "response_onset": False,
                "material_branch": False,
                "rank_switching": False,
                "failed": False,
            }
        )
    return _update_case_payload(payload, updates)


def _direction_point(
    *,
    output: Path,
    reference: ReferenceSlice,
    baseline: Mapping[str, Any],
    parent: Mapping[str, Any],
    problem: Any,
    report: Any,
    mode: TargetMode,
    alpha: float,
    target: float,
    chain: str,
    order: int,
    repeat_noise: float,
    attempt: str = "primary",
    cold_start: bool = False,
) -> dict[str, Any]:
    if parent.get("status") != "succeeded" or not parent.get("restart_file"):
        raise RuntimeError(f"{chain}: parent case is not a successful restart source")
    target_slug = f"{target:.12g}".replace("-", "m").replace(".", "p")
    direction_path = (
        Path(output)
        / "stage2"
        / "directions"
        / f"s{reference.shot}_t{reference.time_ms:05d}"
        / f"mode_{mode.mode_index}"
        / f"target_{target_slug}_{attempt}.nc"
    )
    direction = write_direction_control(
        direction_path,
        problem=problem,
        report=report,
        mode_index=mode.mode_index,
        target=target,
    )
    identity = baseline["identity"]
    scientific = fixed_scientific_config()
    spec = CaseSpec(
        stage=2,
        reference=reference,
        kind=f"{chain}_target_{target_slug}_{attempt}",
        kfile=Path(baseline["kfile"]),
        executable=Path(identity["build_provenance"]["executable"]["path"]),
        scientific_sha256=scientific.sha256,
        table_identity=identity["table_identity"],
        analysis_identity={
            "purpose": "directional_continuation",
            "baseline_linearization_sha256": problem.sha256,
            "parameter_order_sha256": direction["parameter_order_sha256"],
        },
        kfile_semantic_audit=identity.get("kfile_semantic_audit", {}),
        build_provenance=identity["build_provenance"],
        direction_file=direction_path,
        restart_parent=(None if cold_start else Path(parent["restart_file"])),
        parent_case_id=(None if cold_start else parent["case_id"]),
        chain=chain,
        target=target,
    )
    payload = execute_nonlinear_case(spec, output=output, scientific=scientific)
    updates: dict[str, Any] = {
        "reference": reference.key,
        "control": "direction",
        "chain": chain,
        "order": order,
        "attempt": attempt,
        "mode_index": mode.mode_index,
        "mode_selector": mode.selector,
        "alpha": alpha,
        "requested_value": target,
        "direction_control": direction,
        "cold_start": cold_start,
        "converged": payload.get("status") == "succeeded",
    }
    if payload.get("status") == "succeeded":
        baseline_metrics, _ = compare_snapshots(payload["snapshot"], baseline["snapshot"])
        adjacent_metrics, rank_switch = compare_snapshots(
            payload["snapshot"], parent["snapshot"]
        )
        displacement = nonlinear_displacement(baseline_metrics)
        updates.update(
            {
                **asdict(baseline_metrics),
                "displacement": displacement,
                "adjacent_displacement": nonlinear_displacement(adjacent_metrics),
                "response_onset": (
                    displacement >= 0.1 and displacement >= 10.0 * repeat_noise
                ),
                "material_branch": False,
                "rank_switching": rank_switch,
                "failed": False,
            }
        )
    else:
        updates.update(
            {
                "displacement": None,
                "adjacent_displacement": None,
                "response_onset": False,
                "material_branch": False,
                "rank_switching": False,
                "failed": False,
            }
        )
    return _update_case_payload(payload, updates)


def _recover_diamagnetic_failure(
    *,
    output: Path,
    reference: ReferenceSlice,
    baseline: Mapping[str, Any],
    parent: Mapping[str, Any],
    parent_exponent: float,
    target_exponent: float,
    chain: str,
    order: int,
    repeat_noise: float,
    failed_target: Mapping[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Try three bisections and confirm an unrecovered target twice."""
    history: list[dict[str, Any]] = [dict(failed_target)]
    low_exponent = parent_exponent
    high_exponent = target_exponent
    last_good = dict(parent)
    target = dict(failed_target)
    for level in range(1, 4):
        midpoint = 0.5 * (low_exponent + high_exponent)
        middle = _diamagnetic_point(
            output=output,
            reference=reference,
            baseline=baseline,
            parent=last_good,
            exponent=midpoint,
            chain=f"{chain}_failure_refinement",
            order=level,
            repeat_noise=repeat_noise,
            attempt=f"level_{level}",
        )
        history.append(middle)
        if middle["status"] == "succeeded":
            low_exponent, last_good = midpoint, middle
            target = _diamagnetic_point(
                output=output,
                reference=reference,
                baseline=baseline,
                parent=last_good,
                exponent=target_exponent,
                chain=chain,
                order=order,
                repeat_noise=repeat_noise,
                attempt=f"retry_{level}",
            )
            history.append(target)
            if target["status"] == "succeeded":
                return target, history
        else:
            high_exponent = midpoint

    repeats = []
    consecutive_failures = 0
    for repeat_index in (1, 2):
        repeated = _diamagnetic_point(
            output=output,
            reference=reference,
            baseline=baseline,
            parent=last_good,
            exponent=target_exponent,
            chain=chain,
            order=order,
            repeat_noise=repeat_noise,
            attempt=f"failure_repeat_{repeat_index}",
        )
        repeats.append(repeated)
        history.append(repeated)
        target = repeated
        if repeated["status"] == "succeeded":
            target = _annotate_case_payload(
                target,
                {
                    "failed": False,
                    "bisection_exhausted": True,
                    "deterministic_failure_repeats": consecutive_failures,
                    "nondeterministic_failure": True,
                    "failure_repeat_case_ids": [
                        item["case_id"] for item in repeats
                    ],
                    "failure_refinement_case_ids": [
                        item["case_id"] for item in history[1:]
                    ],
                },
            )
            history[-1] = target
            return target, history
        consecutive_failures += 1
    target = _annotate_case_payload(
        target,
        {
            "failed": True,
            "bisection_exhausted": True,
            "deterministic_failure_repeats": consecutive_failures,
            "nondeterministic_failure": False,
            "failure_repeat_case_ids": [item["case_id"] for item in repeats],
            "failure_refinement_case_ids": [item["case_id"] for item in history[1:]],
        },
    )
    history[-1] = target
    return target, history


def _localize_branch(
    *,
    output: Path,
    reference: ReferenceSlice,
    baseline: Mapping[str, Any],
    lower: Mapping[str, Any],
    upper: Mapping[str, Any],
    lower_exponent: float,
    upper_exponent: float,
    chain: str,
    repeat_noise: float,
) -> dict[str, Any]:
    """Bisect a physics jump or converged/failed boundary to <=0.025 dex."""
    low, high = dict(lower), dict(upper)
    low_exp, high_exp = lower_exponent, upper_exponent
    cases = []
    level = 0
    transition_kind = "physics_branch"
    while abs(high_exp - low_exp) > 0.025:
        level += 1
        midpoint = 0.5 * (low_exp + high_exp)
        middle = _diamagnetic_point(
            output=output,
            reference=reference,
            baseline=baseline,
            parent=low,
            exponent=midpoint,
            chain=f"{chain}_branch_localization",
            order=level,
            repeat_noise=repeat_noise,
            attempt=f"level_{level}",
        )
        cases.append(middle)
        if middle["status"] != "succeeded":
            repeats: list[dict[str, Any]] = []
            for repeat_index in (1, 2):
                repeated = _diamagnetic_point(
                    output=output,
                    reference=reference,
                    baseline=baseline,
                    parent=low,
                    exponent=midpoint,
                    chain=f"{chain}_branch_localization",
                    order=level,
                    repeat_noise=repeat_noise,
                    attempt=f"level_{level}_failure_repeat_{repeat_index}",
                )
                repeats.append(repeated)
                cases.append(repeated)
                if repeated["status"] == "succeeded":
                    repeated = _annotate_case_payload(
                        repeated,
                        {
                            "nondeterministic_failure": True,
                            "failed": False,
                            "bisection_exhausted": True,
                            "deterministic_failure_repeats": repeat_index - 1,
                            "failure_repeat_case_ids": [
                                item["case_id"] for item in repeats
                            ],
                        },
                    )
                    cases[-1] = repeated
                    return {
                        "lower_log10": min(low_exp, high_exp),
                        "upper_log10": max(low_exp, high_exp),
                        "width_log10": abs(high_exp - low_exp),
                        "lower_case_id": low["case_id"],
                        "upper_case_id": high["case_id"],
                        "case_ids": [item["case_id"] for item in cases],
                        "resolved": False,
                        "transition_kind": "nondeterministic_outcome",
                        "reason": "branch-localization failure was not repeatable",
                        "_case_records": cases,
                    }
            qualified = _annotate_case_payload(
                repeats[-1],
                {
                    "failed": True,
                    "bisection_exhausted": True,
                    "deterministic_failure_repeats": 2,
                    "nondeterministic_failure": False,
                    "failure_repeat_case_ids": [
                        item["case_id"] for item in repeats
                    ],
                },
            )
            cases[-1] = qualified
            high, high_exp = qualified, midpoint
            transition_kind = "deterministic_outcome"
            continue
        if transition_kind == "deterministic_outcome":
            low, low_exp = middle, midpoint
            continue
        interval_metrics, _ = compare_snapshots(middle["snapshot"], low["snapshot"])
        if nonlinear_displacement(interval_metrics) >= 1.0:
            high, high_exp = middle, midpoint
        else:
            low, low_exp = middle, midpoint
    width = abs(high_exp - low_exp)
    if transition_kind == "deterministic_outcome":
        return {
            "lower_log10": min(low_exp, high_exp),
            "upper_log10": max(low_exp, high_exp),
            "width_log10": width,
            "lower_case_id": low["case_id"],
            "upper_case_id": high["case_id"],
            "case_ids": [item["case_id"] for item in cases],
            "adjacent_displacement": None,
            "transition_kind": transition_kind,
            "qualified_failure_case_id": high["case_id"],
            "resolved": width <= 0.025 and _qualified_failure(high),
            "_case_records": cases,
        }
    final_metrics, _ = compare_snapshots(high["snapshot"], low["snapshot"])
    final_displacement = nonlinear_displacement(final_metrics)
    return {
        "lower_log10": min(low_exp, high_exp),
        "upper_log10": max(low_exp, high_exp),
        "width_log10": width,
        "lower_case_id": low["case_id"],
        "upper_case_id": high["case_id"],
        "case_ids": [item["case_id"] for item in cases],
        "adjacent_displacement": final_displacement,
        "transition_kind": transition_kind,
        "resolved": width <= 0.025 and final_displacement >= 1.0,
        "_case_records": cases,
    }


def _blocked_chain_records(
    reference: ReferenceSlice,
    chain: str,
    exponents: Sequence[float],
    start_order: int,
    blocker: str,
) -> list[dict[str, Any]]:
    return [
        {
            "reference": reference.key,
            "chain": chain,
            "order": start_order + index,
            "control": "diamagnetic_flux_scale",
            "log10_scale": exponent,
            "requested_value": 10.0**exponent,
            "status": "blocked",
            "outcome": "blocked_by_prior_failure",
            "blocked_by_case_id": blocker,
            "converged": False,
        }
        for index, exponent in enumerate(exponents)
    ]


def _baseline_continuation_record(
    baseline: Mapping[str, Any], reference: ReferenceSlice, chain: str, order: int = 0
) -> dict[str, Any]:
    scalars = baseline["snapshot"]["afile"]["scalars"]
    return {
        "case_id": baseline["case_id"],
        "reference": reference.key,
        "chain": chain,
        "order": order,
        "control": "diamagnetic_flux_scale",
        "log10_scale": 0.0,
        "requested_value": 1.0,
        "status": "succeeded",
        "outcome": "accepted",
        "converged": True,
        "displacement": 0.0,
        "adjacent_displacement": 0.0,
        "response_onset": False,
        "material_branch": False,
        "rank_switching": False,
        "beta_p": float(scalars["betap"]),
        "li": float(scalars["li"]),
        "beta_p_relative": 0.0,
        "li_relative": 0.0,
        "snapshot": baseline["snapshot"],
        "parameter_state": baseline.get("parameter_state"),
        "restart_file": baseline["restart_file"],
        "linearization_file": baseline.get("linearization_file"),
        "restart_control_sha256": baseline.get("restart_control_sha256"),
        "workdir": baseline.get("workdir"),
        "identity": baseline.get("identity"),
        "artifacts": baseline.get("artifacts", ()),
        "reused_stage1_baseline": True,
    }


def _reference_repeat_noise(
    *,
    output: Path,
    reference: ReferenceSlice,
    baseline: Mapping[str, Any],
    restart_proof: Mapping[str, Any],
) -> tuple[float, Mapping[str, Any] | None]:
    if reference.key == restart_proof.get("reference"):
        return float(restart_proof["repeat_noise"]), None
    identity = baseline["identity"]
    scientific = fixed_scientific_config()
    repeat = execute_nonlinear_case(
        CaseSpec(
            stage=2,
            reference=reference,
            kind="identical_cold_noise_repeat",
            kfile=Path(baseline["kfile"]),
            executable=Path(identity["build_provenance"]["executable"]["path"]),
            scientific_sha256=scientific.sha256,
            table_identity=identity["table_identity"],
            analysis_identity={"purpose": "identical_run_noise"},
            kfile_semantic_audit=identity.get("kfile_semantic_audit", {}),
            build_provenance=identity["build_provenance"],
            chain="identical_cold_noise_repeat",
        ),
        output=output,
        scientific=scientific,
    )
    if repeat.get("status") != "succeeded":
        repeat = _update_case_payload(
            repeat,
            {
                "reference": reference.key,
                "control": "repeat_noise",
                "chain": "identical_cold_noise_repeat",
                "converged": False,
                "displacement": None,
            },
        )
        return math.inf, repeat
    metrics, _ = compare_snapshots(repeat["snapshot"], baseline["snapshot"])
    repeat = _update_case_payload(
        repeat,
        {
            "reference": reference.key,
            "control": "repeat_noise",
            "chain": "identical_cold_noise_repeat",
            "converged": True,
            "displacement": nonlinear_displacement(metrics),
        },
    )
    return float(repeat["displacement"]), repeat


def _execute_diamagnetic_chain(
    *,
    output: Path,
    reference: ReferenceSlice,
    baseline: Mapping[str, Any],
    chain: str,
    exponents: Sequence[float],
    initial_parent: Mapping[str, Any],
    initial_exponent: float,
    repeat_noise: float,
) -> tuple[list[dict[str, Any]], Mapping[str, Any], bool]:
    results: list[dict[str, Any]] = []
    parent = dict(initial_parent)
    parent_exponent = initial_exponent
    complete = True
    for order, exponent in enumerate(exponents, start=1):
        result = _diamagnetic_point(
            output=output,
            reference=reference,
            baseline=baseline,
            parent=parent,
            exponent=exponent,
            chain=chain,
            order=order,
            repeat_noise=repeat_noise,
        )
        refinement: list[dict[str, Any]] = []
        if result["status"] != "succeeded":
            result, refinement = _recover_diamagnetic_failure(
                output=output,
                reference=reference,
                baseline=baseline,
                parent=parent,
                parent_exponent=parent_exponent,
                target_exponent=exponent,
                chain=chain,
                order=order,
                repeat_noise=repeat_noise,
                failed_target=result,
            )
            results.extend(refinement)
        else:
            results.append(result)
        if result["status"] != "succeeded":
            remaining = exponents[order:]
            results.extend(
                _blocked_chain_records(
                    reference,
                    chain,
                    remaining,
                    order + 1,
                    result["case_id"],
                )
            )
            complete = False
            break
        if float(result.get("adjacent_displacement") or 0.0) >= 1.0:
            localized = _localize_branch(
                output=output,
                reference=reference,
                baseline=baseline,
                lower=parent,
                upper=result,
                lower_exponent=parent_exponent,
                upper_exponent=exponent,
                chain=chain,
                repeat_noise=repeat_noise,
            )
            localization_records = list(localized.pop("_case_records", ()))
            if localized["resolved"]:
                cold_replica = _diamagnetic_point(
                    output=output,
                    reference=reference,
                    baseline=baseline,
                    parent=parent,
                    exponent=exponent,
                    chain=f"{chain}_branch_cold_replica",
                    order=order,
                    repeat_noise=repeat_noise,
                    attempt="cold",
                    cold_start=True,
                )
                localized["cold_start_case_id"] = cold_replica["case_id"]
                localized["cold_start_status"] = cold_replica["status"]
                localization_records.append(cold_replica)
                if cold_replica["status"] == "succeeded":
                    hysteresis, _ = compare_snapshots(
                        cold_replica["snapshot"], result["snapshot"]
                    )
                    localized["cold_start_difference"] = nonlinear_displacement(
                        hysteresis
                    )
            result = _update_case_payload(
                result,
                {
                    "branch_localization": localized,
                    "material_branch": bool(localized["resolved"]),
                    "interval_log10": localized["width_log10"],
                },
            )
            # Replace the main record if it was already appended. Refinement
            # histories intentionally retain their own immutable case entries.
            for index in range(len(results) - 1, -1, -1):
                if results[index].get("case_id") == result["case_id"]:
                    results[index] = result
                    break
            results.extend(localization_records)
        parent, parent_exponent = result, exponent
    return results, parent, complete


def execute_diamagnetic_continuation(output: Path) -> dict[str, Any]:
    """Run the mandatory 1..10^4 forward/reverse native-restart paths."""
    stage1 = require_stage2_gate(output)
    plan_path = Path(output) / STAGE2_PLAN
    if not plan_path.is_file():
        build_stage2_plan(output)
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    proof = require_restart_proof(output, stage1, plan)
    baselines = _stage1_case_payloads(stage1)
    log_schedule = diamagnetic_log10_schedule()
    all_records: list[dict[str, Any]] = []
    noise_records = []
    repeat_noise_by_reference: dict[str, float] = {}
    outcome_populations: dict[str, dict[str, int]] = {}
    for reference in REFERENCE_SLICES:
        baseline = baselines[reference.key]
        repeat_noise, noise_case = _reference_repeat_noise(
            output=output,
            reference=reference,
            baseline=baseline,
            restart_proof=proof,
        )
        if noise_case is not None:
            noise_records.append(noise_case)
        repeat_noise_by_reference[reference.key] = repeat_noise
        if not math.isfinite(repeat_noise):
            blocker = noise_case["case_id"] if noise_case else "noise_gate"
            blocked = [
                *_blocked_chain_records(
                    reference, "diamagnetic_increasing", log_schedule, 0, blocker
                ),
                *_blocked_chain_records(
                    reference,
                    "diamagnetic_decreasing",
                    tuple(reversed(log_schedule)),
                    0,
                    blocker,
                ),
            ]
            all_records.extend(blocked)
            statuses = [item["outcome"] for item in blocked]
            outcome_populations[reference.key] = {
                name: statuses.count(name) for name in sorted(set(statuses))
            }
            outcome_populations[reference.key]["noise_gate_passed"] = 0
            continue
        increasing_baseline = _baseline_continuation_record(
            baseline, reference, "diamagnetic_increasing"
        )
        increasing, high, forward_complete = _execute_diamagnetic_chain(
            output=output,
            reference=reference,
            baseline=baseline,
            chain="diamagnetic_increasing",
            exponents=log_schedule[1:],
            initial_parent=baseline,
            initial_exponent=0.0,
            repeat_noise=repeat_noise,
        )
        forward = [increasing_baseline, *increasing]
        all_records.extend(forward)
        if not forward_complete or float(high.get("log10_scale", -1.0)) != 4.0:
            forward_blocker = next(
                (
                    item
                    for item in reversed(increasing)
                    if _qualified_failure(item)
                ),
                high,
            )
            blocked_reverse = _blocked_chain_records(
                reference,
                "diamagnetic_decreasing",
                tuple(reversed(log_schedule)),
                0,
                forward_blocker.get("case_id", "forward_chain"),
            )
            all_records.extend(blocked_reverse)
            statuses = [
                item.get("outcome", item.get("status", "unknown"))
                for item in (*forward, *blocked_reverse)
            ]
            outcome_populations[reference.key] = {
                name: statuses.count(name) for name in sorted(set(statuses))
            }
            outcome_populations[reference.key]["forward_complete"] = 0
            continue
        reverse_high = {
            **dict(high),
            "chain": "diamagnetic_decreasing",
            "order": 0,
            "reused_forward_high_endpoint": True,
        }
        decreasing, _, reverse_complete = _execute_diamagnetic_chain(
            output=output,
            reference=reference,
            baseline=baseline,
            chain="diamagnetic_decreasing",
            exponents=tuple(reversed(log_schedule[:-1])),
            initial_parent=high,
            initial_exponent=4.0,
            repeat_noise=repeat_noise,
        )
        reverse = [reverse_high, *decreasing]
        forward_by_scale = {
            round(float(item["log10_scale"]), 8): item
            for item in forward
            if item.get("status") == "succeeded" and "log10_scale" in item
        }
        for index, item in enumerate(reverse):
            match = forward_by_scale.get(round(float(item.get("log10_scale", -99)), 8))
            if match is None or item.get("status") != "succeeded":
                continue
            hysteresis_metrics, _ = compare_snapshots(item["snapshot"], match["snapshot"])
            difference = nonlinear_displacement(hysteresis_metrics)
            item = {**item, "reverse_difference": difference}
            if difference >= 1.0:
                item["material_branch"] = True
            if item.get("workdir") and not item.get("reused_forward_high_endpoint"):
                item = _update_case_payload(item, item)
            reverse[index] = item
        all_records.extend(reverse)
        statuses = [item.get("outcome", item.get("status", "unknown")) for item in (*forward, *reverse)]
        outcome_populations[reference.key] = {
            name: statuses.count(name) for name in sorted(set(statuses))
        }
        if not reverse_complete:
            outcome_populations[reference.key]["reverse_complete"] = 0

    outlier_assessment = assess_347_diamagnetic_response(
        baselines["41672:347"], all_records
    )
    payload = {
        "schema_version": SCHEMA_VERSION,
        "issue": ISSUE,
        "stage": 2,
        "control": "diamagnetic_flux_scale",
        "restart_validation": proof,
        "noise_cases": noise_records,
        "repeat_noise_by_reference": repeat_noise_by_reference,
        "continuation_results": all_records,
        "outcome_populations": outcome_populations,
        "outlier_assessment": outlier_assessment,
    }
    result_path = Path(output) / "diamagnetic_continuation.json"
    write_json_atomic(result_path, payload)
    results_path = Path(output) / RESULTS_JSON
    if results_path.is_file():
        combined = json.loads(results_path.read_text(encoding="utf-8"))
        combined["continuation_results"] = replace_control_records(
            combined.get("continuation_results", ()),
            all_records,
            control="diamagnetic_flux_scale",
        )
        combined["restart_validation"] = proof
        combined["outcome_populations"] = outcome_populations
        combined["outlier_assessment"] = outlier_assessment
        prior = combined.get("issue_663_evidence", {}).get("evidence", {})
        combined = merge_nonlinear_family_evidence(
            combined,
            continuation_records=combined["continuation_results"],
            prior_evidence=prior,
        )
        combined["diamagnetic_result_sha256"] = sha256_file(result_path)
        write_report(combined, output)
    incomplete = any(
        item.get("status") in {"failed", "blocked"} for item in all_records
    )
    plan["status"] = (
        "diamagnetic_complete_with_failures" if incomplete else "diamagnetic_complete"
    )
    plan["diamagnetic_result"] = str(result_path)
    plan["diamagnetic_result_sha256"] = sha256_file(result_path)
    plan["outlier_gate_result"] = outlier_assessment
    write_json_atomic(plan_path, plan)
    return payload


def _target_for_alpha(mode: TargetMode, alpha: float) -> float:
    return (
        alpha / mode.singular_value
        if mode.threshold_class == "resolved"
        else alpha
    )


def _localize_direction_branch(
    *,
    output: Path,
    reference: ReferenceSlice,
    baseline: Mapping[str, Any],
    parent: Mapping[str, Any],
    candidate: Mapping[str, Any],
    problem: Any,
    report: Any,
    mode: TargetMode,
    parent_alpha: float,
    candidate_alpha: float,
    chain: str,
    repeat_noise: float,
) -> dict[str, Any]:
    """Require a directional jump to persist through three midpoint solves."""
    near, far = dict(parent), dict(candidate)
    near_alpha, far_alpha = parent_alpha, candidate_alpha
    cases = []
    for level in range(1, 4):
        middle_alpha = 0.5 * (near_alpha + far_alpha)
        middle = _direction_point(
            output=output,
            reference=reference,
            baseline=baseline,
            parent=near,
            problem=problem,
            report=report,
            mode=mode,
            alpha=middle_alpha,
            target=_target_for_alpha(mode, middle_alpha),
            chain=f"{chain}_branch_localization",
            order=level,
            repeat_noise=repeat_noise,
            attempt=f"level_{level}",
        )
        cases.append(middle)
        if middle["status"] != "succeeded":
            return {
                "resolved": False,
                "reason": "branch-localization midpoint did not converge",
                "case_ids": [item["case_id"] for item in cases],
                "_case_records": cases,
            }
        near_metrics, _ = compare_snapshots(middle["snapshot"], near["snapshot"])
        if nonlinear_displacement(near_metrics) >= 1.0:
            far, far_alpha = middle, middle_alpha
        else:
            near, near_alpha = middle, middle_alpha
    final_metrics, _ = compare_snapshots(far["snapshot"], near["snapshot"])
    final_jump = nonlinear_displacement(final_metrics)
    return {
        "resolved": final_jump >= 1.0,
        "near_alpha": near_alpha,
        "far_alpha": far_alpha,
        "width_alpha": abs(far_alpha - near_alpha),
        "adjacent_displacement": final_jump,
        "near_case_id": near["case_id"],
        "far_case_id": far["case_id"],
        "case_ids": [item["case_id"] for item in cases],
        "_case_records": cases,
    }


def _execute_direction_chain(
    *,
    output: Path,
    reference: ReferenceSlice,
    baseline: Mapping[str, Any],
    problem: Any,
    report: Any,
    mode: TargetMode,
    sign: float,
    repeat_noise: float,
    initial_parent: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    name = "negative" if sign < 0 else "positive"
    chain = f"mode_{mode.mode_index}_{name}"
    amplitudes = (0.125, 0.25, 0.5, 1.0, 2.0, 4.0)
    results: list[dict[str, Any]] = []
    parent = dict(initial_parent or baseline)
    parent_alpha = 0.0
    for order, magnitude in enumerate(amplitudes):
        alpha = sign * magnitude
        target = _target_for_alpha(mode, alpha)
        result = _direction_point(
            output=output,
            reference=reference,
            baseline=baseline,
            parent=parent,
            problem=problem,
            report=report,
            mode=mode,
            alpha=alpha,
            target=target,
            chain=chain,
            order=order,
            repeat_noise=repeat_noise,
        )
        history = [result]
        if result["status"] != "succeeded":
            near_alpha, far_alpha = parent_alpha, alpha
            last_good = parent
            for level in range(1, 4):
                middle_alpha = 0.5 * (near_alpha + far_alpha)
                middle = _direction_point(
                    output=output,
                    reference=reference,
                    baseline=baseline,
                    parent=last_good,
                    problem=problem,
                    report=report,
                    mode=mode,
                    alpha=middle_alpha,
                    target=_target_for_alpha(mode, middle_alpha),
                    chain=f"{chain}_failure_refinement",
                    order=level,
                    repeat_noise=repeat_noise,
                    attempt=f"level_{level}",
                )
                history.append(middle)
                if middle["status"] == "succeeded":
                    near_alpha, last_good = middle_alpha, middle
                    result = _direction_point(
                        output=output,
                        reference=reference,
                        baseline=baseline,
                        parent=last_good,
                        problem=problem,
                        report=report,
                        mode=mode,
                        alpha=alpha,
                        target=target,
                        chain=chain,
                        order=order,
                        repeat_noise=repeat_noise,
                        attempt=f"retry_{level}",
                    )
                    history.append(result)
                    if result["status"] == "succeeded":
                        break
                else:
                    far_alpha = middle_alpha
            if result["status"] != "succeeded":
                repeats = []
                consecutive_failures = 0
                for repeat_index in (1, 2):
                    repeated = _direction_point(
                        output=output,
                        reference=reference,
                        baseline=baseline,
                        parent=last_good,
                        problem=problem,
                        report=report,
                        mode=mode,
                        alpha=alpha,
                        target=target,
                        chain=chain,
                        order=order,
                        repeat_noise=repeat_noise,
                        attempt=f"failure_repeat_{repeat_index}",
                    )
                    repeats.append(repeated)
                    history.append(repeated)
                    result = repeated
                    if repeated["status"] == "succeeded":
                        break
                    consecutive_failures += 1
                result = _annotate_case_payload(
                    result,
                    {
                        "failed": result["status"] != "succeeded",
                        "bisection_exhausted": True,
                        "deterministic_failure_repeats": consecutive_failures,
                        "nondeterministic_failure": result["status"] == "succeeded",
                        "failure_repeat_case_ids": [
                            item["case_id"] for item in repeats
                        ],
                        "failure_refinement_case_ids": [
                            item["case_id"] for item in history[1:]
                        ],
                    },
                )
                history[-1] = result
        results.extend(history)
        if result["status"] != "succeeded":
            for blocked_order, blocked_magnitude in enumerate(
                amplitudes[order + 1 :], start=order + 1
            ):
                blocked_alpha = sign * blocked_magnitude
                results.append(
                    {
                        "reference": reference.key,
                        "chain": chain,
                        "order": blocked_order,
                        "control": "direction",
                        "mode_index": mode.mode_index,
                        "alpha": blocked_alpha,
                        "requested_value": _target_for_alpha(mode, blocked_alpha),
                        "status": "blocked",
                        "outcome": "blocked_by_prior_failure",
                        "blocked_by_case_id": result["case_id"],
                        "converged": False,
                    }
                )
            break
        if float(result.get("adjacent_displacement") or 0.0) >= 1.0:
            localized = _localize_direction_branch(
                output=output,
                reference=reference,
                baseline=baseline,
                parent=parent,
                candidate=result,
                problem=problem,
                report=report,
                mode=mode,
                parent_alpha=parent_alpha,
                candidate_alpha=alpha,
                chain=chain,
                repeat_noise=repeat_noise,
            )
            localization_records = list(localized.pop("_case_records", ()))
            cold = _direction_point(
                output=output,
                reference=reference,
                baseline=baseline,
                parent=parent,
                problem=problem,
                report=report,
                mode=mode,
                alpha=alpha,
                target=target,
                chain=f"{chain}_branch_cold_replica",
                order=order,
                repeat_noise=repeat_noise,
                attempt="cold",
                cold_start=True,
            )
            localization_records.append(cold)
            persisted = False
            if cold["status"] == "succeeded":
                branch_metrics, _ = compare_snapshots(cold["snapshot"], parent["snapshot"])
                persisted = nonlinear_displacement(branch_metrics) >= 1.0
            result = _update_case_payload(
                result,
                {
                    "material_branch": bool(localized.get("resolved")),
                    "branch_localization": localized,
                    "branch_cold_replica_case_id": cold["case_id"],
                    "branch_cold_replica_status": cold["status"],
                    "cold_start_branch_persists": persisted,
                },
            )
            for index in range(len(results) - 1, -1, -1):
                if results[index].get("case_id") == result["case_id"]:
                    results[index] = result
                    break
            results.extend(localization_records)
        parent, parent_alpha = result, alpha
    return results


def _direction_linearity(
    records: Sequence[Mapping[str, Any]],
    baseline_state: Sequence[float],
    parameter_scale: Sequence[float],
    response_matrix: np.ndarray,
) -> dict[str, Any]:
    """Compare all four adjacent small-step derivatives in baseline AS norm."""
    baseline = np.asarray(baseline_state, dtype=float)
    scale = np.asarray(parameter_scale, dtype=float)
    matrix = np.asarray(response_matrix, dtype=float)
    if (
        baseline.ndim != 1
        or scale.shape != baseline.shape
        or matrix.ndim != 2
        or matrix.shape[1] != baseline.size
        or np.any(scale <= 0.0)
        or not np.all(np.isfinite(baseline))
        or not np.all(np.isfinite(scale))
        or not np.all(np.isfinite(matrix))
    ):
        raise ValueError("linearity gate requires finite matching baseline A and S")
    states: dict[float, np.ndarray] = {0.0: baseline}
    for item in records:
        alpha = item.get("alpha")
        if item.get("status") != "succeeded" or alpha is None:
            continue
        alpha = float(alpha)
        if alpha in {-0.25, -0.125, 0.125, 0.25}:
            state = np.asarray(item.get("parameter_state", ()), dtype=float)
            if state.shape == baseline.shape and np.all(np.isfinite(state)):
                states[alpha] = state
    required = (-0.25, -0.125, 0.0, 0.125, 0.25)
    if set(states) != set(required):
        return {
            "passed": False,
            "reason": "missing converged -0.25/-0.125/0/+0.125/+0.25 points",
            "adjacent_comparisons": [],
        }
    as_matrix = matrix * scale[np.newaxis, :]
    derivatives = []
    segments = []
    for left, right in zip(required[:-1], required[1:]):
        scaled_derivative = ((states[right] - states[left]) / scale) / (right - left)
        derivatives.append(as_matrix @ scaled_derivative)
        segments.append((left, right))
    result: dict[str, Any] = {"norm": "baseline_row_weighted_AS"}
    comparisons = []
    names = ("negative", "cross_zero", "positive")
    for name, index in zip(names, range(3)):
        left_derivative = derivatives[index]
        right_derivative = derivatives[index + 1]
        relative = _relative_norm(right_derivative, left_derivative)
        denominator = float(
            np.linalg.norm(left_derivative) * np.linalg.norm(right_derivative)
        )
        cosine = (
            float(np.dot(left_derivative, right_derivative) / denominator)
            if denominator > 0.0
            else 1.0
        )
        comparison = {
            "left_segment": list(segments[index]),
            "right_segment": list(segments[index + 1]),
            "derivative_relative_difference": relative,
            "derivative_cosine": cosine,
            "passed": relative <= 0.05 and cosine >= 0.995,
        }
        result[name] = comparison
        comparisons.append(comparison)
    result["adjacent_comparisons"] = comparisons
    result["passed"] = all(item["passed"] for item in comparisons)
    return result


def validate_direction_zero_target(
    case: Mapping[str, Any], *, repeat_noise: float
) -> dict[str, Any]:
    """Gate a one-row constrained solve before any nonzero target is launched."""
    solve_validation = case.get("linear_solve_validation") or {}
    reduced_validation = case.get("reduced_linear_solve_validation") or {}
    reproduction_error = reduced_validation.get("reproduction_relative_error")
    transform_error = reduced_validation.get("transform_relative_error")
    reduced_status = reduced_validation.get("status")
    singular_error = solve_validation.get("singular_value_relative_error")
    equality_residual = solve_validation.get("exact_constraint_residual_norm")
    equality_tolerance = solve_validation.get("exact_constraint_tolerance")
    constrained_solve_validated = bool(
        reproduction_error is not None
        and float(reproduction_error) <= 1.0e-8
        and transform_error is not None
        and float(transform_error) <= 1.0e-10
        and reduced_status == 0
        and singular_error is not None
        and float(singular_error) <= 1.0e-10
        and equality_residual is not None
        and equality_tolerance is not None
        and float(equality_residual) <= float(equality_tolerance)
    )
    expected_method = "truncated_svd_reduced_dgglse_exact_direction"
    attributes = case.get("linearization_attributes") or {}
    algorithm = attributes.get("direction_solver_algorithm")
    solver_method = case.get("solver_method")
    algorithm_validated = bool(
        solver_method == expected_method and algorithm == expected_method
    )
    displacement_value = case.get("displacement")
    displacement = (
        math.inf if displacement_value is None else float(displacement_value)
    )
    maximum = 10.0 * repeat_noise
    passed = bool(
        case.get("status") == "succeeded"
        and case.get("exact_constraint_count") == 1
        and displacement <= maximum
        and constrained_solve_validated
        and algorithm_validated
    )
    return {
        "passed": passed,
        "displacement": displacement,
        "maximum_displacement": maximum,
        "expected_exact_constraint_count": 1,
        "observed_exact_constraint_count": case.get("exact_constraint_count"),
        "constrained_solve_validated": constrained_solve_validated,
        "algorithm_validated": algorithm_validated,
        "solver_method": solver_method,
        "direction_solver_algorithm": algorithm,
        "solver_reproduction_relative_error": reproduction_error,
        "reduced_transform_relative_error": transform_error,
        "reduced_status": reduced_status,
        "singular_value_relative_error": singular_error,
        "equality_residual": equality_residual,
        "equality_tolerance": equality_tolerance,
    }


def execute_directional_continuation(output: Path) -> dict[str, Any]:
    """Run selected weak-mode chains using EFIT's native equality hook."""
    from vaft.code.efit import analyze_efit_identifiability, read_efit_linearization

    stage1 = require_stage2_gate(output)
    plan_path = Path(output) / STAGE2_PLAN
    if not plan_path.is_file():
        build_stage2_plan(output)
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    proof = require_restart_proof(output, stage1, plan)
    baselines = _stage1_case_payloads(stage1)
    all_records: list[dict[str, Any]] = []
    linearity = []
    zero_target_gates = []
    noise_records = []
    repeat_noise_by_reference: dict[str, float] = {}
    for reference in REFERENCE_SLICES:
        baseline = baselines[reference.key]
        repeat_noise, noise_case = _reference_repeat_noise(
            output=output,
            reference=reference,
            baseline=baseline,
            restart_proof=proof,
        )
        if noise_case is not None:
            noise_records.append(noise_case)
        repeat_noise_by_reference[reference.key] = repeat_noise
        if not math.isfinite(repeat_noise):
            continue
        problem = read_efit_linearization(baseline["linearization_file"])
        report = analyze_efit_identifiability(problem)
        modes = select_target_modes_from_report(report.to_dict())
        for mode in modes:
            zero = _direction_point(
                output=output,
                reference=reference,
                baseline=baseline,
                parent=baseline,
                problem=problem,
                report=report,
                mode=mode,
                alpha=0.0,
                target=0.0,
                chain=f"mode_{mode.mode_index}_zero_target_gate",
                order=0,
                repeat_noise=repeat_noise,
                attempt="mandatory",
            )
            zero_gate = validate_direction_zero_target(
                zero, repeat_noise=repeat_noise
            )
            zero_passed = bool(zero_gate["passed"])
            zero = _update_case_payload(
                zero,
                {
                    "zero_target_gate": True,
                    "mode_interpretable": zero_passed,
                    "zero_target_maximum_displacement": zero_gate[
                        "maximum_displacement"
                    ],
                    **zero_gate,
                },
            )
            all_records.append(zero)
            zero_target_gates.append(
                {
                    "reference": reference.key,
                    "mode_index": mode.mode_index,
                    "case_id": zero["case_id"],
                    "scientific_qualification": (
                        None if zero_passed else "nonlinear/unreliable"
                    ),
                    "curvature_uncertainty_reported": False,
                    **zero_gate,
                }
            )
            if not zero_passed:
                for sign in (-1.0, 1.0):
                    name = "negative" if sign < 0 else "positive"
                    for order, magnitude in enumerate((0.125, 0.25, 0.5, 1, 2, 4)):
                        alpha = sign * magnitude
                        all_records.append(
                            {
                                "reference": reference.key,
                                "chain": f"mode_{mode.mode_index}_{name}",
                                "order": order,
                                "control": "direction",
                                "mode_index": mode.mode_index,
                                "alpha": alpha,
                                "requested_value": _target_for_alpha(mode, alpha),
                                "status": "blocked",
                                "outcome": "blocked_by_zero_target_gate",
                                "blocked_by_case_id": zero["case_id"],
                                "converged": False,
                                "mode_interpretable": False,
                            }
                        )
                linearity.append(
                    {
                        "reference": reference.key,
                        "mode_index": mode.mode_index,
                        "passed": False,
                        "reason": "mandatory t=0 warm-start gate failed",
                        "scientific_qualification": "nonlinear/unreliable",
                        "curvature_uncertainty_reported": False,
                    }
                )
                continue
            mode_records = []
            for sign in (-1.0, 1.0):
                mode_records.extend(
                    _execute_direction_chain(
                        output=output,
                        reference=reference,
                        baseline=baseline,
                        problem=problem,
                        report=report,
                        mode=mode,
                        sign=sign,
                        repeat_noise=repeat_noise,
                        initial_parent=zero,
                    )
                )
            all_records.extend(mode_records)
            linearity_result = _direction_linearity(
                mode_records,
                baseline["parameter_state"],
                report.parameter_scale,
                problem.main.weighted_a,
            )
            if not linearity_result["passed"]:
                linearity_result["scientific_qualification"] = "nonlinear/unreliable"
            linearity.append(
                {
                    "reference": reference.key,
                    "mode_index": mode.mode_index,
                    **linearity_result,
                }
            )
    payload = {
        "schema_version": SCHEMA_VERSION,
        "issue": ISSUE,
        "stage": 2,
        "control": "native_direction_constraint",
        "restart_validation": proof,
        "noise_cases": noise_records,
        "repeat_noise_by_reference": repeat_noise_by_reference,
        "zero_target_gates": zero_target_gates,
        "linearity": linearity,
        "continuation_results": all_records,
        "outcome_populations": {
            reference.key: {
                name: outcomes.count(name)
                for name in sorted(set(outcomes))
            }
            for reference in REFERENCE_SLICES
            for outcomes in [[
                str(item.get("outcome", item.get("status", "unknown")))
                for item in all_records
                if item.get("reference") == reference.key
            ]]
        },
    }
    result_path = Path(output) / "directional_continuation.json"
    write_json_atomic(result_path, payload)
    results_path = Path(output) / RESULTS_JSON
    if results_path.is_file():
        combined = json.loads(results_path.read_text(encoding="utf-8"))
        combined["continuation_results"] = replace_control_records(
            combined.get("continuation_results", ()),
            all_records,
            control="direction",
        )
        combined["direction_linearity"] = linearity
        combined["zero_target_gates"] = zero_target_gates
        combined["restart_validation"] = proof
        combined["directional_outcome_populations"] = payload[
            "outcome_populations"
        ]
        combined["directional_result_sha256"] = sha256_file(result_path)
        prior = combined.get("issue_663_evidence", {}).get("evidence", {})
        combined = merge_nonlinear_family_evidence(
            combined,
            continuation_records=combined["continuation_results"],
            prior_evidence=prior,
        )
        write_report(combined, output)
    plan["directional_result"] = str(result_path)
    plan["directional_result_sha256"] = sha256_file(result_path)
    plan["status"] = (
        "directional_complete_with_unreliable_modes"
        if any(not item["passed"] for item in linearity)
        or any(not item["passed"] for item in zero_target_gates)
        else "directional_complete"
    )
    write_json_atomic(plan_path, plan)
    return payload


def continuation_results(
    records: Sequence[Mapping[str, Any]], *, repeat_noise: Mapping[str, float]
) -> list[dict[str, Any]]:
    """Classify normalized continuation records without dropping failures."""
    results = []
    for item in records:
        metrics = TransitionMetrics(
            lcfs_rms_mm=float(item.get("lcfs_rms_mm", 0.0) or 0.0),
            area_relative=float(item.get("area_relative", 0.0) or 0.0),
            volume_relative=float(item.get("volume_relative", 0.0) or 0.0),
            beta_p_relative=float(item.get("beta_p_relative", 0.0) or 0.0),
            li_relative=float(item.get("li_relative", 0.0) or 0.0),
            profile_relative_rms=item.get("profile_relative_rms", {}),
        )
        reference = str(item["reference"])
        classification = classify_transition(
            metrics,
            repeat_noise=float(repeat_noise.get(reference, 0.0)),
            interval_log10=item.get("interval_log10"),
            reverse_difference=item.get("reverse_difference"),
            previous_retained_mask=item.get("previous_retained_mask"),
            retained_mask=item.get("retained_mask"),
            converged=bool(item.get("converged", False)),
            bisection_exhausted=bool(item.get("bisection_exhausted", False)),
            deterministic_failure_repeats=int(
                item.get("deterministic_failure_repeats", 0)
            ),
        )
        results.append({**dict(item), **asdict(classification)})
    return results


def _records_sha256(records: Sequence[Mapping[str, Any]]) -> str:
    normalized = [_json_safe(dict(item)) for item in records]
    normalized.sort(key=lambda item: json.dumps(item, sort_keys=True))
    return _canonical_sha({"records": normalized})


def _local_information_supported(record: Mapping[str, Any]) -> bool:
    """Whether Stage 1 found an actual rank gain or reinforcing effect."""
    gains = record.get("rank_gain_by_cutoff", ())
    labels = {str(value).replace("_", " ") for value in record.get("classification", ())}
    return bool(
        any(float(value) > 0.0 for value in gains if value is not None)
        or "independent information" in labels
        or "reinforcing" in labels
        or float(record.get("weak_singular_value_lift") or 0.0) >= 10.0
        or float(record.get("curvature_inverse_trace_reduction") or 0.0) >= 0.2
    )


def _qualified_failure(record: Mapping[str, Any]) -> bool:
    return bool(
        record.get("case_id")
        and record.get("status") != "succeeded"
        and record.get("failed") is True
        and record.get("bisection_exhausted") is True
        and int(record.get("deterministic_failure_repeats") or 0) >= 2
    )


def _validated_continuation_case(record: Mapping[str, Any]) -> Mapping[str, Any]:
    """Reload one case manifest and revalidate identity, artifacts, and snapshot."""
    case_id = str(record.get("case_id", ""))
    workdir = Path(str(record.get("workdir", "")))
    manifest_path = workdir / STAGE1_CASE_MANIFEST
    if not case_id or not manifest_path.is_file():
        raise ValueError("case manifest is missing")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    identity = manifest.get("identity")
    if (
        not isinstance(identity, Mapping)
        or manifest.get("case_id") != case_id
        or not case_id.endswith(_canonical_sha(identity)[:12])
    ):
        raise ValueError("case identity/artifact validation failed")
    artifacts = manifest.get("artifacts", ())
    artifact_hashes: dict[str, str] = {}
    for artifact in artifacts:
        if not isinstance(artifact, Mapping) or not artifact.get("path"):
            raise ValueError("case artifact record is malformed")
        path = Path(str(artifact["path"])).resolve()
        if not path.is_file() or artifact.get("sha256") != sha256_file(path):
            raise ValueError("case artifact is absent or changed")
        artifact_hashes[str(path)] = str(artifact["sha256"])
    if manifest.get("status") == "succeeded":
        if resumable_case_manifest(manifest_path, identity) is None:
            raise ValueError("successful case artifacts are absent or changed")
        snapshot = manifest.get("snapshot")
        if not isinstance(snapshot, Mapping):
            raise ValueError("successful case has no snapshot")
        compare_snapshots(snapshot, snapshot)
        for key, label in (("gfile_sha256", "g-file"), ("afile_sha256", "a-file")):
            digest = snapshot.get(key)
            if not digest or digest not in artifact_hashes.values():
                raise ValueError(f"successful case does not bind its {label}")
        restart_file = Path(str(manifest.get("restart_file", ""))).resolve()
        if artifact_hashes.get(str(restart_file)) != (
            sha256_file(restart_file) if restart_file.is_file() else None
        ):
            raise ValueError("successful case does not bind its restart output")
        if identity.get("require_linearization") is not False:
            sidecar = Path(str(manifest.get("linearization_file", ""))).resolve()
            if artifact_hashes.get(str(sidecar)) != (
                sha256_file(sidecar) if sidecar.is_file() else None
            ):
                raise ValueError("changed-control case does not bind a native sidecar")
        warm_parent = identity.get("restart_parent_sha256") is not None
        restart_audit = manifest.get("native_restart_audit", {})
        if warm_parent and identity.get("kind") != "restart_proof_unchanged_restart":
            sidecar_audit = manifest.get("sidecar_restart_audit", {})
            if not (
                restart_audit.get("physical_state_loaded") is True
                and restart_audit.get("signature_match") is False
                and restart_audit.get("objective_history_reused") is False
                and restart_audit.get("fresh_response_accepted") is True
                and sidecar_audit.get("physical_state_loaded") is True
                and sidecar_audit.get("signature_match") is False
                and sidecar_audit.get("objective_history_reused") is False
                and sidecar_audit.get("fresh_response_accepted") is True
                and int(sidecar_audit.get("minite_iteration_offset") or 0) > 0
            ):
                raise ValueError(
                    "warm changed-control case lacks native restart/fresh-response proof"
                )
        elif not warm_parent and (
            restart_audit.get("physical_state_loaded") is True
            or restart_audit.get("objective_history_reused") is True
            or manifest.get("sidecar_restart_audit", {}).get(
                "physical_state_loaded"
            ) is True
            or manifest.get("sidecar_restart_audit", {}).get(
                "objective_history_reused"
            ) is True
        ):
            raise ValueError("cold case reports a restored restart state")
        if (
            not warm_parent
            and int(identity.get("stage") or 0) == 2
            and identity.get("require_linearization") is not False
        ):
            sidecar_audit = manifest.get("sidecar_restart_audit", {})
            if sidecar_audit != {
                "physical_state_loaded": False,
                "signature_match": False,
                "objective_history_reused": False,
                "fresh_response_accepted": True,
                "minite_iteration_offset": 0,
            }:
                raise ValueError("cold Stage-2 sidecar has invalid restart provenance")
    else:
        if not artifacts:
            raise ValueError("failed case has no content-bound artifacts")
        if not manifest.get("finished_at") or not manifest.get("outcome"):
            raise ValueError("failed case has no completed runtime evidence")
        if not manifest.get("diagnostic_errors") and manifest.get("slice_status") is None:
            raise ValueError("failed case has no diagnostic or slice-status evidence")
        if identity.get("restart_parent_sha256") is not None:
            restart_audit = manifest.get("native_restart_audit", {})
            if not (
                restart_audit.get("physical_state_loaded") is True
                and restart_audit.get("signature_match") is False
                and restart_audit.get("objective_history_reused") is False
            ):
                raise ValueError("failed warm case lacks native physical-state-load proof")
    if Path(str(manifest.get("workdir", ""))).resolve() != workdir.resolve():
        raise ValueError("aggregate/case workdir differs")
    if record.get("identity") != identity:
        raise ValueError("aggregate/case content identity differs")
    expected_reference = f"{identity.get('shot')}:{identity.get('time_ms')}"
    if record.get("reference") != expected_reference:
        raise ValueError("aggregate/case reference differs")
    if manifest.get("status") != record.get("status"):
        raise ValueError("aggregate/case status differs")
    derived_fields = (
        "identity",
        "snapshot",
        "parameter_state",
        "restart_file",
        "linearization_file",
        "restart_control_sha256",
    )
    if record.get("reused_stage1_baseline") or record.get(
        "reused_forward_high_endpoint"
    ):
        if record.get("reused_stage1_baseline") and not (
            manifest.get("status") == "succeeded"
            and record.get("status") == "succeeded"
            and record.get("outcome") == "accepted"
        ):
            raise ValueError("reused Stage-1 baseline is not an accepted success")
        for field_name in derived_fields:
            if _json_safe(record.get(field_name)) != _json_safe(
                manifest.get(field_name)
            ):
                raise ValueError(
                    f"derived aggregate differs from source case: {field_name}"
                )
    if not record.get("reused_stage1_baseline") and not record.get(
        "reused_forward_high_endpoint"
    ):
        authoritative_fields = (
            "outcome",
            "converged",
            "snapshot",
            "parameter_state",
            "exact_constraint_count",
            "linear_solve_validation",
            "reduced_linear_solve_validation",
            "solver_method",
            "linearization_attributes",
            "native_restart_audit",
            "sidecar_restart_audit",
            "attempt",
            "mode_index",
            "alpha",
            "log10_scale",
            "requested_value",
            "displacement",
            "adjacent_displacement",
            "response_onset",
            "material_branch",
            "rank_switching",
            "failed",
            "bisection_exhausted",
            "deterministic_failure_repeats",
            "nondeterministic_failure",
            "failure_repeat_case_ids",
            "failure_refinement_case_ids",
            "branch_localization",
            "beta_p",
            "li",
            "beta_p_relative",
            "li_relative",
            "zero_target_gate",
            "mode_interpretable",
            "passed",
            "zero_target_maximum_displacement",
            "maximum_displacement",
            "expected_exact_constraint_count",
            "observed_exact_constraint_count",
            "constrained_solve_validated",
            "algorithm_validated",
            "solver_reproduction_relative_error",
            "reduced_transform_relative_error",
            "reduced_status",
            "singular_value_relative_error",
            "equality_residual",
            "equality_tolerance",
        )
        for field_name in authoritative_fields:
            if field_name in record or field_name in manifest:
                if _json_safe(record.get(field_name)) != _json_safe(
                    manifest.get(field_name)
                ):
                    raise ValueError(
                        f"aggregate/case authoritative field differs: {field_name}"
                    )
    if not record.get("reused_stage1_baseline") and not record.get(
        "reused_forward_high_endpoint"
    ):
        if record.get("chain") != identity.get("chain"):
            raise ValueError("aggregate/case chain differs")
        if record.get("control") != manifest.get("control"):
            raise ValueError("aggregate/case control differs")
        requested_value = record.get("requested_value")
        if requested_value is not None and not math.isclose(
            float(requested_value), float(identity.get("target")), rel_tol=1.0e-12, abs_tol=1.0e-12
        ):
            raise ValueError("aggregate coordinate does not match the case target")
    return manifest


def _failure_repeat_errors(
    record: Mapping[str, Any],
    by_case: Mapping[str, Mapping[str, Any]],
    validated_case_ids: set[str] | None,
) -> list[str]:
    """Verify two independent, same-parent failures instead of trusting a count."""
    case_id = str(record.get("case_id", ""))
    prefix = f"{record.get('reference')}/{record.get('chain')}/{case_id}"
    repeat_ids = [str(value) for value in record.get("failure_repeat_case_ids", ())]
    errors: list[str] = []
    if len(repeat_ids) != 2 or len(set(repeat_ids)) != 2:
        return [f"{prefix}: deterministic failure does not bind two unique repeats"]
    if case_id != repeat_ids[-1]:
        errors.append(f"{prefix}: terminal failure is not the final repeat")
    repeat_records = [by_case.get(value) for value in repeat_ids]
    if any(item is None for item in repeat_records):
        errors.append(f"{prefix}: one or more repeat records are absent")
        return errors
    parents = set()
    for repeat_id, item in zip(repeat_ids, repeat_records):
        assert item is not None
        if validated_case_ids is not None and repeat_id not in validated_case_ids:
            errors.append(f"{prefix}: repeat {repeat_id} is not artifact-validated")
        if item.get("status") == "succeeded":
            errors.append(f"{prefix}: repeat {repeat_id} unexpectedly succeeded")
        if (
            item.get("reference") != record.get("reference")
            or item.get("chain") != record.get("chain")
            or item.get("requested_value") != record.get("requested_value")
        ):
            errors.append(f"{prefix}: repeat {repeat_id} has different controls")
        attempt = str(item.get("attempt", ""))
        if not attempt.startswith("failure_repeat_"):
            errors.append(f"{prefix}: repeat {repeat_id} is not labelled as a repeat")
        parents.add(item.get("identity", {}).get("parent_case_id"))
    if len(parents) != 1 or None in parents:
        errors.append(f"{prefix}: repeats do not share one successful warm parent")
    parent_id = next(iter(parents), None)
    parent = by_case.get(str(parent_id)) if parent_id is not None else None
    if (
        parent is None
        or parent.get("status") != "succeeded"
        or (
            validated_case_ids is not None
            and str(parent_id) not in validated_case_ids
        )
    ):
        errors.append(f"{prefix}: repeat parent is absent, failed, or unvalidated")
    return errors


def _schedule_chain_errors(
    records: Sequence[Mapping[str, Any]],
    *,
    reference: str,
    chain: str,
    coordinate: str,
    requested: Sequence[float],
    extra_successes: Sequence[Mapping[str, Any]] = (),
    validated_case_ids: set[str] | None = None,
    allowed_blocker_ids: set[str] | None = None,
) -> list[str]:
    """Validate terminal points and blocked ancestry for one continuation chain."""
    chain_records = [
        item
        for item in records
        if item.get("reference") == reference and item.get("chain") == chain
    ]
    by_case: dict[str, Mapping[str, Any]] = {}
    for item in records:
        if item.get("case_id"):
            by_case[str(item["case_id"])] = item
    for item in extra_successes:
        if item.get("case_id"):
            by_case.setdefault(str(item["case_id"]), item)
    qualified_failures = {
        case_id
        for case_id, item in by_case.items()
        if _qualified_failure(item)
        and (validated_case_ids is None or case_id in validated_case_ids)
    }
    allowed_blocker_ids = allowed_blocker_ids or set()
    errors: list[str] = []
    terminal: list[Mapping[str, Any]] = []
    for value in requested:
        matches = [
            item
            for item in chain_records
            if item.get(coordinate) is not None
            and math.isclose(
                float(item[coordinate]), float(value), rel_tol=0.0, abs_tol=1.0e-8
            )
        ]
        if not matches:
            errors.append(f"{reference}/{chain}: missing {coordinate}={value:g}")
            continue
        item = matches[-1]
        terminal.append(item)
        status = item.get("status")
        if status == "succeeded":
            if item.get("converged") is not True or not item.get("case_id"):
                errors.append(
                    f"{reference}/{chain}/{value:g}: succeeded point is not a "
                    "qualified converged case"
                )
                continue
            if validated_case_ids is not None and str(item["case_id"]) not in validated_case_ids:
                errors.append(
                    f"{reference}/{chain}/{value:g}: succeeded point is not artifact-validated"
                )
            parent_case_id = item.get("identity", {}).get("parent_case_id")
            if parent_case_id is not None:
                parent = by_case.get(str(parent_case_id))
                if (
                    parent is None
                    or parent.get("status") != "succeeded"
                    or (
                        validated_case_ids is not None
                        and str(parent_case_id) not in validated_case_ids
                    )
                ):
                    errors.append(
                        f"{reference}/{chain}/{value:g}: warm parent is absent or failed"
                    )
        elif status == "blocked":
            blocker_id = str(item.get("blocked_by_case_id"))
            blocker = by_case.get(blocker_id)
            ordinary_failure = bool(
                blocker_id in qualified_failures
                and blocker is not None
                and blocker.get("reference") == reference
                and (
                    blocker.get("chain") == chain
                    or (
                        chain == "diamagnetic_decreasing"
                        and blocker.get("chain") == "diamagnetic_increasing"
                        and blocker.get("control") == "diamagnetic_flux_scale"
                    )
                )
            )
            qualified_gate = bool(
                blocker_id in allowed_blocker_ids
                and blocker is not None
                and blocker.get("reference") == reference
                and blocker.get("zero_target_gate") is True
                and blocker.get("mode_interpretable") is False
            )
            if not ordinary_failure and not qualified_gate:
                errors.append(
                    f"{reference}/{chain}/{value:g}: blocked placeholder lacks a "
                    "qualified deterministic-failure ancestor"
                )
        elif _qualified_failure(item):
            errors.extend(
                _failure_repeat_errors(item, by_case, validated_case_ids)
            )
        else:
            errors.append(
                f"{reference}/{chain}/{value:g}: terminal failure is not repeat-qualified"
            )

    first_failure = next(
        (item for item in terminal if _qualified_failure(item)), None
    )
    if first_failure is not None:
        blocker = str(first_failure["case_id"])
        failed_index = terminal.index(first_failure)
        for item in terminal[failed_index + 1 :]:
            if item.get("status") != "blocked" or str(
                item.get("blocked_by_case_id")
            ) != blocker:
                errors.append(
                    f"{reference}/{chain}: chain continues through deterministic failure {blocker}"
                )
                break
    return errors


def _family_record_errors(record: Mapping[str, Any]) -> list[str]:
    """Reject incomplete or internally inconsistent final classifications."""
    reference = str(record.get("reference"))
    family = str(record.get("family"))
    prefix = f"{reference}/{family}"
    allowed = {
        "independent information",
        "reinforcing",
        "inactive/redundant",
        "overwhelmed",
        "structural anchor",
        "accounting-confounded",
        "cutoff-sensitive/inconclusive",
    }
    labels = [str(value).replace("_", " ") for value in record.get("classification", ())]
    errors: list[str] = []
    if not labels:
        errors.append(f"{prefix}: classification is empty")
        return errors
    unknown = sorted(set(labels) - allowed)
    if unknown:
        errors.append(f"{prefix}: unknown classifications: {', '.join(unknown)}")
    primary = set(labels) & {
        "independent information",
        "reinforcing",
        "inactive/redundant",
        "cutoff-sensitive/inconclusive",
    }
    if len(primary) != 1:
        errors.append(f"{prefix}: exactly one local-information label is required")
    if "cutoff-sensitive/inconclusive" in labels:
        errors.append(
            f"{prefix}: rank gain is cutoff-sensitive and needs scientific resolution"
        )
    gains = record.get("rank_gain_by_cutoff")
    try:
        if (
            not isinstance(gains, Sequence)
            or isinstance(gains, (str, bytes))
            or len(gains) != 3
            or any(float(value) < 0.0 or not float(value).is_integer() for value in gains)
        ):
            raise ValueError
    except (TypeError, ValueError, OverflowError):
        errors.append(f"{prefix}: rank_gain_by_cutoff must contain three nonnegative integers")
    for name in (
        "weak_subspace_action_fraction",
        "weak_singular_value_lift",
        "curvature_inverse_trace_reduction",
        "baseline_solver_objective_share",
    ):
        try:
            value = float(record[name])
        except (KeyError, TypeError, ValueError):
            errors.append(f"{prefix}: missing finite {name}")
            continue
        if not math.isfinite(value):
            errors.append(f"{prefix}: {name} is nonfinite")
            continue
        if name in {
            "weak_subspace_action_fraction",
            "curvature_inverse_trace_reduction",
            "baseline_solver_objective_share",
        } and not (-1.0e-12 <= value <= 1.0 + 1.0e-12):
            errors.append(f"{prefix}: {name} is outside [0, 1]")
        if name == "weak_singular_value_lift" and value < 0.0:
            errors.append(f"{prefix}: weak_singular_value_lift is negative")
    if "accounting-confounded" in labels and family != "plasma_current":
        errors.append(f"{prefix}: accounting-confounded is only valid for the Ip family")
    if "structural anchor" in labels and family not in {
        "pf_current",
        "pf_relation",
    }:
        errors.append(f"{prefix}: structural anchor is only valid for PF families")
    try:
        expected = set(classify_family_information(record))
    except (TypeError, ValueError, OverflowError):
        expected = set()
    if expected and set(labels) != expected:
        errors.append(f"{prefix}: classification does not match its evidence fields")
    return errors


def finalize_study(output: Path) -> dict[str, Any]:
    """Write the final acceptance matrix, remaining incomplete when gates fail."""
    stage1 = require_stage2_gate(output)
    output = Path(output)
    results_path = output / RESULTS_JSON
    plan_path = output / STAGE2_PLAN
    if not results_path.is_file() or not plan_path.is_file():
        raise RuntimeError("finalization requires Stage-1 results and a Stage-2 plan")
    payload = json.loads(results_path.read_text(encoding="utf-8"))
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    reasons: list[str] = []
    validated_restart_proof: Mapping[str, Any] | None = None
    try:
        validated_restart_proof = require_restart_proof(output, stage1, plan)
    except Exception as exc:
        reasons.append(
            "native restart proof cannot be reproduced "
            f"({type(exc).__name__}: {exc})"
        )
    artifacts: dict[str, Mapping[str, Any]] = {}
    for key in ("diamagnetic_result", "directional_result"):
        path = Path(plan.get(key, ""))
        expected = plan.get(f"{key}_sha256")
        if not path.is_file() or not expected or sha256_file(path) != expected:
            reasons.append(f"{key} is missing or its hash changed")
            continue
        artifacts[key] = json.loads(path.read_text(encoding="utf-8"))
        if payload.get(f"{key}_sha256") != expected:
            reasons.append(f"combined result is not content-bound to {key}")

    expected_proof_artifact = plan.get("restart_validation", {}).get("artifact", {})
    expected_proof_path = str(expected_proof_artifact.get("path", ""))
    expected_proof_sha256 = str(expected_proof_artifact.get("sha256", ""))
    for label, container in (
        ("combined result", payload),
        ("diamagnetic result", artifacts.get("diamagnetic_result", {})),
        ("directional result", artifacts.get("directional_result", {})),
    ):
        embedded = container.get("restart_validation", {}).get("artifact", {})
        if (
            str(embedded.get("path", "")) != expected_proof_path
            or str(embedded.get("sha256", "")) != expected_proof_sha256
        ):
            reasons.append(f"{label} is not bound to the current restart proof")
    if validated_restart_proof is not None and (
        not expected_proof_sha256
        or expected_proof_sha256 != sha256_file(Path(expected_proof_path))
    ):
        reasons.append("restart proof artifact changed after validation")

    combined_records = payload.get("continuation_results", ())
    for key, control in (
        ("diamagnetic_result", "diamagnetic_flux_scale"),
        ("directional_result", "direction"),
    ):
        artifact = artifacts.get(key)
        if artifact is None:
            continue
        expected_records = artifact.get("continuation_results", ())
        actual_records = [
            item for item in combined_records if item.get("control") == control
        ]
        if _records_sha256(expected_records) != _records_sha256(actual_records):
            reasons.append(f"combined continuation records differ from {key}")
    zero_gates = payload.get("zero_target_gates", ())
    if not zero_gates:
        reasons.append("mandatory directional t=0 gate evidence is missing")
    directional_artifact = artifacts.get("directional_result", {})
    if _records_sha256(zero_gates) != _records_sha256(
        directional_artifact.get("zero_target_gates", ())
    ):
        reasons.append("combined directional t=0 gates are stale or truncated")
    validated_case_ids: set[str] = set()
    validation_candidates = [
        *artifacts.get("diamagnetic_result", {}).get("continuation_results", ()),
        *artifacts.get("diamagnetic_result", {}).get("noise_cases", ()),
        *directional_artifact.get("continuation_results", ()),
        *directional_artifact.get("noise_cases", ()),
    ]
    validation_records_by_id = {
        str(item["case_id"]): item
        for item in validation_candidates
        if item.get("case_id")
    }
    seen_validation_records: set[tuple[str, str]] = set()
    for record in validation_candidates:
        if record.get("status") == "blocked" or not record.get("case_id"):
            continue
        validation_key = (
            str(record["case_id"]),
            str(record.get("chain", "")),
        )
        if validation_key in seen_validation_records:
            continue
        seen_validation_records.add(validation_key)
        try:
            _validated_continuation_case(record)
        except Exception as exc:
            reasons.append(
                f"{record.get('case_id')}: continuation case validation failed "
                f"({type(exc).__name__}: {exc})"
            )
        else:
            validated_case_ids.add(str(record["case_id"]))
    # Bind auxiliary refinement/repeat/cold-replica evidence referenced by a
    # terminal result; a dangling case id must never certify a transition.
    for record in validation_candidates:
        referenced: set[str] = {
            str(value)
            for key in ("failure_refinement_case_ids", "failure_repeat_case_ids")
            for value in record.get(key, ())
        }
        localization = record.get("branch_localization")
        if isinstance(localization, Mapping):
            referenced.update(str(value) for value in localization.get("case_ids", ()))
            for key in (
                "lower_case_id",
                "upper_case_id",
                "near_case_id",
                "far_case_id",
                "cold_start_case_id",
            ):
                if localization.get(key):
                    referenced.add(str(localization[key]))
        if record.get("branch_cold_replica_case_id"):
            referenced.add(str(record["branch_cold_replica_case_id"]))
        missing_references = sorted(referenced - validated_case_ids)
        if missing_references:
            reasons.append(
                f"{record.get('case_id')}: transition evidence references "
                "unvalidated cases: " + ", ".join(missing_references)
            )
        localization = record.get("branch_localization")
        if isinstance(localization, Mapping):
            if localization.get("resolved") is not True:
                reasons.append(
                    f"{record.get('case_id')}: attached branch/outcome localization "
                    "is unresolved"
                )
            elif localization.get("transition_kind") == "deterministic_outcome":
                failure_id = str(localization.get("qualified_failure_case_id", ""))
                failure = validation_records_by_id.get(failure_id, {})
                try:
                    width = float(localization.get("width_log10"))
                except (TypeError, ValueError):
                    width = math.inf
                if (
                    width > 0.025
                    or failure_id not in validated_case_ids
                    or not _qualified_failure(failure)
                ):
                    reasons.append(
                        f"{record.get('case_id')}: outcome transition lacks a "
                        "qualified <=0.025-dex failure boundary"
                    )
                elif failure:
                    reasons.extend(
                        _failure_repeat_errors(
                            failure,
                            validation_records_by_id,
                            validated_case_ids,
                        )
                    )
            else:
                try:
                    jump = float(localization.get("adjacent_displacement"))
                except (TypeError, ValueError):
                    jump = -math.inf
                if jump < 1.0:
                    reasons.append(
                        f"{record.get('case_id')}: localized physics branch has D < 1"
                    )
        elif (
            record.get("status") == "succeeded"
            and float(record.get("adjacent_displacement") or 0.0) >= 1.0
            and record.get("cold_start") is not True
            and record.get("zero_target_gate") is not True
            and "branch_localization" not in str(record.get("chain", ""))
            and "failure_refinement" not in str(record.get("chain", ""))
        ):
            reasons.append(
                f"{record.get('case_id')}: material adjacent jump has no localization"
            )
    for gate_record in zero_gates:
        if str(gate_record.get("case_id")) not in validated_case_ids:
            reasons.append(
                f"{gate_record.get('reference')}/mode_{gate_record.get('mode_index')}: "
                "t=0 case is not artifact-validated"
            )
    diamagnetic_noise = artifacts.get("diamagnetic_result", {}).get(
        "repeat_noise_by_reference", {}
    )
    directional_noise = directional_artifact.get("repeat_noise_by_reference", {})
    if _canonical_sha({"noise": diamagnetic_noise}) != _canonical_sha(
        {"noise": directional_noise}
    ):
        reasons.append("diamagnetic and directional repeat-noise maps differ")
    derived_repeat_noise: dict[str, float] = {}
    baselines = _stage1_case_payloads(stage1)
    for reference in REFERENCE_SLICES:
        if reference.key == "41672:331":
            if validated_restart_proof is not None:
                derived_repeat_noise[reference.key] = float(
                    validated_restart_proof["repeat_noise"]
                )
            continue
        per_artifact = []
        for artifact_key in ("diamagnetic_result", "directional_result"):
            matching = [
                item
                for item in artifacts.get(artifact_key, {}).get("noise_cases", ())
                if item.get("reference") == reference.key
            ]
            if len(matching) != 1:
                reasons.append(
                    f"{reference.key}: {artifact_key} does not bind exactly one noise case"
                )
            else:
                per_artifact.append(matching[0])
        if len(per_artifact) != 2:
            continue
        if per_artifact[0].get("case_id") != per_artifact[1].get("case_id"):
            reasons.append(f"{reference.key}: Stage-2 controls use different noise cases")
            continue
        noise_case = per_artifact[0]
        if (
            noise_case.get("status") != "succeeded"
            or str(noise_case.get("case_id")) not in validated_case_ids
        ):
            reasons.append(f"{reference.key}: repeat-noise case is not a validated success")
            continue
        try:
            metrics, _ = compare_snapshots(
                noise_case["snapshot"], baselines[reference.key]["snapshot"]
            )
            derived_repeat_noise[reference.key] = nonlinear_displacement(metrics)
        except Exception as exc:
            reasons.append(
                f"{reference.key}: repeat noise cannot be rederived "
                f"({type(exc).__name__}: {exc})"
            )
    for reference, value in derived_repeat_noise.items():
        for label, noise_map in (
            ("diamagnetic", diamagnetic_noise),
            ("directional", directional_noise),
        ):
            try:
                recorded = float(noise_map[reference])
            except (KeyError, TypeError, ValueError):
                reasons.append(f"{reference}: {label} repeat noise is missing")
                continue
            if not math.isclose(recorded, value, rel_tol=1.0e-12, abs_tol=1.0e-12):
                reasons.append(f"{reference}: {label} repeat noise cannot be reproduced")
    for gate_record in zero_gates:
        case_id = str(gate_record.get("case_id", ""))
        case = validation_records_by_id.get(case_id)
        noise = derived_repeat_noise.get(str(gate_record.get("reference")))
        if case is None or noise is None:
            continue
        rederived = validate_direction_zero_target(case, repeat_noise=noise)
        recorded = {
            key: gate_record.get(key)
            for key in rederived
        }
        if _json_safe(recorded) != _json_safe(rederived) or case.get(
            "mode_interpretable"
        ) is not bool(rederived["passed"]):
            reasons.append(
                f"{gate_record.get('reference')}/mode_{gate_record.get('mode_index')}: "
                "t=0 gate cannot be rederived from its case and repeat noise"
            )
    outlier = payload.get("outlier_assessment", {})
    if outlier.get("status") != "evaluated" or outlier.get("classification") not in {
        "weak-mode explained",
        "truncation/rank switching",
        "nonlinear/unreliable",
    }:
        reasons.append("41672/347-ms response has no final weak/nonlinear/rank classification")
    prior = payload.get("issue_663_evidence", {})
    if not prior.get("reusable"):
        reasons.append("PF/ablation evidence is absent or provenance-incompatible")
    ablation_effects = payload.get("constraint_ablation_effects", ())
    expected_ablation_keys = {
        (shot, variant)
        for shot in (41672, 39915, 41524)
        for variant in ISSUE_663_ABLATION_VARIANTS
    }
    observed_ablation_keys = {
        (int(item["shot"]), str(item["variant"]))
        for item in ablation_effects
        if item.get("shot") is not None and item.get("variant") is not None
    }
    if observed_ablation_keys != expected_ablation_keys:
        reasons.append("constraint-ablation effect matrix is incomplete")
    if len(observed_ablation_keys) != len(ablation_effects):
        reasons.append("constraint-ablation effect matrix contains duplicate rows")
    for item in ablation_effects:
        normalized = item.get("threshold_normalized", {})
        common_produced = int(item.get("common_produced") or 0)
        acceptance = normalized.get("acceptance_change_percentage_points")
        population_only_material = bool(
            common_produced == 0
            and item.get("material_response") is True
            and acceptance is not None
            and math.isfinite(float(acceptance))
            and float(acceptance) >= 1.0
        )
        try:
            valid_normalized = all(
                normalized.get(name) is not None
                and math.isfinite(float(normalized[name]))
                and float(normalized[name]) >= 0.0
                for name in ISSUE_663_ABLATION_THRESHOLDS
            )
        except (TypeError, ValueError, OverflowError):
            valid_normalized = False
        if not valid_normalized and not population_only_material:
            reasons.append(
                f"constraint-ablation row {item.get('shot')}/{item.get('variant')} "
                "has incomplete metrics"
            )

    diamagnetic_artifact = artifacts.get("diamagnetic_result", {})
    diamagnetic_records = diamagnetic_artifact.get("continuation_results", ())
    requested_logs = diamagnetic_log10_schedule()
    for reference in REFERENCE_SLICES:
        for chain, requested in (
            ("diamagnetic_increasing", requested_logs),
            ("diamagnetic_decreasing", tuple(reversed(requested_logs))),
        ):
            reasons.extend(
                _schedule_chain_errors(
                    diamagnetic_records,
                    reference=reference.key,
                    chain=chain,
                    coordinate="log10_scale",
                    requested=requested,
                    validated_case_ids=validated_case_ids,
                )
            )
        if reference.key not in diamagnetic_artifact.get("outcome_populations", {}):
            reasons.append(f"{reference.key}: outcome population is missing")

    directional_records = directional_artifact.get("continuation_results", ())
    failed_zero_blockers = {
        str(item.get("case_id"))
        for item in zero_gates
        if item.get("passed") is not True and item.get("case_id")
    }
    plan_references = {
        f"{item['reference']['shot']}:{item['reference']['time_ms']}": item
        for item in plan.get("references", ())
    }
    expected_mode_keys: set[tuple[str, int]] = set()
    for reference in REFERENCE_SLICES:
        selected_modes = plan_references.get(reference.key, {}).get("selected_modes", ())
        unique_mode_indices = {
            int(mode["mode_index"])
            for mode in selected_modes
            if mode.get("mode_index") is not None
        }
        if len(selected_modes) != 3 or len(unique_mode_indices) != 3:
            reasons.append(
                f"{reference.key}: plan must contain exactly three unique selected modes"
            )
        for mode in selected_modes:
            mode_index = int(mode["mode_index"])
            expected_mode_keys.add((reference.key, mode_index))
            for sign, name in ((-1.0, "negative"), (1.0, "positive")):
                expected_alpha = tuple(
                    sign * value for value in (0.125, 0.25, 0.5, 1, 2, 4)
                )
                reasons.extend(
                    _schedule_chain_errors(
                        directional_records,
                        reference=reference.key,
                        chain=f"mode_{mode_index}_{name}",
                        coordinate="alpha",
                        requested=expected_alpha,
                        extra_successes=zero_gates,
                        validated_case_ids=validated_case_ids,
                        allowed_blocker_ids=failed_zero_blockers,
                    )
                )
    observed_zero_keys = {
        (str(item.get("reference")), int(item.get("mode_index")))
        for item in zero_gates
        if item.get("mode_index") is not None
    }
    observed_linearity_keys = {
        (str(item.get("reference")), int(item.get("mode_index")))
        for item in payload.get("direction_linearity", ())
        if item.get("mode_index") is not None
    }
    if observed_zero_keys != expected_mode_keys:
        reasons.append("directional t=0 gate coverage differs from the selected-mode plan")
    if observed_linearity_keys != expected_mode_keys:
        reasons.append("directional linearity coverage differs from the selected-mode plan")
    linearity_by_key = {
        (str(item.get("reference")), int(item.get("mode_index"))): item
        for item in payload.get("direction_linearity", ())
        if item.get("mode_index") is not None
    }
    for gate in zero_gates:
        if gate.get("passed") is True:
            continue
        key = (str(gate.get("reference")), int(gate.get("mode_index")))
        linearity_record = linearity_by_key.get(key, {})
        nonzero = [
            item
            for item in directional_records
            if item.get("reference") == key[0]
            and item.get("mode_index") == key[1]
            and item.get("alpha") not in (None, 0, 0.0)
        ]
        if not (
            gate.get("scientific_qualification") == "nonlinear/unreliable"
            and gate.get("curvature_uncertainty_reported") is False
            and linearity_record.get("scientific_qualification")
            == "nonlinear/unreliable"
            and linearity_record.get("curvature_uncertainty_reported") is False
            and len(nonzero) == 12
            and all(
                item.get("status") == "blocked"
                and str(item.get("blocked_by_case_id")) == str(gate.get("case_id"))
                for item in nonzero
            )
        ):
            reasons.append(
                f"{key[0]}/mode_{key[1]}: failed t=0 gate is not a fully "
                "blocked nonlinear/unreliable qualification"
            )
    for item in payload.get("direction_linearity", ()):
        if item.get("passed") is not True and item.get(
            "scientific_qualification"
        ) != "nonlinear/unreliable":
            reasons.append(
                f"{item.get('reference')}/mode_{item.get('mode_index')}: failed "
                "linearity has no nonlinear/unreliable qualification"
            )
    if _records_sha256(payload.get("direction_linearity", ())) != _records_sha256(
        directional_artifact.get("linearity", ())
    ):
        reasons.append("combined directional linearity records are stale or truncated")

    expected_pairs = {
        (reference.key, family)
        for reference in REFERENCE_SLICES
        for family in FAMILY_ORDER
    }
    family_rows = list(payload.get("family_results", ()))
    records = {
        (str(item.get("reference")), str(item.get("family"))): item
        for item in family_rows
    }
    if len(records) != len(family_rows):
        reasons.append("family classifications contain duplicate reference/family keys")
    missing = sorted(expected_pairs - set(records))
    if missing:
        reasons.append(
            "family classifications are missing: "
            + ", ".join(f"{reference}/{family}" for reference, family in missing)
        )
    extra_pairs = sorted(set(records) - expected_pairs)
    if extra_pairs:
        reasons.append(
            "unexpected family classifications: "
            + ", ".join(f"{reference}/{family}" for reference, family in extra_pairs)
        )
    for record in family_rows:
        reasons.extend(_family_record_errors(record))
    matrix = [
        {
            "reference": reference,
            "family": family,
            "classification": records[(reference, family)].get("classification", []),
            "local_information_supported": _local_information_supported(
                records[(reference, family)]
            ),
            "nonlinear_evidence": records[(reference, family)].get(
                "nonlinear_evidence"
            ),
        }
        for reference, family in sorted(expected_pairs & set(records))
    ]
    expected_references = {item.key for item in REFERENCE_SLICES}

    def exact_reference_section(name: str) -> None:
        section = payload.get(name, ())
        references = [str(item.get("reference")) for item in section]
        if (
            len(references) != len(expected_references)
            or set(references) != expected_references
        ):
            reasons.append(f"{name} does not contain exactly one row per reference")

    for section_name in ("spectra", "external_current_spectra", "ip_accounting"):
        exact_reference_section(section_name)
    mode_share_references = {
        str(item.get("reference")) for item in payload.get("mode_family_shares", ())
    }
    if mode_share_references != expected_references:
        reasons.append("mode/family shares do not cover every reference")
    expected_subset_names = {
        "full",
        "core",
        "core_plus_flux_loop",
        "core_plus_bpol_probe",
        "core_plus_magnetics",
        "core_plus_diamagnetic_flux",
        "core_plus_magnetics_plus_diamagnetic",
        *(f"without_{family}" for family in FAMILY_ORDER),
    }
    observed_subsets: dict[str, set[str]] = {
        reference: set() for reference in expected_references
    }
    subset_duplicate = False
    for item in payload.get("subset_results", ()):
        reference = str(item.get("reference"))
        name = str(item.get("name"))
        if reference not in observed_subsets or name in observed_subsets[reference]:
            subset_duplicate = True
            continue
        observed_subsets[reference].add(name)
        if "maximum_principal_angle_degrees" not in item:
            reasons.append(f"{reference}/{name}: principal-angle result is missing")
    if subset_duplicate or any(
        names != expected_subset_names for names in observed_subsets.values()
    ):
        reasons.append("subset/rank/principal-angle matrix is incomplete or duplicated")
    native_rows = payload.get("native_rows", ())
    main_row_pairs = {
        (str(item.get("reference")), str(item.get("family")))
        for item in native_rows
        if item.get("block") == "main"
    }
    if main_row_pairs != expected_pairs:
        reasons.append("native main residual/weight rows do not cover every family/reference")
    external_pf_blocks = {
        str(item.get("reference"))
        for item in native_rows
        if item.get("block") == "external_current"
        and item.get("family") in {"pf_current", "pf_relation"}
    }
    if external_pf_blocks != expected_references:
        reasons.append("native external-current PF rows do not cover every reference")
    summary_pairs = {
        (str(item.get("reference")), str(item.get("family")))
        for item in payload.get("family_residual_summary", ())
        if item.get("block") == "main"
    }
    if summary_pairs != expected_pairs:
        reasons.append("family residual/weight summaries are incomplete")
    exact_reference_section("mfile_family_chi2_audits")
    for population_name, populations in (
        ("diamagnetic", diamagnetic_artifact.get("outcome_populations", {})),
        ("directional", directional_artifact.get("outcome_populations", {})),
    ):
        if set(populations) != expected_references or any(
            not isinstance(counts, Mapping)
            or sum(
                int(value)
                for key, value in counts.items()
                if key not in {"noise_gate_passed", "forward_complete", "reverse_complete"}
            )
            <= 0
            for counts in populations.values()
        ):
            reasons.append(f"{population_name} outcome populations are incomplete")
    # Plot before freezing acceptance so every delivered plot is content-bound
    # in the acceptance matrix.  Plot generation does not depend on acceptance.
    try:
        plot_paths = write_plots(payload, output)
    except Exception as exc:
        plot_paths = ()
        reasons.append(f"plot generation failed ({type(exc).__name__}: {exc})")
    required_plot_names = {
        "constraint_ablation_effect_heatmap.png",
        "singular_spectra.png",
        "mode_family_shares.png",
        "conditional_rank_gain.png",
        "diagnostic_vs_solver_objective.png",
        "ip_vcurrt_accounting.png",
        "main_and_external_pf_solves.png",
        "residual_and_processed_weight_distributions.png",
        "stacked_solver_objective.png",
        "pf_currents_and_relation_residuals.png",
        "diamagnetic_continuation.png",
        "directional_continuation.png",
    }
    missing_plots = required_plot_names - {path.name for path in plot_paths}
    if missing_plots:
        reasons.append("required plots are missing: " + ", ".join(sorted(missing_plots)))
    acceptance = {
        "schema_version": SCHEMA_VERSION,
        "issue": ISSUE,
        "accepted": not reasons,
        "status": "complete" if not reasons else "incomplete",
        "reasons": reasons,
        "stage1_manifest_sha256": sha256_file(output / STUDY_MANIFEST),
        "stage2_plan_sha256": sha256_file(plan_path),
        "restart_proof_sha256": expected_proof_sha256 or None,
        "classification_matrix": matrix,
        "outlier_assessment": outlier,
        "zero_target_gates": zero_gates,
        "direction_linearity": payload.get("direction_linearity", ()),
        "outcome_populations": payload.get("outcome_populations", {}),
        "plot_artifacts": _artifact_records(plot_paths),
    }
    acceptance_path = write_json_atomic(output / "acceptance_matrix.json", acceptance)
    payload["acceptance"] = acceptance
    payload["acceptance_matrix_sha256"] = sha256_file(acceptance_path)
    # Avoid regenerating the already-hashed plots.  The JSON and Markdown are
    # the only report artifacts that depend on the acceptance object itself.
    write_json_atomic(output / RESULTS_JSON, payload)
    (output / RESULTS_MARKDOWN).write_text(
        render_markdown(payload), encoding="utf-8"
    )
    return acceptance


def _cell(value: Any) -> str:
    if value is None:
        return "–"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, (int, float)):
        return f"{value:.6g}"
    return str(value)


def render_markdown(payload: Mapping[str, Any]) -> str:
    """Render a compact review report from normalized study results."""
    gate = payload.get("stage1_gate", {})
    lines = [
        "# EFIT native constraint-identifiability study (#664)",
        "",
        f"Stage-1 gate: **{'PASS' if gate.get('passed') else 'FAIL'}**.",
    ]
    acceptance = payload.get("acceptance")
    if acceptance:
        lines.extend(
            (
                "",
                f"Final study status: **{str(acceptance.get('status', 'incomplete')).upper()}**.",
            )
        )
        lines.extend(f"- {reason}" for reason in acceptance.get("reasons", ()))
    reasons = gate.get("reasons", ())
    if reasons:
        lines.extend(("", "Gate findings:", ""))
        lines.extend(f"- {reason}" for reason in reasons)
    mfile_audits = payload.get("mfile_family_chi2_audits", ())
    if mfile_audits:
        lines.extend(
            (
                "",
                "## Native/m-file diagnostic χ² validation",
                "",
                "| shot/time [ms] | family | native sum | m-file sum | absolute error | relative error | gate |",
                "|---|---|---:|---:|---:|---:|---|",
            )
        )
        for audit in mfile_audits:
            for family, record in audit.get("families", {}).items():
                lines.append(
                    "| {reference} | `{family}` | {native} | {persisted} | {absolute} | {relative} | {passed} |".format(
                        reference=audit.get("reference", "–"),
                        family=family,
                        native=_cell(record.get("native_diagnostic_chi2")),
                        persisted=_cell(record.get("mfile_diagnostic_chi2")),
                        absolute=_cell(record.get("absolute_error")),
                        relative=_cell(record.get("relative_error")),
                        passed=_cell(record.get("passed")),
                    )
                )
    lines.extend(
        (
            "",
            "## Constraint-family information",
            "",
            "| shot/time [ms] | family | classification | rank gain (τ/10, τ, 10τ) | weak action | weak lift | trace reduction | objective share | non-Ip share |",
            "|---|---|---|---:|---:|---:|---:|---:|---:|",
        )
    )
    for record in payload.get("family_results", ()):
        labels = record.get("classification") or classify_family_information(record)
        lines.append(
            "| {reference} | `{family}` | {labels} | {rank} | {action} | {lift} | {trace} | {share} | {non_ip_share} |".format(
                reference=record.get("reference", "–"),
                family=record.get("family", "–"),
                labels=", ".join(labels),
                rank="/".join(str(value) for value in record.get("rank_gain_by_cutoff", ())),
                action=_cell(record.get("weak_subspace_action_fraction")),
                lift=_cell(record.get("weak_singular_value_lift")),
                trace=_cell(record.get("curvature_inverse_trace_reduction")),
                share=_cell(record.get("baseline_solver_objective_share")),
                non_ip_share=_cell(
                    record.get("solver_objective_share_excluding_ip")
                ),
            )
        )
    residual_summary = payload.get("family_residual_summary", ())
    if residual_summary:
        lines.extend(
            (
                "",
                "## Native residual and weight audit",
                "",
                "| shot/time [ms] | solve | family | active/rows | bias | RMS | median | max | σ median | FWT median | processed weight median | solver objective |",
                "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            )
        )
        for record in residual_summary:
            lines.append(
                "| {reference} | `{block}` | `{family}` | {active}/{rows} | {bias} | {rms} | {median} | {maximum} | {uncertainty} | {submitted} | {processed} | {objective} |".format(
                    reference=record.get("reference", "–"),
                    block=record.get("block", "–"),
                    family=record.get("family", "–"),
                    active=record.get("active_channel_count", 0),
                    rows=record.get("row_count", 0),
                    bias=_cell(record.get("residual_bias")),
                    rms=_cell(record.get("residual_rms")),
                    median=_cell(record.get("median_absolute_residual")),
                    maximum=_cell(record.get("maximum_absolute_residual")),
                    uncertainty=_cell(
                        (record.get("uncertainty") or {}).get("median")
                    ),
                    submitted=_cell(
                        (record.get("submitted_fwt") or {}).get("median")
                    ),
                    processed=_cell(
                        (record.get("processed_weight") or {}).get("median")
                    ),
                    objective=_cell(record.get("solver_objective")),
                )
            )
    lines.extend(
        (
            "",
            "## Nonlinear continuation",
            "",
            "| shot/time [ms] | chain | setting | outcome | βp | Δβp | li | Δli | D | onset | branch | rank switch |",
            "|---|---|---:|---|---:|---:|---:|---:|---:|---|---|---|",
        )
    )
    for record in payload.get("continuation_results", ()):
        lines.append(
            "| {reference} | `{chain}` | {setting} | {outcome} | {beta} | {delta_beta} | {li} | {delta_li} | {d} | {onset} | {branch} | {rank} |".format(
                reference=record.get("reference", "–"),
                chain=record.get("chain", "–"),
                setting=_cell(record.get("requested_value")),
                outcome=record.get("outcome", "–"),
                beta=_cell(record.get("beta_p")),
                delta_beta=_cell(record.get("beta_p_relative")),
                li=_cell(record.get("li")),
                delta_li=_cell(record.get("li_relative")),
                d=_cell(record.get("displacement")),
                onset=_cell(record.get("response_onset")),
                branch=_cell(record.get("material_branch")),
                rank=_cell(record.get("rank_switching")),
            )
        )
    outlier = payload.get("outlier_assessment")
    if outlier:
        lines.extend(
            (
                "",
                "## 41672/347-ms diamagnetic response",
                "",
                f"Classification: **{outlier.get('classification', 'incomplete')}**.",
                "",
                "- Minimum weak/borderline-subspace projection at ×1000–×10000: "
                f"{_cell(outlier.get('minimum_target_projection_fraction'))} "
                "(required ≥0.8).",
                "- Retained-mask switch in the response interval: "
                f"{_cell(outlier.get('rank_switching'))}.",
                "- Early same-branch beta/li prediction gate: "
                f"{_cell(outlier.get('prediction_passed'))} "
                "(required sign agreement and ≤25% magnitude error).",
            )
        )
    lines.extend(
        (
            "",
            "`chipasma` is reported only as an Ip/VCURRT accounting diagnostic; it is not reconstructed from the Ip solve row.",
            "",
        )
    )
    return "\n".join(lines)


def write_report(payload: Mapping[str, Any], output: Path) -> tuple[Path, Path, tuple[Path, ...]]:
    """Write normalized JSON, Markdown, and deterministic summary plots."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    json_path = write_json_atomic(output / RESULTS_JSON, payload)
    markdown_path = output / RESULTS_MARKDOWN
    markdown_path.write_text(render_markdown(payload), encoding="utf-8")
    plots = write_plots(payload, output)
    return json_path, markdown_path, plots


def write_plots(payload: Mapping[str, Any], output: Path) -> tuple[Path, ...]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output = Path(output)
    created: list[Path] = []
    ablation_records = payload.get("constraint_ablation_effects", ())
    if ablation_records:
        metric_labels = (
            ("lcfs_rms_mm", "LCFS / 5 mm"),
            ("area_relative", "area / 2%"),
            ("volume_relative", "volume / 2%"),
            ("beta_p_relative", "beta_p / 2%"),
            ("li_relative", "li / 2%"),
            ("profile_relative_rms", "profile RMS / 5%"),
            ("acceptance_change_percentage_points", "acceptance / 10 pp"),
        )
        values = np.full((len(ablation_records), len(metric_labels)), np.nan)
        row_labels = []
        for row, record in enumerate(ablation_records):
            row_labels.append(f"{record['shot']} {record['variant']}")
            normalized = record.get("threshold_normalized", {})
            for column, (name, _) in enumerate(metric_labels):
                value = normalized.get(name)
                if value is not None and math.isfinite(float(value)):
                    values[row, column] = float(value)
        display = np.ma.masked_invalid(
            np.log10(np.maximum(values, np.finfo(float).tiny))
        )
        figure, axis = plt.subplots(
            figsize=(11, max(5.5, 0.34 * len(ablation_records)))
        )
        image = axis.imshow(
            display,
            aspect="auto",
            cmap="coolwarm",
            vmin=-3.0,
            vmax=3.0,
        )
        axis.set_xticks(
            range(len(metric_labels)),
            [label for _, label in metric_labels],
            rotation=30,
            ha="right",
        )
        axis.set_yticks(range(len(row_labels)), row_labels, fontsize=7)
        for row in range(values.shape[0]):
            for column in range(values.shape[1]):
                if np.isfinite(values[row, column]):
                    axis.text(
                        column,
                        row,
                        f"{values[row, column]:.2g}",
                        ha="center",
                        va="center",
                        fontsize=5.5,
                        color=(
                            "white"
                            if abs(display[row, column]) >= 1.4
                            else "black"
                        ),
                    )
        axis.set_title("#663 nonlinear ablation effects (1 = material threshold)")
        figure.colorbar(
            image,
            ax=axis,
            label="log10(effect / material-response threshold)",
        )
        path = output / "constraint_ablation_effect_heatmap.png"
        figure.tight_layout()
        figure.savefig(path, dpi=160)
        plt.close(figure)
        created.append(path)
    spectra = payload.get("spectra", ())
    if spectra:
        figure, axis = plt.subplots(figsize=(8, 5))
        for record in spectra:
            values = np.asarray(record.get("singular_values", ()), dtype=float)
            if values.size:
                axis.semilogy(np.arange(values.size), values, "o-", label=record["reference"])
                tau = record.get("tau")
                if tau:
                    axis.axhline(tau, linewidth=0.7, alpha=0.35)
        axis.set_xlabel("mode index")
        axis.set_ylabel("singular value of ASZ")
        axis.legend(fontsize=8)
        path = output / "singular_spectra.png"
        figure.tight_layout()
        figure.savefig(path, dpi=160)
        plt.close(figure)
        created.append(path)

    mode_records = payload.get("mode_family_shares", ())
    if mode_records:
        references = tuple(dict.fromkeys(item["reference"] for item in mode_records))
        figure, axes = plt.subplots(
            len(references),
            1,
            figsize=(10, max(3.0, 2.7 * len(references))),
            squeeze=False,
        )
        for axis, reference in zip(axes[:, 0], references):
            records = [item for item in mode_records if item["reference"] == reference]
            families = tuple(dict.fromkeys(item["family"] for item in records))
            modes = sorted({int(item["mode_index"]) for item in records})
            family_index = {name: index for index, name in enumerate(families)}
            mode_index = {value: index for index, value in enumerate(modes)}
            values = np.zeros((len(families), len(modes)))
            for item in records:
                values[family_index[item["family"]], mode_index[int(item["mode_index"])]] = float(
                    item["share"]
                )
            image = axis.imshow(values, vmin=0.0, vmax=1.0, aspect="auto", cmap="viridis")
            axis.set_yticks(range(len(families)), families)
            axis.set_xticks(range(len(modes)), modes)
            axis.set_ylabel(reference)
        axes[-1, 0].set_xlabel("mode index")
        figure.colorbar(image, ax=axes[:, 0].tolist(), label="family share")
        path = output / "mode_family_shares.png"
        figure.savefig(path, dpi=160, bbox_inches="tight")
        plt.close(figure)
        created.append(path)

    family_records = payload.get("family_results", ())
    if family_records:
        references = tuple(dict.fromkeys(item["reference"] for item in family_records))
        families = tuple(dict.fromkeys(item["family"] for item in family_records))
        rank_heat = np.zeros((len(families), len(references)))
        for item in family_records:
            gains = item.get("rank_gain_by_cutoff", ())
            rank_heat[families.index(item["family"]), references.index(item["reference"])] = (
                float(gains[len(gains) // 2]) if gains else np.nan
            )
        figure, axis = plt.subplots(figsize=(max(7, len(references) * 1.5), max(4, len(families) * 0.55)))
        image = axis.imshow(rank_heat, aspect="auto", cmap="magma")
        axis.set_xticks(range(len(references)), references, rotation=30, ha="right")
        axis.set_yticks(range(len(families)), families)
        figure.colorbar(image, ax=axis, label="nominal-cutoff rank gain")
        path = output / "conditional_rank_gain.png"
        figure.tight_layout()
        figure.savefig(path, dpi=160)
        plt.close(figure)
        created.append(path)

        figure, axis = plt.subplots(figsize=(max(8, len(family_records) * 0.22), 5))
        x = np.arange(len(family_records))
        diagnostic = np.asarray(
            [item.get("diagnostic_chi2", np.nan) for item in family_records], dtype=float
        )
        objective = np.asarray(
            [item.get("solver_objective", np.nan) for item in family_records], dtype=float
        )
        diagnostic = np.where(np.isfinite(diagnostic), np.maximum(diagnostic, 1e-30), np.nan)
        objective = np.where(np.isfinite(objective), np.maximum(objective, 1e-30), np.nan)
        axis.plot(x, diagnostic, "o", label="diagnostic χ²")
        axis.plot(x, objective, "x", label="solver objective")
        axis.set_yscale("log")
        axis.set_xticks(
            x,
            [f"{item['reference']}\n{item['family']}" for item in family_records],
            rotation=90,
            fontsize=6,
        )
        axis.legend()
        path = output / "diagnostic_vs_solver_objective.png"
        figure.tight_layout()
        figure.savefig(path, dpi=160)
        plt.close(figure)
        created.append(path)

    ip_records = payload.get("ip_accounting", ())
    if ip_records:
        figure, axis = plt.subplots(figsize=(8, 4.5))
        x = np.arange(len(ip_records))
        axis.semilogy(
            x,
            [max(float(item.get("reported_chipasma") or 0.0), 1e-30) for item in ip_records],
            "o-",
            label="reported chipasma",
        )
        axis.semilogy(
            x,
            [max(float(item.get("predicted_vcurrt_chi2") or 0.0), 1e-30) for item in ip_records],
            "x--",
            label="VCURRT prediction",
        )
        axis.set_xticks(x, [item["reference"] for item in ip_records], rotation=30, ha="right")
        axis.set_ylabel("Ip accounting χ²")
        axis.legend()
        path = output / "ip_vcurrt_accounting.png"
        figure.tight_layout()
        figure.savefig(path, dpi=160)
        plt.close(figure)
        created.append(path)

    external_spectra = payload.get("external_current_spectra", ())
    if spectra and external_spectra:
        external_by_reference = {item["reference"]: item for item in external_spectra}
        figure, axis = plt.subplots(figsize=(8, 5))
        colors = plt.cm.tab10.colors
        for index, main in enumerate(spectra):
            reference = main["reference"]
            external = external_by_reference.get(reference)
            if external is None:
                continue
            color = colors[index % len(colors)]
            axis.semilogy(main["singular_values"], "o-", color=color, label=f"{reference} main")
            axis.semilogy(
                external["singular_values"],
                "x--",
                color=color,
                label=f"{reference} external PF",
            )
        axis.set_xlabel("mode index")
        axis.set_ylabel("singular value")
        axis.legend(fontsize=7, ncol=2)
        path = output / "main_and_external_pf_solves.png"
        figure.tight_layout()
        figure.savefig(path, dpi=160)
        plt.close(figure)
        created.append(path)

    row_records = payload.get("native_rows", ())
    if row_records:
        labels = tuple(
            dict.fromkeys(f"{item['block']}:{item['family']}" for item in row_records)
        )
        residual_groups: list[np.ndarray] = []
        weight_groups: list[np.ndarray] = []
        usable_labels: list[str] = []
        for label in labels:
            selected = [
                item
                for item in row_records
                if f"{item['block']}:{item['family']}" == label
            ]
            residual = np.asarray(
                [item.get("physical_residual", np.nan) for item in selected], dtype=float
            )
            weight = np.asarray(
                [item.get("processed_weight", np.nan) for item in selected], dtype=float
            )
            residual = np.abs(residual[np.isfinite(residual)])
            weight = np.abs(weight[np.isfinite(weight) & (weight != 0.0)])
            if not residual.size and not weight.size:
                continue
            usable_labels.append(label)
            residual_groups.append(np.maximum(residual, 1e-30))
            weight_groups.append(np.maximum(weight, 1e-30))
        if usable_labels:
            figure, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
            for axis, groups, label in (
                (axes[0], residual_groups, "|physical residual|"),
                (axes[1], weight_groups, "|processed weight|"),
            ):
                positions = [index + 1 for index, values in enumerate(groups) if values.size]
                values = [values for values in groups if values.size]
                if values:
                    axis.boxplot(values, positions=positions, showfliers=True)
                    axis.set_yscale("log")
                axis.set_ylabel(label)
            axes[1].set_xticks(range(1, len(usable_labels) + 1), usable_labels, rotation=45, ha="right")
            path = output / "residual_and_processed_weight_distributions.png"
            figure.tight_layout()
            figure.savefig(path, dpi=160)
            plt.close(figure)
            created.append(path)

        main_summaries = [
            item
            for item in payload.get("family_residual_summary", ())
            if item.get("block") == "main"
        ]
        if main_summaries:
            references = tuple(dict.fromkeys(item["reference"] for item in main_summaries))
            families = tuple(dict.fromkeys(item["family"] for item in main_summaries))
            x = np.arange(len(references))
            bottom = np.zeros(len(references))
            figure, axis = plt.subplots(figsize=(9, 5))
            for family in families:
                values = np.asarray(
                    [
                        next(
                            (
                                float(item.get("solver_objective") or 0.0)
                                for item in main_summaries
                                if item["reference"] == reference
                                and item["family"] == family
                            ),
                            0.0,
                        )
                        for reference in references
                    ]
                )
                axis.bar(x, values, bottom=bottom, label=family)
                bottom += values
            axis.set_yscale("log")
            axis.set_xticks(x, references, rotation=30, ha="right")
            axis.set_ylabel("solver-objective contribution")
            axis.legend(fontsize=7, ncol=2)
            path = output / "stacked_solver_objective.png"
            figure.tight_layout()
            figure.savefig(path, dpi=160)
            plt.close(figure)
            created.append(path)

        pf_rows = [
            item
            for item in row_records
            if item.get("family") in {"pf_current", "pf_relation"}
        ]
        current_rows = [item for item in pf_rows if item.get("family") == "pf_current"]
        relation_rows = [item for item in pf_rows if item.get("family") == "pf_relation"]
        if current_rows or relation_rows:
            figure, axes = plt.subplots(2, 1, figsize=(10, 7))
            current_groups = dict.fromkeys(
                (item["reference"], item["block"]) for item in current_rows
            )
            for reference, block in current_groups:
                selected = [
                    item
                    for item in current_rows
                    if item["reference"] == reference and item["block"] == block
                ]
                channel = [item["channel"] for item in selected]
                axes[0].plot(
                    channel,
                    [item.get("measurement") for item in selected],
                    "o-",
                    label=f"{reference} {block} measured",
                )
                axes[0].plot(
                    channel,
                    [item.get("reconstruction") for item in selected],
                    "x--",
                    label=f"{reference} {block} reconstructed",
                )
            relation_groups = dict.fromkeys(
                (item["reference"], item["block"]) for item in relation_rows
            )
            for reference, block in relation_groups:
                selected = [
                    item
                    for item in relation_rows
                    if item["reference"] == reference and item["block"] == block
                ]
                axes[1].plot(
                    [item["channel"] for item in selected],
                    [item.get("physical_residual") for item in selected],
                    "o-",
                    label=f"{reference} {block}",
                )
            axes[0].set_ylabel("PF current")
            axes[1].set_ylabel(r"soft relation residual $C^T I-X$")
            axes[1].set_xlabel("native channel")
            axes[1].set_yscale("symlog", linthresh=1e-12)
            axes[0].legend(fontsize=6, ncol=2)
            axes[1].legend(fontsize=7)
            path = output / "pf_currents_and_relation_residuals.png"
            figure.tight_layout()
            figure.savefig(path, dpi=160)
            plt.close(figure)
            created.append(path)
    continuation = payload.get("continuation_results", ())
    diamagnetic = [
        item for item in continuation if item.get("control") == "diamagnetic_flux_scale"
    ]
    if diamagnetic:
        figure, axes = plt.subplots(3, 1, figsize=(8, 9), sharex=True)
        grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
        for item in diamagnetic:
            grouped.setdefault((item["reference"], item["chain"]), []).append(item)
        for (reference, chain), records in sorted(grouped.items()):
            records = sorted(records, key=lambda item: item.get("order", 0))
            x = [item["requested_value"] for item in records]
            axes[0].semilogx(
                x,
                [item.get("beta_p", np.nan) for item in records],
                "o-",
                label=f"{reference} {chain} βp",
            )
            axes[0].semilogx(
                x,
                [item.get("li", np.nan) for item in records],
                "x--",
                label=f"{reference} {chain} li",
            )
            axes[1].semilogx(x, [item.get("beta_p_relative", np.nan) for item in records], "o-", label=f"{reference} {chain}")
            axes[1].semilogx(x, [item.get("li_relative", np.nan) for item in records], "x--")
            axes[2].loglog(x, [max(float(item.get("displacement") or 0.0), 1e-12) for item in records], "o-", label=f"{reference} {chain}")
        axes[0].set_ylabel("raw βp (solid), li (dashed)")
        axes[1].set_ylabel("relative change (βp solid, li dashed)")
        axes[2].set_xlabel("diamagnetic objective scale")
        axes[2].set_ylabel("nonlinear displacement D")
        axes[2].axhline(0.1, color="grey", linewidth=0.7)
        axes[2].axhline(1.0, color="black", linewidth=0.7)
        axes[0].legend(fontsize=6, ncol=2)
        path = output / "diamagnetic_continuation.png"
        figure.tight_layout()
        figure.savefig(path, dpi=160)
        plt.close(figure)
        created.append(path)
    directional = [
        item
        for item in continuation
        if item.get("control") == "direction"
        and item.get("status") == "succeeded"
        and item.get("alpha") not in (None, 0, 0.0)
    ]
    if directional:
        figure, axis = plt.subplots(figsize=(9, 5))
        groups: dict[tuple[str, int, str], list[Mapping[str, Any]]] = {}
        for item in directional:
            groups.setdefault(
                (str(item["reference"]), int(item["mode_index"]), str(item["chain"])),
                [],
            ).append(item)
        for (reference, mode_index, chain), records in sorted(groups.items()):
            records = sorted(records, key=lambda item: abs(float(item["alpha"])))
            axis.loglog(
                [abs(float(item["alpha"])) for item in records],
                [max(float(item.get("displacement") or 0.0), 1e-12) for item in records],
                "o-",
                label=f"{reference} m{mode_index} {chain.rsplit('_', 1)[-1]}",
            )
        axis.axhline(0.1, color="grey", linewidth=0.7)
        axis.axhline(1.0, color="black", linewidth=0.7)
        axis.set_xlabel(r"direction amplitude $|\alpha|$")
        axis.set_ylabel("nonlinear displacement D")
        axis.legend(fontsize=6, ncol=2)
        path = output / "directional_continuation.png"
        figure.tight_layout()
        figure.savefig(path, dpi=160)
        plt.close(figure)
        created.append(path)
    else:
        gates = payload.get("zero_target_gates", ())
        if gates:
            figure, axis = plt.subplots(figsize=(10, 5))
            labels = [
                f"{item.get('reference')} m{item.get('mode_index')}"
                for item in gates
            ]
            values = [
                max(float(item.get("displacement") or 0.0), 1.0e-12)
                for item in gates
            ]
            colors = ["tab:green" if item.get("passed") else "tab:red" for item in gates]
            axis.bar(np.arange(len(gates)), values, color=colors)
            axis.set_yscale("log")
            axis.set_xticks(np.arange(len(gates)), labels, rotation=70, ha="right")
            axis.set_ylabel("zero-target nonlinear displacement D")
            axis.set_title("Directional continuation stopped by mandatory t=0 gates")
            axis.axhline(1.0, color="black", linewidth=0.7)
            path = output / "directional_continuation.png"
            figure.tight_layout()
            figure.savefig(path, dpi=160)
            plt.close(figure)
            created.append(path)
    return tuple(created)


def _read_json_argument(value: str) -> Mapping[str, Any]:
    path = Path(value)
    try:
        if path.is_file():
            return json.loads(path.read_text(encoding="utf-8"))
    except OSError:
        pass
    parsed = json.loads(value)
    if not isinstance(parsed, Mapping):
        raise ValueError("JSON argument must decode to an object")
    return parsed


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    plan = subparsers.add_parser("plan", help="write the content-addressed Stage-1 plan")
    plan.add_argument("--output", type=Path, required=True)
    plan.add_argument("--kfile-root", type=Path, required=True)
    plan.add_argument("--executable", type=Path, required=True)
    plan.add_argument("--scientific-sha256", required=True)
    plan.add_argument("--table-identity", required=True, help="JSON text or path")
    plan.add_argument("--build-provenance", required=True, help="JSON text or path")

    execute = subparsers.add_parser(
        "execute-stage1", help="run and analyze all five gated native baselines"
    )
    execute.add_argument("--output", type=Path, required=True)
    execute.add_argument("--kfile-root", type=Path, required=True)
    execute.add_argument("--executable", type=Path, required=True)
    execute.add_argument("--scientific-sha256", required=True)
    execute.add_argument("--table-identity", required=True, help="JSON text or path")
    execute.add_argument("--build-provenance", required=True, help="JSON text or path")

    analyze = subparsers.add_parser(
        "analyze", help="analyze five already-produced native sidecars"
    )
    analyze.add_argument("--sidecars", type=Path, nargs="+", required=True)
    analyze.add_argument("--output", type=Path, required=True)

    stage2 = subparsers.add_parser(
        "plan-stage2", help="build continuation chains after a passing Stage-1 gate"
    )
    stage2.add_argument("--output", type=Path, required=True)

    restart = subparsers.add_parser(
        "execute-restart-proof",
        help="run cold repeats and the unchanged-control native restart gate",
    )
    restart.add_argument("--output", type=Path, required=True)

    diamagnetic = subparsers.add_parser(
        "execute-diamagnetic",
        help="run the gated bidirectional 0.25-dex diamagnetic continuation",
    )
    diamagnetic.add_argument("--output", type=Path, required=True)

    directional = subparsers.add_parser(
        "execute-directional",
        help="run the gated native weak-mode directional continuation",
    )
    directional.add_argument("--output", type=Path, required=True)

    reproduce = subparsers.add_parser(
        "execute-663-reproduction",
        help="reproduce provenance-incompatible #663 ablation/strength endpoints",
    )
    reproduce.add_argument("--output", type=Path, required=True)

    merge_prior = subparsers.add_parser(
        "merge-663-evidence",
        help="merge #663 evidence only after exact provenance validation",
    )
    merge_prior.add_argument("--output", type=Path, required=True)
    merge_prior.add_argument("--input", type=Path, required=True)

    finalize = subparsers.add_parser(
        "finalize", help="write the gated final classification/acceptance matrix"
    )
    finalize.add_argument("--output", type=Path, required=True)

    report = subparsers.add_parser("report", help="render an existing normalized result JSON")
    report.add_argument("--input", type=Path, required=True)
    report.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "plan":
        provenance = build_provenance_record(
            args.executable, _read_json_argument(args.build_provenance)
        )
        validate_build_provenance(provenance)
        build_plan_manifest(
            output=args.output,
            kfile_root=args.kfile_root,
            executable=args.executable,
            scientific_sha256=args.scientific_sha256,
            table_identity=_read_json_argument(args.table_identity),
            build_provenance=provenance,
        )
        return 0
    if args.command == "execute-stage1":
        execute_stage1(
            output=args.output,
            kfile_root=args.kfile_root,
            executable=args.executable,
            scientific_sha256=args.scientific_sha256,
            table_identity=_read_json_argument(args.table_identity),
            build_provenance=_read_json_argument(args.build_provenance),
        )
        return 0
    if args.command == "analyze":
        analyze_sidecars(args.sidecars, args.output)
        return 0
    if args.command == "plan-stage2":
        build_stage2_plan(args.output)
        return 0
    if args.command == "execute-restart-proof":
        proof = execute_restart_proof(args.output)
        return 0 if proof["passed"] else 3
    if args.command == "execute-diamagnetic":
        execute_diamagnetic_continuation(args.output)
        return 0
    if args.command == "execute-directional":
        execute_directional_continuation(args.output)
        return 0
    if args.command == "execute-663-reproduction":
        execute_issue_663_reproduction(args.output)
        return 0
    if args.command == "merge-663-evidence":
        merge_issue_663_evidence(args.output, args.input)
        return 0
    if args.command == "finalize":
        acceptance = finalize_study(args.output)
        return 0 if acceptance["accepted"] else 4
    payload = json.loads(args.input.read_text(encoding="utf-8"))
    write_report(payload, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
