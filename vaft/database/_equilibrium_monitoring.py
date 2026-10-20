"""Read-only equilibrium monitoring across published and server-side evidence.

Adapters own storage access; the collector owns table semantics. A future HSDS
attempt occurrence/source can implement EvidenceBackend without changing those
semantics. Missing products remain observations, never successful attempts.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Protocol

import numpy as np
import pandas as pd

from vaft.ods_access import path_count, path_value

SCHEMA_VERSION = 1
IDENTITY = ("shot", "source", "family", "backend", "product_id")
TABLE_COLUMNS = {
    "shots": (*IDENTITY, "machine_era", "availability", "stage_status", "runtime_status",
              "replication_state", "attempts_recorded", "equilibria_present", "usable_recorded",
              "attempt_coverage", "errors_json", "io_metrics_json"),
    "attempts": (*IDENTITY, "attempt_id", "time_s", "eq_index", "has_equilibrium",
                 "attempt_evidence", "overall_status", "runtime_ok", "output_ok", "numerical_ok", "physical_ok", "failure_codes_json", "metrics_json",
                 "configuration_json", "kfile_sha256", "provenance_json"),
    "checks": (*IDENTITY, "attempt_id", "time_s", "scope", "check", "status", "unit",
               "provider", "result_json"),
    "channels": (*IDENTITY, "attempt_id", "time_s", "scope", "family_name", "channel",
                 "channel_index", "status", "details_json"),
    "metrics": (*IDENTITY, "attempt_id", "time_s", "scope", "check", "metric", "value"),
    "inputs": (*IDENTITY, "attempt_id", "time_s", "kind", "snapshot_json", "sha256"),
}


def _plain(value):
    if isinstance(value, Mapping):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, np.ndarray):
        return _plain(value.tolist())
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, np.generic):
        return _plain(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def _json(value):
    return json.dumps(_plain(value), sort_keys=True, allow_nan=False)


def _digest(value):
    return hashlib.sha256(_json(value).encode()).hexdigest()


def _collection(ods):
    raw = path_value(ods, "equilibrium.code.parameters", None)
    if isinstance(raw, str):
        try:
            payload = json.loads(raw)
            return payload.get("efit_collection") or payload.get("kinetic_efit", {})
        except (ValueError, AttributeError):
            return {}
    return {}


def _snapshot(node):
    """Detach only a selected subtree while its lazy handle is still open."""
    from omas import ODS
    if isinstance(node, ODS):
        # keys() sees the lazy structure; flat() only sees already cached leaves.
        return {str(key): _snapshot(node[key]) for key in node.keys()}
    return _plain(node)


def _copy_branch(source, target, branch):
    """Copy a selected subtree to an ordinary ODS, safe for deep-copy providers."""
    from omas import ODS
    value = path_value(source, branch, None)
    if value is None:
        return
    def copy(node, prefix):
        if isinstance(node, ODS):
            for key in node.keys():
                copy(node[key], f"{prefix}.{key}")
        else:
            target[prefix] = node
    copy(value, branch)


@dataclass
class ShotEvidence:
    """Backend-neutral observation of one shot/product, including absent output."""
    shot: int
    backend: str
    source: str
    family: str = "magnetic"
    ods: Any = None
    constraints: Any = None
    collection: dict = field(default_factory=dict)
    manifest: dict = field(default_factory=dict)
    replication: dict = field(default_factory=dict)
    diagnostics_manifest: dict = field(default_factory=dict)
    kfiles: list[dict] = field(default_factory=list)
    availability: str = "present"
    errors: list[str] = field(default_factory=list)
    io_metrics: dict = field(default_factory=dict)
    # A new attempt-only HSDS adapter may supply a stable run/product identifier.
    product_id: str | None = None


class EvidenceBackend(Protocol):
    def read(self, shot: int) -> ShotEvidence: ...


class HSDSEvidenceBackend:
    """Lazy HSDS reads, bounded to one shot and requested IDS subtrees.

    Full verification needs 2-D equilibrium fields, and signal QA needs waveforms.
    Those selected subtrees are materialized before pure analysis; unrelated IDS
    (camera data, for example) are never requested. Only occurrence 0 is supported
    by the current database.open API.
    """
    def __init__(self, source="main", family="magnetic", *, magnetics=True, vacuum=False):
        self.source, self.family = source, family
        self.magnetics, self.vacuum = magnetics, vacuum

    def read(self, shot):
        from omas import ODS
        from vaft import database
        from ._summary import _is_connection_error
        evidence = ShotEvidence(shot, "hsds", self.source, self.family,
                                ods=ODS(consistency_check=False))
        branches = ["equilibrium", "tf", "magnetics.time", "magnetics.diamagnetic_flux",
                    "thomson_scattering", "core_profiles"]
        if self.magnetics:
            branches.append("magnetics")
        if self.vacuum:
            branches.extend(("magnetics", "pf_active", "pf_passive", "em_coupling",
                             "spectrometer_uv", "summary"))
        branches = list(dict.fromkeys(branches))
        roots = list(dict.fromkeys(path.split('.')[0] for path in branches))
        if self.vacuum:
            roots.remove("em_coupling")
            roots.insert(0, "em_coupling")
        # Open IDS separately so an absent optional IDS cannot erase equilibrium.
        for root in roots:
            if self.vacuum and root in ("pf_active", "pf_passive", "spectrometer_uv", "summary") and path_value(evidence.ods, "em_coupling", None) is None:
                continue  # No circuit model: a vacuum solve cannot use these bulk reads.
            try:
                with database.open(shot, source=self.source, paths=[root]) as lazy:
                    for branch in branches:
                        if branch.split('.')[0] == root:
                            if hasattr(lazy, "prefetch"):
                                lazy.prefetch(branch)
                            _copy_branch(lazy, evidence.ods, branch)
                    if hasattr(lazy, "store"):
                        for key, value in lazy.store.metrics.items():
                            evidence.io_metrics[key] = evidence.io_metrics.get(key, 0) + value
            except Exception as exc:
                if _is_connection_error(exc):
                    raise ConnectionError(f"HSDS source {self.source!r} is unreachable at shot {shot}") from exc
                if isinstance(exc, (FileNotFoundError, KeyError)) or (isinstance(exc, OSError) and exc.args and exc.args[0] in (404, 410)):
                    evidence.errors.append(f"{root}: unavailable: {exc}")
                else:
                    evidence.errors.append(f"{root}: error: {type(exc).__name__}: {exc}")
        evidence.collection = _collection(evidence.ods)
        if not path_count(evidence.ods, "equilibrium.time_slice"):
            evidence.availability = "no_equilibrium"
        return evidence


class FileDBEvidenceBackend:
    """Read a server FileDB locally or through a read-only mounted filesystem.

    This adapter never starts SSH or solvers. Run the command on vestserver, or
    pass a mounted FileDB root. Source is explicit, since a campaign FileDB need
    not be the product currently published to that HSDS namespace.
    """
    def __init__(self, root, source="main", family="magnetic", *, magnetics=True, vacuum=False):
        from .filedb import FileDB
        self.filedb = root if isinstance(root, FileDB) else FileDB(root)
        if not self.filedb.root.is_dir():
            raise FileNotFoundError(f"FileDB root does not exist: {self.filedb.root}")
        self.source, self.family = source, family
        self.magnetics, self.vacuum = magnetics, vacuum

    def read(self, shot):
        from omas import ODS
        from vaft.omas import load
        from vaft.code.efit.applicability import kfile_manifest_paths
        from vaft.data.keqdsk import read_keqdsk
        db, family = self.filedb, self.family
        stage = {"magnetic": "efit", "kinetic": "kinetic_efit", "electron": "electron_efit"}[family]
        e = ShotEvidence(shot, "filedb", self.source, family, ods=ODS(consistency_check=False))
        def read_json(path):
            if not path.exists():
                return {}
            try:
                return json.loads(path.read_text())
            except (OSError, ValueError) as exc:
                e.errors.append(f"{path}: {exc}")
                return {}
        lineage = {"family": family} if stage == "efit" else {}
        e.manifest = read_json(db.omas_manifest(stage, shot=shot, **lineage))
        e.replication = read_json(db.omas_replication_record(stage, shot=shot, **lineage))
        e.diagnostics_manifest = read_json(db.omas_manifest("diagnostics", shot=shot))
        # Legacy JSON products may predate the canonical suffix. Refuse ambiguity.
        def product(path):
            if path.exists():
                return path
            candidates = sorted(path.parent.glob(path.name.split('.')[0] + ".json*"))
            if len(candidates) > 1:
                e.errors.append(f"ambiguous product at {path.parent}")
                return None
            return candidates[0] if candidates else None
        for name, roots in [(stage, ("equilibrium",)), ("diagnostics", ("magnetics", "tf", "pf_active", "spectrometer_uv", "summary")),
                            ("eddy", ("pf_active", "pf_passive", "em_coupling"))]:
            if name == "eddy" and not self.vacuum:
                continue
            path = product(db.omas_product(name, shot=shot, **(lineage if name == stage else {})))
            if path:
                try:
                    data = load(path)
                    for root in roots:
                        _copy_branch(data, e.ods, root)
                except Exception as exc:
                    e.errors.append(f"{path}: {type(exc).__name__}: {exc}")
        if self.vacuum:
            era = e.diagnostics_manifest.get("machine_version") or e.manifest.get("machine_version")
            if era:
                static_path = db.omas_product("static", machine_version=era)
                if static_path.exists():
                    try:
                        static = load(static_path)
                        for root in ("em_coupling", "wall"):
                            _copy_branch(static, e.ods, root)
                        if not path_count(e.ods, "pf_passive.loop"):
                            _copy_branch(static, e.ods, "pf_passive")
                    except Exception as exc:
                        e.errors.append(f"static vacuum geometry: {exc}")
        e.collection = _collection(e.ods)
        # Missing collection payload is expected for no_output placeholders.
        if not e.collection:
            e.collection = {"slice_statuses": e.manifest.get("slice_statuses", []),
                            "status": e.manifest.get("efit_status"),
                            "efit_preset": e.manifest.get("efit_preset")}
        if stage != "efit":
            records = e.manifest.get("attempts", [])
            if not records and e.manifest.get("reconstruction") and e.manifest.get("configuration", {}).get("time_ms") is not None:
                records = [{"time_ms": e.manifest["configuration"]["time_ms"], **e.manifest["reconstruction"]}]
            if records:
                # A corrective stage's success is not the slice-validation usable verdict.
                e.collection["slice_statuses"] = [{"time": record["time_ms"] / 1000,
                    "overall_status": "stage_" + record.get("status", "unrecorded"), "metrics": {k: v for k, v in record.items() if k not in ("time_ms", "status")},
                    "provenance": {"configuration": e.manifest.get("configuration", {})}} for record in records]
        work = db.omas(stage, shot=shot, **lineage, artifact="work")
        constraints_path = work / "constraints.json"
        if constraints_path.exists():
            try:
                e.constraints = load(constraints_path)
            except Exception as exc:
                e.errors.append(f"{constraints_path}: {exc}")
        input_dir = db.efit(shot, family=family, artifact="input")
        from vaft.code.efit.presets import PRESET_RECORD
        preset_path = input_dir / PRESET_RECORD
        if preset_path.exists():
            e.collection["efit_preset"] = read_json(preset_path)
        runtime_path = db.efit(shot, family=family, artifact="metadata") / "efit_status.txt"
        if not e.collection.get("status") and runtime_path.exists():
            e.collection["status"] = runtime_path.read_text().strip()
        listing = input_dir / "kfiles_generated.txt"
        if listing.exists():
            # Follow only the explicitly listed inputs, never stale wildcard files.
            for text in kfile_manifest_paths(listing.read_text()):
                path = Path(text)
                original = path
                if not path.is_absolute():
                    path = input_dir / path
                if not path.exists():
                    # Mounts change the original /srv prefix. The listed basename
                    # can only resolve under this shot/family's canonical dirs.
                    candidates = [input_dir / original.name,
                                  db.efit(shot, family=family) / "kfile" / original.name]
                    matches = [candidate for candidate in candidates if candidate.is_file()]
                    if len(matches) == 1:
                        path = matches[0]
                try:
                    payload = read_keqdsk(path).namelists
                    from vaft.code.efit.magnetic import _efit_case_time, _efit_case_key
                    e.kfiles.append({"time": _efit_case_time(_efit_case_key(path)),
                                     "path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                                     "namelists": payload})
                except Exception as exc:
                    e.errors.append(f"{path}: {type(exc).__name__}: {exc}")
        e.availability = "present" if path_count(e.ods, "equilibrium.time_slice") else (
            "no_output" if e.manifest.get("status") == "no_output" else "no_equilibrium")
        e.product_id = _digest({"manifest": e.manifest, "collection": e.collection,
                                "kfiles": [{"time": k["time"], "sha256": k["sha256"]} for k in e.kfiles]})
        return e


@dataclass
class EquilibriumSummary:
    shots: pd.DataFrame
    attempts: pd.DataFrame
    checks: pd.DataFrame
    channels: pd.DataFrame
    inputs: pd.DataFrame
    metrics: pd.DataFrame
    metadata: dict

    def export(self, directory):
        """Materialize portable evidence tables plus a reproducibility manifest."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        for name in TABLE_COLUMNS:
            getattr(self, name).to_csv(directory / f"{name}.csv", index=False)
        (directory / "manifest.json").write_text(_json(self.metadata) + "\n")
        (directory / "population.json").write_text(_json(self.population()) + "\n")

    def population(self):
        """Counts with explicit shot/attempt denominators, kept by backend."""
        groups = []
        for (backend, source, family), shots in self.shots.groupby(["backend", "source", "family"]):
            attempts = self.attempts[(self.attempts.backend == backend) &
                                     (self.attempts.source == source) & (self.attempts.family == family)]
            recorded = attempts[attempts.attempt_evidence == "recorded_status"]
            failures = {}
            for text in recorded.failure_codes_json:
                for code in set(json.loads(text)):
                    failures[code] = failures.get(code, 0) + 1
            groups.append({"backend": backend, "source": source, "family": family,
                "requested_shots": len(shots), "shots_with_equilibrium": int((shots.equilibria_present.fillna(0) > 0).sum()),
                "shots_with_recorded_attempts": int((shots.attempts_recorded.fillna(0) > 0).sum()),
                "observed_time_points": len(attempts), "recorded_attempts": len(recorded),
                "outcomes": recorded.overall_status.value_counts().to_dict(), "failure_codes": failures,
                "usable_fraction_recorded": float((recorded.overall_status == "usable").mean()) if len(recorded) and not recorded.overall_status.str.startswith("stage_").any() else None})
        return {"groups": groups, "note": "failure-code counts overlap; output-only or input-only observations are excluded from the recorded-attempt success rate"}


def _times(ods):
    times = path_value(ods, "equilibrium.time", None)
    if times is None:
        times = [path_value(ods, f"equilibrium.time_slice.{i}.time", None)
                 for i in range(path_count(ods, "equilibrium.time_slice"))]
    return list(times)


def _time_key(time):
    value = float(time)
    if not np.isfinite(value):
        raise ValueError("nonfinite attempt time")
    # Join at EFIT's millisecond resolution; report collisions rather than zip.
    return int(round(value * 1000))


def _index_by_time(items, time_getter):
    result = {}
    for item in items:
        key = _time_key(time_getter(item))
        if key in result:
            raise ValueError(f"ambiguous time: multiple records at {key} ms")
        result[key] = item
    return result


def collect_evidence(e: ShotEvidence, *, magnetics=True, vacuum=False):
    """Normalize a storage observation; keep failed and output-less attempts."""
    product = e.product_id or _digest(e.collection or {"shot": e.shot, "source": e.source, "equilibrium": _snapshot(path_value(e.ods, "equilibrium", None))})
    base = dict(shot=e.shot, source=e.source, family=e.family, backend=e.backend, product_id=product)
    tables = {name: [] for name in TABLE_COLUMNS}
    statuses = _index_by_time(e.collection.get("slice_statuses", []), lambda s: s["time"])
    eq = _index_by_time(enumerate(_times(e.ods)), lambda p: p[1])
    constraints = _index_by_time(enumerate(_times(e.constraints)), lambda p: p[1]) if e.constraints is not None else {}
    kfiles = _index_by_time(e.kfiles, lambda k: k["time"])
    for key in set(statuses) & set(kfiles):
        recorded_hash = statuses[key].get("provenance", {}).get("kfile_sha256")
        if recorded_hash and recorded_hash != kfiles[key]["sha256"]:
            raise ValueError(f"k-file generation mismatch at {key} ms; refusing mixed input/output evidence")
    keys = sorted(set(statuses) | set(eq) | set(constraints) | set(kfiles))
    # The temporal check must see every slice, not one-slice copies.
    report = {}
    if eq:
        try:
            from vaft.validation.equilibrium import validate_equilibrium
            report = validate_equilibrium(e.ods, diagnostics=e.ods)
        except Exception as exc:
            e.errors.append(f"equilibrium validation: {type(exc).__name__}: {exc}")
    for key in keys:
        status = statuses.get(key, {})
        index, time = eq.get(key, (None, status.get("time", key / 1000)))
        attempt_id = f"{product}:{key}"
        at = {**base, "attempt_id": attempt_id, "time_s": float(time)}
        provenance = status.get("provenance", {})
        tables["attempts"].append({**at, "eq_index": index, "has_equilibrium": index is not None,
            "attempt_evidence": "recorded_status" if status else "kfile" if key in kfiles else "constraints" if key in constraints else "published_output",
            "overall_status": status.get("overall_status", "unrecorded"),
            **{name: status.get(name) for name in ("runtime_ok", "output_ok", "numerical_ok", "physical_ok")},
            "failure_codes_json": _json(status.get("failure_codes", [])),
            "metrics_json": _json(status.get("metrics", {})),
            "configuration_json": _json(provenance.get("configuration", e.collection.get("efit_preset", {}))),
            "kfile_sha256": provenance.get("kfile_sha256", kfiles.get(key, {}).get("sha256")),
            "provenance_json": _json(provenance)})
        if key in constraints:
            ci = constraints[key][0]
            snapshot = path_value(e.constraints, f"equilibrium.time_slice.{ci}.constraints", None)
            tables["inputs"].append({**at, "kind": "constraint_stage", "snapshot_json": _json(_snapshot(snapshot)), "sha256": _digest(_snapshot(snapshot))})
        if key in kfiles:
            kfile = kfiles[key]
            tables["inputs"].append({**at, "kind": "kfile_namelists", "snapshot_json": _json(kfile["namelists"]), "sha256": kfile["sha256"]})
        if index is not None:
            from vaft.validation.registry import CHECKS
            for category in ("verification", "diagnostic_fit", "physical_validity", "independent_validation"):
                for check, result in report.get(category, {}).items():
                    selected = result
                    if result.get("slices"):
                        slices = [r for r in result["slices"] if r.get("time_slice") == index]
                        # Some providers use index/time_s; match declared report order as fallback.
                        if not slices and len(result["slices"]) == len(report.get("time_slices", [])):
                            slices = [result["slices"][report["time_slices"].index(index)]]
                        if not slices:
                            continue
                        selected = slices[0]
                    if not result.get("slices") and index != min(p[0] for p in eq.values()):
                        continue
                    check_at = at if result.get("slices") else {**base, "attempt_id": None, "time_s": None}
                    name = f"{category}.{check}"
                    spec = CHECKS.get(name)
                    tables["checks"].append({**check_at, "scope": "slice" if result.get("slices") else "product",
                        "check": name, "status": selected.get("status", result.get("status", "not_available")),
                        "unit": spec.unit if spec else None, "provider": spec.provider if spec else None,
                        "result_json": _json(selected)})
            try:
                from vaft.validation.equilibrium_quality import equilibrium_quality_constraint_points
                for point in equilibrium_quality_constraint_points(e.ods, index):
                    tables["channels"].append({**at, "scope": "equilibrium_fit", "family_name": point.get("family"),
                        "channel": point.get("channel"), "channel_index": point.get("index"),
                        "status": "fitted" if point.get("fitted") else "excluded", "details_json": _json(point)})
            except Exception as exc:
                e.errors.append(f"constraint points {time}: {exc}")
    shot_at = {**base, "attempt_id": None, "time_s": None}
    if magnetics:
        try:
            from vaft.validation.magnetics import validate_magnetics_signals, magnetics_quality_metrics
            quality = e.diagnostics_manifest.get("magnetics_quality") or magnetics_quality_metrics(e.ods, validate_magnetics_signals(e.ods))
            qa_status = "evaluated" if quality.get("summary", {}).get("present", 0) else "not_available"
            tables["checks"].append({**shot_at, "scope": "shot", "check": "magnetics.quality", "status": qa_status,
                "provider": "vaft.validation.magnetics", "result_json": _json(quality.get("summary", {}))})
            from vaft.validation.magnetics import channel_node, unusable_channels_at
            known = []
            for channel in quality.get("channels", []):
                kind, channel_index, quantity = channel.get("kind"), channel.get("index"), channel.get("quantity")
                if kind is not None and channel_index is not None and quantity:
                    node = channel_node(kind, channel_index, quantity)
                    if path_value(e.ods, node + ".validity", None) is not None or path_value(e.ods, node + ".validity_timed", None) is not None:
                        known.append((kind, channel_index))
            if tables["attempts"]:
                times = [row["time_s"] for row in tables["attempts"]]
                unusable = unusable_channels_at(e.ods, times) if known else {}
                for i, attempt in enumerate(tables["attempts"]):
                    unavailable = sum(bool(unusable.get(channel, np.zeros(len(times), dtype=bool))[i]) for channel in known)
                    tables["checks"].append({**{key: attempt[key] for key in (*IDENTITY, "attempt_id", "time_s")},
                        "scope": "diagnostic_time_point", "check": "magnetics.diagnostic_validity",
                        "status": "evaluated" if known else "not_available", "provider": "vaft.validation.magnetics.unusable_channels_at",
                        "result_json": _json({"assessed_channels": len(known), "unusable_channels": unavailable,
                                              "unknown_validity_channels": len(quality.get("channels", [])) - len(known)})})
            for channel in quality.get("channels", []):
                tables["channels"].append({**shot_at, "scope": "magnetics_signal", "family_name": channel.get("family"),
                    "channel": channel.get("name"), "channel_index": channel.get("index"),
                    "status": channel.get("status"), "details_json": _json(channel)})
        except Exception as exc:
            e.errors.append(f"magnetics: {exc}")
    if vacuum:
        try:
            from vaft.validation.vacuum_benchmark import run_benchmark_case
            benchmark = run_benchmark_case(e.ods, shot=e.shot)
            tables["checks"].append({**shot_at, "scope": "window", "check": "vacuum.benchmark", "status": "evaluated",
                "provider": "vaft.validation.vacuum_benchmark", "result_json": _json(benchmark)})
            for channel in benchmark.get("metrics", {}).get("channels", []):
                tables["channels"].append({**shot_at, "scope": "vacuum_window", "family_name": channel.get("family"),
                    "channel": channel.get("name"), "channel_index": channel.get("index"),
                    "status": channel.get("status"), "details_json": _json(channel)})
        except Exception as exc:
            tables["checks"].append({**shot_at, "scope": "window", "check": "vacuum.benchmark", "status": "not_available",
                "provider": "vaft.validation.vacuum_benchmark", "result_json": _json({"reason": str(exc)})})
    tables["shots"].append({**base, "machine_era": e.manifest.get("machine_version") or e.diagnostics_manifest.get("machine_version"),
        "availability": e.availability, "stage_status": e.manifest.get("status", "unrecorded"),
        "runtime_status": e.collection.get("status"), "replication_state": e.replication.get("state", "unrecorded"),
        "attempts_recorded": len(statuses), "equilibria_present": len(eq),
        "usable_recorded": sum(s.get("overall_status") == "usable" for s in statuses.values()),
        "attempt_coverage": "recorded_attempts" if statuses else "input_only" if constraints or kfiles else "outputs_only" if eq else "unknown",
        "errors_json": _json(e.errors), "io_metrics_json": _json(e.io_metrics)})
    def numeric_leaves(value, prefix=""):
        if isinstance(value, Mapping):
            for key, item in value.items():
                yield from numeric_leaves(item, f"{prefix}.{key}" if prefix else str(key))
        elif isinstance(value, list):
            for i, item in enumerate(value):
                yield from numeric_leaves(item, f"{prefix}.{i}")
        elif isinstance(value, (float, int)) and not isinstance(value, bool) and np.isfinite(value):
            yield prefix, value
    for row in tables["checks"]:
        for name, value in numeric_leaves(json.loads(row["result_json"])):
            if name in ("time", "time_slice"):
                continue
            tables["metrics"].append({**{key: row.get(key) for key in (*IDENTITY, "attempt_id", "time_s", "scope", "check")},
                                      "metric": name, "value": value})
    for row in tables["attempts"]:
        for name, value in numeric_leaves(json.loads(row["metrics_json"])):
            tables["metrics"].append({**{key: row.get(key) for key in (*IDENTITY, "attempt_id", "time_s")},
                                      "scope": "slice", "check": "efit.collection", "metric": name, "value": value})
    return tables


def equilibrium_summary(shots: Iterable[int], *, mode="hsds", source=None, family="magnetic",
                        filedb=None, magnetics=True, vacuum=False, backends=None):
    """Collect identical tables from HSDS only, or HSDS plus server FileDB.

    Backend is part of the observation key: local and remote products must not
    be pooled in statistical denominators. No-output absence in HSDS is unknown,
    never evidence of solver failure. Supply backends= for future storage adapters.
    """
    from .sources import resolve
    if mode not in ("hsds", "server"):
        raise ValueError("mode must be 'hsds' or 'server'")
    if family not in ("magnetic", "electron", "kinetic"):
        raise ValueError("family must be magnetic, electron or kinetic")
    source = resolve(source or {"magnetic": "main", "electron": "electron-efit", "kinetic": "kinetic-efit"}[family])
    shots = list(shots)
    if any(isinstance(s, bool) or not isinstance(s, (int, np.integer)) or s < 0 for s in shots):
        raise ValueError("shots must be non-negative integers")
    shots = sorted(set(int(s) for s in shots))
    if backends is None:
        backends = [HSDSEvidenceBackend(source, family, magnetics=magnetics, vacuum=vacuum)]
        if mode == "server":
            if filedb is None:
                raise ValueError("server mode requires a local or mounted filedb root")
            backends.append(FileDBEvidenceBackend(filedb, source, family, magnetics=magnetics, vacuum=vacuum))
    backends = list(backends)
    tables = {name: [] for name in TABLE_COLUMNS}
    for shot in shots:
        for backend in backends:
            evidence = backend.read(shot)
            try:
                rows = collect_evidence(evidence, magnetics=magnetics, vacuum=vacuum)
            except Exception as exc:
                # Invalid joins are visible, not quietly attributed to a different slice.
                rows = {name: [] for name in TABLE_COLUMNS}
                rows["shots"].append(dict(shot=shot, backend=evidence.backend, source=evidence.source,
                    family=evidence.family, product_id=evidence.product_id or _digest(evidence.collection),
                    availability="analysis_error", attempt_coverage="unknown", errors_json=_json([*evidence.errors, str(exc)])))
            for name in tables:
                tables[name].extend(rows[name])
    import vaft
    metadata = {"schema_version": SCHEMA_VERSION, "vaft_version": vaft.__version__,
                "generated_at": datetime.now(timezone.utc).isoformat(), "mode": mode,
                "source": source, "family": family, "shots": shots, "magnetics": magnetics,
                "vacuum": vacuum, "backend_types": [type(b).__name__ for b in backends],
                "denominator": "group by backend/source/family; attempts include recorded failures; HSDS absence is unknown"}
    return EquilibriumSummary(**{name: pd.DataFrame(rows, columns=TABLE_COLUMNS[name]) for name, rows in tables.items()}, metadata=metadata)
