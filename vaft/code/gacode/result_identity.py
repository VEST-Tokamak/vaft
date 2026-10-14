"""Compact scientific identity for solver collections and local runs (#2026).

Only standard code.parameters XML is used; native arrays stay in their source.
"""

import hashlib
import json
import re
import xml.etree.ElementTree as ET
from collections.abc import Mapping
from pathlib import Path

import numpy as np

__all__ = [
    "collection_record",
    "digest",
    "encode",
    "input_value_status",
    "parameters",
    "plain",
    "prepare_target",
    "preserve_time_semantics",
    "read_native_input",
]


def plain(value):
    if isinstance(value, np.ndarray):
        return plain(value.tolist())
    if isinstance(value, np.generic):
        return plain(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, Mapping):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    return value


def digest(value):
    return hashlib.sha256(
        json.dumps(
            plain(value), sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def parameters(text):
    """Decode our XML envelope, including OMAS-decoded dictionary representations."""
    if isinstance(text, Mapping):
        node = text.get("parameters", text)
        node = node.get("gyrokinetic_result", {}) if isinstance(node, Mapping) else {}
        if isinstance(node, Mapping):
            node = node.get("#text", "")
        try:
            return json.loads(node) if node else None
        except (TypeError, ValueError):
            return None
    try:
        node = ET.fromstring(str(text)).find("gyrokinetic_result")
        return json.loads(node.text) if node is not None else None
    except (ET.ParseError, TypeError, ValueError):
        return None


def encode(record, existing=None):
    if isinstance(existing, Mapping):
        import xmltodict

        existing = xmltodict.unparse(
            existing if "parameters" in existing else {"parameters": existing}
        )
    try:
        root = ET.fromstring(str(existing))
    except ET.ParseError:
        root = ET.Element("parameters")
    if root.tag != "parameters":
        raise ValueError("code.parameters must have parameters root")
    old = root.find("gyrokinetic_result")
    if old is not None:
        root.remove(old)
    ET.SubElement(root, "gyrokinetic_result").text = json.dumps(
        plain(record), sort_keys=True, allow_nan=False
    )
    return ET.tostring(root, encoding="unicode")


def _parse_input(path):
    result = {}
    if not path.is_file():
        return result
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].split("!", 1)[0].strip()
        if not line:
            continue
        if "=" in line:
            k, v = line.split("=", 1)
        else:
            parts = line.rsplit(None, 1)
            if len(parts) != 2:
                continue
            v, k = parts
        k = k.strip().upper()
        v = v.strip().strip("'\"")
        if not re.fullmatch(r"[A-Z][A-Z0-9_]*", k):
            continue
        if v.lower() in (".true.", ".false.", "t", "f"):
            v = v.lower() in (".true.", "t")
        else:
            try:
                v = float(v.replace("D", "e").replace("d", "e"))
            except ValueError:
                pass
        result[k] = v
    return result


def read_native_input(directory, solver):
    """Read generated value/KEY records or requested KEY=VALUE syntax."""
    directory = Path(directory)
    path = directory / f"input.{solver.lower()}.gen"
    if not path.is_file():
        path = directory / f"input.{solver.lower()}"
    return _parse_input(path), path if path.is_file() else None


# Exclude only surface-dependent physical inputs, retaining all other controls.
_LOCAL_TGLF = {
    "SIGN_BT",
    "SIGN_IT",
    "BETAE",
    "XNUE",
    "ZEFF",
    "DEBYE",
    "VEXB_SHEAR",
    "NS",
}
_LOCAL_CGYRO = {
    "IPCCW",
    "BTCCW",
    "RMIN",
    "RMAJ",
    "Q",
    "S",
    "SHIFT",
    "KAPPA",
    "S_KAPPA",
    "DELTA",
    "S_DELTA",
    "ZETA",
    "S_ZETA",
    "ZMAG",
    "DZMAG",
    "BETAE_UNIT",
    "NU_EE",
    "LAMBDA_STAR",
    "BETA_STAR",
    "GAMMA_E",
    "GAMMA_P",
    "MACH",
    "N_SPECIES",
}


def _local_key(key, solver):
    return (
        key in (_LOCAL_TGLF if solver == "TGLF" else _LOCAL_CGYRO)
        or (solver == "TGLF" and key.endswith("_LOC"))
        or bool(re.search(r"_\d+$", key))
    )


def collection_record(
    solver,
    surfaces,
    *,
    state_identity=None,
    realization_id=None,
    configuration=None,
    windows=None,
):
    configs = []
    runs = []
    states = []
    species_identities = []
    for j, (local, native) in enumerate(surfaces):
        charges = getattr(local, "zs", None)
        masses = getattr(local, "mass", None)
        if hasattr(local, "species"):
            charges = local.species["Z"]
            masses = local.species["MASS"]
        species_identities.append(
            digest(
                {
                    "names": getattr(local, "names", ()),
                    "charges": charges,
                    "masses": masses,
                }
            )
        )
        inputs, path = read_native_input(getattr(native, "directory", ""), solver)
        version = getattr(native, "version", None)
        cfg = {k: v for k, v in inputs.items() if not _local_key(k, solver)}
        saturation = getattr(native, "saturation_parameters", None) or {}
        if solver == "TGLF":
            for key in ("SAT_RULE", "XNU_MODEL", "WDIA_TRAPPED", "UNITS"):
                if key in saturation:
                    cfg[key] = saturation[key]
        if solver == "TGLF" and str((version or {}).get("revision", "")).startswith(
            "b493397"
        ):
            # tglf_startup.f90 at this pinned revision hard-codes USE_PRESETS=.TRUE.
            sat = int(cfg.get("SAT_RULE", -1))
            if sat in (0, 1, 2, 3):
                cfg["WDIA_TRAPPED"] = 1.0 if sat in (2, 3) else 0.0
                cfg["XNU_MODEL"] = 3 if sat in (2, 3) else 2
                if sat == 0:
                    cfg["UNITS"] = "GYRO"
                elif sat in (2, 3) and cfg.get("UNITS") == "GYRO":
                    cfg["UNITS"] = "CGYRO"
                if sat == 0 and cfg.get("NMODES", 0) > 2:
                    cfg["NMODES"] = 4
                if cfg.get("USE_BPER"):
                    cfg["ALPHA_MACH"] = 0.0
                cfg["preset_evidence"] = "tglf_startup.f90 b493397 USE_PRESETS=.TRUE."
        if configuration is not None:
            supplied = plain(configuration)
            if any(key in cfg and cfg[key] != value for key, value in supplied.items()):
                raise ValueError(
                    "supplied configuration contradicts effective native settings"
                )
            cfg.update(supplied)
        # Without auditable configuration, isolate this source rather than merging it.
        if not cfg:
            cfg = {
                "unverified_source": str(getattr(native, "directory", "")),
                "local_type": type(local).__name__,
            }
        cfg["solver_version"] = (version or {}).get("revision")
        configs.append(cfg)
        radius = float(getattr(local, "r_over_a", getattr(local, "rho", 0)))
        state = {
            k: plain(getattr(local, k))
            for k in (
                "geometry",
                "species",
                "zs",
                "mass",
                "as_",
                "taus",
                "rlns",
                "rlts",
                "q_loc",
                "rmaj_loc",
                "rmin_loc",
                "vexb_shear",
            )
            if hasattr(local, k)
        }
        norm = getattr(local, "normalisation", None)
        if norm is not None:
            state["normalisation"] = plain(vars(norm))
        states.append(
            {
                "radius": radius,
                "inputs": state,
                "native_local_inputs": {
                    k: v for k, v in inputs.items() if _local_key(k, solver)
                },
            }
        )
        run = {
            "radius": radius,
            "source": str(getattr(native, "directory", "")),
            "input_sha256": hashlib.sha256(path.read_bytes()).hexdigest()
            if path
            else None,
            "input_provenance": plain(getattr(local, "provenance", {})),
            "native_input": inputs,
            "requested_input": _parse_input(
                Path(getattr(native, "directory", "")) / f"input.{solver.lower()}"
            ),
            "effective_configuration": cfg,
            "effective_model_evidence": "input.gen and native saturation metadata when available; omitted defaults remain unverified",
            "simulation_time_unit": "a/c_s" if solver == "CGYRO" else None,
            "averaging_window": list(windows[j]) if windows is not None else None,
        }
        if solver == "CGYRO":
            from vaft.code.gacode.cgyro.time import time_coordinate_metadata

            run["time_normalization"] = time_coordinate_metadata(native, norm)
            run["averaging_window_unit"] = "a/c_s"
            if (
                windows is not None
                and run["time_normalization"]["status"] == "available"
            ):
                run["averaging_window_elapsed_s"] = (
                    np.asarray(windows[j], dtype=float)
                    * run["time_normalization"]["gyrokinetic_time_scale_s"]
                ).tolist()
        run["physical_input_fingerprint"] = digest(state)
        run["local_run_id"] = digest(
            {
                "source": run["source"],
                "input_sha256": run["input_sha256"],
                "version": version,
            }
        )
        run["input_value_status"] = input_value_status(
            inputs, run["requested_input"], getattr(local, "provenance", {})
        )
        if windows is not None:
            run["analysis_identity"] = digest(
                {
                    "input_sha256": run["input_sha256"],
                    "window": windows[j],
                    "method": "gap-aware piecewise-linear window average",
                }
            )
            lo, hi = map(float, windows[j])
            if not np.isfinite([lo, hi]).all() or hi <= lo:
                raise ValueError("invalid nonlinear averaging window")
            t = np.asarray(
                getattr(native, "time", [])
                if getattr(native, "time", None) is not None
                else [],
                float,
            )
            selected = t[(t >= lo) & (t <= hi)]
            run["averaging_sample_count"] = int(selected.size)
            run["observed_sample_bounds"] = (
                selected[[0, -1]].tolist() if selected.size else None
            )
            run["averaging_method"] = (
                "time-weighted native flux; selected interval is not proof of saturation"
            )
            if hasattr(native, "flux_time_trace"):
                trace = native.flux_time_trace()
                if trace is not None and len(t) >= 2:
                    from vaft.process.gyrokinetics import reduce_time_series

                    energy = np.sum(trace[:, 1, :], axis=0)
                    run["selected_window_statistics"] = reduce_time_series(
                        t[: energy.size], energy, window=(lo, hi)
                    )
        native_record = Path(getattr(native, "directory", "")) / "record.json"
        if native_record.is_file():
            try:
                source_record = json.loads(native_record.read_text(encoding="utf-8"))
            except (ValueError, OSError):
                source_record = {}
            if source_record.get("run_identity"):
                run["local_run_id"] = source_record["run_identity"]
                run["identity_evidence"] = "retained VAFT run record"
        runs.append(run)
    if len({run["radius"] for run in runs}) != len(runs):
        raise ValueError("duplicate radial surfaces in one collection")
    runs.sort(key=lambda run: run["radius"])
    states.sort(key=lambda state: state["radius"])
    for index, run in enumerate(runs):
        run["grid_index"] = index
    if len(set(species_identities)) != 1:
        raise ValueError("surfaces disagree on species identity/order")
    if len({digest(c) for c in configs}) != 1:
        raise ValueError("surfaces disagree on effective model configuration/revision")
    record = {
        "schema_version": 1,
        "solver": solver,
        "configuration": configs[0],
        "realization_id": realization_id
        or digest({"solver": solver, "configuration": configs[0]}),
        "state_identity": state_identity or digest(states),
        "state_evidence": "explicit" if state_identity else "inferred_local_inputs",
        "local_runs": runs,
    }
    return record


def prepare_target(ods, record, time, time_index):
    """Resolve collection/time, rejecting ambiguous replacement before any write."""
    from vaft.ods_access import path_count

    if not np.isfinite(time) or time_index < 0:
        raise ValueError("finite experimental time and nonnegative index required")
    count = path_count(ods, "core_transport.model")
    model = count
    previous = None
    matches = []
    for i in range(count):
        base = f"core_transport.model.{i}"
        old = parameters(ods.get(f"{base}.code.parameters", None))
        if (
            old
            and old.get("solver") == record["solver"]
            and old.get("realization_id") == record["realization_id"]
        ):
            matches.append((i, old))
    if len(matches) > 1:
        raise ValueError("ambiguous duplicate realization identity in existing ODS")
    if matches:
        model, previous = matches[0]
        if previous.get("configuration") != record["configuration"]:
            raise ValueError("realization identity collides with a different configuration")
    base = f"core_transport.model.{model}"
    slices = list((previous or {}).get("slices", []))
    same = [s for s in slices if s["time"] == float(time)]
    if len(same) > 1:
        raise ValueError("ambiguous duplicate experimental time in existing realization")
    if same:
        if same[0]["state_identity"] != record["state_identity"]:
            raise ValueError("different plasma state at an existing experimental time")
        previous_runs = {r["radius"]: r for r in same[0].get("local_runs", [])}
        for run in record["local_runs"]:
            old_run = previous_runs.get(run["radius"])
            if old_run and old_run.get("physical_input_fingerprint") != run.get(
                "physical_input_fingerprint"
            ):
                raise ValueError(
                    "same state identity has incompatible local physical inputs"
                )
        time_index = same[0]["index"]
    elif any(s["index"] == time_index for s in slices):
        raise ValueError("time_index already belongs to another experimental time")
    slices = [s for s in slices if s["index"] != time_index]
    slices.append(
        {
            "index": time_index,
            "time": float(time),
            "state_identity": record["state_identity"],
            "state_evidence": record["state_evidence"],
            "local_runs": record["local_runs"],
        }
    )
    record = {
        k: v
        for k, v in record.items()
        if k not in ("state_identity", "state_evidence", "local_runs")
    }
    record["slices"] = sorted(slices, key=lambda s: s["time"])
    return model, time_index, record


def input_value_status(effective, requested, provenance):
    """Distinguish physical evidence from the effective numeric solver input."""
    result = {}
    aliases = {
        "GAMMA_E": "gamma_e",
        "VEXB_SHEAR": "vexb_shear",
        "MACH": "mach",
        "GAMMA_P": "gamma_p",
    }
    for key, alias in aliases.items():
        source = dict(provenance.get(alias, {}) or {})
        physical = source.get("kind", "unavailable")
        if key not in effective:
            status = "unavailable"
        elif physical in ("unavailable", "solver_default") and source:
            status = "solver_default"
        elif physical in ("measured", "inferred"):
            status = physical
        elif physical == "derived":
            status = "inferred"
        elif key in requested:
            status = "assumed"
        else:
            status = "solver_default"
        result[key] = {
            "effective_value": effective.get(key),
            "status": status,
            "requested": key in requested and status != "solver_default",
            "present_in_input_file": key in requested,
            "physical_evidence": source or {"kind": "unavailable"},
        }
    return result


def preserve_time_semantics(ods):
    """Use heterogeneous IDS time when per-model stored slice coordinates differ."""
    from vaft.ods_access import path_count

    if ods.get("core_transport.ids_properties.homogeneous_time") == 0:
        return
    axes = []
    for m in range(path_count(ods, "core_transport.model")):
        base = f"core_transport.model.{m}.profiles_1d"
        axes.append([ods.get(f"{base}.{i}.time") for i in range(path_count(ods, base))])
    if len(axes) > 1 and any(axis != axes[0] for axis in axes[1:]):
        ods["core_transport.ids_properties.homogeneous_time"] = 0
