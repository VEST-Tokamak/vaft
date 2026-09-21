"""Prepare a standalone NICE native-input directory from a canonical ODS."""

from __future__ import annotations

from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
from typing import Any, Iterable
import xml.etree.ElementTree as ET

import numpy as np

from .config import NiceConfig, NiceInputs
from .diagnostics import diagnostics_from_ods
from .geometry import geometry_hash, nice_geometry_from_ods


def _hash_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _git_revision(path: Path) -> str | None:
    try:
        completed = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            check=True,
            text=True,
            capture_output=True,
            timeout=5,
        )
        return completed.stdout.strip() or None
    except Exception:
        return None


def _write(path: Path, lines: Iterable[str]) -> Path:
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _infer_shot(ods: Any, config: NiceConfig) -> int:
    if config.shot is not None:
        return int(config.shot)
    for path in (
        "dataset_description.data_entry.pulse",
        "summary.global_quantities.pulse",
    ):
        try:
            return int(ods[path])
        except Exception:
            pass
    raise ValueError("NICE shot is required in NiceConfig.shot or ODS metadata")


def _resolve_time(ods: Any, config: NiceConfig) -> float:
    if config.time_index is not None:
        return float(np.asarray(ods["equilibrium.time"], float)[int(config.time_index)])
    if config.time is not None:
        return float(config.time)
    raise ValueError("NiceConfig.time or time_index is required")


def _interp(ods: Any, data_path: str, time_path: str, time: float) -> float:
    return float(
        np.interp(
            time, np.asarray(ods[time_path], float), np.asarray(ods[data_path], float)
        )
    )


def _passive_currents(ods: Any, time: float, config: NiceConfig) -> tuple[float, ...]:
    count = len(ods["pf_passive.loop"])
    if config.passive_currents is not None:
        values = tuple(float(v) for v in config.passive_currents)
        if len(values) != count:
            raise ValueError(
                f"passive_currents has {len(values)} entries; pf_passive has {count} loops"
            )
        return values
    try:
        times = np.asarray(ods["pf_passive.time"], float)
    except Exception:
        # Compact/legacy VAFT samples store passive currents on the shared PF
        # timebase without duplicating pf_passive.time.
        try:
            times = np.asarray(ods["pf_active.time"], float)
        except Exception:
            times = None
    values = []
    for i in range(count):
        current = np.asarray(ods[f"pf_passive.loop.{i}.current"], float)
        if times is None or current.size != times.size:
            raise ValueError(f"No time-aligned fixed current for pf_passive.loop.{i}")
        values.append(float(np.interp(time, times, current)))
    return tuple(values)


def _four_points(entry: dict[str, Any]) -> list[list[float]]:
    points = entry["outline"]
    if len(points) != 4:
        raise ValueError(
            f"{entry['ods_path']} has {len(points)} outline points; standalone NICE "
            "rectangular coils.txt requires exactly four"
        )
    return points


def _subtract_fixed_passive_response(geometry, diagnostics, currents):
    """Remove prescribed wall-current signals before the plasma fit."""
    from vaft.formula.magnetics import project_poloidal_field
    from vaft.process.electromagnetics import compute_point_response_matrices

    passive_geometry = geometry["pf_passive"]
    source_points = np.asarray(
        [
            np.mean(np.asarray(entry["outline"], float), axis=0)
            for entry in passive_geometry
        ]
    )
    groups = np.asarray([entry["loop_index"] for entry in passive_geometry], dtype=int)
    n_passive = len(currents)
    corrected = []
    for diagnostic in diagnostics:
        if not diagnostic.enabled or diagnostic.family == "plasma_current":
            corrected.append(diagnostic)
            continue
        if diagnostic.family == "bpol_probe":
            positions = [[diagnostic.geometry["r"], diagnostic.geometry["z"]]]
        else:
            positions = [[p["r"], p["z"]] for p in diagnostic.geometry["positions"]]
        positions = np.asarray(positions, float)
        psi, bz, br = compute_point_response_matrices(
            positions[:, 0],
            positions[:, 1],
            source_points[:, 0],
            source_points[:, 1],
            turns=np.ones(len(source_points)),
            groups=groups,
            n_groups=n_passive,
            components=("psi", "bz", "br"),
        )
        if diagnostic.family == "bpol_probe":
            response = project_poloidal_field(
                br[0], bz[0], diagnostic.geometry["poloidal_angle"]
            )
        else:
            response = np.mean(psi, axis=0)
        passive_value = float(np.asarray(response) @ np.asarray(currents, float))
        corrected.append(
            replace(
                diagnostic,
                value=diagnostic.value - passive_value,
                passive_response=passive_value,
            )
        )
    return tuple(corrected)


def prepare_nice_inputs(ods: Any, config: NiceConfig) -> NiceInputs:
    """Write deterministic NICE text inputs without importing or executing NICE."""
    if config.cocos_in != 11 or config.cocos_out != 11:
        raise ValueError(
            "The validated standalone adapter currently supports COCOS 11 only"
        )
    if config.flux_loop_input_sign not in (-1, 1):
        raise ValueError("flux_loop_input_sign must be +1 or -1")
    workdir = Path(config.workdir).expanduser()
    input_dir, output_dir = workdir / "input", workdir / "output"
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError(
            "NICE output directory is not empty; use a fresh case to avoid stale success"
        )
    input_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    (workdir / "restart").mkdir(parents=True, exist_ok=True)
    shot, time = _infer_shot(ods, config), _resolve_time(ods, config)
    geometry = nice_geometry_from_ods(ods)
    passive = _passive_currents(ods, time, config)
    diagnostics = diagnostics_from_ods(ods, time, config)
    if config.passive_current_mode not in {
        "diagnostic_subtraction",
        "external_coils",
        "ignore",
    }:
        raise ValueError(
            "passive_current_mode must be diagnostic_subtraction, external_coils, or ignore"
        )
    if config.passive_current_mode == "diagnostic_subtraction":
        diagnostics = _subtract_fixed_passive_response(geometry, diagnostics, passive)
    active_audit = []
    if config.correct_active_response:
        from .response import active_response

        diagnostics, active_audit = active_response(ods, geometry, diagnostics, time)

    files: list[Path] = []
    files.append(
        _write(
            input_dir / "limiter.txt",
            [str(len(geometry["limiter"]))]
            + [f"{r:.17g} {z:.17g}" for r, z in geometry["limiter"]],
        )
    )

    conductors = list(geometry["pf_active"])
    conductor_currents = []
    for entry in conductors:
        idx = int(entry["coil_index"])
        conductor_currents.append(
            _interp(ods, f"pf_active.coil.{idx}.current.data", "pf_active.time", time)
        )
    if config.passive_current_mode == "external_coils":
        for entry in geometry["pf_passive"]:
            conductors.append(entry)
            conductor_currents.append(passive[int(entry["loop_index"])])
    if len(conductors) > 100:
        raise ValueError(
            f"NICE supports at most 100 coils, but this mapping produced {len(conductors)}; "
            "use passive_current_mode='diagnostic_subtraction' for VEST"
        )
    coil_lines = [str(len(conductors))]
    for entry in conductors:
        p = _four_points(entry)
        points = np.asarray(p, float)
        r0, r1 = points[:, 0].min(), points[:, 0].max()
        z0, z1 = points[:, 1].min(), points[:, 1].max()
        coil_lines.append(
            f"{0.5 * (r0 + r1):.17g} {0.5 * (z0 + z1):.17g} "
            f"{r1 - r0:.17g} {z1 - z0:.17g} {entry['turns']:.17g}"
        )
    files.append(_write(input_dir / "coils.txt", coil_lines))
    files.append(
        _write(
            input_dir / "Icoils.txt",
            [str(len(conductor_currents))] + [f"{v:.17g}" for v in conductor_currents],
        )
    )

    # Fixed passives are represented once, as prescribed external conductors.
    files.append(_write(input_dir / "passive_structure.txt", ["0"]))
    files.append(_write(input_dir / "passive_structure_conductivity.txt", ["0"]))

    bpol = [d for d in diagnostics if d.family == "bpol_probe" and d.enabled]
    flux = [d for d in diagnostics if d.family == "flux_loop" and d.enabled]
    files.append(
        _write(
            input_dir / "Bprobes.txt",
            [str(len(bpol))]
            + [
                f"{d.geometry['r']:.17g} {d.geometry['z']:.17g} {d.geometry['poloidal_angle']:.17g}"
                for d in bpol
            ],
        )
    )
    files.append(
        _write(
            input_dir / "Bprobes_meas.txt",
            [str(len(bpol))] + [f"{d.value:.17g} {d.uncertainty:.17g} 0" for d in bpol],
        )
    )
    flux_lines = [str(len(flux))]
    for d in flux:
        positions = d.geometry["positions"]
        flux_lines.append(
            " ".join(
                [str(len(positions))]
                + [f"{p[k]:.17g}" for p in positions for k in ("r", "z", "phi")]
            )
        )
    files.append(_write(input_dir / "fluxloops.txt", flux_lines))
    files.append(
        _write(
            input_dir / "fluxloops_meas.txt",
            [str(len(flux))]
            + [
                f"{config.flux_loop_input_sign * d.value:.17g} {d.uncertainty:.17g} 0"
                for d in flux
            ],
        )
    )

    ip = next(
        (d.value for d in diagnostics if d.family == "plasma_current" and d.enabled),
        float("nan"),
    )
    if not np.isfinite(ip):
        raise ValueError("NICE requires a finite enabled plasma-current diagnostic")
    f0 = _interp(ods, "tf.b_field_tor_vacuum_r.data", "tf.time", time)
    files.append(
        _write(
            input_dir / "Ip_B0.txt", [f"{ip:.17g}", f"{f0 / config.major_radius:.17g}"]
        )
    )

    if config.parameter_file is not None:
        source = Path(config.parameter_file).expanduser()
        if not source.is_file():
            raise FileNotFoundError(f"NICE parameter file not found: {source}")
        target = input_dir / "param.xml"
        shutil.copyfile(source, target)
        root = ET.parse(target).getroot()
        cocos_node = root.find(".//inCOCOS")
        if cocos_node is None or int(cocos_node.text) != config.cocos_in:
            found = None if cocos_node is None else cocos_node.text
            raise ValueError(
                f"param.xml inCOCOS={found!r} does not match NiceConfig.cocos_in={config.cocos_in}"
            )
        output_cocos_node = root.find(".//outCOCOS")
        if output_cocos_node is None or int(output_cocos_node.text) != config.cocos_out:
            found = None if output_cocos_node is None else output_cocos_node.text
            raise ValueError(
                f"param.xml outCOCOS={found!r} does not match NiceConfig.cocos_out={config.cocos_out}"
            )
        files.append(target)
        if (
            root.findtext("useNewCOCOSManager") != "1"
            or root.findtext("inoutCOCOS") != str(config.cocos_in)
            or config.cocos_in != config.cocos_out
        ):
            raise ValueError(
                "Effective NICE inoutCOCOS does not match config; explicitly set useNewCOCOSManager=1 and identical input/output COCOS"
            )
        if root.find("r_eqx_contour") is not None:
            from .geometry import validate_contour

            if int(root.findtext("n_points_eqx_contour", "0")) != len(
                np.fromstring(root.findtext("r_eqx_contour"), sep=" ")
            ):
                raise ValueError("Computational contour node count mismatch")
            validate_contour(
                np.fromstring(root.findtext("r_eqx_contour"), sep=" "),
                np.fromstring(root.findtext("z_eqx_contour"), sep=" "),
                geometry,
            )

    passive_bytes = json.dumps(passive, separators=(",", ":")).encode()
    channel_payload = [asdict(d) for d in diagnostics]
    native_hashes = {p.name: _hash_bytes(p.read_bytes()) for p in files}
    diagnostic_hash = _hash_bytes(
        json.dumps(
            channel_payload, sort_keys=True, separators=(",", ":"), allow_nan=True
        ).encode()
    )
    channel_set = [
        {
            key: row[key]
            for key in (
                "family",
                "ods_path",
                "identifier",
                "geometry",
                "enabled",
                "reason",
            )
        }
        for row in channel_payload
    ]
    channel_set_hash = _hash_bytes(
        json.dumps(
            channel_set, sort_keys=True, separators=(",", ":"), allow_nan=True
        ).encode()
    )
    semantic_hashes = {
        "geometry": geometry_hash(geometry),
        "diagnostics": diagnostic_hash,
        "passive_current": _hash_bytes(passive_bytes),
    }
    snapshot_hash = config.input_snapshot_hash or _hash_bytes(
        json.dumps(semantic_hashes, sort_keys=True, separators=(",", ":")).encode()
    )
    nice_revision = config.source_revision
    if nice_revision is None and config.nice_home is not None:
        nice_revision = _git_revision(Path(config.nice_home).expanduser())
    vaft_revision = config.vaft_revision or _git_revision(Path(__file__).parents[3])
    manifest = {
        "schema_version": 1,
        "solver": "NICE",
        "shot": shot,
        "time_s": time,
        "nice_source_revision": nice_revision,
        "cocos_in": config.cocos_in,
        "cocos_out": config.cocos_out,
        "effective_cocos": config.cocos_in,
        "flux_loop_input_sign": config.flux_loop_input_sign,
        "active_response_audit": active_audit,
        "diagnostic_source": config.diagnostic_source,
        "geometry_hash": semantic_hashes["geometry"],
        "diagnostic_channels": channel_payload,
        "diagnostic_channel_hash": diagnostic_hash,
        "diagnostic_channel_set_hash": channel_set_hash,
        "passive_current_A": list(passive),
        "passive_current_hash": semantic_hashes["passive_current"],
        "passive_current_treatment": {
            "diagnostic_subtraction": "fixed_forward_model_subtracted_from_diagnostics",
            "external_coils": "fixed_external_one_turn_conductors",
            "ignore": "geometry_only",
        }[config.passive_current_mode],
        "native_input_hashes": native_hashes,
        "native_input_hash": _hash_bytes(
            json.dumps(native_hashes, sort_keys=True).encode()
        ),
        "input_snapshot_hash": snapshot_hash,
        "vaft_revision": vaft_revision,
        "build_options": dict(config.build_options),
        "profile_basis": dict(config.profile_basis),
        "solver_tolerances": dict(config.solver_tolerances),
        "initialization_method": config.initialization_method,
    }
    manifest_file = workdir / "nice_case_manifest.json"
    manifest_file.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=True), encoding="utf-8"
    )
    files.append(manifest_file)
    return NiceInputs(
        workdir,
        input_dir,
        output_dir,
        shot,
        time,
        geometry,
        diagnostics,
        passive,
        manifest,
        manifest_file,
        tuple(files),
        ods,
    )
