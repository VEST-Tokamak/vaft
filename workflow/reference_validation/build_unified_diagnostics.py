"""Regenerate the small, explicitly cross-shot VEST diagnostic fixture.

Run from a repository checkout. Source artifacts are intentionally not needed by
users of the generated fixture (and several are excluded from distributions).
"""

from __future__ import annotations

import copy
import gzip
import hashlib
import json
from pathlib import Path
import re
from tempfile import TemporaryDirectory

import numpy as np
import yaml
from omas import ODS

from vaft.machine_mapping.coils_non_axisymmetric import coils_non_axisymmetric
from vaft.machine_mapping.ec_launchers import resolve_ec_launcher_geometry
from vaft.machine_mapping.interferometer import interferometer
from vaft.machine_mapping.nbi import nbi
from vaft.machine_mapping.registry import port_phi
from vaft.machine_mapping.thomson_scattering import (
    LASER_DUMP_PORT, LASER_ENTRY_PORT, PORT_MAJOR_RADIUS_M,
    scattering_volume_phi,
)


ROOT = Path(__file__).resolve().parents[2] / "vaft/data"
OUTPUT = ROOT / "unified/vest_diagnostics"
NOTICE = "Cross-shot composite fixture — not a physical VEST discharge"


def _read(name: str) -> dict:
    with gzip.open(ROOT / name, "rt", encoding="utf-8") as handle:
        return json.load(handle)


def _hash(name: str) -> str:
    digest = hashlib.sha256()
    with (ROOT / name).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _slice_signal(signal: dict, indices: list[int]) -> None:
    for key in ("time", "data"):
        if key not in signal:
            continue
        values = signal[key]
        if isinstance(values, list) and len(values) == 1 and isinstance(values[0], list):
            signal[key] = [[values[0][index] for index in indices]]
        else:
            signal[key] = [values[index] for index in indices]


def _indices(time: list[float], start: float, end: float, maximum: int) -> list[int]:
    candidates = [index for index, value in enumerate(time) if start <= value <= end]
    if not candidates:
        raise ValueError(f"source has no samples in [{start}, {end}] s")
    stride = max(1, (len(candidates) + maximum - 1) // maximum)
    return candidates[::stride]


def _source(shot: int, artifact: str, ids: list[str], time: str, processing: str,
            nature: str, geometry: str) -> dict:
    return {
        "source_shot": shot,
        "source_artifact": artifact,
        "source_sha256": _hash(artifact),
        "ids": ids,
        "source_time": time,
        "processing": processing,
        "value_kind": nature,
        "geometry_compatibility": geometry,
    }


def _geometry_records(fixture: dict) -> list[dict]:
    """Small derived geometry with exact source coordinates and derivations."""
    records = []
    records.append({
        "family": "thomson_scattering", "semantic": "trajectory",
        "label": "Thomson laser 8MM10 to 1MM10 (derived port chord)",
        "r_m": [PORT_MAJOR_RADIUS_M] * 2, "z_m": [0.0, 0.0],
        "phi_rad": [port_phi(LASER_ENTRY_PORT), port_phi(LASER_DUMP_PORT)],
        "source_keys": ["thomson_scattering"],
        "value_kind": "derived geometry; not surveyed as-built trajectory",
        "derivation": "thomson_scattering.py entry/dump ports, vest port map, 0.803 m port-flange radius; midplane path",
    })
    for index, channel in enumerate(fixture["thomson_scattering"]["channel"][:5]):
        r = float(channel["position"]["r"])
        z = float(channel["position"]["z"])
        if not np.isclose(z, 0.0, rtol=0, atol=1e-12):
            raise ValueError(f"Thomson channel {index} is not on the mapped midplane chord")
        records.append({
            "family": "thomson_scattering", "semantic": "point",
            "label": f"Thomson scattering site {index + 1} (derived)",
            "r_m": [r], "z_m": [z],
            "phi_rad": [scattering_volume_phi(r)],
            "source_keys": ["thomson_scattering"],
            "value_kind": "derived local scattering position; measurement values remain direct",
            "derivation": f"thomson_scattering.scattering_volume_phi applied to stored channel {index} radius; port-flange chord, not survey",
        })

    ec = resolve_ec_launcher_geometry(39915)
    if ec is None:
        raise ValueError("reference shot 39915 has no EC geometry revision")
    phi = float(ec["phi"])
    k_r, k_phi, k_z = ec["direction"]
    ec_direction = [float(k_r * np.cos(phi) - k_phi * np.sin(phi)),
                    float(k_r * np.sin(phi) + k_phi * np.cos(phi)), float(k_z)]
    records.append({
        "family": "ec_launchers", "semantic": "point", "label": "EC 6 kW launch position (provisional CAD)",
        "r_m": [ec["r"]], "z_m": [ec["z"]], "phi_rad": [phi],
        "source_keys": ["ec_launcher_geometry"], "value_kind": "provisional CAD-derived geometry",
        "derivation": f"resolve_ec_launcher_geometry(39915), port={ec['port']}, status={ec['status']}, revision={ec['revision']}; no power implication",
    })
    records.append({
        "family": "ec_launchers", "semantic": "directed_axis", "label": "EC launch axis (provisional; displayed extent only)",
        "r_m": [ec["r"]], "z_m": [ec["z"]], "phi_rad": [phi],
        "direction_xyz": ec_direction, "source_keys": ["ec_launcher_geometry"],
        "value_kind": "provisional CAD-derived launch direction; not wave propagation/absorption",
        "derivation": "resolve_ec_launcher_geometry(39915) stored R/phi/Z direction rotated to machine XYZ at stored launch phi",
    })

    mapped = ODS(consistency_check=False)
    nbi(mapped)
    group = "nbi.unit.0.beamlets_group.0"
    r = float(mapped[f"{group}.position.r"])
    phi = float(mapped[f"{group}.position.phi"])
    z = float(mapped[f"{group}.position.z"])
    tangent = float(mapped[f"{group}.tangency_radius"])
    sense = int(mapped[f"{group}.direction"])
    # NUBEAM source and aperture are both at z=0; verify the raw model input,
    # rather than inferring a horizontal axis from source elevation alone.
    model_text = (ROOT / "nubeam/vest_case/mdescr_VEST_190307.dat").read_text()
    aperture_height = re.search(r"^\s*Zbap\(1\)\s*=\s*([-+\d.]+)", model_text, re.M)
    if aperture_height is None or float(aperture_height[1]) != z:
        raise ValueError("NUBEAM aperture elevation does not match source")
    distance = float(np.sqrt(r * r - tangent * tangent))
    foot_phi = phi + sense * float(np.arctan2(distance, tangent))
    source_xy = r * np.array([np.cos(phi), np.sin(phi)])
    foot_xy = tangent * np.array([np.cos(foot_phi), np.sin(foot_phi)])
    along = (foot_xy - source_xy) / np.linalg.norm(foot_xy - source_xy)
    records.append({
        "family": "nbi", "semantic": "point", "label": "NBI B_001 source (NUBEAM model)",
        "r_m": [r], "z_m": [z], "phi_rad": [phi], "source_keys": ["nbi_model"],
        "value_kind": "NUBEAM model-derived geometry, not as-built",
        "derivation": "nbi mapper source position from vest.yaml/mdescr model",
    })
    records.append({
        "family": "nbi", "semantic": "directed_axis", "label": "NBI B_001 beam axis (NUBEAM model; displayed extent only)",
        "r_m": [r], "z_m": [z], "phi_rad": [phi],
        "direction_xyz": [float(along[0]), float(along[1]), 0.0],
        "source_keys": ["nbi_model"], "value_kind": "NUBEAM model-derived beam direction, not as-built",
        "derivation": "nbi mapper source r/phi, model tangency radius and clockwise sense; source Zbsc(1) and aperture Zbap(1) both 0 in mdescr_VEST_190307.dat",
    })
    return records


def main() -> None:
    machine = _read("samples/39915/omas.json.gz")
    kinetic = _read("samples/48224/omas.json.gz")
    langmuir = _read("legacy/langmuir_probes_42699.json.gz")
    sxr = _read("samples/45531/omas.json.gz")

    # Keep only the anchored machine outlines and sensor locations. No 39915
    # equilibrium is mixed with the internally consistent 48224 kinetic subset.
    fixture = {key: copy.deepcopy(machine[key]) for key in ("wall", "pf_active", "pf_passive")}
    fixture["magnetics"] = {
        "ids_properties": copy.deepcopy(machine["magnetics"]["ids_properties"]),
        "b_field_pol_probe": [
            {key: copy.deepcopy(value) for key, value in channel.items()
             if key in ("name", "identifier", "position", "poloidal_angle", "length", "type")}
            for channel in machine["magnetics"]["b_field_pol_probe"]
        ],
        "flux_loop": [
            {key: copy.deepcopy(value) for key, value in channel.items()
             if key in ("name", "identifier", "position")}
            for channel in machine["magnetics"]["flux_loop"]
        ],
    }
    for family in ("thomson_scattering", "charge_exchange", "core_profiles", "equilibrium"):
        fixture[family] = copy.deepcopy(kinetic[family])

    probes = copy.deepcopy(langmuir["langmuir_probes"])
    for probe in probes["embedded"]:
        indices = _indices(probe["time"], 0.35, 0.46, 160)
        probe["time"] = [probe["time"][index] for index in indices]
        for signal in ("n_e", "t_e"):
            _slice_signal(probe[signal], indices)
            if "validity_timed" in probe[signal]:
                probe[signal]["validity_timed"] = [
                    probe[signal]["validity_timed"][index] for index in indices
                ]
    fixture["langmuir_probes"] = probes

    channels = copy.deepcopy(sxr["soft_x_rays"]["channel"][:2])
    for channel in channels:
        indices = _indices(channel["brightness"]["time"], 0.299, 0.303, 192)
        _slice_signal(channel["brightness"], indices)
        if "validity_timed" in channel:
            _slice_signal(channel["validity_timed"], indices)
    fixture["soft_x_rays"] = {
        "ids_properties": copy.deepcopy(sxr["soft_x_rays"]["ids_properties"]),
        "channel": channels,
    }

    mapped = ODS(consistency_check=False)
    interferometer(
        mapped, 47230,
        mat_file_94ghz=ROOT / "legacy/47230_056789_LID_1_100.mat",
        mat_file_282ghz=ROOT / "legacy/47230_ALL_LID_1_100.mat",
    )
    with TemporaryDirectory() as temporary:
        artifact = Path(temporary) / "mapped.json"
        mapped.save(str(artifact))
        interferometer_data = json.loads(artifact.read_text(encoding="utf-8"))["interferometer"]
    interferometer_data["channel"] = [
        interferometer_data["channel"][index] for index in (0, 5)
    ]
    for channel in interferometer_data["channel"]:
        signal = channel["n_e_line"]
        indices = _indices(signal["time"], 0.299, 0.303, 192)
        _slice_signal(signal, indices)
    fixture["interferometer"] = interferometer_data

    # One exact mapped filament, with every stored element endpoint retained.
    coils_non_axisymmetric(mapped, options={"coil_sets": ["MID"]})
    with TemporaryDirectory() as temporary:
        artifact = Path(temporary) / "coils.json"
        mapped.save(str(artifact))
        mapped_coils = json.loads(artifact.read_text(encoding="utf-8"))["coils_non_axisymmetric"]
    mapped_coils["coil"] = mapped_coils["coil"][:1]
    fixture["coils_non_axisymmetric"] = mapped_coils

    fixture["dataset_description"] = {
        "data_entry": {"machine": "VEST", "pulse_type": "cross-shot diagnostic fixture", "user": "vaft"},
        "ids_properties": {"comment": NOTICE + "; geometry reference: shot 39915", "homogeneous_time": 0},
    }

    OUTPUT.mkdir(parents=True, exist_ok=True)
    artifact = OUTPUT / "omas.json.gz"
    encoded = json.dumps(fixture, separators=(",", ":"), allow_nan=True).encode("utf-8")
    with artifact.open("wb") as stream:
        with gzip.GzipFile(filename="", mode="wb", fileobj=stream, compresslevel=9, mtime=0) as handle:
            handle.write(encoded)

    manifest = {
        "schema_version": 1,
        "kind": "cross-shot-diagnostic-fixture",
        "machine": "VEST",
        "physical_discharge": False,
        "notice": NOTICE,
        "geometry_reference": {
            "source_shot": 39915, "pf_geometry_version": "1906", "wall_geometry_version": "1512",
            "meaning": "Other-shot diagnostic geometry is shown against this reference machine; operation was not simultaneous.",
        },
        "artifact": {"path": "omas.json.gz", "sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
                     "size": artifact.stat().st_size},
        "sources": {
            "machine_geometry": _source(39915, "samples/39915/omas.json.gz", ["wall", "pf_active", "pf_passive"], "static geometry", "copied, geometry only", "mapped", "reference"),
            "magnetics": _source(39915, "samples/39915/omas.json.gz", ["magnetics"], "static sensor positions only", "geometry selection", "mapped", "reference"),
            "thomson_scattering": _source(48224, "samples/48224/omas.json.gz", ["thomson_scattering"], "native 0.298–0.307 s", "copied from the 48224 kinetic subset without refitting", "measured local n_e and T_e", "48224 positions projected on 39915 geometry reference"),
            "charge_exchange": _source(48224, "samples/48224/omas.json.gz", ["charge_exchange"], "native 0.297–0.301 s", "copied from the 48224 kinetic subset without refitting", "measured local impurity-ion T_i and V_phi", "48224 positions projected on 39915 geometry reference"),
            "core_profiles": _source(48224, "samples/48224/omas.json.gz", ["core_profiles"], "profile slice at 0.300 s", "copied together with its 48224 measurements and equilibrium", "fitted n_e, T_e, T_i and V_phi", "48224 PF era 2507; not a 39915 machine profile"),
            "equilibrium": _source(48224, "samples/48224/omas.json.gz", ["equilibrium"], "reconstruction slice at 0.300 s", "copied together with the 48224 kinetic subset", "reconstructed equilibrium", "48224 PF era 2507; never overlay as the 39915 equilibrium"),
            "interferometer_94ghz": _source(47230, "legacy/47230_056789_LID_1_100.mat", ["interferometer"], "native samples within 0.299–0.303 s", "vaft.machine_mapping.interferometer; one horizontal chord; sparse sample selection", "measured line-integrated density", "geometry projected on reference; no local density inference"),
            "interferometer_282ghz": _source(47230, "legacy/47230_ALL_LID_1_100.mat", ["interferometer"], "native samples within 0.299–0.303 s", "vaft.machine_mapping.interferometer; one vertical chord; sparse sample selection", "measured line-integrated density", "geometry projected on reference"),
            "langmuir_probes": _source(42699, "legacy/langmuir_probes_42699.json.gz", ["langmuir_probes"], "native samples within 0.35–0.46 s", "existing mapped triple-probe output; sparse sample selection", "derived local n_e and T_e", "positions projected on reference; shot-era compatibility to be checked before physical interpretation"),
            "soft_x_rays": _source(45531, "samples/45531/omas.json.gz", ["soft_x_rays"], "native samples within 0.299–0.303 s", "existing mapped output; first two vertical chords; sparse sample selection", "relative calibrated SXR chord signal proxy (not absolute brightness)", "LOS projected on reference; no local temperature inference"),
            "ec_launcher_geometry": {
                "source_shot": 39915,
                "source_artifact": "vaft/machine_mapping/vest.yaml",
                "source_sha256": hashlib.sha256((ROOT.parents[1] / "vaft/machine_mapping/vest.yaml").read_bytes()).hexdigest(),
                "ids": ["ec_launchers"], "source_time": "static revision from shot 29500",
                "processing": "resolve_ec_launcher_geometry(39915); provisional CAD and port map",
                "value_kind": "provisional CAD-derived launch geometry; no power",
                "geometry_compatibility": "mapped geometry compatible with shot-39915 reference era",
            },
        },
    }
    manifest["geometry_sources"] = {
        "coil_filament_model": {
            "source_artifact": "gpec/vest_MID.dat", "source_sha256": _hash("gpec/vest_MID.dat"),
            "model_reference_shot": 48226,
            "ids": ["coils_non_axisymmetric"], "source_time": "static reference geometry",
            "processing": "coils_non_axisymmetric mapper; MID sector 1 only, exact conductor endpoints",
            "value_kind": "mapped 3-D filament; excitation absent",
            "geometry_compatibility": "shot-48226 reference model projected on 39915 geometry; no simultaneous operation claim",
        },
        "nbi_model": {
            "source_artifact": "nubeam/vest_case/mdescr_VEST_190307.dat",
            "source_sha256": _hash("nubeam/vest_case/mdescr_VEST_190307.dat"),
            "source_model_key": 0,
            "ids": ["nbi"], "source_time": "static NUBEAM case model",
            "processing": "nbi mapper and vest.yaml, source/aperture heights checked against mdescr",
            "value_kind": "model-derived; not as-built or discharge measurement",
            "geometry_compatibility": "model geometry projected on shot-39915 reference",
        },
    }
    manifest["geometry_records"] = _geometry_records(fixture)
    for family in ("interferometer_94ghz", "interferometer_282ghz"):
        manifest["sources"][family]["mapping_sources"] = [
            {"path": path, "sha256": hashlib.sha256(
                (ROOT.parents[1] / path).read_bytes()
            ).hexdigest()}
            for path in (
                "vaft/machine_mapping/vest.yaml",
                "vaft/machine_mapping/interferometer.py",
            )
        ]
    mapping_sources = {
        "thomson_scattering": ("vaft/machine_mapping/thomson_scattering.py", "vaft/machine_mapping/registry.py"),
        "ec_launcher_geometry": ("vaft/machine_mapping/ec_launchers.py", "vaft/machine_mapping/registry.py"),
        "nbi_model": ("vaft/machine_mapping/nbi.py", "vaft/machine_mapping/vest.yaml", "vaft/machine_mapping/registry.py"),
        "coil_filament_model": ("vaft/machine_mapping/coils_non_axisymmetric.py",),
    }
    for source_key, paths in mapping_sources.items():
        (manifest["sources"].get(source_key) or manifest["geometry_sources"][source_key])["mapping_sources"] = [
            {"path": path, "sha256": hashlib.sha256((ROOT.parents[1] / path).read_bytes()).hexdigest()}
            for path in paths
        ]
    manifest["camera_calibration_reference"] = {
        "camera_shot": 39915,
        "meaning": "Calibration for geometric projection only; no camera frame copied or simultaneous observation implied",
        "assets": [
            {"path": path, "sha256": _hash(path)}
            for path in ("geometry/camera_visible/intrinsics.json", "geometry/camera_visible/pose_39915.json")
        ],
    }
    (OUTPUT / "manifest.yaml").write_text(yaml.safe_dump(manifest, sort_keys=False, allow_unicode=True), encoding="utf-8")
    print(f"{artifact}: {artifact.stat().st_size} bytes")


if __name__ == "__main__":
    main()
