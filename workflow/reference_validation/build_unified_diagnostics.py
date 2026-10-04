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
from tempfile import TemporaryDirectory

import yaml
from omas import ODS

from vaft.machine_mapping.interferometer import interferometer


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
            "kinetic": _source(48224, "samples/48224/omas.json.gz", ["thomson_scattering", "charge_exchange", "core_profiles", "equilibrium"], "Thomson 0.298–0.307 s; equilibrium/core_profiles near 0.300 s; CX native time retained", "copied together without refitting or time alignment", "measured; fitted/reconstructed", "2507 PF era; equilibrium must not be presented as the 39915 machine configuration"),
            "interferometer_94ghz": _source(47230, "legacy/47230_056789_LID_1_100.mat", ["interferometer"], "native samples within 0.299–0.303 s", "vaft.machine_mapping.interferometer; one horizontal chord; sparse sample selection", "measured line-integrated density", "geometry projected on reference; no local density inference"),
            "interferometer_282ghz": _source(47230, "legacy/47230_ALL_LID_1_100.mat", ["interferometer"], "native samples within 0.299–0.303 s", "vaft.machine_mapping.interferometer; one vertical chord; sparse sample selection", "measured line-integrated density", "geometry projected on reference"),
            "langmuir_probes": _source(42699, "legacy/langmuir_probes_42699.json.gz", ["langmuir_probes"], "native samples within 0.35–0.46 s", "existing mapped triple-probe output; sparse sample selection", "derived local n_e and T_e", "positions projected on reference; shot-era compatibility to be checked before physical interpretation"),
            "soft_x_rays": _source(45531, "samples/45531/omas.json.gz", ["soft_x_rays"], "native samples within 0.299–0.303 s", "existing mapped output; first two vertical chords; sparse sample selection", "measured chord brightness", "LOS projected on reference; no local temperature inference"),
        },
    }
    (OUTPUT / "manifest.yaml").write_text(yaml.safe_dump(manifest, sort_keys=False, allow_unicode=True), encoding="utf-8")
    print(f"{artifact}: {artifact.stat().st_size} bytes")


if __name__ == "__main__":
    main()
