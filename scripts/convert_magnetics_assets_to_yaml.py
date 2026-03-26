"""Convert donor magnetics assets into minimal YAML payloads.

This keeps only the fields currently consumed by the canonical magnetics port:

- `MD.mat` -> ordered channel list with `{field_code, kind, calibration}`
- `VEST_MagneticsGeometry_Full_ver_2302.mat` -> ordered geometry list with
  `{field_code, kind, r, z}`
- `table.dat` -> only the names referenced by the magnetics geometry/code files
"""

from __future__ import annotations

from pathlib import Path

import scipy.io
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_ROOT = REPO_ROOT.parent / "vest_database" / "OMAS" / "Geometry"
DEFAULT_TARGET_ROOT = REPO_ROOT / "vaft" / "data" / "geometry"


def _kind_from_nature(nature: float) -> str:
    return "flux_loop" if int(nature) == 1 else "b_field_pol_probe"


def _load_md_rows(path: Path, include_geometry: bool) -> list[dict[str, float | int | str]]:
    rows = scipy.io.loadmat(path)["md"]
    payload = []
    for row in rows:
        entry = {
            "field_code": int(row[4]),
            "kind": _kind_from_nature(row[3]),
        }
        if include_geometry:
            entry["r"] = float(row[0])
            entry["z"] = float(row[1])
        else:
            entry["calibration"] = float(row[2])
        payload.append(entry)
    return payload


def _load_referenced_names(path: Path, used_codes: set[int]) -> list[dict[str, int | str]]:
    entries = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        parts = line.split("'")
        field_code = int(parts[0])
        if field_code not in used_codes:
            continue
        entries.append(
            {
                "field_code": field_code,
                "name": parts[1].strip(),
            }
        )
    entries.sort(key=lambda item: item["field_code"])
    return entries


def _dump_yaml(path: Path, payload: dict) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False, allow_unicode=False)


def main() -> None:
    source_root = DEFAULT_SOURCE_ROOT
    target_root = DEFAULT_TARGET_ROOT
    target_root.mkdir(parents=True, exist_ok=True)

    md_path = source_root / "MD.mat"
    geometry_path = source_root / "VEST_MagneticsGeometry_Full_ver_2302.mat"
    table_path = source_root / "table.dat"

    dynamic_channels = _load_md_rows(md_path, include_geometry=False)
    geometry_channels = _load_md_rows(geometry_path, include_geometry=True)
    used_codes = {entry["field_code"] for entry in dynamic_channels} | {
        entry["field_code"] for entry in geometry_channels
    }
    names = _load_referenced_names(table_path, used_codes)

    _dump_yaml(
        target_root / "MD.yaml",
        {
            "source": "MD.mat",
            "description": "Minimal ordered channel metadata used by vfit_md",
            "channels": dynamic_channels,
        },
    )
    _dump_yaml(
        target_root / "VEST_MagneticsGeometry_Full_ver_2302.yaml",
        {
            "source": "VEST_MagneticsGeometry_Full_ver_2302.mat",
            "description": "Minimal ordered geometry metadata used by magnetics static geometry population",
            "channels": geometry_channels,
        },
    )
    _dump_yaml(
        target_root / "table.yaml",
        {
            "source": "table.dat",
            "description": "Only the field-code labels referenced by the magnetics assets",
            "entries": names,
        },
    )


if __name__ == "__main__":
    main()
