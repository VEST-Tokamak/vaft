#!/usr/bin/env python3
"""Verify that built VAFT distributions obey the PyPI data policy."""

from __future__ import annotations

import argparse
import tarfile
import zipfile
from pathlib import Path


# Single source of truth for the ``vaft/data`` policy: individually whitelisted
# files, plus the suffixes allowed inside each shipped category directory.
# Keep in sync with ``[tool.setuptools.package-data]`` and ``MANIFEST.in``.
_ALLOWED_DATA_FILES = {
    "geometry/Coil_info.mat",
    "geometry/VEST_DiscretizedCoilGeometry_Full_ver_1906.mat",
    "geometry/VEST_DiscretizedCoilGeometry_Full_ver_2507.mat",
    "omas/39915.json",
}
_ALLOWED_DATA_SUFFIXES = {
    "geometry/": (".yaml", ".csv"),
    "gpec/": (".in", ".dat"),
    "legacy/": (".txt",),
}

REQUIRED_FILES = {
    "vaft/.hscfg.example",
    "vaft/machine_mapping/vest.yaml",
    "vaft/data/geometry/MD.yaml",
    "vaft/data/geometry/line_of_sight_endpoints.csv",
    "vaft/data/gpec/gpec.in",
    "vaft/data/legacy/sql_table.txt",
} | {f"vaft/data/{name}" for name in _ALLOWED_DATA_FILES}


def _distribution_names(path: Path) -> set[str]:
    if path.suffix == ".whl":
        with zipfile.ZipFile(path) as archive:
            return set(archive.namelist())

    with tarfile.open(path) as archive:
        names = set()
        for info in archive.getmembers():
            if not info.isfile():
                continue
            _, separator, relative = info.name.partition("/")
            if separator and relative:
                names.add(relative)
        return names


def _allowed_data_file(name: str) -> bool:
    if not name.startswith("vaft/data/"):
        return True

    relative = name.removeprefix("vaft/data/")
    if "/" not in relative:
        return relative.endswith(".py")
    if relative in _ALLOWED_DATA_FILES:
        return True
    for category, suffixes in _ALLOWED_DATA_SUFFIXES.items():
        if relative.startswith(category):
            return relative.endswith(suffixes)
    return False


def _verify_distribution(path: Path) -> None:
    names = _distribution_names(path)
    missing = sorted(REQUIRED_FILES - names)
    if missing:
        raise ValueError(f"{path.name}: missing required files: {', '.join(missing)}")

    forbidden_data = sorted(name for name in names if not _allowed_data_file(name))
    if forbidden_data:
        raise ValueError(
            f"{path.name}: contains repository-only data: {', '.join(forbidden_data)}"
        )

    if path.suffix != ".whl" and any(name.startswith("test/") for name in names):
        raise ValueError(f"{path.name}: source distribution must not include repository tests")

    print(f"verified {path.name} ({len(names)} files)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dist_dir", type=Path, help="directory containing wheel and sdist artifacts")
    parser.add_argument("--max-wheel-mib", type=float, default=25.0)
    args = parser.parse_args()

    wheels = sorted(args.dist_dir.glob("*.whl"))
    sdists = sorted(args.dist_dir.glob("*.tar.gz"))
    if len(wheels) != 1 or len(sdists) != 1:
        raise ValueError("expected exactly one wheel and one source distribution")

    wheel_size_mib = wheels[0].stat().st_size / 1024 / 1024
    if wheel_size_mib > args.max_wheel_mib:
        raise ValueError(
            f"{wheels[0].name}: {wheel_size_mib:.1f} MiB exceeds "
            f"the {args.max_wheel_mib:.1f} MiB limit"
        )

    for path in (*wheels, *sdists):
        _verify_distribution(path)
    print(f"wheel size: {wheel_size_mib:.1f} MiB")


if __name__ == "__main__":
    main()
