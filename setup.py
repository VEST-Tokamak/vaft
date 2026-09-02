"""Setuptools hooks for the compact wheel-only reference sample."""

from __future__ import annotations

from pathlib import Path
from shutil import copy2

from setuptools import setup
from setuptools.command.build_py import build_py as _build_py


class build_py(_build_py):
    """Replace checkout-sized shot 39915 artifacts in build outputs only."""

    def run(self) -> None:
        super().run()
        project_root = Path(__file__).parent
        wheel_variant = project_root / "packaging" / "wheel_samples" / "39915"
        destination = Path(self.build_lib) / "vaft" / "data" / "samples" / "39915"
        destination.mkdir(parents=True, exist_ok=True)
        for filename in ("manifest.yaml", "omas.json.gz", "imas.nc"):
            source = wheel_variant / filename
            if not source.is_file():
                raise FileNotFoundError(f"Missing generated wheel sample artifact: {source}")
            copy2(source, destination / filename)


setup(cmdclass={"build_py": build_py})
