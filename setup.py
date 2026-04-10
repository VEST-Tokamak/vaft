"""Legacy setuptools entrypoint.

VAFT packaging metadata and dependencies are defined in `pyproject.toml`.
This file is kept for compatibility with tooling that still invokes `setup.py`.
"""

from setuptools import setup

if __name__ == "__main__":
    setup()