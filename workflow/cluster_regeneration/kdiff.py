#!/usr/bin/env python3
"""Numeric diff of two directories of EFIT k-files.

usage: kdiff.py <dir A> <dir B>

For every k-file present in both, every namelist variable is parsed (``n*value`` repeats
expanded, ``*_DIR`` paths skipped) and compared as max|a - b| / max|b|. Prints, per variable,
the worst relative difference over all files and the file it occurred in; variables equal
everywhere are not printed.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np

ASSIGNMENT = re.compile(r"\s*([A-Za-z_][A-Za-z0-9_()]*)\s*=(.*)")
REPEAT = re.compile(r"^(\d+)\*(.*)$")


def _number(token: str) -> float:
    return float(token.replace("d", "e").replace("D", "E"))


def parse(path: Path) -> dict[str, np.ndarray]:
    values: dict[str, list[float]] = {}
    current = None
    for line in path.read_text().splitlines():
        line = line.split("!")[0]
        match = ASSIGNMENT.match(line)
        if match:
            current, rest = match.group(1).upper(), match.group(2)
            values[current] = []
        else:
            rest = line
        if current is None or "DIR" in current:
            continue
        for token in re.split(r"[,\s]+", rest.strip()):
            token = token.strip().rstrip("/")
            if not token:
                continue
            repeat = REPEAT.match(token)
            try:
                if repeat:
                    values[current] += [_number(repeat.group(2))] * int(repeat.group(1))
                else:
                    values[current].append(_number(token))
            except ValueError:
                pass
    return {name: np.array(v) for name, v in values.items() if v}


def main(a_dir: Path, b_dir: Path) -> None:
    worst: dict[str, tuple[float, str]] = {}
    for a_file in sorted(a_dir.glob("k*")):
        b_file = b_dir / a_file.name
        if not b_file.exists():
            print("missing in B", a_file.name)
            continue
        a, b = parse(a_file), parse(b_file)
        for name in set(a) | set(b):
            if name not in a or name not in b or a[name].shape != b[name].shape:
                worst[name] = (float("inf"), a_file.name)
                continue
            scale = max(float(np.max(np.abs(b[name]))), 1e-300)
            rel = float(np.max(np.abs(a[name] - b[name]))) / scale
            if rel > worst.get(name, (0.0, ""))[0]:
                worst[name] = (rel, a_file.name)
    for name, (rel, where) in sorted(worst.items(), key=lambda item: -item[1][0]):
        print(f"{name:12s} {'shape/missing' if rel == float('inf') else f'{rel:.3e}'} {where}")


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
