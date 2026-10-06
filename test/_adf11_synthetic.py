"""Synthetic ADF11 tables for C and O, written once per session: no network, no OPEN-ADAS.

The shape follows test/test_impurity_charge_states.py: ionisation that turns on
with T_e stage by stage, constant recombination, on a 2 x 4 (n_e, T_e) grid.
"""

from __future__ import annotations

import functools
import tempfile
from pathlib import Path

LOG_NE = (10.0, 14.0)            # cm^-3
LOG_TE = (0.0, 1.0, 2.0, 3.0)    # 1 eV .. 1 keV


def _table(path: Path, blocks) -> Path:
    lines = [f"{len(blocks)} {len(LOG_NE)} {len(LOG_TE)} / synthetic", "",
             " ".join(map(str, LOG_NE)), " ".join(map(str, LOG_TE))]
    for index, values in enumerate(blocks, start=1):
        lines.append(f"---------------- /IPRT=1/IGRD=1/TYPE=TEST/Z1={index}/")
        lines.append(" ".join(f"{v:.3f}" for v in values))
    path.write_text("\n".join(lines) + "\n", encoding="ascii")
    return path


def _element(directory: Path, symbol: str, z: int) -> tuple[Path, Path]:
    scd, acd = [], []
    for j in range(z):
        threshold = 0.6 + 0.35 * j          # log10 T_e where stage j ionises fast
        per_te = [-8.0 - 3.0 * max(threshold - t, 0.0) for t in LOG_TE]
        scd.append([v for v in per_te for _ in LOG_NE])
        acd.append([-11.0] * (len(LOG_NE) * len(LOG_TE)))
    return (_table(directory / f"acd96_{symbol.lower()}.dat", acd),
            _table(directory / f"scd96_{symbol.lower()}.dat", scd))


@functools.lru_cache(maxsize=None)
def synthetic_adf11_tables() -> dict[str, tuple[str, str]]:
    """``{element: (acd path, scd path)}`` for C and O, as the ``adf11_tables=`` option takes them."""
    directory = Path(tempfile.mkdtemp(prefix="vaft-adf11-synthetic-"))
    return {symbol: tuple(str(p) for p in _element(directory, symbol, z)) for symbol, z in (("C", 6), ("O", 8))}
