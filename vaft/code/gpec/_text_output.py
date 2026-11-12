"""DCON/RDCON ASCII output, for runs that predate VAFT's netCDF-reading path.

The legacy shot-first FileDB (``public/<shot>/linear_stability/<ms>/
{dcon,rdcon}/nn=<n>/``) kept only the solvers' text files: ``dcon.out`` and,
for RDCON, ``delta_gw.out``. :func:`read_dcon_output` and
:func:`read_pest3_matching_output` need the ``*_output_n<n>.nc`` those runs
never wrote. This module reads the same quantities from the text so that the
legacy runs and the current ones go through one container:

* ``dcon.out`` has the equilibrium scalars, the ``mlow``..``psilim`` table, the
  ``Singular Surfaces`` table and the ``Total Energy Eigenvalues`` table. Row
  ``isol = 1`` of that last table is the least-stable total energy, which the
  netCDF carries as ``W_t_eigenvalue[0]`` / ``total1``.
* ``delta_gw.out`` is ``gal_write_delta`` (``rdcon/gal.f``): the raw
  ``(2 msing, 2 msing)`` Galerkin matrix ``delta`` with left/right halves per
  surface. The netCDF stores the same matrix transposed, because a Fortran
  column-major array read back by xarray swaps its indices. The PEST3
  matrices then follow from the left/right sums of ``rdcon_netcdf.f``
  (the ``construct PEST3 matching data`` loop; lines 488-506 at GPEC e68d7ac2), so ``Delta_prime`` here means what it means in the netCDF.

The text prints four significant figures. ``Delta_prime`` is a difference of
four such entries, so it carries an absolute error of a few units in the
fourth digit of the largest of them. On a run that wrote both outputs, the
diagonal matched the netCDF within 0.6 %.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np

from ._matching_output import Pest3MatchingOutput

_NUMBER = r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[EeDd][-+]?\d+)?"
_ROW = re.compile(rf"^\s*(\d+)\s+((?:{_NUMBER}\s*)+)$")
_DELTA_ROW = re.compile(r"^\s*(\d+)([lr])\s+(.*)$")


def _floats(text: str) -> list[float]:
    return [float(x.replace("D", "E").replace("d", "e")) for x in re.findall(_NUMBER, text)]


def _header_table(lines: list[str], header: str) -> dict[str, float]:
    """The one-row table printed under a whitespace-separated header line."""
    names = header.split()
    for index, line in enumerate(lines):
        if line.split() == names:
            for row in lines[index + 1:]:
                if row.strip():
                    values = row.split()
                    out: dict[str, float] = {}
                    for name, value in zip(names, values):
                        try:
                            out[name] = float(value)
                        except ValueError:  # lim_fl is a logical (T/F)
                            out[name] = value
                    return out
    return {}


def _numbered_rows(lines: list[str], start: int, width: int) -> list[list[float]]:
    """Consecutive ``i  v1 .. v<width>`` rows after ``start``, skipping blank lines before the first."""
    rows: list[list[float]] = []
    for line in lines[start:]:
        if not line.strip():
            if rows:
                break
            continue
        match = _ROW.match(line)
        if match is None:
            if rows:
                break
            continue
        values = _floats(match.group(2))
        if len(values) < width:
            break
        rows.append([float(match.group(1)), *values[:width]])
    return rows


def _section_start(lines: list[str], title: str) -> int | None:
    for index, line in enumerate(lines):
        if line.strip().rstrip(":") == title:
            return index
    return None


def read_dcon_text(path: str | Path) -> dict[str, Any]:
    """Equilibrium scalars, truncation, singular surfaces and energies from one ``dcon.out``.

    Returns a mapping with:

    * ``scalars``: every name in the equilibrium summary tables (``betap1``,
      ``li1``, ``q0``, ``q95``, ``qmax`` ...), as DCON printed them;
    * ``truncation``: ``mlow``, ``mhigh``, ``mpert``, ``mband``, ``nn``,
      ``qlim``, ``psilim`` and ``mpsi``;
    * ``singular_surfaces``: one dict per row (``i``, ``psi``, ``rho``,
      ``q``, ``q1``, ``di0``, ``di``, ``err``);
    * ``energies``: one dict per ``isol`` (``plasma``, ``vacuum``, ``total``
      as a complex number), least-stable first, as DCON orders them.

    A section DCON did not print (a run that stopped before the energy
    solve) is empty rather than an error, so the caller decides what an
    incomplete run means.
    """
    lines = Path(path).read_text(encoding="utf-8", errors="replace").splitlines()
    scalars: dict[str, Any] = {}
    for header in (
        "mpsi mtheta psilow psihigh rs_right rs_left zs_top zs_bot",
        "amean rmean aratio kappa delta1 delta2 volume",
        "li1 li2 li3 ro zo psio",
        "betap1 betap2 betat betan betaj bwall bt0",
        "q0 qmin q95 qmax qa crnt I/aB",
    ):
        scalars.update(_header_table(lines, header))
    truncation = _header_table(lines, "mlow mhigh mpert mband nn lim_fl dmlim qlim psilim")
    if "mpsi" in scalars:
        truncation["mpsi"] = scalars["mpsi"]

    surfaces: list[dict[str, float]] = []
    start = _section_start(lines, "Singular Surfaces")
    if start is not None:
        names = ("i", "psi", "rho", "q", "q1", "di0", "di", "err")
        for row in _numbered_rows(lines, start + 1, len(names) - 1):
            surfaces.append(dict(zip(names, row)))
            surfaces[-1]["i"] = int(surfaces[-1]["i"])

    energies: list[dict[str, Any]] = []
    start = _section_start(lines, "Total Energy Eigenvalues")
    if start is not None:
        for row in _numbered_rows(lines, start + 1, 4):
            energies.append({"isol": int(row[0]), "plasma": row[1], "vacuum": row[2],
                             "total": complex(row[3], row[4])})
    return {"scalars": scalars, "truncation": truncation,
            "singular_surfaces": surfaces, "energies": energies}


def read_delta_gw(path: str | Path) -> np.ndarray:
    """The first ``Delta matrix`` block of a ``delta_gw.out``, as a complex array in file order.

    Rows and columns are the ``1l, 1r, 2l, 2r, ...`` labels the file prints,
    so element ``[a, b]`` is the Fortran ``delta(a+1, b+1)``.
    """
    rows: list[list[float]] = []
    labels: list[tuple[int, str]] = []
    for line in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
        match = _DELTA_ROW.match(line)
        if match is None:
            if rows and line.strip() and not line.lstrip().startswith("i\\j"):
                break
            continue
        label = (int(match.group(1)), match.group(2))
        if label in labels:  # a second block starts
            break
        labels.append(label)
        rows.append(_floats(match.group(3)))
    if not rows:
        raise ValueError(f"{path}: no Delta matrix rows")
    width = {len(row) for row in rows}
    size = len(rows)
    if width != {2 * size}:
        raise ValueError(f"{path}: Delta matrix is not square ({size} rows, widths {sorted(width)})")
    values = np.asarray(rows, dtype=float)
    return values[:, 0::2] + 1j * values[:, 1::2]


def pest3_matrices_from_delta(delta: np.ndarray) -> dict[str, np.ndarray]:
    """``A_prime``, ``B_prime``, ``Gamma_prime``, ``Delta_prime`` from the Galerkin ``delta``.

    ``delta`` is in Fortran index order, rows and columns ``1l, 1r, 2l, 2r, ...``,
    the way :func:`read_delta_gw` returns it. The result is transposed into the
    orientation :func:`read_pest3_matching_output` gets from the netCDF, so one
    container means one thing whichever file it was read from.
    """
    delta = np.asarray(delta)
    if delta.ndim != 2 or delta.shape[0] != delta.shape[1] or delta.shape[0] % 2:
        raise ValueError(f"delta must be a square (2 msing, 2 msing) matrix, got {delta.shape}")
    left = slice(0, None, 2)
    right = slice(1, None, 2)
    ll, lr = delta[left, left], delta[left, right]
    rl, rr = delta[right, left], delta[right, right]
    # rdcon_netcdf.f:488-506, with 2*ising -> right and 2*ising-1 -> left.
    return {
        "A_prime": (rr + rl + lr + ll).T,
        "B_prime": (rr - rl + lr - ll).T,
        "Gamma_prime": (rr + rl - lr - ll).T,
        "Delta_prime": (rr - rl - lr + ll).T,
    }


def read_pest3_matching_text(run_dir: str | Path, *, solver: str = "rdcon", mode: int) -> Pest3MatchingOutput:
    """A :class:`Pest3MatchingOutput` from an RDCON run that left only ``dcon.out`` and ``delta_gw.out``.

    The poloidal mode number of each singular surface is ``round(n q)``, the
    resonance condition. The text does not print it, but the netCDF ``r``
    variable holds exactly that value. ``total1`` is the ``isol = 1`` total
    energy of the run's ``dcon.out``. The local-stability profiles are not
    read: the text prints them on DCON's ``ipsi`` grid only to the precision
    of the table, and no comparison here needs them.
    """
    if solver != "rdcon":
        raise ValueError(f"only RDCON writes delta_gw.out; got solver={solver!r}")
    run_dir = Path(run_dir)
    text = read_dcon_text(run_dir / "dcon.out")
    delta = read_delta_gw(run_dir / "delta_gw.out")
    surfaces = text["singular_surfaces"]
    msing = delta.shape[0] // 2
    if len(surfaces) != msing:
        raise ValueError(
            f"{run_dir}: dcon.out lists {len(surfaces)} singular surfaces but delta_gw.out is {msing}-surface"
        )
    truncation = text["truncation"]
    n_tor = int(truncation.get("nn", mode))
    if n_tor != int(mode):
        raise ValueError(f"{run_dir}: dcon.out is an n = {n_tor} run, not the requested n = {mode}")
    q = np.array([s["q"] for s in surfaces], dtype=float)
    energies = text["energies"]
    total = energies[0] if energies else None
    return Pest3MatchingOutput(
        solver=solver,
        n_tor=n_tor,
        mlow=int(truncation["mlow"]),
        mhigh=int(truncation["mhigh"]),
        mpert=int(truncation["mpert"]),
        mband=int(truncation["mband"]),
        m=np.rint(n_tor * q).astype(int),
        psi_n_rational=np.array([s["psi"] for s in surfaces], dtype=float),
        q_rational=q,
        Delta=delta.T,
        plasma1=None if total is None else complex(total["plasma"]),
        vacuum1=None if total is None else complex(total["vacuum"]),
        total1=None if total is None else total["total"],
        metadata={"run_dir": str(run_dir), "text_files": ["dcon.out", "delta_gw.out"],
                  "mpsi": truncation.get("mpsi")},
        **pest3_matrices_from_delta(delta),
    )
