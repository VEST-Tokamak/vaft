"""Synthetic GPEC ASCII outputs, shaped like the real ``gpec_*_n<mode>.out`` files.

Layouts follow the shipped DIII-D examples (GPEC v1.5.5): a ``KIND: description``
line, a version line, file-level ``key = value`` scalars, then titled sections
whose blocks carry their own scalars, a column header and fixed-width numeric
rows.  The awkward details are deliberate, because each is somewhere a reader
can go wrong:

* ``gpec_response`` repeats the section title "Eigenvectors" for two sections
  with different columns;
* a legend line such as ``rho = Reluctance (power norm)`` is ``key = value``
  shaped but is prose, and must not become a scalar;
* one real singcoup section title contains GPEC's own typo, "to to";
* a run whose count is zero writes a column header with no rows under it.

Nothing here is copied from a real file: the numbers are synthetic, so the
fixtures carry no run's data.
"""

from __future__ import annotations

import numpy as np

SINGCOUP_SECTIONS = (
    ("The coupling matrix to effective resonant fields", "C_f"),
    ("The coupling matrix to singular currents", "C_i"),
    ("The coupling matrix to square of island half-widths", "C_w"),
    # GPEC's own typo, kept so a reader that "corrects" it is caught.
    ("The coupling matrix to to penetrated resonant fields", "C_p"),
    ("The coupling matrix to unitless Delta", "C_d"),
)

VERSION = "v1.5.5-test"


def _rows(values) -> str:
    return "\n".join(
        "  " + "".join(f"{value:17.8E}" if isinstance(value, float) else f"{value:5d}" for value in row)
        for row in values
    )


def write_singcoup_matrix(path, *, n=1, m_range=(-2, 2), rational_q=(2.0, 3.0)):
    """Write a miniature ``gpec_singcoup_matrix_n<mode>.out``; returns its data."""
    m = np.arange(m_range[0], m_range[1] + 1)
    blocks = {}
    lines = [
        " GPEC_SINGCOUP_MATRIX: Coupling matrices between resonant field and external field",
        f" {VERSION}",
        "",
        "    jac_out = boozer     tmag_out = 1",
        f"      msing =   {len(rational_q)}      mpert = {m.size}       mlow = {m_range[0]}      mhigh =  {m_range[1]}",
        "     psilim =  9.91538663E-001       qlim =  5.20000000E+000",
    ]
    for section_index, (title, symbol) in enumerate(SINGCOUP_SECTIONS):
        lines += ["", f" {title}"]
        for surface_index, q in enumerate(rational_q):
            psi = 0.5 + 0.1 * surface_index
            scale = (section_index + 1) * (surface_index + 1)
            values = (m * 1e-5 * scale) + 1j * (m * 1e-6 * scale)
            blocks[(title, q)] = values
            lines += [
                "",
                f"  q = {q:.3f}  psi = {psi: .8E}",
                "",
                f"    m        real({symbol})        imag({symbol})",
                _rows([(int(mm), float(v.real), float(v.imag)) for mm, v in zip(m, values)]),
            ]
    path = path / f"gpec_singcoup_matrix_n{n}.out"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"m": m, "blocks": blocks, "rational_q": tuple(rational_q), "n": n}


def write_response(path, *, n=1, mode_count=3, m_range=(-1, 1)):
    """Write a miniature ``gpec_response_n<mode>.out``; returns its data."""
    modes = np.arange(1, mode_count + 1)
    m = np.arange(m_range[0], m_range[1] + 1)
    energy = np.stack([modes * 0.1, modes * 0.2, modes * 0.3], axis=1)
    indices = np.stack([modes * -1.0, modes * -2.0], axis=1)
    eigenvector = np.array(
        [[float(mode), float(mm), mode * 1e-3, mm * 1e-4] for mode in modes for mm in m]
    )
    lines = [
        " GPEC_RESPONSE: Response parameters",
        f" {VERSION}",
        "",
        f"      mpert =  {m.size}       mlow = {m_range[0]}      mhigh =  {m_range[1]}",
        "     psilim =  9.91538663E-001       qlim =  5.20000000E+000",
        "",
        " Energy for dcon eigenmodes",
        "",
        " mode          ev0          ev1          iv1",
        _rows([(int(mode), *row) for mode, row in zip(modes, energy)]),
        "",
        " Stability indices",
        "",
        " mode            s           se",
        _rows([(int(mode), *row) for mode, row in zip(modes, indices)]),
        "",
        " Eigenvalues (e) and Singular Values (s)",
        "  jac_type = hamada          ",
        "   L = Vacuum Inductance",
        "   rho = Reluctance (power norm)",
        "   P = Permeability   *Complex (not Hermitian)",
        "",
        " mode              e_L         e_rho",
        _rows([(int(mode), float(mode) * 1e-6, float(mode) * 1e4) for mode in modes]),
        "",
        # Two sections with the same title and different columns.
        " Eigenvectors",
        "",
        " mode    m      real(K_x^L)      imag(K_x^L)",
        _rows([(int(row[0]), int(row[1]), row[2], row[3]) for row in eigenvector]),
        "",
        " Eigenvectors",
        "",
        " mode    m        real(V_L)        imag(V_L)",
        _rows([(int(row[0]), int(row[1]), row[2] * 2, row[3] * 2) for row in eigenvector]),
    ]
    path = path / f"gpec_response_n{n}.out"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {
        "modes": modes,
        "m": m,
        "energy": energy,
        "indices": indices,
        "eigenvector": eigenvector,
        "n": n,
    }


def write_empty_singfld(path, *, n=1):
    """A run with no rational surfaces: a column header and no rows under it."""
    lines = [
        " GPEC_SINGFLD: Resonant fields and islands from coils",
        f" {VERSION}",
        "",
        "    jac_out = hamada     tmag_out = 1",
        " sweet-spot = 0",
        "      msing =    0",
        "",
        "      q              psi    real(singflx)    imag(singflx)",
    ]
    path = path / f"gpec_vsingfld_n{n}.out"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"columns": ("q", "psi", "real(singflx)", "imag(singflx)")}
