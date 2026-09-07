"""Synthetic GPEC ASCII outputs, shaped like the real ``gpec_*_n<mode>.out`` files.

Layouts follow the shipped examples (GPEC v1.5.5): a ``KIND: description``
line, a version line, file-level ``key = value`` scalars, then titled sections
whose blocks carry their own scalars, a column header and numeric rows.

The awkward details are deliberate.  Each is somewhere a reader can go wrong,
and each was found by parsing real output rather than imagined:

* ``gpec_response`` repeats the section title "Eigenvectors" for two sections
  with different columns;
* its glossary is written as ``key = value`` with a one-word value
  (``Lambda = Inductance``) -- indistinguishable in shape from a real scalar
  such as ``jac_type = hamada``;
* ``gpec_control`` writes scalars whose *key* contains a space
  (``vacuum energy =  3.28697546E+000``);
* ``gpec_singfld`` puts an untitled block before a titled one, and its overlap
  block repeats the column name ``overlap(%)`` once per coupling matrix;
* ``gpec_recon_*`` has no version line and puts Fortran ``c`` comments between
  a column header and its rows;
* numbers are Fortran-formatted with three-digit exponents, and a denormal can
  lose its ``E`` entirely (``5.84973725-321`` appears in a real run);
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

#: The glossary lines GPEC prints under a section title.  The first three have
#: one-word values, so they are shaped exactly like a scalar and must not be
#: read as one; ``jac_type`` is a genuine scalar in the same shape, which is
#: why the parser treats the whole family as prose.
RESPONSE_GLOSSARY = (
    "  jac_type = hamada          ",
    "   L = Vacuum Inductance",
    "   Lambda = Inductance",
    "   rho = Reluctance",
    "   P = Permeability   *Complex (not Hermitian)",
)


def _fortran(value: float) -> str:
    """``1.234E-001``: Fortran's three-digit exponent, as the real files write it."""
    text = f"{value: .8E}"
    mantissa, _, exponent = text.partition("E")
    sign, digits = exponent[0], exponent[1:]
    return f"{mantissa}E{sign}{int(digits):03d}"


def _rows(values) -> str:
    return "\n".join(
        "  " + "".join(f"{v:5d}" if isinstance(v, (int, np.integer)) else f"{_fortran(v):>18s}" for v in row)
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
                f"  q = {q:.3f}  psi = {_fortran(psi)}",
                "",
                f"    m        real({symbol})        imag({symbol})",
                _rows([(int(mm), float(v.real), float(v.imag)) for mm, v in zip(m, values)]),
            ]
    path = path / f"gpec_singcoup_matrix_n{n}.out"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"m": m, "blocks": blocks, "rational_q": tuple(rational_q), "n": n}


def write_singcoup_svd(path, *, n=1, m_range=(-1, 1), mode_count=2):
    """``gpec_singcoup_svd_n<mode>.out``: free-text notes above the version line."""
    m = np.arange(m_range[0], m_range[1] + 1)
    notes = (
        "Note the right singular vectors have an additional half-area weighting,",
        "such that their inner product with b_x is the overlap field.",
    )
    lines = [" GPEC_SINGCOUP_SVD: SVD analysis for coupling matrices", *(f" {note}" for note in notes), f" {VERSION}", ""]
    lines += ["    jac_out = boozer     tmag_out = 1", f"      msing =   {mode_count}      mpert = {m.size}", ""]
    lines += [" Right singular vectors of the coupling matrix to effective resonant fields"]
    for mode in range(1, mode_count + 1):
        lines += [
            "",
            f" mode =   {mode}    s =  {_fortran(4.5 / mode)}",
            "",
            "    m        real(v)        imag(v)",
            _rows([(int(mm), float(mm) * 1e-3 * mode, float(mm) * 1e-4) for mm in m]),
        ]
    path = path / f"gpec_singcoup_svd_n{n}.out"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"notes": notes, "m": m, "mode_count": mode_count}


def write_response(path, *, n=1, mode_count=3, m_range=(-1, 1), denormal=True):
    """Write a miniature ``gpec_response_n<mode>.out``; returns its data."""
    modes = np.arange(1, mode_count + 1)
    m = np.arange(m_range[0], m_range[1] + 1)
    energy = np.stack([modes * 0.1, modes * 0.2, modes * 0.3], axis=1)
    eigenvector = np.array(
        [[float(mode), float(mm), mode * 1e-3, mm * 1e-4] for mode in modes for mm in m]
    )
    eigenvalue_rows = []
    for mode in modes:
        e_l, e_rho = float(mode) * 1e-6, float(mode) * 1e4
        row = f"  {int(mode):5d}{_fortran(e_l):>18s}{_fortran(e_rho):>18s}"
        if denormal and mode == modes[-1]:
            # A real run wrote a denormal with the exponent's E dropped.
            row = f"  {int(mode):5d}{_fortran(e_l):>18s}{'5.84973725-321':>18s}"
        eigenvalue_rows.append(row)

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
        " Eigenvalues (e) and Singular Values (s)",
        *RESPONSE_GLOSSARY,
        "",
        " mode              e_L         e_rho",
        "\n".join(eigenvalue_rows),
        "",
        # Two sections with the same title and different columns.
        " Eigenvectors",
        "",
        " mode    m      real(K_x^L)      imag(K_x^L)",
        _rows([(int(row[0]), int(row[1]), row[2], row[3]) for row in eigenvector]),
        "",
        " Eigenvectors",
        "",
        # An unpaired real() column: no imag() partner, so it is not complex.
        " mode    m        real(V_L)          norm(V)",
        _rows([(int(row[0]), int(row[1]), row[2] * 2, abs(row[3])) for row in eigenvector]),
    ]
    path = path / f"gpec_response_n{n}.out"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {
        "modes": modes,
        "m": m,
        "energy": energy,
        "eigenvector": eigenvector,
        "denormal": 5.84973725e-321 if denormal else None,
        "glossary": RESPONSE_GLOSSARY,
        "n": n,
    }


def write_control(path, *, n=1, m_range=(-1, 1)):
    """``gpec_control_n<mode>.out``: scalars whose key contains a space."""
    m = np.arange(m_range[0], m_range[1] + 1)
    scalars = {
        "vacuum energy": 3.28697546,
        "surface energy": 7.94775984,
        "plasma energy": 3.76560559,
        "toroidal torque": 1.17037378e-4,
    }
    lines = [
        " GPEC_CONTROL: Plasma response for an external perturbation on the control surface",
        f" {VERSION}",
        "",
        f"      mpert =  {m.size}",
        *(f"  {key} = {_fortran(value)}" for key, value in scalars.items()),
        "",
        " jac_type = hamada          ",
        "",
        "    m        real(bin)        imag(bin)",
        _rows([(int(mm), float(mm) * 1e-7, float(mm) * 1e-6) for mm in m]),
    ]
    path = path / f"gpec_control_n{n}.out"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"m": m, "scalars": scalars}


def write_singfld(path, *, n=1, rational_q=(2.0, 3.0), overlap_matrices=3):
    """``gpec_singfld_n<mode>.out``: an untitled block, then a titled one whose
    header repeats ``overlap(%)`` once per coupling matrix."""
    resonant = [
        (q, 0.5 + 0.1 * index, 3.0e-4 * (index + 1), -1.1e-4 * (index + 1))
        for index, q in enumerate(rational_q)
    ]
    overlap_columns = "   mode" + "".join(
        f"{f'real(ov{k})':>17s}{f'imag(ov{k})':>17s}{'overlap(%)':>17s}" for k in range(overlap_matrices)
    )
    overlap_rows = [
        [float(mode)] + [v for k in range(overlap_matrices) for v in (mode * 1e-4, mode * 1e-5, mode * 10.0 + k)]
        for mode in range(1, len(rational_q) + 1)
    ]
    lines = [
        " GPEC_SINGFLD: Resonant fields, singular currents, and islands",
        f" {VERSION}",
        "",
        "    jac_out = boozer     tmag_out = 1",
        " sweet-spot =  5.00000000E-004",
        f"      msing =    {len(rational_q)}",
        "",
        "      q              psi    real(singflx)    imag(singflx)",
        _rows(resonant),
        "",
        " Overlap fields, overlap singular currents, and overlap islands",
        "",
        overlap_columns,
        _rows([[int(row[0])] + list(row[1:]) for row in overlap_rows]),
    ]
    path = path / f"gpec_singfld_n{n}.out"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {
        "resonant": np.asarray(resonant),
        "overlap": np.asarray(overlap_rows),
        "overlap_matrices": overlap_matrices,
    }


def write_recon(path, *, rows=3):
    """``gpec_recon_*_sol1.out``: no version line, Fortran ``c`` comments
    between the column header and its rows, and trailing summary rows that
    belong to no block."""
    comments = (
        "c  psifac-grid integration results",
        "c  C2/mu0 = int J |C|^2 / mu0",
        "c  recon_int = spline",
    )
    lines = [
        "              psi           c2_mu0        k_xin2_re",
        *comments,
        *[
            "  " + "".join(f"{_fortran(value):>18s}" for value in (1e-4 * (k + 1), 1.6e-4, 3e-10))
            for k in range(rows)
        ],
        "c  Final C2 component totals:",
        # Two values where the block has three columns: belongs to no block.
        f"  {_fortran(3.2e-1)}{_fortran(4.2e-1)}",
    ]
    path = path / "gpec_recon_integration_sol1.out"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"comments": comments, "rows": rows}


def write_empty_singfld(path, *, n=1):
    """A run with no rational surfaces: a column header and no rows under it."""
    columns = ("q", "psi", "real(singflx)", "imag(singflx)", "islandhwidth", "chirikov")
    lines = [
        " GPEC_SINGFLD: Resonant fields and islands from coils",
        f" {VERSION}",
        "",
        "    jac_out = hamada     tmag_out = 1",
        " sweet-spot = 0",
        "      msing =    0",
        "",
        "      q              psi    real(singflx)    imag(singflx)     islandhwidth         chirikov",
    ]
    path = path / f"gpec_vsingfld_n{n}.out"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"columns": columns}
