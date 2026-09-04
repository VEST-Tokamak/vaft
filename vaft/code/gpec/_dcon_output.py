"""DCON's own native output: a typed container over ``dcon_output_n<mode>.nc``.

DCON solves the ideal-MHD marginal-stability (delta-W) problem for a single
toroidal mode number. Several of its core outputs have no home in the IMAS
``mhd_linear`` IDS at all (the plasma/vacuum/total energy decomposition,
Mercier diagnostics, mode-range provenance), and the Fourier-space
eigenfunction reconstructed from ``solutions.bin`` has only a closest-fit one
-- so :class:`DconOutput` is where all of it lives losslessly and at full
radial resolution, while ``vaft.machine_mapping.mhd_linear`` writes a strided
view of the eigenfunction into the IDS with its unit and normalization
mismatches recorded explicitly (issue #170).

This is DCON's own schema, not a cross-solver abstraction: RDCON/STRIDE's
native output (:mod:`vaft.code.gpec._matching_output`) has a genuinely
different netCDF schema and is deliberately not folded in here.

Every field here is verified against the DCON Fortran source in GPEC
(``dcon/dcon.f``, ``dcon/dcon_netcdf.f``, ``dcon/free.f``,
``dcon/ode_output.f``, ``match/ideal.f``); see the module-level notes on
:class:`DconEigenfunction` for the one quantity (``v4``) whose physical
identity is not fully documented in the source and is kept only as a raw,
labeled value rather than interpreted.

``mlow``/``mhigh``/``mpert``/``mband`` are read from
``dcon_output_n<mode>.nc``'s global attributes, not by parsing DCON's ASCII
log: ``dcon/dcon_netcdf.f`` writes them (``nf90_put_att(ncid,nf90_global,
'mlow',mlow)`` etc.) directly onto the same netCDF file VAFT already reads
for ``W_t_eigenvalue``, which is both simpler and more robust than scraping
``dcon.out``'s free-format numeric row.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
import struct
from typing import Any, Optional, Sequence
import warnings

import numpy as np

from ._netcdf import complex_var

#: Sidecar payload version.  v1 carried only the mode-range provenance, energy
#: eigenvalues, Mercier diagnostics and the eigenfunction; v2 adds the
#: equilibrium scalars, coordinate system, 1-D profiles and the edge scan.
#: Every v2 key is read through ``.get()``, so a v1 payload still loads.
_SCHEMA_VERSION = 2


def _opt_float(attrs: Any, key: str) -> Optional[float]:
    """A netCDF global attribute as a plain ``float``, or ``None`` when absent.

    The coercion is not cosmetic: xarray surfaces netCDF globals as
    ``np.float64``/``np.int32``/``np.str_``, none of which ``json.dumps`` can
    serialize, so an uncoerced value only fails once a real file reaches
    :meth:`DconOutput.write_json`.
    """
    if key not in attrs:
        return None
    try:
        return float(np.asarray(attrs[key]).reshape(-1)[0])
    except (TypeError, ValueError, IndexError):
        return None


def _opt_int(attrs: Any, key: str) -> Optional[int]:
    value = _opt_float(attrs, key)
    return None if value is None else int(value)


def _opt_str(attrs: Any, key: str) -> str:
    return "" if key not in attrs else str(attrs[key]).strip()


def _opt_array(ds: Any, name: str, dtype=float) -> Optional[np.ndarray]:
    return None if name not in ds.variables else np.asarray(ds[name].values, dtype=dtype)


def _read_fortran_record_length(stream) -> Optional[int]:
    """Read one Fortran unformatted record-length marker, or ``None`` at EOF.

    A well-formed marker is a non-negative count of bytes holding ``REAL*4``
    values, so it must be a multiple of 4. Anything else means the stream is
    not (or is no longer) a ``solutions.bin`` written by ``match``: reading on
    from a bogus length would desynchronise the parser and yield silently
    garbage harmonics, so this raises instead.
    """
    raw = stream.read(4)
    if len(raw) < 4:
        return None
    length = struct.unpack("<i", raw)[0]
    if length < 0 or length % 4:
        raise ValueError(
            f"malformed Fortran record-length marker {length!r} at byte "
            f"{stream.tell() - 4} (expected a non-negative multiple of 4)"
        )
    return length


def _read_n_floats(stream, n: int) -> np.ndarray:
    raw = stream.read(n * 4)
    if len(raw) < n * 4:
        raise EOFError("Unexpected EOF while reading float data.")
    return np.frombuffer(raw, dtype="<f4")


def _read_solutions_bin_blocks(path: Path) -> list[list[np.ndarray]]:
    """Parse ``solutions.bin``'s Fortran unformatted records into per-``ipert`` blocks.

    Layout (``match/ideal.f:378-390``): outer loop over poloidal-harmonic
    blocks (``ipert=1,mpert``), inner loop over radial steps
    (``istep=0,mstep``), each record 7 ``REAL*4`` values -- ``psi, rho, q,
    Re(xi.grad(psi)), Im(xi.grad(psi)), Re(v4), Im(v4)`` -- with a blank
    (zero-length) record separating blocks.
    """
    blocks: list[list[np.ndarray]] = []
    with open(path, "rb") as stream:
        while True:
            length = _read_fortran_record_length(stream)
            if length is None:
                break
            if length == 0:
                continue
            vec = _read_n_floats(stream, length // 4)
            _read_fortran_record_length(stream)  # trailing record-length marker

            steps = [vec]
            while True:
                length2 = _read_fortran_record_length(stream)
                if length2 is None or length2 == 0:
                    break
                vec2 = _read_n_floats(stream, length2 // 4)
                _read_fortran_record_length(stream)
                steps.append(vec2)
            blocks.append(steps)
    return blocks


@dataclass
class DconEigenfunction:
    """Per-poloidal-harmonic normal-displacement eigenfunction from ``solutions.bin``.

    ``m`` is the true physical poloidal mode number for each block (``mlow +
    ipert``, 0-based ``ipert``), never a raw array index. ``xi_psi_real`` /
    ``xi_psi_imag`` are ``Re/Im(xi . grad(psi))`` -- confirmed both from
    ``match/ideal.f``'s ``b(:,istep)=ifac*singfac*v(:,1,istep)`` (the standard
    ``b_psi = i(m-nq).xi_psi`` relation) and independently from a
    physicist-authored reference reader that labels the same column
    ``r"Re$(\\xi \\cdot \\nabla \\psi)$"``.

    ``v4_real``/``v4_imag`` is the second Euler-Lagrange dependent variable
    (``v(:,4,istep)``, sourced from ``ud`` in the underlying ODE solution,
    i.e. plausibly ``d(pi)/dpsi``). Its physical identity was *not* found
    documented anywhere in the DCON/match source, so it is kept as a raw,
    labeled value rather than interpreted -- do not assume a meaning for it
    without independent confirmation.

    Rows are padded with NaN out to the longest block's step count, since
    integration step counts can differ slightly harmonic to harmonic.
    """

    m: np.ndarray
    psi: np.ndarray
    rho: np.ndarray
    q: np.ndarray
    xi_psi_real: np.ndarray
    xi_psi_imag: np.ndarray
    v4_real: np.ndarray
    v4_imag: np.ndarray

    @property
    def amplitude(self) -> np.ndarray:
        """``|xi . grad(psi)|`` per (harmonic, step), preserving the NaN padding."""
        return np.hypot(self.xi_psi_real, self.xi_psi_imag)

    def b_normal(self, n_tor: int) -> np.ndarray:
        """Normal perturbed field per (harmonic, step), complex.

        ``b = i (m - n q) xi.grad(psi)`` -- exactly ``match/ideal.f:372``'s
        ``b(:,istep)=ifac*singfac*v(:,1,istep)`` with ``singfac = m - n*q``,
        recomputed here because ``match`` forms it internally and never writes
        it out.  Everything it needs is already in ``solutions.bin``: ``q`` is
        the third column and ``m`` comes from the run's own ``mlow``.

        Carries the same arbitrary eigenvector normalization as ``xi``, so only
        its shape and relative harmonic content are meaningful.
        """
        xi = self.xi_psi_real + 1j * self.xi_psi_imag
        singular_factor = self.m[:, np.newaxis] - int(n_tor) * self.q
        return 1j * singular_factor * xi

    def to_dict(self) -> dict[str, Any]:
        return {
            "m": self.m.tolist(),
            "psi": self.psi.tolist(),
            "rho": self.rho.tolist(),
            "q": self.q.tolist(),
            "xi_psi_real": self.xi_psi_real.tolist(),
            "xi_psi_imag": self.xi_psi_imag.tolist(),
            "v4_real": self.v4_real.tolist(),
            "v4_imag": self.v4_imag.tolist(),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "DconEigenfunction":
        return cls(
            m=np.asarray(payload["m"], dtype=int),
            psi=np.asarray(payload["psi"], dtype=float),
            rho=np.asarray(payload["rho"], dtype=float),
            q=np.asarray(payload["q"], dtype=float),
            xi_psi_real=np.asarray(payload["xi_psi_real"], dtype=float),
            xi_psi_imag=np.asarray(payload["xi_psi_imag"], dtype=float),
            v4_real=np.asarray(payload["v4_real"], dtype=float),
            v4_imag=np.asarray(payload["v4_imag"], dtype=float),
        )


def read_solutions_bin(path: str | Path, *, mlow: int) -> DconEigenfunction:
    """Parse ``solutions.bin`` and label its blocks by true poloidal mode number.

    ``mlow`` must come from the companion run's ``dcon_output_n<mode>.nc``
    global attribute (see :func:`read_dcon_output`), not be guessed -- DCON
    computes it at runtime from the equilibrium (``mlow=MIN(nn*qmin,zero)-4-
    delta_mlow``, ``dcon/dcon.f:195``), so it is not a static default.
    """
    blocks = _read_solutions_bin_blocks(Path(path))
    n_ipert = len(blocks)
    if n_ipert == 0:
        empty = np.zeros((0, 0), dtype=np.float32)
        return DconEigenfunction(
            m=np.zeros(0, dtype=int),
            psi=empty, rho=empty, q=empty,
            xi_psi_real=empty, xi_psi_imag=empty, v4_real=empty, v4_imag=empty,
        )

    max_steps = max(len(steps) for steps in blocks)
    arr = np.full((n_ipert, max_steps, 7), np.nan, dtype=np.float32)
    for i_block, steps in enumerate(blocks):
        for j_step, vec7 in enumerate(steps):
            arr[i_block, j_step, : vec7.size] = vec7

    m = mlow + np.arange(n_ipert, dtype=int)
    return DconEigenfunction(
        m=m,
        psi=arr[:, :, 0],
        rho=arr[:, :, 1],
        q=arr[:, :, 2],
        xi_psi_real=arr[:, :, 3],
        xi_psi_imag=arr[:, :, 4],
        v4_real=arr[:, :, 5],
        v4_imag=arr[:, :, 6],
    )


@dataclass
class DconEquilibrium:
    """Equilibrium scalars DCON computed from the g-file, from the netCDF globals.

    Every field here is n-independent -- a property of the equilibrium DCON was
    given, not of the mode it solved for.  Definitions and units are read off
    ``equil/equil_out.f`` rather than assumed, because several differ from the
    same-named quantity elsewhere in VAFT: ``crnt`` is in MA, ``betan`` is a
    percentage, and there are three mutually inconsistent ``betap``/``li``
    normalizations that only their subscript distinguishes.

    ``None`` for any attribute a given file does not carry, so a run written by
    a different GPEC version parses rather than raising.
    """

    # Shape, from the separatrix midplane intercepts (equil_out.f:261-266).
    amean: Optional[float] = None      # minor radius [m], (rsep(1)-rsep(2))/2
    rmean: Optional[float] = None      # major radius [m], (rsep(1)+rsep(2))/2
    aratio: Optional[float] = None     # rmean/amean [-]
    kappa: Optional[float] = None      # elongation [-]
    delta1: Optional[float] = None     # (rmean-rext(1))/amean [-]
    delta2: Optional[float] = None     # (rmean-rext(2))/amean [-]
    # Axis and flux normalization.
    ro: Optional[float] = None         # magnetic axis R [m]
    zo: Optional[float] = None         # magnetic axis Z [m]
    psio: Optional[float] = None       # poloidal flux normalization
    psilow: Optional[float] = None     # inner integration boundary, psi_n [-]
    # Safety factor (equil_out.f:443-449).
    q0: Optional[float] = None         # sq%fs(0,4)-sq%fs1(0,4)*sq%xs(0), extrapolated to the axis
    qmin: Optional[float] = None
    qmax: Optional[float] = None
    qa: Optional[float] = None         # edge q, extrapolated to psi_n=1
    q95: Optional[float] = None
    # Current and field.
    crnt: Optional[float] = None       # plasma current [MA] -- equil_out.f:299 divides by 1e6*mu0
    bt0: Optional[float] = None        # vacuum toroidal field at rmean [T], equil_out.f:268
    # Beta and inductance.  The suffixes are different normalizations of the
    # same physical quantity, not different quantities (equil_out.f:363-368).
    betat: Optional[float] = None      # toroidal beta [-], equil_out.f:360
    betan: Optional[float] = None      # normalized beta [% m T MA^-1], equil_out.f:362
    betap1: Optional[float] = None     # bp0-normalized
    betap2: Optional[float] = None     # ro-normalized
    betap3: Optional[float] = None     # rmean-normalized
    li1: Optional[float] = None        # bp0-normalized
    li2: Optional[float] = None        # ro-normalized
    li3: Optional[float] = None        # rmean-normalized

    _FIELDS = (
        "amean", "rmean", "aratio", "kappa", "delta1", "delta2",
        "ro", "zo", "psio", "psilow",
        "q0", "qmin", "qmax", "qa", "q95",
        "crnt", "bt0",
        "betat", "betan", "betap1", "betap2", "betap3", "li1", "li2", "li3",
    )

    @classmethod
    def from_attrs(cls, attrs: Any) -> Optional["DconEquilibrium"]:
        """Build from a dataset's global attributes, or ``None`` if it carries none of them."""
        values = {name: _opt_float(attrs, name) for name in cls._FIELDS}
        if all(value is None for value in values.values()):
            return None
        return cls(**values)

    def to_dict(self) -> dict[str, Any]:
        return {name: getattr(self, name) for name in self._FIELDS}

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "DconEquilibrium":
        return cls(**{name: payload.get(name) for name in cls._FIELDS})


@dataclass
class DconCoordinates:
    """The flux coordinate system and grid DCON mapped the equilibrium onto.

    Kept apart from :class:`DconEquilibrium` because it describes the *solver's*
    working coordinates rather than the plasma.  ``jacobian`` in particular is
    load-bearing downstream: it is what decides which IMAS ``grid_type`` (if
    any) DCON's Fourier-space output can honestly declare, and it is read off
    the output file here rather than inferred from the input template, so a run
    prepared with a non-default ``equil.in`` reports its own truth.
    """

    jacobian: str = ""                 # jac_type, e.g. "hamada" (dcon_netcdf.f:93)
    power_bp: Optional[float] = None
    power_b: Optional[float] = None
    power_r: Optional[float] = None
    mpsi: Optional[int] = None         # radial grid intervals
    mtheta: Optional[int] = None       # poloidal grid intervals

    @classmethod
    def from_attrs(cls, attrs: Any) -> Optional["DconCoordinates"]:
        jacobian = _opt_str(attrs, "jacobian")
        values = {
            "power_bp": _opt_float(attrs, "power_bp"),
            "power_b": _opt_float(attrs, "power_b"),
            "power_r": _opt_float(attrs, "power_r"),
            "mpsi": _opt_int(attrs, "mpsi"),
            "mtheta": _opt_int(attrs, "mtheta"),
        }
        if not jacobian and all(value is None for value in values.values()):
            return None
        return cls(jacobian=jacobian, **values)

    def to_dict(self) -> dict[str, Any]:
        return {
            "jacobian": self.jacobian,
            "power_bp": self.power_bp,
            "power_b": self.power_b,
            "power_r": self.power_r,
            "mpsi": self.mpsi,
            "mtheta": self.mtheta,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "DconCoordinates":
        return cls(
            jacobian=str(payload.get("jacobian", "")),
            power_bp=payload.get("power_bp"),
            power_b=payload.get("power_b"),
            power_r=payload.get("power_r"),
            mpsi=payload.get("mpsi"),
            mtheta=payload.get("mtheta"),
        )


@dataclass
class DconEdgeScan:
    """DCON's edge scan: the least-stable total energy against edge safety factor.

    Written only when ``size_edge > 0`` (``dcon/dcon_netcdf.f:149,202-211``),
    which DCON sets from ``psiedge < psilim`` (``dcon/sing.f:224``).  VAFT's
    packaged ``dcon.in`` ships ``psiedge=1`` against ``psihigh=0.994``, so this
    is **absent by default** and appears only when a caller lowers
    ``DCONOptions.psiedge`` below ``psihigh``.

    It is DCON's own per-run stability-limit curve: ``dW`` is
    ``long_name="Least Stable Total Energy Eigenvalues"`` sampled against a
    scan of edge q, so a sign change along it locates the truncation at which
    the equilibrium becomes unstable.
    """

    psi_n: np.ndarray                  # (n_edge,)
    q: np.ndarray                      # (n_edge,)
    dW: np.ndarray                     # (n_edge,) complex

    @classmethod
    def from_dataset(cls, ds: Any) -> Optional["DconEdgeScan"]:
        psi_n = _opt_array(ds, "psi_n_edge")
        q = _opt_array(ds, "q_edge")
        dW = complex_var(ds, "dW_edge")
        if psi_n is None or q is None or dW is None:
            return None
        return cls(psi_n=psi_n, q=q, dW=dW)

    def to_dict(self) -> dict[str, Any]:
        return {
            "psi_n": self.psi_n.tolist(),
            "q": self.q.tolist(),
            "dW": {"real": np.real(self.dW).tolist(), "imag": np.imag(self.dW).tolist()},
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "DconEdgeScan":
        return cls(
            psi_n=np.asarray(payload["psi_n"], dtype=float),
            q=np.asarray(payload["q"], dtype=float),
            dW=(
                np.asarray(payload["dW"]["real"], dtype=float)
                + 1j * np.asarray(payload["dW"]["imag"], dtype=float)
            ),
        )


@dataclass
class DconOutput:
    """Typed container over one DCON run's native output (one ``n_tor``).

    Field names mirror ``dcon_output_n<mode>.nc``'s own variable/attribute
    names directly (``W_t_eigenvalue``, ``di``, ``dr``, ``ca1``, ``mlow``,
    ``mhigh``, ``mpert``, ``mband``) rather than inventing new terminology.
    """

    n_tor: int
    mlow: int
    mhigh: int
    mpert: int
    mband: int
    psi_n: Optional[np.ndarray] = None
    m: Optional[np.ndarray] = None
    #: The netCDF ``mode`` coordinate (``1..mpert``); eigenvalue entries are
    #: addressed by this *label*, never by array position -- see
    #: :meth:`_least_stable`.
    mode: Optional[np.ndarray] = None
    W_p_eigenvalue: Optional[np.ndarray] = None  # complex, (mode,)
    W_v_eigenvalue: Optional[np.ndarray] = None
    W_t_eigenvalue: Optional[np.ndarray] = None
    di: Optional[np.ndarray] = None
    dr: Optional[np.ndarray] = None
    ca1: Optional[np.ndarray] = None
    #: DCON's integration limits for *this* toroidal mode.  Deliberately not on
    #: :class:`DconEquilibrium`: under ``sas_flag`` the edge limit is rounded onto
    #: the mode's own rational spacing (``qlim=(INT(nn*qlim)+dmlim)/nn``,
    #: ``dcon/sing.f:186``) after being capped at the ``qhigh`` that VAFT itself
    #: templates into ``dcon.in``.  It is a solver truncation, not a property of
    #: the equilibrium -- never read it as ``qa``.
    qlim: Optional[float] = None
    psilim: Optional[float] = None
    #: 1-D equilibrium profiles on ``psi_n`` (``dcon/dcon_netcdf.f:165-171``).
    f: Optional[np.ndarray] = None          # F = R*B_tor
    mu0p: Optional[np.ndarray] = None       # mu0 * pressure
    dvdpsi: Optional[np.ndarray] = None
    q: Optional[np.ndarray] = None          # long_name "Safety Factor"
    equilibrium: Optional[DconEquilibrium] = None
    coordinates: Optional[DconCoordinates] = None
    edge_scan: Optional[DconEdgeScan] = None
    eigenfunction: Optional[DconEigenfunction] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    #: DCON sorts its energy eigenvalues so the least-stable one is mode 1
    #: (``dcon/free.f``'s ``plasma1``/``vacuum1``/``total1`` are the
    #: ``ep(1)``/``ev(1)``/``et(1)`` entries).
    _LEAST_STABLE_MODE_LABEL = 1

    def _least_stable(self, eigenvalues: Optional[np.ndarray]) -> Optional[complex]:
        """The least-stable entry of an eigenvalue array, selected by mode *label*.

        The netCDF writes a ``mode`` coordinate of ``1..mpert``
        (``dcon/dcon_netcdf.f``'s ``nf90_put_var(ncid,mo_id,(/(i,i=1,mpert)/))``),
        and the value we want is the one labelled ``1``. Position 0 is the
        same entry for every file the suite writes today, but selecting by
        label keeps that an explicit, checkable assumption rather than a
        silent one -- a differently-ordered ``mode`` coordinate would
        otherwise change which eigenvalue is reported with no visible signal.
        """
        if eigenvalues is None or eigenvalues.size == 0:
            return None
        if self.mode is None:
            return complex(eigenvalues[0])
        matches = np.flatnonzero(np.asarray(self.mode) == self._LEAST_STABLE_MODE_LABEL)
        if matches.size == 0:
            return None
        return complex(eigenvalues[int(matches[0])])

    @property
    def plasma1(self) -> Optional[complex]:
        """Least-stable plasma-response energy eigenvalue (``dcon/free.f``'s ``plasma1``)."""
        return self._least_stable(self.W_p_eigenvalue)

    @property
    def vacuum1(self) -> Optional[complex]:
        return self._least_stable(self.W_v_eigenvalue)

    @property
    def total1(self) -> Optional[complex]:
        """Least-stable total-energy eigenvalue -- ``dcon/free.f``'s ``total1``, the
        ``W_t_eigenvalue`` entry labelled by mode 1."""
        return self._least_stable(self.W_t_eigenvalue)

    @property
    def m_pol_dominant(self) -> Optional[int]:
        """The poloidal mode number carrying the largest peak ``|xi . grad(psi)|``.

        ``None`` without an eigenfunction (no ``solutions.bin``, i.e. no
        ``match``), and ``None`` if every harmonic is entirely NaN.

        Reportable despite the eigenfunction's arbitrary normalization: that
        normalization (``match/ideal.f:318-325``, plus DCON's ``ucrit``
        re-scaling during integration) is a single global factor multiplying
        every harmonic alike, so which harmonic is largest does not depend on
        it -- unlike the amplitudes themselves, which do.
        """
        if self.eigenfunction is None or self.eigenfunction.m.size == 0:
            return None
        amplitude = self.eigenfunction.amplitude
        usable = np.isfinite(amplitude)
        if not usable.any():
            return None
        peaks = np.where(usable, amplitude, -np.inf).max(axis=1)
        return int(self.eigenfunction.m[int(np.argmax(peaks))])

    @property
    def stable_free_boundary(self) -> Optional[bool]:
        """Free-boundary stability from ``sign(Re(total1))`` (``dcon/dcon.f:306-314``).

        ``None`` when no total-energy eigenvalue was computed (``vac_flag=false``
        runs). This does not cover fixed-boundary stability (``nzero``), which
        DCON only reports on stdout, not in any file VAFT retains today.
        """
        t1 = self.total1
        return None if t1 is None else bool(t1.real >= 0)

    def to_dict(self) -> dict[str, Any]:
        def _c(arr: Optional[np.ndarray]) -> Optional[dict[str, list[float]]]:
            if arr is None:
                return None
            arr = np.asarray(arr)
            return {"real": np.real(arr).tolist(), "imag": np.imag(arr).tolist()}

        def _r(arr: Optional[np.ndarray]) -> Optional[list[float]]:
            return None if arr is None else np.asarray(arr).tolist()

        return {
            "schema": "vaft.code.gpec.DconOutput",
            "schema_version": _SCHEMA_VERSION,
            "n_tor": self.n_tor,
            "mlow": self.mlow,
            "mhigh": self.mhigh,
            "mpert": self.mpert,
            "mband": self.mband,
            "psi_n": None if self.psi_n is None else np.asarray(self.psi_n).tolist(),
            "m": None if self.m is None else np.asarray(self.m).tolist(),
            "mode": None if self.mode is None else np.asarray(self.mode).tolist(),
            "W_p_eigenvalue": _c(self.W_p_eigenvalue),
            "W_v_eigenvalue": _c(self.W_v_eigenvalue),
            "W_t_eigenvalue": _c(self.W_t_eigenvalue),
            "di": None if self.di is None else np.asarray(self.di).tolist(),
            "dr": None if self.dr is None else np.asarray(self.dr).tolist(),
            "ca1": None if self.ca1 is None else np.asarray(self.ca1).tolist(),
            "qlim": self.qlim,
            "psilim": self.psilim,
            "f": _r(self.f),
            "mu0p": _r(self.mu0p),
            "dvdpsi": _r(self.dvdpsi),
            "q": _r(self.q),
            "equilibrium": None if self.equilibrium is None else self.equilibrium.to_dict(),
            "coordinates": None if self.coordinates is None else self.coordinates.to_dict(),
            "edge_scan": None if self.edge_scan is None else self.edge_scan.to_dict(),
            "eigenfunction": None if self.eigenfunction is None else self.eigenfunction.to_dict(),
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "DconOutput":
        def _c(block: Optional[dict[str, list[float]]]) -> Optional[np.ndarray]:
            if block is None:
                return None
            return np.asarray(block["real"], dtype=float) + 1j * np.asarray(block["imag"], dtype=float)

        def _r(values: Optional[list[float]]) -> Optional[np.ndarray]:
            return None if values is None else np.asarray(values, dtype=float)

        # A payload written by a *newer* VAFT may carry fields whose meaning this
        # code does not know; silently dropping them would turn a version skew
        # into a quiet data loss on the next round trip.
        version = int(payload.get("schema_version", 1))
        if version > _SCHEMA_VERSION:
            raise ValueError(
                f"DconOutput sidecar schema_version {version} is newer than this "
                f"VAFT understands ({_SCHEMA_VERSION}); upgrade VAFT to read it"
            )

        return cls(
            n_tor=payload["n_tor"],
            mlow=payload["mlow"],
            mhigh=payload["mhigh"],
            mpert=payload["mpert"],
            mband=payload["mband"],
            psi_n=None if payload.get("psi_n") is None else np.asarray(payload["psi_n"], dtype=float),
            m=None if payload.get("m") is None else np.asarray(payload["m"], dtype=int),
            mode=None if payload.get("mode") is None else np.asarray(payload["mode"], dtype=int),
            W_p_eigenvalue=_c(payload.get("W_p_eigenvalue")),
            W_v_eigenvalue=_c(payload.get("W_v_eigenvalue")),
            W_t_eigenvalue=_c(payload.get("W_t_eigenvalue")),
            di=None if payload.get("di") is None else np.asarray(payload["di"], dtype=float),
            dr=None if payload.get("dr") is None else np.asarray(payload["dr"], dtype=float),
            ca1=None if payload.get("ca1") is None else np.asarray(payload["ca1"], dtype=float),
            qlim=payload.get("qlim"),
            psilim=payload.get("psilim"),
            f=_r(payload.get("f")),
            mu0p=_r(payload.get("mu0p")),
            dvdpsi=_r(payload.get("dvdpsi")),
            q=_r(payload.get("q")),
            equilibrium=(
                None if payload.get("equilibrium") is None
                else DconEquilibrium.from_dict(payload["equilibrium"])
            ),
            coordinates=(
                None if payload.get("coordinates") is None
                else DconCoordinates.from_dict(payload["coordinates"])
            ),
            edge_scan=(
                None if payload.get("edge_scan") is None
                else DconEdgeScan.from_dict(payload["edge_scan"])
            ),
            eigenfunction=(
                None if payload.get("eigenfunction") is None else DconEigenfunction.from_dict(payload["eigenfunction"])
            ),
            metadata=payload.get("metadata", {}),
        )

    def write_json(self, path: str | Path) -> Path:
        target = Path(path).expanduser()
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return target

    @classmethod
    def read_json(cls, path: str | Path) -> "DconOutput":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


#: One flat row per DCON run, for a scan across time slices or shots.
#:
#: Deliberately a fixed, ordered tuple rather than "whatever the run happened to
#: carry": a scan is a table, and a table whose columns vary per row cannot be
#: compared across shots.  Missing values are ``None``, never omitted.
SCAN_COLUMNS: tuple[str, ...] = (
    "shot",
    "time_ms",
    "n_tor",
    "q0",
    "qmin",
    "qmax",
    "qa",
    "q95",
    "qlim",
    "betan",
    "betat",
    "betap1",
    "betap3",
    "li1",
    "li3",
    "crnt",
    "bt0",
    "amean",
    "rmean",
    "aratio",
    "kappa",
    "delta1",
    "delta2",
    "total1_real",
    "stable_free_boundary",
)

#: Columns taken straight off :class:`DconEquilibrium`.
_SCAN_EQUILIBRIUM_COLUMNS = tuple(
    name for name in SCAN_COLUMNS if name in DconEquilibrium._FIELDS
)


def dcon_scan_row(
    result: DconOutput,
    *,
    shot: Optional[int] = None,
    time_ms: Optional[float] = None,
) -> dict[str, Any]:
    """Flatten one DCON run into a row keyed by :data:`SCAN_COLUMNS`.

    ``shot`` and ``time_ms`` are supplied by the caller rather than read from
    the file.  DCON writes them as ``INT(shotnum)``/``INT(shottime)``, which are
    both 0 for a GEQDSK whose header carries no shot -- so the run's own
    directory is the authority on its identity, and the file's values are kept
    in ``DconOutput.metadata`` for cross-checking instead.
    """
    row: dict[str, Any] = {name: None for name in SCAN_COLUMNS}
    row["shot"] = None if shot is None else int(shot)
    row["time_ms"] = None if time_ms is None else float(time_ms)
    row["n_tor"] = int(result.n_tor)
    # qlim rides on the run, not the equilibrium: it is the mode-dependent
    # truncation DCON integrated to, and a scan that confuses it with `qa`
    # compares different quantities across rows.
    row["qlim"] = result.qlim
    if result.equilibrium is not None:
        for name in _SCAN_EQUILIBRIUM_COLUMNS:
            row[name] = getattr(result.equilibrium, name)
    total1 = result.total1
    row["total1_real"] = None if total1 is None else float(total1.real)
    row["stable_free_boundary"] = result.stable_free_boundary
    return row


def read_dcon_scan(
    workdir: str | Path,
    *,
    modes: Sequence[int] = (1,),
    shot: Optional[int] = None,
) -> list[dict[str, Any]]:
    """Harvest one row per DCON run under a prepared case directory.

    Walks the ``<workdir>/<time>/<module>/nn=<mode>/`` layout
    :func:`vaft.code.gpec._runtime.module_dir` writes, so the scan reads exactly
    what the runner produced rather than a directory convention maintained
    separately from it.

    A run whose netCDF cannot be read is warned about and skipped: one bad cell
    in a scan of hundreds should cost that cell, not the scan -- but it must not
    cost it silently, because a quietly shorter table looks exactly like a
    shorter run list.

    DCON-only by construction, with no ``module`` parameter: every column in
    :data:`SCAN_COLUMNS` comes from :class:`DconOutput`, so pointing this at
    ``rdcon``/``stride`` could only ever find their files and then fail to read
    them as DCON's.
    """
    root = Path(workdir).expanduser()
    rows: list[dict[str, Any]] = []
    for time_dir in sorted(path for path in root.glob("*") if path.is_dir()):
        for mode in modes:
            run_dir = time_dir / "dcon" / f"nn={mode}"
            if not (run_dir / f"dcon_output_n{mode}.nc").exists():
                continue
            try:
                result = read_dcon_output(run_dir, mode=mode)
            except Exception as exc:  # noqa: BLE001 -- any unreadable file, same handling
                warnings.warn(
                    f"skipping unreadable DCON output in {run_dir}: {type(exc).__name__}: {exc}",
                    RuntimeWarning,
                    stacklevel=2,
                )
                continue
            rows.append(dcon_scan_row(result, shot=shot, time_ms=_time_ms(time_dir.name)))
    return rows


def _time_ms(label: str) -> Optional[float]:
    """The run directory's time label as milliseconds, or ``None`` if it is not one."""
    try:
        return float(label)
    except ValueError:
        return None


def read_dcon_output(run_dir: str | Path, *, mode: int) -> DconOutput:
    """Build a :class:`DconOutput` from one completed DCON run directory.

    Reads ``dcon_output_n<mode>.nc`` for mode-range provenance, the equilibrium
    scalars and coordinate system DCON derived from the g-file, energy
    eigenvalues, Mercier diagnostics, the 1-D profiles, and the edge scan when
    the run produced one; reads ``solutions.bin`` (if
    present -- it is written by the companion ``match`` tool, not DCON
    itself) for the Fourier-space eigenfunction. Never reads ``euler.bin``
    (out of scope: too large for routine conversion).
    """
    import xarray as xr

    run_dir = Path(run_dir)
    nc_path = run_dir / f"dcon_output_n{mode}.nc"
    with xr.open_dataset(nc_path) as ds:
        mlow = int(ds.attrs["mlow"])
        mhigh = int(ds.attrs["mhigh"])
        mpert = int(ds.attrs["mpert"])
        mband = int(ds.attrs["mband"])
        n_tor = int(ds.attrs.get("n", mode))

        psi_n = np.asarray(ds["psi_n"].values, dtype=float) if "psi_n" in ds.variables else None
        m = np.asarray(ds["m"].values, dtype=int) if "m" in ds.variables else None
        mode_coord = np.asarray(ds["mode"].values, dtype=int) if "mode" in ds.variables else None
        W_p_eigenvalue = complex_var(ds, "W_p_eigenvalue")
        W_v_eigenvalue = complex_var(ds, "W_v_eigenvalue")
        W_t_eigenvalue = complex_var(ds, "W_t_eigenvalue")
        di = np.asarray(ds["di"].values, dtype=float) if "di" in ds.variables else None
        dr = np.asarray(ds["dr"].values, dtype=float) if "dr" in ds.variables else None
        ca1 = np.asarray(ds["ca1"].values, dtype=float) if "ca1" in ds.variables else None
        profiles = {name: _opt_array(ds, name) for name in ("f", "mu0p", "dvdpsi", "q")}
        equilibrium = DconEquilibrium.from_attrs(ds.attrs)
        coordinates = DconCoordinates.from_attrs(ds.attrs)
        edge_scan = DconEdgeScan.from_dataset(ds)
        qlim = _opt_float(ds.attrs, "qlim")
        psilim = _opt_float(ds.attrs, "psilim")
        # DCON writes the shot/time it was told, as INT(shotnum)/INT(shottime)
        # -- both 0 for a g-file whose header carries no shot.  Kept for
        # cross-checking only: a run's identity comes from its directory, never
        # from these.
        nc_shot = _opt_int(ds.attrs, "shot")
        nc_time = _opt_int(ds.attrs, "time")

    eigenfunction = None
    bin_path = run_dir / "solutions.bin"
    if bin_path.exists():
        eigenfunction = read_solutions_bin(bin_path, mlow=mlow)

    metadata: dict[str, Any] = {"run_dir": str(run_dir), "nc_file": nc_path.name}
    if nc_shot is not None:
        metadata["nc_shot"] = nc_shot
    if nc_time is not None:
        metadata["nc_time"] = nc_time
    if eigenfunction is not None and eigenfunction.m.size and eigenfunction.m.size != mpert:
        # match/ideal.f's ipert loop should span 1..mpert, so a different block
        # count means solutions.bin and this netCDF do not describe the same
        # run (a stale file from an earlier mpert, most likely). The m labels
        # are derived from mlow, so they are then not trustworthy -- record it
        # in the metadata *and* warn, rather than let a silently mislabeled
        # eigenfunction look like a clean parse.
        metadata["eigenfunction_mpert_mismatch"] = {
            "solutions_bin_n_ipert": int(eigenfunction.m.size),
            "netcdf_mpert": mpert,
        }
        warnings.warn(
            f"{bin_path} has {eigenfunction.m.size} harmonic block(s) but "
            f"{nc_path.name} reports mpert={mpert}; the poloidal mode numbers "
            "derived from mlow may be mislabeled for this run",
            RuntimeWarning,
            stacklevel=2,
        )

    return DconOutput(
        n_tor=n_tor,
        mlow=mlow,
        mhigh=mhigh,
        mpert=mpert,
        mband=mband,
        psi_n=psi_n,
        m=m,
        mode=mode_coord,
        W_p_eigenvalue=W_p_eigenvalue,
        W_v_eigenvalue=W_v_eigenvalue,
        W_t_eigenvalue=W_t_eigenvalue,
        di=di,
        dr=dr,
        ca1=ca1,
        qlim=qlim,
        psilim=psilim,
        f=profiles["f"],
        mu0p=profiles["mu0p"],
        dvdpsi=profiles["dvdpsi"],
        q=profiles["q"],
        equilibrium=equilibrium,
        coordinates=coordinates,
        edge_scan=edge_scan,
        eigenfunction=eigenfunction,
        metadata=metadata,
    )
