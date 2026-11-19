"""Toroidal Fourier conventions as declared metadata, and the explicit conversion between them.

A complex toroidal harmonic ``C_n`` does not say which reconstruction it belongs
to.  ``Re(C e^{-i n phi})`` and ``Re(C e^{+i n phi})`` are different fields,
and a coefficient normalised as a plain sample mean is half the one normalised
as a peak amplitude.  VAFT uses several conventions on purpose, each matching
the code it talks to.  A convention recorded only in a docstring cannot be
checked; this module names them, so two coefficients can be compared or converted only when
both say how they were made.

A convention here fixes the coefficient of a **real** field ``f(phi)``:

``C_n = c * mean_phi( f(phi) exp(s i n phi) )``,
``f(phi) = (2 / c) Re( C_n exp(-s i n phi) )`` for ``n != 0``,

with ``s`` the kernel sign and ``c`` the coefficient scale.  Changing ``s``
conjugates the coefficient; changing ``c`` rescales it.  Nothing else is
convertible: a different toroidal-angle orientation, phase origin or poloidal
angle changes what ``phi`` and ``theta`` are, and that is never bridged here.

Nothing in this module inspects array values or dtypes to decide a convention.
A coefficient is converted only from a declared source convention to a declared
target convention.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional

__all__ = [
    "FourierConvention",
    "UnresolvedConventionError",
    "COIL_SECTOR_DECOMPOSITION",
    "GPEC_HARMONIC",
    "HELICAL_HARMONIC",
    "CONVENTIONS",
    "fourier_conversion",
    "convert_harmonic",
    "reconstruct_real_field",
]

#: The IMAS toroidal angle: counter-clockwise seen from above, origin at the
#: machine's phi = 0.  VEST's port clock maps onto it through
#: ``vaft.machine_mapping.conventions.VEST_PORT_CLOCK_TO_PHI_SIGN``.
IMAS_PHI = "imas_phi"


class UnresolvedConventionError(ValueError):
    """The conventions are not declared completely enough, or differ in a way no conversion can bridge."""


@dataclass(frozen=True)
class FourierConvention:
    """How a complex toroidal harmonic of a real field was defined.

    ``kernel_sign`` is ``s`` in the forward kernel ``exp(s i n phi)``;
    ``coefficient_scale`` is ``c``, 1 for the plain sample mean and 2 for the
    peak-amplitude coefficient.  ``toroidal_angle`` and ``phase_origin`` name
    the angle and its zero; ``poloidal_angle`` is set only for a helical
    ``(m, n)`` harmonic.  A field left ``None`` is undeclared, and an undeclared
    field makes every conversion refuse.
    """

    name: str
    kernel_sign: Optional[int]
    coefficient_scale: Optional[float]
    toroidal_angle: Optional[str] = IMAS_PHI
    phase_origin: Optional[str] = "phi = 0"
    poloidal_angle: Optional[str] = None
    real_field: bool = True
    source: str = ""

    def __post_init__(self):
        if self.kernel_sign not in (None, -1, 1):
            raise ValueError(f"kernel_sign must be -1, +1 or None, got {self.kernel_sign!r}")
        if self.coefficient_scale is not None and not self.coefficient_scale > 0:
            raise ValueError(f"coefficient_scale must be positive, got {self.coefficient_scale!r}")

    @property
    def undeclared(self) -> tuple[str, ...]:
        """Names of the fields a conversion needs and this convention leaves ``None``."""
        needed = ("kernel_sign", "coefficient_scale", "toroidal_angle", "phase_origin")
        return tuple(name for name in needed if getattr(self, name) is None)


#: :func:`vaft.process.coils_non_axisymmetric.toroidal_mode_decomposition`:
#: ``C_n = (1/K) sum_k v_k exp(-i n phi_k)``, so ``A cos(n phi + d)`` gives
#: ``(A/2) exp(+i d)``.
COIL_SECTOR_DECOMPOSITION = FourierConvention(
    name="coil_sector_decomposition",
    kernel_sign=-1,
    coefficient_scale=1.0,
    source="vaft.process.coils_non_axisymmetric.toroidal_mode_decomposition",
)

#: The harmonic GPEC writes and FLARE reads, and the coefficient
#: :func:`vaft.process.perturbation.toroidal_phase_audit` extracts:
#: ``C = 2 mean_phi(B exp(+i n phi))``, ``B = Re(C exp(-i n phi))``.
GPEC_HARMONIC = FourierConvention(
    name="gpec_harmonic",
    kernel_sign=+1,
    coefficient_scale=2.0,
    source="vaft.process.perturbation.toroidal_phase_audit",
)

#: :func:`vaft.formula.stability.helical_harmonic`, ``Re[b exp(i(m theta - n phi))]``
#: with ``theta`` the PEST straight-field-line angle: toroidally the GPEC
#: convention, ``b = 2 conj(C_n)`` of the coil-sector decomposition.
HELICAL_HARMONIC = FourierConvention(
    name="helical_harmonic",
    kernel_sign=+1,
    coefficient_scale=2.0,
    poloidal_angle="pest",
    source="vaft.formula.stability.helical_harmonic",
)

#: Every named convention, by name.
CONVENTIONS: Mapping[str, FourierConvention] = {
    c.name: c for c in (COIL_SECTOR_DECOMPOSITION, GPEC_HARMONIC, HELICAL_HARMONIC)
}


def fourier_conversion(source: FourierConvention, target: FourierConvention) -> tuple[bool, float]:
    """``(conjugate, factor)`` that turn a ``source`` coefficient into a ``target`` one.

    ``C_target = factor * (conj(C_source) if conjugate else C_source)``.  Raises
    :class:`UnresolvedConventionError` when either convention leaves a needed
    field undeclared, describes a complex field, or differs from the other in
    toroidal angle, phase origin or poloidal angle.
    """
    for convention in (source, target):
        if convention.undeclared:
            raise UnresolvedConventionError(
                f"Fourier convention {convention.name!r} leaves {', '.join(convention.undeclared)} undeclared"
            )
        if not convention.real_field:
            raise UnresolvedConventionError(
                f"Fourier convention {convention.name!r} describes a complex field; "
                "conjugating its coefficients is not a change of kernel"
            )
    for attribute in ("toroidal_angle", "phase_origin", "poloidal_angle"):
        a, b = getattr(source, attribute), getattr(target, attribute)
        if a != b:
            raise UnresolvedConventionError(
                f"{attribute} differs ({a!r} against {b!r}); a coefficient conversion cannot bridge it"
            )
    return source.kernel_sign != target.kernel_sign, target.coefficient_scale / source.coefficient_scale


def convert_harmonic(coefficient, source: FourierConvention, target: FourierConvention):
    """Return ``coefficient`` re-expressed from ``source`` into ``target``; the input is not modified.

    Works on a complex scalar or array.  The same real-space field is
    reconstructed from the result under ``target`` as from the input under
    ``source``; see :func:`reconstruct_real_field`.
    """
    import numpy as np

    conjugate, factor = fourier_conversion(source, target)
    value = np.asarray(coefficient)
    result = factor * (np.conj(value) if conjugate else value)
    return complex(result) if result.ndim == 0 else result


def reconstruct_real_field(coefficient, phi_rad, n_tor: int, convention: FourierConvention):
    """The real field ``f(phi)`` one harmonic describes under ``convention``.

    ``f = (2/c) Re(C exp(-s i n phi))`` for ``n != 0`` and ``f = Re(C)/c`` for
    ``n = 0``.  ``coefficient`` may be an array; ``phi_rad`` broadcasts against it.
    """
    import numpy as np

    if convention.undeclared:
        raise UnresolvedConventionError(
            f"Fourier convention {convention.name!r} leaves {', '.join(convention.undeclared)} undeclared"
        )
    c = np.asarray(coefficient)
    if int(n_tor) == 0:
        return np.real(c) / convention.coefficient_scale * np.ones_like(np.asarray(phi_rad, dtype=float))
    kernel = np.exp(-convention.kernel_sign * 1j * int(n_tor) * np.asarray(phi_rad, dtype=float))
    return 2.0 / convention.coefficient_scale * np.real(c * kernel)
