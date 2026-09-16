r"""Electron-cyclotron resonance relations.

This module contains machine-independent scalar relations only. Launcher
geometry, injected power, and device-specific operating frequencies belong to
the diagnostic/actuator mapping layer or to the calling analysis.
"""

from __future__ import annotations

import numpy as np

from .constants import ME, QE

__all__ = ["electron_cyclotron_resonance_field"]


def electron_cyclotron_resonance_field(frequency_hz, harmonic=1):
    r"""Magnetic-field magnitude resonant with an electron-cyclotron frequency.

    The non-relativistic electron cyclotron frequency is

    .. math::

        f = h\,\frac{e B}{2\pi m_e},

    so the field for harmonic number :math:`h` is

    .. math::

        B_{\mathrm{ECR}} = \frac{2\pi m_e f}{h e}.

    Parameters
    ----------
    frequency_hz : float or np.ndarray
        Resonant electromagnetic frequency [Hz]. Values must be finite and
        positive.
    harmonic : int, optional
        Positive electron-cyclotron harmonic number. The default is the
        fundamental, ``1``.

    Returns
    -------
    float or np.ndarray
        Resonant magnetic-field magnitude [T]. A scalar frequency returns a
        Python ``float``; an array keeps its broadcast shape.

    Notes
    -----
    This is the cold, non-relativistic resonance condition. Doppler shift,
    relativistic mass correction, finite launch angle, and accessibility are
    intentionally outside this relation.
    """
    frequency = np.asarray(frequency_hz, dtype=float)
    if not np.all(np.isfinite(frequency)) or np.any(frequency <= 0.0):
        raise ValueError(
            f"frequency_hz must be finite and positive; got {frequency_hz!r}"
        )
    if (
        isinstance(harmonic, (bool, np.bool_))
        or not isinstance(harmonic, (int, np.integer))
        or harmonic < 1
    ):
        raise ValueError(f"harmonic must be a positive integer; got {harmonic!r}")

    field = 2.0 * np.pi * ME * frequency / (harmonic * QE)
    if np.isscalar(frequency_hz) or np.ndim(frequency_hz) == 0:
        return float(field)
    return field
