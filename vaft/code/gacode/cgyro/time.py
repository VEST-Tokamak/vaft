"""CGYRO elapsed-time coordinates, separate from the input discharge timestamp.

Native time is t*c_s/a. SI conversion preserves the solver origin and does not
advance a frozen experimental background. GKDB time uses R0/(sqrt(2)*c_s) instead.
Native metadata units follow GACODE b493397 cgyro_write_initdata.f90 and
cgyro_make_profiles.F90. Local-profile scales reuse TGLFNormalisation (#1064).
"""

from collections.abc import Mapping
from dataclasses import asdict, dataclass

import numpy as np

from vaft.code.gacode.tglf.inputs import _MASS_DEUTERIUM_G
from vaft.formula.constants import QE

__all__ = [
    "CgyroTimeNormalization",
    "gyrokinetic_elapsed_time_s",
    "gyrokinetic_normalized_time",
    "gyrokinetic_time_scale_s",
]

# Reuse the GACODE mass used by the existing profile-to-local conversion.
_DEUTERIUM_KG = _MASS_DEUTERIUM_G * 1e-3


def _positive(value, name):
    if value is None or not np.isscalar(value):
        raise ValueError(f"missing scalar {name} for CGYRO time normalization")
    value = float(value)
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return value


@dataclass(frozen=True)
class CgyroTimeNormalization:
    """Verified per-run SI reference scales; the mass convention is deuterium.

    Parameters are SI: length [m], electron temperature [J], mass [kg], speed
    [m/s]. Source identifies the effective run metadata or retained local input.
    GACODE rounds its energy conversion constant; the recorded speed and mass are retained
    after relative 1e-4 consistency checks, rather than recomputed silently.
    """

    reference_length_m: float
    electron_temperature_j: float
    reference_mass_kg: float
    sound_speed_m_s: float
    source: str

    def __post_init__(self):
        for key in (
            "reference_length_m",
            "electron_temperature_j",
            "reference_mass_kg",
            "sound_speed_m_s",
        ):
            object.__setattr__(self, key, _positive(getattr(self, key), key))
        if not self.source:
            raise ValueError("normalization source is required")
        if not np.isclose(self.reference_mass_kg, _DEUTERIUM_KG, rtol=1e-4, atol=0):
            raise ValueError(
                "CGYRO reference mass must use GACODE deuterium convention"
            )
        expected = np.sqrt(self.electron_temperature_j / self.reference_mass_kg)
        if not np.isclose(self.sound_speed_m_s, expected, rtol=1e-4, atol=0):
            raise ValueError(
                "recorded sound speed contradicts electron temperature / deuterium mass"
            )

    @classmethod
    def from_equilibrium(cls, equilibrium, *, source="out.cgyro.equilibrium"):
        """Read native a_meters, temp_norm [keV], mass_norm [1e-27 kg], vth_norm [m/s].

        Zero scales from dimensionless PROFILE_MODEL=1 runs are unavailable. No
        machine radius or ion-species mass is substituted for missing scales.
        """
        eq = equilibrium or {}
        length = _positive(eq.get("a_meters"), "a_meters")
        temperature = _positive(eq.get("temp_norm"), "temp_norm") * 1e3 * QE
        mass = _positive(eq.get("mass_norm"), "mass_norm") * 1e-27
        speed = eq.get("vth_norm")
        speed = np.sqrt(temperature / mass) if speed is None else speed
        return cls(length, temperature, mass, speed, source)

    @classmethod
    def from_normalisation(
        cls, normalisation, *, source="retained local input normalisation"
    ):
        """Use an explicit TGLFNormalisation shared with the CGYRO local input.

        Caller must associate these scales with the run; a file path alone does
        not establish that association. Units are the existing SI contract.
        """
        return cls(
            normalisation.minor_radius,
            normalisation.electron_temperature,
            _DEUTERIUM_KG,
            normalisation.sound_speed,
            source,
        )

    @classmethod
    def from_metadata(cls, metadata):
        """Restore available serialized scales, rejecting a contradictory saved factor."""
        if metadata.get("native_unit") != "a/c_s":
            raise ValueError("saved normalization must use native CGYRO a/c_s units")
        if metadata.get("status") != "available":
            raise ValueError("SI time normalization unavailable")
        result = cls(
            **{
                key: metadata[key]
                for key in (
                    "reference_length_m",
                    "electron_temperature_j",
                    "reference_mass_kg",
                    "sound_speed_m_s",
                    "source",
                )
            }
        )
        if not np.isclose(
            metadata.get("gyrokinetic_time_scale_s", np.nan),
            result.gyrokinetic_time_scale_s,
            rtol=1e-12,
            atol=0,
        ):
            raise ValueError("saved time scale contradicts reference quantities")
        return result

    @property
    def gyrokinetic_time_scale_s(self):
        """One native a/c_s unit in seconds."""
        return self.reference_length_m / self.sound_speed_m_s

    def metadata(self):
        """Serializable reference scales and temporal-domain provenance."""
        return {
            **asdict(self),
            "status": "available",
            "gyrokinetic_time_scale_s": self.gyrokinetic_time_scale_s,
            "native_coordinate": "gyrokinetic_normalized_time",
            "native_unit": "a/c_s",
            "si_coordinate": "gyrokinetic_elapsed_time_s",
            "origin": "solver time origin",
            "mass_convention": "GACODE deuterium",
            "experimental_trajectory_equivalence": False,
        }


def _normalization(source):
    if isinstance(source, CgyroTimeNormalization):
        return source
    if hasattr(source, "equilibrium"):
        return CgyroTimeNormalization.from_equilibrium(
            source.equilibrium, source=f"{source.directory}/out.cgyro.equilibrium"
        )
    if isinstance(source, Mapping):
        return CgyroTimeNormalization.from_equilibrium(source)
    raise TypeError(
        "expected CgyroTimeNormalization, CgyroOutputs or native equilibrium mapping"
    )


def gyrokinetic_time_scale_s(source):
    """Return the verified native a/c_s time scale [s]; missing scales raise ValueError."""
    return _normalization(source).gyrokinetic_time_scale_s


def gyrokinetic_elapsed_time_s(time, normalization):
    """Convert CGYRO normalized elapsed time to seconds, preserving the solver origin."""
    return np.asarray(time, dtype=float) * gyrokinetic_time_scale_s(normalization)


def gyrokinetic_normalized_time(elapsed_time_s, normalization):
    """Convert elapsed seconds to CGYRO t*c_s/a; discharge timestamps are not inputs."""
    return np.asarray(elapsed_time_s, dtype=float) / gyrokinetic_time_scale_s(
        normalization
    )


def time_coordinate_metadata(outputs, normalisation=None):
    """Record available or unavailable scales without altering existing native products."""
    eq = getattr(outputs, "equilibrium", None) or {}
    try:
        # An explicit retained local normalization can supply scales for native
        # dimensionless runs. Positive native scales take precedence and must agree.
        if (
            all(
                eq.get(key) in (None, 0)
                for key in ("a_meters", "temp_norm", "vth_norm", "mass_norm")
            )
            and normalisation is not None
        ):
            result = CgyroTimeNormalization.from_normalisation(normalisation)
        else:
            result = _normalization(outputs)
            if normalisation is not None:
                local = CgyroTimeNormalization.from_normalisation(normalisation)
                for key in (
                    "reference_length_m",
                    "electron_temperature_j",
                    "sound_speed_m_s",
                ):
                    if not np.isclose(
                        getattr(local, key), getattr(result, key), rtol=1e-4, atol=0
                    ):
                        raise ValueError(
                            f"retained local and native normalization disagree: {key}"
                        )
        return result.metadata()
    except (ValueError, TypeError, AttributeError) as exc:
        return {
            "status": "unavailable",
            "reason": str(exc),
            "native_unit": "a/c_s",
            "native_coordinate": "gyrokinetic_normalized_time",
            "experimental_trajectory_equivalence": False,
        }


def display_time(time, *, unit="native", normalization=None):
    """Prepare plot coordinates outside renderers; no discharge-time alignment."""
    if unit not in ("native", "s", "ms"):
        raise ValueError("time_unit must be native, s or ms (gyrokinetic elapsed time)")
    if unit == "native":
        return np.asarray(time, dtype=float), "Gyrokinetic elapsed time [a/c_s]"
    values = gyrokinetic_elapsed_time_s(time, normalization)
    return values * (1e3 if unit == "ms" else 1), f"Gyrokinetic elapsed time [{unit}]"
