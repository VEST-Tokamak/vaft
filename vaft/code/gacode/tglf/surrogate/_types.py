"""What a TGLF surrogate prediction is, and what it has to carry with it.

A neural network answers every question it is asked.  Ask one about a plasma unlike
anything in its training set and it returns a finite, plausible-looking flux with no
outward sign that nothing behind it is grounded -- which is why issue #553 section 12
requires validity and extrapolation to be *outputs*, not diagnostics a caller may
choose to look at.  The types here exist to make that structural: a
:class:`SurrogatePrediction` cannot be built without a :class:`DomainAudit`, and
:func:`~vaft.code.gacode.tglf.surrogate.inference.run_surrogate` refuses by default
when the audit says the input is outside the model's training distribution.

One honest limitation is recorded rather than papered over.  The true per-input
training bounds live inside the upstream Julia ``.bson`` artifacts; the ONNX
distribution ships only the normalisation moments (``xm``/``xsigma``).  So the measure
here is a standard-deviation distance, not a min/max containment test, and
:attr:`DomainAudit.bounds_available` says so on every audit.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Optional

import numpy as np

#: What :attr:`DomainAudit.z_scores` measures. Named in the audit so a reader never has
#: to infer the statistic from the number.
DOMAIN_MEASURE = "standard-deviation distance from the model's training mean (xm/xsigma)"

#: Distance beyond which an input is reported as outside the training distribution.
#: A proxy, not a bound -- see the module docstring.
DEFAULT_DOMAIN_THRESHOLD = 3.0

__all__ = [
    "DEFAULT_DOMAIN_THRESHOLD",
    "DOMAIN_MEASURE",
    "DomainAudit",
    "ModelContractError",
    "ModelIdentity",
    "ModelResolutionError",
    "SurrogateError",
    "SurrogateMetadata",
    "SurrogatePrediction",
    "TrainingDomainError",
]


class SurrogateError(RuntimeError):
    """Base for every refusal this backend makes."""


class ModelResolutionError(SurrogateError):
    """No model artifact could be found, or the search was ambiguous.

    Carries what was searched so the message is actionable offline, which is the whole
    requirement of issue #553 section 10.
    """


class ModelContractError(SurrogateError):
    """The model asks for inputs this local state cannot supply.

    Raised rather than substituting a zero: a feature the caller never provided is the
    one thing a normalised feature vector cannot express.
    """


class TrainingDomainError(SurrogateError):
    """The input lies outside the model's training distribution and was refused.

    Raised by default so that obtaining an extrapolated number is a deliberate act.
    The audit that caused it is attached as :attr:`audit`.
    """

    def __init__(self, message: str, audit: "DomainAudit") -> None:
        super().__init__(message)
        self.audit = audit


@dataclass(frozen=True)
class ModelIdentity:
    """Which model this is, where it came from, and what physics it claims.

    ``sat_rule``, ``electromagnetic`` and ``devices`` are read from the upstream naming
    convention (``sat3_em_d3d+mastu+nstx_azf-1``), which is the only machine-readable
    statement of them the ONNX distribution carries; :attr:`physics_source` says so
    rather than letting a parsed filename pass for a manifest.  Anything the name does
    not spell is left ``None``.
    """

    #: Upstream family name, e.g. ``sat3_em_d3d+mastu+nstx_azf-1``.
    name: str
    #: Directory holding the ensemble and its normalisation text files.
    directory: str
    #: Ensemble member file names, sorted.
    members: tuple[str, ...]
    #: SHA-256 of every file that defines the model's behaviour -- each ensemble
    #: member *and* each normalisation file. Integrity metadata for artifacts VAFT does
    #: not vendor. The moments belong in it: identical weights read through different
    #: ``xm``/``xsigma`` are a different model.
    sha256: Mapping[str, str]
    #: Which resolution step found it: ``"explicit path"``, ``"model directory"``,
    #: ``"TURBULENTTRANSPORTHOME"`` or ``"julia depot"``.
    resolved_by: str
    format: str = "onnx"
    sat_rule: Optional[int] = None
    electromagnetic: Optional[bool] = None
    devices: tuple[str, ...] = ()
    #: Name fragments that are not a device and not the saturation rule -- ``azf-1``,
    #: ``gknn31``, ``withnegD``. Kept verbatim; their meaning is upstream's.
    tags: tuple[str, ...] = ()
    #: Upstream package version, when the artifact was found inside a Julia depot.
    upstream_version: Optional[str] = None
    #: Other directories holding a byte-identical copy of this ensemble. Non-empty
    #: means the search matched more than once and the copies agreed; when they
    #: disagree the resolver refuses instead of choosing.
    alternatives: tuple[str, ...] = ()
    physics_source: str = "parsed from the upstream model name"

    @property
    def ensemble_size(self) -> int:
        return len(self.members)


@dataclass(frozen=True)
class SurrogateMetadata:
    """The normalisation contract shipped beside an ONNX ensemble.

    ``xnames``/``ynames`` are the input and output channels in the order the network's
    tensors use them; the moments are what map between physical and normalised space.
    A name ending ``_log10`` asks for the base-10 logarithm of the TGLF key with the
    suffix removed, which is upstream's convention in ``src/tglf_nn.jl``.
    """

    xnames: tuple[str, ...]
    ynames: tuple[str, ...]
    xm: np.ndarray
    xsigma: np.ndarray
    ym: np.ndarray
    ysigma: np.ndarray

    def __post_init__(self) -> None:
        for label, names, moments in (
            ("input", self.xnames, (self.xm, self.xsigma)),
            ("output", self.ynames, (self.ym, self.ysigma)),
        ):
            for moment in moments:
                if np.size(moment) != len(names):
                    raise ModelContractError(
                        f"the {label} normalisation has {np.size(moment)} entries for "
                        f"{len(names)} {label} names; the model directory is inconsistent"
                    )
        if np.any(np.asarray(self.xsigma, dtype=float) == 0.0):
            raise ModelContractError(
                "an input normalisation sigma is zero, so no z-score is defined for "
                "that channel; the model directory is unusable for a domain audit"
            )

    @property
    def requires_base_model_outputs(self) -> tuple[str, ...]:
        """Input channels that are another model's *outputs*.

        The ``*_gknn*`` families are two-stage corrections: they take a base network's
        fluxes as features.  They are not drop-in replacements, and assembling their
        feature vector from TGLF keys alone silently yields NaN.
        """
        return tuple(name for name in self.xnames if name.startswith("OUT_"))


@dataclass(frozen=True)
class DomainAudit:
    """How far a local input sits from what the model was trained on.

    Attributes
    ----------
    z_scores
        Per input channel, ``(x - xm) / xsigma``.
    violations
        Channels beyond :attr:`threshold`, worst first.  The names are the model's, so
        ``TAUS_2`` names the ion-to-electron temperature ratio of species 2.
    assumed
        Channels whose value rests on a TGLF default standing in for a quantity the
        state could not supply -- ``VEXB_SHEAR`` on a profile with no ``w0``.  Reported
        rather than refused: the native run writes the same default, and a surrogate
        that rejected what TGLF accepts would no longer be a surrogate of it.
    bounds_available
        False for every ONNX artifact upstream publishes.  True min/max bounds exist
        only inside the Julia ``.bson``, so :attr:`z_scores` is a proxy for containment
        and this flag is what stops it being read as one.
    """

    model: str
    z_scores: Mapping[str, float]
    threshold: float
    violations: tuple[str, ...]
    max_abs_z: float
    assumed: tuple[str, ...] = ()
    measure: str = DOMAIN_MEASURE
    bounds_available: bool = False

    @property
    def in_domain(self) -> bool:
        return not self.violations

    def summary(self) -> str:
        """One line a refusal message or a log can carry verbatim."""
        if self.in_domain:
            return (
                f"{self.model}: within {self.threshold:g} sigma on every input "
                f"(worst |z| = {self.max_abs_z:.2f})"
            )
        worst = ", ".join(
            f"{name} z={self.z_scores[name]:+.1f}" for name in self.violations[:4]
        )
        more = "" if len(self.violations) <= 4 else f" and {len(self.violations) - 4} more"
        return (
            f"{self.model}: {len(self.violations)} of {len(self.z_scores)} inputs beyond "
            f"{self.threshold:g} sigma ({worst}{more})"
        )


@dataclass(frozen=True)
class SurrogatePrediction:
    """One surrogate evaluation, with the grounds for trusting it or not.

    ``outputs`` are the ensemble mean per output channel and ``uncertainty`` the
    ensemble standard deviation.  The spread is not a calibrated error bar, but it does
    track extrapolation: members agree where they were trained and diverge where they
    were not.
    """

    outputs: Mapping[str, float]
    uncertainty: Mapping[str, float]
    model: ModelIdentity
    domain: DomainAudit
    #: The vector the network was fed, in physical space -- after the ``_log10``
    #: transform the channel names ask for, before the ``(x - xm) / xsigma``
    #: normalisation, which happens inside the inference call.
    features: Mapping[str, float]
    provenance: Mapping[str, Any] = field(default_factory=dict)

    @property
    def qualified(self) -> bool:
        """Whether this prediction rests on an input the model was trained for."""
        return self.domain.in_domain

    def relative_spread(self) -> dict[str, float]:
        """Ensemble spread as a fraction of the mean, per channel.

        ``inf`` where the mean is zero: the spread is then real and the ratio is not,
        and reporting zero would invert the meaning.
        """
        return {
            name: (
                float("inf") if self.outputs[name] == 0.0
                else abs(self.uncertainty[name] / self.outputs[name])
            )
            for name in self.outputs
        }
