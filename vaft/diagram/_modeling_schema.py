"""The vocabulary of the integrated-modeling diagrams (#1085), kept apart from their drawing.

A model is placed by three independent axes -- where its knowledge comes
from, how it is evaluated, and at what physical description level it
represents the system -- and, separately, by the role it plays in a
workflow. Models are connected by typed couplings. The diagrams in
``_integrated_modeling`` draw these tuples; documentation, tutorials and
future provenance tooling can read the same vocabulary. Nothing here is a
stable public API yet.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

#: knowledge basis, first-principles to data-driven: (key, label, position on the axis in [0, 1], generic example)
KNOWLEDGE_BASIS: Tuple[Tuple[str, str, float, str], ...] = (
    ("first_principles", "mechanistic / first-principles", 0.07, "conservation laws and governing equations"),
    ("semi_empirical", "semi-empirical / closure", 0.40, "physical form, fitted coefficients"),
    ("empirical", "empirical", 0.68, "fitted relation without a derived form"),
    ("data_driven", "data-driven", 0.93, "learned from data, no prescribed form"),
)
#: physics-informed is a bridge over the knowledge axis, not one point on it: its span in [0, 1]
PHYSICS_INFORMED_SPAN: Tuple[float, float] = (0.2, 0.86)

#: computational realization, analytical to fully numerical: (key, label, position in [0, 1], generic example)
COMPUTATIONAL_REALIZATION: Tuple[Tuple[str, str, float, str], ...] = (
    ("analytical", "analytical", 0.07, "closed-form solution"),
    ("semi_analytical", "semi-analytical", 0.33, "asymptotic expansion, Green-function reduction, quadrature"),
    ("reduced_numerical", "reduced numerical", 0.58, "reduced-order solver: 1-D ODE, averaged model"),
    ("fully_numerical", "fully numerical", 0.88, "full iterative PDE solver"),
)

#: physical abstraction, most to least resolved: (key, label, what the level resolves)
PHYSICAL_ABSTRACTION: Tuple[Tuple[str, str, str], ...] = (
    ("particle", "particle / orbit", "individual trajectories"),
    ("kinetic", "kinetic", "the distribution in velocity space"),
    ("fluid", "moment / fluid", "densities, flows, temperatures"),
    ("mhd", "MHD", "one magnetised conducting fluid"),
    ("equilibrium", "equilibrium / static", "force balance, no dynamics"),
)
#: the reduction from each level to the next one down
ABSTRACTION_REDUCTIONS: Tuple[str, ...] = (
    "ensemble average",
    "velocity moments + closure",
    "single fluid, low frequency",
    "stationary force balance",
)

#: the role a model plays in a workflow: metadata, not an axis
MODEL_ROLES: Tuple[str, ...] = ("forward", "inverse", "closure", "surrogate", "estimator", "classifier")

#: the explanatory layer: conceptual and heuristic models explain; they are not a fourth axis
CONCEPTUAL_LAYER: Tuple[str, ...] = ("physical picture", "cartoon model", "scaling argument", "toy model")

#: typed couplings between models, data and states: (key, label); the template style is "coupling <key>"
COUPLING_TYPES: Tuple[Tuple[str, str], ...] = (
    ("data_flow", "data flow"),
    ("closure", "closure"),
    ("calibration", "parameter inference / calibration"),
    ("surrogate", "surrogate replacement"),
    ("residual", "residual correction"),
    ("validation", "validation / benchmarking"),
    ("feedback", "feedback / control"),
    ("iterative", "iterative coupling"),
)


@dataclass(frozen=True)
class ModelDescriptor:
    """One model placed in the integrated modeling space.

    ``knowledge`` and ``realization`` are positions in [0, 1] along the two
    axes and are semantic: models of the same knowledge basis share an x and
    are separated in y. A learned model is not a realization rung: a
    surrogate takes the y of the model it ``emulates`` (and is linked to
    it); one that emulates no model sits at the rung of its evaluation.
    ``abstraction`` is a key of ``PHYSICAL_ABSTRACTION`` or None for a
    global (0-D) quantity.
    """

    name: str
    knowledge: float
    realization: float
    abstraction: Optional[str]
    role: str
    emulates: Optional[str] = None

    def __post_init__(self):
        if not (0.0 <= self.knowledge <= 1.0 and 0.0 <= self.realization <= 1.0):
            raise ValueError(f"{self.name}: positions must lie in [0, 1]")
        if self.abstraction is not None and self.abstraction not in {k for k, _, _ in PHYSICAL_ABSTRACTION}:
            raise ValueError(f"{self.name}: unknown abstraction {self.abstraction!r}")
        if self.role not in MODEL_ROLES:
            raise ValueError(f"{self.name}: unknown role {self.role!r}")


@dataclass(frozen=True)
class ModelCoupling:
    """A typed edge of the integrated modeling process."""

    source: str
    target: str
    coupling: str

    def __post_init__(self):
        if self.coupling not in {k for k, _ in COUPLING_TYPES}:
            raise ValueError(f"unknown coupling type {self.coupling!r}")


#: domain-independent examples of the modeling space
GENERIC_MODELS: Tuple[ModelDescriptor, ...] = (
    ModelDescriptor("analytic equilibrium", 0.14, 0.07, "equilibrium", "forward"),
    ModelDescriptor("linear stability code", 0.14, 0.80, "mhd", "forward"),
    ModelDescriptor("orbit-following code", 0.14, 0.92, "particle", "forward"),
    ModelDescriptor("quasilinear reduced model", 0.40, 0.55, "kinetic", "closure"),
    ModelDescriptor("transport solver with closures", 0.42, 0.72, "fluid", "forward"),
    ModelDescriptor("physics-informed surrogate", 0.76, 0.55, "kinetic", "surrogate",
                    emulates="quasilinear reduced model"),
    ModelDescriptor("empirical scaling law", 0.70, 0.07, None, "estimator"),
    ModelDescriptor("event classifier", 0.88, 0.40, None, "classifier"),
)

#: a small fusion overlay: where familiar codes and models sit
FUSION_MODELS: Tuple[ModelDescriptor, ...] = (
    ModelDescriptor("Solov'ev", 0.09, 0.07, "equilibrium", "forward"),
    ModelDescriptor("CHEASE", 0.09, 0.93, "equilibrium", "forward"),
    ModelDescriptor("GPEC", 0.09, 0.855, "mhd", "forward"),
    ModelDescriptor("ASCOT5", 0.09, 0.78, "particle", "forward"),
    ModelDescriptor("BEAMS3D", 0.09, 0.705, "particle", "forward"),
    ModelDescriptor("DCON / RDCON", 0.09, 0.62, "mhd", "forward"),
    ModelDescriptor("EFIT", 0.30, 0.86, "equilibrium", "inverse"),
    ModelDescriptor("TGLF", 0.42, 0.55, "kinetic", "closure"),
    ModelDescriptor("TRANSP", 0.44, 0.74, "fluid", "inverse"),
    ModelDescriptor("IPB98(y,2) scaling", 0.70, 0.07, None, "estimator"),
    ModelDescriptor("neural-operator surrogate", 0.80, 0.55, "kinetic", "surrogate", emulates="TGLF"),
    ModelDescriptor("event classifier", 0.90, 0.40, None, "classifier"),
)

#: one phenomenon at successive realizations: the Rutherford equation carries fitted coefficients
TEARING_PATH: Tuple[ModelDescriptor, ...] = (
    ModelDescriptor("reduced analytical model", 0.15, 0.07, "mhd", "forward"),
    ModelDescriptor("Rutherford equation", 0.28, 0.33, "mhd", "forward"),
    ModelDescriptor("reduced numerical model", 0.15, 0.58, "mhd", "forward"),
    ModelDescriptor("nonlinear MHD simulation", 0.16, 0.88, "mhd", "forward"),
)

#: the generic integrated modeling process: (role, label) of each node
PROCESS_NODES: Tuple[Tuple[str, str], ...] = (
    ("experiment", "Experiment / plant"),
    ("measurement", "Measurement"),
    ("processing", "Data processing"),
    ("inverse", "Inverse model"),
    ("state", "Physical state"),
    ("forward", "Forward model"),
    ("data_driven", "Data-driven model"),
    ("prediction", "Prediction"),
    ("validation", "Validation / Control"),
)
PROCESS_COUPLINGS: Tuple[ModelCoupling, ...] = (
    ModelCoupling("experiment", "measurement", "data_flow"),
    ModelCoupling("measurement", "processing", "data_flow"),
    ModelCoupling("processing", "inverse", "calibration"),
    ModelCoupling("inverse", "state", "data_flow"),
    ModelCoupling("state", "forward", "iterative"),
    ModelCoupling("state", "data_driven", "data_flow"),
    ModelCoupling("data_driven", "forward", "closure"),
    ModelCoupling("data_driven", "forward", "surrogate"),
    ModelCoupling("forward", "prediction", "data_flow"),
    ModelCoupling("data_driven", "prediction", "residual"),
    ModelCoupling("prediction", "validation", "validation"),
    ModelCoupling("processing", "validation", "data_flow"),
    ModelCoupling("validation", "data_driven", "calibration"),
    ModelCoupling("validation", "experiment", "feedback"),
)
