"""Which detailed diagram draws each science box of the platform figure, or which gap stands in for one.

:data:`COVERAGE` pairs every box of :func:`vaft.diagram.vest_data_platform`
(read from :mod:`vaft.diagram._platform`, never restated) with the workflow
diagrams that draw it and the tracked gaps that remain. :func:`coverage_markdown`
is the table the Diagrams guide carries; a test keeps the two equal and checks
that every named diagram is a public builder.
"""

from __future__ import annotations

from typing import List, NamedTuple, Tuple

from ._platform import DERIVED_PHYSICS, EXPERIMENTAL_PROCESSING, RECONSTRUCTION, SIMULATION


class Coverage(NamedTuple):
    area: str
    box: str
    diagrams: Tuple[str, ...]
    gaps: str


def _simulation_box(domain: str, concept: str, code: str) -> str:
    return f"{domain}: {concept} ({code})"


COVERAGE: Tuple[Coverage, ...] = (
    Coverage("Experimental data processing", EXPERIMENTAL_PROCESSING[0], ("machine_model_history",),
             "no single shot-to-machine object; VEST COCOS index unresolved"),
    Coverage("Experimental data processing", EXPERIMENTAL_PROCESSING[1], ("signal_processing",),
             "no shared drift-correction, synchronization or integration stage"),
    Coverage("Experimental data processing", EXPERIMENTAL_PROCESSING[2], ("quality_validation",),
             "only source validity reaches IMAS"),
    Coverage("Experimental data processing", EXPERIMENTAL_PROCESSING[3], ("fault_anomaly_detection",),
             "no general anomaly model runs in the pipeline"),
    Coverage("Experimental data processing", EXPERIMENTAL_PROCESSING[4], ("shot_classification",),
             "the four-way physics class is not implemented (#57); no disruption labels"),
    Coverage("Experimental data processing", EXPERIMENTAL_PROCESSING[5], ("event_detection",),
             "no breakdown, flat-top, IRE, sawtooth or disruption detector"),
    Coverage("Reconstruction", "Eddy Current Model", ("eddy_current_model",), ""),
    Coverage("Reconstruction", "Magnetic EFIT", ("magnetic_efit",), ""),
    Coverage("Reconstruction", "Profile Fitting", ("kinetic_profile_fitting",),
             "no joint multi-diagnostic fit (#1096)"),
    Coverage("Reconstruction", "Plasma Parameter Inference",
             ("plasma_parameter_inference", "parameter_inference_overview", "parameter_inference_dependency_graph",
              "romero_transformer_balance", "resistive_zeff_inference"), "E_r and gamma_E not inferred: gamma_E = 0 is an assumption"),
    Coverage("Reconstruction", "Kinetic EFIT", ("kinetic_efit",), ""),
    Coverage("Derived physics", "Vacuum Field Proxies", ("vacuum_field_proxies",),
             "vacuum products are returned, not written to IMAS (no issue yet)"),
    Coverage("Derived physics", "MHD Parameters", ("derived_mhd_parameters",),
             "two poloidal-beta definitions (#318); corrupt vacuum_toroidal_field.r0 (#325)"),
    Coverage("Derived physics", "Synthetic Diagnostics", ("synthetic_diagnostics",),
             "no synthetic Thomson, interferometer or camera (#2046); no IMAS storage of synthetic signals (#2053)"),
    Coverage("Derived physics", "Coordinate Conversion", ("coordinate_cocos_conversions",),
             "the stored flux-surface grid is geometric, not straight-field-line (#472)"),
    Coverage("Derived physics", "Power Balance", ("power_balance_confinement",),
             "no measured P_rad (no bolometer); nothing written to the summary IDS (#1060, #1905)"),
    Coverage("Simulation", _simulation_box("Equilibrium", *SIMULATION["Equilibrium"][0]), ("chease_coupling",), ""),
    Coverage("Simulation", _simulation_box("Equilibrium", *SIMULATION["Equilibrium"][1]), ("tokamaker_coupling",), ""),
    Coverage("Simulation", _simulation_box("Equilibrium", *SIMULATION["Equilibrium"][2]),
             ("analytic_mhd_equilibrium",), ""),
    Coverage("Simulation", _simulation_box("Stability", *SIMULATION["Stability"][0]), ("dcon_rdcon_stability",), ""),
    Coverage("Simulation", _simulation_box("Stability", *SIMULATION["Stability"][1]), ("dcon_rdcon_stability",), ""),
    Coverage("Simulation", _simulation_box("3D Response & Topology", *SIMULATION["3D Response & Topology"][0]),
             ("gpec_plasma_response",), ""),
    Coverage("Simulation", _simulation_box("3D Response & Topology", *SIMULATION["3D Response & Topology"][1]),
             ("flare_field_line_topology",), ""),
    Coverage("Simulation", _simulation_box("Transport", *SIMULATION["Transport"][0]), ("transport_hierarchy",),
             "classical transport has a private identifier.index"),
    Coverage("Simulation", _simulation_box("Transport", *SIMULATION["Transport"][1]),
             ("transport_hierarchy", "neo_neoclassical", "neoclassical_model_comparison"),
             "Sauter / Redl results have no core_transport writer"),
    Coverage("Simulation", _simulation_box("Transport", *SIMULATION["Transport"][2]),
             ("transport_hierarchy", "tglf_cgyro_local_transport"), "the TGLF-NN surrogate has no IMAS writer"),
)


def platform_boxes() -> List[Tuple[str, str]]:
    """Every science box of the platform figure, as ``(area, box)``, in the order the figure declares them."""
    boxes = [("Experimental data processing", b) for b in EXPERIMENTAL_PROCESSING]
    boxes += [("Reconstruction", b) for b in RECONSTRUCTION]
    boxes += [("Derived physics", b) for b in DERIVED_PHYSICS]
    boxes += [("Simulation", _simulation_box(domain, concept, code))
              for domain, leaves in SIMULATION.items() for concept, code in leaves]
    return boxes


def coverage_markdown() -> str:
    """The coverage table, as the Diagrams guide carries it (a test keeps the two equal)."""
    lines = ["| Area | Box | Detailed diagram | Tracked gaps |", "| --- | --- | --- | --- |"]
    for row in COVERAGE:
        diagrams = ", ".join(f"`{key}`" for key in row.diagrams) or "--"
        lines.append(f"| {row.area} | {row.box} | {diagrams} | {row.gaps or '--'} |")
    return "\n".join(lines)
