"""Regenerate or check the committed reference diagrams.

::

    python -m vaft.diagram.build            # re-render diagrams whose source changed
    python -m vaft.diagram.build --force    # re-render all of them
    python -m vaft.diagram.build --check    # verify; needs no TeX

Freshness is judged on the generated TikZ source, not on SVG bytes: two
``dvisvgm`` releases write different (equally correct) SVG for the same
picture, so a byte comparison would fail on every machine but the last one
to render. ``manifest.json`` records, per asset, the SHA-256 of the TikZ
document it was rendered from and of the SVG itself. ``--check`` rebuilds
the TikZ (pure Python), and fails when either hash disagrees -- a stale
asset, or a hand-edited one -- or when an asset is missing or orphaned.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

MANIFEST = "manifest.json"

#: The reference island: a 3/2 mode on a D-shaped (elongated, triangular) plasma.
REFERENCE_ISLAND = {
    "m": 3, "n": 2, "width": 0.16, "phase": 0.0,
    "r_s": 0.55, "elongation": 1.7, "triangularity": 0.4,
}

#: asset file name -> (builder name in vaft.diagram, keyword arguments)
CANONICAL: Dict[str, Tuple[str, dict]] = {
    **{
        f"magnetic_island_{projection}.svg": ("magnetic_island", {**REFERENCE_ISLAND, "projection": projection})
        for projection in ("poloidal", "top", "3d")
    },
    # stability and operational-space charts, at their documented defaults
    **{f"{name}.svg": (name, {}) for name in ("peeling_ballooning", "s_alpha_ballooning", "hugill", "troyon")},
    # single-particle motion
    **{f"{name}.svg": (name, {}) for name in ("exb_drift", "curvature_drift", "magnetization_current",
                                              "toroidal_drift")},
    **{f"{name}_{projection}.svg": (name, {"projection": projection}) for name, projection in (
        ("exb_drift", "3d"), ("curvature_drift", "poloidal"), ("curvature_drift", "top"),
        ("magnetization_current", "3d"), ("toroidal_drift", "poloidal"), ("toroidal_drift", "top"))},
    # tearing physics, one concept per diagram
    **{f"{name}.svg": (name, {}) for name in ("rational_surface", "delta_prime", "tearing_layer_matching")},
    # 3-D perturbation harmonics
    **{f"{name}.svg": (name, {}) for name in ("normal_field_component", "complex_harmonic",
                                              "toroidal_harmonic_phase", "harmonic_real_space_projection",
                                              "complex_field_superposition")},
    # concept diagrams
    "collision_processes.svg": ("collision_processes", {}),
    # geometric approximations
    "geometry_ordering_map.svg": ("geometry_ordering_map", {}),
    **{f"field_line_geometry_{g}.svg": ("field_line_geometry", {"geometry": g})
       for g in ("toroidal", "cylindrical", "slab")},
    "mode_number_mapping.svg": ("mode_number_mapping", {}),
    # tokamak geometry and flux coordinates
    **{f"tokamak_torus_{p}.svg": ("tokamak_torus", {"projection": p}) for p in ("3d", "poloidal")},
    **{f"flux_surfaces_{s}.svg": ("flux_surfaces", {"shape": s}) for s in ("circular", "shifted")},
    **{f"{name}.svg": (name, {}) for name in ("shaping_family", "hfs_lfs_field", "safety_factor_winding",
                                              "flux_coordinates", "poloidal_angle_comparison",
                                              "unwrapped_flux_surface", "field_line_pitch")},
    # toroidicity and TF ripple
    **{f"{name}.svg": (name, {}) for name in ("trapped_and_passing_orbits", "toroidal_field_ripple",
                                              "ripple_well_formation", "stochastic_ripple_orbit")},
    # guiding-centre invariants and toroidal symmetry
    **{f"{name}.svg": (name, {}) for name in ("guiding_center_invariants", "canonical_toroidal_momentum",
                                              "toroidal_symmetry_breaking")},
    # straight-field-line coordinates
    **{f"{name}.svg": (name, {}) for name in ("sfl_coordinate_grids", "sfl_coordinate_taxonomy",
                                              "sfl_fourier_convergence")},
    # Clebsch labels, field-aligned and ballooning representations
    **{f"{name}.svg": (name, {}) for name in ("clebsch_field_line_label", "ballooning_curvature_drive",
                                              "ballooning_newcomb_test", "ballooning_harmonic_envelope",
                                              "ballooning_workflow")},
    # slab resonant layers: parity and harmonic coupling
    **{f"slab_parity_{p}.svg": ("slab_parity", {"parity": p}) for p in ("tearing", "twisting")},
    **{f"{name}.svg": (name, {}) for name in ("slab_parity_comparison", "poloidal_harmonic_coupling",
                                              "resonant_layer_matching")},
    # disruption physics: the quench sequence, causal chain, runaway generation, energy paths (#1041)
    **{f"{name}.svg": (name, {}) for name in ("disruption_timeline", "disruption_causal_chain",
                                              "runaway_generation", "disruption_energy_pathways")},
    # equilibrium-aware phenomena on the default Solov'ev equilibrium (#1209)
    "kink_mode_1_1_internal.svg": ("kink_mode", {}),
    "kink_mode_2_1_global.svg": ("kink_mode", {"m": 2, "n": 1, "radial_profile": "global", "amplitude": 0.08,
                                               "harmonics": {2: 1.0, 3: 0.3}}),
    **{f"sawtooth_{stage}.svg": ("sawtooth", {"stage": stage}) for stage in ("precursor", "reconnection",
                                                                             "post_crash")},
    **{f"stochastic_layer_{regime}.svg": ("stochastic_layer", {"regime": regime})
       for regime in ("isolated", "touching", "overlapping")},
    "separatrix_lobes.svg": ("separatrix_lobes", {}),
    # vertical displacement events: hot/cold VDE, halo currents, timescales (#1042)
    **{f"{name}.svg": (name, {}) for name in ("hot_vde_sequence", "cold_vde_bifurcation",
                                              "plasma_wall_halo_current", "vde_timescales")},
    # the Grad-Shafranov problem: regions, boundaries, topology, problem classes (#1052)
    **{f"{name}.svg": (name, {}) for name in ("grad_shafranov_domain_decomposition",
                                              "fixed_vs_free_boundary_equilibrium",
                                              "limiter_and_diverted_topologies", "equilibrium_problem_taxonomy",
                                              "poloidal_flux_source_decomposition")},
    # cylindrical geometry: profiles, mode shapes, matching
    **{f"{name}.svg": (name, {}) for name in ("current_to_q_profile", "cylindrical_rational_surfaces",
                                              "cylindrical_mode_morphology", "internal_external_kink",
                                              "plasma_vacuum_wall", "cylindrical_tearing_outer")},
    # canonical field configurations, reconnection topology and ideal-MHD waves (#1063)
    **{f"slab_field_configuration_{k}.svg": ("slab_field_configuration", {"kind": k})
       for k in ("uniform", "sheared", "reversed", "guide")},
    "current_sheet.svg": ("current_sheet", {}),
    "current_sheet_guide_field.svg": ("current_sheet", {"guide_field": True}),
    **{f"{name}.svg": (name, {}) for name in ("harris_sheet", "x_point", "magnetic_reconnection",
                                              "island_formation", "shear_alfven_wave",
                                              "fast_magnetosonic_wave", "mhd_wave_family")},
    # plasma-wall interaction concepts (#1047)
    **{f"{name}.svg": (name, {}) for name in ("plasma_wall_interaction_processes", "plasma_wall_interaction_reflection",
                                              "plasma_wall_interaction_recycling",
                                              "plasma_wall_interaction_energy_partition")},
    # E_s = 8.68 eV: the sublimation energy of W, the usual surface binding energy (Behrisch & Eckstein,
    # "Sputtering by Particle Bombardment", Springer 2007, tables) -- an input, shown on the figure
    "plasma_wall_interaction_sputtering.svg": ("plasma_wall_interaction_sputtering", {"surface_binding_energy": 8.68}),
    # spectroscopy and ionization concepts (#1046)
    "spectroscopy_ionization_stages.svg": ("spectroscopy_ionization_stages", {"term": "C III"}),
    **{f"spectroscopy_transitions_{name}.svg": ("spectroscopy_transitions", {"term": term})
       for name, term in (("h_alpha", "H-alpha"), ("oi_7770", "OI_7770"))},
    "spectroscopy_energy_levels.svg": ("spectroscopy_energy_levels", {"term": "D-alpha"}),
    "spectroscopy_spectrum.svg": ("spectroscopy_spectrum", {}),
    # neutral beam injection: lifecycle and reduced attenuation (#1136)
    **{f"{name}.svg": (name, {}) for name in ("nbi_particle_lifecycle", "nbi_neutral_attenuation")},
    # iteration behaviour, branch bifurcation and branch selection (#1093)
    **{f"{name}.svg": (name, {}) for name in ("iteration_behavior", "branch_bifurcation", "basin_of_attraction",
                                              "grid_induced_two_cycle", "branch_selection")},
    # cold-plasma waves from their equations (#1113)
    **{f"{name}.svg": (name, {}) for name in ("o_mode_cutoff", "x_mode_dispersion", "cma_diagram",
                                              "profile_propagation")},
    # neoclassical and NTV collisionality regimes (#1111)
    "neoclassical_collisionality.svg": ("neoclassical_collisionality", {}),
    "ntv_collisionality.svg": ("ntv_collisionality", {}),
    "ntv_precession_regimes.svg": ("ntv_precession_regimes", {}),
    # wall conditioning as wall-state transitions (#1051)
    "wall_conditioning_baking.svg": ("wall_conditioning_baking", {}),
    "wall_conditioning_gdc_deuterium.svg": ("wall_conditioning_gdc", {"gas": "D2"}),
    "wall_conditioning_gdc_helium.svg": ("wall_conditioning_gdc", {"gas": "He"}),
    "wall_conditioning_boronization.svg": ("wall_conditioning_boronization", {}),
    "wall_conditioning_sequence.svg": ("wall_conditioning_sequence", {}),
    # field-aligned coordinates, flux tubes, shear and the ballooning eigenfunction (#1075 part 2)
    **{f"{name}.svg": (name, {}) for name in ("field_aligned_basis", "flux_tube_patch",
                                              "magnetic_shear_field_aligned", "ballooning_eigenfunction")},
    # transits, boundary conditions and the X-point limit (#1075 remainder)
    **{f"{name}.svg": (name, {}) for name in ("ballooning_transit_map", "ballooning_boundary_conditions",
                                              "field_aligned_xpoint_limitation")},
    # a toroidal mode number: the shift nu couples harmonics in every angle but PEST (#1074)
    "sfl_fourier_convergence_n2.svg": ("sfl_fourier_convergence", {"n": 2}),
    # SFL coordinates part 2: action-angle, validity near a separatrix, coordinates vs COCOS (#1074)
    **{f"{name}.svg": (name, {}) for name in ("field_line_action_angle", "sfl_coordinate_validity",
                                              "coordinates_vs_cocos")},
    # SOL blobs and filaments: mechanism, velocity scaling, regimes (#1211)
    **{f"{name}.svg": (name, {}) for name in ("blob_polarization", "blob_velocity_scaling", "blob_regimes")},
    "blob_polarization_hole.svg": ("blob_polarization", {"perturbation": "hole"}),
    **{f"blob_current_closure_{r}.svg": ("blob_current_closure", {"regime": r}) for r in ("sheath", "inertial")},
    # MARFE on the high-field side and next to the X-point, with Drake's condition (#1209)
    "marfe.svg": ("marfe", {}),
    "marfe_xpoint.svg": ("marfe", {"localization": "xpoint"}),
    # Eich target profile on a diverted equilibrium (#1209)
    "divertor_heat_footprint.svg": ("divertor_heat_footprint", {}),
    # integrated modeling: three independent axes, their space and typed model coupling (#1085)
    "knowledge_basis.svg": ("knowledge_basis", {}),
    "computational_realization.svg": ("computational_realization", {}),
    "physical_abstraction.svg": ("physical_abstraction", {}),
    "integrated_modeling_space.svg": ("integrated_modeling_space", {}),
    "integrated_modeling_space_fusion.svg": ("integrated_modeling_space", {"examples": "fusion"}),
    "integrated_modeling_space_tearing.svg": ("integrated_modeling_space", {"examples": "tearing"}),
    "integrated_modeling_process.svg": ("integrated_modeling_process", {}),
    # the VAFT framework: lifecycle, pillars, workflow, interoperability, provenance, architecture (#1090)
    "fusion_science_knowledge_lifecycle.svg": ("fusion_science_knowledge_lifecycle", {}),
    "vaft_four_pillars.svg": ("vaft_four_pillars", {}),
    "scientific_workflow.svg": ("scientific_workflow", {}),
    "interoperability_layers.svg": ("interoperability_layers", {}),
    "scientific_provenance_chain.svg": ("scientific_provenance_chain", {}),
    "scientific_infrastructure_principles.svg": ("scientific_infrastructure_principles", {}),
    "machine_agnostic_architecture.svg": ("machine_agnostic_architecture", {}),
    "experiment_modeling_theory_data_network.svg": ("experiment_modeling_theory_data_network", {}),
    "experiment_modeling_theory_data_network_point_to_point.svg": ("experiment_modeling_theory_data_network",
                                                                   {"communication": "point_to_point"}),
    "experiment_modeling_theory_data_network_equilibrium.svg": ("experiment_modeling_theory_data_network",
                                                                {"communication": "equilibrium"}),
    "human_ai_interface.svg": ("human_ai_interface", {}),
    # the machine and research archive since 2012 (#497)
    "machine_research_archive.svg": ("machine_research_archive", {}),
}


def default_output() -> Path:
    """``docs/assets/diagrams`` of the source checkout this module runs from.

    An installed package has no ``docs/`` next to it; rather than write into
    ``site-packages``, that asks for an explicit ``--output``.
    """
    root = Path(__file__).resolve().parents[2]
    if not (root / "pyproject.toml").is_file() or not (root / "docs").is_dir():
        raise SystemExit("not running from a VAFT source checkout: pass --output DIR")
    return root / "docs" / "assets" / "diagrams"


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _call_text(builder: str, kwargs: dict) -> str:
    args = ", ".join(f"{key}={value!r}" for key, value in kwargs.items())
    return f"vaft.diagram.{builder}({args})"


def _diagram(builder: str, kwargs: dict):
    import vaft.diagram

    return getattr(vaft.diagram, builder)(**kwargs)


def _read_manifest(out_dir: Path) -> dict:
    path = out_dir / MANIFEST
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8")).get("diagrams", {})


def check(out_dir: Optional[Path] = None) -> List[str]:
    """Every way the committed assets disagree with the current source."""
    out_dir = Path(out_dir or default_output())
    recorded = _read_manifest(out_dir)
    problems: List[str] = []
    for name, (builder, kwargs) in CANONICAL.items():
        entry = recorded.get(name)
        svg = out_dir / name
        if entry is None:
            problems.append(f"{name}: not in {MANIFEST}")
            continue
        if not svg.exists():
            problems.append(f"{name}: missing")
            continue
        try:
            source_sha256 = _diagram(builder, kwargs).source_sha256
        except Exception as exc:  # noqa: BLE001 -- one broken builder must not hide the report on the others
            problems.append(f"{name}: builder raised {type(exc).__name__}: {exc}")
            continue
        if source_sha256 != entry.get("source_sha256"):
            problems.append(f"{name}: stale -- its TikZ source or the render recipe changed; run python -m vaft.diagram.build")
        if _sha256(svg.read_bytes()) != entry.get("svg_sha256"):
            problems.append(f"{name}: the SVG does not match {MANIFEST} (edited by hand?)")
    for name in sorted(set(recorded) - set(CANONICAL)):
        problems.append(f"{name}: recorded in {MANIFEST} but no longer canonical")
    for svg in sorted(out_dir.glob("*.svg")):
        if svg.name not in CANONICAL:
            problems.append(f"{svg.name}: orphaned asset, not produced by the build")
    return problems


def build(out_dir: Optional[Path] = None, *, force: bool = False) -> List[str]:
    """Render stale (or, with ``force``, all) canonical diagrams; return what was written."""
    out_dir = Path(out_dir or default_output())
    out_dir.mkdir(parents=True, exist_ok=True)
    recorded = _read_manifest(out_dir)
    manifest: Dict[str, dict] = {}
    written: List[str] = []
    for name, (builder, kwargs) in CANONICAL.items():
        diagram = _diagram(builder, kwargs)
        svg_path = out_dir / name
        entry = recorded.get(name, {})
        fresh = (
            not force
            and svg_path.exists()
            and entry.get("source_sha256") == diagram.source_sha256
            and entry.get("svg_sha256") == _sha256(svg_path.read_bytes())
        )
        if not fresh:
            svg_path.write_text(diagram.svg, encoding="utf-8", newline="\n")
            written.append(name)
        manifest[name] = {
            "call": _call_text(builder, kwargs),
            "source_sha256": diagram.source_sha256,
            "svg_sha256": _sha256(svg_path.read_bytes()),
        }
    document = {
        "generator": "python -m vaft.diagram.build",
        "note": "source_sha256 hashes the render recipe and the generated TikZ document; --check compares it without TeX.",
        "diagrams": manifest,
    }
    (out_dir / MANIFEST).write_text(json.dumps(document, indent=2, sort_keys=True) + "\n",
                                    encoding="utf-8", newline="\n")
    return written


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m vaft.diagram.build", description=__doc__.split("\n\n")[0])
    parser.add_argument("--check", action="store_true", help="verify the committed assets; needs no TeX")
    parser.add_argument("--force", action="store_true", help="re-render every diagram, fresh or not")
    parser.add_argument("--output", type=Path, default=None, help="asset directory (default: docs/assets/diagrams)")
    args = parser.parse_args(argv)
    if args.check:
        problems = check(args.output)
        for problem in problems:
            print(problem, file=sys.stderr)
        print(f"{len(CANONICAL)} diagrams, {len(problems)} problems")
        return 1 if problems else 0
    written = build(args.output, force=args.force)
    print(f"rendered {len(written)} of {len(CANONICAL)} diagrams" + (f": {', '.join(written)}" if written else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
