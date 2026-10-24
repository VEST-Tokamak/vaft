"""Which ordering contracts a calculation consumes, from its plasma formalism (#1823)."""

from __future__ import annotations

import math
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from vaft.code.formalism import PlasmaFormalism
from vaft.validation.applicability import APPLICABILITY_STATUSES
from vaft.validation.model_contracts import (
    ADAPTER_MODES,
    ROLES,
    ContractUse,
    adapter_mode,
    advisory_ordering_check,
    contracts_for,
)


def _uses(formalism):
    return [(u.contract, u.role) for u in contracts_for(formalism)[0]]


def test_the_registry_imports_without_matplotlib_or_pandas():
    code = ("import sys, vaft.validation.model_contracts as m; list(m.ADAPTER_MODES); "
            "print('matplotlib' in sys.modules, 'pandas' in sys.modules)")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout.split()
    assert out == ["False", "False"]


def test_matched_and_hybrid_calculations_return_contract_role_pairs():
    assert _uses(adapter_mode("gpec:rdcon_rmatch").formalism) == [
        ("ideal_single_fluid_mhd", "outer_solution"), ("resistive_mhd", "layer_solution")]
    assert _uses(adapter_mode("gpec:dcon_kinetic").formalism) == [
        ("ideal_single_fluid_mhd", "fluid_part"), ("drift_kinetic", "kinetic_response")]
    assert _uses(adapter_mode("pentrc:ntv_closure").formalism) == [
        ("ideal_single_fluid_mhd", "fluid_part"), ("drift_kinetic", "closure")]
    # an ideal Delta-prime calculation (STRIDE, RDCON without a solved layer) is the outer region only
    assert _uses(adapter_mode("gpec:stride").formalism) == [("ideal_single_fluid_mhd", "outer_solution")]


def test_each_family_maps_to_its_contracts():
    assert _uses(adapter_mode("efit:reconstruction").formalism) == [("quasi_static_equilibrium", "governing")]
    assert _uses(adapter_mode("gpec:dcon_ideal").formalism) == [("ideal_single_fluid_mhd", "governing")]
    assert _uses(adapter_mode("gacode:neo").formalism) == [("drift_kinetic", "governing"),
                                                           ("local_neoclassical", "governing")]
    assert _uses(adapter_mode("gacode:cgyro_linear").formalism) == [("gyrokinetic_delta_f", "governing")]
    # TGLF inherits the gyrokinetic foundations without solving gyrokinetics; CGYRO solves them
    for key in ("gacode:tglf", "gacode:tglf_nn", "gacode:tglf_linear"):
        uses, unmapped = contracts_for(adapter_mode(key).formalism)
        assert [(u.contract, u.role) for u in uses] == [("gyrokinetic_delta_f", "inherited")], key
        assert unmapped and "closure" in unmapped[0]


def test_gk_transport_separation_only_for_flux_driven_transport_coupling():
    # a standalone nonlinear run holds the background fixed under the delta-f ordering itself
    for key in ("gacode:cgyro_nonlinear", "gacode:tglf"):
        assert "gyrokinetic_transport_separation" not in [c for c, _ in _uses(adapter_mode(key).formalism)], key
        coupled = contracts_for(adapter_mode(key).formalism, transport_coupled=True)[0]
        assert ("gyrokinetic_transport_separation", "transport_coupling") in [(u.contract, u.role) for u in coupled]
    # a linear run has no fluxes to couple
    linear = contracts_for(adapter_mode("gacode:cgyro_linear").formalism, transport_coupled=True)[0]
    assert "gyrokinetic_transport_separation" not in [u.contract for u in linear]
    assert not any(m.transport_coupled for m in ADAPTER_MODES)       # no VAFT workflow couples them yet


def test_unmapped_parts_are_reasons_never_guessed_contracts():
    uses, unmapped = adapter_mode("nubeam:fast_ions").contracts()
    assert uses == () and "test-particle" in unmapped[0]
    full_f = PlasmaFormalism(scientific_operation="turbulent_transport", bulk_description="kinetic",
                             kinetic_equation="gyrokinetic", distribution_formulation="full_f")
    uses, unmapped = contracts_for(full_f)
    assert uses == () and "full_f" in unmapped[0]
    for mode in ADAPTER_MODES:
        uses, unmapped = mode.contracts()
        assert uses or unmapped, mode.key
        assert all(u.role in ROLES for u in uses)
    with pytest.raises(ValueError):
        ContractUse("ideal_single_fluid_mhd", "whole")
    with pytest.raises(KeyError, match="known"):
        adapter_mode("efit:magic")


def test_cgyro_modes_are_the_backends_own_records():
    from vaft.code.gacode.cgyro import CGYROConfig, plasma_formalism

    assert adapter_mode("gacode:cgyro_linear").formalism == plasma_formalism(CGYROConfig())
    assert adapter_mode("gacode:cgyro_nonlinear").formalism.regime == "nonlinear"


# -- the registry against the GPEC-suite resolver (#1734): same contracts from the real namelists --

def _cell(tmp_path: Path, name: str, **overrides: str) -> Path:
    from vaft.data.resources import data_path

    cell = tmp_path / name.split(".")[0]
    cell.mkdir(parents=True, exist_ok=True)
    text = (Path(data_path("gpec")) / name).read_text(encoding="utf-8")
    for key, value in overrides.items():
        text, count = re.subn(rf"(?im)^(\s*{key}\s*=\s*)[^!\n]*", rf"\g<1>{value} ", text)
        assert count == 1
    (cell / name).write_text(text, encoding="utf-8")
    return cell


def test_gpec_registry_modes_map_like_the_resolved_records(tmp_path):
    from vaft.code.gpec._formalism import resolve_plasma_formalism
    from vaft.data.resources import data_path

    rdcon = _cell(tmp_path / "r", "rdcon.in")
    shutil.copy(Path(data_path("gpec")) / "rmatch.in", rdcon / "rmatch.in")
    (rdcon / "delta.out").write_text("", encoding="utf-8")
    ideal_dcon = _cell(tmp_path / "di", "dcon.in")
    kinetic_dcon = _cell(tmp_path / "dk", "dcon.in", kin_flag="t")
    gpec = _cell(tmp_path / "g", "gpec.in")
    resolved = {
        "gpec:dcon_ideal": resolve_plasma_formalism("dcon", ideal_dcon),
        "gpec:dcon_kinetic": resolve_plasma_formalism("dcon", kinetic_dcon),
        "gpec:gpec_ideal": resolve_plasma_formalism("gpec", gpec, dcon_dir=ideal_dcon),
        "gpec:gpec_kinetic": resolve_plasma_formalism("gpec", gpec, dcon_dir=kinetic_dcon),
        "gpec:rdcon_rmatch": resolve_plasma_formalism("rdcon", rdcon),
        "gpec:stride": resolve_plasma_formalism("stride", _cell(tmp_path / "s", "stride.in")),
        "pentrc:ntv_closure": resolve_plasma_formalism("pentrc", _cell(tmp_path / "p", "pentrc.in")),
    }
    def fields(record):
        d = record.as_dict()
        d.pop("extensions")
        d.pop("solver")
        return d

    for key, record in resolved.items():
        assert record is not None, key
        assert fields(record) == fields(adapter_mode(key).formalism), key   # the whole record, not only its contracts
        assert _uses(record) == _uses(adapter_mode(key).formalism), key


# -- the advisory check --

def _formalism(key):
    return adapter_mode(key).formalism


def test_advisory_check_evaluates_state_level_contracts_on_the_state():
    state = {"tau_evolution_over_tau_alfven": 3e4, "sonic_mach_number": 0.05, "alfven_mach_number": 0.01}
    results, unmapped = advisory_ordering_check(_formalism("efit:reconstruction"), state)
    (result,) = results
    assert unmapped == () and result.evaluation.status == "SUPPORTED" and result.point is None
    assert result.evidence.axis == "applicability"
    assert result.evidence.key == "applicability.efit.quasi_static_equilibrium.governing"
    assert result.as_dict()["margins"]["tau_evolution_over_tau_alfven"] == pytest.approx(math.log10(3e4))


def test_missing_mode_and_layer_inputs_are_unassessed_never_a_pass():
    state = {"debye_length_over_L": 1e-5, "lundquist_number": 1e5, "ion_skin_depth_over_a": 0.3,
             "rho_i_over_LTi": 0.02, "pressure_anisotropy": 0.01}
    results, _ = advisory_ordering_check(_formalism("gpec:rdcon_rmatch"), state)
    statuses = {r.use.role: r.evaluation.status for r in results}
    assert statuses == {"outer_solution": "UNASSESSED", "layer_solution": "UNASSESSED"}   # no omega, k or layer width
    for r in results:
        assert r.evaluation.status in APPLICABILITY_STATUSES
        missing = {a.quantity for a in r.evaluation.assumptions if a.status == "UNASSESSED"}
        assert "k_perp_rho_i" in missing
    layer = [r for r in results if r.use.role == "layer_solution"][0]
    assert {"ion_skin_depth_over_layer", "rho_s_over_layer"} <= {
        a.quantity for a in layer.evaluation.assumptions if a.status == "UNASSESSED"}


def test_local_contracts_are_evaluated_per_point_with_the_state_attached():
    state = {"debye_length_over_L": 1e-5, "omega_over_ion_gyrofrequency": 1e-3, "fluctuation_amplitude": 0.01,
             "k_par_over_k_perp": 1e-3, "sonic_mach_number": 0.05}
    points = {"rho_i_over_LTi": [0.01, 0.05, 2.0], "debye_length_over_L": [1e-5, 1e-5, 1e-5]}
    results, _ = advisory_ordering_check(_formalism("gacode:cgyro_linear"), state, points)
    (result,) = results
    assert result.counts == {"SUPPORTED": 2, "OUTSIDE": 1}
    assert result.evaluation.status == "OUTSIDE" and result.point == 2           # the most constraining point
    assert result.evaluation.limiting == "rho_i_over_LTi"
    assert set(result.counts) <= set(APPLICABILITY_STATUSES) and "MARGINAL" not in result.counts


def test_a_formalism_without_contracts_returns_its_reasons():
    results, unmapped = advisory_ordering_check(_formalism("nubeam:fast_ions"), {})
    assert results == () and unmapped


def test_reduced_and_linear_modes_and_an_unknown_parent():
    assert _uses(adapter_mode("gacode:neo_analytic").formalism) == [
        ("drift_kinetic", "inherited"), ("local_neoclassical", "inherited")]
    # VAFT never runs the RPEC path (rdcon.in sets coil%rpec_flag=f), so it is not a mode
    with pytest.raises(KeyError):
        adapter_mode("gpec:rpec")
    assert adapter_mode("gpec:thresholds").formalism is None and adapter_mode("gpec:thresholds").contracts()[1]
    orphan = PlasmaFormalism(scientific_operation="turbulent_transport", bulk_description="reduced",
                             derived_from="unknown")
    uses, unmapped = contracts_for(orphan)
    assert uses == () and "names no parent theory" in unmapped[0]


def test_profile_inputs_broadcast_scalars_keep_finite_state_values_and_label_points():
    pd = pytest.importorskip("pandas")
    efit = _formalism("efit:reconstruction")
    state = {"tau_evolution_over_tau_alfven": 3e4, "sonic_mach_number": 0.05, "alfven_mach_number": 0.01}
    # a NaN at a point never hides the state's finite value
    (r,), _ = advisory_ordering_check(efit, state, {"sonic_mach_number": [math.nan, math.nan]})
    assert r.evaluation.status == "SUPPORTED"
    # scalars broadcast; strings are scalars; a 2-D column is refused
    gk = _formalism("gacode:cgyro_linear")
    base = {"debye_length_over_L": 1e-5, "omega_over_ion_gyrofrequency": 1e-3, "fluctuation_amplitude": 0.01,
            "k_par_over_k_perp": 1e-3, "sonic_mach_number": 0.05}
    (r,), _ = advisory_ordering_check(gk, base, {"rho_i_over_LTi": [0.01, 0.02], "label": "mid", "x": 1.0})
    assert r.counts == {"SUPPORTED": 2}
    import numpy as np

    with pytest.raises(ValueError, match="one-dimensional"):
        advisory_ordering_check(gk, base, {"rho_i_over_LTi": np.ones((2, 2))})
    # a DataFrame's index labels the most constraining point
    frame = pd.DataFrame({"rho_i_over_LTi": [0.01, 3.0]}, index=pd.Index([0.2, 0.8], name="rho"))
    (r,), _ = advisory_ordering_check(gk, base, frame)
    assert r.evaluation.status == "OUTSIDE" and r.point == 0.8


def test_equilibrium_topology_follows_the_boundary_condition():
    assert adapter_mode("chease:fixed_boundary").formalism.topology_domain == "closed_flux_surface"
    for key in ("efit:reconstruction", "tokamaker:free_boundary", "tes:free_boundary", "nice:reconstruction"):
        assert adapter_mode(key).formalism.topology_domain == "across_separatrix", key


def test_scalar_only_points_are_one_point_not_none():
    base = {"debye_length_over_L": 1e-5, "omega_over_ion_gyrofrequency": 1e-3, "fluctuation_amplitude": 0.01,
            "k_par_over_k_perp": 1e-3, "sonic_mach_number": 0.05}
    (r,), _ = advisory_ordering_check(_formalism("gacode:cgyro_linear"), base, {"rho_i_over_LTi": 5.0})
    assert r.evaluation.status == "OUTSIDE" and r.counts == {"OUTSIDE": 1} and r.point == 0


def test_unmapped_hybrid_fluid_and_non_fluid_equilibrium_give_reasons():
    hybrid = PlasmaFormalism(scientific_operation="ideal_stability", bulk_description="hybrid", fluid_model="unknown",
                             kinetic_equation="drift_kinetic", kinetic_coupling="energy")
    uses, unmapped = contracts_for(hybrid)
    assert [u.contract for u in uses] == ["drift_kinetic"] and "fluid_model='unknown'" in unmapped[0]
    kinetic_eq = PlasmaFormalism(scientific_operation="equilibrium", bulk_description="kinetic",
                                 kinetic_equation="drift_kinetic")
    uses, unmapped = contracts_for(kinetic_eq)
    assert [u.contract for u in uses] == ["quasi_static_equilibrium"] and "not classified" in unmapped[0]


def test_tglf_modes_follow_the_adapters_transport_switch():
    from vaft.code.gacode.tglf import TGLFConfig

    assert TGLFConfig().use_transport_model is True            # quasilinear fluxes by default: the tglf mode
    assert adapter_mode("gacode:tglf").formalism.regime == "quasilinear"
    assert adapter_mode("gacode:tglf_linear").formalism.regime == "linear"   # use_transport_model=False


def test_every_physics_adapter_in_vaft_code_has_a_mode():
    """#1823 acceptance: every adapter is in the registry, classified or explicitly not."""
    from pathlib import Path

    import vaft.code as code

    root = Path(code.__file__).parent
    adapters = {p.stem for p in root.glob("*.py")} | {p.name for p in root.iterdir() if (p / "__init__.py").exists()}
    infrastructure = {"__init__", "_executables", "_launch", "_memory_gate", "_process_tree", "base", "execution",
                      "formalism", "remote", "resources", "slurm", "snakemake"}
    # adapters folded into another registry adapter: CHEASE helpers into chease, PENTRC's package into pentrc
    folded = {"chease_kinetic_iteration": "chease", "chease_scan": "chease", "chease_synthesis": "chease",
              "chease_synthesis_targets": "chease"}
    registered = {m.adapter for m in ADAPTER_MODES}
    for name in sorted(adapters - infrastructure):
        assert folded.get(name, name) in registered, name
    for key in ("flare:field_lines", "mitim:drivers"):
        mode = adapter_mode(key)
        assert mode.formalism is None and mode.contracts() == ((), (mode.note,))
