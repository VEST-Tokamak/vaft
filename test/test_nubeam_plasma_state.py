"""Plasma State specification for VAFT's NUBEAM generator.

Runs without NUBEAM: the generator itself is only exercised by the
integration test at the end, which skips unless ``$NUBEAMHOME`` holds a build
with ``bin/vaft_plasma_state``.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import numpy as np
import pytest

from vaft.code import nubeam
from vaft.code.nubeam import plasma_state as ps
from vaft.compat import short_temporary_directory

INSTALLED_NUBEAM_HOME = os.environ.get("NUBEAMHOME")


@pytest.fixture
def vest_case() -> nubeam.NUBEAMCase:
    return nubeam.packaged_vest_case()


def _profiles(n: int = 5, **overrides) -> ps.PlasmaStateProfiles:
    x = np.linspace(0.0, 1.0, n)
    values = dict(
        x=x,
        ne=np.full(n, 1e19),
        te_kev=np.full(n, 0.02),
        ti_kev=np.full(n, 0.01),
        zeff=np.full(n, 1.5),
        ion_densities=(np.full(n, 9e18), np.full(n, 1e19 / 60)),
    )
    values.update(overrides)
    return ps.PlasmaStateProfiles(**values)


# --------------------------------------------------------------------------
# The legacy VEST case
# --------------------------------------------------------------------------


def test_the_packaged_profiles_file_reads_as_the_state_it_produced(vest_case):
    """Values checked against the Plasma State the case is validated with."""
    legacy = ps.read_legacy_profiles(vest_case.input_dir / "profiles")
    profiles = legacy.profiles
    assert profiles.x.size == 101 and profiles.coordinate == "rho_tor"
    assert profiles.ne[0] == pytest.approx(2e19)
    assert profiles.ne[-1] == pytest.approx(1e18)
    assert profiles.te_kev[0] == pytest.approx(0.02)
    assert np.allclose(profiles.ti_kev, 0.01)
    assert np.allclose(profiles.zeff, 1.5)
    hydrogen, carbon = profiles.ion_densities
    assert hydrogen[-1] == pytest.approx(9e17)
    assert carbon[-1] == pytest.approx(1e18 / 60)
    assert legacy.beam_power_w == (200000.0,)
    assert legacy.beam_energy_kev == (10.0,)
    assert legacy.edge_neutral_density == pytest.approx(6e16)
    assert legacy.neutral_energy_kev == (0.00015,)


def test_a_non_zero_unmapped_block_is_refused_not_dropped(vest_case, tmp_path):
    lines = (vest_case.input_dir / "profiles").read_text(encoding="utf-8").splitlines()
    lines[4] = "1.0"  # first value of the block VAFT does not map
    path = tmp_path / "profiles"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with pytest.raises(ps.PlasmaStateInputError, match="refuses the file"):
        ps.read_legacy_profiles(path)


def test_a_truncated_profiles_file_says_where_it_ended(vest_case, tmp_path):
    lines = (vest_case.input_dir / "profiles").read_text(encoding="utf-8").splitlines()
    path = tmp_path / "profiles"
    path.write_text("\n".join(lines[:300]) + "\n", encoding="utf-8")
    with pytest.raises(ps.PlasmaStateInputError, match="ends early"):
        ps.read_legacy_profiles(path)


def test_the_legacy_spec_takes_names_and_times_from_inputf(vest_case, tmp_path):
    for item in vest_case.input_dir.iterdir():
        shutil.copy2(item, tmp_path / item.name)
    spec = ps.legacy_case_spec(tmp_path, gfile=tmp_path / "g020000.015100")
    assert (spec.t0, spec.t1) == (0.0, 0.005)
    assert spec.output_name == "FUSMA_NUBEAM.cdf"
    assert spec.runid == "FUSMA_NUBEAM"
    assert spec.mdescr.name == "mdescr_VEST_190307.dat"
    assert spec.sconfig.name == "sconfig_VEST_190307.dat"
    ps.check_spec_against_namelists(spec)


def test_the_vest_shot_configuration(vest_case):
    config = ps.read_shot_configuration(vest_case.input_dir / "sconfig_VEST_190307.dat")
    assert config.ion_charge_numbers == (1, 6)
    assert config.ion_mass_numbers == (1, 12)
    assert config.gas_sources == ("H0rcy",)
    assert ps.count_beams(vest_case.input_dir / "mdescr_VEST_190307.dat") == 1


# --------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------


def test_profiles_must_span_the_whole_plasma():
    with pytest.raises(ps.PlasmaStateInputError, match="does not extrapolate"):
        _profiles(x=np.linspace(0.0, 0.9, 5))


@pytest.mark.parametrize("name", ["ne", "te_kev", "ti_kev"])
def test_non_positive_kinetic_profiles_are_refused(name):
    values = np.full(5, 1.0)
    values[-1] = 0.0
    with pytest.raises(ps.PlasmaStateInputError, match="positive"):
        _profiles(**{name: values})


def test_a_profile_on_another_grid_is_refused():
    with pytest.raises(ps.PlasmaStateInputError, match="has 4 points"):
        _profiles(ne=np.full(4, 1e19))


def test_an_unknown_coordinate_is_refused():
    with pytest.raises(ps.PlasmaStateInputError, match="coordinate"):
        _profiles(coordinate="rho_pol")


def _spec(vest_case, **overrides) -> ps.PlasmaStateSpec:
    values = dict(
        mdescr=vest_case.input_dir / "mdescr_VEST_190307.dat",
        sconfig=vest_case.input_dir / "sconfig_VEST_190307.dat",
        gfile=vest_case.gfile,
        profiles=_profiles(),
        beam_power_w=(2e5,),
        beam_energy_kev=(10.0,),
        neutral_energy_kev=(0.005,),
    )
    values.update(overrides)
    return ps.PlasmaStateSpec(**values)


def test_counts_are_checked_against_the_machine_description(vest_case):
    ps.check_spec_against_namelists(_spec(vest_case))
    with pytest.raises(ps.PlasmaStateInputError, match="beam source"):
        ps.check_spec_against_namelists(
            _spec(vest_case, beam_power_w=(1.0, 1.0), beam_energy_kev=(1.0, 1.0))
        )
    with pytest.raises(ps.PlasmaStateInputError, match="thermal ion"):
        ps.check_spec_against_namelists(
            _spec(vest_case, profiles=_profiles(ion_densities=(np.full(5, 1e19),)))
        )
    with pytest.raises(ps.PlasmaStateInputError, match="neutral gas"):
        ps.check_spec_against_namelists(_spec(vest_case, neutral_energy_kev=()))


def test_bdy_crat_outside_what_ntcc_accepts_is_refused(vest_case):
    with pytest.raises(ps.PlasmaStateInputError, match="bdy_crat"):
        _spec(vest_case, bdy_crat=0.2)


@pytest.mark.parametrize(
    "form, text",
    [
        ("comma", "iZatom_S(1) = 1, 6\niAMU_S(1) = 1, 12\n"),
        ("blank-separated", "iZatom_S = 1 6\niAMU_S = 1 12\n"),
        ("repeat count", "iZatom_S = 1, 6\niAMU_S = 2*12\n"),
        ("two per line", "iZatom_S(1)=1,6 iAMU_S(1)=1,12\n"),
        ("per index", "iZatom_S(1)=1\niZatom_S(2)=6\niAMU_S(1)=1\niAMU_S(2)=12\n"),
    ],
)
def test_fortran_namelist_value_forms_are_read(tmp_path, form, text):
    """Blank separators, ``r*value`` repeats and several assignments per
    record are legal namelist input that NTCC's reader accepts."""
    path = tmp_path / "sconfig.dat"
    path.write_text("&sconfig\n" + text + "/\n", encoding="utf-8")
    config = ps.read_shot_configuration(path)
    assert config.ion_charge_numbers == (1, 6)
    assert config.ion_mass_numbers == ((12, 12) if form == "repeat count" else (1, 12))


def test_an_unreadable_namelist_value_names_the_file_line_and_token(tmp_path):
    path = tmp_path / "sconfig.dat"
    path.write_text("&sconfig\niZatom_S(1) = 1, 6\niAMU_S(1) = 1, twelve\n/\n", encoding="utf-8")
    with pytest.raises(ps.PlasmaStateInputError, match=r"sconfig\.dat:3: cannot read iAMU_S value 'twelve'"):
        ps.read_shot_configuration(path)


def test_the_error_is_a_nubeam_input_error():
    assert issubclass(ps.PlasmaStateInputError, nubeam.NUBEAMInputError)


# --------------------------------------------------------------------------
# Zeff
# --------------------------------------------------------------------------


def test_ion_densities_from_zeff_reproduce_the_vest_case():
    ne = np.array([2e19, 1e18])
    hydrogen, carbon = ps.ion_densities_from_zeff(ne, np.full(2, 1.5), (1, 6))
    assert np.allclose(hydrogen, 0.9 * ne)
    assert np.allclose(carbon, ne / 60)
    # Quasi-neutrality and the Zeff definition both hold.
    assert np.allclose(hydrogen + 6 * carbon, ne)
    assert np.allclose(hydrogen + 36 * carbon, 1.5 * ne)


def test_a_zeff_the_species_cannot_make_is_refused():
    with pytest.raises(ps.PlasmaStateInputError, match="outside"):
        ps.ion_densities_from_zeff(np.full(2, 1e19), np.full(2, 7.0), (1, 6))
    with pytest.raises(ps.PlasmaStateInputError, match="exactly two"):
        ps.ion_densities_from_zeff(np.full(2, 1e19), np.full(2, 1.5), (1,))


# --------------------------------------------------------------------------
# The namelist
# --------------------------------------------------------------------------


def _parse_namelist(text: str) -> dict[str, str]:
    body = text.strip().splitlines()
    assert body[0] == "&vaft_plasma_state" and body[-1] == "/"
    entries: dict[str, str] = {}
    for line in body[1:-1]:
        for part in line.split(", ") if line.count("=") > 1 else [line]:
            key, value = part.split("=", 1)
            entries[key.strip()] = value.strip()
    return entries


def test_the_namelist_carries_every_input(vest_case, tmp_path):
    spec = _spec(vest_case, runid="RUN1", output_name="RUN1.cdf", t1=0.005)
    entries = _parse_namelist(ps.render_plasma_state_namelist(spec, tmp_path))
    assert entries["runid"] == "'RUN1'"
    assert entries["output_file"] == "'RUN1.cdf'"
    assert entries["nx"] == "5" and entries["nion"] == "2" and entries["nbeam"] == "1"
    assert entries["x_coordinate"] == "'rho_tor'"
    assert entries["power_nbi"] == "200000.0"
    assert entries["is_recycling"] == "1"
    assert "ni(1:5,2)" in entries
    assert entries["mdescr_file"].endswith("mdescr_VEST_190307.dat'")


def test_files_inside_the_workdir_are_named_relative_to_it(vest_case, tmp_path):
    shutil.copy2(vest_case.input_dir / "mdescr_VEST_190307.dat", tmp_path)
    spec = _spec(vest_case, mdescr=tmp_path / "mdescr_VEST_190307.dat")
    entries = _parse_namelist(ps.render_plasma_state_namelist(spec, tmp_path))
    assert entries["mdescr_file"] == "'mdescr_VEST_190307.dat'"


def test_a_quote_cannot_break_out_of_a_namelist_string(vest_case, tmp_path):
    spec = _spec(vest_case, runid="a'b")
    with pytest.raises(ps.PlasmaStateInputError, match="quotes"):
        ps.render_plasma_state_namelist(spec, tmp_path)


# --------------------------------------------------------------------------
# core_profiles
# --------------------------------------------------------------------------


def _ods(*, with_ions: bool = True, span: float = 1.0) -> dict:
    """A plain nested mapping shaped like the IDS paths the reader follows."""
    psi = np.linspace(0.0, span, 21) * 0.02 + 0.1  # axis 0.1, boundary 0.12
    profile = {
        "grid": {"psi": psi},
        "electrons": {
            "density_thermal": np.linspace(2e19, 1e18, 21),
            "temperature": np.linspace(20.0, 10.0, 21),
        },
        "zeff": np.full(21, 1.5),
        "ion": {},
    }
    if with_ions:
        profile["ion"] = {
            0: {
                "element": {0: {"z_n": 1.0, "a": 1.0}},
                "density_thermal": np.linspace(1.8e19, 9e17, 21),
                "temperature": np.full(21, 10.0),
            },
            1: {
                "element": {0: {"z_n": 6.0, "a": 12.0}},
                "density_thermal": np.linspace(2e19, 1e18, 21) / 60,
                "temperature": np.full(21, 10.0),
            },
        }
    else:
        profile["ion"] = {0: {"element": {0: {"z_n": 1.0, "a": 1.0}}, "temperature": np.full(21, 10.0)}}
    return {
        "core_profiles": {"time": np.array([0.3]), "profiles_1d": {0: profile}},
        "equilibrium": {
            "time": np.array([0.3]),
            "time_slice": {0: {"global_quantities": {"psi_axis": 0.1, "psi_boundary": 0.12}}},
        },
    }


def test_core_profiles_map_onto_sqrt_psi_n_in_kev():
    profiles, provenance = ps.profiles_from_core_profiles(
        _ods(), time=0.3, equilibrium_index=0, ion_charges=(1, 6), ion_masses=(1, 12)
    )
    assert profiles.coordinate == "sqrt_psi_n"
    assert profiles.te_kev[0] == pytest.approx(0.02)
    assert profiles.ti_kev[-1] == pytest.approx(0.01)
    assert profiles.ion_densities[1][0] == pytest.approx(2e19 / 60)
    assert provenance["ne"].endswith("electrons.density_thermal")
    # The abscissa is sqrt(psi_N): the grid point at psi_N = 0.25 is x = 0.5.
    middle = np.searchsorted(profiles.x, 0.5)
    assert profiles.ne[middle] == pytest.approx(np.interp(0.25, np.linspace(0, 1, 21), np.linspace(2e19, 1e18, 21)))


def test_missing_ion_densities_are_derived_from_zeff():
    profiles, provenance = ps.profiles_from_core_profiles(
        _ods(with_ions=False), time=0.3, equilibrium_index=0, ion_charges=(1, 6), ion_masses=(1, 12)
    )
    assert provenance["ion_densities"] == "derived from ne and Zeff"
    assert np.allclose(profiles.ion_densities[0], 0.9 * profiles.ne)


def test_profiles_that_stop_short_of_the_edge_are_refused():
    with pytest.raises(ps.PlasmaStateInputError, match="does not extrapolate"):
        ps.profiles_from_core_profiles(
            _ods(span=0.8), time=0.3, equilibrium_index=0, ion_charges=(1, 6), ion_masses=(1, 12)
        )


def test_a_time_with_no_profiles_is_refused():
    with pytest.raises(ps.PlasmaStateInputError, match="no sample within"):
        ps.profiles_from_core_profiles(
            _ods(), time=0.5, equilibrium_index=0, ion_charges=(1, 6), ion_masses=(1, 12)
        )


# --------------------------------------------------------------------------
# Staging for a run built from data
# --------------------------------------------------------------------------


def test_staging_namelists_reads_the_state_nubeam_will_open(vest_case, tmp_path):
    with short_temporary_directory(max_length=60) as scratch:
        staged = nubeam.stage_nubeam_namelists(vest_case.input_dir, workdir=scratch / "run")
        assert staged.state_name == "FUSMA_NUBEAM.cdf"
        assert staged.mdescr.is_file() and staged.sconfig.is_file()
        assert not (staged.workdir / "inputf").exists()


def test_init_and_step_must_read_the_same_state(vest_case, tmp_path):
    case = tmp_path / "case"
    shutil.copytree(vest_case.input_dir, case)
    step = case / "nubeam_step_files.dat"
    step.write_text(step.read_text().replace("FUSMA_NUBEAM.cdf", "OTHER.cdf"))
    with short_temporary_directory(max_length=60) as scratch:
        with pytest.raises(nubeam.NUBEAMInputError, match="different Plasma States"):
            nubeam.stage_nubeam_namelists(case, workdir=scratch / "run")


# --------------------------------------------------------------------------
# Integration
# --------------------------------------------------------------------------


@pytest.mark.skipif(
    not INSTALLED_NUBEAM_HOME
    or not (Path(INSTALLED_NUBEAM_HOME) / "bin" / "vaft_plasma_state").exists(),
    reason="needs $NUBEAMHOME with bin/vaft_plasma_state",
)
def test_the_packaged_case_builds_a_state(vest_case, monkeypatch):
    xr = pytest.importorskip("xarray")
    monkeypatch.setenv("NUBEAMHOME", INSTALLED_NUBEAM_HOME)
    with short_temporary_directory(max_length=60) as scratch:
        inputs = nubeam.prepare_nubeam_inputs(
            vest_case.input_dir, gfile=vest_case.gfile, workdir=scratch / "run"
        )
        state = nubeam.generate_plasma_state(inputs)
        with xr.open_dataset(state, decode_times=False) as data:
            assert str(data["tokamak_id"].values.astype(str)).strip() == "VEST"
            assert data["power_nbi"].values[0] == pytest.approx(2e5)
            assert data["ns"].shape == (3, 100)
