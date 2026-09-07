"""The GACODE runtime and the NEO adapter (issue #550).

Everything here runs without GACODE installed. Executable resolution is checked
against launchable stubs written by `test/external_code_stubs.py`, which travel
exactly the path a real `neo` does, and the parsers are checked against stored
output from two real NEO runs in different regimes.

The single test that needs a real installation is gated on `$GACODEHOME`, in the
style `test_nubeam_adapter.py` established: the variable is read once at import,
because the autouse fixture below removes it before any test body runs.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from external_code_stubs import write_launchable_stub, write_unlaunchable_file

from vaft.code._executables import ExecutableNotLaunchable
from vaft.code.gacode import (
    GACODE_HOME_ENV,
    GACODEConfig,
    available_platforms,
    find_gacode_executable,
    gacode_environment,
    gacode_home,
    gacode_platform,
    launcher_relative_path,
    require_gacode_executable,
    run_gacode,
)
from vaft.code.gacode._profiles import GACODEProfile
from vaft.code.gacode.neo import (
    NEOConfig,
    NEOExecutionError,
    NeoOutputs,
    collect_neo_outputs,
    neo_parameters,
    prepare_neo_case,
    run_neo,
    write_input_neo,
)

FIXTURES = Path(__file__).parent / "data" / "gacode"
REG18 = FIXTURES / "neo_reg18"
VEST = FIXTURES / "neo_vest_48224"

#: Read at import: the autouse fixture removes it before any test body runs, and
#: the skipif below is evaluated here too, so this is the value it consulted.
INSTALLED_GACODE_HOME = os.environ.get(GACODE_HOME_ENV) or os.environ.get("GACODE_ROOT")
INSTALLED_GACODE_PLATFORM = os.environ.get("GACODE_PLATFORM")

_ENVIRONMENT = (GACODE_HOME_ENV, "GACODE_ROOT", "GACODE_PLATFORM")


@pytest.fixture(autouse=True)
def clear_gacode_environment(monkeypatch):
    for name in _ENVIRONMENT:
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def installation(tmp_path) -> Path:
    """A minimal tree with the shape the resolver documents."""
    root = tmp_path / "gacode"
    write_launchable_stub(root / launcher_relative_path("neo"))
    (root / "platform" / "build").mkdir(parents=True, exist_ok=True)
    (root / "platform" / "build" / "make.inc.GFORTRAN_OSX_BREW").write_text("")
    (root / "platform" / "build" / "make.inc.CI_CPU").write_text("")
    return root


# --------------------------------------------------------------------------
# Importability
# --------------------------------------------------------------------------


def test_importing_the_package_does_not_need_gacode():
    """The whole point of resolving lazily: `from vaft.code import *` must work."""
    import vaft.code
    import vaft.code.gacode
    import vaft.code.gacode.neo  # noqa: F401

    assert "gacode" in vaft.code.__all__
    assert gacode_home(GACODEConfig()) is None


def test_the_suite_layout_is_per_code_not_a_shared_bin():
    """GACODE gives every member its own bin, unlike every other adapter here."""
    assert launcher_relative_path("neo") == Path("neo") / "bin" / "neo"
    assert launcher_relative_path("TGLF") == Path("tglf") / "bin" / "tglf"


def test_a_name_outside_the_suite_is_refused():
    with pytest.raises(ValueError, match="not a GACODE suite member"):
        launcher_relative_path("tokamaker")


# --------------------------------------------------------------------------
# Executable and platform resolution
# --------------------------------------------------------------------------


def test_the_canonical_layout_resolves(monkeypatch, installation):
    monkeypatch.setenv(GACODE_HOME_ENV, str(installation))
    resolved = find_gacode_executable(GACODEConfig(), "neo")
    assert resolved == installation / "neo" / "bin" / "neo"


def test_gacode_root_is_accepted_as_a_compatibility_fallback(monkeypatch, installation):
    """VAFT adds $GACODEHOME; it does not take $GACODE_ROOT away."""
    monkeypatch.setenv("GACODE_ROOT", str(installation))
    assert gacode_home(GACODEConfig()) == installation
    assert find_gacode_executable(GACODEConfig(), "neo") is not None


def test_the_config_wins_over_the_environment(monkeypatch, installation, tmp_path):
    monkeypatch.setenv(GACODE_HOME_ENV, str(tmp_path / "elsewhere"))
    assert gacode_home(GACODEConfig(home=str(installation))) == installation


def test_an_unconfigured_installation_is_not_an_error_until_something_runs():
    assert find_gacode_executable(GACODEConfig(), "neo") is None


def test_requiring_an_unconfigured_installation_says_what_to_set():
    with pytest.raises(FileNotFoundError) as error:
        require_gacode_executable(GACODEConfig(), "neo")
    message = str(error.value)
    assert GACODE_HOME_ENV in message
    assert "neo/bin/neo" in message
    assert "$GACODE_ROOT" in message, "the compatibility variable must be named"


def test_a_configured_root_with_no_executable_names_the_expected_path(tmp_path):
    with pytest.raises(FileNotFoundError) as error:
        find_gacode_executable(GACODEConfig(home=str(tmp_path / "unbuilt")), "neo")
    assert str(tmp_path / "unbuilt" / "neo" / "bin" / "neo") in str(error.value)
    assert "Compile or install" in str(error.value)


@pytest.mark.skipif(os.name == "nt", reason="POSIX permission bits")
def test_a_present_but_unrunnable_launcher_is_a_permission_error(tmp_path):
    root = tmp_path / "gacode"
    write_unlaunchable_file(root / "neo" / "bin" / "neo")
    with pytest.raises(PermissionError):
        find_gacode_executable(GACODEConfig(home=str(root)), "neo")


def test_an_unset_platform_lists_what_the_installation_provides(installation):
    with pytest.raises(ValueError) as error:
        gacode_platform(GACODEConfig(home=str(installation)))
    message = str(error.value)
    assert "GACODE_PLATFORM" in message
    assert "GFORTRAN_OSX_BREW" in message and "CI_CPU" in message


def test_a_platform_the_installation_does_not_have_is_refused(installation):
    config = GACODEConfig(home=str(installation), platform="SUMMIT")
    with pytest.raises(ValueError, match="is not built in this installation"):
        gacode_platform(config)


def test_the_platform_comes_from_the_environment_when_the_config_omits_it(
    monkeypatch, installation
):
    monkeypatch.setenv("GACODE_PLATFORM", "CI_CPU")
    assert gacode_platform(GACODEConfig(home=str(installation))) == "CI_CPU"


def test_available_platforms_reads_the_build_directory(installation):
    assert available_platforms(installation) == ("CI_CPU", "GFORTRAN_OSX_BREW")


# --------------------------------------------------------------------------
# The subprocess environment
# --------------------------------------------------------------------------


def test_the_environment_sets_gacodes_own_variables(installation):
    config = GACODEConfig(home=str(installation), platform="CI_CPU")
    environment = gacode_environment(config, "neo")
    assert environment["GACODE_ROOT"] == str(installation)
    assert environment[GACODE_HOME_ENV] == str(installation)
    assert environment["GACODE_PLATFORM"] == "CI_CPU"


def test_pygacode_is_on_the_pythonpath(installation):
    """Without it the launcher's parse step fails and NEO blames input.neo.gen."""
    config = GACODEConfig(home=str(installation), platform="CI_CPU")
    entries = gacode_environment(config, "neo")["PYTHONPATH"].split(os.pathsep)
    assert str(installation / "f2py" / "pygacode") in entries


def test_path_and_pythonpath_are_prefixed_not_replaced(monkeypatch, installation):
    monkeypatch.setenv("PATH", "/sentinel/bin")
    monkeypatch.setenv("PYTHONPATH", "/sentinel/lib")
    config = GACODEConfig(home=str(installation), platform="CI_CPU")
    environment = gacode_environment(config, "neo")
    assert environment["PATH"].endswith("/sentinel/bin")
    assert environment["PYTHONPATH"].endswith("/sentinel/lib")
    assert str(installation / "shared" / "bin") in environment["PATH"]


def test_config_env_has_the_last_word(installation):
    config = GACODEConfig(
        home=str(installation), platform="CI_CPU", env={"GACODE_PLATFORM": "OVERRIDE"}
    )
    assert gacode_environment(config, "neo")["GACODE_PLATFORM"] == "OVERRIDE"


def test_a_launcher_the_system_refuses_to_start_is_reported_as_such(tmp_path):
    root = tmp_path / "gacode"
    (root / "platform" / "build").mkdir(parents=True)
    (root / "platform" / "build" / "make.inc.CI_CPU").write_text("")
    missing = root / "neo" / "bin" / "neo"
    missing.parent.mkdir(parents=True)
    with pytest.raises(ExecutableNotLaunchable):
        run_gacode(
            missing,
            ["-e", "case"],
            cwd=tmp_path,
            log_path=tmp_path / "neo.log",
            config=GACODEConfig(home=str(root), platform="CI_CPU"),
        )


def test_a_run_captures_its_log_and_returns_the_status(tmp_path, installation):
    stub = write_launchable_stub(installation / "neo" / "bin" / "neo", exit_code=3)
    config = GACODEConfig(home=str(installation), platform="CI_CPU")
    returncode, log = run_gacode(
        stub, [], cwd=tmp_path, log_path=tmp_path / "neo.log", config=config
    )
    assert returncode == 3
    assert log.is_file()


# --------------------------------------------------------------------------
# input.neo
# --------------------------------------------------------------------------


def _profile(n_ion: int = 1) -> GACODEProfile:
    rho = np.linspace(0.0, 1.0, 8)
    ones = np.ones((n_ion, rho.size))
    return GACODEProfile(
        rho=rho,
        z=np.arange(1, n_ion + 1, dtype=float),
        mass=np.full(n_ion, 2.0),
        name=tuple(f"i{i}" for i in range(n_ion)),
        rmin=np.linspace(0.0, 0.3, rho.size),
        rmaj=np.full(rho.size, 0.4),
        polflux=np.linspace(0.0, 0.05, rho.size),
        q=np.linspace(1.0, 3.0, rho.size),
        ne=np.linspace(1.0, 0.2, rho.size),
        te=np.linspace(1.0, 0.1, rho.size),
        ni=ones,
        ti=ones,
        torfluxa=0.02,
        rcentr=0.4,
        bcentr=0.15,
        current=0.1,
    )


def test_the_species_count_includes_electrons():
    assert neo_parameters(NEOConfig(), _profile(n_ion=2))["N_SPECIES"] == 3


def test_every_setting_is_written_rather_than_left_to_neos_default(tmp_path):
    """A file that omits a setting cannot later be told from a deliberate choice."""
    parameters = neo_parameters(NEOConfig(), _profile())
    text = write_input_neo(parameters, tmp_path / "input.neo").read_text()
    for key in ("N_ENERGY", "N_XI", "N_THETA", "COLLISION_MODEL", "PROFILE_MODEL",
                "ROTATION_MODEL", "N_SPECIES", "IPCCW", "BTCCW"):
        assert f"{key}=" in text


def test_extra_parameters_are_written_verbatim():
    config = NEOConfig(extra_parameters={"threed_model": 1})
    assert neo_parameters(config, _profile())["THREED_MODEL"] == 1


def test_input_neo_generation_is_deterministic(tmp_path):
    first = write_input_neo(neo_parameters(NEOConfig(), _profile()), tmp_path / "a")
    second = write_input_neo(neo_parameters(NEOConfig(), _profile()), tmp_path / "b")
    assert first.read_text() == second.read_text()


def test_a_species_count_that_forgets_the_electrons_is_refused():
    with pytest.raises(ValueError, match="counts electrons too"):
        NEOConfig(n_species=1)


@pytest.mark.parametrize("radius", [0.0, 1.0, 1.5, -0.2])
def test_a_surface_that_is_not_one_is_refused(radius):
    with pytest.raises(ValueError, match="rmin_over_a must lie in"):
        NEOConfig(rmin_over_a=radius)


def test_staging_writes_both_input_files(tmp_path):
    staged = prepare_neo_case(_profile(), tmp_path / "case")
    assert staged.input_gacode.is_file() and staged.input_neo.is_file()
    assert set(staged.files) == {staged.input_gacode, staged.input_neo}
    assert staged.provenance["n_ion"] == 1


def test_staging_refuses_a_profile_that_cannot_feed_profile_model_2(tmp_path):
    """PROFILE_MODEL=2 reads input.gacode, so what it needs must be there."""
    incomplete = GACODEProfile(rho=np.linspace(0, 1, 5), z=np.array([1.0]))
    with pytest.raises(ValueError, match="missing"):
        prepare_neo_case(incomplete, tmp_path / "case")


# --------------------------------------------------------------------------
# Parsing stored runs
# --------------------------------------------------------------------------


@pytest.mark.parametrize("directory", [REG18, VEST], ids=["reg18", "vest_48224"])
def test_a_stored_run_parses(directory):
    native = collect_neo_outputs(directory)
    assert native is not None
    assert native.grid is not None
    assert native.precision is not None
    assert native.version["platform"] == "GFORTRAN_OSX_BREW"


def test_the_theory_layout_holds_for_two_and_three_species():
    """The column layout is neo_theory.f90's, not pygacode's stale one.

    pygacode reads the per-species block three-wide; the writer emits two values
    per species and then two trailing scalars. Checking both species counts is
    what separates the two readings -- they differ by `n_species` columns.
    """
    for directory, species in ((REG18, 3), (VEST, 2)):
        native = collect_neo_outputs(directory)
        assert native.n_species == species
        assert np.shape(native.theory["hirshman_sigmar_particle_flux"]) == (species, 1)
        assert native.theory["redl_bootstrap_current"].shape == (1,)


def test_radial_and_species_dimensions_survive_parsing():
    native = collect_neo_outputs(REG18)
    assert np.shape(native.transport["energy_flux"]) == (3, 1)
    assert np.shape(native.equilibrium["density"]) == (3, 1)
    assert native.grid.theta.size == native.grid.n_theta


def test_the_normalisation_and_coordinate_bridges_are_kept():
    """These two files are what make a normalised result mean anything.

    `expnorm` carries the SI scales, `exprhon` the map from NEO's r/a back to
    rho_tor_norm and psi_norm, and so back into IMAS.
    """
    native = collect_neo_outputs(REG18)
    assert native.normalisation.a_meters[0] > 0.0
    assert native.normalisation.b_unit[0] > 0.0
    assert 0.0 < native.coordinates["rho_tor_norm"][0] < 1.0
    assert 0.0 < native.coordinates["psi_norm"][0] < 1.0


def test_an_unwritten_product_is_absent_not_zero():
    """The VEST fixture stores fewer products, and they read back as None."""
    native = collect_neo_outputs(VEST)
    assert "rotation" in native.missing()
    assert native.rotation is None
    # A product that *was* written and happens to be small stays a number.
    assert native.transport["potential_squared"] is not None


def test_a_directory_with_no_neo_output_reads_as_nothing(tmp_path):
    assert collect_neo_outputs(tmp_path) is None
    assert collect_neo_outputs(tmp_path / "absent") is None


def test_describe_names_the_normalisation():
    native = collect_neo_outputs(REG18)
    assert "n_0 v_t0" in native.describe("particle_flux")
    with pytest.raises(KeyError, match="not a NEO transport quantity"):
        native.describe("nonsense")


def test_the_native_result_round_trips_through_json(tmp_path):
    native = collect_neo_outputs(REG18)
    path = native.write_json(tmp_path / "native.json")
    reloaded = NeoOutputs.read_json(path)
    np.testing.assert_allclose(reloaded.bootstrap_current, native.bootstrap_current)
    np.testing.assert_allclose(reloaded.grid.theta, native.grid.theta)
    np.testing.assert_allclose(
        reloaded.theory["sauter_bootstrap_current"],
        native.theory["sauter_bootstrap_current"],
    )
    assert reloaded.version == native.version
    assert reloaded.missing() == native.missing()


def test_a_payload_from_a_newer_schema_is_refused():
    payload = collect_neo_outputs(REG18).to_dict()
    payload["schema_version"] = 99
    with pytest.raises(ValueError, match="schema version 99"):
        NeoOutputs.from_dict(payload)


def test_the_stored_payload_names_its_schema():
    payload = collect_neo_outputs(REG18).to_dict()
    assert payload["schema"] == "vaft.code.gacode.neo.NeoOutputs"


# --------------------------------------------------------------------------
# Running
# --------------------------------------------------------------------------


def test_a_run_that_produces_nothing_fails_even_with_a_zero_exit(tmp_path, installation):
    """NEO can exit cleanly having written nothing usable, so status is not enough."""
    write_launchable_stub(installation / "neo" / "bin" / "neo", exit_code=0)
    config = NEOConfig(home=str(installation), platform="CI_CPU")
    staged = prepare_neo_case(_profile(), tmp_path / "case", config)
    with pytest.raises(NEOExecutionError, match="wrote no readable output"):
        run_neo(staged, config)


def test_a_failed_run_is_returned_rather_than_raised_when_asked(tmp_path, installation):
    write_launchable_stub(installation / "neo" / "bin" / "neo", exit_code=2)
    config = NEOConfig(home=str(installation), platform="CI_CPU")
    staged = prepare_neo_case(_profile(), tmp_path / "case", config)
    result = run_neo(staged, config, check=False)
    assert result.returncode == 2 and not result.ok
    assert result.logs and result.logs[0].is_file()


def test_the_run_records_what_produced_it(tmp_path, installation):
    write_launchable_stub(installation / "neo" / "bin" / "neo", exit_code=1)
    config = NEOConfig(home=str(installation), platform="CI_CPU")
    staged = prepare_neo_case(_profile(), tmp_path / "case", config)
    result = run_neo(staged, config, check=False)
    assert result.provenance["platform"] == "CI_CPU"
    assert result.provenance["parameters"]["N_SPECIES"] == 2
    assert Path(result.provenance["executable"]).name.startswith("neo")


@pytest.mark.skipif(
    not INSTALLED_GACODE_HOME, reason="NEO integration test requires $GACODEHOME"
)
def test_an_installed_neo_reproduces_the_reg18_regression_case(tmp_path):
    """The shipped reg18 case, run through the VAFT adapter, end to end.

    `out.neo.prec` is what GACODE's own `neo -rc` compares, so reproducing it is
    reproducing the regression.
    """
    from vaft.code.gacode._input_gacode import read_input_gacode
    from vaft.code.gacode.neo import run_neo_case

    profile = read_input_gacode(REG18 / "input.gacode")
    config = NEOConfig(
        home=INSTALLED_GACODE_HOME,
        platform=INSTALLED_GACODE_PLATFORM,
        n_species=3,
        rotation_model=2,
    )
    result = run_neo_case(profile, tmp_path / "reg18", config)
    assert result.ok
    expected = float((REG18 / "out.neo.prec").read_text().split()[0])
    assert result.outputs_native.precision == pytest.approx(expected, rel=1e-7)


@pytest.mark.parametrize(
    ("filename", "message"),
    [
        ("out.neo.transport", "out.neo.transport has"),
        ("out.neo.equil", "out.neo.equil has"),
        ("out.neo.theory", "columns but 3 species imply"),
    ],
)
def test_a_table_of_the_wrong_width_is_refused_not_strided_over(tmp_path, filename, message):
    """A stride over a mis-shaped table mis-assigns species instead of failing.

    Silently attributing one species' flux to another is worse than a parse
    error, so the width is checked against the species count first.
    """
    import shutil

    case = tmp_path / "case"
    shutil.copytree(REG18, case)
    values = (case / filename).read_text().split()
    (case / filename).write_text(" ".join(values[:-1]) + "\n")
    with pytest.raises(ValueError, match=message):
        collect_neo_outputs(case)
