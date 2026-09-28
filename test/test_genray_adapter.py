"""GENRAY EC ray-tracing adapter (#264), without GENRAY.

The launch-angle conversion, the input file, the refusals, the netCDF ->
``waves`` mapping and the run contract are all tested here with synthetic
data and a stand-in backend. The real executable is exercised only by
``test_genray_integration.py``, which skips without ``$GENRAYHOME``.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest
from omas import ODS

from vaft.code.execution import ExecutionResult
from vaft.code.genray import (
    GENRAYConfig,
    eccone_angles,
    find_genray_executable,
    genray_to_waves,
    prepare_genray_inputs,
    read_genray_netcdf,
    run,
)

netCDF4 = pytest.importorskip("netCDF4")


# --------------------------------------------------------------------------- #
# launch angles
# --------------------------------------------------------------------------- #


def _genray_direction(alfast_deg, betast_deg, phist_deg):
    """GENRAY cone_ec.f: the Cartesian launch direction."""
    a, b, p = map(math.radians, (alfast_deg, betast_deg, phist_deg))
    return np.array([math.cos(b) * math.cos(a + p), math.cos(b) * math.sin(a + p), math.sin(b)])


def _steering_angles(k):
    """IMAS DD 3.41: angle_pol = atan2(-k_Z, -k_R), angle_tor = arcsin(k_phi / |k|)."""
    k_r, k_phi, k_z = np.asarray(k, dtype=float) / np.linalg.norm(k)
    return math.atan2(-k_z, -k_r), math.asin(k_phi)


def _imas_direction(k_rpz, phi):
    """Cartesian direction of an (R, phi, Z) vector at toroidal angle phi (IMAS, right-handed)."""
    k_r, k_phi, k_z = k_rpz
    return np.array([k_r * math.cos(phi) - k_phi * math.sin(phi), k_r * math.sin(phi) + k_phi * math.cos(phi), k_z])


@pytest.mark.parametrize(
    "k_rpz",
    [(-1.0, 0.0, 0.0), (-1.0, 0.0, -0.3), (-1.0, 0.4, 0.2), (-0.5, -0.8, 0.1), (0.2, 0.9, -0.3)],
)
@pytest.mark.parametrize("phi_deg", [0.0, 210.0, -35.0])
def test_eccone_angles_launch_along_the_imas_wave_vector(k_rpz, phi_deg):
    k = np.asarray(k_rpz) / np.linalg.norm(k_rpz)
    angle_pol, angle_tor = _steering_angles(k)

    alfast, betast = eccone_angles(angle_pol, angle_tor)

    np.testing.assert_allclose(
        _genray_direction(alfast, betast, phi_deg), _imas_direction(k, math.radians(phi_deg)), atol=1e-12
    )


def test_a_radial_launch_is_alfast_180_betast_0():
    assert eccone_angles(0.0, 0.0) == pytest.approx((180.0, 0.0))


# --------------------------------------------------------------------------- #
# configuration
# --------------------------------------------------------------------------- #


def test_the_mode_is_required_and_spelled_o_or_x():
    with pytest.raises(TypeError):
        GENRAYConfig(time=0.3)  # type: ignore[call-arg]
    with pytest.raises(ValueError):
        GENRAYConfig(mode="R", time=0.3)
    assert GENRAYConfig(mode="O", time=0.3).ioxm == 1
    assert GENRAYConfig(mode="X", time=0.3).ioxm == -1


@pytest.mark.parametrize(
    "kwargs",
    [{"power_w": 0.0}, {"zeff": 0.5}, {"n_rho": 2}, {"harmonic": 0}, {"minimum_temperature_ev": 0.0},
     {"time_tolerance": -1.0}],
)
def test_nonsense_configuration_is_refused(kwargs):
    with pytest.raises(ValueError):
        GENRAYConfig(mode="X", time=0.3, **kwargs)


# --------------------------------------------------------------------------- #
# inputs
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def vest_48224():
    import vaft.data
    import vaft.omas

    return vaft.omas.load(vaft.data.sample(48224))


def _with_launcher(ods, *, time=0.3, power=None):
    ods = ods.copy()
    beam = "ec_launchers.beam.0"
    ods["ec_launchers.ids_properties.homogeneous_time"] = 0
    ods[f"{beam}.name"] = "ECH 6 kW 2.45 GHz"
    ods[f"{beam}.time"] = np.array([time])
    for path, value in {
        "launching_position.r": 0.8021,
        "launching_position.z": -0.36,
        "launching_position.phi": math.radians(210.0),
        "steering_angle_pol": 0.0,
        "steering_angle_tor": 0.0,
    }.items():
        ods[f"{beam}.{path}"] = np.array([value])
    ods[f"{beam}.frequency.time"] = np.array([time])
    ods[f"{beam}.frequency.data"] = np.array([2.45e9])
    if power is not None:
        ods[f"{beam}.power_launched.time"] = np.array([time])
        ods[f"{beam}.power_launched.data"] = np.array([power])
    return ods


def _config(**overrides):
    base = {"mode": "X", "time": 0.3, "power_w": 3000.0, "zeff": 2.0, "minimum_temperature_ev": 1.0}
    base.update(overrides)
    return GENRAYConfig(**base)


def _namelist_value(text, group, name):
    body = text.split(f" &{group}\n", 1)[1].split(" &end", 1)[0]
    for line in body.splitlines():
        key, _, value = line.strip().partition("=")
        if key == name:
            return value
    raise KeyError(name)


def _number(value):
    return float(value.replace("d", "e"))


def test_inputs_carry_the_launcher_in_genray_units(vest_48224, tmp_path):
    inputs = prepare_genray_inputs(_with_launcher(vest_48224), _config(), tmp_path)
    text = inputs.genray_in.read_text()

    assert _number(_namelist_value(text, "wave", "frqncy")) == pytest.approx(2.45)  # GHz
    assert int(_namelist_value(text, "wave", "ioxm")) == -1
    assert _number(_namelist_value(text, "eccone", "powtot")) == pytest.approx(3.0e-3)  # MW
    assert _number(_namelist_value(text, "eccone", "rst")) == pytest.approx(0.8021)
    assert _number(_namelist_value(text, "eccone", "zst")) == pytest.approx(-0.36)
    assert _number(_namelist_value(text, "eccone", "phist")) == pytest.approx(210.0)
    assert _number(_namelist_value(text, "eccone", "alfast")) == pytest.approx(180.0)
    assert _number(_namelist_value(text, "eccone", "betast")) == pytest.approx(0.0)
    assert int(_namelist_value(text, "tokamak", "indexrho")) == 4
    assert inputs.eqdsk.is_file()
    record = json.loads((tmp_path / "vaft_genray_inputs.json").read_text())
    assert record["equilibrium_time_s"] == pytest.approx(0.3)
    assert record["core_profiles_time_s"] == pytest.approx(0.3)
    assert record["temperature_points_floored"] >= 1  # the fit pins Te = 0 at the separatrix


def test_profiles_are_tabulated_on_sqrt_psi_n_in_1e19(vest_48224, tmp_path):
    inputs = prepare_genray_inputs(_with_launcher(vest_48224), _config(n_rho=11), tmp_path)
    text = inputs.genray_in.read_text()
    density = [_number(v) for v in _namelist_value(text, "dentab", "prof").split(",")]
    temperature = [_number(v) for v in _namelist_value(text, "temtab", "prof").split(",")]

    assert len(density) == len(temperature) == 11
    # 48224 at 0.3 s: ne(axis) = 1.02e19 m^-3, Te ~ 0.1 keV mid-radius.
    assert density[0] == pytest.approx(1.017, rel=1e-3)
    assert 0.05 < max(temperature) < 0.2
    assert temperature[-1] == pytest.approx(1.0e-3)  # the explicit 1 eV floor


def test_a_zero_edge_temperature_is_refused_without_an_explicit_floor(vest_48224, tmp_path):
    with pytest.raises(ValueError, match="minimum_temperature_ev"):
        prepare_genray_inputs(_with_launcher(vest_48224), _config(minimum_temperature_ev=None), tmp_path)


def test_a_time_mismatch_beyond_tolerance_is_refused(vest_48224, tmp_path):
    with pytest.raises(ValueError, match="within"):
        prepare_genray_inputs(_with_launcher(vest_48224), _config(time=0.31), tmp_path)


def test_launched_power_comes_from_the_ods_or_is_refused(vest_48224, tmp_path):
    with pytest.raises(ValueError):  # no power_launched and no power_w
        prepare_genray_inputs(_with_launcher(vest_48224), _config(power_w=None), tmp_path / "a")
    with pytest.raises(ValueError, match="power_w"):  # NaN is not a power
        prepare_genray_inputs(_with_launcher(vest_48224, power=float("nan")), _config(power_w=None), tmp_path / "b")
    inputs = prepare_genray_inputs(_with_launcher(vest_48224, power=2500.0), _config(power_w=None), tmp_path / "c")
    assert inputs.provenance["launcher"]["power_w"] == pytest.approx(2500.0)
    assert "power_launched" in inputs.provenance["launcher"]["power_source"]


def test_missing_zeff_is_refused(vest_48224, tmp_path):
    with pytest.raises(ValueError, match="zeff"):
        prepare_genray_inputs(_with_launcher(vest_48224), _config(zeff=None), tmp_path)


def test_overrides_are_applied_last_and_recorded(vest_48224, tmp_path):
    config = _config(namelist_overrides={"numercl": {"prmt6": 5.0e-4}})
    inputs = prepare_genray_inputs(_with_launcher(vest_48224), config, tmp_path)
    assert _number(_namelist_value(inputs.genray_in.read_text(), "numercl", "prmt6")) == pytest.approx(5.0e-4)
    assert inputs.provenance["namelist_overrides"] == {"numercl": {"prmt6": 5.0e-4}}


# --------------------------------------------------------------------------- #
# outputs
# --------------------------------------------------------------------------- #


def _write_genray_nc(path: Path, *, frequency=2.45e9, rays=None, status=None, totals=True, refl_loss=None):
    """A minimal genray.nc in GENRAY's CGS units."""
    rays = rays or [
        {  # a ray going straight in from R = 80 cm at z = -36 cm, losing half its power
            "wr": [80.0, 70.0, 60.0], "wz": [-36.0, -36.0, -36.0], "wphi": [0.0, 0.0, 0.0],
            "delpwr": [3.0e10, 2.0e10, 1.5e10],
            "wn_r": [-1.0, -0.8, -0.5], "wn_z": [0.0, 0.0, 0.0], "wn_phi": [0.0, 0.1, 0.2],
            "wnpar": [0.0, 0.1, 0.2], "wnper": [1.0, 0.8, 0.5],
        }
    ]
    neltmax = max(len(r["wr"]) for r in rays)
    with netCDF4.Dataset(str(path), "w") as data:
        data.createDimension("nrays", len(rays))
        data.createDimension("neltmax", neltmax)
        data.createDimension("char64dim", 64)
        version = data.createVariable("version", "S1", ("char64dim",))
        version[:] = np.array(list("genray_test".ljust(64)), dtype="S1")
        data.createVariable("freqcy", "f8")[...] = frequency
        data.createVariable("ioxm", "i4")[...] = -1
        data.createVariable("nrayelt", "i4", ("nrays",))[:] = [len(r["wr"]) for r in rays]
        for name in ("wr", "wz", "wphi", "delpwr", "wn_r", "wn_z", "wn_phi", "wnpar", "wnper"):
            values = np.zeros((len(rays), neltmax))
            for i, ray in enumerate(rays):
                values[i, : len(ray[name])] = ray[name]
            data.createVariable(name, "f8", ("nrays", "neltmax"))[:] = values
        if totals:
            data.createVariable("power_inj_total", "f8")[...] = 3.0e10
            data.createVariable("power_total", "f8")[...] = 1.5e10
            data.createVariable("powtot_e", "f8")[...] = 1.5e10
            data.createDimension("nfreq", 1)
            codes = status if status is not None else [3] * len(rays)
            data.createVariable("iray_status_nc", "i4", ("nfreq", "nrays"))[:] = [codes]
        if refl_loss is not None:
            data.createVariable("refl_loss", "f8")[...] = refl_loss
    return path


def test_netcdf_is_read_in_si(tmp_path):
    parsed = read_genray_netcdf(_write_genray_nc(tmp_path / "genray.nc"))
    ray = parsed["rays"][0]

    np.testing.assert_allclose(ray["r"], [0.8, 0.7, 0.6])
    np.testing.assert_allclose(ray["z"], [-0.36] * 3)
    np.testing.assert_allclose(ray["power"], [3000.0, 2000.0, 1500.0])  # erg/s -> W
    np.testing.assert_allclose(ray["length"], [0.0, 0.1, 0.2])  # arc length, m
    assert parsed["power_absorbed_w"] == pytest.approx(1500.0)
    assert parsed["version"] == "genray_test"


def test_rays_map_into_waves_without_invented_quantities(tmp_path):
    parsed = read_genray_netcdf(_write_genray_nc(tmp_path / "genray.nc"))
    ods = ODS()

    genray_to_waves(parsed, ods, time=0.3, beam_name="ECH 6 kW 2.45 GHz", provenance={"mode": "X"})

    wave = ods["waves.coherent_wave.0"]
    assert wave["identifier.type.name"] == "EC" and wave["identifier.type.index"] == 1
    beam = wave["beam_tracing.0.beam.0"]
    assert beam["power_initial"] == pytest.approx(3000.0)
    np.testing.assert_allclose(beam["electrons.power"], [0.0, 1000.0, 1500.0])
    k0 = 2 * math.pi * 2.45e9 / 2.99792458e8
    np.testing.assert_allclose(beam["wave_vector.k_r"], np.array([-1.0, -0.8, -0.5]) * k0)
    np.testing.assert_allclose(beam["wave_vector.k_tor"], np.array([0.0, 0.1, 0.2]) * k0)
    for absent in ("spot", "phase", "e_field"):
        assert absent not in beam
    assert wave["global_quantities.0.power"] == pytest.approx(1500.0)
    parameters = json.loads(ods["waves.code.parameters"])
    assert parameters["inputs"] == {"mode": "X"}
    assert ods["waves.code.name"] == "GENRAY"


def test_mapped_waves_round_trip_through_json(tmp_path):
    ods = ODS()
    genray_to_waves(read_genray_netcdf(_write_genray_nc(tmp_path / "genray.nc")), ods, time=0.3)
    path = tmp_path / "waves.json"
    ods.save(str(path))

    reloaded = ODS().load(str(path), consistency_check=True)

    np.testing.assert_allclose(
        reloaded["waves.coherent_wave.0.beam_tracing.0.beam.0.position.r"], [0.8, 0.7, 0.6]
    )


# --------------------------------------------------------------------------- #
# the run contract
# --------------------------------------------------------------------------- #


def test_an_unconfigured_installation_says_how_to_build_one(monkeypatch):
    monkeypatch.delenv("GENRAYHOME", raising=False)
    with pytest.raises(FileNotFoundError, match="install_genray.sh"):
        find_genray_executable(GENRAYConfig(mode="X", time=0.3))


class _WritingBackend:
    """Stands in for GENRAY: records the launch and writes a genray.nc, or does not."""

    def __init__(self, write=True, returncode=0):
        self.write, self.returncode, self.requests = write, returncode, []

    def run(self, request):
        self.requests.append(request)
        if self.write:
            _write_genray_nc(Path(request.workdir) / "genray.nc")
        return ExecutionResult(returncode=self.returncode)


def _executable(tmp_path):
    from external_code_stubs import write_launchable_stub

    return write_launchable_stub(tmp_path / "bin" / "xgenray")


def test_run_prepares_launches_and_maps_into_waves(vest_48224, tmp_path):
    backend = _WritingBackend()
    config = _config(executable=str(_executable(tmp_path)), backend=backend)
    ods = _with_launcher(vest_48224)

    result = run(ods, config, workdir=tmp_path / "case")

    assert result.ok
    assert Path(backend.requests[0].workdir) == tmp_path / "case"
    assert (tmp_path / "case" / "genray.dat").is_file()
    assert "waves.coherent_wave.0.beam_tracing.0.beam.0.position.r" in ods
    assert json.loads(ods["waves.code.parameters"])["inputs"]["mode"] == "X"


def test_a_run_without_genray_nc_is_not_ok_and_leaves_waves_alone(vest_48224, tmp_path):
    # A stale genray.nc from an earlier run must not pass for this one.
    case = tmp_path / "case"
    case.mkdir()
    _write_genray_nc(case / "genray.nc")
    config = _config(executable=str(_executable(tmp_path)), backend=_WritingBackend(write=False))
    ods = _with_launcher(vest_48224)

    result = run(ods, config, workdir=case)

    assert not result.ok
    assert "waves" not in ods


def test_a_failed_exit_is_not_ok_even_with_output(vest_48224, tmp_path):
    config = _config(executable=str(_executable(tmp_path)), backend=_WritingBackend(returncode=3))
    ods = _with_launcher(vest_48224)

    result = run(ods, config, workdir=tmp_path / "case")

    assert not result.ok
    assert "waves" not in ods


def test_a_ray_that_never_entered_the_plasma_is_not_a_result(tmp_path):
    untraced = {"wr": [80.0], "wz": [-36.0], "wphi": [0.0], "delpwr": [3.0e10], "wn_r": [-1.0],
                "wn_z": [0.0], "wn_phi": [0.0], "wnpar": [0.0], "wnper": [1.0]}
    parsed = read_genray_netcdf(_write_genray_nc(tmp_path / "genray.nc", rays=[untraced], status=[-1]))

    assert not parsed["complete"]
    assert parsed["stop_reasons"] == ["never entered the plasma (dinit_1ray found no boundary crossing)"]
    with pytest.raises(ValueError, match="traced rays 0/1"):
        genray_to_waves(parsed, ODS(), time=0.3)


def test_a_file_without_totals_is_incomplete(tmp_path):
    parsed = read_genray_netcdf(_write_genray_nc(tmp_path / "genray.nc", totals=False))
    assert not parsed["complete"]
    assert "power_total" in parsed["missing_totals"]


def test_reflection_loss_is_not_reported_as_absorption(tmp_path):
    parsed = read_genray_netcdf(_write_genray_nc(tmp_path / "genray.nc", refl_loss=0.1))
    ods = ODS()
    genray_to_waves(parsed, ods, time=0.3)
    assert "electrons" not in ods["waves.coherent_wave.0.beam_tracing.0.beam.0"]


def test_a_rerun_replaces_the_previous_rays_instead_of_merging(tmp_path):
    two = [
        {"wr": [80.0, 70.0], "wz": [-36.0, -36.0], "wphi": [0.0, 0.0], "delpwr": [3.0e10, 2.0e10],
         "wn_r": [-1.0, -1.0], "wn_z": [0.0, 0.0], "wn_phi": [0.0, 0.0], "wnpar": [0.0, 0.0], "wnper": [1.0, 1.0]},
    ] * 2
    ods = ODS()
    genray_to_waves(read_genray_netcdf(_write_genray_nc(tmp_path / "a.nc", rays=two)), ods, time=0.3)
    assert len(ods["waves.coherent_wave.0.beam_tracing.0.beam"]) == 2

    genray_to_waves(read_genray_netcdf(_write_genray_nc(tmp_path / "b.nc")), ods, time=0.3)

    assert len(ods["waves.coherent_wave.0.beam_tracing.0.beam"]) == 1


def test_a_second_time_is_refused_in_the_same_waves(tmp_path):
    ods = ODS()
    parsed = read_genray_netcdf(_write_genray_nc(tmp_path / "genray.nc"))
    genray_to_waves(parsed, ods, time=0.3)
    with pytest.raises(ValueError, match="separate ODS"):
        genray_to_waves(parsed, ods, time=0.31, coherent_wave_index=1)


def test_a_homogeneous_time_launcher_uses_the_ids_time(vest_48224, tmp_path):
    ods = _with_launcher(vest_48224)
    for node in ("ec_launchers.beam.0.time", "ec_launchers.beam.0.frequency.time"):
        del ods[node]
    ods["ec_launchers.ids_properties.homogeneous_time"] = 1
    ods["ec_launchers.time"] = np.array([0.3])

    inputs = prepare_genray_inputs(ods, _config(), tmp_path)

    assert inputs.provenance["launcher"]["frequency_hz"] == pytest.approx(2.45e9)


@pytest.mark.parametrize(
    ("zeff", "reason"),
    [(0.5, "< 1"), (float("nan"), "not finite")],
)
def test_an_unusable_zeff_profile_is_refused_or_named(vest_48224, tmp_path, zeff, reason):
    ods = _with_launcher(vest_48224)
    n = len(ods["core_profiles.profiles_1d.0.grid.psi"])
    ods["core_profiles.profiles_1d.0.zeff"] = np.full(n, zeff)

    with pytest.raises(ValueError, match="zeff"):
        prepare_genray_inputs(ods, _config(zeff=None), tmp_path / "a")
    inputs = prepare_genray_inputs(ods, _config(zeff=2.0), tmp_path / "b")
    assert reason in inputs.provenance["zeff_source"]


def test_a_psi_convention_mismatch_between_ids_is_refused(vest_48224, tmp_path):
    ods = _with_launcher(vest_48224)
    axis = float(ods["equilibrium.time_slice.0.global_quantities.psi_axis"])
    ods["core_profiles.profiles_1d.0.grid.psi_magnetic_axis"] = axis * 2 * math.pi + 1.0
    with pytest.raises(ValueError, match="psi convention"):
        prepare_genray_inputs(ods, _config(), tmp_path)


def test_a_stopped_genray_run_is_a_failed_result(tmp_path):
    """#1016: returncode None and runtime_status "timeout" (it was 124)."""
    from external_code_stubs import RecordingBackend, write_launchable_stub
    from vaft.code.execution import ExecutionResult
    from vaft.code.genray.config import GENRAYConfig, GENRAYInputs
    from vaft.code.genray.runner import run_genray

    executable = write_launchable_stub(tmp_path / "xgenray")
    backend = RecordingBackend(ExecutionResult(returncode=None, timed_out=True, elapsed_s=9.0))
    config = GENRAYConfig(executable=str(executable), timeout=9.0, backend=backend)
    inputs = GENRAYInputs(
        workdir=tmp_path, genray_in=tmp_path / "genray.in", eqdsk=tmp_path / "eqdsk", provenance={}
    )
    result = run_genray(inputs, config)
    assert (result.status, result.runtime_status, result.returncode) == ("failed", "timeout", None)
    assert result.elapsed_s == 9.0 and result.parsed is None
    assert result.stderr == "GENRAY timed out after 9 s of running"
