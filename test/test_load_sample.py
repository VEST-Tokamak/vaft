"""Paired OMAS/IMAS sample registry and semantic compatibility tests."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import subprocess
import sys

import h5py
import numpy as np
import pytest
import yaml

import vaft
from vaft.data.reference_samples import semantic_sample_view, verify_sample_artifacts
from vaft.omas import compare_ods


def test_sample_registry_returns_both_packaged_artifacts():
    assert vaft.data.available_samples() == (39915, 40600, 41524, 41672, 45531, 48224)
    omas_path = vaft.data.sample(39915, representation="omas")
    imas_path = vaft.data.sample(39915, representation="imas")

    assert omas_path.name == "omas.json.gz"
    assert imas_path.name == "imas.nc"
    assert omas_path.is_file()
    assert imas_path.is_file()
    manifest = vaft.data.sample_manifest(39915)
    assert manifest["imas_dd_version"] == "3.41.0"
    assert verify_sample_artifacts(omas_path.parent, manifest) == {
        "omas": manifest["representations"]["omas"]["sha256"],
        "imas": manifest["representations"]["imas"]["sha256"],
    }


def test_sample_registry_errors_are_actionable():
    with pytest.raises(ValueError, match="available shots: 39915, 40600, 41524, 41672, 45531, 48224"):
        vaft.data.sample(1)
    with pytest.raises(ValueError, match="expected one of: imas, omas"):
        vaft.data.sample(39915, representation="pickle")


@pytest.mark.parametrize("shot", [41524, 41672])
def test_pipeline_samples_carry_repository_only_omas_and_imas(shot):
    omas_path = vaft.data.sample(shot, representation="omas")
    imas_path = vaft.data.sample(shot, representation="imas")

    assert omas_path.name == "omas.json.gz"
    assert imas_path.name == "imas.nc"
    manifest = vaft.data.sample_manifest(shot)
    for representation in ("omas", "imas"):
        assert manifest["representations"][representation]["package"] == "repository-only"
    assert verify_sample_artifacts(omas_path.parent, manifest) == {
        "omas": manifest["representations"]["omas"]["sha256"],
        "imas": manifest["representations"]["imas"]["sha256"],
    }


@pytest.mark.parametrize("shot", [41524, 41672])
def test_full_imas_samples_round_trip_through_both_adapters(shot):
    manifest = vaft.data.sample_manifest(shot)
    path = vaft.data.sample(shot, representation="imas")
    version = manifest["imas_dd_version"]

    via_omas = vaft.omas.load(path, imas_version=version)
    with vaft.imas.load(path, imas_version=version) as handle:
        assert handle.info.format == "imas_netcdf"
        assert handle.info.converted is False
        via_imas = handle.to_omas()
    native_omas = vaft.omas.sample_ods(shot)

    reference = semantic_sample_view(native_omas, manifest)
    for candidate in (via_omas, via_imas):
        result = compare_ods(
            reference, semantic_sample_view(candidate, manifest), scope="union"
        )
        assert result.passed
    np.testing.assert_allclose(
        via_omas["equilibrium.time"], manifest["acceptance"]["equilibrium_times"]
    )
    assert via_omas["em_coupling.mutual_passive_passive"].shape == (950, 950)
    assert via_omas["magnetics.ip.0.data"].size > 100
    for index in range(len(via_omas["magnetics.b_field_pol_probe"])):
        probe = via_omas[f"magnetics.b_field_pol_probe.{index}"]
        if "poloidal_angle" in probe:
            # DD: clockwise from +R, so +Bz is 3*pi/2. See issue #288.
            np.testing.assert_allclose(probe["poloidal_angle"], 3 * np.pi / 2)
    if shot in (41524, 41672):
        assert manifest["source"]["kind"] == "pipeline-until-efit"
        assert (
            manifest["pipeline"]["efit"]["successful_time_slices"]
            == len(manifest["acceptance"]["equilibrium_times"])
        )
        ip = np.asarray(
            [
                via_omas[f"equilibrium.time_slice.{index}.global_quantities.ip"]
                for index in range(len(via_omas["equilibrium.time_slice"]))
            ],
            dtype=float,
        )
        assert np.isfinite(ip).all()
        assert np.max(ip) > 1.0e5


def test_repository_only_lookup_explains_missing_wheel_artifact(monkeypatch, tmp_path):
    from vaft.data import resources

    manifest_path = resources.data_path("samples/41524/manifest.yaml")

    def installed_data_path(name=""):
        if name == "samples/41524/manifest.yaml":
            return manifest_path
        return tmp_path / name

    monkeypatch.setattr(resources, "data_path", installed_data_path)
    with pytest.raises(FileNotFoundError, match="Clone the VAFT GitHub repository"):
        resources.sample(41524, representation="omas")


def test_legacy_sample_ods_wrapper_uses_registry_and_sample_odc_is_removed():
    ods = vaft.omas.sample_ods()
    assert ods["dataset_description.data_entry.pulse"] == 39915
    assert "equilibrium" in ods
    assert not hasattr(vaft.omas, "sample_odc")


@pytest.mark.parametrize("shot", [41524, 41672])
def test_sample_ods_loads_every_registered_shot_as_omas(shot):
    ods = vaft.omas.sample_ods(shot)
    assert ods["dataset_description.data_entry.pulse"] == shot
    assert "equilibrium" in ods and "ec_launchers" in ods
    assert len(ods["magnetics.rogowski_coil"]) == 2


def test_reference_probe_metadata_describes_positive_bz():
    ods = vaft.omas.load(vaft.data.sample(39915, representation="omas"))
    assert len(ods["magnetics.b_field_pol_probe"]) == 64
    assert len(ods["magnetics.flux_loop"]) == 11
    for index in range(64):
        angle = ods[f"magnetics.b_field_pol_probe.{index}.poloidal_angle"]
        # The DD's poloidal_angle is clockwise from +R, so the sensitive axis is
        # (cos, -sin).  This assertion used (cos, +sin), which is why a stored
        # pi/2 looked like +Bz here while telling a DD reader -Bz (issue #288).
        np.testing.assert_allclose(
            (np.cos(angle), -np.sin(angle)), (0.0, 1.0), atol=1e-15
        )


@pytest.mark.parametrize("shot", [39915, 41524, 41672])
def test_native_imas_sample_records_explicit_dd_version(shot):
    with h5py.File(vaft.data.sample(shot, "imas"), "r") as handle:
        version = handle.attrs["data_dictionary_version"]
    if isinstance(version, bytes):
        version = version.decode()
    assert str(version) == "3.41.0"


def test_sample_artifact_checksum_failure_is_actionable(tmp_path):
    artifact = tmp_path / "imas.nc"
    artifact.write_bytes(b"tampered")
    manifest = {
        "representations": {
            "imas": {
                "path": artifact.name,
                "sha256": "0" * 64,
                "size": artifact.stat().st_size,
            }
        }
    }
    with pytest.raises(ValueError, match="Checksum mismatch"):
        verify_sample_artifacts(tmp_path, manifest)


def test_pinned_legacy_source_checksum_failure_is_actionable(monkeypatch):
    script = (
        Path(__file__).resolve().parents[1]
        / "workflow"
        / "reference_validation"
        / "generate_legacy_imas_sample.py"
    )
    spec = importlib.util.spec_from_file_location("legacy_sample_generator", script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    monkeypatch.setattr(
        module.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args[0], returncode=0, stdout=b"tampered", stderr=b""
        ),
    )
    manifest = {
        "source": {
            "git_commit": "deadbeef",
            "git_path": "sample.json",
            "sha256": "0" * 64,
        }
    }
    with pytest.raises(ValueError, match="source checksum mismatch"):
        module._git_source_bytes(Path.cwd(), manifest)


def test_paired_sample_exercises_complete_adapter_matrix():
    manifest = vaft.data.sample_manifest(39915)
    version = manifest["imas_dd_version"]
    omas_path = vaft.data.sample(39915, "omas")
    imas_path = vaft.data.sample(39915, "imas")

    omas_native = vaft.omas.load(omas_path, imas_version=version)
    with vaft.imas.load(omas_path, imas_version=version) as handle:
        omas_converted_to_imas = handle.to_omas()
    imas_converted_to_omas = vaft.omas.load(imas_path, imas_version=version)
    with vaft.imas.load(imas_path, imas_version=version) as handle:
        assert handle.info.format == "imas_netcdf"
        assert handle.info.converted is False
        imas_native = handle.to_omas()

    reference = semantic_sample_view(omas_native, manifest)
    assert len(reference) > 100
    for candidate in (
        omas_converted_to_imas,
        imas_converted_to_omas,
        imas_native,
    ):
        result = compare_ods(
            reference,
            semantic_sample_view(candidate, manifest),
            scope="union",
        )
        assert result.passed

    np.testing.assert_allclose(
        omas_native["equilibrium.time"],
        [0.316, 0.317, 0.318, 0.319, 0.320, 0.323, 0.325, 0.326, 0.331],
    )
    assert len(omas_native["equilibrium.time"]) == manifest["pipeline"]["efit"][
        "successful_time_slices"
    ]
    assert len(omas_native["magnetics.b_field_pol_probe"]) == 64
    assert len(omas_native["magnetics.flux_loop"]) == 11
    assert omas_native["magnetics.ids_properties.homogeneous_time"] == 0
    for family, signal in (
        ("b_field_pol_probe", "field"),
        ("b_field_pol_probe", "voltage"),
        ("flux_loop", "flux"),
    ):
        for index in range(len(omas_native[f"magnetics.{family}"])):
            base = f"magnetics.{family}.{index}.{signal}"
            if f"{base}.data" not in omas_native:
                continue
            assert omas_native[f"{base}.data"].shape == omas_native[
                f"{base}.time"
            ].shape
    assert manifest["packaging_policy"]["repository_equilibrium_time_slices"] == "all"
    assert omas_native["wall.description_2d.0.limiter.unit.0.outline.r"].size > 10
    assert (
        omas_native["magnetics.ip.0.data"].shape
        == omas_native["magnetics.ip.0.time"].shape
    )
    assert (
        omas_native["pf_active.coil.0.current.data"].shape
        == omas_native["pf_active.time"].shape
    )
    assert omas_native["equilibrium.time_slice.0.profiles_1d.psi"].size > 10


def test_wheel_build_uses_the_three_slice_39915_variant(tmp_path):
    root = Path(__file__).resolve().parents[1]
    build_lib = tmp_path / "build"
    subprocess.run(
        [sys.executable, "setup.py", "build_py", "--build-lib", str(build_lib)],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    sample_root = build_lib / "vaft" / "data" / "samples" / "39915"
    # Only 39915 ships: the other samples' manifests do, their artifacts do not.
    for shot in (40600, 41524, 41672, 45531, 48224):
        shot_root = build_lib / "vaft" / "data" / "samples" / str(shot)
        assert (shot_root / "manifest.yaml").is_file()
        assert not (shot_root / "omas.json.gz").exists()
    assert not (build_lib / "vaft" / "data" / "kineticEfit").exists()
    manifest = yaml.safe_load((sample_root / "manifest.yaml").read_text(encoding="utf-8"))
    assert manifest["generation"]["distribution_variant"] == "wheel"
    assert manifest["generation"]["equilibrium_time_slices"] == 3
    verify_sample_artifacts(sample_root, manifest)
    wheel_ods = vaft.omas.load(sample_root / "omas.json.gz")
    np.testing.assert_allclose(wheel_ods["equilibrium.time"], [0.316, 0.317, 0.318])
    assert len(wheel_ods["magnetics.b_field_pol_probe"]) == 64
    assert len(wheel_ods["magnetics.flux_loop"]) == 11
    with vaft.imas.load(sample_root / "imas.nc") as handle:
        wheel_native = handle.to_omas()
    np.testing.assert_allclose(
        wheel_native["equilibrium.time"], [0.316, 0.317, 0.318]
    )
    assert len(wheel_native["magnetics.b_field_pol_probe"]) == 64
    assert len(wheel_native["magnetics.flux_loop"]) == 11
    # The wheel variant is substituted into every build, so a probe convention
    # fixed only in vaft/data/samples would ship inverted (issue #288). Both
    # representations must carry the DD-conformant angle, not just the checkout.
    from vaft.machine_mapping.magnetics import POLOIDAL_ANGLE

    for label, source in (("omas", wheel_ods), ("imas", wheel_native)):
        angles = [
            float(source[f"magnetics.b_field_pol_probe.{index}.poloidal_angle"])
            for index in range(len(source["magnetics.b_field_pol_probe"]))
            if "poloidal_angle" in source[f"magnetics.b_field_pol_probe.{index}"]
        ]
        assert angles, label
        np.testing.assert_allclose(angles, POLOIDAL_ANGLE, err_msg=label)


def test_39915_sample_carries_the_complete_pf_active_machine_state():
    """The compact sample must keep every PF coil (regression: it once shipped
    only PF1-PF5, which cannot even hold the measured equilibrium — the
    outboard coils carry real current, so free-boundary work silently broke
    offline)."""
    ods = vaft.omas.load(vaft.data.sample(39915))

    assert len(ods["pf_active.coil"]) == 10
    time = np.asarray(ods["pf_active.time"], dtype=float)
    currents = {
        str(ods[f"pf_active.coil.{i}.name"]): float(
            np.interp(0.325, time, ods[f"pf_active.coil.{i}.current.data"])
        )
        for i in range(10)
    }
    # the outboard coils' measured currents, not zero-padded placeholders
    assert currents["PF6"] == pytest.approx(-1431.8, abs=1.0)
    assert currents["PF9"] == pytest.approx(-522.1, abs=1.0)
    assert currents["PF10"] == pytest.approx(-522.1, abs=1.0)
    # every coil still carries its element geometry for adapter meshing
    assert all(len(ods[f"pf_active.coil.{i}.element"]) > 0 for i in range(10))


def test_packaged_sample_carries_passive_geometry_without_derivable_data():
    """The 39915 sample stores measurements, not reconstructions.

    ``pf_passive`` loop currents are ~48 MB of the canonical source and
    ``em_coupling``'s matrices are a function of the PF geometry version, so the
    sample keeps the geometry and leaves both to
    :func:`vaft.machine_mapping.em_coupling.em_coupling`.
    """
    ods = vaft.omas.sample_ods()

    assert "em_coupling" not in ods
    assert len(ods["pf_passive.loop"]) == 950
    assert "time" not in ods["pf_passive"]
    currents = [
        path for path in ods.flat()
        if str(path).startswith("pf_passive.") and str(path).endswith(".current")
    ]
    assert currents == []

    # The geometry that makes the omission safe is all present.
    for index in (0, 949):
        loop = ods[f"pf_passive.loop.{index}"]
        assert len(loop["element.0.geometry.outline.r"]) > 0
        assert float(loop["resistance"]) > 0.0


def test_em_coupling_is_reconstructible_from_the_packaged_geometry():
    """The omission is only defensible while the reconstruction works."""
    from vaft.machine_mapping.em_coupling import em_coupling

    ods = vaft.omas.sample_ods()
    assert "em_coupling" not in ods

    em_coupling(ods, shot=39915)

    assert np.shape(ods["em_coupling.mutual_active_active"]) == (10, 10)
    assert np.shape(ods["em_coupling.mutual_passive_active"]) == (950, 10)
    # Richer than the sample ever stored: the passive-passive matrix was never
    # in the compact selectors at all.
    assert np.shape(ods["em_coupling.mutual_passive_passive"]) == (950, 950)
    assert len(ods["em_coupling.active_coils"]) == len(ods["pf_active.coil"])
    assert len(ods["em_coupling.passive_loops"]) == len(ods["pf_passive.loop"])


@pytest.mark.parametrize("shot", [39915, 41524, 41672])
def test_pipeline_samples_carry_efit_reconstructed_constraints(shot):
    """The m-file replay (#952) filled reconstructed/chi_squared, and nothing else.

    EFIT's reported chi-square on these runs is the Ip row's vessel accounting
    term: the magnetics are weighted out (sigma = weight * 1e4 under
    legacy_weight), so they contribute ~1e-7 of it at most.  The recomputed
    plasma-only Ip term is ~1e-13, so in the metrics the magnetics dominate.
    """
    manifest = vaft.data.sample_manifest(shot)
    record = manifest["generation"]["efit_constraints"]
    assert manifest["pipeline"]["efit"]["optional_outputs"]["mfile"]["status"] == "regenerated"
    assert record["reproduction"]["max_psi_relative_difference"] < 1e-3
    ods = vaft.omas.sample_ods(shot)
    for index in range(len(ods["equilibrium.time_slice"])):
        root = f"equilibrium.time_slice.{index}.constraints"
        for family in ("bpol_probe", "flux_loop"):
            for channel in range(len(ods[f"{root}.{family}"])):
                assert np.isfinite(ods[f"{root}.{family}.{channel}.reconstructed"])
                assert np.isfinite(ods[f"{root}.{family}.{channel}.chi_squared"])
        fitted_pf = [
            channel for channel in range(len(ods[f"{root}.pf_current"]))
            if np.isfinite(ods[f"{root}.pf_current.{channel}.reconstructed"])
        ]
        # the run's table had 16 coil groups; the other ten entries carry no current
        assert len(fitted_pf) == 16
        ip_chi = float(ods[f"{root}.ip.chi_squared"])
        magnetics_chi = sum(
            float(ods[f"{root}.{family}.{channel}.chi_squared"])
            for family in ("bpol_probe", "flux_loop")
            for channel in range(len(ods[f"{root}.{family}"]))
        )
        assert np.isfinite(ip_chi) and ip_chi > 0.0
        assert magnetics_chi < 1e-5 * max(ip_chi, 1.0)
        if f"equilibrium.time_slice.{index}.boundary.outline.r" in ods:
            assert float(ods[f"{root}.ip.reconstructed"]) == pytest.approx(
                float(ods[f"{root}.ip.measured"]), rel=1e-6
            )
        else:
            # the dead trailing slice: EFIT failed in `bound` and wrote zero current
            assert float(ods[f"{root}.ip.reconstructed"]) == 0.0
    # the flux-loop reconstruction is stored in Wb like the measurement
    from vaft.omas.efit_quality import fit_quality_metrics

    metrics = fit_quality_metrics(ods, time_slice=0)
    assert metrics["families"]["flux_loop"]["sigma_unit_factor"] == pytest.approx(
        2 * np.pi, rel=1e-4
    )
    # The stored Ip chi-square is EFIT's report, which adds the prescribed vessel
    # current to the model; the metrics recompute the plasma-only residual the fit
    # actually matched (#952 follow-up, see test_efit_ip_vessel_chi_squared.py).
    # So the reported value is the vessel accounting term, and Ip owns almost none
    # of the recomputed total -- the magnetics do.
    ip = metrics["scalars"]["ip"]
    assert ip["chi_squared"] < 1e-9
    assert ip["chi_squared_efit_reported"] == pytest.approx(
        ip["vessel_accounting_term"], rel=1e-9
    )
    assert metrics["chi_squared_share"]["ip"] < 1e-3


def test_kinetic_sample_48224_loads_as_omas_with_its_diagnostics():
    manifest = vaft.data.sample_manifest(48224)
    path = vaft.data.sample(48224, representation="omas")
    assert manifest["representations"]["omas"]["package"] == "repository-only"
    assert verify_sample_artifacts(path.parent, manifest) == {
        "omas": manifest["representations"]["omas"]["sha256"]
    }
    ods = vaft.omas.sample_ods(48224)
    assert ods["dataset_description.data_entry.pulse"] == 48224
    acceptance = manifest["acceptance"]
    assert len(ods["thomson_scattering.channel"]) == acceptance["thomson_scattering"]["channel_count"]
    assert len(ods["charge_exchange.channel"]) == acceptance["charge_exchange"]["channel_count"]
    assert len(ods["core_profiles.profiles_1d"]) == acceptance["core_profiles"]["slice_count"]
    np.testing.assert_allclose(ods["equilibrium.time"], acceptance["equilibrium_times"])
    for ids in acceptance["required_ids"]:
        assert ids in ods
    # the full 17-point limiter, not the database's 5-point stub
    from omas import ODS
    from vaft.machine_mapping.wall import wall

    reference = ODS()
    wall(reference)
    for axis in ("r", "z"):
        np.testing.assert_allclose(
            ods[f"wall.description_2d.0.limiter.unit.0.outline.{axis}"],
            reference[f"wall.description_2d.0.limiter.unit.0.outline.{axis}"],
        )


def test_kinetic_sample_48224_carries_three_distinct_equilibria():
    equilibria = vaft.omas.sample_equilibria(48224)
    assert list(equilibria) == ["efit_magnetic", "efit_kinetic", "chease"]
    gq = "equilibrium.time_slice.0.global_quantities"
    for ods in equilibria.values():
        np.testing.assert_allclose(ods["equilibrium.time"], [0.3])
        assert abs(float(ods[f"{gq}.ip"])) > 1.0e5
    assert equilibria["chease"]["equilibrium.time_slice.0.profiles_2d.0.psi"].shape == (513, 513)
    magnetic, kinetic = equilibria["efit_magnetic"], equilibria["efit_kinetic"]
    # the magnetic run has no pressure constraint; the kinetic run fits six points
    assert "pressure" not in magnetic["equilibrium.time_slice.0.constraints"]
    pressure = kinetic["equilibrium.time_slice.0.constraints.pressure"]
    assert len(pressure) == 6
    for j in range(6):
        for leaf in ("measured", "reconstructed", "chi_squared", "position.psi"):
            assert np.isfinite(float(pressure[f"{j}.{leaf}"])), (j, leaf)
    # the pressure points roughly double beta_p on the same magnetics and Ip
    assert float(kinetic[f"{gq}.beta_pol"]) == pytest.approx(0.0290, rel=0.01)
    assert float(magnetic[f"{gq}.beta_pol"]) == pytest.approx(0.0148, rel=0.01)
    assert float(kinetic[f"{gq}.ip"]) == float(magnetic[f"{gq}.ip"])
    # EFIT's chi-square is the plasma-current term; the probes are weighted out
    root = "equilibrium.time_slice.0.constraints"
    assert float(magnetic[f"{root}.ip.chi_squared"]) == pytest.approx(686.53, rel=1e-4)
    probes = magnetic[f"{root}.bpol_probe"]
    assert sum(float(probes[f"{j}.chi_squared"]) for j in range(len(probes))) < 1e-3
    # flux loops are stored in Wb like the other samples
    loops = magnetic[f"{root}.flux_loop"]
    enabled = [j for j in range(len(loops)) if float(loops[f"{j}.weight"]) > 0]
    assert len(enabled) == 4


def test_sample_equilibria_rejects_a_sample_without_them():
    with pytest.raises(ValueError, match="declares no equilibria"):
        vaft.omas.sample_equilibria(39915)


def test_fluctuation_sample_45531_carries_the_three_angle_array_at_native_rate():
    manifest = vaft.data.sample_manifest(45531)
    path = vaft.data.sample(45531, representation="omas")
    assert manifest["representations"]["omas"]["package"] == "repository-only"
    assert verify_sample_artifacts(path.parent, manifest) == {
        "omas": manifest["representations"]["omas"]["sha256"]
    }
    ods = vaft.omas.sample_ods(45531)
    assert ods["dataset_description.data_entry.pulse"] == 45531
    acceptance = manifest["acceptance"]
    for ids in acceptance["required_ids"]:
        assert ids in ods
    assert "equilibrium" not in ods
    magnetics = acceptance["magnetics"]
    probes = ods["magnetics.b_field_pol_probe"]
    assert len(probes) == magnetics["b_field_pol_probe_count"]
    assert len(ods["magnetics.flux_loop"]) == magnetics["flux_loop_count"]

    first = magnetics["fluctuation_first_index"]
    rates: dict[int, int] = {}
    angles = set()
    window = manifest["generation"]["window"]
    for index in range(first, first + magnetics["fluctuation_count"]):
        assert str(ods[f"magnetics.b_field_pol_probe.{index}.identifier"]).startswith("OutMirnov_")
        time = np.asarray(ods[f"magnetics.b_field_pol_probe.{index}.voltage.time"])
        rate = int(round(1.0 / float(np.median(np.diff(time))), -3))
        rates[rate] = rates.get(rate, 0) + 1
        assert window["tstart"] <= time[0] and time[-1] < window["tend"]
        angles.add(round(float(np.degrees(ods[f"magnetics.b_field_pol_probe.{index}.position.phi"])), 3))
    assert rates == {int(k): v for k, v in magnetics["fluctuation_rates_hz"].items()}
    assert angles == {135.0, 225.0, 315.0}
    # The plasma window the sample was cut around is inside it.
    assert window["tstart"] < window["plasma_onset_s"] < window["plasma_offset_s"] < window["tend"]

    sxr = acceptance["soft_x_rays"]
    assert len(ods["soft_x_rays.channel"]) == sxr["channel_count"]
    time = np.asarray(ods["soft_x_rays.channel.0.brightness.time"])
    assert 1.0 / float(np.median(np.diff(time))) == pytest.approx(sxr["sample_rate_hz"], rel=1e-6)
    from vaft.process.soft_x_rays import sxr_te_pairs_from_ods

    assert len(sxr_te_pairs_from_ods(ods, "bottom")) == 16

    from vaft.plot.backend.recipes import _mirnov_phase_available

    assert _mirnov_phase_available(ods) is None


def test_camera_sample_40600_carries_the_frames_and_the_lfs_probe():
    manifest = vaft.data.sample_manifest(40600)
    path = vaft.data.sample(40600, representation="omas")
    assert manifest["representations"]["omas"]["package"] == "repository-only"
    assert verify_sample_artifacts(path.parent, manifest) == {
        "omas": manifest["representations"]["omas"]["sha256"]
    }
    ods = vaft.omas.sample_ods(40600)
    assert ods["dataset_description.data_entry.pulse"] == 40600
    acceptance = manifest["acceptance"]
    for ids in acceptance["required_ids"]:
        assert ids in ods
    camera = acceptance["camera_visible"]
    detector = "camera_visible.channel.0.detector.0"
    frames = ods[f"{detector}.frame"]
    assert len(frames) == camera["frame_count"]
    assert ods[f"{detector}.lines_n"] == camera["lines_n"]
    assert ods[f"{detector}.columns_n"] == camera["columns_n"]
    assert np.asarray(ods[f"{detector}.frame.0.image_raw"]).shape == (camera["lines_n"], camera["columns_n"])
    times = np.asarray([float(ods[f"{detector}.frame.{i}.time"]) for i in range(len(frames))])
    assert 1.0 / float(np.median(np.diff(times))) == pytest.approx(camera["frame_rate_hz"], rel=1e-6)
    assert float(ods[f"{detector}.exposure_time"]) == pytest.approx(1.91e-5)

    magnetics = acceptance["magnetics"]
    probe = f"magnetics.b_field_pol_probe.{magnetics['lfs_probe_index']}"
    assert ods[f"{probe}.identifier"] == "MagneticFieldProbe_C2-05_Bz"
    time = np.asarray(ods[f"{probe}.voltage.time"])
    assert 1.0 / float(np.median(np.diff(time))) == pytest.approx(magnetics["lfs_probe_rate_hz"], rel=1e-6)
    # One clock: the camera window lies inside the probe record.
    assert time[0] <= times[0] and times[-1] <= time[-1]
    assert "ip" in ods["magnetics"]
    assert "alpha" in str(ods["spectrometer_uv.channel.0.name"]).lower()
