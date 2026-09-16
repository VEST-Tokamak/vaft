"""Ideal-GPEC output mapping into ``mhd_linear`` and the stage builder.

Synthetic fixtures shaped like the real ideal-GPEC netCDF layout (see
``gpec_nc_fixtures``) exercise the mapping hermetically: complex plasma
response fields, the derived (total minus plasma) vacuum field, the dense
toroidal-mode grid, shot/time injection, provenance, and the
``build_gpec_ideal_ods`` stage contract.
"""

from __future__ import annotations

import contextlib
import io
import re
import warnings

import numpy as np
import pytest
from omas import ODS

from gpec_nc_fixtures import write_control_nc, write_cylindrical_nc, write_profile_nc
from vaft.code.gpec import _runtime as gpec_runtime
from vaft.machine_mapping.gpec_ideal import gpec_ideal
from vaft.omas.vest_upstream import build_gpec_ideal_ods


REFERENCE_COIL_IN = """&COIL_CONTROL
 coil_num=1
 coil_name(1)="MID"
 coil_cur(1,1)=200.0
 coil_cur(1,2)=200.0
 coil_cur(1,3)=0.0
 coil_cur(1,4)=-200.0
 coil_cur(1,5)=-200.0
 coil_cur(1,6)=0.0
/
"""


@pytest.fixture()
def run_dir(tmp_path):
    write_control_nc(tmp_path)
    write_cylindrical_nc(tmp_path)
    return tmp_path


def test_maps_the_spectral_field_onto_a_declared_psi_m_grid(run_dir):
    """``plasma`` carries one grid per mode, and it carries the spectral
    field -- the one representation the resonant derivation can read back and
    the one the slot's only registered renderer expects. The cylindrical
    decomposition stays in the sidecar."""
    ods = ODS(consistency_check=False)
    write_cylindrical_nc(run_dir)
    profile = write_profile_nc(run_dir)
    extras = gpec_ideal(ods, str(run_dir), {"time_s": 0.3})

    entry = ods["mhd_linear.time_slice.0.toroidal_mode.0"]
    assert entry["n_tor"] == 1
    assert entry["energy_perturbed"] == pytest.approx(1.5)

    np.testing.assert_allclose(entry["plasma.grid.dim1"], profile["psi_n"])
    np.testing.assert_allclose(entry["plasma.grid.dim2"], profile["m_out"])
    assert entry["plasma.grid_type.index"] == -1
    assert entry["plasma.grid_type.name"] == "inverse_psi_hamada_fourier"

    field = (
        entry["plasma.b_field_perturbed.coordinate1.real"]
        + 1j * entry["plasma.b_field_perturbed.coordinate1.imaginary"]
    )
    # Native (m, psi) lands as (dim1, dim2) = (psi, m).
    np.testing.assert_allclose(field, profile["Jbgradpsi"].T)

    assert extras[1]["energy_perturbed"] == pytest.approx(1.5)
    assert (run_dir / "gpec_ideal_native_n1.json").exists()


def test_the_cylindrical_field_is_not_written_into_the_spectral_slot(run_dir):
    """Two representations, one grid. Writing the (R, z) field here left it
    declared on whatever grid the last writer set, and the slot's registered
    renderer reads it as harmonics against flux."""
    ods = ODS(consistency_check=False)
    data = write_cylindrical_nc(run_dir)
    write_profile_nc(run_dir)
    gpec_ideal(ods, str(run_dir))
    entry = ods["mhd_linear.time_slice.0.toroidal_mode.0"]
    assert np.size(entry["plasma.grid.dim1"]) != np.size(data["R"])
    assert "vacuum" not in entry or "b_field_perturbed" not in entry["vacuum"]


def test_everything_the_derivation_needs_is_in_the_ods(run_dir):
    """The point of this mapping: a consumer reads the resonant response out
    of `mhd_linear` without going back to the run directory. chi1 is the one
    that nearly got away -- it scales the jump, it is a control-file
    attribute, and the round trip I first called "from the IDS alone" was
    quietly still opening the .nc for it."""
    ods = ODS(consistency_check=False)
    write_profile_nc(run_dir)
    gpec_ideal(ods, str(run_dir))

    entry = ods["mhd_linear.time_slice.0.toroidal_mode.0"]
    parameters = ods["mhd_linear.code.parameters"]
    assert entry["n_tor"] == 1
    assert np.size(entry["plasma.grid.dim1"]) and np.size(entry["plasma.grid.dim2"])
    assert "real" in entry["plasma.b_field_perturbed.coordinate1"]
    chi1 = re.search(r'<rational_surfaces chi1="([^"]+)"', parameters)
    assert chi1 is not None, "chi1 is not recoverable from the IDS"
    assert float(chi1.group(1)) != 0.0
    for field in ("psi_n", "q", "dq_dpsi_n", "area", "geometric_factor"):
        assert f'{field}="' in parameters


def test_each_modes_geometry_is_inside_its_own_solver_element(run_dir):
    """`code.parameters` is one IDS-global string that accumulates a fragment
    per mode. As siblings, N <solver>, N <spectral_field> and N
    <rational_surfaces> could only be paired by document order -- and the
    surfaces of n = 1 and n = 3 are different surfaces."""
    import xml.etree.ElementTree as ET

    ods = ODS(consistency_check=False)
    write_profile_nc(run_dir, n=1, rational_q=(2.0, 3.0))
    write_control_nc(run_dir, n=1)
    write_cylindrical_nc(run_dir, n=1)
    write_profile_nc(run_dir, n=2, rational_q=(2.0, 2.5, 3.0))
    write_control_nc(run_dir, n=2)
    write_cylindrical_nc(run_dir, n=2)
    gpec_ideal(ods, str(run_dir), {"mode": 1, "modes": [1, 2]})
    gpec_ideal(ods, str(run_dir), {"mode": 2, "modes": [1, 2]})

    root = ET.fromstring(ods["mhd_linear.code.parameters"])
    solvers = root.findall("solver")
    assert [s.get("n_tor") for s in solvers] == ["1", "2"]
    counts = {
        s.get("n_tor"): len(s.find("rational_surfaces").findall("surface"))
        for s in solvers
    }
    assert counts == {"1": 2, "2": 3}
    for solver in solvers:
        assert solver.find("spectral_field") is not None
        assert solver.find("rational_surfaces").get("chi1")
    # Nothing left at the top level to be paired by position.
    assert root.findall("rational_surfaces") == []
    assert root.findall("spectral_field") == []


def test_geometry_is_withheld_rather_than_written_without_its_normalisation(run_dir):
    """A geometry block without chi1 reads as complete and is not: a consumer
    would have to go back to the control file for one number."""
    from vaft.machine_mapping import gpec_ideal as module

    write_profile_nc(run_dir)
    profile = module.read_gpec_netcdf(str(run_dir)).profile
    assert module._rational_surface_geometry(profile, 1, None) is None
    assert module._rational_surface_geometry(profile, 1, 0.0) is None
    assert module._rational_surface_geometry(profile, 1, 1.6) is not None


def test_one_ods_cannot_hold_both_the_dcon_and_the_gpec_product(run_dir, tmp_path):
    """`mhd_linear`'s plasma region carries one grid per mode, and the two
    writers fill it with different quantities on different grids. Merging
    them left the second writer's grid describing the first writer's array,
    with grid_type.index and grid_type.name contradicting each other."""
    from vaft.machine_mapping.mhd_linear import claim_ids

    ods = ODS(consistency_check=False)
    write_profile_nc(run_dir)
    gpec_ideal(ods, str(run_dir))
    assert ods["mhd_linear.code.name"] == "GPEC"
    with pytest.raises(ValueError, match="cannot hold both"):
        claim_ids(ods, "mhd_linear", "GPEC-suite")

    other = ODS(consistency_check=False)
    other["mhd_linear.code.name"] = "GPEC-suite"
    with pytest.raises(ValueError, match="cannot hold both"):
        gpec_ideal(other, str(run_dir))


def test_the_same_writer_may_add_more_modes(run_dir):
    """The guard separates products, not calls: one GPEC ODS holds every
    mode of that run."""
    ods = ODS(consistency_check=False)
    write_profile_nc(run_dir)
    gpec_ideal(ods, str(run_dir), {"modes": [1]})
    gpec_ideal(ods, str(run_dir), {"modes": [1]})
    assert ods["mhd_linear.code.name"] == "GPEC"


def test_the_geometry_the_derivation_needs_is_recorded(run_dir):
    """psi, q, dq/dpsi_N and the area are equilibrium quantities, and the
    geometric factor absorbs a vacuum surface inductance that nothing
    downstream can rebuild -- so it is measured here or not at all."""
    ods = ODS(consistency_check=False)
    write_profile_nc(run_dir)
    gpec_ideal(ods, str(run_dir))
    surfaces = re.findall(r"<surface ([^/]*)/>", ods["mhd_linear.code.parameters"])
    assert surfaces
    fields = dict(re.findall(r'(\w+)="([^"]+)"', surfaces[0]))
    assert set(fields) == {"psi_n", "q", "dq_dpsi_n", "area", "geometric_factor"}
    # Plain floats, not numpy reprs: this XML is read back by a parser.
    for value in fields.values():
        assert not value.startswith("np."), fields
        float(value)


def test_time_injection_overrides_zero_attrs(run_dir):
    ods = ODS(consistency_check=False)
    gpec_ideal(ods, str(run_dir), {"time_s": 0.3})
    assert ods["mhd_linear.ids_properties.homogeneous_time"] == 1
    np.testing.assert_allclose(ods["mhd_linear.time"], [0.3])
    assert ods["mhd_linear.time_slice.0.time"] == pytest.approx(0.3)


def test_a_call_without_time_s_still_leaves_the_ids_homogeneous_in_time(run_dir):
    """``time_slice`` is a dynamic AOS: without a time mode and a ``time`` as
    long as the AOS, imas-python refuses to write the IDS at all. A slice
    nobody timed reads NaN rather than a fabricated instant."""
    ods = ODS(consistency_check=False)
    gpec_ideal(ods, str(run_dir), {"modes": [1]})
    assert ods["mhd_linear.ids_properties.homogeneous_time"] == 1
    times = np.asarray(ods["mhd_linear.time"], dtype=float)
    assert times.shape == (len(ods["mhd_linear.time_slice"]),)
    assert np.all(np.isnan(times))
    assert "time" not in ods["mhd_linear.time_slice.0"]


def test_a_time_base_laid_out_by_the_pipeline_is_kept(run_dir):
    """The pipeline writes the whole time base before any solver runs; a
    mapper call for one slice must neither shorten nor overwrite it."""
    ods = ODS(consistency_check=False)
    ods["mhd_linear.ids_properties.homogeneous_time"] = 1
    ods["mhd_linear.time"] = [0.30, 0.31]
    gpec_ideal(ods, str(run_dir), {"time_slice": 1, "modes": [1]})
    np.testing.assert_allclose(ods["mhd_linear.time"], [0.30, 0.31])
    assert ods["mhd_linear.time_slice.1.time"] == pytest.approx(0.31)
    assert len(ods["mhd_linear.time_slice"]) == 2


def test_the_mapped_ods_round_trips_through_imas(run_dir, tmp_path):
    """The reason the time base matters: ``vaft.imas.save`` validates the IDS,
    and the native (IMAS-entry) plot path reads the file back."""
    imas = pytest.importorskip("imas")
    import vaft.imas

    ods = ODS(consistency_check=False)
    profile = write_profile_nc(run_dir, rational_q=(2.0, 3.0))
    gpec_ideal(ods, str(run_dir), {"modes": [1]})
    target = tmp_path / "round_trip.nc"
    with warnings.catch_warnings(), contextlib.redirect_stderr(io.StringIO()):
        warnings.simplefilter("ignore")
        vaft.imas.save(ods, str(target))

    with imas.DBEntry(str(target), "r", dd_version="3.41.0") as entry:
        ids = entry.get("mhd_linear")
        assert int(ids.ids_properties.homogeneous_time) == 1
        assert len(ids.time) == len(ids.time_slice) == 1
        mode = ids.time_slice[0].toroidal_mode[0]
        assert int(mode.n_tor) == 1
        np.testing.assert_allclose(mode.plasma.grid.dim1, profile["psi_n"])
        np.testing.assert_allclose(mode.plasma.grid.dim2, profile["m_out"])
        np.testing.assert_allclose(
            mode.plasma.b_field_perturbed.coordinate1.imaginary,
            ods["mhd_linear.time_slice.0.toroidal_mode.0.plasma.b_field_perturbed.coordinate1.imaginary"],
        )


def test_dense_mode_grid_positions(run_dir):
    ods = ODS(consistency_check=False)
    gpec_ideal(ods, str(run_dir), {"modes": [1, 2]})
    modes = ods["mhd_linear.time_slice.0.toroidal_mode"]
    assert len(modes) == 2
    assert modes[0]["n_tor"] == 1
    assert modes[1]["n_tor"] == 2  # padded entry: n_tor only
    assert "energy_perturbed" not in modes[1]


def test_include_spectral_false_leaves_the_profile_file_unopened(run_dir, monkeypatch):
    """Mapping the spectral field means reading the profile output, which is
    144 MB on the DIII-D example. A caller that wants only the control-level
    mapping must be able to decline that."""
    write_profile_nc(run_dir)
    from vaft.code.gpec import _profile_output

    monkeypatch.setattr(
        _profile_output, "_read_profile",
        lambda path: pytest.fail(f"the profile file was opened: {path}"),
    )
    ods = ODS(consistency_check=False)
    extras = gpec_ideal(ods, str(run_dir), {"include_spectral": False})
    entry = ods["mhd_linear.time_slice.0.toroidal_mode.0"]
    assert "grid" not in entry["plasma"]
    # The provenance flag still answers, because it reads the recorded path.
    assert extras[1]["has_profile"] is True


def test_code_parameters_provenance(run_dir):
    ods = ODS(consistency_check=False)
    gpec_ideal(ods, str(run_dir))
    parameters = ods["mhd_linear.code.parameters"]
    for token in (
        'solver name="gpec"',
        'derivation="energy_vacuum+energy_surface+energy_plasma"',
        'derivation="total_minus_plasma"',
        "<jacobian>hamada</jacobian>",
    ):
        assert token in parameters
    assert ods["mhd_linear.code.name"] == "GPEC"
    assert ods["mhd_linear.code.version"] == "v1.5.5-test"
    assert ods["mhd_linear.code.output_flag"][0] == 0


def test_provenance_reports_which_native_outputs_the_run_wrote(run_dir):
    """``has_cylindrical`` and ``has_profile``: what is there, not what is mapped.

    The profile file's resonant quantities have no IMAS home yet (vaft#170),
    so a consumer of the provenance is the only way to learn the run produced
    one.
    """
    ods = ODS(consistency_check=False)
    extras = gpec_ideal(ods, str(run_dir))
    assert extras[1]["has_cylindrical"] is True
    assert extras[1]["has_profile"] is False

    write_profile_nc(run_dir)
    extras = gpec_ideal(ODS(consistency_check=False), str(run_dir))
    assert extras[1]["has_profile"] is True


def test_the_profile_flag_is_read_from_the_path_not_the_contents(run_dir):
    """``has_profile`` answers "was there one?" from the recorded source
    path. With the spectral mapping on, the file is opened anyway -- but the
    flag must not be what opens it, which
    ``test_include_spectral_false_leaves_the_profile_file_unopened`` pins."""
    run_dir_no_profile = run_dir
    extras = gpec_ideal(
        ODS(consistency_check=False), str(run_dir_no_profile),
        {"include_spectral": False},
    )
    assert extras[1]["has_profile"] is False
    write_profile_nc(run_dir_no_profile)
    extras = gpec_ideal(
        ODS(consistency_check=False), str(run_dir_no_profile),
        {"include_spectral": False},
    )
    assert extras[1]["has_profile"] is True


def test_a_profile_for_another_mode_is_not_counted(run_dir):
    """``gpec_profile_output_n2.nc`` beside an n=1 run is not this run's."""
    write_profile_nc(run_dir, n=2)
    extras = gpec_ideal(ODS(consistency_check=False), str(run_dir))
    assert extras[1]["has_profile"] is False


def test_control_only_run_still_maps_energy(tmp_path):
    write_control_nc(tmp_path)
    ods = ODS(consistency_check=False)
    gpec_ideal(ods, str(tmp_path))
    entry = ods["mhd_linear.time_slice.0.toroidal_mode.0"]
    assert entry["energy_perturbed"] == pytest.approx(1.5)
    assert "grid" not in entry["plasma"] if "plasma" in entry else True


def _make_cell(root, time_ms, mode):
    run_dir = gpec_runtime.module_dir(root, time_ms, "gpec", mode)
    run_dir.mkdir(parents=True, exist_ok=True)
    write_control_nc(run_dir, n=mode)
    write_cylindrical_nc(run_dir, n=mode)
    (run_dir / "coil.in").write_text(REFERENCE_COIL_IN, encoding="utf-8")
    return run_dir


def test_build_gpec_ideal_ods(tmp_path):
    _make_cell(tmp_path, 300, 1)
    ods, manifest = build_gpec_ideal_ods(
        shot=48226, time_values=[300], workdir=tmp_path, modes=[1]
    )

    assert manifest["stage"] == "gpec_ideal"
    assert manifest["status"] == "success"
    cell = manifest["modules_modes"]["t=300/gpec/n=1"]
    assert cell["status"] == "success"
    hashed = manifest["input"]
    assert "t=300/gpec/n=1/gpec_control_output_n1.nc" in hashed
    assert "t=300/gpec/n=1/coil.in" in hashed

    # Field and cause travel together: the run's excitation reaches the
    # canonical coil geometry, matched by identifier with turns preserved.
    identifiers = [
        ods[f"coils_non_axisymmetric.coil.{i}.identifier"]
        for i in range(len(ods["coils_non_axisymmetric.coil"]))
    ]
    assert len(identifiers) == 18
    mid01 = identifiers.index("VEST_3D_MID_01")
    np.testing.assert_allclose(
        ods[f"coils_non_axisymmetric.coil.{mid01}.current.data"], [200.0]
    )
    assert ods[f"coils_non_axisymmetric.coil.{mid01}.turns"] == 20.0

    assert ods["mhd_linear.time_slice.0.toroidal_mode.0.n_tor"] == 1
    np.testing.assert_allclose(ods["mhd_linear.time"], [0.3])


def test_build_gpec_ideal_ods_records_missing_cells(tmp_path):
    ods, manifest = build_gpec_ideal_ods(
        shot=48226, time_values=[300], workdir=tmp_path, modes=[1]
    )
    assert manifest["status"] == "empty"
    assert manifest["modules_modes"]["t=300/gpec/n=1"]["status"] == "missing"
    # The dense grid still exists with a padded, flagged entry.
    assert ods["mhd_linear.time_slice.0.toroidal_mode.0.n_tor"] == 1
    assert ods["mhd_linear.code.output_flag"][0] == -1


def test_build_gpec_ideal_ods_mode_workdirs(tmp_path):
    root = tmp_path / "cellroot"
    _make_cell(root, 300, 1)
    ods, manifest = build_gpec_ideal_ods(
        shot=48226,
        time_values=[300],
        mode_workdirs={1: root},
        modes=[1],
    )
    assert manifest["status"] == "success"


def test_build_gpec_ideal_ods_requires_a_workdir():
    with pytest.raises(ValueError, match="workdir"):
        build_gpec_ideal_ods(shot=48226, time_values=[300], modes=[1])
