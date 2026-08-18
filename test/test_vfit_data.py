import matplotlib
import numpy as np
import pytest
from scipy.io import savemat

from vaft.data.vfit import read_vfit

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _grid():
    return {"ContourR": np.array([0.1, 0.2, 0.3]), "ContourZ": np.array([-0.1, 0.1])}


def _surfaces(count: int):
    r = np.array([[0.15, 0.16, 0.15, 0.14], [0.1, 0.3, 0.3, 0.1]], dtype=float)
    z = np.array([[0.0, 0.01, 0.0, -0.01], [-0.1, -0.1, 0.1, 0.1]], dtype=float)
    if count == 1:
        return {"R": r, "Z": z}
    return {"R": np.stack((r, r + 0.01), axis=-1), "Z": np.stack((z, z), axis=-1)}


def _common(count: int):
    wall = np.array([[0.8, -0.1, 0.02, 0.03], [0.8, 0.1, 0.02, 0.03]])
    current = np.array([[1.0, 2.0], [3.0, 4.0]])[:, :count]
    return {
        "Grid": _grid(),
        "VESTGeometry": {"Wall": wall},
        "Eddy": {"WallCurrent": current},
        "Magnetics": {
            "R": np.array([0.15, 0.25]),
            "Z": np.array([-0.05, 0.05]),
            "Residual": np.array([0.1, -0.2]),
        },
        "shotNumber": 12345,
    }


def _gse_payload(count: int = 2):
    psi = np.array([[0.0, 0.1, 0.2], [0.3, 0.4, 0.5]])
    if count > 1:
        psi = np.stack((psi, psi + 1.0), axis=-1)
    payload = _common(count)
    payload.update(
        {
            "ProfileTime": np.arange(1, count + 1) * 100,
            "ConstBtor": 0.17,
            "ProfFitContour": {"Psi": psi},
            "ProfFitShape": {
                "Rmag": np.linspace(0.15, 0.16, count),
                "Zmag": np.zeros(count),
                "PsiA": np.zeros(count),
                "PsiB": np.ones(count),
            },
            "ProfFitConstMHD": {
                "Ip": np.linspace(10_000.0, 20_000.0, count),
                "BetaP": np.linspace(0.1, 0.2, count),
                "BetaT": np.linspace(0.01, 0.02, count),
                "BetaN": np.linspace(1.0, 2.0, count),
                "Lint": np.linspace(0.3, 0.4, count),
                "Wmag": np.linspace(100.0, 200.0, count),
                "q0": np.linspace(1.0, 1.1, count),
            },
            "ProfFitFluxSurf": _surfaces(count),
            "ProfFitProfile": {
                "PsiN": np.tile(np.array([0.0, 1.0]), (count, 1)),
                "F": np.tile(np.array([0.2, 0.1]), (count, 1)),
                "P": np.tile(np.array([1000.0, 0.0]), (count, 1)),
                "q": np.tile(np.array([1.0, 2.0]), (count, 1)),
                "J": np.tile(np.array([3.0, 0.0]), (count, 1)),
            },
        }
    )
    return payload


def _fem_payload(count: int = 1):
    psi = np.array([[0.0, 0.1, 0.2], [0.3, 0.4, 0.5]])
    if count > 1:
        psi = np.stack((psi, psi + 1.0), axis=-1)
    payload = _common(count)
    payload.update(
        {
            "FitTime": np.arange(1, count + 1) * 300,
            "Contour": {"Psi": psi},
            "ConstShape": {
                "Rmag": np.linspace(0.15, 0.16, count),
                "Zmag": np.zeros(count),
                "PsiA": np.zeros(count),
                "PsiB": np.ones(count),
            },
            "ConstMHD": {
                "Ip": np.linspace(10_000.0, 20_000.0, count),
                "Lint": np.ones(count) * 0.3,
            },
            "FluxSurface": _surfaces(count),
            "Profile": {
                "PsiN": np.tile(np.array([0.0, 1.0]), (count, 1)),
                "q": np.tile(np.array([1.0, 2.0]), (count, 1)),
                "J": np.tile(np.array([3.0, 0.0]), (count, 1)),
            },
        }
    )
    return payload


def _write(path, payload):
    savemat(path, payload)
    return path


def test_gse_multislice_maps_profiles_flux_and_passive_wall(tmp_path):
    result = read_vfit(_write(tmp_path / "Equilibrium.mat", _gse_payload()))

    assert result.kind == "gse"
    assert result.shot == 12345
    assert np.allclose(result.times, [0.1, 0.2])

    ods = result.to_omas()
    assert np.allclose(ods["equilibrium.time"], [0.1, 0.2])
    assert ods["equilibrium.time_slice.0.profiles_2d.0.psi"].shape == (3, 2)
    assert np.allclose(
        ods["equilibrium.time_slice.0.profiles_2d.0.psi"],
        np.array([[0.0, 0.3], [0.1, 0.4], [0.2, 0.5]]) * (2.0 * np.pi),
    )
    assert np.allclose(
        ods["equilibrium.time_slice.0.profiles_1d.pressure"], [1000.0, 0.0]
    )
    assert "equilibrium.time_slice.0.profiles_1d.f_df_dpsi" in ods
    assert ods["equilibrium.time_slice.1.global_quantities.beta_pol"] == pytest.approx(
        0.2
    )
    assert ods["equilibrium.vacuum_toroidal_field.r0"] == pytest.approx(0.4)
    assert np.allclose(ods["equilibrium.vacuum_toroidal_field.b0"], [0.17, 0.17])
    assert len(ods["pf_passive.loop"]) == 2
    assert np.allclose(ods["pf_passive.loop.1.current"], [3.0, 4.0])


def test_fem_omits_unavailable_pressure_and_appends(tmp_path):
    gse = read_vfit(_write(tmp_path / "Equilibrium.mat", _gse_payload(count=1)))
    fem = read_vfit(_write(tmp_path / "ElementAnalysis.mat", _fem_payload()))

    ods = gse.to_omas(include_pf_passive=False)
    fem.to_omas(ods, include_pf_passive=False)

    assert fem.kind == "fem"
    assert len(ods["equilibrium.time_slice"]) == 2
    assert ods["equilibrium.time_slice.1.time"] == pytest.approx(0.3)
    assert "equilibrium.time_slice.1.profiles_1d.pressure" not in ods
    assert np.allclose(ods["equilibrium.time_slice.1.profiles_1d.j_tor"], [3.0, 0.0])


def test_vfit_plot_helpers_cover_equilibrium_profiles_wall_and_residuals(tmp_path):
    result = read_vfit(_write(tmp_path / "Equilibrium.mat", _gse_payload(count=1)))
    equilibrium_ax = result.plot_equilibrium(show_flux_surfaces=False)
    profiles_figure, profile_axes = result.plot_profiles()
    wall_ax = result.plot_wall_currents()
    residual_ax = result.plot_magnetics_residuals()

    assert equilibrium_ax.get_title().startswith("VFIT GSE psi")
    assert len(profile_axes.reshape(-1)) == 4
    assert len(wall_ax.collections) == 1
    assert len(residual_ax.collections) == 1
    plt.close(equilibrium_ax.figure)
    plt.close(profiles_figure)
    plt.close(wall_ax.figure)
    plt.close(residual_ax.figure)

    fem = read_vfit(_write(tmp_path / "ElementAnalysis.mat", _fem_payload()))
    with pytest.raises(ValueError, match="Pressure"):
        fem.plot_equilibrium(quantity="pressure")


def test_rejects_unknown_or_mismatched_result_kind(tmp_path):
    invalid = _write(tmp_path / "unknown.mat", {"not_vfit": 1})
    with pytest.raises(ValueError, match="Unsupported VFIT MAT result"):
        read_vfit(invalid)

    gse = _write(tmp_path / "Equilibrium.mat", _gse_payload(count=1))
    with pytest.raises(ValueError, match="Requested kind"):
        read_vfit(gse, kind="fem")


def test_v73_error_is_actionable(tmp_path):
    import h5py

    source = tmp_path / "v73.mat"
    with h5py.File(source, "w") as handle:
        handle["not_a_v5_mat"] = [1]
    with pytest.raises(ValueError, match="v7.3/HDF5"):
        read_vfit(source)


def test_to_imas_forwards_local_target_as_hdf5_uri(tmp_path, monkeypatch):
    import vaft.imas.omas_imas

    result = read_vfit(_write(tmp_path / "Equilibrium.mat", _gse_payload(count=1)))
    called = {}

    def fake_save(ods, **kwargs):
        called["ods"] = ods
        called.update(kwargs)
        return ["equilibrium"]

    monkeypatch.setattr(vaft.imas.omas_imas, "save_omas_imas", fake_save)
    assert result.to_imas(tmp_path / "imas-output", new=True) == ["equilibrium"]
    assert called["new"] is True
    assert called["uri"].startswith("imas:hdf5?path=")
    assert "equilibrium" in called["ods"]


def test_to_imas_hdf5_roundtrip_when_imas_is_installed(tmp_path):
    pytest.importorskip("imas")
    from vaft.imas.omas_imas import load_omas_imas

    result = read_vfit(_write(tmp_path / "Equilibrium.mat", _gse_payload(count=1)))
    target = tmp_path / "imas-output"
    # OMAS bundled with the test runtime supplies the 3.40 schema; the VFIT
    # writer itself accepts any caller-selected IMAS DD version.
    result.to_imas(target, new=True, imas_version="3.40.0")
    loaded = load_omas_imas(
        uri="imas:hdf5?path=" + str(target),
        paths=[["equilibrium"]],
        verbose=False,
        imas_version="3.40.0",
    )
    assert np.allclose(loaded["equilibrium.time"], [0.1])
    assert loaded["equilibrium.time_slice.0.profiles_2d.0.psi"].shape == (3, 2)
