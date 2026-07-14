from pathlib import Path


def test_data_and_code_import_smoke():
    import vaft
    import vaft.code
    import vaft.data

    assert vaft.data is not None
    assert vaft.code is not None
    assert "data" in dir(vaft)


def test_geqdsk_roundtrip(tmp_path):
    from vaft.data import read_geqdsk, write_geqdsk
    from vaft.data.resources import data_path

    geqdsk = read_geqdsk(data_path("efit/g039915.00319"))
    out = tmp_path / "g039915.roundtrip"
    write_geqdsk(geqdsk, out)
    reread = read_geqdsk(out)

    assert reread["NW"] == geqdsk["NW"]
    assert reread["NH"] == geqdsk["NH"]
    assert len(reread["FPOL"]) == len(geqdsk["FPOL"])


def test_geqdsk_to_and_from_omas():
    from vaft.data import from_omas, read_geqdsk
    from vaft.data.resources import data_path

    geqdsk = read_geqdsk(data_path("efit/g039915.00319"))
    ods = geqdsk.to_omas(allow_derived_data=False)
    converted = from_omas(ods, allow_derived_data=False)

    assert ods["equilibrium.time_slice.0.profiles_2d.0.psi"].shape == (
        geqdsk["NW"],
        geqdsk["NH"],
    )
    assert converted["NW"] == geqdsk["NW"]
    assert converted["NH"] == geqdsk["NH"]


def test_collect_efit_outputs_discovers_and_parses(tmp_path):
    from vaft.code.efit import EFITConfig, collect_efit_outputs
    from vaft.data.resources import data_path

    gdir = tmp_path / "gfile"
    adir = tmp_path / "afile"
    kdir = tmp_path / "kfile"
    gdir.mkdir()
    adir.mkdir()
    kdir.mkdir()
    (gdir / "g039915.00319").write_text(data_path("efit/g039915.00319").read_text())
    (adir / "a039915.00319").write_text("dummy")
    (kdir / "k039915.00319").write_text("dummy")

    result = collect_efit_outputs(tmp_path, EFITConfig(shot=39915))

    assert len(result.gfiles) == 1
    assert len(result.afiles) == 1
    assert len(result.kfiles) == 1
    assert len(result.geqdsk) == 1
    assert result.returncode is None


def test_no_omfit_runtime_import_for_geqdsk():
    import sys
    from vaft.data import read_geqdsk
    from vaft.data.resources import data_path

    read_geqdsk(data_path("efit/g039915.00319"))

    assert "omfit_classes.omfit_eqdsk" not in sys.modules
    assert "omfit_classes.fluxSurface" not in sys.modules


def test_sfl_update_is_standalone():
    import sys
    from vaft.data import read_geqdsk
    from vaft.data.resources import data_path
    from vaft.omas.update import update_equilibrium_profiles_2d_sfl_coordinates

    ods = read_geqdsk(data_path("efit/g039915.00319")).to_omas()
    update_equilibrium_profiles_2d_sfl_coordinates(
        ods,
        time_slice=0,
        profiles_2d_idx=1,
        n_theta=32,
    )
    prof = ods["equilibrium.time_slice.0.profiles_2d.1"]

    assert prof["r"].shape == (129, 32)
    assert prof["z"].shape == (129, 32)
    assert prof["psi"].shape == (129, 32)
    assert "omfit_classes.omfit_eqdsk" not in sys.modules
    assert "omfit_classes.fluxSurface" not in sys.modules
