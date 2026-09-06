"""Issue #478: the plotting layer labels poloidal flux by its stored convention.

``vaft.plot.backend.convention.psi_convention`` reads a declared COCOS index
through the accessor, else probes the data the way ``vaft.data.eqdsk`` does,
so OMAS and IMAS inputs agree without converting one into the other.
"""

from __future__ import annotations

import copy

import numpy as np
import pytest

import vaft
from vaft.plot.backend.convention import FLUX_CONVENTIONS, declared_cocos, psi_convention

TWO_PI = 2.0 * np.pi


@pytest.fixture(scope="module")
def weber():
    """A DD weber ODS from the packaged g-file (slope test available: phi stored)."""
    from vaft.data.resources import sample_geqdsk

    return sample_geqdsk("efit/g039915.00319").to_omas()


def _per_radian(ods):
    legacy = copy.deepcopy(ods)
    ts = legacy["equilibrium.time_slice.0"]
    for leaf in ("global_quantities.psi_axis", "global_quantities.psi_boundary"):
        ts[leaf] = float(ts[leaf]) / TWO_PI
    for leaf in ("profiles_1d.psi", "profiles_2d.0.psi"):
        ts[leaf] = np.asarray(ts[leaf], float) / TWO_PI
    for leaf in ("profiles_1d.dpressure_dpsi", "profiles_1d.f_df_dpsi"):
        ts[leaf] = np.asarray(ts[leaf], float) * TWO_PI
    return legacy


def test_the_vocabulary_is_the_two_canonical_unit_tokens():
    assert FLUX_CONVENTIONS == ("Wb", "Wb/rad")


def test_a_declared_index_settles_it_without_probing(weber):
    from vaft.omas.general import set_ods_cocos

    declared = copy.deepcopy(weber)
    set_ods_cocos(declared, 3)  # deliberately contradicts the data: the label wins
    del declared["equilibrium.time_slice.0.profiles_1d.phi"]
    assert declared_cocos(declared) == 3
    assert psi_convention(declared) == "Wb/rad"
    set_ods_cocos(declared, 13)
    assert psi_convention(declared) == "Wb"


def test_an_undeclared_weber_ods_is_read_from_the_slope(weber):
    assert declared_cocos(weber) is None
    assert psi_convention(weber) == "Wb"
    assert psi_convention(_per_radian(weber)) == "Wb/rad"


def test_an_undeclared_ods_without_phi_is_read_from_ampere(weber):
    legacy = _per_radian(weber)
    del legacy["equilibrium.time_slice.0.profiles_1d.phi"]
    assert psi_convention(legacy) == "Wb/rad"
    weber_only = copy.deepcopy(weber)
    del weber_only["equilibrium.time_slice.0.profiles_1d.phi"]
    assert psi_convention(weber_only) == "Wb"


def test_nothing_to_probe_falls_back_to_the_dd_convention(weber):
    bare = copy.deepcopy(weber)
    ts = bare["equilibrium.time_slice.0"]
    for leaf in ("profiles_1d.phi", "boundary.outline.r", "boundary.outline.z"):
        del ts[leaf]
    assert psi_convention(bare) == "Wb"


def test_the_requested_slice_is_probed_first_and_a_dead_slice_does_not_decide():
    sample = vaft.omas.load(vaft.data.sample(39915, representation="omas"))
    del sample["equilibrium.code.parameters"]
    assert declared_cocos(sample) is None
    # Slice 8 is a degenerate EFIT solution; the file's convention still comes through.
    assert psi_convention(sample, time_slice=8) == "Wb"


def test_omas_and_imas_entries_agree_on_the_packaged_shots():
    from vaft.imas.access import IDSEntry

    omas_ods = vaft.omas.load(vaft.data.sample(39915, representation="omas"))
    assert declared_cocos(omas_ods) == 11 and psi_convention(omas_ods) == "Wb"
    with vaft.imas.load(vaft.data.sample(39915, representation="imas"), imas_version="3.41.0") as handle:
        entry = IDSEntry(handle)
        # The declaration is parsed from the DD's XML string, not converted.
        from vaft.plot.backend.access import get

        assert isinstance(get(entry, "equilibrium.code.parameters"), str)
        assert declared_cocos(entry) == 11
        assert psi_convention(entry, time_slice=4) == "Wb"
    # 41524 is repository-only and still the legacy Wb/rad artifact, undeclared.
    try:
        path = vaft.data.sample(41524, representation="imas")
    except Exception:
        pytest.skip("41524 is not packaged in this build")
    with vaft.imas.load(path, imas_version="3.41.0") as handle:
        entry = IDSEntry(handle)
        assert declared_cocos(entry) is None
        assert psi_convention(entry) == "Wb/rad"


def test_the_probe_never_converts_the_imas_entry(monkeypatch):
    from vaft.imas.access import IDSEntry

    def boom(*args, **kwargs):  # pragma: no cover - guards the contract
        raise AssertionError("psi_convention must not convert the entry to an ODS")

    monkeypatch.setattr(IDSEntry, "as_ods_for", boom, raising=False)
    with vaft.imas.load(vaft.data.sample(39915, representation="imas"), imas_version="3.41.0") as handle:
        assert psi_convention(IDSEntry(handle)) == "Wb"


def test_the_module_imports_no_data_model():
    import importlib

    module = importlib.import_module("vaft.plot.backend.convention")
    source = open(module.__file__, encoding="utf-8").read()
    assert "import omas" not in source and "from omas" not in source
    assert "vaft.omas" not in source and "vaft.imas" not in source


# ---------------------------------------------------------------------------
# display policy
# ---------------------------------------------------------------------------

def test_the_display_policy_offers_per_radian_flux_only_to_the_equilibrium():
    from vaft.plot.display import QUANTITIES, allowed_units, resolve_display

    assert set(QUANTITIES["magnetic_flux"].units) == {"Wb", "mWb", "Wb/rad", "mWb/rad"}
    assert set(QUANTITIES["poloidal_flux"].units) == {"Wb", "mWb", "Wb/rad", "mWb/rad"}
    assert allowed_units("magnetic_flux", "flux_loop") == ("Wb", "mWb")
    assert allowed_units("magnetic_flux", "equilibrium") == ("Wb", "mWb", "Wb/rad", "mWb/rad")
    # Exact 2*pi, one direction per call.
    assert resolve_display("Wb/rad", unit="mWb", subject="equilibrium").scale == pytest.approx(1e3 * TWO_PI)
    assert resolve_display("Wb", unit="mWb/rad", subject="equilibrium").scale == pytest.approx(1e3 / TWO_PI)
    assert resolve_display("Wb/rad", subject="equilibrium").unit == "mWb/rad"
    assert resolve_display("Wb", subject="equilibrium").unit == "mWb"
    with pytest.raises(ValueError, match="flux_loop.*supported units: Wb, mWb"):
        resolve_display("Wb", unit="Wb/rad", subject="flux_loop")


def test_pass_through_stays_silent_for_real_units_and_warns_for_display_ones():
    import warnings

    from vaft.plot.display import resolve_display

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert resolve_display("a.u.").scale == 1.0
        assert resolve_display("Pa/Wb", subject="equilibrium").unit == "Pa/Wb"
    with pytest.warns(UserWarning, match="'mWb' is a display unit"):
        resolve_display("mWb")


def test_unit_markup_leaves_the_per_radian_units_readable():
    from vaft.plot.display import unit_markup

    assert unit_markup("mWb/rad") == "mWb/rad"
    assert unit_markup("mWb/rad", flavor="html") == "mWb/rad"


# ---------------------------------------------------------------------------
# the psi maps, the vacuum map and the slice summary
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def sample():
    return vaft.omas.load(vaft.data.sample(39915, representation="omas"))


def test_the_psi_map_takes_units_and_converts_exactly(sample):
    from vaft.plot.backend.recipes import build_model
    from vaft.omas.entries import normalize_entries

    entries = normalize_entries(sample)
    default = build_model("equilibrium_field_psi", entries, time_slice=4)
    assert default.display.unit == "mWb" and default.value_label == "Poloidal Flux [mWb]"
    per_radian = build_model("equilibrium_field_psi", entries, time_slice=4, units="Wb/rad")
    assert per_radian.display.unit == "Wb/rad" and per_radian.value_label == "Poloidal Flux [Wb/rad]"
    np.testing.assert_allclose(per_radian.values, default.values / (1e3 * TWO_PI))
    np.testing.assert_allclose(np.asarray(per_radian.contour_levels), np.asarray(default.contour_levels) / (1e3 * TWO_PI))
    with pytest.raises(ValueError, match="supported units: Wb, Wb/rad, mWb, mWb/rad"):
        build_model("equilibrium_field_psi", entries, time_slice=4, units="furlongs")
    with pytest.raises(ValueError, match="use units="):
        build_model("equilibrium_field_psi", entries, time_slice=4, yunit="mWb")


def test_a_legacy_per_radian_ods_is_labelled_per_radian_by_default(sample):
    from vaft.plot.backend.recipes import build_model
    from vaft.omas.entries import normalize_entries

    legacy = copy.deepcopy(sample)
    del legacy["equilibrium.code.parameters"]
    for index in range(len(legacy["equilibrium.time_slice"])):
        ts = legacy[f"equilibrium.time_slice.{index}"]
        for leaf in ("global_quantities.psi_axis", "global_quantities.psi_boundary"):
            ts[leaf] = float(ts[leaf]) / TWO_PI
        for leaf in ("profiles_1d.psi", "profiles_2d.0.psi"):
            ts[leaf] = np.asarray(ts[leaf], float) / TWO_PI
    entries = normalize_entries(legacy)
    model = build_model("equilibrium_field_psi", entries, time_slice=4)
    assert model.display.unit == "mWb/rad" and model.value_label == "Poloidal Flux [mWb/rad]"
    reference = build_model("equilibrium_field_psi", normalize_entries(sample), time_slice=4)
    # The same physics: per-radian numbers are 2*pi smaller than the weber ones.
    np.testing.assert_allclose(model.values, reference.values / TWO_PI)
    # Asked for weber, the legacy file converts by 2*pi and matches the DD file.
    converted = build_model("equilibrium_field_psi", entries, time_slice=4, units="mWb")
    assert converted.display.unit == "mWb"
    np.testing.assert_allclose(converted.values, reference.values)


def test_the_vacuum_map_goes_through_the_same_policy(sample):
    from vaft.plot.backend.recipes import build_model
    from vaft.omas.entries import normalize_entries

    entries = normalize_entries(sample)
    vacuum = build_model("equilibrium_field_psi_vacuum", entries)
    assert vacuum.display.unit == "mWb" and vacuum.value_label == "Vacuum Poloidal Flux [mWb]"
    per_radian = build_model("equilibrium_field_psi_vacuum", entries, units="Wb/rad")
    np.testing.assert_allclose(per_radian.values, vacuum.values / (1e3 * TWO_PI))
    plasma = build_model("equilibrium_field_psi", entries, time_slice=4)
    assert plasma.display.unit == vacuum.display.unit


def test_the_overview_text_and_map_share_one_unit(sample):
    from vaft.plot.backend.recipes import build_model
    from vaft.omas.entries import normalize_entries

    entries = normalize_entries(sample)
    for units, expected in ((None, "mWb"), ("Wb", "Wb"), ("mWb/rad", "mWb/rad")):
        options = {"time_slice": 4, **({"units": units} if units else {})}
        panels = build_model("equilibrium_overview", entries, **options)
        field = panels.models[0]
        text = next(m for m in panels.models if hasattr(m, "lines"))
        psi_line = next(line for line in text.lines if line.startswith("psi_axis"))
        assert field.display.unit == expected
        assert psi_line.endswith(f" {expected}"), psi_line
        stored = float(sample["equilibrium.time_slice.4.global_quantities.psi_axis"])
        assert float(psi_line.split()[1]) == pytest.approx(stored * field.display.scale, rel=2e-3)


def test_flux_loops_never_offer_per_radian(sample):
    from vaft.plot.backend.recipes import build_model
    from vaft.omas.entries import normalize_entries

    entries = normalize_entries(sample)
    with pytest.raises(ValueError, match="flux_loop"):
        build_model("flux_loop_time_flux", entries, yunit="Wb/rad")
    catalog = vaft.omas.available_plots(sample)
    loops = next(r for r in catalog if r.name == "flux_loop_time_flux")
    assert "Wb/rad" not in loops.display["units"]
    psi = next(r for r in catalog if r.name == "equilibrium_field_psi")
    assert psi.display["units"] == ("Wb", "mWb", "Wb/rad", "mWb/rad")
    assert psi.display["unit"] == "mWb" and psi.display["convention"] == "Wb"
    vacuum = next(r for r in catalog if r.name == "equilibrium_field_psi_vacuum")
    assert vacuum.display["convention"] == "Wb"


def test_both_renderers_label_the_colorbar_from_the_display(sample):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = vaft.omas.plot_equilibrium_field_psi(sample, time_slice=4, units="mWb/rad")
    labels = [a.get_ylabel() for a in figure.axes if a is not axes]
    assert any("mWb/rad" in label for label in labels)
    plt.close(figure)
    plotly_figure = vaft.omas.plot_equilibrium_field_psi(sample, time_slice=4, units="mWb/rad", backend="plotly")
    titles = [
        trace.colorbar.title.text
        for trace in plotly_figure.data
        if getattr(trace, "colorbar", None) is not None and trace.colorbar.title.text
    ]
    assert any("mWb/rad" in title for title in titles)
