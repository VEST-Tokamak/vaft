"""Integration test for the kinetic chain on the packaged shot-48224 dataset.

Uses the physically-paired sample under ``vaft/data/kineticEfit/`` (equilibrium
``g048224.00300`` + Thomson ``NeTe_48224.mat`` + ion Doppler ``IDS_48224.mat``, all
shot 48224 @ 300 ms). No external binary is exercised -- this covers
``build_kinetic_core_profiles`` and ``kinetic_pressure_points`` end to end on real data
(the same slice the end-to-end notebook drives).
"""
import numpy as np
import pytest

pytest.importorskip("omas")

from vaft.data.resources import data_path

DATA = data_path("kineticEfit")
GFILE = DATA / "g048224.00300"
TS_MAT = DATA / "NeTe_48224.mat"
ION_MAT = DATA / "IDS_48224.mat"

pytestmark = pytest.mark.skipif(
    not (GFILE.exists() and TS_MAT.exists() and ION_MAT.exists()),
    reason="vaft/data/kineticEfit sample (g048224.00300 + NeTe/IDS_48224.mat) not present",
)

SHOT = 48224
TIME_MS = 300.0


@pytest.fixture(scope="module")
def kinetic_ods():
    import vaft

    vaft.apply_omfit_compat_patches()
    from omfit_classes.omfit_eqdsk import OMFITgeqdsk
    from vaft.code.kineticEfit import build_kinetic_core_profiles

    geq = OMFITgeqdsk(str(GFILE))
    geq["fluxSurfaces"].load()
    ods = geq.to_omas()
    ods["equilibrium.ids_properties.homogeneous_time"] = 1
    vaft.machine_mapping.dataset_description(
        ods, source=SHOT,
        options={"source_type": "shot", "description": "kinetic chain test"},
    )
    vaft.machine_mapping.thomson_scattering(ods, SHOT, str(TS_MAT))
    vaft.machine_mapping.charge_exchange(ods, shotnumber=SHOT, options="ids", mat_file=str(ION_MAT))
    ods = build_kinetic_core_profiles(ods, geq, TIME_MS, ion_index=0, time_tolerance_ms=3.0)
    return ods, geq


def test_core_profiles_on_equilibrium_grid_are_physical(kinetic_ods):
    ods, _ = kinetic_ods
    cp = "core_profiles.profiles_1d.0"
    # the equilibrium-grid path must run (rho_tor_norm present, not the rho_pol fallback)
    rho = np.asarray(ods[f"{cp}.grid.rho_tor_norm"], dtype=float)
    ne = np.asarray(ods[f"{cp}.electrons.density"], dtype=float)
    te = np.asarray(ods[f"{cp}.electrons.temperature"], dtype=float)
    ti = np.asarray(ods[f"{cp}.ion.0.temperature"], dtype=float)
    pth = np.asarray(ods[f"{cp}.pressure_thermal"], dtype=float)

    assert rho.size > 10 and np.all(np.isfinite(rho))
    assert 1e18 <= ne[0] <= 1e20, ne[0]        # VEST ohmic density
    assert 20.0 <= te[0] <= 300.0, te[0]        # eV
    assert 2.0 <= ti[0] <= 60.0, ti[0]          # eV (CX C3+, well below Te here)
    assert np.all(pth >= 0.0) and pth[0] > 0.0  # kinetic pressure written


def test_raw6_pressure_points_drop_invalid_ts_channels(kinetic_ods):
    ods, geq = kinetic_ods
    from vaft.code.kineticEfit import kinetic_pressure_points

    pts = kinetic_pressure_points(ods, TIME_MS, geq=geq, encoding="raw6")
    # NeTe_48224 has 7 polychromators, 2 padded with NaN -> 5 valid + 1 edge anchor
    assert len(pts.rpress) == 6, len(pts.rpress)
    assert np.all(np.isfinite(pts.pressr)) and np.all(np.isfinite(pts.sigpre))
    assert pts.rpress[-1] == -1.0 and pts.pressr[-1] == 0.0   # psi_N=1, p=0 edge anchor
    assert max(pts.pressr) > 0.0


def test_spline_encoding_has_129_points(kinetic_ods):
    ods, geq = kinetic_ods
    from vaft.code.kineticEfit import kinetic_pressure_points

    pts = kinetic_pressure_points(ods, TIME_MS, geq=geq, encoding="spline")
    assert len(pts.rpress) == 129
    assert np.all(np.asarray(pts.rpress) <= 0.0)   # spline uses RPRESS = -psi_N


# --------------------------------------------------------------------------- #
# Partial-diagnostic builds: only-Thomson / only-ion / neither
# --------------------------------------------------------------------------- #

def _build(with_ts, with_cx, **kwargs):
    import vaft

    vaft.apply_omfit_compat_patches()
    from omfit_classes.omfit_eqdsk import OMFITgeqdsk
    from vaft.code.kineticEfit import build_kinetic_core_profiles

    geq = OMFITgeqdsk(str(GFILE))
    geq["fluxSurfaces"].load()
    ods = geq.to_omas()
    ods["equilibrium.ids_properties.homogeneous_time"] = 1
    if with_ts:
        vaft.machine_mapping.thomson_scattering(ods, SHOT, str(TS_MAT))
    if with_cx:
        vaft.machine_mapping.charge_exchange(ods, shotnumber=SHOT, options="ids", mat_file=str(ION_MAT))
    build_kinetic_core_profiles(ods, geq, TIME_MS, ion_index=0, time_tolerance_ms=3.0, **kwargs)
    return ods


def _has(ods, path):
    try:
        ods[path]
        return True
    except Exception:
        return False


def test_thomson_only_writes_statistical_kinetic_slice():
    from vaft.process.profile import TI_TE_RATIO_VEST

    cp = "core_profiles.profiles_1d.0"
    ods = _build(with_ts=True, with_cx=False)
    assert _has(ods, f"{cp}.electrons.density")
    assert _has(ods, f"{cp}.electrons.temperature")
    assert not _has(ods, f"{cp}.ion.0.velocity.toroidal")  # no ion diagnostic
    # default ti_te_ratio='auto': Ti = TI_TE_RATIO_VEST * Te and the TS-only
    # kinetic pressure e*ne*(1+ratio)*Te are written
    te = np.asarray(ods[f"{cp}.electrons.temperature"], dtype=float)
    ne = np.asarray(ods[f"{cp}.electrons.density"], dtype=float)
    ti = np.asarray(ods[f"{cp}.ion.0.temperature"], dtype=float)
    pth = np.asarray(ods[f"{cp}.pressure_thermal"], dtype=float)
    np.testing.assert_allclose(ti, TI_TE_RATIO_VEST * te, rtol=1e-10)
    np.testing.assert_allclose(
        pth, 1.602176634e-19 * ne * (1.0 + TI_TE_RATIO_VEST) * te, rtol=1e-10
    )


def test_thomson_only_legacy_mode_unchanged():
    cp = "core_profiles.profiles_1d.0"
    ods = _build(with_ts=True, with_cx=False, ti_te_ratio=None)
    te = np.asarray(ods[f"{cp}.electrons.temperature"], dtype=float)
    ti = np.asarray(ods[f"{cp}.ion.0.temperature"], dtype=float)
    np.testing.assert_allclose(ti, te, rtol=1e-12)         # legacy Ti = Te
    assert not _has(ods, f"{cp}.pressure_thermal")         # and no pressure


def test_ion_only_writes_ion_profiles_without_electrons():
    cp = "core_profiles.profiles_1d.0"
    ods = _build(with_ts=False, with_cx=True)
    assert _has(ods, f"{cp}.ion.0.temperature")
    assert _has(ods, f"{cp}.ion.0.velocity.toroidal")
    assert not _has(ods, f"{cp}.electrons.density")        # no Thomson
    assert not _has(ods, f"{cp}.ion.0.density")            # quasi-neutral density needs ne
    assert not _has(ods, f"{cp}.pressure_thermal")


def test_no_diagnostics_raises():
    import vaft

    vaft.apply_omfit_compat_patches()
    from omfit_classes.omfit_eqdsk import OMFITgeqdsk
    from vaft.code.kineticEfit import build_kinetic_core_profiles

    geq = OMFITgeqdsk(str(GFILE))
    geq["fluxSurfaces"].load()
    ods = geq.to_omas()
    with pytest.raises(Exception):
        build_kinetic_core_profiles(ods, geq, TIME_MS, time_tolerance_ms=3.0)
