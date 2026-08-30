"""Derived equilibrium quantities the native EQDSK path produces.

``read_geqdsk(...).to_omas()`` used to write ``rho_tor_norm = sqrt(psi_N)`` and no
volume at all, so it was not a drop-in for the OMFIT conversion the pipelines
used before issue #192. These tests pin the replacements: the coordinate is now
integrated from ``q``, and ``profiles_1d.volume`` comes from tracing the flux
surfaces.

The reference numbers below were captured from ``OMFITgeqdsk.to_omas()`` on the
packaged g-files while both paths were still available. They are hard-coded on
purpose -- nothing here imports OMFIT.
"""
import numpy as np
import pytest

pytest.importorskip("omas")

from vaft.data import read_geqdsk
from vaft.data.resources import data_path
from vaft.formula.equilibrium import exact_volume_from_RZ_contour

#: ``(g-file, OMFIT profiles_1d.volume[-1] in m^3)``.
EDGE_VOLUME_REFERENCE = (
    ("efit/g039915.00319", 1.08445),
    ("efit/g040330.00320", 0.70213),
    ("kineticEfit/g048224.00300", 1.01871),
)


@pytest.fixture(scope="module")
def slices():
    out = {}
    for name, _ in EDGE_VOLUME_REFERENCE:
        path = data_path(name)
        if not path.exists():
            continue
        out[name] = read_geqdsk(path).to_omas()["equilibrium.time_slice"][0]
    if not out:
        pytest.skip("no packaged g-file available")
    return out


@pytest.mark.parametrize("name", [entry[0] for entry in EDGE_VOLUME_REFERENCE])
def test_rho_tor_norm_is_a_normalized_toroidal_flux_coordinate(slices, name):
    eqt = slices.get(name) or pytest.skip(f"{name} not present")
    rho = np.asarray(eqt["profiles_1d.rho_tor_norm"], dtype=float)
    phi = np.asarray(eqt["profiles_1d.phi"], dtype=float)
    rho_tor = np.asarray(eqt["profiles_1d.rho_tor"], dtype=float)

    assert rho[0] == 0.0
    assert rho[-1] == pytest.approx(1.0)
    assert np.all(np.diff(rho) > 0.0)
    assert phi[0] == 0.0
    # rho_tor ~ sqrt(Phi) means rho_tor_norm must sit above the sqrt(psi_N)
    # proxy nowhere by accident: check the two really differ, so a regression
    # back to the proxy fails here.
    psi_norm = np.linspace(0.0, 1.0, rho.size)
    assert np.max(np.abs(rho - np.sqrt(psi_norm))) > 1e-3
    np.testing.assert_allclose(rho, rho_tor / rho_tor[-1], rtol=1e-12)


@pytest.mark.parametrize("name,edge_volume", EDGE_VOLUME_REFERENCE)
def test_volume_profile_matches_the_omfit_reference(slices, name, edge_volume):
    eqt = slices.get(name) or pytest.skip(f"{name} not present")
    volume = np.asarray(eqt["profiles_1d.volume"], dtype=float)

    assert volume[0] == 0.0
    assert np.all(np.isfinite(volume))
    assert np.all(np.diff(volume) >= 0.0)
    # 1% covers the difference between tracing psi_N levels on the g-file's own
    # grid and OMFIT's independent flux-surface solve.
    assert volume[-1] == pytest.approx(edge_volume, rel=0.01)
    assert float(eqt["global_quantities.volume"]) == pytest.approx(volume[-1])


def test_volume_is_omitted_when_derived_data_is_declined():
    eqt = read_geqdsk(data_path("efit/g039915.00319")).to_omas(
        allow_derived_data=False
    )["equilibrium.time_slice"][0]

    assert "profiles_1d.volume" not in eqt
    # rho_tor is not "derived data" in that sense -- it replaces a field the
    # conversion always wrote.
    assert "profiles_1d.rho_tor_norm" in eqt


def test_exact_volume_reproduces_the_analytic_torus():
    """A circular cross-section torus has the closed form V = 2*pi^2*R0*a^2."""
    r0, minor = 0.45, 0.30
    theta = np.linspace(0.0, 2.0 * np.pi, 720, endpoint=False)
    r = r0 + minor * np.cos(theta)
    z = minor * np.sin(theta)

    assert exact_volume_from_RZ_contour(r, z) == pytest.approx(
        2.0 * np.pi**2 * r0 * minor**2, rel=1e-4
    )


def test_mean_radius_factorization_diverges_on_a_real_boundary():
    """Why ``volume_from_RZ_boundary`` is not what the volume profile uses.

    Its ``2*pi*A_poly*mean(R)`` form is exact for a circular cross-section,
    where ``mean(R)`` is the centroid, and drifts as the surface departs from
    one. On VEST's shaped boundary that is several percent -- large enough to
    matter for a reported volume, which is why the profile uses the contour
    integral instead.
    """
    from vaft.formula.equilibrium import volume_from_RZ_boundary

    geqdsk = read_geqdsk(data_path("efit/g039915.00319"))
    n_boundary = int(geqdsk["NBBBS"])
    r = np.asarray(geqdsk["RBBBS"], dtype=float)[:n_boundary]
    z = np.asarray(geqdsk["ZBBBS"], dtype=float)[:n_boundary]

    exact = exact_volume_from_RZ_contour(r, z)
    approximate = volume_from_RZ_boundary(r, z)

    assert exact == pytest.approx(1.08445, rel=0.01)  # the OMFIT reference
    assert approximate / exact - 1.0 > 0.03


def test_exact_volume_closes_an_open_contour_and_ignores_orientation():
    theta = np.linspace(0.0, 2.0 * np.pi, 361)
    r, z = 0.5 + 0.1 * np.cos(theta), 0.1 * np.sin(theta)

    closed = exact_volume_from_RZ_contour(r, z)
    open_contour = exact_volume_from_RZ_contour(r[:-1], z[:-1])
    reversed_contour = exact_volume_from_RZ_contour(r[::-1], z[::-1])

    assert closed == pytest.approx(open_contour, rel=1e-12)
    assert closed == pytest.approx(reversed_contour, rel=1e-12)


def test_trailing_efit_namelists_reach_code_parameters():
    """EFIT appends &OUT1/&BASIS/&CHIOUT after the g-file body.

    Those are the reconstruction's own inputs and fit diagnostics, recorded
    nowhere else in the equilibrium, and the reader used to walk straight past
    them. Values were checked against OMFIT's parse of the same file.
    """
    geqdsk = read_geqdsk(data_path("kineticEfit/g048224.00300"))

    assert set(geqdsk.namelists) == {"out1", "basis", "chiout"}
    assert geqdsk.namelists["out1"]["betap0"] == pytest.approx(0.5)
    assert geqdsk.namelists["basis"]["kppfnc"] == 0

    eqt = geqdsk.to_omas()["equilibrium.code.parameters.time_slice.0"]
    assert eqt["out1.betap0"] == pytest.approx(0.5)
    assert eqt["chiout.chipasma"] is not None


def test_a_gfile_without_trailing_namelists_still_reads():
    """CHEASE-written g-files carry no namelist block."""
    geqdsk = read_geqdsk(data_path("efit/g040330.00320"))

    assert geqdsk.namelists == {}
    assert "equilibrium.code.parameters" not in geqdsk.to_omas(
        allow_derived_data=False
    )
