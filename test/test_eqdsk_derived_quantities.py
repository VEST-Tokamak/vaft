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

#: ``g-file -> {leaf: OMFIT edge value}`` for the shape profiles a g-file does
#: not store. Triangularity is the loosest: it depends on where the boundary
#: happens to be sampled near its Z extremum.
SHAPE_REFERENCE = {
    "efit/g039915.00319": {
        "area": (0.43738, 0.01),
        "elongation": (1.35297, 0.01),
        "triangularity_upper": (0.36753, 0.05),
        "triangularity_lower": (0.36753, 0.05),
    },
    "efit/g040330.00320": {
        "area": (0.32030, 0.01),
        "elongation": (1.49283, 0.01),
        "triangularity_upper": (0.28357, 0.05),
        "triangularity_lower": (0.27497, 0.05),
    },
}


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

    ods = geqdsk.to_omas()
    params = ods["equilibrium.code.parameters.time_slice.0"]
    assert params["out1.betap0"] == pytest.approx(0.5)
    assert params["chiout.chipasma"] is not None
    # Array entries must be ndarrays: omas' code-parameters encoder recurses
    # into a list and then reindexes it by string key, which raises on save.
    assert isinstance(params["out1.brsp"], np.ndarray)


def test_a_gfile_without_trailing_namelists_still_reads():
    """CHEASE-written g-files carry no namelist block."""
    geqdsk = read_geqdsk(data_path("efit/g040330.00320"))

    assert geqdsk.namelists == {}
    assert "equilibrium.code.parameters" not in geqdsk.to_omas(
        allow_derived_data=False
    )


@pytest.mark.parametrize("name", sorted(SHAPE_REFERENCE))
def test_flux_surface_shape_profiles_match_the_omfit_reference(slices, name):
    """A g-file stores no shape profiles; OMFIT solved for them.

    Without these, `update_equilibrium_boundary` has nothing to derive
    `boundary.elongation`/`triangularity` from and silently writes neither --
    which is what broke the chease-history summaries.
    """
    eqt = slices.get(name) or pytest.skip(f"{name} not present")

    for leaf, (reference, tolerance) in SHAPE_REFERENCE[name].items():
        edge = float(np.asarray(eqt[f"profiles_1d.{leaf}"], dtype=float)[-1])
        assert edge == pytest.approx(reference, rel=tolerance), leaf

    assert float(eqt["global_quantities.area"]) == pytest.approx(
        float(np.asarray(eqt["profiles_1d.area"], dtype=float)[-1])
    )
    assert np.all(np.diff(np.asarray(eqt["profiles_1d.area"], dtype=float)) >= 0.0)
    assert np.all(np.isfinite(np.asarray(eqt["profiles_1d.elongation"], dtype=float)))


def test_boundary_helper_picks_up_the_native_shape_profiles():
    """The chain the chease-history scripts actually depend on."""
    from vaft.omas.update import update_equilibrium_boundary

    ods = read_geqdsk(data_path("efit/g039915.00319")).to_omas()
    update_equilibrium_boundary(ods)
    eqt = ods["equilibrium.time_slice.0"]

    assert float(eqt["boundary.elongation"]) == pytest.approx(1.35297, rel=0.01)
    assert float(eqt["boundary.triangularity"]) == pytest.approx(0.36753, rel=0.05)


def test_r_at_z_extremum_beats_the_nearest_vertex():
    """Triangularity is read at the true Z extremum, not the sampled one."""
    from vaft.data.eqdsk import _r_at_z_extremum

    # A parabola peaking between two samples: Z is symmetric about the gap, so
    # the extremum sits half a vertex away from either neighbour.
    r = np.array([0.0, 1.0, 2.0, 3.0])
    z = np.array([0.0, 1.0, 1.0, 0.0])

    assert _r_at_z_extremum(r, z, upper=True) == pytest.approx(1.5)
    # Degenerate input falls back to the sampled vertex rather than raising.
    flat = np.zeros(4)
    assert _r_at_z_extremum(r, flat, upper=True) == pytest.approx(r[int(np.argmax(flat))])


@pytest.mark.parametrize(
    "namelist,label",
    [
        ("&IN1\n A(3) = 1.0\n A(5) = 2.0\n/\n", "sparse assignment pads with None"),
        ("&IN1\n B(1,1)=1.0\n B(2,1)=2.0\n B(1,2)=3.0\n/\n", "sparse 2-D is ragged"),
    ],
)
def test_awkward_namelist_shapes_still_serialize(namelist, label, tmp_path):
    """f90nml shapes that neither np.asarray nor omas takes as they come."""
    import f90nml
    from omas import ODS, save_omas_h5

    from vaft.data.keqdsk import _plain, write_namelists_to_ods

    ods = ODS()
    write_namelists_to_ods(ods, _plain(f90nml.reads(namelist)))
    save_omas_h5(ods, str(tmp_path / "x.h5"))  # must not raise -- see label


def test_kinetic_grid_lookup_skips_the_flux_surface_trace(monkeypatch):
    """core_profiles only wants the 1D grid, not seconds of contour tracing."""
    from vaft.data import eqdsk as eqdsk_module
    from vaft.process.profile import _grid_from_geq

    def _fail(*args, **kwargs):
        raise AssertionError("_grid_from_geq must not trace flux surfaces")

    monkeypatch.setattr(eqdsk_module, "_surface_geometry", _fail)

    grid = _grid_from_geq(read_geqdsk(data_path("efit/g039915.00319")))

    assert grid is not None
    rho, psi, psi_n = grid
    assert rho.size == psi.size and psi_n[0] == 0.0 and psi_n[-1] == pytest.approx(1.0)
