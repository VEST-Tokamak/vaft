"""GEQDSK <-> OMAS round-trip fidelity, in particular profiles_2d.psi.

`from_omas()` used to guess whether `profiles_2d.psi` needed transposing by
comparing its shape against `(len(dim2), len(dim1))`. That check is
ambiguous whenever the grid is square (`nw == nh`) -- VEST's EFIT/CHEASE
grids always are (129x129, 513x513) -- and silently transposed psi that
`to_omas()` had already written in the DD-correct `[dim1, dim2]` = (R, Z)
orientation. The corrupted psi map fed CHEASE a self-inconsistent
equilibrium, which failed deep inside CHEASE's spline setup with
"xin not in ascending order" for every ODS-sourced refinement.
"""

from __future__ import annotations

import numpy as np

from vaft.data.eqdsk import from_omas, read_geqdsk
from vaft.data.resources import data_path


def test_geqdsk_to_omas_round_trip_preserves_every_field():
    direct = read_geqdsk(data_path("efit/g039915.00319"))
    roundtrip = from_omas(direct.to_omas())

    for key in direct.mapping:
        original = direct[key]
        restored = roundtrip.get(key)
        assert restored is not None, f"{key} missing after round trip"
        if isinstance(original, str):
            assert original == restored, key
            continue
        original_arr = np.asarray(original)
        restored_arr = np.asarray(restored)
        assert original_arr.shape == restored_arr.shape, key
        if original_arr.size:
            np.testing.assert_allclose(original_arr, restored_arr, err_msg=key)


def test_geqdsk_to_omas_round_trip_does_not_transpose_psirz_on_a_square_grid():
    """Direct regression for the transpose bug: NW == NH is VEST's normal case."""
    direct = read_geqdsk(data_path("efit/g039915.00319"))
    assert direct["NW"] == direct["NH"], "fixture must be square to exercise the bug"

    roundtrip = from_omas(direct.to_omas())

    original_psi = np.asarray(direct["PSIRZ"])
    restored_psi = np.asarray(roundtrip["PSIRZ"])
    # psi gains 2*pi on the way in and loses it on the way out (issue #236),
    # which re-rounds the last ulp; tight allclose keeps the transpose check.
    np.testing.assert_allclose(original_psi, restored_psi, rtol=1e-12)
    # A transposed square array is not generally equal to itself unless it
    # happens to be symmetric -- confirm the fixture's psi genuinely isn't,
    # so this test could actually have caught the bug.
    assert not np.allclose(original_psi, original_psi.T)


def test_a_weber_family_geqdsk_is_not_scaled_by_two_pi_again():
    """A g-file storing psi in Wb (COCOS 11-18, e.g. TCV's 17) reaches the ODS as it is (#294).

    The same equilibrium written in both families must give the same ODS. Before #294,
    `to_omas` multiplied the weber file by 2*pi as well, leaving its psi (2*pi)^2 off
    Ampere's law and its li_3 about 40 times too large.
    """
    from vaft.data.eqdsk import TWO_PI, to_omas
    from vaft.process.cocos import identify_flux_exponent
    from vaft.process.equilibrium import as_equilibrium

    per_radian = read_geqdsk(data_path("efit/g039915.00319"))
    weber = dict(per_radian.mapping)
    for key in ("SIMAG", "SIBRY", "PSIRZ"):
        weber[key] = np.asarray(per_radian[key], dtype=float) * TWO_PI
    for key in ("FFPRIM", "PPRIME"):
        weber[key] = np.asarray(per_radian[key], dtype=float) / TWO_PI
    assert as_equilibrium(weber).convention.psi_per_radian is False   # the fixture is a weber file

    reference, converted = to_omas(per_radian), to_omas(weber)
    slice_ = "equilibrium.time_slice.0"
    for leaf in ("global_quantities.psi_axis", "global_quantities.psi_boundary", "profiles_1d.psi",
                 "profiles_1d.f_df_dpsi", "profiles_1d.dpressure_dpsi", "profiles_1d.phi",
                 "profiles_2d.0.psi"):
        np.testing.assert_allclose(converted[f"{slice_}.{leaf}"], reference[f"{slice_}.{leaf}"], rtol=1e-12,
                                   err_msg=leaf)
    exponent, ratio = identify_flux_exponent(as_equilibrium(converted))
    assert exponent == 1 and abs(ratio / TWO_PI - 1.0) < 0.05   # Ampere: weber, not (2*pi)^2
