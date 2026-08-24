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
    np.testing.assert_array_equal(original_psi, restored_psi)
    # A transposed square array is not generally equal to itself unless it
    # happens to be symmetric -- confirm the fixture's psi genuinely isn't,
    # so this test could actually have caught the bug.
    assert not np.allclose(original_psi, original_psi.T)
