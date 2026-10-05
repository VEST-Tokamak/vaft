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
import pytest

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



_FLUX_LEAVES = ("global_quantities.psi_axis", "global_quantities.psi_boundary", "profiles_1d.psi",
                "profiles_1d.f_df_dpsi", "profiles_1d.dpressure_dpsi", "profiles_2d.0.psi")


def _families(**overrides):
    """The packaged 39915 g-file per radian, and the same equilibrium written in weber (COCOS 11-18)."""
    from vaft.data.eqdsk import TWO_PI

    per_radian = dict(read_geqdsk(data_path("efit/g039915.00319")).mapping, **overrides)
    weber = dict(per_radian)
    for key in ("SIMAG", "SIBRY", "PSIRZ"):
        weber[key] = np.asarray(per_radian[key], dtype=float) * TWO_PI
    for key in ("FFPRIM", "PPRIME"):
        weber[key] = np.asarray(per_radian[key], dtype=float) / TWO_PI
    return per_radian, weber


def _assert_same_flux(converted, reference, leaves=_FLUX_LEAVES):
    for leaf in leaves:
        path = f"equilibrium.time_slice.0.{leaf}"
        np.testing.assert_allclose(converted[path], reference[path], rtol=1e-12, err_msg=leaf)


@pytest.mark.parametrize("family, case", [
    ("weber", "plain"),
    ("weber", "TCVlike COCOS=17"),
    ("weber", "stale COCOS=1"),          # a token the data contradict does not rescale the file
    ("per_radian", "stale COCOS=11"),    # ... in either direction
])
def test_the_flux_family_of_a_geqdsk_comes_from_its_data_not_a_stale_token(family, case):
    """A weber-family g-file (e.g. TCV's COCOS 17) is not scaled by 2*pi again (#294).

    Before #294 to_omas scaled every file by 2*pi, leaving a weber file (2*pi)^2 off
    Ampere's law: TCV-X21 65402 gave li_3 = 32.9 instead of 0.83.
    """
    from vaft.data.eqdsk import TWO_PI, to_omas
    from vaft.process.cocos import identify_flux_exponent
    from vaft.process.equilibrium import as_equilibrium

    per_radian, weber = _families()
    source = dict(weber if family == "weber" else per_radian, CASE=case)
    reference, converted = to_omas(per_radian), to_omas(source)
    _assert_same_flux(converted, reference, _FLUX_LEAVES + ("profiles_1d.phi",))
    exponent, ratio = identify_flux_exponent(as_equilibrium(converted))
    assert exponent == 1 and abs(ratio / TWO_PI - 1.0) < 0.05   # Ampere: weber, not (2*pi)^2


def test_a_zero_current_weber_geqdsk_is_decided_by_its_q_profile():
    """With no current the signs identify nothing; the contour-q probe still knows the family."""
    from vaft.data.eqdsk import to_omas

    per_radian, weber = _families(CURRENT=0.0)
    _assert_same_flux(to_omas(weber), to_omas(per_radian))


def test_a_weber_geqdsk_survives_an_ods_round_trip_with_its_cocos_token():
    """from_omas writes per radian, so it must not copy a weber COCOS token into CASE verbatim."""
    from vaft.data.eqdsk import to_omas

    per_radian, weber = _families()
    first = to_omas(dict(weber, CASE="TCVlike COCOS=12"))
    exported = from_omas(first)
    assert "COCOS=2" in exported["CASE"]
    _assert_same_flux(to_omas(exported), to_omas(per_radian))
