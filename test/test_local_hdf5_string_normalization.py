"""An ODS must load the same whether it was written as HDF5 or as JSON.

HDF5 stores strings as bytes; JSON has no bytes type. OMAS's consistency
checker is what normally reconciles the two, and `vaft.database._local.load_ods`
deliberately runs without it so a non-conformant artifact still opens. That left
the container leaking into the data: the same ODS came back with `str` from JSON
and `numpy.bytes_` from HDF5 (#734).

The leak is not cosmetic. `replicate_stage` validates a publication by comparing
the replica against the product it sent; with the product read through this
loader and the replica read through the checker, every string leaf differed by
type alone and the publication was recorded as unvalidated.
"""

from __future__ import annotations

import numpy as np
import pytest

from omas import ODS

import vaft.omas as vaft_omas
from vaft.database._local import _as_text


def _sample() -> ODS:
    ods = ODS(consistency_check=True)
    ods["soft_x_rays.ids_properties.homogeneous_time"] = 1
    ods["soft_x_rays.ids_properties.comment"] = "VEST SXR digitizer data"
    ods["soft_x_rays.time"] = np.array([0.30, 0.31])
    ods["soft_x_rays.channel.0.name"] = "Vertical SXR Ch 1"
    ods["soft_x_rays.channel.0.identifier"] = "17592:vertical:none:1"
    ods["soft_x_rays.channel.0.brightness.time"] = np.array([0.30, 0.31])
    ods["soft_x_rays.channel.0.brightness.data"] = np.array([[1.0, 2.0]])
    return ods


@pytest.fixture
def written(tmp_path):
    ods = _sample()
    paths = {}
    for suffix in (".h5", ".json"):
        target = tmp_path / f"product{suffix}"
        ods.save(str(target))
        paths[suffix] = target
    return paths


class TestContainerDoesNotLeakIntoTheData:
    @pytest.mark.parametrize("suffix", (".h5", ".json"))
    def test_string_leaves_load_as_text(self, written, suffix):
        ods = vaft_omas.load(written[suffix])
        for path in (
            "soft_x_rays.channel.0.name",
            "soft_x_rays.channel.0.identifier",
            "soft_x_rays.ids_properties.comment",
        ):
            value = ods[path]
            assert isinstance(value, str), f"{suffix} {path} -> {type(value).__name__}"
            assert not isinstance(value, bytes)

    def test_the_two_containers_agree(self, written):
        h5 = vaft_omas.load(written[".h5"])
        js = vaft_omas.load(written[".json"])
        for path in (
            "soft_x_rays.channel.0.name",
            "soft_x_rays.channel.0.identifier",
            "soft_x_rays.ids_properties.comment",
        ):
            assert h5[path] == js[path]
            assert type(h5[path]) is type(js[path])

    def test_an_array_of_strings_agrees_too(self, tmp_path):
        """A scalar-only check missed that STR_1D still differed by dtype.

        Decoding to an object array left the characters right and the dtype
        wrong, which `compare_ods` can still flag -- the same failure this fix
        exists to remove, one leaf shape over.
        """
        ods = ODS(consistency_check=True)
        ods["core_profiles.ids_properties.homogeneous_time"] = 1
        ods["core_profiles.ids_properties.provenance.node.0.sources"] = ["alpha", "beta"]
        path = "core_profiles.ids_properties.provenance.node.0.sources"

        written = {}
        for suffix in (".h5", ".json"):
            target = tmp_path / f"provenance{suffix}"
            ods.save(str(target))
            written[suffix] = np.asarray(vaft_omas.load(target)[path])

        assert written[".h5"].dtype == written[".json"].dtype
        assert list(written[".h5"]) == list(written[".json"]) == ["alpha", "beta"]
        assert written[".h5"].dtype.kind == "U", "object arrays compare unequal by dtype"

    def test_numeric_leaves_are_left_alone(self, written):
        """The decode must not touch, re-type or re-shape any array."""
        h5 = vaft_omas.load(written[".h5"])
        reference = _sample()
        for path in ("soft_x_rays.time", "soft_x_rays.channel.0.brightness.data"):
            got = np.asarray(h5[path])
            want = np.asarray(reference[path])
            assert got.dtype == want.dtype, path
            assert got.shape == want.shape, path
            np.testing.assert_array_equal(got, want)

    def test_an_integer_image_keeps_its_narrow_width(self, tmp_path):
        """Camera frames are stored int32 on purpose; decoding must not widen them."""
        ods = ODS(consistency_check=True)
        ods["camera_visible.ids_properties.homogeneous_time"] = 1
        ods["camera_visible.time"] = np.array([0.3])
        ods["camera_visible.channel.0.detector.0.frame.0.image_raw"] = np.full((4, 4), 7)
        ods["camera_visible.channel.0.detector.0.frame.0.time"] = 0.3
        ods.consistency_check = False
        path = "camera_visible.channel.0.detector.0.frame.0.image_raw"
        ods[path] = np.asarray(ods[path]).astype(np.int32)

        target = tmp_path / "camera.h5"
        ods.save(str(target))

        loaded = np.asarray(vaft_omas.load(target)[path])
        assert loaded.dtype == np.int32
        np.testing.assert_array_equal(loaded, np.full((4, 4), 7))


class TestTheDecodeHelper:
    def test_a_byte_string_becomes_text(self):
        assert _as_text(np.bytes_(b"Vertical SXR Ch 1")) == "Vertical SXR Ch 1"
        assert _as_text(b"plain") == "plain"

    @pytest.mark.parametrize("dtype", ("S2", object))
    def test_an_array_of_byte_strings_becomes_text(self, dtype):
        """Fixed-width and the object form HDF5 uses for variable-length strings."""
        out = _as_text(np.array([b"a", b"bb"], dtype=dtype))
        assert list(out) == ["a", "bb"]
        assert out.shape == (2,)
        assert out.dtype.kind == "U"

    def test_an_object_array_holding_no_bytes_is_declined(self):
        assert _as_text(np.array(["a", "bb"], dtype=object)) is None

    def test_an_empty_string_array_stays_a_string_array(self):
        """Letting NumPy infer from an empty list gives float64.

        That turns a declared-but-empty string leaf numeric, which is a
        `structure.type` difference against the JSON path -- the class of
        mismatch this whole normalization exists to remove.
        """
        out = _as_text(np.array([], dtype="S5"))
        assert out is not None
        assert out.dtype.kind == "U", f"empty leaf became {out.dtype}"
        assert out.shape == (0,)

    def test_a_zero_dimensional_byte_array_becomes_a_plain_string(self):
        """A 0-d array is a scalar leaf; the checker gives `str` for one.

        Returning a 0-d array would leave it comparing unequal by type against
        the JSON path for that leaf shape.
        """
        out = _as_text(np.array(b"Vertical SXR Ch 1"))
        assert isinstance(out, str)
        assert out == "Vertical SXR Ch 1"

    def test_a_mixed_object_array_is_declined_rather_than_stringified(self):
        """One stray byte string must not take `str()` to the rest.

        Requiring every element to be bytes, not any, keeps a number a number.
        """
        assert _as_text(np.array([b"a", 3], dtype=object)) is None

    def test_an_empty_object_array_is_declined(self):
        assert _as_text(np.array([], dtype=object)) is None

    @pytest.mark.parametrize(
        "value",
        ["already text", 3, 3.5, np.array([1.0, 2.0]), np.array([1, 2], dtype=np.int32), None],
    )
    def test_everything_else_is_declined(self, value):
        assert _as_text(value) is None

    def test_undecodable_bytes_do_not_raise(self):
        """A malformed artifact must still open -- that is why the checker is off."""
        assert _as_text(b"\xff\xfe") is not None
