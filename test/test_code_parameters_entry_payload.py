"""The save half of the `code.parameters` pair, in isolation (#642).

`test_code_parameters_contract.py` measures what an entry does with each shape.
This module tests the thing that decides which shape it is given: what is kept,
what is left behind, what is said about it, and -- the part a caller depends on
without ever asking -- that the ODS it handed over comes back unchanged.
"""

from __future__ import annotations

import logging

import numpy as np
import pytest
from omas import ODS
from omas.omas_core import CodeParameters

from vaft.imas.code_parameters import (
    CACHE_OMITTED_KEY,
    as_entry_payload,
    promote_in_place,
)

FIELD = "equilibrium.code.parameters"


def _ods(**leaves) -> ODS:
    ods = ODS(consistency_check=False)
    ods["equilibrium.ids_properties.homogeneous_time"] = 2
    for path, value in leaves.items():
        ods[f"{FIELD}.{path}"] = value
    return ods


def test_a_string_is_left_exactly_as_it_is():
    """The one shape that needs nothing done to it."""
    ods = ODS(consistency_check=False)
    ods[FIELD] = '<parameters><solver name="dcon"/></parameters>'

    with as_entry_payload(ods) as prepared:
        assert prepared[FIELD] == '<parameters><solver name="dcon"/></parameters>'


def test_a_flat_block_travels_whole_and_unannotated():
    """Nothing was left behind, so nothing is said about it."""
    ods = _ods(cocos=11, cocos_source="declared")

    with as_entry_payload(ods) as prepared:
        payload = prepared[FIELD]
        assert isinstance(payload, CodeParameters)
        assert dict(payload) == {"cocos": 11, "cocos_source": "declared"}
        assert CACHE_OMITTED_KEY not in payload


def test_a_nested_block_is_named_rather_than_carried():
    ods = _ods(**{"time_slice.0.aeqdsk.terror": 2.5e-6, "cocos": 11})

    with as_entry_payload(ods) as prepared:
        payload = prepared[FIELD]
        assert payload["cocos"] == 11
        assert "time_slice" not in payload
        assert payload[CACHE_OMITTED_KEY] == "time_slice"


def test_an_array_leaf_counts_as_something_an_entry_cannot_carry():
    """Measured, not assumed: the XML encoder has no array representation.

    An array put through it returns as the repr of its elements joined by
    spaces -- present, named and useless.  Leaving it behind and saying so
    beats carrying a corrupted copy.
    """
    ods = ODS(consistency_check=False)
    ods["equilibrium.ids_properties.homogeneous_time"] = 2
    block = CodeParameters()
    block["cocos"] = 11
    block["weights"] = np.array([1.0, 2.0, 3.0])
    ods[FIELD] = block

    with as_entry_payload(ods) as prepared:
        payload = prepared[FIELD]
        assert payload["cocos"] == 11
        assert "weights" not in payload
        assert payload[CACHE_OMITTED_KEY] == "weights"


def test_every_omitted_name_is_listed():
    ods = _ods(**{"time_slice.0.x": 1.0, "artifacts.gfile.sha256": "abc", "cocos": 11})

    with as_entry_payload(ods) as prepared:
        omitted = prepared[FIELD][CACHE_OMITTED_KEY]

    assert set(omitted.split(", ")) == {"time_slice", "artifacts"}


def test_the_caller_gets_its_own_ods_back():
    """The write borrows the ODS; it does not get to keep the change.

    Content, not object identity: OMAS copies a `CodeParameters` on
    assignment, so no code anywhere can hand back the same object through
    ``ods[path] = ...``.  What a caller can rely on is that everything it put
    there is there afterwards, and that nothing was added.
    """
    ods = _ods(**{"time_slice.0.aeqdsk.terror": 2.5e-6, "cocos": 11})
    before = dict(ods[FIELD])

    with as_entry_payload(ods):
        pass

    assert dict(ods[FIELD]) == before
    assert ods[f"{FIELD}.time_slice.0.aeqdsk.terror"] == 2.5e-6
    assert CACHE_OMITTED_KEY not in ods[FIELD], "the note belongs to the entry, not the ODS"


def test_the_ods_is_restored_even_when_the_write_fails():
    """A failed write must not leave the caller holding a stripped ODS."""
    ods = _ods(**{"time_slice.0.aeqdsk.terror": 2.5e-6, "cocos": 11})
    before = dict(ods[FIELD])

    with pytest.raises(RuntimeError, match="backend is on fire"):
        with as_entry_payload(ods):
            raise RuntimeError("backend is on fire")

    assert dict(ods[FIELD]) == before
    assert ods[f"{FIELD}.time_slice.0.aeqdsk.terror"] == 2.5e-6


def test_what_stayed_behind_is_logged_once_per_field(caplog):
    """A pipeline log is where this becomes visible without reading a replica."""
    ods = _ods(**{"time_slice.0.aeqdsk.terror": 2.5e-6, "cocos": 11})

    with caplog.at_level(logging.INFO, logger="vaft.imas.code_parameters"):
        with as_entry_payload(ods):
            pass

    messages = [record.getMessage() for record in caplog.records]
    assert len(messages) == 1
    assert "equilibrium.code.parameters" in messages[0]
    assert "time_slice" in messages[0]


def test_an_ids_without_the_field_is_not_given_one():
    ods = ODS(consistency_check=False)
    ods["equilibrium.ids_properties.homogeneous_time"] = 2

    with as_entry_payload(ods) as prepared:
        assert FIELD not in prepared


def test_every_ids_is_prepared_not_only_the_first():
    ods = ODS(consistency_check=False)
    ods["equilibrium.code.parameters.time_slice.0.x"] = 1.0
    ods["equilibrium.code.parameters.cocos"] = 11
    ods["core_profiles.code.parameters.fits.0.chi2"] = 0.5
    ods["core_profiles.code.parameters.method"] = "spline"

    with as_entry_payload(ods) as prepared:
        assert prepared["equilibrium.code.parameters"][CACHE_OMITTED_KEY] == "time_slice"
        assert prepared["core_profiles.code.parameters"]["method"] == "spline"
        assert prepared["core_profiles.code.parameters"][CACHE_OMITTED_KEY] == "fits"


# --- the load half ----------------------------------------------------------


def _plain_branch(tmp_path, ods):
    """The shape a stage product restores: a plain ODS branch, not a `CodeParameters`.

    It cannot be built by hand -- OMAS makes a `CodeParameters` for any
    sub-path written under this field -- so it is round-tripped through a
    product, which is where it comes from in the pipeline.
    """
    import vaft

    product = tmp_path / "stage.json"
    vaft.omas.save(ods, product)
    restored = ODS(consistency_check=False)
    restored.load(str(product), consistency_check=False)
    assert not isinstance(restored[FIELD], CodeParameters), "the fixture proves its own premise"
    return restored


def test_promotion_turns_a_flat_branch_into_code_parameters(tmp_path):
    restored = _plain_branch(tmp_path, _ods(cocos=11, cocos_source="declared"))

    promote_in_place(restored)

    assert isinstance(restored[FIELD], CodeParameters)
    assert restored[FIELD]["cocos"] == 11


def test_promotion_leaves_a_nested_branch_alone(tmp_path):
    """It is a local parser cache; what an entry sees is the save half's decision."""
    restored = _plain_branch(
        tmp_path, _ods(**{"time_slice.0.aeqdsk.terror": 2.5e-6, "cocos": 11})
    )

    promote_in_place(restored)

    assert not isinstance(restored[FIELD], CodeParameters)
    assert restored[f"{FIELD}.time_slice.0.aeqdsk.terror"] == 2.5e-6


def test_a_string_field_is_still_a_string_after_a_save(tmp_path):
    """The promise this module makes, on the shape it deliberately skips.

    OMAS's own `codeparams_xml_save` parses any XML-shaped string back into a
    `CodeParameters` when it exits, so without putting it back a save rewrites
    the caller's ODS.  `vaft.plot.backend.recipes._mhd_linear_radial_stride`
    branches on that type, which is a plot behaving differently after a save
    than before it.
    """
    import vaft

    document = '<parameters><solver name="dcon" n_tor="1"/></parameters>'
    ods = ODS(consistency_check=False)
    ods["mhd_linear.ids_properties.homogeneous_time"] = 2
    ods["mhd_linear.code.parameters"] = document

    vaft.imas.save(ods, tmp_path / "entry")

    assert ods["mhd_linear.code.parameters"] == document


def test_an_empty_block_gains_nothing_from_the_split():
    """Nothing to carry means nothing invented.

    A missing `code.parameters` is materialized by the act of reading it (the
    hazard #478 documents), and an empty node is an accident rather than a
    declaration.  The split leaves it exactly as it found it -- no promotion,
    no `parameters_cache_omitted` note.

    What OMAS then does with an empty `CodeParameters` -- it writes
    `<parameters></parameters>` -- predates this module and is not changed
    here.
    """
    ods = ODS(consistency_check=False)
    ods["equilibrium.ids_properties.homogeneous_time"] = 2
    ods[FIELD] = ODS(consistency_check=False)

    with as_entry_payload(ods) as prepared:
        assert dict(prepared[FIELD]) == {}
