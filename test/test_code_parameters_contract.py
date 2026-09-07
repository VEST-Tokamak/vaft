"""What `code.parameters` may hold, measured through the Access Layer (#561).

`code.parameters` is a `STR_0D` in the Data Dictionary -- one string, XML by
IMAS habit -- and #380 found the consequence the hard way: 4096 paths of EFIT
collection provenance reached the local FileDB product and never reached the
HSDS replica.  #478 found the same loss for a flat block and fixed it the other
way, with `vaft.database._local._promote_code_parameters`.  Neither issue wrote
down what the field actually supports, and #527 now wants to embed a versioned
DCON payload in it, so this module measures every shape end to end rather than
arguing from either precedent.

The mechanism, which is not what either issue assumed:

* **In memory nothing is wrong.**  omas builds a `CodeParameters` object for
  any sub-path written under `code.parameters`, however deep, and serializes
  that to XML on the way to an entry.  A freshly built ODS survives whatever
  its shape -- which is exactly why the defect is invisible to the code that
  writes it.
* **The stage product is where the shape is lost.**  A local product is JSON or
  HDF5 (`vaft.database.filedb.OMAS_PRODUCT_SUFFIX`), and a reload restores the
  block as a plain ODS branch, not a `CodeParameters`.  The Access Layer
  discards a plain branch: no exception, no returned path, one `WARNING: ... is
  not part of IMAS` line on stderr that a pipeline log buries.
* **The loader rescues one shape only.**  `vaft.omas.load` promotes a *flat,
  leaf-only* block back to `CodeParameters`.  A nested block is declined by
  design, so it never reaches a replica -- and any flat leaf sitting beside it
  goes down with it, which is how an EFIT product loses its declared COCOS
  index (see `test_a_flat_leaf_beside_a_nested_block_is_dropped_with_it`).

Every test below writes through the real Access Layer, which is a hard
dependency (`imas_core`, `imas_python`) exercised in CI by
`test_local_importers.py`.  One round trip costs about a quarter of a second.
"""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET

import pytest
from omas import ODS
from omas.omas_core import CodeParameters

import vaft
from vaft.imas import IMAS_DD_VERSION_CONVERSION
from vaft.imas.omas_imas import save_omas_imas

IDS = "equilibrium"
FIELD = f"{IDS}.code.parameters"

#: The envelope `vaft.machine_mapping.mhd_linear` writes, in miniature.
MHD_LINEAR_XML = (
    '<parameters><solver name="dcon" n_tor="1"><mlow>-8</mlow></solver></parameters>'
)


def _product(**leaves) -> ODS:
    """A minimal one-IDS ODS carrying the given `code.parameters` content."""
    ods = ODS(consistency_check=False)
    ods[f"{IDS}.ids_properties.homogeneous_time"] = 2
    ods[f"{IDS}.code.name"] = "probe"
    for path, value in leaves.items():
        ods[f"{FIELD}{'.' + path if path else ''}"] = value
    return ods


def _through_the_product_leg(tmp_path, ods, *, promote=True):
    """Save as a stage product, load it back, write it to an IMAS entry.

    This is the pipeline `vaft.database.replication` runs: a JSON product on
    disk, `vaft.omas.load`, then an Access-Layer write.  ``promote=False``
    bypasses `vaft.omas.load` for the one test that isolates what the promotion
    is doing.

    Returns ``(paths_the_entry_accepted, value_after_the_round_trip)``, where
    the value is ``None`` when the field did not survive.
    """
    product = tmp_path / "stage.json"
    vaft.omas.save(ods, product)
    if promote:
        reloaded = vaft.omas.load(product)
    else:
        reloaded = ODS(consistency_check=False)
        reloaded.load(str(product), consistency_check=False)

    entry = tmp_path / "entry"
    written = save_omas_imas(
        reloaded,
        uri="imas:hdf5?path=" + str(entry),
        new=True,
        verbose=False,
        imas_version=IMAS_DD_VERSION_CONVERSION,
    )
    accepted = [tuple(path) for path in written if tuple(path)[:3] == (IDS, "code", "parameters")]
    with vaft.imas.load(entry) as handle:
        restored = handle.to_omas()
    value = restored[FIELD] if FIELD in restored else None
    return accepted, value


# --- what the field is in memory -------------------------------------------


def test_any_subpath_becomes_a_code_parameters_object_in_memory():
    """Why the loss is invisible where it is written.

    omas gives `code.parameters` a `CodeParameters` rather than an ODS branch
    for any sub-path, at any depth, so the producer sees a well-formed object
    and an in-memory write reaches an entry whatever its shape.  Nothing about
    a nested write looks wrong until the product is reloaded.
    """
    nested = _product(**{"time_slice.0.aeqdsk.terror": 2.5e-6})
    flat = _product(cocos=11)

    assert isinstance(nested[FIELD], CodeParameters)
    assert isinstance(flat[FIELD], CodeParameters)


# --- the shapes that survive the product leg --------------------------------


def test_a_json_string_survives_the_product_and_comes_back_verbatim(tmp_path):
    """One string is the only shape that returns byte-identical.

    `CodeParameters.from_string` refuses it -- JSON is not XML -- so nothing
    re-parses it and the payload a consumer wrote is the payload it reads.
    """
    payload = json.dumps({"efit_collection": {"status": "completed"}}, sort_keys=True)

    accepted, value = _through_the_product_leg(tmp_path, _product(**{"": payload}))

    assert accepted == [(IDS, "code", "parameters")]
    assert isinstance(value, str)
    assert value == payload


def test_an_xml_string_survives_but_comes_back_a_tree(tmp_path):
    """The round trip is not symmetric, and consumers have to know it.

    An XML string reaches the entry unchanged, but the loader parses anything
    XML-shaped into a `CodeParameters`.  A reader that runs `ET.fromstring` or
    a regular expression over this field works on a freshly mapped ODS and
    silently finds nothing on one loaded from an entry.
    """
    accepted, value = _through_the_product_leg(tmp_path, _product(**{"": MHD_LINEAR_XML}))

    assert accepted == [(IDS, "code", "parameters")]
    assert not isinstance(value, str)  # one fragment; see the several-fragment case below
    assert isinstance(value, CodeParameters)
    # The content is all there; only its form changed.  Attributes carry an
    # `@` prefix, which is the shape a consumer must accept after a load.
    assert value["solver"]["@name"] == "dcon"


def test_an_envelope_of_several_fragments_comes_back_a_string(tmp_path):
    """Which of the two forms you get depends on the document, not on the field.

    omas decodes the envelope with a parser that cannot represent repeated
    sibling elements: two `<solver>` fragments make it raise internally, and it
    leaves the string alone.  So a single-module shot returns a tree and a
    multi-module shot -- the ordinary case, since every solver appends its own
    fragment -- returns the string.

    A consumer of this field must therefore accept both forms whatever it
    knows about the producer; `vaft.plot.backend.recipes._mhd_linear_radial_stride`
    does, and says why.
    """
    two_solvers = (
        '<parameters><solver name="dcon" n_tor="1"/>'
        '<solver name="rdcon" n_tor="2"/></parameters>'
    )

    accepted, value = _through_the_product_leg(tmp_path, _product(**{"": two_solvers}))

    assert accepted == [(IDS, "code", "parameters")]
    assert isinstance(value, str)
    assert value == two_solvers


def test_a_flat_block_survives_only_because_the_loader_promotes_it(tmp_path):
    """`_promote_code_parameters` is load-bearing, not defensive (#478).

    The product leg turns the block into a plain ODS branch.  Promoted, it is
    a `CodeParameters` again and reaches the entry; unpromoted, the entry never
    hears about it.
    """
    leaves = {"cocos": 11, "cocos_source": "declared"}

    accepted, value = _through_the_product_leg(tmp_path, _product(**leaves))
    assert accepted == [(IDS, "code", "parameters")]
    assert value["cocos"] == 11
    assert value["cocos_source"] == "declared"

    bypassed, lost = _through_the_product_leg(
        tmp_path / "unpromoted", _product(**leaves), promote=False
    )
    assert bypassed == []
    assert lost is None


# --- the shape that does not ------------------------------------------------


def test_a_nested_block_is_dropped_on_the_way_to_an_entry(tmp_path):
    """#380's loss, reproduced end to end.

    The promotion declines a nested block -- `CodeParameters` cannot address a
    per-slice parser cache -- so the Access Layer sees a plain branch and drops
    it.  This is the EFIT `time_slice` shape, and it is why that cache is a
    local-product convenience and never provenance.
    """
    ods = _product(**{"time_slice.0.aeqdsk.terror": 2.5e-6})

    accepted, value = _through_the_product_leg(tmp_path, ods)

    assert accepted == []
    assert value is None


def test_nothing_raises_and_no_path_is_returned_for_what_was_dropped(tmp_path):
    """The silence is the defect: the write reports success either way.

    `save_omas_imas` returns the paths it actually wrote, and that list is the
    only machine-readable evidence a caller gets.  A dropped field is simply
    absent from it -- no exception, no return code, nothing a pipeline checks.
    """
    ods = _product(**{"time_slice.0.aeqdsk.terror": 2.5e-6})
    product = tmp_path / "stage.json"
    vaft.omas.save(ods, product)
    reloaded = vaft.omas.load(product)

    written = save_omas_imas(
        reloaded,
        uri="imas:hdf5?path=" + str(tmp_path / "entry"),
        new=True,
        verbose=False,
        imas_version=IMAS_DD_VERSION_CONVERSION,
    )

    assert written, "the write did happen; only the parameters were discarded"
    assert not [path for path in written if "parameters" in path]


def test_a_flat_leaf_beside_a_nested_block_is_dropped_with_it(tmp_path):
    """The consequence VAFT actually ships, and the reason to record all this.

    `vaft.omas.general` writes the declared COCOS index as a flat leaf of
    `equilibrium.code.parameters`; the EFIT mappers write their per-slice
    parser cache into the same field.  Promotion is all-or-nothing on the
    block, so an EFIT product loses the COCOS index it declares -- the leaf is
    not nested, it is merely standing next to something that is.

    Tracked as a live defect in issue #561; this test pins the behaviour, it
    does not bless it.
    """
    ods = _product(**{"time_slice.0.aeqdsk.terror": 2.5e-6, "cocos": 11})

    accepted, value = _through_the_product_leg(tmp_path, ods)

    assert accepted == []
    assert value is None, "cocos went down with the parser cache beside it"


# --- what the envelope does to values ---------------------------------------


def test_values_inside_the_envelope_are_reinterpreted_on_the_way_back(tmp_path):
    """Escaped text in the XML envelope is not returned verbatim.

    `CodeParameters.from_string` runs every text node through
    `ast.literal_eval`, so an EFIT case label -- a string that happens to look
    like a number -- comes back as a float.  This is the reason arbitrary keys
    *and arbitrary values* belong inside a payload the XML loader cannot parse,
    which in this repository means JSON.
    """
    label = "039915.00316"

    _, from_envelope = _through_the_product_leg(tmp_path, _product(label=label))
    _, from_json = _through_the_product_leg(
        tmp_path / "json", _product(**{"": json.dumps({"label": label})})
    )

    assert from_envelope["label"] != label
    assert from_envelope["label"] == pytest.approx(39915.00316)
    assert json.loads(from_json)["label"] == label


# --- the mhd_linear envelope (#527 builds on this) --------------------------


def test_the_mhd_linear_envelope_survives_the_product_leg_as_a_string(tmp_path):
    """A stage product keeps the field a string, so `ET.fromstring` still works.

    `vaft.omas.save`/`load` is the local leg every stage product takes, and it
    does not parse the field -- only the Access Layer does.  This is what
    `test_machine_mapping_mhd_linear.py` and the replication test rely on.
    """
    product = tmp_path / "stage.json"
    vaft.omas.save(_product(**{"": MHD_LINEAR_XML}), product)

    value = vaft.omas.load(product)[FIELD]

    assert isinstance(value, str)
    assert ET.fromstring(value).find(".//solver").get("name") == "dcon"


def test_a_fragment_carrying_the_close_tag_would_corrupt_the_envelope():
    """The splice hazard #527's payload has to respect.

    `vaft.machine_mapping.mhd_linear._append_code_parameters` extends the field
    by string surgery on the trailing `</parameters>`.  A fragment whose own
    text contains that literal closes the document early, and everything
    appended afterwards lands outside the root element -- where a parser
    refuses it.  A payload must escape it; nothing in the producer checks.
    """
    from vaft.machine_mapping.mhd_linear import _append_code_parameters

    ods = ODS(consistency_check=False)
    _append_code_parameters(ods, IDS, "<solver name='dcon'>free text </parameters></solver>",
                            code_name="DCON")
    _append_code_parameters(ods, IDS, "<solver name='rdcon'/>", code_name="GPEC-suite")

    with pytest.raises(ET.ParseError):
        ET.fromstring(ods[FIELD])
