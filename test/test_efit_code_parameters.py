"""`equilibrium.code.parameters` must be the string the DD says it is (#380).

The EFIT stage wrote a nested tree there. It looked right in the local product
and was dropped entirely on the way through the IMAS Access Layer, so 4096
paths of collection provenance never reached the HSDS replica. Nothing warned:
the loss is only visible when something reads the replica back.
"""

import importlib.util
import json
from pathlib import Path

import pytest
from omas import ODS


WORKFLOW = (
    Path(__file__).resolve().parents[1]
    / "workflow"
    / "automatic_pipeline_1_routine_data_processing"
)

pytestmark = pytest.mark.skipif(
    not WORKFLOW.exists(), reason="workflow scripts are not part of the distribution"
)


def _module(stem):
    spec = importlib.util.spec_from_file_location(stem, WORKFLOW / f"{stem}.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class _Slice:
    def __init__(self, label, disposition):
        self._d = {"label": label, "disposition": disposition}

    def to_dict(self):
        return dict(self._d)


def _payload():
    return _module("generate_efit_ods").efit_collection_parameters(
        status="completed: returncode=0; gfiles=9",
        slice_statuses=[_Slice("039915.00316", "converged"),
                        _Slice("039915.00319", "rejected")],
        mapping_diagnostics=[{"case": "039915.00316", "differences": []}],
        artifact_hashes={"g039915.00316": "abc123"},
        artifact_manifest={"kfile": ["k039915.00316"]},
    )


def test_the_payload_is_a_string_not_a_tree():
    """A STR_0D field holds one string. Anything else is silently discarded."""
    value = _payload()

    assert isinstance(value, str)
    ods = ODS(consistency_check=False)
    ods["equilibrium.code.parameters"] = value
    # No sub-paths: nothing for the Access Layer to drop.
    nested = [p for p in ods.paths() if len(p) > 3 and p[:3] == ("equilibrium", "code", "parameters")]
    assert nested == []


def test_every_field_survives_a_serialize_parse_cycle():
    """The point of the string is that the provenance still arrives."""
    parsed = json.loads(_payload())["efit_collection"]

    assert parsed["status"].startswith("completed:")
    assert [s["label"] for s in parsed["slice_statuses"]] == [
        "039915.00316",
        "039915.00319",
    ]
    assert parsed["slice_statuses"][1]["disposition"] == "rejected"
    assert parsed["mapping_diagnostics"][0]["case"] == "039915.00316"
    assert parsed["artifact_hashes"]["g039915.00316"] == "abc123"
    assert parsed["artifact_manifest"]["kfile"] == ["k039915.00316"]


def test_numeric_case_labels_stay_inside_the_json():
    """`039915.00316` is a label, not two ODS path segments.

    Keeping the arbitrary keys inside the serialized document is why they can be
    numeric at all -- as ODS paths they would be reinterpreted on reload.
    """
    parsed = json.loads(_payload())["efit_collection"]

    assert "g039915.00316" in parsed["artifact_hashes"]


def test_the_payload_is_deterministic():
    """Two runs over equal inputs must produce byte-identical parameters, or
    replication re-sends a product that did not change."""
    assert _payload() == _payload()


def test_the_stage_no_longer_writes_nested_parameter_paths():
    source = (WORKFLOW / "generate_efit_ods.py").read_text(encoding="utf-8")

    assert 'ods["equilibrium.code.parameters"]' in source
    assert "code.parameters.efit_collection" not in source
