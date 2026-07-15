"""Unit tests for TES parameter-scan argument validation.

``scan_tes`` feeds ``param`` straight into ``dataclasses.replace``, so a typo
used to surface as an opaque ``TypeError`` from deep inside the stdlib. It is
validated against ``TESConfig`` up front instead.
"""

import pytest

from vaft.code.tes import TESConfig, scan_tes


def test_scan_rejects_unknown_param():
    with pytest.raises(ValueError, match="Unknown TESConfig field"):
        scan_tes(
            ods=None,
            base_config=TESConfig(),
            values=[100.0],
            param="ip0_ka",  # correct spelling is ip0_kA
        )


def test_scan_error_lists_valid_fields():
    with pytest.raises(ValueError) as excinfo:
        scan_tes(ods=None, base_config=TESConfig(), values=[100.0], param="nonsense")

    message = str(excinfo.value)
    assert "ip0_kA" in message
    assert "betap" in message


def test_scan_validates_before_touching_the_ods():
    # ods=None would fail loudly downstream; the ValueError proves validation
    # happens before any solve is attempted.
    with pytest.raises(ValueError):
        scan_tes(ods=None, base_config=TESConfig(), values=[], param="bogus")
