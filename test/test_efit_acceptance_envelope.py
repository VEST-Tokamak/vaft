"""EFIT's acceptance envelope, and the VEST one derived from the machine (#171).

The claim being pinned is narrow and it is the one that matters: the geometric
bounds are derived from the limiter, never fitted to a discharge. A test that
only checked "the proposal accepts 39915" would pass for a bound tuned until
it did, which would prove nothing.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from dataclasses import fields, replace
from pathlib import Path

import f90nml
import numpy as np
import pytest

from vaft.code.efit.config import EFITAcceptanceEnvelope
from vaft.data.resources import data_path
from vaft.machine_mapping.efund_geometry import vest_acceptance_envelope

TABLE = Path(__file__).resolve().parent / "data" / "efit_acceptance_envelope.json"
SCRIPT = Path(__file__).resolve().parents[1] / "workflow" / "efit_numerics" / "acceptance_envelope.py"


@pytest.fixture(scope="module")
def module():
    spec = importlib.util.spec_from_file_location("acceptance_envelope", SCRIPT)
    loaded = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = loaded
    try:
        spec.loader.exec_module(loaded)
    except Exception:
        del sys.modules[spec.name]
        raise
    yield loaded
    sys.modules.pop(spec.name, None)


@pytest.fixture(scope="module")
def static():
    from vaft.omas.vest_upstream import build_static_ods

    return build_static_ods("vest-pre-43017-pf1906")


@pytest.fixture(scope="module")
def table():
    return json.loads(TABLE.read_text(encoding="utf-8"))


def test_the_defaults_are_the_packaged_bounds_not_something_new():
    """The type must describe what EFIT reads today before it proposes anything."""
    bundled = f90nml.read(str(data_path("efit/mhdin.dat")))["incheck"]
    envelope = EFITAcceptanceEnvelope()
    for field in fields(EFITAcceptanceEnvelope):
        if field.name in bundled:
            assert getattr(envelope, field.name) == pytest.approx(float(bundled[field.name])), field.name
    assert set(envelope.to_namelist()) >= set(bundled)


def test_the_envelope_refuses_bounds_that_cannot_hold_a_plasma():
    with pytest.raises(ValueError, match="aminor_min must be below aminor_max"):
        EFITAcceptanceEnvelope(aminor_min=80.0)
    with pytest.raises(ValueError, match="a plasma has a size"):
        EFITAcceptanceEnvelope(aminor_min=0.0)
    with pytest.raises(ValueError, match="rcntr_min must be below rcntr_max"):
        EFITAcceptanceEnvelope(rcntr_min=200.0)
    assert EFITAcceptanceEnvelope().sha256 != EFITAcceptanceEnvelope(aminor_min=5.0).sha256


def test_the_geometric_bounds_come_from_the_limiter_and_the_grid(static):
    """Derived, not fitted: every geometric bound is a function of the machine."""
    ods, _ = static
    outline = ods["wall.description_2d.0.limiter.unit.0.outline"]
    r = np.asarray(outline["r"], dtype=float) * 100.0
    z = np.asarray(outline["z"], dtype=float) * 100.0
    cell = (1.2 - 0.05) / 128 * 100.0

    envelope = vest_acceptance_envelope(ods)
    assert envelope.aminor_max == pytest.approx(0.5 * (r.max() - r.min()))
    assert envelope.aminor_min == pytest.approx(5 * cell)
    assert envelope.rcntr_min == pytest.approx(r.min() + envelope.aminor_min)
    assert envelope.rcntr_max == pytest.approx(r.max() - envelope.aminor_min)
    assert envelope.zcntr_min == pytest.approx(z.min() + envelope.aminor_min)
    assert envelope.rcurrt_min == envelope.rcntr_min

    # The floor scales with the grid, which is what makes it a resolution
    # statement rather than a number someone liked.
    coarser = vest_acceptance_envelope(ods, nw=65)
    assert coarser.aminor_min == pytest.approx(2 * envelope.aminor_min, rel=1e-6)


def test_the_physics_bounds_are_left_alone(static):
    """Geometry is derived; li, betap, qstar and plasma_diff are not touched."""
    ods, _ = static
    packaged = EFITAcceptanceEnvelope()
    proposed = vest_acceptance_envelope(ods)
    for name in (
        "li_min", "li_max", "betap_max", "betat_max", "qstar_min", "qstar_max",
        "qout_min", "qout_max", "elong_min", "elong_max", "plasma_diff",
    ):
        assert getattr(proposed, name) == getattr(packaged, name), name


def test_the_virial_checks_do_not_gate_vest_acceptance(static):
    """Issue #649, and a policy decision rather than a derivation.

    At A ~ 1.45 `sbpp` is a difference of two terms near 0.8 that leaves 0.01
    to 0.2 and goes negative, and `sbli` divides by `alpha - 1`. A gate that
    cannot tell a good reconstruction from a bad one must not decide
    acceptance. The quantities stay computed and written to the a-file; only
    the rejection stops, and it can be switched back on explicitly.
    """
    from vaft.code.efit.config import IGNORE_CRITERION

    ods, _ = static
    disabled = vest_acceptance_envelope(ods)
    assert disabled.delbp_diff == IGNORE_CRITERION
    assert disabled.dbpli_diff == IGNORE_CRITERION
    # EFIT tests `value >= tolerance`, so the criterion cannot fire.
    assert IGNORE_CRITERION > 1e3

    kept = vest_acceptance_envelope(ods, virial_checks=True)
    assert kept.delbp_diff == EFITAcceptanceEnvelope().delbp_diff
    assert kept.dbpli_diff == EFITAcceptanceEnvelope().dbpli_diff
    # Only these two move: the decision is about the virial gate, nothing else.
    from dataclasses import fields

    differing = {
        field.name
        for field in fields(EFITAcceptanceEnvelope)
        if getattr(disabled, field.name) != getattr(kept, field.name)
    }
    assert differing == {"delbp_diff", "dbpli_diff"}


def test_a_supplied_base_survives_the_derivation(static):
    ods, _ = static
    base = EFITAcceptanceEnvelope(li_max=2.0, plasma_diff=0.2)
    proposed = vest_acceptance_envelope(ods, base=base)
    assert proposed.li_max == 2.0 and proposed.plasma_diff == 0.2
    assert proposed.aminor_min != base.aminor_min
    # The virial decision overrides the base: it is a policy, not a default.
    assert proposed.delbp_diff != base.delbp_diff


def test_the_derivation_refuses_a_grid_too_coarse_for_the_machine(static):
    ods, _ = static
    with pytest.raises(ValueError, match="resolved floor"):
        vest_acceptance_envelope(ods, nw=5)


def test_the_audit_names_the_bounds_that_reject_real_equilibria(table):
    """The finding: three geometric bounds reject VEST plasmas for being small."""
    offenders = table["summary"]["rejecting_real_equilibria"]
    assert set(offenders) == {"aminor", "rcntr", "rcurrt"}
    by_name = {row["scalar"]: row for row in table["criteria"]}
    assert by_name["aminor"]["packaged"] == [25.0, 75.0]
    assert by_name["aminor"]["outside_packaged_excluding_collapsed"] > 0
    assert by_name["aminor"]["median"] < by_name["aminor"]["packaged"][0]
    for row in table["criteria"]:
        assert row["outside_proposed_excluding_collapsed"] == 0, row["scalar"]


def test_the_audit_leaves_the_passing_criteria_untouched(table):
    """A proposal that moved bounds nothing violated would be loosening, not fixing."""
    for row in table["criteria"]:
        if not row["rejects_real_equilibria"]:
            assert row["changed"] is (row["scalar"] in {"zcntr", "zcurrt"}), row["scalar"]


def test_write_mhdin_emits_the_envelope_and_leaves_the_geometry_alone(static, tmp_path):
    """EFUND ignores &incheck; EFIT reads it from the same file."""
    from vaft.code.efit.efund import EFUNDConfig, write_mhdin
    from vaft.machine_mapping.efund_geometry import efund_geometry_from_static

    ods, manifest = static
    geometry = efund_geometry_from_static(ods, manifest=manifest)
    config = EFUNDConfig(workdir=tmp_path)
    envelope = vest_acceptance_envelope(ods)

    plain = f90nml.read(str(write_mhdin(geometry, config, tmp_path / "plain.dat")))
    with_envelope = f90nml.read(
        str(write_mhdin(geometry, config, tmp_path / "envelope.dat", envelope=envelope))
    )

    assert "incheck" not in plain
    assert with_envelope["incheck"]["aminor_min"] == pytest.approx(envelope.aminor_min)
    assert len(with_envelope["incheck"]) == len(envelope.to_namelist())
    # The table itself must be unaffected: the envelope decides acceptance,
    # not geometry, and a table generated with one must equal a table
    # generated without.
    assert plain["machinein"] == with_envelope["machinein"]
    for key in ("rvs", "zvs", "rf", "zf", "xmp2", "rsi"):
        assert plain["in3"][key] == with_envelope["in3"][key], key
    assert plain["in5"] == with_envelope["in5"]
