"""The Gaussian recovery backend, outside the constraint builder (issue #296)."""

from __future__ import annotations

import numpy as np
import pytest
from omas import ODS

import vaft.code.efit.recovery as recovery
from vaft.code.efit.legacy import gauss_fit4
from vaft.validation.channel_decision import (
    REASON_CONDEMNED,
    REASON_MANUAL_LIST,
    RECOVERED,
    REJECTED,
    USABLE,
    ChannelDecisions,
    missing_decision,
    rejected_decision,
    usable_decision,
)

TIMES = np.array([0.300, 0.301, 0.302])
COEF = np.array([0.08, 0.0, 0.25, -0.02])
KIND = "b_field_pol_probe"


def _families(n: int = 8) -> recovery.ProbeFamilies:
    return recovery.ProbeFamilies(inboard=np.arange(n), side=np.zeros(0, int), outboard=np.zeros(0, int))


def _equilibrium(n: int = 8, ip=(6.0e4, 6.0e4, 1.0e4)) -> ODS:
    """Probe readings that lie exactly on one Gaussian, so a fit through the
    healthy members reproduces any member's value."""
    EQ = ODS(consistency_check=False)
    for i, current in enumerate(ip):
        EQ[f"time_slice.{i}.constraints.ip.measured"] = current
        for j in range(n):
            EQ[f"time_slice.{i}.constraints.bpol_probe.{j}.measured"] = gauss_fit4(COEF, recovery.PROBE_Z_TABLE[j])
            EQ[f"time_slice.{i}.constraints.bpol_probe.{j}.measured_error_upper"] = 1e-3
    return EQ


def _decisions(n: int = 8, overrides: dict | None = None) -> ChannelDecisions:
    entries = {}
    for j in range(n):
        entries[(KIND, j)] = (overrides or {}).get(j, usable_decision(KIND, j, f"P{j}", TIMES.size))
    return ChannelDecisions(TIMES, entries)


def _run(EQ, decisions, **kwargs):
    return recovery.gaussian_probe_recovery(EQ, decisions, families=_families(), **{"mode": 1, **kwargs})


def test_rejected_and_missing_probes_are_not_fit_points(monkeypatch):
    seen = []
    original = recovery.min_gauss_fit4

    def spy(coef, x, y):
        seen.append(len(x))
        return original(coef, x, y)

    monkeypatch.setattr(recovery, "min_gauss_fit4", spy)
    decisions = _decisions(overrides={
        2: rejected_decision(KIND, 2, "P2", 3, reason=REASON_CONDEMNED),
        5: missing_decision(KIND, 5, "P5", 3),
    })
    EQ = _equilibrium()
    EQ["time_slice.0.constraints.bpol_probe.2.measured"] = 99.0  # would wreck the fit if used
    out = _run(EQ, decisions)
    assert seen and set(seen) == {6}  # 8 members minus the rejected and the missing one
    recovered = out.get(KIND, 2)
    assert recovered.state.tolist() == [RECOVERED, RECOVERED, REJECTED]  # slice 2 is below the Ip floor
    assert recovered.value[0] == pytest.approx(gauss_fit4(COEF, recovery.PROBE_Z_TABLE[2]), rel=1e-4)
    assert out.get(KIND, 5).state.tolist() == [3, 3, 3]  # missing stays missing, never a target


def test_mode_one_recovers_only_rejected_members_and_mode_two_every_member():
    decisions = _decisions(overrides={3: rejected_decision(KIND, 3, "P3", 3, reason=REASON_CONDEMNED)})
    only = _run(_equilibrium(), decisions, mode=1)
    assert [key[1] for key, d in only.entries.items() if d.recovered_mask().any()] == [3]
    every = _run(_equilibrium(), decisions, mode=2)
    assert sorted(key[1] for key, d in every.entries.items() if d.recovered_mask().any()) == list(range(8))
    healthy = every.get(KIND, 0)
    assert healthy.state.tolist() == [RECOVERED, RECOVERED, USABLE]
    assert healthy.weight_factor.tolist() == [1.0, 1.0, 1.0]  # a usable member keeps its weight under mode 2


def test_recovered_carries_value_uncertainty_and_provenance_and_is_distinguishable():
    decisions = _decisions(overrides={3: rejected_decision(KIND, 3, "P3", 3, reason=REASON_CONDEMNED)})
    out = _run(_equilibrium(), decisions)
    recovered = out.get(KIND, 3)
    assert recovered.provenance == recovery.GAUSSIAN_PROVENANCE
    assert np.isfinite(recovered.value[:2]).all() and np.isfinite(recovered.uncertainty[:2]).all()
    assert recovered.uncertainty[0] >= 0.0
    assert "recovered_by_gaussian_fit4" in recovered.reasons and REASON_CONDEMNED in recovered.reasons
    assert out.get(KIND, 0).provenance == "measured"
    assert recovered.as_dict()["state"] == ["recovered", "recovered", "rejected"]


def test_a_manual_rejection_recovers_with_nominal_weight_but_a_validity_rejection_does_not():
    decisions = _decisions(overrides={
        3: rejected_decision(KIND, 3, "P3", 3, reason=f"{REASON_MANUAL_LIST}:4"),
        4: rejected_decision(KIND, 4, "P4", 3, reason=REASON_CONDEMNED),
        6: rejected_decision(KIND, 6, "P6", 3, reason=(REASON_CONDEMNED, f"{REASON_MANUAL_LIST}:7")),
    })
    out = _run(_equilibrium(), decisions)
    assert out.get(KIND, 3).weight_factor.tolist() == [1.0, 1.0, 0.0]
    assert out.get(KIND, 4).weight_factor.tolist() == [0.0, 0.0, 0.0]
    assert out.get(KIND, 6).weight_factor.tolist() == [0.0, 0.0, 0.0]
    assert out.get(KIND, 4).state.tolist() == [RECOVERED, RECOVERED, REJECTED]


def test_no_fit_below_the_ip_floor_and_the_input_is_untouched():
    decisions = _decisions(overrides={3: rejected_decision(KIND, 3, "P3", 3, reason=REASON_CONDEMNED)})
    EQ = _equilibrium(ip=(1.0e4, 1.0e4, 1.0e4))
    before = float(EQ["time_slice.0.constraints.bpol_probe.3.measured"])
    out = _run(EQ, decisions)
    assert out.get(KIND, 3).state.tolist() == [REJECTED] * 3
    assert out.get(KIND, 3) is decisions.get(KIND, 3)
    assert float(EQ["time_slice.0.constraints.bpol_probe.3.measured"]) == before


def test_legacy_uncertainty_mode_keeps_the_builders_error():
    decisions = _decisions(overrides={3: rejected_decision(KIND, 3, "P3", 3, reason=REASON_CONDEMNED)})
    out = _run(_equilibrium(), decisions, uncertainty="legacy")
    assert out.get(KIND, 3).uncertainty is None
    with pytest.raises(ValueError, match="uncertainty must be"):
        _run(_equilibrium(), decisions, uncertainty="guess")
    with pytest.raises(ValueError, match="mode must be"):
        _run(_equilibrium(), decisions, mode=3)


def test_vest_probe_families_are_disjoint_and_cover_the_efit_probes():
    from vaft.omas.sample import sample_ods

    ods = sample_ods()
    families = recovery.probe_families(ods["magnetics"], count=64)
    sets = [set(families.inboard.tolist()), set(families.side.tolist()), set(families.outboard.tolist())]
    assert [len(s) for s in sets] == [27, 16, 21]
    assert not (sets[0] & sets[1]) and not (sets[1] & sets[2]) and not (sets[0] & sets[2])
    assert sets[0] | sets[1] | sets[2] == set(range(64))
    assert recovery.family_of(0, families) == "inboard"
    assert recovery.family_of(int(families.outboard[0]), families) == "outboard"
    assert recovery.family_of(int(families.side[0]), families) == "side"
    assert len(recovery.PROBE_Z_TABLE) == 64
