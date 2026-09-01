"""What signs a normal VEST discharge must carry to mean what IMAS says it means.

Issue #288. The contract comes first and the constants follow from it: IMAS
defines positive ``Ip`` and positive ``B0`` as counter-clockwise viewed from
above, so the working physical assumption -- ``Ip`` clockwise, ``Bt``
counter-clockwise -- fixes the mapped discharge signs outright::

    Ip_imas < 0        B0_imas > 0

Two things are deliberately kept apart here:

* **The IMAS target signs.** Settled. A convention question; no measurement
  bears on it.
* **The diagnostic polarities.** Open. Whether a positive VEST Rogowski signal
  really is a clockwise plasma current, and whether a positive stored ``bt_vest``
  really is a counter-clockwise field. Hardware questions.

`confirmed = False` on the transformations refers only to the second.
"""

from __future__ import annotations

import numpy as np
import pytest

SHOTS = (39915, 41524, 41672)


def _load(shot):
    import vaft

    return vaft.omas.load(vaft.data.sample(shot, representation="omas"))


def _native_discharge(ods):
    """The VEST-native ``(ip, b0)`` pair the mapping writes for a discharge."""
    ip = np.asarray(ods["magnetics.ip.0.data"], float)
    peak = float(ip[np.nanargmax(np.abs(ip))])
    b0 = float(np.median(np.asarray(ods["equilibrium.vacuum_toroidal_field.b0"], float)))
    return peak, b0


# --- the contract itself ---------------------------------------------------


def test_the_discharge_contract_is_stated_explicitly():
    """Ip negative, B0 positive, antiparallel -- named, not left implicit."""
    from vaft.machine_mapping import IMAS_DISCHARGE_SIGNS

    assert IMAS_DISCHARGE_SIGNS.ip == -1, "clockwise Ip is negative in IMAS"
    assert IMAS_DISCHARGE_SIGNS.b0 == +1, "counter-clockwise Bt is positive in IMAS"
    assert IMAS_DISCHARGE_SIGNS.antiparallel is True


def test_the_contract_is_a_convention_fact_not_a_measurement():
    """It follows from the assumption alone, so no datum can settle or unsettle it.

    This is the split the module is built around: the target signs are known
    while the diagnostic polarities that would let us *trust* the mapped signals
    are not.
    """
    from vaft.machine_mapping import (
        IMAS_DISCHARGE_SIGNS,
        equilibrium_orientation_is_resolved,
        sign_transformations,
    )

    # The contract is definite ...
    assert IMAS_DISCHARGE_SIGNS.ip in (-1, +1)
    assert IMAS_DISCHARGE_SIGNS.b0 in (-1, +1)
    # ... while every polarity behind it is still open.
    assert all(t.confirmed is False for t in sign_transformations())
    assert equilibrium_orientation_is_resolved() is False


def test_a_contract_rejects_a_value_that_is_not_a_sign():
    from vaft.machine_mapping import DischargeSignContract

    with pytest.raises(ValueError, match="must be \\+1 or -1"):
        DischargeSignContract(ip=0, b0=+1)


# --- discharge level, on representative shots ------------------------------


def test_transformed_discharges_satisfy_the_contract_on_every_packaged_shot():
    """The point of the transformations, checked on real mapped signals.

    Takes what the mapping actually writes for each representative discharge,
    applies the two declared transformations, and requires the result to comply
    with the contract.
    """
    from vaft.machine_mapping import IMAS_DISCHARGE_SIGNS, to_imas_discharge_signs

    for shot in SHOTS:
        ip_vest, bt_vest = _native_discharge(_load(shot))
        ip_imas, b0_imas = to_imas_discharge_signs(ip_vest, bt_vest)
        assert ip_imas < 0, f"shot {shot}: Ip must be negative in IMAS terms"
        assert b0_imas > 0, f"shot {shot}: B0 must be positive in IMAS terms"
        assert IMAS_DISCHARGE_SIGNS.satisfied_by(ip_imas, b0_imas), shot


def test_the_transformations_are_exactly_what_the_contract_requires():
    """Each constant is the contract divided by the native sign, not a free choice."""
    from vaft.machine_mapping import (
        BT_SIGN_VEST_TO_IMAS,
        IMAS_DISCHARGE_SIGNS,
        IP_SIGN_VEST_TO_IMAS,
    )

    for shot in SHOTS:
        ip_vest, bt_vest = _native_discharge(_load(shot))
        assert IP_SIGN_VEST_TO_IMAS.sign == IMAS_DISCHARGE_SIGNS.ip * int(np.sign(ip_vest))
        assert BT_SIGN_VEST_TO_IMAS.sign == IMAS_DISCHARGE_SIGNS.b0 * int(np.sign(bt_vest))


def test_the_packaged_shots_all_map_to_the_same_native_signs():
    """The transformations are uniform only if the native signs are."""
    for shot in SHOTS:
        ip_vest, bt_vest = _native_discharge(_load(shot))
        assert ip_vest > 0, shot
        assert bt_vest > 0, shot


def test_the_stored_discharge_does_not_yet_satisfy_the_contract_for_ip():
    """IP_SIGN is declared but not applied, and that gap is deliberate.

    Applying it flips Ip and, through Eq. 23, q and the orientation of psi, so
    the whole equilibrium must move together -- a data migration in the class of
    #275. Pinning the gap stops anyone reading the constant as a description of
    what is currently in the IDS.
    """
    from vaft.machine_mapping import IMAS_DISCHARGE_SIGNS

    for shot in SHOTS:
        ip_vest, bt_vest = _native_discharge(_load(shot))
        assert not IMAS_DISCHARGE_SIGNS.satisfied_by(ip_vest, bt_vest), (
            f"shot {shot}: stored Ip already complies -- the migration has landed "
            "and this test plus the declared-not-applied docs need updating"
        )
        # Specifically: B0 already complies, Ip does not.
        assert np.sign(bt_vest) == IMAS_DISCHARGE_SIGNS.b0
        assert np.sign(ip_vest) != IMAS_DISCHARGE_SIGNS.ip


def test_the_toroidal_side_complying_is_not_evidence_for_the_assumption():
    """B0 agrees because the assumption met a positive bt_vest. That is not proof."""
    from vaft.machine_mapping import BT_SIGN_VEST_TO_IMAS

    assert BT_SIGN_VEST_TO_IMAS.sign == +1
    for shot in SHOTS:
        _, bt_vest = _native_discharge(_load(shot))
        assert BT_SIGN_VEST_TO_IMAS.apply(bt_vest) == bt_vest
    assert BT_SIGN_VEST_TO_IMAS.confirmed is False


# --- equilibrium-level consequence -----------------------------------------


def test_the_contract_implies_negative_q_in_imas_cocos_eleven():
    """Antiparallel plus sigma_rhotp = +1 gives q < 0, by Sauter Eq. 23."""
    from vaft.data.cocos import VAFT_INTERNAL_COCOS, cocos_spec
    from vaft.machine_mapping import IMAS_DISCHARGE_SIGNS, expected_q_sign

    assert VAFT_INTERNAL_COCOS == 11
    assert cocos_spec(11).sigma_rhotp == +1
    assert IMAS_DISCHARGE_SIGNS.antiparallel
    assert expected_q_sign(11) == -1


def test_the_same_contract_implies_positive_q_in_the_native_cocos_three():
    """The native EFIT file is not in conflict: sigma_rhotp = -1 flips it back.

    COCOS 3/13 with q > 0 and COCOS 11 with q < 0 are the same equilibrium.
    """
    from vaft.data.cocos import cocos_spec
    from vaft.machine_mapping import expected_q_sign

    for index in (3, 13):
        assert cocos_spec(index).sigma_rhotp == -1
        assert expected_q_sign(index) == +1


def test_the_contract_admits_exactly_cocos_three_and_thirteen():
    """Re-derived from the contract through Eq. 23, not read off a comment."""
    from vaft.data.cocos import COCOS_INDICES, cocos_spec
    from vaft.machine_mapping import IMAS_DISCHARGE_SIGNS

    admissible = [
        index
        for index in COCOS_INDICES
        if cocos_spec(index).sigma_rpz == +1
        and cocos_spec(index).expected_sign(
            "q", sigma_ip=IMAS_DISCHARGE_SIGNS.ip, sigma_b0=IMAS_DISCHARGE_SIGNS.b0) > 0
        and cocos_spec(index).expected_sign(
            "dpsi", sigma_ip=IMAS_DISCHARGE_SIGNS.ip, sigma_b0=IMAS_DISCHARGE_SIGNS.b0) > 0
    ]
    assert admissible == [3, 13]


def test_converting_the_index_flips_q_but_never_ip_or_b0():
    """Why the discharge contract is orthogonal to the COCOS index.

    COCOS 3 and 11 share sigma_RphiZ = +1, so they already agree on which way
    +phi runs. Converting between them rescales psi and flips q; it cannot move
    Ip or B0. Anything wrong with those signs is a machine-convention or
    diagnostic-polarity problem, never a relabelling problem.
    """
    from omas.omas_physics import cocos_transform

    for source in (3, 13):
        factors = cocos_transform(source, 11)
        assert factors["Q"] == -1, source
        assert factors["IP"] == +1, source
        assert factors["BT"] == +1, source


def test_the_native_gfile_carries_positive_q_as_cocos_three_predicts():
    """The packaged reconstruction agrees with the contract once read correctly.

    g039915 is the EFIT reconstruction of packaged shot 39915. Its q > 0 is what
    COCOS 3/13 requires under this contract, and converting it to COCOS 11 gives
    the q < 0 the contract predicts there.
    """
    from omas.omas_physics import cocos_transform

    from vaft.data.eqdsk import read_geqdsk
    from vaft.data.resources import data_path, require_repository_sample
    from vaft.machine_mapping import expected_q_sign

    mapping = read_geqdsk(
        require_repository_sample(data_path("efit/g039915.00319"))
    ).mapping
    q = np.asarray(mapping["QPSI"], float)
    assert np.all(q > 0), "the native EFIT file carries positive q"
    assert expected_q_sign(3) == +1, "which is what COCOS 3 predicts here"

    q_imas = q * cocos_transform(3, 11)["Q"]
    assert np.all(q_imas < 0)
    assert expected_q_sign(11) == -1, "and COCOS 11 predicts the flipped sign"


# --- provenance: the open half ---------------------------------------------


def test_both_transformations_are_declared_with_their_provenance():
    """A sign without its source and evidence is the thing this module replaces."""
    from vaft.machine_mapping import (
        BT_SIGN_VEST_TO_IMAS,
        IP_SIGN_VEST_TO_IMAS,
        sign_transformations,
    )

    assert sign_transformations() == (IP_SIGN_VEST_TO_IMAS, BT_SIGN_VEST_TO_IMAS)
    for transformation in sign_transformations():
        assert transformation.sign in (-1, +1)
        assert transformation.vest_native_source, transformation.quantity
        assert transformation.evidence, transformation.quantity
        assert transformation.needed_to_confirm, transformation.quantity


def test_neither_polarity_claims_confirmation_it_does_not_have():
    """Guard against promotion without a spatially oriented reference.

    Internal agreement among the DAQ, magnetics, PF and EFIT chains cannot
    confirm these: those chains can share a global inversion. Flipping
    `confirmed` must come with evidence naming an oriented reference.
    """
    from vaft.machine_mapping import BT_SIGN_VEST_TO_IMAS, IP_SIGN_VEST_TO_IMAS

    assert IP_SIGN_VEST_TO_IMAS.confirmed is False
    assert BT_SIGN_VEST_TO_IMAS.confirmed is False
    # The Bt entry must record both the evidence found and why it falls short:
    # the alignment shots that give alpha ~ +1, and the fact that the two gains
    # share a datasheet and so show consistency rather than orientation.
    evidence = BT_SIGN_VEST_TO_IMAS.evidence
    assert "35376" in evidence, "the alignment shots that produced alpha ~ +1"
    assert "datasheet" in evidence, "why that agreement is not an absolute orientation"
    assert "mutual consistency" in evidence


def test_a_transformation_rejects_a_value_that_is_not_a_sign():
    from vaft.machine_mapping import SignTransformation

    with pytest.raises(ValueError, match="must be \\+1 or -1"):
        SignTransformation(
            quantity="ip", sign=0, confirmed=False,
            vest_native_source="x", evidence="y", needed_to_confirm="z",
        )
