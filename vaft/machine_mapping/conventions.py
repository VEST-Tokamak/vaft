"""The VEST discharge sign contract, and the transformations that realise it.

The primary statement here is a **discharge-level contract**: during a normal
VEST plasma discharge, what signs must the mapped ``Ip`` and ``B0`` carry for the
IDS to mean, in IMAS terms, what the machine physically does?

IMAS defines positive plasma current and positive toroidal field as
**counter-clockwise viewed from above**.  Under the VEST team's working physical
assumption -- ``Ip`` clockwise, ``Bt`` counter-clockwise -- the answer is fixed
and unambiguous::

    Ip_imas < 0        (clockwise is the negative sense)
    B0_imas > 0        (counter-clockwise is the positive sense)

That is :data:`IMAS_DISCHARGE_SIGNS`.  It is a statement about conventions, not
about VEST hardware: given the physical assumption, no measurement is needed to
derive it and none could contradict it.

Everything else in this module follows from it.  Every DAQ sign in VAFT is a
**VEST-native** convention -- it says which way a voltage moves, not which way a
current or a field points in space -- so two transformations carry the DAQ world
into the contract::

    ip_imas = IP_SIGN_VEST_TO_IMAS * ip_vest
    bt_imas = BT_SIGN_VEST_TO_IMAS * bt_vest

Each is fixed by the contract divided by the sign the mapping natively produces.
Every packaged shot maps to ``ip_vest > 0`` and ``bt_vest > 0``, so the
transformations are ``-1`` and ``+1``.  ``ip_vest`` and ``bt_vest`` are what the
mapping already writes: ``magnetics.ip`` from the Rogowski chain in
:mod:`vaft.machine_mapping.magnetics` and ``tf.b_field_tor_vacuum_r`` from the TF
chain in :mod:`vaft.machine_mapping.tf`.

Two separate questions
----------------------
These must not be conflated, and the split is why the contract is stated first:

1. **The IMAS target signs.**  Settled.  A convention question, answered above.
2. **The diagnostic polarities.**  Open.  Does a positive VEST Rogowski signal
   really correspond to a clockwise plasma current?  Does a positive stored
   ``bt_vest`` independently establish a counter-clockwise ``Bt``?  These are
   questions about hardware wiring, and no amount of reasoning about IMAS
   answers them.

The ``confirmed = False`` on both transformations refers to (2) alone.  It does
not mean the contract is uncertain.

**Declared, not applied.**  ``IP_SIGN_VEST_TO_IMAS`` is ``-1`` while the mapping
still writes ``ip_vest`` unchanged, so the stored discharge does not yet satisfy
the contract for ``Ip``.  Applying it is not a one-leaf edit: flipping ``Ip``
flips, through Sauter Eq. 23, the sign of ``q`` and the orientation of ``psi``
-- the whole equilibrium has to move together, which is a data-migration change
of the same class as issue #275 and needs the same before/after numerical
verification.  ``BT_SIGN_VEST_TO_IMAS`` is ``+1``, so the toroidal side already
satisfies the contract; that is the assumption meeting a positive ``bt_vest``,
not evidence for the assumption.

The equilibrium-level consequence
---------------------------------
``Ip_imas < 0`` with ``B0_imas > 0`` is **antiparallel**, and that propagates to
``q`` through Sauter Eq. 23, ``sign(q) = sigma_rhotp * sigma_Ip * sigma_B0``:

* In **IMAS/COCOS 11** (``sigma_rhotp = +1``): ``q < 0``.
* In the **native EFIT gEQDSK**, admissible as **COCOS 3** (weber/radian) or
  **13** (weber) under this contract, ``sigma_rhotp = -1``: ``q > 0``.

Both are correct simultaneously; they are the same equilibrium in two
conventions.  The packaged g-files do carry ``q > 0``, which is consistent with
COCOS 3/13 rather than evidence against the contract.  It is mild corroboration
that COCOS 3 is the index Sauter section IX and OMAS give for EFIT, the code that
produced these reconstructions.

Crucially, ``cocos_transform(3, 11)`` carries ``Q = -1`` but ``IP = +1`` and
``BT = +1``: **converting the index never changes the sign of Ip or B0.**  Both
indices share ``sigma_RphiZ = +1``, so the physical directions are already the
same and only the flux/safety-factor conventions differ.  The discharge contract
is therefore orthogonal to the COCOS index, which is exactly why it is stated on
its own rather than folded into the index discussion.

Why internal agreement cannot settle the polarities
---------------------------------------------------
The Rogowski, the poloidal probes, the PF coil currents, EFIT and the ODS are
mutually consistent -- the audit for issue #288 verified that the coil-subtracted
outboard-probe response tracks ``magnetics.ip`` with a positive slope on shot
39915.  That is a closed loop: every member of it could carry the same global
inversion and no comparison among them would notice.  The same holds on the
toroidal side, where the TF current gain and the IMPA Hall gain are two columns
of one DAQ wiring datasheet.  Establishing either polarity needs one reference
that is oriented in space, not one that is merely consistent with the others.

What each polarity needs
------------------------
``IP_SIGN_VEST_TO_IMAS`` -- whether a positive ``ip_vest`` really is a clockwise
plasma current.  The Rogowski calibration in ``vest.yaml`` is ``divide by
2.0e-5``, a magnitude with no recorded sense, and the ``sign.multiply`` flip from
``+1`` to ``-1`` at shot 20259 carries no comment and traces to a bulk WIP
commit.  Any of these settles it: a Rogowski calibration record stating the
output polarity for a known current direction; a shot in which a known current
was driven through a known conductor in a known direction with the Rogowski
recording it; or the DAQ wiring datasheet for the Rogowski channel, the
counterpart of the one that exists for IMPA.  This is the more consequential of
the two, because it is the one carrying a pending migration.

``BT_SIGN_VEST_TO_IMAS`` -- whether a positive ``bt_vest`` really is a
counter-clockwise toroidal field.  The IMPA Hall array is an independently
calibrated toroidal-field measurement (``-2/15`` T/V, DAQ wiring datasheet) and
``vest.yaml`` records that the 2022-04-23 alignment shots 35376-35385 give
``alpha ~ +1``: the Hall-derived and TF-derived fields agree in sign.  Two things
stop that from closing it.  Both gains come from the same datasheet, so the
agreement shows the two columns are mutually consistent rather than absolutely
oriented; and the datasheet does not record which toroidal direction the Hall
sensors face.  So it needs the raw alignment-shot data, which is in the VEST
database but not packaged here, plus one statement of the Hall sensors' facing
(issue #298).  Failing that: the TF winding sense with its supply polarity.  The
packaged 3D coil geometry (:mod:`vaft.machine_mapping.coil_geometry_3d`) covers
only the non-axisymmetric RMP sets, so it cannot supply the TF winding path.

Until both polarities are confirmed the VEST COCOS index is not resolved, because
it is their *product* that fixes ``sigma_Ip * sigma_B0`` and hence the sign of
``q``.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = [
    "BT_SIGN_VEST_TO_IMAS",
    "DischargeSignContract",
    "IMAS_DISCHARGE_SIGNS",
    "IP_SIGN_VEST_TO_IMAS",
    "SignTransformation",
    "equilibrium_orientation_is_resolved",
    "expected_q_sign",
    "sign_transformations",
    "to_imas_discharge_signs",
]


@dataclass(frozen=True)
class DischargeSignContract:
    """The signs a normal VEST discharge must carry in the IMAS convention.

    A convention statement, not a measurement: IMAS calls counter-clockwise
    positive for both quantities, so the working physical assumption fixes these
    outright.  Confidence in the *diagnostic polarities* is tracked separately,
    on :class:`SignTransformation`.
    """

    ip: int
    b0: int

    def __post_init__(self) -> None:
        for name in ("ip", "b0"):
            if getattr(self, name) not in (-1, +1):
                raise ValueError(f"{name} must be +1 or -1, not {getattr(self, name)!r}")

    @property
    def antiparallel(self) -> bool:
        """Whether the plasma current opposes the toroidal field."""
        return self.ip * self.b0 < 0

    def satisfied_by(self, ip, b0) -> bool:
        """Whether a mapped ``(Ip, B0)`` pair, already in IMAS terms, complies."""
        import numpy as np

        return bool(np.sign(ip) == self.ip and np.sign(b0) == self.b0)


IMAS_DISCHARGE_SIGNS = DischargeSignContract(ip=-1, b0=+1)
"""Ip clockwise and Bt counter-clockwise, expressed in IMAS signs."""


def expected_q_sign(cocos_index: int, contract: DischargeSignContract | None = None) -> int:
    """The sign ``q`` must carry in ``cocos_index`` under the discharge contract.

    Sauter Eq. 23 gives ``sign(q) = sigma_rhotp * sigma_Ip * sigma_B0``, so the
    antiparallel contract yields ``q < 0`` in IMAS/COCOS 11 and ``q > 0`` in the
    native EFIT COCOS 3/13.  Both describe the same equilibrium.
    """
    from vaft.data.cocos import cocos_spec

    contract = IMAS_DISCHARGE_SIGNS if contract is None else contract
    sign = cocos_spec(cocos_index).expected_sign(
        "q", sigma_ip=contract.ip, sigma_b0=contract.b0,
    )
    return int(round(sign))


def to_imas_discharge_signs(ip_vest, bt_vest):
    """Carry a VEST-native ``(ip, bt)`` discharge pair into the IMAS convention."""
    return IP_SIGN_VEST_TO_IMAS.apply(ip_vest), BT_SIGN_VEST_TO_IMAS.apply(bt_vest)


@dataclass(frozen=True)
class SignTransformation:
    """One VEST-native to IMAS sign transformation and its standing."""

    quantity: str
    sign: int
    confirmed: bool
    vest_native_source: str
    evidence: str
    needed_to_confirm: str

    def __post_init__(self) -> None:
        if self.sign not in (-1, +1):
            raise ValueError(f"a sign transformation must be +1 or -1, not {self.sign!r}")

    def apply(self, value):
        """Map a VEST-native value into the IMAS convention."""
        return self.sign * value


IP_SIGN_VEST_TO_IMAS = SignTransformation(
    quantity="ip",
    sign=-1,
    confirmed=False,
    vest_native_source=(
        "magnetics.ip, from raw field 109 (Rogowski RC03) divided by 2.0e-5, "
        "FL10-referenced, then multiplied by vest.yaml plasma_current.processing.sign"
    ),
    evidence=(
        "Working assumption only: the VEST team states the plasma current runs "
        "clockwise viewed from above, so sigma_Ip = -1, and every packaged shot "
        "maps to ip_vest > 0; hence -1. No independent measurement supports it. The "
        "Rogowski calibration records a magnitude only, the sign.multiply flip at "
        "shot 20259 is uncommented, and the chain's agreement with the poloidal "
        "probes and the PF coils is a closed loop that does not orient it in space."
    ),
    needed_to_confirm=(
        "A Rogowski calibration record giving the output polarity for a known "
        "current direction, a known-current reference shot, or the DAQ wiring "
        "datasheet for the Rogowski channel."
    ),
)

BT_SIGN_VEST_TO_IMAS = SignTransformation(
    quantity="b_field_tor_vacuum_r",
    sign=+1,
    confirmed=False,
    vest_native_source=(
        "tf.b_field_tor_vacuum_r, from raw field 1 times -3.0e4 A/V, then "
        "mu0 * 24 * I_TF / (2 * pi) per vaft.machine_mapping.tf"
    ),
    evidence=(
        "Working assumption plus partial corroboration. The VEST team states the "
        "toroidal field runs counter-clockwise viewed from above, so sigma_B0 = +1, "
        "and every packaged shot maps to bt_vest > 0; hence +1. Separately, the IMPA "
        "Hall array is an independently calibrated toroidal-field measurement "
        "(-2/15 T/V) and vest.yaml records alpha ~ +1 on the 2022-04-23 alignment "
        "shots 35376-35385, so the Hall-derived and TF-derived fields agree in sign. "
        "Both gains are columns of one DAQ wiring datasheet, so that shows mutual "
        "consistency, not absolute orientation, and it cannot confirm the assumption "
        "on its own. That this sign is the identity is a property of the assumption "
        "meeting a positive bt_vest, not evidence for either."
    ),
    needed_to_confirm=(
        "The raw alignment-shot data (35376-35385, not packaged here) together with "
        "the toroidal direction the IMPA Hall sensors face (issue #298); or the TF "
        "winding sense with its supply polarity."
    ),
)


def sign_transformations() -> tuple[SignTransformation, ...]:
    """Both transformations, in the order they appear in the COCOS product."""
    return (IP_SIGN_VEST_TO_IMAS, BT_SIGN_VEST_TO_IMAS)


def equilibrium_orientation_is_resolved() -> bool:
    """True only when both transformations rest on an oriented reference.

    ``sigma_Ip * sigma_B0`` is the product of the two, so one unconfirmed
    transformation leaves the VEST COCOS index unresolved even if the other is
    certain.
    """
    return all(item.confirmed for item in sign_transformations())
