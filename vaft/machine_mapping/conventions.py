"""VEST-native to IMAS sign transformations for the equilibrium orientation.

Every DAQ sign in VAFT is a **VEST-native** convention: it says which way a
voltage moves, not which way a current or a field points in space.  Two
transformations carry that DAQ world into the IMAS physical convention, and
between them they fix the COCOS index of every VEST equilibrium::

    ip_imas = IP_SIGN_VEST_TO_IMAS * ip_vest
    bt_imas = BT_SIGN_VEST_TO_IMAS * bt_vest

``ip_vest`` and ``bt_vest`` are what the mapping already produces:
``magnetics.ip`` from the Rogowski chain in :mod:`vaft.machine_mapping.magnetics`
and ``tf.b_field_tor_vacuum_r`` from the TF chain in
:mod:`vaft.machine_mapping.tf`.

Neither transformation is established yet.  Both carry ``confirmed = False``.
Their values come from the VEST team's **working assumption** that the plasma
current runs clockwise and the toroidal field counter-clockwise, both viewed
from above.  In a right-handed ``(R, phi, Z)`` frame with ``phi``
counter-clockwise from above that is ``sigma_Ip = -1`` and ``sigma_B0 = +1``,
and every packaged shot maps to ``ip_vest > 0`` and ``bt_vest > 0``, so the
transformations are ``-1`` and ``+1`` respectively.

**Declared, not applied.**  ``IP_SIGN_VEST_TO_IMAS`` is ``-1`` while the mapping
still writes ``ip_vest`` unchanged, so the two disagree by design until the
assumption is confirmed.  Applying it is not a one-leaf edit: flipping ``Ip``
flips, through Sauter Eq. 23, the sign of ``q`` and the orientation of ``psi``
-- the whole equilibrium has to move together, which is a data-migration change
of the same class as issue #275 and needs the same before/after numerical
verification.  :func:`equilibrium_orientation_is_resolved` stays ``False``
meanwhile.  ``BT_SIGN_VEST_TO_IMAS`` is ``+1``, so the toroidal side needs no
migration under this assumption; that is a property of the assumption, not
evidence for it.

Implied COCOS
-------------
With ``sigma_Ip = -1`` and ``sigma_B0 = +1``, and taking the reconstructions'
own ``q > 0`` and ``psi_edge - psi_axis > 0``, Sauter Eq. 23 admits exactly
**COCOS 3** (weber/radian) or **COCOS 13** (weber) for a ``phi``
counter-clockwise frame.  That is the index VEST equilibria should carry once
the transformations are applied, and it is a prediction the confirmation work
can test rather than an established fact.  It is mild corroboration that COCOS 3
is also the index Sauter section IX and OMAS's own tables give for EFIT, which
is the code that produced these reconstructions.

Why internal agreement cannot settle them
-----------------------------------------
The Rogowski, the poloidal probes, the PF coil currents, EFIT and the ODS are
mutually consistent -- the audit for issue #288 verified that the coil-subtracted
outboard-probe response tracks ``magnetics.ip`` with a positive slope on shot
39915.  That is a closed loop: every member of it could carry the same global
inversion and no comparison among them would notice.  The same holds on the
toroidal side, where the TF current gain and the IMPA Hall gain are two columns
of one DAQ wiring datasheet.  Establishing either transformation needs one
reference that is oriented in space, not one that is merely consistent with the
others.

What each one needs
-------------------
``IP_SIGN_VEST_TO_IMAS`` -- the net sign between ``ip_vest`` and a plasma
current flowing in the IMAS ``+phi`` direction.  The Rogowski calibration in
``vest.yaml`` is ``divide by 2.0e-5``, a magnitude with no recorded sense, and
the ``sign.multiply`` flip from ``+1`` to ``-1`` at shot 20259 carries no
comment and traces to a bulk WIP commit.  Any of these settles it: a Rogowski
calibration record stating the output polarity for a known current direction; a
shot in which a known current was driven through a known conductor in a known
direction with the Rogowski recording it; or the DAQ wiring datasheet for the
Rogowski channel, the counterpart of the one that exists for IMPA.

``BT_SIGN_VEST_TO_IMAS`` -- the net sign between ``bt_vest`` and a toroidal
field pointing along IMAS ``+phi``.  This one is close to resolved.  The IMPA
Hall array is an independently calibrated toroidal-field measurement
(``-2/15`` T/V, DAQ wiring datasheet) and ``vest.yaml`` records that the
2022-04-23 alignment shots 35376-35385 give ``alpha ~ +1``: the Hall-derived and
TF-derived fields agree in sign.  Two things stop that from closing it.  Both
gains come from the same datasheet, so the agreement shows the two columns are
mutually consistent rather than absolutely oriented; and the datasheet does not
record which toroidal direction the Hall sensors face.  So it needs the raw
alignment-shot data, which is in the VEST database but not packaged here, plus
one statement of the Hall sensors' facing (issue #298).  Failing that: the TF
winding sense with its supply polarity.  The packaged 3D coil geometry
(:mod:`vaft.machine_mapping.coil_geometry_3d`) covers only the non-axisymmetric
RMP sets, so it cannot supply the TF winding path.

Until both are confirmed the VEST COCOS index is not resolved, because it is
their *product* that fixes ``sigma_Ip * sigma_B0`` and hence the sign of ``q``.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = [
    "BT_SIGN_VEST_TO_IMAS",
    "IP_SIGN_VEST_TO_IMAS",
    "SignTransformation",
    "equilibrium_orientation_is_resolved",
    "sign_transformations",
]


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
