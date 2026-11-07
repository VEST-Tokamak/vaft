"""Licence-verified public equilibria for multi-machine comparison (#2285).

Each external equilibrium here is one pinned artefact whose *data* licence was
read at the file (or Zarr group) level, not inferred from a software licence or
from the file being reachable.  The artefact is fetched once with a SHA-256
check (:mod:`._fetch`), read by a source-specific adapter and returned as an
OMAS ODS plus a :class:`PublicEquilibriumSource` record that carries the
producer, licence, attribution obligations, revision, selection rule and the
convention evidence.

Sources (audit of 2026-11-07):

========  ======================================  =================  ==============================
device    artefact                                 data licence        contract
========  ======================================  =================  ==============================
VEST      packaged sample 39915                    VAFT reference      full EFIT
TCV       TCV-X21 ``65402_t1.eqdsk``               CC BY 4.0           full GEQDSK (experimental)
DIII-D    NeurIPS challenge ``diii_d_train`` row   CC BY 4.0           flux map + LCFS + scalars
MAST      FAIR-MAST level-2 30420 ``equilibrium``  CC BY-SA 4.0        full EFIT++ slice
STEP      OpenSTEP EB-CC flat-top JETTO g-file     CC BY 4.0           full GEQDSK (design)
SPARC     SPARCPublic ``sparc_1593`` g-file        MIT (CFS)           full GEQDSK (design)
ITER      ITER IMAS example scenario 53298 (DD3)   CC BY 4.0           full IMAS slice (modelled)
========  ======================================  =================  ==============================

JET (FUSE ``JET_99948``) and MAST-U (FUSE ``MASTU_Hmode``) are listed in
:data:`EXCLUDED_EQUILIBRIUM_SOURCES` and cannot be loaded: the redistribution
rights of their underlying data were not established.  MAST is not MAST-U and
does not stand in for it.  ITER is read from the ITER Organization's own CC BY
4.0 Zenodo release, not from the FUSE ``ITER_equilibrium_ods`` sample.

What is *not* done: no profile, pressure, ``F`` or ``q`` is constructed for a
source that does not ship it (DIII-D carries a flux map, an LCFS and EFIT
scalars only), no vacuum field is invented, and no physical equivalence
between machines is implied.  Values keep their source sign and unit except
for the documented ``2*pi`` that puts a per-radian poloidal flux into the
ODS's full-weber psi (the same rule :func:`vaft.data.eqdsk.to_omas` applies).
A few ODS fields are *derived* from shipped data rather than copied, and each
is named in :attr:`PublicEquilibrium.notes` and in
``equilibrium.ids_properties.comment``: the DIII-D ``psi_axis`` and
``psi_boundary`` (the flux map interpolated at the EFIT axis and averaged
along the shipped LCFS) and plasma current (the magnetics Ip interpolated onto
the EFIT time), and the MAST ``profiles_1d.psi`` (built from the shipped
``psi_norm`` grid).  A COCOS index is written on the ODS only when one index
is pinned by the data or by a producer statement the data do not contradict.

Failures are kept apart by stage: :class:`~._fetch.FetchError` (network),
:class:`~._fetch.ChecksumError` (bytes are not the pinned artefact),
:class:`EquilibriumLicenceError` (licence not established or changed
upstream), :class:`EquilibriumParseError` (artefact layout not as audited) and
:class:`EquilibriumCompatibilityError` (no usable slice).
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
import json
import os
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np

from ._fetch import cache_dir, fetch, sha256_of

__all__ = [
    "EXCLUDED_EQUILIBRIUM_SOURCES",
    "EquilibriumCompatibilityError",
    "EquilibriumLicenceError",
    "EquilibriumParseError",
    "PUBLIC_EQUILIBRIUM_DEVICES",
    "PUBLIC_EQUILIBRIUM_SOURCES",
    "PublicEquilibrium",
    "PublicEquilibriumSource",
    "load_public_equilibrium",
    "public_equilibrium_provenance",
]

TWO_PI = 2.0 * np.pi

#: Fraction of the peak |Ip| that defines the flat-top window of the
#: representative-slice rule (:func:`_flat_top_midpoint`).
FLAT_TOP_FRACTION = 0.9


class EquilibriumLicenceError(RuntimeError):
    """The data licence of the requested equilibrium is not established, or
    the licence an artefact declares differs from the audited one."""


class EquilibriumParseError(ValueError):
    """The artefact arrived intact but its layout is not the audited one."""


class EquilibriumCompatibilityError(ValueError):
    """The artefact holds no slice the common geometry workflow can use."""


@dataclass(frozen=True)
class PublicEquilibriumSource:
    """Audited provenance of one public equilibrium artefact.

    Attributes
    ----------
    device : str
        Device name as used in VAFT tables, e.g. ``"MAST"`` [str].
    producer : str
        Who produced and released the data [str].
    dataset : str
        Published dataset or repository name [str].
    revision : str
        Pinned upstream revision (git commit, HF dataset commit, ingestion
        commit) [str].
    url : str
        Revision-pinned URL of the artefact; for a Zarr store, of the store
        root [str].
    artefact : str
        The selected file or group inside the dataset [str].
    sha256 : str or None
        SHA-256 of the artefact (of ``zarr.json`` for a Zarr store); ``None``
        only for the packaged VEST sample, hashed at load time [str].
    size_bytes : int or None
        Size of the pinned artefact [byte].
    licence : str
        Licence name of the data [str].
    licence_url : str
        Licence text URL [str].
    licence_basis : str
        Where the licence statement for *this* artefact was read [str].
    attribution : str
        Attribution / acknowledgement text the licensor asks for [str].
    share_alike : bool
        Whether adaptations must carry the same licence [bool].
    references : tuple of str
        Citations, DOI first where one exists [str].
    data_kind : str
        ``"experimental reconstruction"``, ``"design"``, ... [str].
    contract : str
        ``"full"`` (flux map, profiles, LCFS) or ``"flux-map+lcfs"`` [str].
    selection : str
        Selected shot / scenario and the slice rule [str].
    convention : str
        Units, signs and COCOS evidence as audited [str].
    stated_cocos : int or None
        COCOS index the producer states for the artefact, if any [-].
    reader : str
        Adapter key [str].
    """

    device: str
    producer: str
    dataset: str
    revision: str
    url: str
    artefact: str
    sha256: str | None
    size_bytes: int | None
    licence: str
    licence_url: str
    licence_basis: str
    attribution: str
    share_alike: bool
    references: tuple[str, ...]
    data_kind: str
    contract: str
    selection: str
    convention: str
    stated_cocos: int | None
    reader: str


@dataclass(frozen=True)
class PublicEquilibrium:
    """One loaded public equilibrium.

    Attributes
    ----------
    ods : omas.ODS
        Equilibrium ODS (psi in Wb, source signs) [ODS].
    source : PublicEquilibriumSource
        Audited provenance record [record].
    time_index : int
        Index of the representative slice in ``ods`` [-].
    time_s : float or None
        Time of that slice [s]; ``None`` when the artefact carries none.
    sha256 : str
        SHA-256 of the artefact actually read [str].
    cocos : int or None
        COCOS index written on the ODS, ``None`` when not pinned [-].
    cocos_candidates : tuple of int
        Indices the data are consistent with [-].
    missing : tuple of str
        Equilibrium quantities the source does not provide [str].
    notes : tuple of str
        Conversion and selection notes, in order [str].
    """

    ods: Any
    source: PublicEquilibriumSource
    time_index: int
    time_s: float | None
    sha256: str
    cocos: int | None
    cocos_candidates: tuple[int, ...]
    missing: tuple[str, ...]
    notes: tuple[str, ...] = field(default_factory=tuple)


_CC_BY = "https://creativecommons.org/licenses/by/4.0/"
_CC_BY_SA = "https://creativecommons.org/licenses/by-sa/4.0/"

_TCV_COMMIT = "9a0a5eb13e2f0ff34494c31a66a3d32b042ddd7d"
_SPARC_COMMIT = "5b913b8216d05346b22e2e1e87c2ff100761613f"
_OPENSTEP_COMMIT = "6f66ae09c60b361d600b178416ef3f22b8d0aa6a"
_HF_REVISION = "1e280905b85f2a6fdde7e06fca8cf3a1edf447cb"
_MAST_SHOT = 30420
_MAST_STORE = f"https://s3.echo.stfc.ac.uk/mast/level2/shots/{_MAST_SHOT}.zarr/"
_MAST_INGEST_COMMIT = "ed6ea4e2a81f2317ae474f7c5421071994220948"
_MAST_LICENCE_NAME = "Creative Commons 4.0 BY-SA"
_ITER_RECORD = "17062700"
_ITER_DD_VERSION = "3.39.0"

_SOURCES = (
    PublicEquilibriumSource(
        device="VEST",
        producer="VEST team, Seoul National University",
        dataset="VAFT packaged reference sample",
        revision="packaged with this VAFT version",
        url="vaft.data.sample(39915, representation='imas')",
        artefact="39915 EFIT reconstruction",
        sha256=None,
        size_bytes=None,
        licence="VAFT packaged reference data",
        licence_url="",
        licence_basis="Native VAFT reference sample; no third-party licence involved.",
        attribution="VEST discharge 39915, VAFT reference sample.",
        share_alike=False,
        references=(),
        data_kind="experimental reconstruction (EFIT)",
        contract="full",
        selection="shot 39915; flat-top midpoint slice (|Ip| >= 0.9 max)",
        convention="VAFT EFIT product, COCOS labelled by VAFT.",
        stated_cocos=None,
        reader="vaft_sample",
    ),
    PublicEquilibriumSource(
        device="TCV",
        producer="Swiss Plasma Center, EPFL (TCV-X21 team)",
        dataset="TCV-X21 validation case",
        revision=_TCV_COMMIT,
        url=(
            f"https://raw.githubusercontent.com/SPCData/TCV-X21/{_TCV_COMMIT}/"
            "1.experimental_data/reference_scenario/65402_t1.eqdsk"
        ),
        artefact="1.experimental_data/reference_scenario/65402_t1.eqdsk",
        sha256="c6e2bfdff4d33937ef2e617bd2e883161eb12d8ee1d059409c5dbafd5560035e",
        size_bytes=3942925,
        licence="CC BY 4.0",
        licence_url=_CC_BY,
        licence_basis=(
            "TCV-X21 README 'License and attribution notice': the TCV-X21 datasets are "
            "licenced under CC BY 4.0 (repository LICENSE); the MIT licence covers only "
            "the tcvx21 analysis code."
        ),
        attribution=(
            "Cite Oliveira, Body et al., Nucl. Fusion 62 (2022) 096001 and acknowledge the "
            "TCV-X21 dataset(s) publicly released under the CC-BY 4.0 license."
        ),
        share_alike=False,
        references=("doi:10.1088/1741-4326/ac4cde",),
        data_kind="experimental reference equilibrium",
        contract="full",
        selection="TCV 65402 reference scenario, the single g-file slice",
        convention=(
            "psi stored in Wb (the 11-18 family; Ip < 0, psi decreasing outward); "
            "BCENTR/FPOL inconsistency noted in #325."
        ),
        stated_cocos=None,
        reader="geqdsk",
    ),
    PublicEquilibriumSource(
        device="DIII-D",
        producer="Sophelio and General Atomics (DIII-D National Fusion Facility)",
        dataset="Fusion Equilibrium Challenge (NeurIPS 2026), config diii_d_train",
        revision=_HF_REVISION,
        url=(
            "https://huggingface.co/datasets/Sophelio/fusion-equilibrium-challenge/resolve/"
            f"{_HF_REVISION}/data/diii_d_train/d3d_shot_0003ff34e7.parquet"
        ),
        artefact="data/diii_d_train/d3d_shot_0003ff34e7.parquet",
        sha256="6892d65616a7c033dde3888befb452a920aa7043856e40aa467cc836746181cc",
        size_bytes=4191877,
        licence="CC BY 4.0",
        licence_url=_CC_BY,
        licence_basis=(
            "Dataset card (README.md front matter 'license: cc-by-4.0' and 'License & "
            "acknowledgement' section) of the pinned dataset revision; covers every "
            "released config including diii_d_train."
        ),
        attribution=(
            "Michoski et al., The Fusion Equilibrium Challenge (2026), Sophelio and General "
            "Atomics. Work supported by the U.S. DOE, Office of Science, Office of Fusion "
            "Energy Sciences, using the DIII-D National Fusion Facility, a DOE Office of "
            "Science user facility, under Award No. DE-FC02-04ER54698."
        ),
        share_alike=False,
        references=("https://huggingface.co/datasets/Sophelio/fusion-equilibrium-challenge",),
        data_kind="experimental reconstruction (EFIT)",
        contract="flux-map+lcfs",
        selection=(
            "training record d3d_shot_0003ff34e7 (anonymised shot id; training split only, "
            "never a test split); flat-top midpoint EFIT slice (|Ip| >= 0.9 max); every "
            "shipped slice is diverted"
        ),
        convention=(
            "efit_psirz in Wb/rad with the magnetic axis at the minimum; times in ms; Ip in "
            "kA. No toroidal field, F, p or q profile is shipped."
        ),
        stated_cocos=None,
        reader="neurips_parquet",
    ),
    PublicEquilibriumSource(
        device="MAST",
        producer="UKAEA (FAIR-MAST)",
        dataset=f"FAIR-MAST level 2, shot {_MAST_SHOT}",
        revision=f"fair-mast-ingestion {_MAST_INGEST_COMMIT}, ingested 2026-09-24",
        url=_MAST_STORE,
        artefact="equilibrium group (EFIT++ reconstruction)",
        sha256="b1f29c28f0c25f5768a21889f2ec45ebc4243e626aa4743845dc58eb33ccd637",
        size_bytes=783496,
        licence="CC BY-SA 4.0",
        licence_url=_CC_BY_SA,
        licence_basis=(
            "The equilibrium group's own Zarr attributes declare license_name "
            f"'{_MAST_LICENCE_NAME}' and license_url {_CC_BY_SA}. zarr.json is hash-pinned, "
            "so an upstream licence change first appears as a ChecksumError; the "
            "attributes are re-checked after the hash."
        ),
        attribution=(
            "MAST data from the FAIR-MAST service (UKAEA), CC BY-SA 4.0; cite Jackson et "
            "al., SoftwareX 27 (2024) 101869."
        ),
        share_alike=True,
        references=("doi:10.1016/j.softx.2024.101869", "doi:10.1109/TPS.2025.3583419"),
        data_kind="experimental reconstruction (EFIT++)",
        contract="full",
        selection=(
            f"MAST shot {_MAST_SHOT}; flat-top midpoint EFIT++ slice (|Ip| >= 0.9 max); "
            "only reconstructed fields, no *_measured diagnostics"
        ),
        convention=(
            "psi in Wb/rad with the magnetic axis at the maximum (EFIT++ sign); Ip > 0, "
            "B0 < 0 at the selected slice."
        ),
        stated_cocos=None,
        reader="fair_mast_zarr",
    ),
    PublicEquilibriumSource(
        device="STEP",
        producer="UKIFS / UKAEA (OpenSTEP)",
        dataset="OpenSTEP SPP-001, EB-CC flat-top (JINTRAC/JETTO)",
        revision=_OPENSTEP_COMMIT,
        url=(
            f"https://raw.githubusercontent.com/ukaea/OpenSTEP/{_OPENSTEP_COMMIT}/"
            "eb-cc/ftop/_src/jetto.eqdsk_out"
        ),
        artefact="eb-cc/ftop/_src/jetto.eqdsk_out",
        sha256="36bc3166586cb13fa925edf20aaf9c42e2f816d5d0124d393df09cfe9cb14509",
        size_bytes=384341,
        licence="CC BY 4.0",
        licence_url=_CC_BY,
        licence_basis=(
            "OpenSTEP is a data repository; its LICENSE.txt and README 'License' section "
            "license the repository content under CC BY 4.0."
        ),
        attribution=(
            "Cite OpenSTEP (doi:10.14468/07jt-s540) and Tholerus et al., Nucl. Fusion 64 "
            "(2024) 106030."
        ),
        share_alike=False,
        references=("doi:10.14468/07jt-s540", "doi:10.1088/1741-4326/ad6ea2"),
        data_kind="design (JETTO fixed-boundary flat-top)",
        contract="full",
        selection=(
            "SPP-001 EB-CC flat-top operating point, the single JETTO g-file slice; the "
            "free-boundary Fiesta case ships the byte-identical file as its target"
        ),
        convention=(
            "README: 'Eqdsk output is COCOS 1'. The IMAS netCDF beside it is stamped DD "
            "4.0.0 but its psi increases outward with Ip > 0 (COCOS 11, not DD 4's COCOS "
            "17), so the g-file is the artefact used."
        ),
        stated_cocos=1,
        reader="geqdsk",
    ),
    PublicEquilibriumSource(
        device="SPARC",
        producer="Commonwealth Fusion Systems",
        dataset="SPARCPublic EQlibrary",
        revision=_SPARC_COMMIT,
        url=(
            f"https://raw.githubusercontent.com/cfs-energy/SPARCPublic/{_SPARC_COMMIT}/"
            "EQlibrary/sparc_1593.EQDSK_PsiOver2pi_negIp_negBt_negFpol"
        ),
        artefact="EQlibrary/sparc_1593.EQDSK_PsiOver2pi_negIp_negBt_negFpol",
        sha256="b6459951b3ed75f7c3f78725322a7e0643f61592991f65d17efe78fc90d74898",
        size_bytes=137217,
        licence="MIT (Copyright CFS)",
        licence_url=f"https://github.com/cfs-energy/SPARCPublic/blob/{_SPARC_COMMIT}/LICENSE",
        licence_basis=(
            "The repository-wide MIT licence (README 'License', Copyright 2020 CFS) has no "
            "carve-out for EQlibrary and is treated as the licence of this discharge file "
            "(VAFT maintainer decision, #2285). No separate data licence exists."
        ),
        attribution=(
            "The information, data, or work presented herein builds on the SPARC primary "
            "reference discharge and X-point target discharge data provided by "
            "Commonwealth Fusion Systems."
        ),
        share_alike=False,
        references=("https://github.com/cfs-energy/SPARCPublic",),
        data_kind="design (GEQDSK library)",
        contract="full",
        selection="EQlibrary case 1593, the single g-file slice",
        convention="File name states psi/2pi, negative Ip, Bt and F (SPARC helicity).",
        stated_cocos=None,
        reader="geqdsk",
    ),
    PublicEquilibriumSource(
        device="ITER",
        producer="ITER Organization",
        dataset="Example of ITER simulation results in IMAS (Zenodo, version 4.0.0)",
        revision=f"zenodo record {_ITER_RECORD} (doi:10.5281/zenodo.{_ITER_RECORD}), 2025-09-05",
        url=(
            f"https://zenodo.org/records/{_ITER_RECORD}/files/"
            "iter_scenario_53298_seq1_DD3.nc?download=1"
        ),
        artefact="iter_scenario_53298_seq1_DD3.nc",
        sha256="5792cce5815f53195eea1671647491c9502e49664a122ad3c14582e62346dd10",
        size_bytes=14646060,
        licence="CC BY 4.0",
        licence_url=_CC_BY,
        licence_basis=(
            "The Zenodo record's own licence field is CC BY 4.0 (creator: ITER "
            "Organization); the title's 'for code testing purpose' describes intent and "
            "places no restriction on the licence."
        ),
        attribution=(
            "ITER Organization, Example of ITER simulation results in IMAS (for code "
            f"testing purpose), Zenodo (2025), doi:10.5281/zenodo.{_ITER_RECORD}, CC BY 4.0."
        ),
        share_alike=False,
        references=(f"doi:10.5281/zenodo.{_ITER_RECORD}",),
        data_kind="modelled scenario (JINTRAC core-edge, 15 MA / 5.3 T H-mode)",
        contract="full",
        selection=(
            "ITER scenario 53298 seq. 1 (JINTRAC), the single equilibrium slice; only the "
            "equilibrium IDS is kept"
        ),
        convention=(
            f"IMAS DD {_ITER_DD_VERSION} netCDF (DD3 is COCOS 11); psi in Wb, Ip < 0, B0 < 0. "
            "The DD4 conversion in the same record is not used."
        ),
        stated_cocos=11,
        reader="imas_netcdf",
    ),
)

#: Audited sources keyed by device, in display order.
PUBLIC_EQUILIBRIUM_SOURCES: Mapping[str, PublicEquilibriumSource] = MappingProxyType(
    {source.device: source for source in _SOURCES}
)
#: Device names :func:`load_public_equilibrium` accepts, in display order.
PUBLIC_EQUILIBRIUM_DEVICES: tuple[str, ...] = tuple(PUBLIC_EQUILIBRIUM_SOURCES)

#: Equilibria deliberately not loadable, with the reason.
EXCLUDED_EQUILIBRIUM_SOURCES: Mapping[str, str] = MappingProxyType(
    {
        "JET": (
            "FUSE.jl sample JET_99948_STATE0_eq.json only; redistribution rights of the "
            "original JET data are not established."
        ),
        "MAST-U": (
            "FUSE.jl sample MASTU_Hmode.json only; original source and rights unresolved "
            "(and its slice fails Ampere's law). MAST is a different device and is not a "
            "substitute."
        ),
    }
)

# FAIR-MAST level-2 chunk files read by the MAST adapter, with their SHA-256.
_MAST_ARRAYS = (
    "time", "ip", "psi", "major_radius", "z", "psi_axis", "psi_boundary",
    "magnetic_axis_r", "magnetic_axis_z", "lcfs_r", "lcfs_z", "psi_norm",
    "f", "pressure", "q95", "x_point_r", "x_point_z", "bvac_val", "bvac_r",
)
_MAST_CHUNKS: Mapping[str, str] = MappingProxyType(
    {
        "equilibrium/bvac_r/c/0": "9db11358f749c6378f3562ee81863c380c17c785d62065a3dfa1ffc7af46a38d",
        "equilibrium/bvac_val/c/0": "b3a8b2a84820754cb23b405829f6cd125097bfbe34a3a36b5ed5c8d4ddfbea7e",
        "equilibrium/f/c/0/0": "a6b3c627eb407f38b8970ae7a68199e3769f0355b3b14ad8f010c0d8675bef04",
        "equilibrium/ip/c/0": "b2f464ff735f3dd25de2b923e0ea20b88663b9166b51fc0ceb8b6bd5f11b6770",
        "equilibrium/lcfs_r/c/0/0": "07cb31c414d8f696d04f129d44845b20d2668f97ca8b0100323fdc45eeeb998b",
        "equilibrium/lcfs_z/c/0/0": "4a0fa0afc55094eb088db72c314e123b5372d8170dd260284a40dce707d8a5de",
        "equilibrium/magnetic_axis_r/c/0": "190843591e8b7a53ab5f8467f6eb01dc2cbe9793572289fd6015935e5c0be687",
        "equilibrium/magnetic_axis_z/c/0": "1784c6dbe503bb825a9f0aa4115957de3e7997a9c481acfd95852c263896239c",
        "equilibrium/major_radius/c/0": "839c9730367e4e818d46c4311f35de73785a94a71b49ee0c7c8533858180a134",
        "equilibrium/pressure/c/0/0": "bae50ea49859aa86c315dd5c2075c9f944b280e61d595ed12bb5122f6a261f68",
        "equilibrium/psi/c/0/0/0": "767bf61d757d672b034b98b0a392fa0c5bdd34e4584dfc25fcecbb2f6c2db017",
        "equilibrium/psi/c/0/0/1": "05bec20dae5b9b526814ae87ebfee89cf9bb382ef4ab92940dd48ab0c93daffa",
        "equilibrium/psi/c/0/1/0": "5173b5200216411641b91544f701cf13b0abc6d633a1bbb8689fc2e06fbccb49",
        "equilibrium/psi/c/0/1/1": "b5f25c7b4976cff6c7d0fe333e68009128eed015f5a53f1bb4e436683d65f6f2",
        "equilibrium/psi/c/1/0/0": "b981b0e1882d021567ca81b88067f2dffb2f7a502339b74bb7661c76df67366f",
        "equilibrium/psi/c/1/0/1": "bf807faddf0e7cdc78b5f02f1bea8f704d8775b3f7586eed5228c60cfaf723c3",
        "equilibrium/psi/c/1/1/0": "0b90fe6b61ccfab9d62cf4d5a1fdb5563a88014c201528cbefcf1d4cefbf87d9",
        "equilibrium/psi/c/1/1/1": "8b97be63d1b83896779b1da9bd94ae8d0bb0f2aa44b66285fc8acf7bdb038bfc",
        "equilibrium/psi_axis/c/0": "b678b0c71a198880d262430c6dc31cd2158ad97d224079a106fffd31a6410349",
        "equilibrium/psi_boundary/c/0": "580e2562eff4565c20ee960bd5954fbb4bce436c6c504d6935ddac7472fa66d5",
        "equilibrium/psi_norm/c/0": "18cc60bbd551fcad6b553188127cf43d56a34fc9a7ba54a1c2ac98ea80a7d008",
        "equilibrium/q95/c/0": "38f129e57c87513824b0245e20d77a94b4009a13904bf235919442dc4a1bc853",
        "equilibrium/time/c/0": "50a0ba7353f524b0d7eb9b6df884e980dac9dcdd985ddda2574dbc19051ad3e3",
        "equilibrium/x_point_r/c/0/0": "622c1da5bf6fbcacbea41fae7d70543e9779671c320b1ad6e62d206651a977cc",
        "equilibrium/x_point_z/c/0/0": "8f2ebccc1171449164b0b8998aa1981d3ff0b329af85c917836a478002964caf",
        "equilibrium/z/c/0": "ce8263327297e0458437aed08c001a119c2889660c86708f294ee5c00e3dda95",
    }
)


# --------------------------------------------------------------------- public API


def load_public_equilibrium(
    device: str,
    *,
    cache: str | os.PathLike[str] | None = None,
    timeout: float = 120.0,
) -> PublicEquilibrium:
    """Fetch, verify and adapt one audited public equilibrium.

    Parameters
    ----------
    device : str
        One of :data:`PUBLIC_EQUILIBRIUM_DEVICES` [str].
    cache : path-like or None, optional
        Cache directory, default ``None`` for :func:`~._fetch.cache_dir` [path].
    timeout : float, optional
        Network timeout per file, default 120 [s].

    Returns
    -------
    PublicEquilibrium
        The ODS, its provenance record and the selected slice.

    Raises
    ------
    EquilibriumLicenceError
        ``device`` is excluded or unknown, or the artefact declares a licence
        other than the audited one.
    FetchError, ChecksumError
        The artefact could not be fetched, or is not the pinned one. A
        ``ChecksumError`` means the upstream bytes changed and the source needs
        a fresh audit, licence included.
    EquilibriumParseError
        The artefact's layout is not the audited one.
    EquilibriumCompatibilityError
        No representative slice could be selected.
    ImportError
        An optional reader dependency (``pyarrow``) is missing.
    """
    if device in EXCLUDED_EQUILIBRIUM_SOURCES:
        raise EquilibriumLicenceError(
            f"{device} is excluded: {EXCLUDED_EQUILIBRIUM_SOURCES[device]}"
        )
    if device not in PUBLIC_EQUILIBRIUM_SOURCES:
        raise EquilibriumLicenceError(
            f"No licence-verified equilibrium for {device!r}; audited devices are "
            f"{list(PUBLIC_EQUILIBRIUM_DEVICES)}."
        )
    source = PUBLIC_EQUILIBRIUM_SOURCES[device]
    reader = _READERS[source.reader]
    return reader(source, cache=cache, timeout=timeout)


def public_equilibrium_provenance(loaded: Iterable[PublicEquilibrium]):
    """Source, licence and selection table of the equilibria actually loaded.

    Parameters
    ----------
    loaded : iterable of PublicEquilibrium
        Results of :func:`load_public_equilibrium` [record].

    Returns
    -------
    pandas.DataFrame
        One row per loaded equilibrium: device, producer, data kind, contract,
        licence, licence basis, attribution, share-alike flag, revision,
        artefact, SHA-256 read, selected time, COCOS label and candidates, and
        the quantities the source does not provide.
    """
    import pandas as pd

    rows = []
    for item in loaded:
        source = item.source
        rows.append(
            {
                "device": source.device,
                "producer": source.producer,
                "data_kind": source.data_kind,
                "contract": source.contract,
                "licence": source.licence,
                "licence_url": source.licence_url,
                "licence_basis": source.licence_basis,
                "attribution": source.attribution,
                "share_alike": source.share_alike,
                "dataset": source.dataset,
                "revision": source.revision,
                "artefact": source.artefact,
                "sha256": item.sha256,
                "selection": source.selection,
                "time_s": item.time_s,
                "cocos": item.cocos,
                "cocos_candidates": item.cocos_candidates,
                "missing": item.missing,
                "references": source.references,
            }
        )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------- shared helpers


def _flat_top_midpoint(time: np.ndarray, ip: np.ndarray) -> int:
    """Index of the middle slice of the flat-top window.

    The window is every slice with ``|Ip| >= FLAT_TOP_FRACTION * max|Ip|``;
    the middle of those slices (lower middle for an even count) is returned.
    Deterministic, and away from the ramp edges where the peak |Ip| often sits.
    """
    time = np.asarray(time, dtype=float)
    ip = np.abs(np.asarray(ip, dtype=float))
    valid = np.isfinite(time) & np.isfinite(ip)
    if not valid.any() or np.nanmax(ip[valid]) <= 0.0:
        raise EquilibriumCompatibilityError("no slice with a finite, non-zero plasma current")
    window = np.flatnonzero(valid & (ip >= FLAT_TOP_FRACTION * np.max(ip[valid])))
    return int(window[(len(window) - 1) // 2])


def _convention_of(ods, time_index: int = 0):
    import copy

    from vaft.process.equilibrium import as_equilibrium

    # On a copy: reading an absent ODS path materializes an empty node, and
    # the returned ODS must hold only what the source provides.
    return as_equilibrium(copy.deepcopy(ods), time_index=time_index).convention


def _label_convention(ods, source: PublicEquilibriumSource, time_index: int, notes: list[str]):
    """Write a COCOS label when one index is pinned; return (label, candidates)."""
    from vaft.omas.general import ods_cocos, set_ods_cocos

    try:
        convention = _convention_of(ods, time_index)
    except Exception as exc:  # identification is evidence, not a reason to fail
        notes.append(f"COCOS identification failed: {exc}")
        return ods_cocos(ods), ()
    candidates = tuple(convention.identified or convention.candidates or ())
    label = ods_cocos(ods)
    if label is None and source.stated_cocos is not None:
        # The ODS holds full-weber psi, so a per-radian statement moves into
        # the 11-18 family (the rule vaft.data.eqdsk._label_cocos documents).
        stated = source.stated_cocos + 10 if source.stated_cocos < 10 else source.stated_cocos
        if stated in candidates:
            label = set_ods_cocos(
                ods, stated, source=f"producer statement COCOS {source.stated_cocos} ({source.device})"
            )
            notes.append(
                f"COCOS {stated} labelled from the producer's COCOS {source.stated_cocos} "
                f"statement; the data's candidates {candidates} include it."
            )
        else:
            notes.append(
                f"Producer states COCOS {source.stated_cocos} but the data's candidates are "
                f"{candidates}; left unlabelled."
            )
    elif label is None and len(candidates) == 1:
        label = set_ods_cocos(ods, candidates[0], source="sign identification")
    if label is None and candidates:
        notes.append(f"COCOS not pinned; the data are consistent with {candidates}.")
    elif label is None:
        notes.append(
            "COCOS identification abstained: it needs a q profile with the Ip and B0 signs, "
            "which this source does not provide together. psi keeps the source sign."
        )
    return label, candidates


def _missing(ods, time_index: int) -> tuple[str, ...]:
    from vaft.ods_access import path_value

    base = f"equilibrium.time_slice.{time_index}"
    wanted = {
        "psi map": f"{base}.profiles_2d.0.psi",
        "LCFS": f"{base}.boundary.outline.r",
        "q profile": f"{base}.profiles_1d.q",
        "pressure profile": f"{base}.profiles_1d.pressure",
        "F profile": f"{base}.profiles_1d.f",
        "vacuum B0": "equilibrium.vacuum_toroidal_field.b0",
    }
    out = []
    for label, path in wanted.items():
        value = path_value(ods, path)
        if value is None or np.size(value) == 0:
            out.append(label)
    return tuple(out)


def _provenance(ods, source: PublicEquilibriumSource, sha256: str, notes: list[str]) -> None:
    ods["dataset_description.data_entry.machine"] = source.device
    ods["equilibrium.ids_properties.homogeneous_time"] = 1
    comment = (
        f"{source.dataset} ({source.producer}), revision {source.revision}, "
        f"artefact {source.artefact} sha256:{sha256}. Licence {source.licence} "
        f"({source.licence_url}). {source.attribution} Adapted by "
        "vaft.data.public.equilibria. Conversion notes: " + " ".join(notes)
    )
    ods["equilibrium.ids_properties.comment"] = comment
    ods["equilibrium.ids_properties.provenance.node.0.path"] = "equilibrium"
    ods["equilibrium.ids_properties.provenance.node.0.sources"] = [source.url]


def _finish(ods, source, sha256, time_index, time_s, notes) -> PublicEquilibrium:
    label, candidates = _label_convention(ods, source, time_index, notes)
    _provenance(ods, source, sha256, notes)
    return PublicEquilibrium(
        ods=ods,
        source=source,
        time_index=time_index,
        time_s=time_s,
        sha256=sha256,
        cocos=label,
        cocos_candidates=tuple(int(c) for c in candidates),
        missing=_missing(ods, time_index),
        notes=tuple(notes),
    )


def _fetch_artefact(source: PublicEquilibriumSource, filename: str, cache, timeout) -> Path:
    assert source.sha256 is not None
    return fetch(source.url, sha256=source.sha256, filename=filename, cache=cache, timeout=timeout)


# --------------------------------------------------------------------- adapters


def _read_vest_sample(source, *, cache=None, timeout=120.0) -> PublicEquilibrium:
    import vaft.data
    import vaft.omas

    path = Path(vaft.data.sample(39915, representation="imas"))
    ods = vaft.omas.load(path)
    sha256 = sha256_of(path) if path.is_file() else "packaged sample (directory)"
    times = np.asarray(ods["equilibrium.time"], dtype=float)
    ip = [ods[f"equilibrium.time_slice.{i}.global_quantities.ip"] for i in range(len(times))]
    index = _flat_top_midpoint(times, ip)
    notes = [f"Representative slice {index} of {len(times)} by the flat-top midpoint rule."]
    from vaft.omas.general import ods_cocos

    label = ods_cocos(ods)
    try:
        candidates = tuple(_convention_of(ods, index).identified or ())
    except Exception as exc:
        candidates = ()
        notes.append(f"COCOS identification failed: {exc}")
    return PublicEquilibrium(
        ods=ods,
        source=source,
        time_index=index,
        time_s=float(times[index]),
        sha256=sha256,
        cocos=label,
        cocos_candidates=candidates,
        missing=_missing(ods, index),
        notes=tuple(notes),
    )


def _read_geqdsk(source, *, cache=None, timeout=120.0) -> PublicEquilibrium:
    import warnings

    import vaft.omas

    filename = f"equilibrium_{source.device.lower().replace('-', '')}_{source.revision[:12]}.geqdsk"
    path = _fetch_artefact(source, filename, cache, timeout)
    notes: list[str] = []
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            ods = vaft.omas.load(path)
        except Exception as exc:
            raise EquilibriumParseError(f"{source.device} g-file could not be read: {exc}") from exc
    for message in dict.fromkeys(str(w.message) for w in caught):
        if "IMAS DD version" not in message:
            notes.append(f"reader warning: {message}")
    from vaft.data.eqdsk import infer_source_shot_time

    # to_omas writes 0.0 when it cannot read a time; report that as unknown.
    time = infer_source_shot_time(path)[1]
    if time is None:
        notes.append(
            "The g-file states no time the reader can parse; time_s is None and the ODS "
            "slice time 0.0 is a placeholder."
        )
    return _finish(ods, source, source.sha256, 0, time, notes)


_NEURIPS_COLUMNS = (
    "source",
    "efit_times",
    "efit_psirz",
    "efit_grid_R",
    "efit_grid_Z",
    "efit_lcfs_n",
    "efit_lcfs_r",
    "efit_lcfs_z",
    "efit_r_axis",
    "efit_z_axis",
    "efit_q95",
    "efit_beta_n",
    "magnetics_plasma_current",
    "magnetics_plasma_current_times",
)


def _import_pyarrow():
    try:
        import pyarrow  # noqa: F401
        import pyarrow.parquet  # noqa: F401
    except ImportError as exc:  # pragma: no cover - exercised without the extra
        raise ImportError(
            "Reading the DIII-D and MAST public equilibria needs pyarrow; install "
            "'vaft[public]'."
        ) from exc
    import pyarrow

    return pyarrow


def _read_neurips_parquet(source, *, cache=None, timeout=120.0) -> PublicEquilibrium:
    from omas import ODS

    pa = _import_pyarrow()
    filename = f"equilibrium_diiid_{Path(source.artefact).stem}_{source.revision[:12]}.parquet"
    path = _fetch_artefact(source, filename, cache, timeout)
    try:
        table = pa.parquet.read_table(path, columns=list(_NEURIPS_COLUMNS))
    except Exception as exc:
        raise EquilibriumParseError(f"DIII-D parquet layout differs from the audit: {exc}") from exc
    if table.num_rows != 1:
        raise EquilibriumParseError(f"expected one shot row, found {table.num_rows}")
    row = {name: table.column(name)[0].as_py() for name in table.column_names}
    if row["source"] != "DIII-D":
        raise EquilibriumParseError(f"record is from {row['source']!r}, not DIII-D")

    times_ms = np.asarray(row["efit_times"], dtype=float)
    ip_ka = np.interp(
        times_ms,
        np.asarray(row["magnetics_plasma_current_times"], dtype=float),
        np.asarray(row["magnetics_plasma_current"], dtype=float),
        left=np.nan,
        right=np.nan,
    )
    index = _flat_top_midpoint(times_ms, ip_ka)
    psirz = np.asarray(row["efit_psirz"][index], dtype=float)  # rows Z, columns R
    r_grid = np.asarray(row["efit_grid_R"], dtype=float)
    z_grid = np.asarray(row["efit_grid_Z"], dtype=float)
    if psirz.shape != (z_grid.size, r_grid.size):
        raise EquilibriumParseError(
            f"efit_psirz slice {psirz.shape} does not match the (Z, R) grid "
            f"({z_grid.size}, {r_grid.size})"
        )
    n_lcfs = int(row["efit_lcfs_n"][index])
    lcfs_r = np.asarray(row["efit_lcfs_r"][index], dtype=float)[:n_lcfs]
    lcfs_z = np.asarray(row["efit_lcfs_z"][index], dtype=float)[:n_lcfs]
    if n_lcfs < 4 or not (np.isfinite(lcfs_r).all() and np.isfinite(lcfs_z).all()):
        raise EquilibriumCompatibilityError(f"DIII-D slice {index} has no usable LCFS")
    psi_wb = psirz.T * TWO_PI  # (R, Z), full weber like every ODS psi
    r_axis = float(row["efit_r_axis"][index])
    z_axis = float(row["efit_z_axis"][index])
    psi_axis = _bilinear(r_grid, z_grid, psi_wb, r_axis, z_axis)
    psi_boundary = float(np.mean([_bilinear(r_grid, z_grid, psi_wb, r, z) for r, z in zip(lcfs_r, lcfs_z)]))

    ods = ODS(consistency_check=False)
    t_s = float(times_ms[index]) * 1e-3
    ods["equilibrium.time"] = np.array([t_s])
    slice_ = ods["equilibrium.time_slice.0"]
    slice_["time"] = t_s
    slice_["profiles_2d.0.grid_type.index"] = 1
    slice_["profiles_2d.0.grid.dim1"] = r_grid
    slice_["profiles_2d.0.grid.dim2"] = z_grid
    slice_["profiles_2d.0.psi"] = psi_wb
    slice_["boundary.outline.r"] = lcfs_r
    slice_["boundary.outline.z"] = lcfs_z
    slice_["global_quantities.magnetic_axis.r"] = r_axis
    slice_["global_quantities.magnetic_axis.z"] = z_axis
    slice_["global_quantities.psi_axis"] = psi_axis
    slice_["global_quantities.psi_boundary"] = psi_boundary
    slice_["global_quantities.ip"] = float(ip_ka[index]) * 1e3
    slice_["global_quantities.q_95"] = float(row["efit_q95"][index])
    slice_["global_quantities.beta_normal"] = float(row["efit_beta_n"][index])
    notes = [
        f"EFIT slice {index} of {times_ms.size} at {times_ms[index]:g} ms by the flat-top "
        "midpoint rule.",
        "Ip is magnetics_plasma_current interpolated onto the EFIT time (derived; the "
        "record ships no EFIT Ip).",
        "psi = 2*pi * efit_psirz (Wb/rad -> Wb), transposed from (Z, R) to (R, Z); sign kept.",
        "psi_axis and psi_boundary interpolated from the flux map at the EFIT axis and "
        "averaged along the shipped LCFS (derived, not shipped).",
        "efit_li is not mapped: the card does not say which internal-inductance "
        "definition it is.",
        "No F, p or q profile and no toroidal field are shipped; none is constructed.",
    ]
    return _finish(ods, source, source.sha256, 0, t_s, notes)


def _bilinear(r_grid, z_grid, values, r, z) -> float:
    """Bilinear interpolation of ``values[(R, Z)]`` at one point."""
    i = int(np.clip(np.searchsorted(r_grid, r) - 1, 0, r_grid.size - 2))
    j = int(np.clip(np.searchsorted(z_grid, z) - 1, 0, z_grid.size - 2))
    tr = (r - r_grid[i]) / (r_grid[i + 1] - r_grid[i])
    tz = (z - z_grid[j]) / (z_grid[j + 1] - z_grid[j])
    return float(
        values[i, j] * (1 - tr) * (1 - tz)
        + values[i + 1, j] * tr * (1 - tz)
        + values[i, j + 1] * (1 - tr) * tz
        + values[i + 1, j + 1] * tr * tz
    )


# ------------------------------------------------------------- FAIR-MAST (Zarr v3)


def _zstd_decompress(payload: bytes, size: int) -> bytes:
    try:
        from compression import zstd  # Python >= 3.14

        return zstd.decompress(payload)
    except ImportError:
        pa = _import_pyarrow()
        return pa.Codec("zstd").decompress(payload, decompressed_size=size).to_pybytes()


def _mast_cache(cache) -> Path:
    return cache_dir(cache) / "fair_mast" / str(_MAST_SHOT)


def _zarr_array(meta: Mapping[str, Any], name: str, cache, timeout) -> np.ndarray:
    key = f"equilibrium/{name}"
    try:
        node = meta[key]
        shape = tuple(int(s) for s in node["shape"])
        chunk = tuple(int(c) for c in node["chunk_grid"]["configuration"]["chunk_shape"])
        codecs = node["codecs"]
        dtype = node["data_type"]
        separator = node["chunk_key_encoding"]["configuration"]["separator"]
    except (KeyError, TypeError) as exc:
        raise EquilibriumParseError(f"FAIR-MAST array {key} missing or malformed: {exc}") from exc
    names = [codec.get("name") for codec in codecs]
    if dtype not in ("float32", "float64") or names != ["bytes", "zstd"] or separator != "/":
        raise EquilibriumParseError(
            f"FAIR-MAST array {key}: dtype {dtype}, codecs {names}, separator {separator!r} "
            "differ from the audited float / bytes+zstd / '/' layout"
        )
    item = np.dtype(dtype).newbyteorder("<")
    if codecs[0].get("configuration", {}).get("endian", "little") != "little":
        raise EquilibriumParseError(f"FAIR-MAST array {key} is not little-endian")
    out = np.full(shape, np.nan)
    counts = [-(-s // c) for s, c in zip(shape, chunk)] if shape else []
    for flat in range(int(np.prod(counts)) if counts else 1):
        index = np.unravel_index(flat, counts) if counts else ()
        chunk_key = f"{key}/c" + "".join(f"/{int(i)}" for i in index)
        if chunk_key not in _MAST_CHUNKS:
            raise EquilibriumParseError(f"FAIR-MAST chunk {chunk_key} has no pinned SHA-256")
        path = fetch(
            _MAST_STORE + chunk_key,
            sha256=_MAST_CHUNKS[chunk_key],
            filename=chunk_key.replace("/", "__"),
            cache=_mast_cache(cache),
            timeout=timeout,
        )
        try:
            raw = _zstd_decompress(path.read_bytes(), int(np.prod(chunk)) * item.itemsize)
        except Exception as exc:
            raise EquilibriumParseError(f"FAIR-MAST chunk {chunk_key} did not decode: {exc}") from exc
        block = np.frombuffer(raw, dtype=item).astype(float)
        if block.size != int(np.prod(chunk)):
            raise EquilibriumParseError(f"FAIR-MAST chunk {chunk_key} decodes to {block.size} values")
        block = block.reshape(chunk)
        region = tuple(slice(i * c, min((i + 1) * c, s)) for i, c, s in zip(index, chunk, shape))
        out[region] = block[tuple(slice(0, r.stop - r.start) for r in region)]
    return out


def _read_fair_mast(source, *, cache=None, timeout=120.0) -> PublicEquilibrium:
    from omas import ODS

    root = fetch(
        _MAST_STORE + "zarr.json",
        sha256=source.sha256,
        filename="zarr.json",
        cache=_mast_cache(cache),
        timeout=timeout,
    )
    try:
        meta = json.loads(root.read_text(encoding="utf-8"))["consolidated_metadata"]["metadata"]
        attributes = meta["equilibrium"]["attributes"]
    except (KeyError, ValueError, TypeError) as exc:
        raise EquilibriumParseError(f"FAIR-MAST zarr.json layout differs from the audit: {exc}") from exc
    if (
        attributes.get("license_name") != _MAST_LICENCE_NAME
        or attributes.get("license_url") != _CC_BY_SA
    ):
        raise EquilibriumLicenceError(
            "FAIR-MAST equilibrium group declares licence "
            f"{attributes.get('license_name')!r} ({attributes.get('license_url')!r}), not the "
            f"audited {_MAST_LICENCE_NAME!r}."
        )
    data = {name: _zarr_array(meta, name, cache, timeout) for name in _MAST_ARRAYS}
    index = _flat_top_midpoint(data["time"], data["ip"])
    has_x_point = np.isfinite(data["x_point_r"][:, index]).any()

    psi_wb = data["psi"][:, :, index].T * TWO_PI  # (z, R) -> (R, Z), Wb/rad -> Wb
    psi_axis = float(data["psi_axis"][index]) * TWO_PI
    psi_boundary = float(data["psi_boundary"][index]) * TWO_PI
    lcfs = np.isfinite(data["lcfs_r"][:, index]) & np.isfinite(data["lcfs_z"][:, index])
    if lcfs.sum() < 4 or not np.isfinite(psi_wb).all():
        raise EquilibriumCompatibilityError(f"MAST slice {index} has no complete flux map and LCFS")

    ods = ODS(consistency_check=False)
    t_s = float(data["time"][index])
    ods["dataset_description.data_entry.pulse"] = _MAST_SHOT
    ods["equilibrium.time"] = np.array([t_s])
    ods["equilibrium.vacuum_toroidal_field.r0"] = float(data["bvac_r"][index])
    ods["equilibrium.vacuum_toroidal_field.b0"] = np.array([float(data["bvac_val"][index])])
    slice_ = ods["equilibrium.time_slice.0"]
    slice_["time"] = t_s
    slice_["profiles_2d.0.grid_type.index"] = 1
    slice_["profiles_2d.0.grid.dim1"] = data["major_radius"]
    slice_["profiles_2d.0.grid.dim2"] = data["z"]
    slice_["profiles_2d.0.psi"] = psi_wb
    slice_["boundary.outline.r"] = data["lcfs_r"][lcfs, index]
    slice_["boundary.outline.z"] = data["lcfs_z"][lcfs, index]
    slice_["global_quantities.magnetic_axis.r"] = float(data["magnetic_axis_r"][index])
    slice_["global_quantities.magnetic_axis.z"] = float(data["magnetic_axis_z"][index])
    slice_["global_quantities.psi_axis"] = psi_axis
    slice_["global_quantities.psi_boundary"] = psi_boundary
    slice_["global_quantities.ip"] = float(data["ip"][index])
    psi_norm = data["psi_norm"]
    slice_["profiles_1d.psi"] = psi_axis + psi_norm * (psi_boundary - psi_axis)
    slice_["profiles_1d.f"] = data["f"][:, index]
    slice_["profiles_1d.pressure"] = data["pressure"][:, index]
    slice_["global_quantities.q_95"] = float(data["q95"][index])
    notes = [
        f"EFIT++ slice {index} of {data['time'].size} at {t_s:g} s by the flat-top midpoint "
        f"rule; X-point present: {bool(has_x_point)}.",
        "psi, psi_axis, psi_boundary = 2*pi * stored Wb/rad values; psi transposed from "
        "(z, R) to (R, Z); sign kept.",
        "profiles_1d.psi = psi_axis + psi_norm * (psi_boundary - psi_axis) on the shipped "
        "psi_norm grid; f and pressure copied on that grid.",
        "The q profile is not mapped: FAIR-MAST stores it on profile_r, a midplane "
        "coordinate running inboard to outboard, not on psi_norm. q95 is mapped as a scalar.",
        "psi_axis/psi_boundary are labelled 'Wb' in the store but equal the Wb/rad map at "
        "the axis, so they are scaled with psi.",
        "B0 and r0 are EFIT++ bvac_val and bvac_r at the slice.",
        "Measured diagnostics (*_measured) are not used.",
    ]
    return _finish(ods, source, source.sha256, 0, t_s, notes)


def _read_imas_netcdf(source, *, cache=None, timeout=120.0) -> PublicEquilibrium:
    import warnings

    from omas import ODS

    import vaft.omas

    filename = f"equilibrium_{source.device.lower()}_{source.artefact}"
    path = _fetch_artefact(source, filename, cache, timeout)
    notes: list[str] = []
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            full = vaft.omas.load(path, imas_version=_ITER_DD_VERSION)
        except Exception as exc:
            raise EquilibriumParseError(
                f"{source.device} IMAS netCDF could not be read: {exc}"
            ) from exc
    for message in dict.fromkeys(str(w.message) for w in caught):
        if "IMAS DD version" not in message and "binary incompatibility" not in message:
            notes.append(f"reader warning: {message}")
    if "equilibrium" not in full.keys():
        raise EquilibriumParseError(f"{source.device} artefact holds no equilibrium IDS")
    put = full["equilibrium.ids_properties"].get("version_put.data_dictionary", None)
    if put != _ITER_DD_VERSION:
        raise EquilibriumParseError(
            f"{source.device} artefact declares DD {put!r}, audited as {_ITER_DD_VERSION}"
        )
    times = np.atleast_1d(np.asarray(full["equilibrium.time"], dtype=float))
    ip = [full[f"equilibrium.time_slice.{i}.global_quantities.ip"] for i in range(len(times))]
    index = _flat_top_midpoint(times, ip)
    ods = ODS(consistency_check=False)
    ods["equilibrium"] = full["equilibrium"]
    dropped = sorted(k for k in full.keys() if k != "equilibrium")
    if dropped:
        notes.append(f"Only the equilibrium IDS is returned; not adapted: {', '.join(dropped)}.")
    if len(times) > 1:
        notes.append(f"Representative slice {index} of {len(times)} by the flat-top midpoint rule.")
    return _finish(ods, source, source.sha256, index, float(times[index]), notes)


_READERS = {
    "vaft_sample": _read_vest_sample,
    "geqdsk": _read_geqdsk,
    "neurips_parquet": _read_neurips_parquet,
    "fair_mast_zarr": _read_fair_mast,
    "imas_netcdf": _read_imas_netcdf,
}
