"""The table VAFT ships, and the claim that it knows where it came from (#695).

Until #695 the packaged Green tables had no recoverable origin: which EFUND
build produced them, from which input, was not answerable. They are now the
output of a recorded run over the canonical static geometry, and the manifest
that says so sits beside them. These tests hold that shipped directory to the
two things that make the claim mean anything -- the manifest describes the
files actually present, and it names the assets it was built from.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from vaft.data.resources import data_path

pytest.importorskip("f90nml")

TABLE_DIRECTORY = Path(data_path("efit"))
MANIFEST = TABLE_DIRECTORY / "efund_table_manifest.json"


@pytest.fixture(scope="module")
def manifest():
    if not MANIFEST.is_file():
        pytest.fail(f"{MANIFEST} is missing; the packaged table must carry its provenance")
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


def test_the_packaged_table_records_its_own_provenance(manifest):
    from vaft.code.efit.efund import table_identity

    record = table_identity(TABLE_DIRECTORY)
    assert record["provenance"] == "manifest"
    assert record["identity"] == manifest["table"]["identity"]
    # The era and the EFUND build, which is what "recorded origin" means.
    assert manifest["machine"]["era"] == "vest-pre-43017-pf1906"
    assert manifest["efund"]["executable"]["sha256"]
    assert manifest["efund"]["config"]["grid"] == {
        "nw": 129, "nh": 129, "rleft": 0.05, "rright": 1.2, "zbotto": -1.5, "ztop": 1.5,
    }
    # Every static asset it was projected from, by hash.
    assets = manifest["machine"]["static_inputs"]
    assert set(assets) >= {"static_geometry", "pf_geometry", "magnetics_geometry"}
    for name, record in assets.items():
        assert len(record["sha256"]) == 64, name


def test_the_manifest_describes_the_files_that_are_there(manifest):
    """A manifest that does not match the bytes beside it is worse than none.

    The packaged `.ddd` files are Git LFS objects. In a checkout without them
    this reads pointer files and the hashes will not match, which is why the
    LFS guard runs first: a missing asset must skip, never fail as a
    provenance defect.
    """
    from lfs_assets import skip_unless_materialized

    skip_unless_materialized(*sorted(TABLE_DIRECTORY.glob("*.ddd")))

    for name, record in manifest["table"]["files"].items():
        path = TABLE_DIRECTORY / name
        assert path.is_file(), name
        assert path.stat().st_size == record["size"], name
        assert hashlib.sha256(path.read_bytes()).hexdigest() == record["sha256"], name


def test_the_shipped_namelist_now_matches_the_tables(manifest):
    """Reversed by #708, and it had to be.

    Until #708 the shipped `mhdin.dat` was the legacy 16-lumped-conductor file
    while the tables beside it were built from 302 filaments -- inert, because
    EFIT takes the coil response from the tables and only `nfsum` from the
    namelist, and the two agreed on that one number.

    They no longer do. The table describes 26 current groups, so a namelist
    saying 16 would have EFIT reading a different coilset from the one the
    k-file writes, which `generate_kfile` now refuses outright. The shipped
    namelist is therefore the one that built the tables.
    """
    import f90nml

    packaged = f90nml.read(str(TABLE_DIRECTORY / "mhdin.dat"))
    assert packaged["machinein"]["nfcoil"] == 530
    assert manifest["machine"]["counts"]["nfcoil"] == 530
    assert manifest["machine"]["counts"]["nfsum"] == packaged["machinein"]["nfsum"] == 26


def test_the_legacy_namelist_is_preserved_as_an_independent_reference():
    """The one thing the switch could have quietly cost, kept on purpose.

    The legacy namelist describes the same machine written by other hands, so
    `test_efund_geometry.py` uses it to check that the canonical projection is
    right rather than merely self-consistent. It could not stay in the table
    directory -- its `nfsum` is 16 and would be read by EFIT -- so it moved to
    `legacy/`, with a README saying why it is there and that it is no longer
    loadable as a table.
    """
    import f90nml

    legacy = TABLE_DIRECTORY / "legacy" / "mhdin.dat"
    assert legacy.is_file(), "the independent geometry reference must not be dropped"
    assert (TABLE_DIRECTORY / "legacy" / "README.md").is_file()

    parsed = f90nml.read(str(legacy))
    assert parsed["machinein"]["nfcoil"] == 16
    assert parsed["machinein"]["nfsum"] == 16
    # And it is genuinely a different description, not a copy of the new one.
    shipped = f90nml.read(str(TABLE_DIRECTORY / "mhdin.dat"))
    assert parsed["machinein"]["nfcoil"] != shipped["machinein"]["nfcoil"]


def test_the_acceptance_envelope_was_not_changed_by_the_switch():
    """Scope: #695 moved the table, and nothing else.

    The `&incheck` bounds EFIT reads are a separate decision (#649), still
    carried by the packaged namelist at their packaged values. A switch that
    quietly moved them too would confound the two.
    """
    import f90nml

    from vaft.code.efit.config import EFITAcceptanceEnvelope

    bundled = f90nml.read(str(TABLE_DIRECTORY / "mhdin.dat"))["incheck"]
    envelope = EFITAcceptanceEnvelope()
    for field in ("aminor_min", "aminor_max", "rcntr_min", "rcntr_max"):
        assert bundled[field] == pytest.approx(getattr(envelope, field)), field


# --- machine era (#805) -----------------------------------------------------


def test_the_table_says_which_machine_era_it_was_built_for(manifest):
    """A Green table is only valid for the geometry it was projected from."""
    from vaft.code.efit.efund import table_identity, table_machine_era

    assert table_machine_era(TABLE_DIRECTORY) == manifest["machine"]["era"]
    assert table_identity(TABLE_DIRECTORY)["era"] == manifest["machine"]["era"]


def test_a_directory_with_no_manifest_makes_no_era_claim(tmp_path):
    """Unknown is not the same as matching, and must not read as a pass."""
    from vaft.code.efit.efund import table_machine_era

    (tmp_path / "mhdin.dat").write_text(" &machinein\n nfsum = 26\n /\n", encoding="utf-8")
    assert table_machine_era(tmp_path) is None


def test_a_table_from_another_era_is_refused(tmp_path):
    """#805: the mismatch that reconstructed shot 46742 against wrong coils.

    The packaged table is `vest-pre-43017-pf1906`. Shot 46742 is
    `vest-45967-plus-pf2507`, where twenty-four filaments belong to PF6
    instead of PF7 and sit 16 cm further out in z. Running it against the
    packaged table produced two g-files from twenty-five slices, both of them
    vacuum -- no plasma equilibrium at all -- where the era-matched table
    produced eight. Nothing reported a problem; that is what this refuses.

    The era is compared as an opaque string the caller supplies, so the
    writer stays free of any particular machine's era list.
    """
    from vaft.code.efit import generate_constraints_ods

    with pytest.raises(ValueError, match="was built for machine era"):
        generate_constraints_ods(
            None, 46742, str(tmp_path), str(TABLE_DIRECTORY) + "/", [], [], [],
            expected_table_era="vest-45967-plus-pf2507",
        )


def test_the_matching_era_is_not_refused(tmp_path, manifest):
    """The other half: the guard must not block the routine case.

    It has to fail somewhere further on -- the ODS here is `None` -- but it
    must not fail on the era.
    """
    from vaft.code.efit import generate_constraints_ods

    with pytest.raises(Exception) as caught:
        generate_constraints_ods(
            None, 39915, str(tmp_path), str(TABLE_DIRECTORY) + "/", [], [], [],
            expected_table_era=manifest["machine"]["era"],
        )
    assert "was built for machine era" not in str(caught.value)


def test_a_generated_table_directory_is_runnable(tmp_path):
    """#805: EFUND's output alone is not a table EFIT can read.

    EFIT also reads a limiter contour and a probe description from the table
    directory, and EFUND writes neither. A generated directory without them
    fails in `read_limiter.f90` with a Fortran runtime error naming
    `lim.dat` -- a long way from "the table is incomplete". Both files are
    era-independent, so the generator copies them from the packaged
    directory.
    """
    import importlib.util
    import sys
    from pathlib import Path

    script = Path(__file__).resolve().parents[1] / "workflow" / "efit_tables" / "regenerate_legacy_table.py"
    spec = importlib.util.spec_from_file_location("regenerate_legacy_table", script)
    module = importlib.util.module_from_spec(spec)
    sys.modules["regenerate_legacy_table"] = module
    spec.loader.exec_module(module)

    assert module._copy_runtime_companions(tmp_path) == list(module.RUNTIME_COMPANIONS)
    for name in module.RUNTIME_COMPANIONS:
        assert (tmp_path / name).is_file(), name
        assert (tmp_path / name).read_bytes() == (TABLE_DIRECTORY / name).read_bytes()

    # Idempotent, and it never overwrites a file the generator itself wrote.
    (tmp_path / "lim.dat").write_text("mine", encoding="utf-8")
    assert module._copy_runtime_companions(tmp_path) == []
    assert (tmp_path / "lim.dat").read_text(encoding="utf-8") == "mine"


# --- identity reproducibility (#793) ----------------------------------------


def test_the_identity_is_decided_by_the_table_and_not_by_the_run():
    """#793: the same inputs must produce the same identity.

    `table.identity` exists so two runs can be compared. It could not be:
    `mhdout.dat` was in the digest, and it carries an uninitialised Fortran
    value. `KUBICS` is 4 in the input and came back 83664424 from the run that
    built the #695 table and 4890152 from the run that built the #708 one --
    an order of magnitude apart, from the same input.

    So any check of the form "regenerate and confirm the identity is
    unchanged" failed for a reason with nothing to do with the table. #708 had
    to fall back to comparing `ep129129.ddd` byte for byte, which worked only
    because that one file happens to have no `nfsum` term.
    """
    from vaft.code.efit.efund import IDENTITY_EXCLUDED_FILES, _table_digest

    assert "mhdout.dat" in IDENTITY_EXCLUDED_FILES

    table = {"rfcoil.ddd": "aaa", "ec129129.ddd": "bbb", "mhdout.dat": "run-one"}
    rerun = {"rfcoil.ddd": "aaa", "ec129129.ddd": "bbb", "mhdout.dat": "run-two"}
    assert _table_digest(table) == _table_digest(rerun)

    # And it is still an identity: a table file that moves changes it.
    moved = dict(table, **{"ec129129.ddd": "ccc"})
    assert _table_digest(moved) != _table_digest(table)


def test_the_excluded_file_is_still_recorded(manifest):
    """Excluded from the identity is not excluded from the record.

    `mhdout.dat` is what EFUND echoed back, which is worth keeping even though
    it cannot be compared across runs. It stays under `table.files` with its
    hash and size; it simply does not decide identity.
    """
    assert "mhdout.dat" in manifest["table"]["files"]
    assert manifest["table"]["files"]["mhdout.dat"]["sha256"]

    from vaft.code.efit.efund import _table_digest

    hashes = {name: record["sha256"] for name, record in manifest["table"]["files"].items()}
    assert manifest["table"]["identity"] == _table_digest(hashes)
    # The shipped manifest says why, so a reader does not have to find #793.
    assert "KUBICS" in manifest["extra"]["identity_excludes_mhdout"]
