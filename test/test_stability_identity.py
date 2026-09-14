"""The identity contract for equilibrium families and stability products (#527 stage 2).

A stability result is identified by

    (shot, time, family, refinement, product, n_tor)

where `family` is the reconstruction lineage (`magnetic`, `electron`, `kinetic`),
`refinement` is the equilibrium actually handed to the solver (`chease`), and
`product` is the stability calculation (`dcon-peeling`, `dcon-kink`, `rdcon`,
`stride`, `ideal-gpec`).

DCON's two edge treatments are separate *products* rather than two views of one
run: with `psiedge < psilim` DCON overwrites its own controls and re-integrates
(`dcon/dcon.F:262-279`), so a run's eigenvalues and `euler.bin` describe the
truncated solution and the full-edge one is discarded. One run yields one of the
two and can never yield both.

These tests are the executable specification for that identity. They were written
against a tree that could not satisfy them, and each one failed for the reason it
names before the resolver was extended.
"""

from __future__ import annotations

import pytest

from vaft.database import sources as _sources
from vaft.database import filedb
from vaft.database.filedb import FileDB

FAMILIES = ("magnetic", "electron", "kinetic")
DCON_PRODUCTS = ("dcon-peeling", "dcon-kink")
SHOT = 39915


@pytest.fixture()
def db(tmp_path):
    return FileDB(tmp_path / "FileDB")


# --- FileDB identity ---------------------------------------------------------


def test_each_equilibrium_family_gets_its_own_stability_identity(db):
    """The three families must not collide before all three pipelines exist.

    `electron-efit` and `kinetic-efit` are already catalogued lineages with no
    pipeline behind them yet, so the storage model has to keep them apart now
    rather than after the first collision.
    """
    paths = {
        family: db.gpec("dcon-kink", SHOT, 1, family=family, refinement="chease")
        for family in FAMILIES
    }

    assert len(set(paths.values())) == len(FAMILIES)
    for family, path in paths.items():
        assert family in path.parts
        assert "chease" in path.parts
        assert "dcon-kink" in path.parts


def test_the_two_dcon_edge_products_are_independent_identities(db):
    """A peeling run and a kink run of the same cell are different products.

    They are separate DCON executions with different integration boundaries, so
    sharing a leaf would mean the second overwrote the first.
    """
    peeling, kink = (
        db.gpec(product, SHOT, 1, family="magnetic", refinement="chease")
        for product in DCON_PRODUCTS
    )

    assert peeling != kink
    # They differ in exactly one segment: the product.
    differing = [a for a, b in zip(peeling.parts, kink.parts) if a != b]
    assert differing == ["dcon-peeling"]


def test_the_artifact_class_stays_the_leaf_under_the_new_dimensions(db):
    cell = db.gpec("dcon-kink", SHOT, 1, family="magnetic", refinement="chease")

    for artifact in ("work", "output", "log", "metadata"):
        assert db.gpec(
            "dcon-kink", SHOT, 1, family="magnetic", refinement="chease", artifact=artifact
        ) == cell / artifact


def test_the_reconstruction_and_refinement_domains_carry_the_family_too(db):
    """Otherwise a kinetic-EFIT run overwrites the magnetic one for the same shot."""
    assert len({db.efit(SHOT, family=family) for family in FAMILIES}) == len(FAMILIES)
    assert len({db.chease(SHOT, family=family) for family in FAMILIES}) == len(FAMILIES)


@pytest.mark.xfail(
    strict=True,
    reason=(
        "Per-product stage assembly is not built yet (#137). The resolver already "
        "carries the dimensions; what is missing is a builder that emits one "
        "`mhd_linear` per product instead of folding DCON, RDCON and STRIDE into "
        "one file per shot, and the per-product HSDS sources to receive them. "
        "strict=True so this turns red -- not quietly green -- the day "
        "`_PRODUCT_STAGES` is populated."
    ),
)
def test_the_stage_product_is_per_family_and_per_product(db):
    """One logical product owns one `mhd_linear`, so its path must say which."""
    kink = db.omas_product(
        "mhd_linear", shot=SHOT, family="magnetic", refinement="chease", product="dcon-kink"
    )
    peeling = db.omas_product(
        "mhd_linear", shot=SHOT, family="magnetic", refinement="chease", product="dcon-peeling"
    )
    electron = db.omas_product(
        "mhd_linear", shot=SHOT, family="electron", refinement="chease", product="dcon-kink"
    )

    assert len({kink, peeling, electron}) == 3


def test_the_deferred_product_branch_resolves_when_it_is_switched_on(db, monkeypatch):
    """Exercise the branch #137 will enable, before #137 enables it.

    `_PRODUCT_STAGES` is empty today, so the code that appends a refinement and
    a product to a stage path never runs -- and would execute for the first time
    in production path resolution the day that set is populated. Populating it
    here runs it now: segment order, the shot that follows, and the refusal of a
    product for a stage that still does not take one.
    """
    monkeypatch.setattr(
        filedb, "_PRODUCT_STAGES", frozenset({"mhd_linear"}), raising=True
    )

    assert db.omas(
        "mhd_linear", shot=SHOT, family="magnetic", refinement="chease",
        product="dcon-kink", artifact="output",
    ) == db.root / f"omas/mhd_linear/magnetic/chease/dcon-kink/{SHOT}/output"

    # A stage outside the set still refuses both, so enabling one stage cannot
    # quietly loosen the others.
    with pytest.raises(filedb.FileDBPathError, match="product is not valid"):
        db.omas("chease", shot=SHOT, family="magnetic", product="dcon-kink")

    # And the enabled stage still requires them.
    with pytest.raises(filedb.FileDBPathError, match="refinement"):
        db.omas("mhd_linear", shot=SHOT, family="magnetic")


def test_a_stability_path_without_a_family_is_refused(db):
    """A missing dimension must be an error, never a silent default to magnetic.

    The message has to name `family`: today this raises for an unrelated reason
    (`gpec()` has no such parameter at all), and a test that accepts any
    TypeError would keep passing without specifying anything.
    """
    with pytest.raises(TypeError, match="family"):
        db.gpec("dcon-kink", SHOT, 1, refinement="chease")


def test_an_unknown_family_or_product_is_refused_by_name(db):
    with pytest.raises(Exception) as unknown_family:
        db.gpec("dcon-kink", SHOT, 1, family="not-a-family", refinement="chease")
    assert "family" in str(unknown_family.value).lower()

    with pytest.raises(Exception) as unknown_product:
        db.gpec("dcon-sideways", SHOT, 1, family="magnetic", refinement="chease")
    assert "product" in str(unknown_product.value).lower()


# --- logical source identity -------------------------------------------------
#
# Everything below specifies the HSDS side, which is not built yet: `_NAME` in
# `vaft/database/sources.py` rejects a `/`-joined name outright, there is no
# alias mechanism, and `STAGE_REPLICATION` is a 1:1 stage -> source map. These
# are `xfail(strict=True)` rather than deleted or commented out, so the target
# stays executable and the suite turns red the day the implementation lands
# without them being updated.
pytestmark_hsds = pytest.mark.xfail(
    strict=True,
    reason="Hierarchical HSDS sources are not implemented yet (#77/#94).",
)


def test_the_logical_source_hierarchy_resolves_as_written():
    """Logical source names mirror the scientific chain and stay distinct."""
    names = (
        "main",
        "main/chease",
        "main/chease/dcon-peeling",
        "main/chease/dcon-kink",
        "main/chease/rdcon",
        "main/chease/stride",
    )

    resolved = {name: _sources.resolve(name) for name in names}

    assert len(set(resolved.values())) == len(names)
    assert resolved["main"] == "main"


def test_the_three_families_are_three_sources_before_their_pipelines_exist():
    resolved = {
        family: _sources.resolve(f"{family}/chease/dcon-kink")
        for family in ("main", "electron-efit", "kinetic-efit")
    }

    assert len(set(resolved.values())) == 3


def test_magnetic_efit_is_a_read_only_projection_of_main():
    """The alias names an equilibrium family, and creates no dataset of its own.

    It resolves to `main`'s storage, so it can never be a write destination --
    otherwise the same product would have two writable canonical homes, which is
    the collision named sources exist to prevent.
    """
    assert _sources.resolve("magnetic-efit") == "main"

    entry = _sources.CATALOG["magnetic-efit"]
    assert entry.projects_to == "main"
    assert entry.writable is False

    with pytest.raises(_sources.ReadOnlySourceError, match="read-only projection"):
        _sources.resolve("magnetic-efit", writable=True)


@pytest.mark.xfail(
    strict=True,
    reason=(
        "`chease-mhd-stability` is still the live destination for mhd_linear, "
        "so it must stay writable until the per-product assembly (#137) moves "
        "that stage onto the hierarchy. Retiring it before then would leave the "
        "stability stage with nowhere to write. It becomes read-only in the "
        "same change that moves the destinations, and is deleted by the gated "
        "migration in #94."
    ),
)
def test_the_legacy_combined_source_stays_readable_and_unwritable():
    """`chease-mhd-stability` is migrated and retired, not silently redirected.

    It must not alias onto the new hierarchy: that would be implicit composition
    of several products behind one name.
    """
    assert _sources.resolve("chease-mhd-stability") == "chease-mhd-stability"

    with pytest.raises(_sources.HSDSSourceError):
        _sources.resolve("chease-mhd-stability", writable=True)


def test_a_malformed_hierarchical_name_is_still_refused():
    """Admitting `/` must not admit traversal, empty segments or shot folders."""
    for name in ("main/", "/main", "main//chease", "main/39915", "main/../public", "Main/chease"):
        with pytest.raises(_sources.HSDSSourceError):
            _sources.resolve(name)


def test_loading_a_parent_never_composes_its_children():
    """`main` must not acquire its refinement or stability products implicitly.

    The hierarchy makes the relationship *visible*, which is the whole point --
    and is also the thing that could quietly turn into composition. Naming a
    parent resolves to that parent alone; walking to a child is a separate call
    a caller has to make.
    """
    parent = _sources.resolve("main")
    child = _sources.resolve("main/chease/dcon-kink")

    assert child != parent
    assert isinstance(parent, str), "resolution is one source, never a union"

    # The children are discoverable, and discovering them composes nothing.
    assert [entry.name for entry in _sources.children("main")] == ["main/chease"]
    assert _sources.resolve("main") == parent
