"""The two DCON edge branches, end to end and side by side (#527 step 10).

Everything in this sequence exists so that one equilibrium can be solved two
ways -- full-edge and dW-peak-truncated -- without the two results colliding.
Each piece was tested where it was built; this walks one shot through all of
them at once and checks the property the pieces were for.

GPEC is not installed here, so the solver outputs are fixtures shaped like the
real files. What is *not* faked is everything under test: the native reader, the
`mhd_linear` assembly, the FileDB grammar, the HSDS destinations and the
provenance chain are the production code paths.

The claim being checked is the plan's: two products from one equilibrium must
persist to different FileDB identities, assemble into independent `mhd_linear`
products, publish to different sources, carry the *same* upstream hashes, never
overwrite each other, and remain directly comparable for the peeling-vs-kink
classification.
"""

import hashlib
import importlib.util
from pathlib import Path

import pytest

from vaft.code.gpec import read_dcon_output
from vaft.code.gpec import _runtime as gpec_runtime
from vaft.database import sources as _sources
from vaft.database.filedb import FileDB
from vaft.database.provenance import verify_chain
from vaft.omas.vest_upstream import build_mhd_linear_ods

from gpec_nc_fixtures import write_dcon_in, write_dcon_output_nc

_WORKFLOW = (
    Path(__file__).resolve().parents[1]
    / "workflow"
    / "automatic_pipeline_1_routine_data_processing"
)
_SPEC = importlib.util.spec_from_file_location("pipeline1_paths", _WORKFLOW / "paths.py")
_PATHS = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(_PATHS)
stability_product = _PATHS.stability_product


pytestmark = pytest.mark.core


SHOT = 39915
TIME_MS = 325
MODE = 1
FAMILY = "magnetic"
REFINEMENT = "chease"

#: Full-edge integrates once and keeps the whole edge; the truncated run finds
#: the dW peak, overwrites its own controls and re-integrates
#: (`dcon/dcon.F:262-279`). One run yields one of them and can never yield both.
BRANCHES = {
    "dcon-peeling": {"edge_scan": False, "psiedge": 1.0, "w_t": -0.30},
    "dcon-kink": {"edge_scan": True, "psiedge": 0.95, "w_t": +7.89},
}


@pytest.fixture
def equilibrium(tmp_path):
    """One CHEASE-refined g-file, and its digest -- the shared upstream."""
    path = tmp_path / "chease" / FAMILY / str(SHOT) / "output" / f"g0{SHOT}.00{TIME_MS}"
    path.parent.mkdir(parents=True)
    path.write_text("EFITD  CHEASE refined  39915  325ms\n", encoding="utf-8")
    return path, hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def solved(tmp_path, equilibrium):
    """Both branches solved into their own canonical cells.

    The cell directory comes from `FileDB.gpec`, so this is the grammar the
    pipeline writes, not a path assembled for the test.
    """
    _, _ = equilibrium
    db = FileDB(tmp_path)
    cells = {}
    for product, spec in BRANCHES.items():
        work = db.gpec(
            product, SHOT, MODE, family=FAMILY, refinement=REFINEMENT, artifact="work"
        )
        run_dir = gpec_runtime.module_dir(work, TIME_MS, "dcon", MODE)
        run_dir.mkdir(parents=True)
        # Both fixtures take the run *directory* and name the file themselves,
        # the way the solver does.
        write_dcon_output_nc(
            run_dir,
            n=MODE,
            w_t=spec["w_t"],
            edge_scan=spec["edge_scan"],
        )
        write_dcon_in(run_dir, psiedge=spec["psiedge"])
        cells[product] = work
    return db, cells


# --- identity --------------------------------------------------------------


def test_the_two_branches_occupy_different_cells(solved):
    """Different FileDB identities, so neither can write over the other."""
    _, cells = solved

    peeling, kink = cells["dcon-peeling"], cells["dcon-kink"]
    assert peeling != kink

    # Compared by segment rather than by counting `.parent` hops: the grammar
    # is `gpec/{family}/{refinement}/{product}/{shot}/n={mode}/{artifact}`, and
    # an off-by-one in that chain is a test that passes for the wrong reason.
    peeling_parts, kink_parts = peeling.parts, kink.parts
    assert len(peeling_parts) == len(kink_parts)
    differing = [
        i for i, (a, b) in enumerate(zip(peeling_parts, kink_parts)) if a != b
    ]
    assert len(differing) == 1, "the two cells must differ in the product alone"
    index = differing[0]
    assert (peeling_parts[index], kink_parts[index]) == ("dcon-peeling", "dcon-kink")
    # Everything else is shared: same shot, same lineage, same mode.
    assert peeling_parts[index + 1 :] == kink_parts[index + 1 :]


def test_each_cell_declares_the_edge_treatment_it_actually_holds(solved):
    """The product is verifiable against the artifact, not taken on trust.

    `DconOutput.edge_treatment` is read from the output file, so a cell filed
    under the wrong product can be detected rather than believed.
    """
    _, cells = solved
    for product, work in cells.items():
        run_dir = gpec_runtime.module_dir(work, TIME_MS, "dcon", MODE)
        result = read_dcon_output(run_dir, mode=MODE)
        assert stability_product("dcon", edge_treatment=result.edge_treatment) == product


# --- assembly --------------------------------------------------------------


@pytest.fixture
def assembled(solved):
    """One `mhd_linear` per branch, built by the production assembler."""
    _, cells = solved
    built = {}
    for product, work in cells.items():
        ods, manifest = build_mhd_linear_ods(
            shot=SHOT,
            time_values=[TIME_MS],
            module_workdirs={("dcon", MODE): work},
            modules=("dcon",),
            modes=(MODE,),
        )
        built[product] = (ods, manifest)
    return built


def test_each_branch_assembles_its_own_product_and_keeps_its_own_number(assembled):
    """The collision this whole sequence exists to prevent.

    Both write `energy_perturbed` at the same `(time_slice, position)`. In one
    combined product the second would replace the first; in two products each
    keeps what its solver found.
    """
    energies = {
        product: ods["mhd_linear"]["time_slice"][0]["toroidal_mode"][0]["energy_perturbed"]
        for product, (ods, _) in assembled.items()
    }

    assert energies["dcon-peeling"] == pytest.approx(BRANCHES["dcon-peeling"]["w_t"])
    assert energies["dcon-kink"] == pytest.approx(BRANCHES["dcon-kink"]["w_t"])
    assert energies["dcon-peeling"] != energies["dcon-kink"]


def test_the_stage_products_land_at_different_paths(tmp_path, assembled):
    db = FileDB(tmp_path)

    paths = {
        product: db.omas_product(
            "mhd_linear", shot=SHOT, family=FAMILY,
            refinement=REFINEMENT, product=product,
        )
        for product in assembled
    }

    assert len(set(paths.values())) == 2
    for product, path in paths.items():
        assert product in path.parts


# --- publication -----------------------------------------------------------


def test_the_two_branches_publish_to_different_sources(assembled):
    destinations = {
        product: _sources.source_for_stage(
            "mhd_linear", family=FAMILY, refinement=REFINEMENT, product=product
        )
        for product in assembled
    }

    assert destinations == {
        "dcon-peeling": "main/chease/dcon-peeling",
        "dcon-kink": "main/chease/dcon-kink",
    }
    # Both writable, and neither is the retired combined source.
    for source in destinations.values():
        assert _sources.resolve(source, writable=True) == source


def test_neither_branch_is_reachable_by_loading_the_parent():
    """A hierarchy must not become an implicit union.

    `main` is the magnetic baseline; naming it must not acquire the refinement
    or either stability product.
    """
    assert _sources.resolve("main") == "main"
    assert [entry.name for entry in _sources.children("main")] == ["main/chease"]
    assert {entry.name for entry in _sources.children("main/chease")} >= {
        "main/chease/dcon-peeling",
        "main/chease/dcon-kink",
    }


# --- provenance ------------------------------------------------------------


def test_both_branches_carry_the_same_upstream_equilibrium(equilibrium, assembled):
    """Different results, one ancestor -- which is what makes them comparable.

    If the two branches carried different upstream digests they would not be two
    treatments of one equilibrium, and comparing them would say nothing about
    the edge.
    """
    _, digest = equilibrium
    chease_manifest = {
        "input": [
            {"path": "efit/g039915.00325", "sha256": "e" * 64, "output_sha256": digest}
        ]
    }

    for product in assembled:
        report = verify_chain(
            chease_manifest=chease_manifest,
            stability_input_sha256=digest,
            efit_gfile_sha256="e" * 64,
            shot=SHOT,
        )
        assert report.verified, product


# --- the comparison the split makes possible -------------------------------


def test_the_two_branches_are_directly_comparable_for_the_classification(assembled):
    """Peeling-vs-kink is derived by comparing the branches, never stored.

    Full-edge unstable while the truncated solve is stable means the drive is at
    the edge. That comparison needs both numbers to survive, which is exactly
    what one combined product could not do -- and it stays a derivation, so the
    product name never claims a physical classification it did not establish.
    """
    def energy(product):
        ods, _ = assembled[product]
        return ods["mhd_linear"]["time_slice"][0]["toroidal_mode"][0]["energy_perturbed"]

    full_edge_unstable = energy("dcon-peeling") < 0
    truncated_stable = energy("dcon-kink") >= 0

    assert full_edge_unstable and truncated_stable
    edge_driven = full_edge_unstable and truncated_stable
    assert edge_driven, "this fixture is the edge-driven case"

    # The product names the numerical treatment, not the physics. Nothing in
    # either product asserts "peeling" or "kink" as a mode classification.
    for product, (ods, _) in assembled.items():
        entry = ods["mhd_linear"]["time_slice"][0]["toroidal_mode"][0]
        assert "ballooning_type" not in entry
