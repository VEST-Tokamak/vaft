"""Promoting a NEO result to the database (issue #742, #550 phase 8).

Three things are asserted here, and they are the issue's own acceptance criteria:

* the qualification rule is evaluated **from an ODS**, never from a shot list, and
  it refuses what the converter refuses rather than restating its rules;
* promotion goes NEO -> native result -> standardized product -> database, with the
  native result preserved *beside* the product and never on the query path;
* what is stored survives a save and a reload.

Nothing here needs GACODE installed: the run under test is the seven-surface NEO
result committed under `test/data/gacode/neo_vest_48224_profile/`.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from vaft.code.gacode.neo.outputs import collect_neo_outputs
from vaft.validation.neoclassical import REQUIREMENTS, input_readiness

FIXTURES = Path(__file__).parent / "data" / "gacode"
PROFILE_RUN = FIXTURES / "neo_vest_48224_profile"

SAMPLE = None
try:  # pragma: no cover - depends on the repository-only sample being present
    from vaft.data.resources import data_path

    _candidate = Path(data_path("kineticEfit/ods_48224_300ms.json"))
    SAMPLE = _candidate if _candidate.exists() else None
except Exception:
    SAMPLE = None

requires_sample = pytest.mark.skipif(
    SAMPLE is None, reason="the packaged 48224 kinetic sample is a repository-only asset"
)


@pytest.fixture(scope="module")
def state():
    from omas import load_omas_json

    return load_omas_json(str(SAMPLE), consistency_check=False)


@pytest.fixture(scope="module")
def native():
    return collect_neo_outputs(PROFILE_RUN)


# --------------------------------------------------------------------------
# the qualification rule
# --------------------------------------------------------------------------


@requires_sample
def test_the_canonical_kinetic_state_qualifies(state):
    """48224 is the state the whole chain was built on; it must pass its own rule."""
    readiness = input_readiness(state, z_eff=2.0)
    assert readiness["qualified"] is True
    assert readiness["unmet"] == ()
    assert tuple(check["name"] for check in readiness["requirements"]) == REQUIREMENTS


@requires_sample
def test_every_requirement_says_why_it_passed_not_merely_that_it_did(state):
    """A qualified state has to record what it qualified on, or the rule is a boolean."""
    for check in input_readiness(state, z_eff=2.0)["requirements"]:
        assert check["detail"].strip(), check["name"]


@requires_sample
def test_the_assumptions_travel_with_the_verdict(state):
    """An assumed Z_eff is the difference between a measurement and a model input."""
    readiness = input_readiness(state, z_eff=2.0)
    assumed = {entry["quantity"]: entry for entry in readiness["assumptions"]}
    assert assumed["z_eff"]["kind"] == "caller_supplied"
    assert assumed["z_eff"]["value"] == 2.0
    assert assumed["rho_max"]["value"] == 0.95


@requires_sample
def test_a_state_the_converter_refuses_does_not_qualify_and_quotes_the_refusal(state):
    """No truncation means a zero density at the boundary, which GACODE cannot take.

    The rule must report the converter's own words: a paraphrase here would drift
    from what actually refuses the state at run time.
    """
    readiness = input_readiness(state, rho_max=None, z_eff=2.0)
    assert readiness["qualified"] is False
    assert "convertible" in readiness["unmet"]
    detail = readiness["requirements"][0]["detail"]
    assert "not positive" in detail and "logarithmic" in detail


@requires_sample
def test_a_state_without_an_ion_temperature_does_not_qualify(state):
    """NEO needs the ion collisionality; an electron-only fit is not a NEO input."""
    import copy

    electron_only = copy.deepcopy(state)
    del electron_only["core_profiles.profiles_1d.0.ion.0.temperature"]
    readiness = input_readiness(electron_only, z_eff=2.0)
    assert readiness["qualified"] is False
    assert readiness["unmet"], readiness["requirements"]


@requires_sample
def test_the_rule_reads_a_state_not_a_shot_number(state):
    """The signature is the contract: a rule taking a shot would go stale on reprocess."""
    import inspect

    parameters = inspect.signature(input_readiness).parameters
    assert "shot" not in parameters
    assert list(parameters)[0] == "ods"


# --------------------------------------------------------------------------
# the product
# --------------------------------------------------------------------------


@requires_sample
def test_the_product_carries_what_neo_produced_and_the_identity_of_its_input(tmp_path):
    from vaft.omas.vest_upstream import build_neoclassical_ods

    ods, manifest = build_neoclassical_ods(
        shot=48224, state=SAMPLE, run_directory=PROFILE_RUN, z_eff=2.0
    )
    assert manifest["stage"] == "neoclassical"
    assert manifest["status"] == "success"
    assert "j_bootstrap" in manifest["written"]
    assert manifest["input"]["state_sha256"]
    assert manifest["solver"]["code"] == "neo"
    assert manifest["readiness"]["qualified"] is True

    # What NEO produced, and not a copy of the state it was computed from.
    assert "core_profiles.profiles_1d.0.j_bootstrap" in ods
    assert len(ods["core_transport.model"]) == 1
    assert "core_profiles.profiles_1d.0.electrons.temperature" not in ods
    assert "equilibrium.time_slice" not in ods


@requires_sample
def test_an_unqualified_state_is_refused_rather_than_promoted_with_a_caveat(tmp_path):
    """The rule gates the write. A product built from a rejected input is the thing
    #550 defers, and a manifest field saying so would not stop anyone querying it."""
    from vaft.omas.vest_upstream import build_neoclassical_ods

    with pytest.raises(ValueError, match="does not qualify"):
        build_neoclassical_ods(
            shot=48224,
            state=SAMPLE,
            run_directory=PROFILE_RUN,
            rho_max=None,
            z_eff=2.0,
        )


@requires_sample
def test_the_manifest_is_json_and_the_readiness_survives_it(tmp_path):
    """The manifest is written as JSON; a numpy scalar in it would raise at write time."""
    from vaft.omas.vest_upstream import build_neoclassical_ods

    _, manifest = build_neoclassical_ods(
        shot=48224, state=SAMPLE, run_directory=PROFILE_RUN, z_eff=2.0
    )
    reloaded = json.loads(json.dumps(manifest))
    assert reloaded["readiness"]["unmet"] == []
    assert reloaded["readiness"]["assumptions"][0]["quantity"] == "z_eff"


@requires_sample
def test_the_native_result_is_written_beside_the_product_not_inside_it(tmp_path):
    """#527: preserve the solver-native result; #742: never make it the query contract."""
    from vaft.code.gacode.neo.outputs import NeoOutputs
    from vaft.omas.vest_upstream import build_neoclassical_ods, write_neoclassical_product

    ods, manifest = build_neoclassical_ods(
        shot=48224, state=SAMPLE, run_directory=PROFILE_RUN, z_eff=2.0
    )
    output = tmp_path / "48224_neoclassical.json"
    native_path = write_neoclassical_product(
        ods,
        manifest,
        collect_neo_outputs(PROFILE_RUN),
        output=output,
        metadata=tmp_path / "48224_neoclassical.meta.json",
    )

    assert output.is_file() and native_path.is_file()
    assert native_path != output
    # It is NEO's own result, in NEO's own normalisation, and it reloads as one.
    restored = NeoOutputs.read_json(native_path)
    assert restored.solved
    np.testing.assert_allclose(
        np.asarray(restored.bootstrap_current),
        np.asarray(collect_neo_outputs(PROFILE_RUN).bootstrap_current),
    )


@requires_sample
def test_the_product_survives_a_save_and_a_reload(tmp_path):
    """An acceptance criterion in its own right: what is stored must come back."""
    from omas import load_omas_json
    from vaft.omas.vest_upstream import build_neoclassical_ods, write_neoclassical_product

    ods, manifest = build_neoclassical_ods(
        shot=48224, state=SAMPLE, run_directory=PROFILE_RUN, z_eff=2.0
    )
    output = tmp_path / "48224_neoclassical.json"
    write_neoclassical_product(
        ods, manifest, collect_neo_outputs(PROFILE_RUN),
        output=output, metadata=tmp_path / "meta.json",
    )

    reloaded = load_omas_json(str(output), consistency_check=False)
    np.testing.assert_allclose(
        np.asarray(reloaded["core_profiles.profiles_1d.0.j_bootstrap"]),
        np.asarray(ods["core_profiles.profiles_1d.0.j_bootstrap"]),
        equal_nan=True,
    )
    assert len(reloaded["core_transport.model"]) == 1


# --------------------------------------------------------------------------
# the stage, and the summary that reads it
# --------------------------------------------------------------------------


def test_the_stage_is_registered_optional_and_opt_in():
    """Opt-in is the acceptance criterion: no pipeline-1 rule may demand it."""
    from vaft.database.filedb import OMASStage
    from vaft.database.sources import replication_for_stage

    entry = replication_for_stage(OMASStage.NEOCLASSICAL)
    assert entry.ids == ("core_profiles", "core_transport")
    assert entry.optional is True
    assert entry.produced_by == "corrective"
    assert entry.source == "kinetic-efit/neoclassical"


def test_the_source_hangs_beneath_the_lineage_that_can_feed_it():
    from vaft.database.sources import describe

    source = describe("kinetic-efit/neoclassical")
    assert source.parent == "kinetic-efit"


def test_it_does_not_publish_into_the_source_that_owns_the_kinetic_profiles():
    """core_profiles has two owners now; only a source split keeps them apart."""
    from vaft.database.sources import STAGE_REPLICATION

    owners = {
        stage: entry.source
        for stage, entry in STAGE_REPLICATION.items()
        if "core_profiles" in entry.ids
    }
    assert set(owners) == {"core_profiles", "neoclassical"}
    assert len(set(owners.values())) == 2


@requires_sample
def test_the_summary_reads_the_product_and_not_a_native_file(tmp_path):
    from vaft.database._summary import extract_neoclassical
    from vaft.omas.vest_upstream import build_neoclassical_ods

    ods, _ = build_neoclassical_ods(
        shot=48224, state=SAMPLE, run_directory=PROFILE_RUN, z_eff=2.0
    )
    rows = extract_neoclassical(ods, 48224)
    assert len(rows) == 1
    row = rows[0]
    assert row["shot"] == 48224
    assert row["points_solved"] > 1
    assert 0.0 < row["rho_solved_min"] < row["rho_solved_max"] < 1.0
    assert row["has_core_transport"] is True
    assert np.isfinite(row["j_bootstrap_peak_A_m2"])


def test_the_preset_is_registered_with_the_columns_the_extractor_emits():
    from vaft.database import get_summary_preset

    preset = get_summary_preset("neoclassical")
    assert preset.key_columns == ("shot", "cp_index")
    assert "i_bootstrap_kA" in preset.columns
    assert "points_solved" in preset.columns


@requires_sample
def test_a_partial_run_is_reported_as_partial_not_as_a_small_current(tmp_path):
    """NEO solved 7 surfaces of a 129-point grid; the row must say so.

    Without the solved span and count, a partial profile and a full one are two
    numbers in the same column with no way to tell them apart.
    """
    from vaft.database._summary import extract_neoclassical
    from vaft.omas.vest_upstream import build_neoclassical_ods

    ods, _ = build_neoclassical_ods(
        shot=48224, state=SAMPLE, run_directory=PROFILE_RUN, z_eff=2.0
    )
    row = extract_neoclassical(ods, 48224)[0]
    grid = np.asarray(ods["core_profiles.profiles_1d.0.grid.rho_tor_norm"])
    assert row["points_solved"] == int(
        np.count_nonzero(
            np.isfinite(np.asarray(ods["core_profiles.profiles_1d.0.j_bootstrap"]))
        )
    )
    assert row["points_solved"] <= grid.size


@requires_sample
def test_the_integrated_current_needs_the_equilibrium_the_summary_composes(tmp_path):
    """I_bs is an area integral, so it is NaN on the product alone and a number
    once the shot is composed -- which is what `paths` on the preset asks for.

    Not a defect: the product owns what NEO produced and the equilibrium stays in
    the source that owns it. The column would be dishonest computed from an assumed
    geometry, so it is NaN until the real one is present.
    """
    from omas import load_omas_json
    from vaft.database._summary import extract_neoclassical
    from vaft.omas.vest_upstream import build_neoclassical_ods

    product, _ = build_neoclassical_ods(
        shot=48224, state=SAMPLE, run_directory=PROFILE_RUN, z_eff=2.0
    )
    assert np.isnan(extract_neoclassical(product, 48224)[0]["i_bootstrap_kA"])

    composed = load_omas_json(str(SAMPLE), consistency_check=False)
    for leaf in ("j_bootstrap", "grid.rho_tor_norm"):
        composed[f"core_profiles.profiles_1d.0.{leaf}"] = np.asarray(
            product[f"core_profiles.profiles_1d.0.{leaf}"]
        )
    row = extract_neoclassical(composed, 48224)[0]
    # 2.19 kA on this state through the notebook's own interpolated profile; the
    # seven surfaces NEO actually solved give the same current to within a percent.
    assert row["i_bootstrap_kA"] == pytest.approx(2.18, abs=0.1)


@requires_sample
def test_a_lineage_with_no_destination_is_refused_before_anything_is_written():
    """No ODS records which reconstruction it came from, so the caller states it.

    An electron-only fit carries an ion temperature that looks exactly like a
    measured one -- the rule cannot tell them apart, and a product published into
    `kinetic-efit/neoclassical` from an electron state would misdescribe itself.
    """
    from vaft.omas.vest_upstream import build_neoclassical_ods

    with pytest.raises(ValueError, match="no neoclassical destination"):
        build_neoclassical_ods(
            shot=48224,
            state=SAMPLE,
            run_directory=PROFILE_RUN,
            z_eff=2.0,
            lineage="electron",
        )

    _, manifest = build_neoclassical_ods(
        shot=48224, state=SAMPLE, run_directory=PROFILE_RUN, z_eff=2.0
    )
    assert manifest["lineage"] == "kinetic"


@requires_sample
def test_the_rule_does_not_claim_the_ion_temperature_was_measured(state):
    """It cannot know, and a rule that overstates its evidence is worse than none."""
    readiness = input_readiness(state, z_eff=2.0)
    detail = next(
        check["detail"]
        for check in readiness["requirements"]
        if check["name"] == "ion_temperature"
    )
    assert "not the same as measured" in detail
