"""The TGLF-NN surrogate backend (issue #553, increment 2).

Three things are under test, and only the first needs a network to exist at all:

* **resolution** -- VAFT vendors no weights, so it must find artifacts the user already
  has, refuse ambiguity rather than choosing a version silently, and say where it
  looked when it finds nothing;
* **the domain audit** -- which runs without ``onnxruntime`` and without a prediction,
  because "does this model apply to this plasma" is the question that comes first;
* **the arithmetic** -- exercised against `data/gacode/tglf_surrogate/synthetic_linear`,
  a ~500-byte two-member ensemble whose answer is hand-computable (see its README).

The upstream models are not fixtures here. Where a test needs them it says so and skips
when they are absent.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from vaft.code.gacode.tglf import TGLFConfig, TGLFInput
from vaft.code.gacode.tglf.surrogate import (
    METADATA_FILES,
    DomainAudit,
    ModelContractError,
    ModelResolutionError,
    SurrogatePrediction,
    TrainingDomainError,
    audit_training_domain,
    available_models,
    build_features,
    ensemble_predict,
    load_metadata,
    parse_model_name,
    resolve_model,
    run_surrogate,
)

SYNTHETIC = Path(__file__).parent / "data" / "gacode" / "tglf_surrogate" / "synthetic_linear"


def isolated(tmp_path, **overrides) -> dict:
    """An environment with no model roots, so a test sees only what it creates."""
    environment = {"JULIA_DEPOT_PATH": str(tmp_path / "no-julia-depot")}
    environment.update(overrides)
    return environment


def local_input(**overrides) -> TGLFInput:
    """A minimal two-species local input; the fields tests vary are keywords."""
    values = dict(
        rho=0.5,
        rmin_loc=1.0, rmaj_loc=3.0, drmajdx_loc=0.0, zmaj_loc=0.0, dzmajdx_loc=0.0,
        q_loc=3.0, q_prime_loc=16.0, p_prime_loc=0.0,
        kappa_loc=1.0, s_kappa_loc=0.0, delta_loc=0.0, s_delta_loc=0.0,
        zeta_loc=0.0, s_zeta_loc=0.0,
        zs=np.array([-1.0, 1.0]), mass=np.array([2.7e-4, 1.0]),
        as_=np.array([1.0, 1.0]), taus=np.array([1.0, 1.0]),
        rlns=np.array([1.0, 1.0]), rlts=np.array([3.0, 3.0]),
        betae=10.0, xnue=1.0, zeff=1.0, debye=1.0, sign_bt=1.0, sign_it=1.0,
    )
    values.update(overrides)
    return TGLFInput(**values)


def write_model(
    directory: Path,
    *,
    xnames=("Q_LOC", "RMIN_LOC"),
    ynames=("OUT_Q_elec",),
    xm=(0.0, 0.0),
    xsigma=(1.0, 1.0),
    ym=(0.0,),
    ysigma=(1.0,),
    members=("a.onnx", "b.onnx"),
    payload=b"not a real graph",
) -> Path:
    """A model directory shaped like upstream's, for the paths that never run a graph."""
    directory.mkdir(parents=True, exist_ok=True)
    for name, values in (
        ("xnames.txt", xnames), ("ynames.txt", ynames),
        ("xm.txt", xm), ("xsigma.txt", xsigma), ("ym.txt", ym), ("ysigma.txt", ysigma),
    ):
        (directory / name).write_text("\n".join(str(value) for value in values) + "\n")
    for member in members:
        (directory / member).write_bytes(payload)
    return directory


# --------------------------------------------------------------------------
# what the upstream naming convention states
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name, sat_rule, electromagnetic, devices, tags",
    [
        ("sat3_em_d3d+mastu+nstx_azf-1", 3, True, ("d3d", "mastu", "nstx"), ("azf-1",)),
        ("sat1_es_ukstep_azf-1", 1, False, ("ukstep",), ("azf-1",)),
        ("sat0quench_em_mastu_azf+1", 0, True, ("mastu",), ("quench", "azf+1")),
        ("sat1geo_es_nstx_azf-1", 1, False, ("nstx",), ("geo", "azf-1")),
        ("sat1_em_fpp_d3d", 1, True, ("fpp", "d3d"), ()),
        ("sat3_em_d3d_azf-1_withnegD", 3, True, ("d3d",), ("azf-1", "withnegD")),
    ],
)
def test_the_model_name_is_read_for_the_physics_it_states(
    name, sat_rule, electromagnetic, devices, tags
):
    parsed = parse_model_name(name)
    assert parsed["sat_rule"] == sat_rule
    assert parsed["electromagnetic"] is electromagnetic
    assert parsed["devices"] == devices
    assert parsed["tags"] == tags


def test_a_name_that_states_no_physics_claims_none():
    """A parsed filename must not become a physics claim it never made."""
    parsed = parse_model_name("synthetic_linear")
    assert parsed["sat_rule"] is None
    assert parsed["electromagnetic"] is None
    assert parsed["devices"] == ()


def test_identity_records_that_the_physics_came_from_the_name(tmp_path):
    identity = resolve_model(write_model(tmp_path / "sat3_em_mastu_azf-1"))
    assert identity.sat_rule == 3
    assert identity.devices == ("mastu",)
    assert "name" in identity.physics_source


# --------------------------------------------------------------------------
# resolution
# --------------------------------------------------------------------------


def test_an_explicit_directory_resolves_and_is_hashed(tmp_path):
    """The digest covers the normalisation too: the moments are part of the model."""
    directory = write_model(tmp_path / "family", members=("m1.onnx", "m2.onnx"))
    identity = resolve_model(directory)
    assert identity.resolved_by == "explicit path"
    assert identity.members == ("m1.onnx", "m2.onnx")
    assert set(identity.sha256) == {"m1.onnx", "m2.onnx", *METADATA_FILES}
    assert len(identity.sha256["m1.onnx"]) == 64


def test_copies_differing_only_in_their_normalisation_are_refused(tmp_path):
    """Same weights read through different moments is a different model.

    Upstream's `models/SEMVER` calls a patch bump "training data / space ... updated",
    which is exactly a change to these files and to nothing else.
    """
    write_model(tmp_path / "a" / "models" / "family", payload=b"same weights")
    write_model(
        tmp_path / "b" / "models" / "family", payload=b"same weights", xm=(0.0, 1.0),
    )
    with pytest.raises(ModelResolutionError, match="not the same bytes"):
        resolve_model(
            "family",
            model_dir=tmp_path / "a" / "models",
            env=isolated(tmp_path, TURBULENTTRANSPORTHOME=str(tmp_path / "b")),
        )


def test_the_upstream_version_is_recorded_on_the_route_the_readme_recommends(tmp_path):
    checkout = tmp_path / "checkout"
    write_model(checkout / "models" / "family")
    (checkout / "Project.toml").write_text('name = "TurbulentTransport"\nversion = "9.9.9"\n')
    identity = resolve_model(
        "family", env=isolated(tmp_path, TURBULENTTRANSPORTHOME=str(checkout))
    )
    assert identity.resolved_by == "TURBULENTTRANSPORTHOME"
    assert identity.upstream_version == "9.9.9"


def test_a_configured_model_directory_resolves_by_name(tmp_path):
    write_model(tmp_path / "models" / "family")
    identity = resolve_model(
        "family", model_dir=tmp_path / "models", env=isolated(tmp_path)
    )
    assert identity.resolved_by == "model directory"


@pytest.mark.parametrize("variable", ["TURBULENTTRANSPORTHOME", "TURBULENTTRANSPORT_ROOT"])
def test_the_environment_root_resolves_by_name(tmp_path, variable):
    """VAFT's own ``<CODE>HOME`` spelling, with the issue's ``_ROOT`` form accepted."""
    write_model(tmp_path / "checkout" / "models" / "family")
    identity = resolve_model(
        "family", env=isolated(tmp_path, **{variable: str(tmp_path / "checkout")})
    )
    assert identity.resolved_by == variable


def test_a_missing_model_names_every_place_that_was_searched(tmp_path):
    write_model(tmp_path / "models" / "other")
    with pytest.raises(ModelResolutionError) as error:
        resolve_model("absent", model_dir=tmp_path / "models", env=isolated(tmp_path))
    message = str(error.value)
    assert str(tmp_path / "models") in message
    assert "TURBULENTTRANSPORTHOME" in message
    assert "other" in message, "the resolvable names are the actionable part"


def test_a_directory_without_the_normalisation_files_is_refused(tmp_path):
    directory = tmp_path / "family"
    directory.mkdir()
    (directory / "m.onnx").write_bytes(b"x")
    with pytest.raises(ModelResolutionError, match="xnames.txt"):
        resolve_model(directory)


def test_a_bson_only_family_says_so_rather_than_resolving_empty(tmp_path):
    """Upstream ships most families as Julia .bson; those have no ONNX to run."""
    directory = write_model(tmp_path / "family", members=())
    with pytest.raises(ModelResolutionError, match="bson"):
        resolve_model(directory)


def test_identical_copies_in_two_roots_resolve_and_record_the_other(tmp_path):
    for root in ("a", "b"):
        write_model(tmp_path / root / "models" / "family", payload=b"same bytes")
    identity = resolve_model(
        "family",
        model_dir=tmp_path / "a" / "models",
        env=isolated(tmp_path, TURBULENTTRANSPORTHOME=str(tmp_path / "b")),
    )
    assert identity.directory == str(tmp_path / "a" / "models" / "family")
    assert identity.alternatives == (str(tmp_path / "b" / "models" / "family"),)


def test_two_different_models_sharing_a_name_are_refused_not_chosen_between(tmp_path):
    """A Julia depot carries several installed versions; upstream's SEMVER says a
    version bump can change the training set behind an unchanged family name."""
    write_model(tmp_path / "a" / "models" / "family", payload=b"version one")
    write_model(tmp_path / "b" / "models" / "family", payload=b"version two")
    with pytest.raises(ModelResolutionError) as error:
        resolve_model(
            "family",
            model_dir=tmp_path / "a" / "models",
            env=isolated(tmp_path, TURBULENTTRANSPORTHOME=str(tmp_path / "b")),
        )
    message = str(error.value)
    assert str(tmp_path / "a" / "models" / "family") in message
    assert str(tmp_path / "b" / "models" / "family") in message


def test_a_bare_name_stays_a_name_beside_a_folder_that_shares_it(tmp_path, monkeypatch):
    """A stray directory in the working tree must not shadow a family name."""
    (tmp_path / "family").mkdir()  # not a model: no normalisation files
    write_model(tmp_path / "models" / "family")
    monkeypatch.chdir(tmp_path)
    identity = resolve_model(
        "family", model_dir=tmp_path / "models", env=isolated(tmp_path)
    )
    assert identity.resolved_by == "model directory"


def test_available_models_answers_what_a_name_would_resolve_to(tmp_path):
    write_model(tmp_path / "a" / "models" / "family", payload=b"first")
    write_model(tmp_path / "b" / "models" / "family", payload=b"second")
    write_model(tmp_path / "b" / "models" / "only-in-b")
    found = available_models(
        model_dir=tmp_path / "a" / "models",
        env=isolated(tmp_path, TURBULENTTRANSPORTHOME=str(tmp_path / "b")),
    )
    assert found["family"] == tmp_path / "a" / "models" / "family"
    assert found["only-in-b"] == tmp_path / "b" / "models" / "only-in-b"


@pytest.mark.parametrize("environment", [{}, {"TURBULENTTRANSPORTHOME": "/nonexistent"}])
def test_a_passed_environment_isolates_the_julia_depot_too(environment):
    """``env=`` that the depot search reached past would be a half-truth.

    Neither of these names ``JULIA_DEPOT_PATH`` or ``HOME``, and the machine running
    this very likely has a real depot holding that family.
    """
    with pytest.raises(ModelResolutionError):
        resolve_model("sat3_em_d3d+mastu_azf-1", env=environment)


# --------------------------------------------------------------------------
# the feature vector
# --------------------------------------------------------------------------


def test_features_follow_the_models_own_order_and_log_convention(tmp_path):
    metadata = load_metadata(
        resolve_model(write_model(
            tmp_path / "family",
            xnames=("BETAE_log10", "Q_LOC"), xm=(0.0, 0.0), xsigma=(1.0, 1.0),
        ))
    )
    values, _ = build_features(local_input(betae=100.0, q_loc=7.0), metadata)
    assert values == pytest.approx([2.0, 7.0])


def test_a_channel_no_tglf_key_supplies_is_refused_not_filled(tmp_path):
    """A NaN feature normalises to NaN and the network answers anyway."""
    metadata = load_metadata(resolve_model(write_model(
        tmp_path / "family", xnames=("Q_LOC", "NO_SUCH_KEY"),
    )))
    with pytest.raises(ModelContractError, match="NO_SUCH_KEY"):
        build_features(local_input(), metadata)


def test_a_non_finite_plain_channel_is_not_blamed_on_the_log_transform(tmp_path):
    """ZEFF is never logged; a message about logarithms sends the reader elsewhere."""
    metadata = load_metadata(resolve_model(write_model(
        tmp_path / "family", xnames=("ZEFF", "Q_LOC"),
    )))
    with pytest.raises(ModelContractError) as error:
        build_features(local_input(zeff=float("nan")), metadata)
    assert "ZEFF (ZEFF=nan)" in str(error.value)
    assert "log10 of ZEFF" not in str(error.value)


def test_a_density_fraction_record_is_matched_despite_its_field_spelling(tmp_path):
    """The array is `as_`, not `as`; a candidate list that missed it would say nothing."""
    metadata = load_metadata(resolve_model(write_model(
        tmp_path / "family", xnames=("AS_2", "Q_LOC"),
    )))
    absent = {"kind": "unavailable", "reason": "no ion density fractions on the profile"}
    _, assumed = build_features(local_input(provenance={"as_": absent}), metadata)
    assert assumed == ("AS_2",)


def test_a_non_positive_value_under_the_log_transform_is_refused(tmp_path):
    metadata = load_metadata(resolve_model(write_model(
        tmp_path / "family", xnames=("XNUE_log10", "Q_LOC"),
    )))
    with pytest.raises(ModelContractError, match="XNUE_log10"):
        build_features(local_input(xnue=0.0), metadata)


def test_a_feature_resting_on_an_unavailable_quantity_is_reported(tmp_path):
    """The native run writes the same default; the surrogate must not hide it."""
    metadata = load_metadata(resolve_model(write_model(
        tmp_path / "family", xnames=("VEXB_SHEAR", "Q_LOC"),
    )))
    absent = {"kind": "unavailable", "reason": "no w0 on the profile"}
    _, assumed = build_features(
        local_input(provenance={"vexb_shear": absent}), metadata
    )
    assert assumed == ("VEXB_SHEAR",)


def test_a_two_stage_correction_network_is_refused_with_its_reason(tmp_path):
    """The ``*_gknn*`` families take a base model's fluxes as inputs."""
    directory = write_model(
        tmp_path / "family_gknn31",
        xnames=("Q_LOC", "OUT_Q_elec"), xm=(0.0, 0.0), xsigma=(1.0, 1.0),
    )
    with pytest.raises(ModelContractError, match="two-stage"):
        load_metadata(resolve_model(directory))


def test_a_normalisation_that_does_not_match_its_names_is_refused(tmp_path):
    with pytest.raises(ModelContractError, match="inconsistent"):
        load_metadata(resolve_model(write_model(
            tmp_path / "family", xnames=("Q_LOC", "RMIN_LOC"), xm=(0.0,), xsigma=(1.0,),
        )))


def test_a_zero_sigma_channel_is_refused_because_no_z_score_exists(tmp_path):
    with pytest.raises(ModelContractError, match="sigma is zero"):
        load_metadata(resolve_model(write_model(
            tmp_path / "family", xnames=("Q_LOC", "RMIN_LOC"), xsigma=(1.0, 0.0),
        )))


# --------------------------------------------------------------------------
# the domain audit -- no onnxruntime, no prediction
# --------------------------------------------------------------------------


def audited(tmp_path, *, threshold=3.0, **input_overrides) -> DomainAudit:
    identity = resolve_model(write_model(
        tmp_path / "family",
        xnames=("Q_LOC", "RMIN_LOC"), xm=(2.0, 0.5), xsigma=(0.5, 0.25),
    ))
    return audit_training_domain(
        local_input(**input_overrides), identity, load_metadata(identity),
        threshold=threshold,
    )


def test_the_audit_is_the_standard_deviation_distance_from_the_training_mean(tmp_path):
    audit = audited(tmp_path, q_loc=3.0, rmin_loc=1.0)
    assert audit.z_scores["Q_LOC"] == pytest.approx(2.0)
    assert audit.z_scores["RMIN_LOC"] == pytest.approx(2.0)
    assert audit.max_abs_z == pytest.approx(2.0)
    assert audit.in_domain is True


def test_violations_are_listed_worst_first(tmp_path):
    audit = audited(tmp_path, q_loc=4.0, rmin_loc=2.0)  # z = 4 and 6
    assert audit.violations == ("RMIN_LOC", "Q_LOC")
    assert audit.in_domain is False


def test_the_threshold_is_what_decides_and_it_is_honoured(tmp_path):
    assert audited(tmp_path, q_loc=4.0, threshold=3.0).in_domain is False
    assert audited(tmp_path, q_loc=4.0, threshold=5.0).in_domain is True


def test_a_non_positive_threshold_is_a_programming_error(tmp_path):
    with pytest.raises(ValueError):
        audited(tmp_path, threshold=0.0)


def test_the_audit_never_claims_to_be_a_containment_test(tmp_path):
    """True min/max bounds ship only inside the Julia .bson, not beside the ONNX."""
    audit = audited(tmp_path)
    assert audit.bounds_available is False
    assert "sigma" in audit.measure or "deviation" in audit.measure


def test_the_summary_names_the_offending_inputs(tmp_path):
    summary = audited(tmp_path, q_loc=4.0, rmin_loc=2.0).summary()
    assert "RMIN_LOC" in summary and "Q_LOC" in summary


def test_importing_the_surrogate_does_not_import_onnxruntime():
    """Issue #553 section 11: the ML runtime stays optional and unimported."""
    import subprocess
    import sys

    completed = subprocess.run(
        [
            sys.executable, "-c",
            "import sys; import vaft.code.gacode.tglf.surrogate as s;"
            " print('onnxruntime' in sys.modules)",
        ],
        capture_output=True, text=True, check=True,
    )
    assert completed.stdout.strip() == "False"


# --------------------------------------------------------------------------
# the arithmetic, against a hand-computable ensemble
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def synthetic():
    pytest.importorskip("onnxruntime")
    identity = resolve_model(SYNTHETIC)
    return identity, load_metadata(identity)


def test_the_ensemble_reproduces_the_fixtures_hand_computed_answer(synthetic):
    identity, metadata = synthetic
    features = np.array([3.0, 1.0, 1.0])  # z = [2, 2, 1]; see the fixture README
    mean, spread = ensemble_predict(features, identity, metadata)
    assert mean["OUT_Q_elec"] == pytest.approx(4.5, abs=1e-5)
    assert mean["OUT_Q_ions"] == pytest.approx(1.125, abs=1e-5)
    assert spread["OUT_Q_elec"] == pytest.approx(1.5, abs=1e-5)
    assert spread["OUT_Q_ions"] == pytest.approx(0.375, abs=1e-5)


def test_a_feature_vector_of_the_wrong_length_is_refused(synthetic):
    identity, metadata = synthetic
    with pytest.raises(ModelContractError, match="3 inputs"):
        ensemble_predict(np.array([1.0, 2.0]), identity, metadata)


def test_a_run_outside_the_training_distribution_is_refused_by_default(synthetic):
    identity, _ = synthetic
    with pytest.raises(TrainingDomainError) as error:
        run_surrogate(local_input(q_loc=99.0), SYNTHETIC)
    assert error.value.audit.violations[0] == "Q_LOC"
    assert "allow_extrapolation" in str(error.value)


def test_an_extrapolated_prediction_is_returned_marked_not_qualified(synthetic):
    prediction = run_surrogate(local_input(q_loc=99.0), SYNTHETIC, allow_extrapolation=True)
    assert isinstance(prediction, SurrogatePrediction)
    assert prediction.qualified is False
    assert prediction.provenance["extrapolation_allowed"] is True
    assert prediction.provenance["training_bounds_available"] is False
    assert np.isfinite(prediction.outputs["OUT_Q_elec"]), (
        "the point is that the network answers confidently; qualified is what says not to"
    )


def test_an_in_domain_prediction_is_qualified_and_matches_the_ensemble(synthetic):
    identity, metadata = synthetic
    prediction = run_surrogate(local_input(q_loc=3.0, rmin_loc=1.0, betae=10.0), SYNTHETIC)
    assert prediction.qualified is True
    assert prediction.outputs["OUT_Q_elec"] == pytest.approx(4.5, abs=1e-5)
    assert prediction.model.ensemble_size == 2
    assert prediction.provenance["sha256"] == dict(identity.sha256)


def test_the_saturation_rule_the_run_asked_for_is_recorded_against_the_models_own(
    tmp_path, synthetic
):
    """Fidelity, not domain: SAT_RULE is no input channel, so the audit cannot see it.

    A SAT3-trained network standing in for a SAT1 run is a real mismatch and belongs in
    the provenance (issue #553 sections 12-13) -- but not in ``qualified``, which is
    about whether the *input* was one the model was trained for.
    """
    import shutil

    target = tmp_path / "sat2_em_d3d_azf-1"
    shutil.copytree(SYNTHETIC, target)

    matched = run_surrogate(local_input(), target, config=TGLFConfig(sat_rule=2))
    assert matched.provenance["requested_sat_rule"] == 2
    assert matched.provenance["sat_rule_matches_model"] is True

    mismatched = run_surrogate(local_input(), target, config=TGLFConfig(sat_rule=1))
    assert mismatched.provenance["sat_rule_matches_model"] is False
    assert mismatched.qualified is True


def test_a_model_whose_name_states_no_rule_claims_no_match_either_way(synthetic):
    prediction = run_surrogate(local_input(), SYNTHETIC, config=TGLFConfig(sat_rule=1))
    assert prediction.provenance["sat_rule_matches_model"] is None


def test_relative_spread_does_not_report_zero_where_the_ratio_is_undefined():
    prediction = SurrogatePrediction(
        outputs={"a": 0.0, "b": 2.0},
        uncertainty={"a": 0.5, "b": 1.0},
        model=resolve_model(SYNTHETIC),
        domain=DomainAudit(model="x", z_scores={}, threshold=3.0, violations=(), max_abs_z=0.0),
        features={},
    )
    spread = prediction.relative_spread()
    assert spread["a"] == float("inf")
    assert spread["b"] == pytest.approx(0.5)


# --------------------------------------------------------------------------
# against the real upstream models, when the machine has them
#
# These are the tests that answer issue #553 section 16 -- whether a public model
# qualifies for VEST -- and they are the reason the audit exists. They need artifacts
# VAFT does not vendor, so they skip rather than fail when those are absent.
# --------------------------------------------------------------------------

UPSTREAM = available_models()
D3D_FAMILY = "sat3_em_d3d+mastu+nstx_azf-1"

requires_upstream = pytest.mark.skipif(
    D3D_FAMILY not in UPSTREAM,
    reason="needs a TurbulentTransport.jl checkout; VAFT vendors no model weights",
)

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
def vest_local():
    from omas import load_omas_json

    from vaft.code.gacode.inputs import prepare_gacode_profile
    from vaft.code.gacode.tglf import prepare_tglf_input

    ods = load_omas_json(str(SAMPLE), consistency_check=False)
    profile = prepare_gacode_profile(ods, rho_max=0.95, z_eff=2.0, impurity="C")
    return prepare_tglf_input(profile, 0.5)


@requires_upstream
def test_upstreams_own_sample_is_in_domain_for_the_model_trained_on_it():
    """The oracle for the feature convention.

    ``test/data/sample_input.tglf`` is upstream's DIII-D case, and this family was
    trained on DIII-D among others. If the log transform, the key naming or the
    channel order were wrong, that sample would not land inside its own training
    distribution -- ``BETAE_log10`` alone would be tens of sigma out.
    """
    from vaft.code.gacode.tglf.surrogate.resolver import load_metadata as _load

    directory = UPSTREAM[D3D_FAMILY]
    sample = directory.parent.parent / "test" / "data" / "sample_input.tglf"
    if not sample.is_file():
        pytest.skip("this checkout carries no sample_input.tglf")

    keys = {}
    for line in sample.read_text(encoding="utf-8").splitlines():
        line = line.split("#")[0].strip()
        if "=" in line:
            name, _, raw = line.partition("=")
            try:
                keys[name.strip()] = float(raw)
            except ValueError:
                continue

    identity = resolve_model(directory)
    metadata = _load(identity)
    values = np.array([
        np.log10(keys[name[: -len("_log10")]]) if name.endswith("_log10") else keys[name]
        for name in metadata.xnames
    ])
    z = (values - metadata.xm) / metadata.xsigma
    assert np.max(np.abs(z)) < 3.0, (
        f"upstream's own sample is out of its own training distribution on "
        f"{[metadata.xnames[i] for i in np.where(np.abs(z) >= 3.0)[0]]}; the feature "
        f"convention is wrong"
    )


@requires_upstream
@requires_sample
def test_no_public_model_has_vest_in_domain_and_the_temperature_ratio_is_why(vest_local):
    """Issue #553 section 16: a number is not qualification.

    Every public TGLF-NN family was trained where the ions are about as hot as the
    electrons. VEST's ohmic plasma runs an order of magnitude colder in ``T_i/T_e``,
    which is what ``TAUS`` measures, so no amount of agreement in the other thirty
    inputs makes any of these models applicable.
    """
    from vaft.code.gacode.tglf.surrogate.resolver import load_metadata as _load

    audited_families = 0
    for name, directory in UPSTREAM.items():
        identity = resolve_model(directory)
        try:
            metadata = _load(identity)
            audit = audit_training_domain(vest_local, identity, metadata)
        except ModelContractError:
            continue  # two-stage families and differing input conventions
        audited_families += 1
        assert not audit.in_domain, f"{name} unexpectedly reports VEST in domain"
        assert any(entry.startswith("TAUS") for entry in audit.violations), (
            f"{name} is out of domain on {audit.violations}, but not on the "
            f"temperature ratio; the physics reason has changed and this test's "
            f"claim needs rechecking"
        )
    assert audited_families >= 3, "too few families audited for the claim to mean much"


@requires_upstream
@requires_sample
def test_running_a_public_model_on_vest_refuses_before_it_predicts(vest_local):
    with pytest.raises(TrainingDomainError) as error:
        run_surrogate(vest_local, D3D_FAMILY)
    assert any(name.startswith("TAUS") for name in error.value.audit.violations)


@requires_upstream
@requires_sample
def test_the_ensemble_disagrees_with_itself_where_it_was_never_trained(vest_local):
    """The spread is not calibrated, but it is not blind either."""
    prediction = run_surrogate(vest_local, D3D_FAMILY, allow_extrapolation=True)
    spread = prediction.relative_spread()
    assert max(spread.values()) > 0.5, (
        f"an out-of-domain prediction whose members agree to within "
        f"{max(spread.values()):.0%} would leave nothing at all to warn on"
    )
