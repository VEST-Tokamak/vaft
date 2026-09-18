"""The framework-free ML contract: datasets, group splits, artifacts, resolution (#669).

Everything here runs on the closed-form NumPy backend, so the whole lifecycle
is exercised on the gate without an ML framework installed.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys

import numpy as np
import pytest
import yaml

from vaft.process import ml


def _toy(n_groups=20, per_group=40, seed=0):
    rng = np.random.default_rng(seed)
    latent = rng.standard_normal((n_groups * per_group, 2))
    x = np.hstack([latent, latent @ rng.standard_normal((2, 4))])
    x += 0.01 * rng.standard_normal(x.shape)
    groups = np.repeat(np.arange(40000, 40000 + n_groups), per_group)
    time = np.tile(np.arange(per_group) * 1e-5, n_groups)
    spec = ml.DatasetSpec(name="toy", task="reconstruction", group_key="shot")
    return ml.build_dataset(x, groups, spec=spec, sample_meta={"time": time})


def _flip_last_byte(path):
    payload = bytearray(path.read_bytes())
    payload[-1] ^= 0xFF
    path.write_bytes(bytes(payload))


def _trained(dataset=None, seed=1):
    dataset = dataset or _toy()
    split = ml.split_groups(dataset, ml.SplitSpec(seed=seed))
    spec = ml.ModelSpec(
        architecture="pca_autoencoder", task="toy_ae", backend="linear", hyperparameters={"n_components": 2}
    )
    return dataset, split, ml.train_model(dataset, split, spec)


# ---------------------------------------------------------------------------
# Datasets and fingerprints
# ---------------------------------------------------------------------------


def test_fingerprint_is_stable_and_sensitive_to_one_value():
    a, b = _toy(), _toy()
    assert a.fingerprint == b.fingerprint == ml.dataset_fingerprint(a)
    changed = a.inputs.copy()
    changed[7, 3] += 1e-12
    c = ml.build_dataset(changed, a.groups, spec=a.spec, sample_meta=dict(a.sample_meta))
    assert c.fingerprint != a.fingerprint


def test_fingerprint_covers_groups_and_spec():
    a = _toy()
    regrouped = ml.build_dataset(a.inputs, a.groups[::-1], spec=a.spec, sample_meta=dict(a.sample_meta))
    renamed = ml.build_dataset(
        a.inputs, a.groups, spec=ml.DatasetSpec(name="other", task="reconstruction", group_key="shot"),
        sample_meta=dict(a.sample_meta),
    )
    assert len({a.fingerprint, regrouped.fingerprint, renamed.fingerprint}) == 3


def test_build_dataset_refuses_non_finite_and_mismatched_lengths():
    spec = ml.DatasetSpec(name="x", task="t")
    with pytest.raises(ml.ModelContractError, match="non-finite"):
        ml.build_dataset(np.array([[1.0], [np.nan]]), [1, 2], spec=spec)
    with pytest.raises(ml.ModelContractError, match="groups"):
        ml.build_dataset(np.ones((3, 2)), [1, 2], spec=spec)


def test_feature_dataset_blocks_select_their_columns():
    fspec = ml.FeatureSpec(
        feature_names=("ip_mean", "ip_std", "ha_mean"),
        blocks={"stage1": ("ip_mean", "ip_std"), "stage2": ("ha_mean",)},
        version="v4",
    )
    x = np.arange(12.0).reshape(4, 3)
    ds = ml.build_dataset(x, [1, 1, 2, 2], spec=ml.DatasetSpec(name="f", task="t"), feature_spec=fspec)
    assert isinstance(ds, ml.FeatureDataset)
    stage2 = ds.select_block("stage2")
    np.testing.assert_array_equal(stage2.inputs, x[:, [2]])
    assert stage2.feature_spec.feature_names == ("ha_mean",)
    assert stage2.fingerprint == ml.dataset_fingerprint(stage2) != ds.fingerprint
    with pytest.raises(ml.ModelContractError):
        ml.build_dataset(x[:, :2], [1, 1, 2, 2], spec=ml.DatasetSpec(name="f", task="t"), feature_spec=fspec)


# ---------------------------------------------------------------------------
# Group splits before windowing
# ---------------------------------------------------------------------------


def test_split_assigns_every_group_to_exactly_one_partition():
    ds = _toy(n_groups=23)
    split = ml.split_groups(ds, ml.SplitSpec(seed=5))
    seen = [g for p in split.partitions for g in split.groups(p)]
    assert sorted(seen) == sorted(set(ds.group_labels)) and len(seen) == len(set(seen))
    masks = [split.mask(ds, p) for p in split.partitions]
    assert np.array_equal(np.sum(masks, axis=0), np.ones(ds.n_samples))
    counts = [len(split.groups(p)) for p in split.partitions]
    assert counts == [16, 4, 3]  # 16.1, 3.45, 3.45 of 23; the remainder goes to the first short partition


def test_split_depends_on_the_group_set_and_seed_not_sample_order():
    ds = _toy()
    order = np.random.default_rng(3).permutation(ds.n_samples)
    shuffled = ml.build_dataset(ds.inputs[order], ds.groups[order], spec=ds.spec)
    a = ml.split_groups(ds, ml.SplitSpec(seed=9))
    b = ml.split_groups(shuffled, ml.SplitSpec(seed=9))
    c = ml.split_groups(ds, ml.SplitSpec(seed=10))
    assert dict(a.assignment) == dict(b.assignment)
    assert dict(a.assignment) != dict(c.assignment)


def test_split_honours_pinned_groups_and_gives_every_partition_a_group():
    ds = _toy(n_groups=3)
    split = ml.split_groups(ds, ml.SplitSpec(seed=0, fixed={"40002": "test"}))
    assert split.groups("test") == ("40002",)
    assert all(len(split.groups(p)) == 1 for p in split.partitions)
    with pytest.raises(ml.ModelContractError, match="cannot fill"):
        ml.split_groups(_toy(n_groups=2))
    with pytest.raises(ml.ModelContractError, match="not present"):
        ml.split_groups(ds, ml.SplitSpec(fixed={"1": "test"}))


def test_a_group_the_split_never_saw_is_refused_not_guessed():
    ds = _toy(n_groups=10)
    split = ml.split_groups(ds)
    extra = ml.build_dataset(ds.inputs[:5], np.full(5, 99999), spec=ds.spec)
    with pytest.raises(ml.ModelContractError, match="not in the split"):
        split.mask(extra, "train")


def test_windows_never_cross_groups_and_inherit_the_split():
    ds = _toy(n_groups=6, per_group=25)
    split = ml.split_groups(ds, ml.SplitSpec(seed=2))
    win = ml.window_dataset(ds, window=10, step=5)
    starts, stops = win.sample_meta["window_start"], win.sample_meta["window_stop"]
    for s, e, g in zip(starts, stops, win.group_labels):
        assert set(ds.group_labels[s:e]) == {g}
    # 25 samples per group, window 10, step 5 -> starts 0,5,10,15 per group
    assert win.n_samples == 6 * 4 and win.inputs.shape == (24, 10, 6)
    np.testing.assert_array_equal(win.sample_meta["time"], ds.sample_meta["time"][starts])
    for p in split.partitions:
        assert set(win.group_labels[split.mask(win, p)]) == set(split.groups(p))


def test_window_target_modes():
    x = np.arange(20.0).reshape(10, 2)
    ds = ml.build_dataset(x, np.zeros(10), targets=np.arange(10.0), spec=ml.DatasetSpec(name="w", task="t"))
    last = ml.window_dataset(ds, 4, 3, target_mode="last")
    np.testing.assert_array_equal(last.targets[:, 0], [3.0, 6.0, 9.0])
    assert ml.window_dataset(ds, 4, 3, target_mode="window").targets.shape == (3, 4, 1)
    assert ml.window_dataset(ds, 4, 3, target_mode="none").targets is None
    with pytest.raises(ml.ModelContractError):
        ml.window_dataset(ds, 0, 1)


# ---------------------------------------------------------------------------
# Training, evaluation, calibration, inference
# ---------------------------------------------------------------------------


def test_training_records_what_identifies_the_model():
    ds, split, result = _trained()
    record = result.artifact.training
    assert record["dataset_fingerprint"] == ds.fingerprint
    assert record["split"]["groups"]["test"] == list(split.groups("test"))
    assert record["vaft_version"] and "numpy" in record["runtimes"]
    assert record["seed"] == 0 and record["config"]["train_partition"] == "train"


def test_normalisation_uses_the_train_partition_only():
    ds, split, result = _trained()
    train = ds.inputs[split.mask(ds, "train")]
    np.testing.assert_allclose(result.artifact.normalization["input_mean"], train.mean(axis=0))


def test_ridge_recovers_a_linear_map_and_evaluates_per_group():
    rng = np.random.default_rng(4)
    x = rng.standard_normal((400, 3, 2))
    w = rng.standard_normal((6, 4))
    y = (x.reshape(400, -1) @ w).reshape(400, 2, 2)
    ds = ml.build_dataset(x, np.repeat(np.arange(20), 20), targets=y, spec=ml.DatasetSpec(name="r", task="t"))
    split = ml.split_groups(ds)
    model = ml.train_model(ds, split, ml.ModelSpec(architecture="ridge", task="r", backend="linear")).artifact
    evaluation = ml.evaluate_model(model, ds, split, "test")
    assert evaluation.metrics["r2"] > 0.999
    assert set(evaluation.per_group) == set(split.groups("test"))
    assert ml.predict(model, ds).outputs["prediction"].shape == (400, 2, 2)


def test_supervised_architecture_without_targets_is_refused():
    ds = _toy()
    with pytest.raises(ml.ModelContractError, match="needs targets"):
        ml.train_model(ds, ml.split_groups(ds), ml.ModelSpec(architecture="ridge", task="r", backend="linear"))


def test_calibration_flags_the_expected_share_and_is_carried_by_predict():
    ds, split, result = _trained()
    cal = ml.calibrate_threshold(result.artifact, ds, split, quantile=0.9, name="stage1")
    assert cal.partition == "validation" and cal.groups == split.groups("validation")
    model = result.artifact.with_calibration(cal)
    out = ml.predict(model, ds).outputs
    val = split.mask(ds, "validation")
    assert out["stage1_exceeds"][val].mean() == pytest.approx(0.1, abs=0.02)
    with pytest.raises(ml.ModelContractError, match="per-sample scalar"):
        ml.calibrate_threshold(model, ds, split, output="reconstruction")


def test_predict_refuses_a_different_shape_or_feature_schema():
    ds, split, result = _trained()
    with pytest.raises(ml.ModelContractError, match="expects inputs"):
        ml.predict(result.artifact, np.ones((3, 5)))
    fspec = ml.FeatureSpec(feature_names=tuple("abcdef"))
    other = ml.build_dataset(ds.inputs, ds.groups, spec=ds.spec, feature_spec=fspec)
    model = result.artifact.replace(input_names=tuple("uvwxyz"))
    with pytest.raises(ml.ModelContractError, match="columns"):
        ml.predict(model, other)


def test_unknown_backend_and_architecture_are_named():
    ds = _toy()
    split = ml.split_groups(ds)
    with pytest.raises(ml.ModelContractError, match="unknown ML backend"):
        ml.train_model(ds, split, ml.ModelSpec(architecture="x", task="t", backend="jax"))
    with pytest.raises(ml.ModelContractError, match="no architecture"):
        ml.train_model(ds, split, ml.ModelSpec(architecture="transformer", task="t", backend="linear"))


# ---------------------------------------------------------------------------
# Artifacts
# ---------------------------------------------------------------------------


def test_saved_artifact_round_trips_and_carries_its_identity(tmp_path):
    ds, split, result = _trained()
    cal = ml.calibrate_threshold(result.artifact, ds, split, quantile=0.95)
    saved = ml.save_model_artifact(result.artifact.with_calibration(cal), tmp_path / "a", version="0.1.0")
    manifest_bytes = (tmp_path / "a" / "manifest.json").read_bytes()
    assert saved.identity.manifest_sha256 == hashlib.sha256(manifest_bytes).hexdigest()
    loaded = ml.load_model_artifact(tmp_path / "a")
    assert loaded.identity.manifest_sha256 == saved.identity.manifest_sha256
    assert loaded.identity.version == "0.1.0" and loaded.state == saved.state
    a, b = ml.predict(saved, ds), ml.predict(loaded, ds)
    for key in a.outputs:
        np.testing.assert_array_equal(a.outputs[key], b.outputs[key])
    assert b.model_identity.version == "0.1.0"
    assert b.provenance["dataset_fingerprint"] == ds.fingerprint


def test_manifest_is_plain_sorted_json_with_every_file_pinned(tmp_path):
    _, _, result = _trained()
    ml.save_model_artifact(result.artifact, tmp_path / "a", version="1.2.3")
    manifest = json.loads((tmp_path / "a" / "manifest.json").read_text())
    assert manifest["schema_version"] == 0 and manifest["model"] == "toy_ae"
    assert set(manifest["files"]) == {"weights.npz", "normalization.npz"}
    for name, record in manifest["files"].items():
        payload = (tmp_path / "a" / name).read_bytes()
        assert record == {"sha256": hashlib.sha256(payload).hexdigest(), "size": len(payload)}
    for key in ("vaft_version", "vaft_revision", "dataset_fingerprint", "seed", "split"):
        assert key in manifest["training"]


def test_a_tampered_artifact_is_refused(tmp_path):
    _, _, result = _trained()
    ml.save_model_artifact(result.artifact, tmp_path / "a", version="0.1.0")
    weights = tmp_path / "a" / "weights.npz"
    _flip_last_byte(weights)
    with pytest.raises(ml.ModelResolutionError, match="weights.npz"):
        ml.load_model_artifact(tmp_path / "a")


def test_save_refuses_to_overwrite_and_checks_name_and_version(tmp_path):
    _, _, result = _trained()
    ml.save_model_artifact(result.artifact, tmp_path / "a", version="0.1.0")
    with pytest.raises(ml.ModelContractError, match="not empty"):
        ml.save_model_artifact(result.artifact, tmp_path / "a", version="0.1.1")
    with pytest.raises(ml.ModelContractError, match="semantic"):
        ml.save_model_artifact(result.artifact, tmp_path / "b", version="latest")
    with pytest.raises(ml.ModelContractError, match="must match"):
        ml.save_model_artifact(result.artifact.replace(name="Toy-AE"), tmp_path / "c", version="0.1.0")


def test_calibrating_drops_the_saved_identity(tmp_path):
    ds, split, result = _trained()
    saved = ml.save_model_artifact(result.artifact, tmp_path / "a", version="0.1.0")
    recalibrated = saved.with_calibration(ml.calibrate_threshold(saved, ds, split))
    assert recalibrated.identity is None
    assert recalibrated.resolved_identity().resolved_by == "in-memory"


def test_linear_backend_cannot_export(tmp_path):
    _, _, result = _trained()
    with pytest.raises(ml.ModelContractError, match="cannot export"):
        ml.export_model(result.artifact, tmp_path / "x")


# ---------------------------------------------------------------------------
# Registry resolution
# ---------------------------------------------------------------------------


def _publish(tmp_path, versions=("0.1.0", "0.2.0"), stages=None, statuses=None):
    """Lay out a vaft-nn registry and a filled cache the way PUBLISHING.md does."""
    registry, cache = tmp_path / "vaft-nn", tmp_path / "cache"
    entries = []
    for i, version in enumerate(versions):
        _, _, result = _trained(seed=i + 1)
        out = tmp_path / "build" / version
        saved = ml.save_model_artifact(result.artifact, out, version=version)
        target = registry / "models" / "toy_ae" / "versions" / version
        target.mkdir(parents=True)
        shutil.copy(out / "manifest.json", target / "manifest.json")
        shutil.copytree(out, cache / "toy_ae" / version)
        entries.append({
            "version": version,
            "status": (statuses or {}).get(version, "validated"),
            "manifest_sha256": saved.identity.manifest_sha256,
            "release": f"toy_ae-v{version}",
        })
    index = {"schema_version": 0, "model": "toy_ae", "versions": entries,
             "stages": stages if stages is not None else {"production": versions[0]}}
    (registry / "models" / "toy_ae" / "releases.yaml").write_text(yaml.safe_dump(index))
    return registry, cache


def test_exact_version_resolves_and_loads_verified(tmp_path):
    registry, cache = _publish(tmp_path)
    identity = ml.resolve_model("toy_ae", version="0.2.0", registry=registry, cache=cache)
    assert identity.version == "0.2.0" and identity.stage is None and identity.resolved_by == "registry"
    model = ml.load_model("toy_ae", version="0.2.0", registry=registry, cache=cache)
    assert model.identity == identity
    assert ml.predict(model, _toy()).model_identity == identity


def test_a_stage_alias_resolves_to_and_records_an_exact_version(tmp_path):
    registry, cache = _publish(tmp_path)
    env = {"VAFT_NN_HOME": str(registry), "VAFT_NN_CACHE": str(cache)}
    model = ml.load_model("toy_ae", stage="production", env=env)
    result = ml.predict(model, _toy())
    assert result.model_identity.version == "0.1.0"
    assert result.model_identity.stage == "production"
    assert result.model_identity.manifest_sha256


def test_resolution_refuses_a_cache_that_differs_from_the_registry(tmp_path):
    registry, cache = _publish(tmp_path)
    _flip_last_byte(cache / "toy_ae" / "0.1.0" / "weights.npz")
    with pytest.raises(ml.ModelResolutionError, match="gh release download toy_ae-v0.1.0"):
        ml.resolve_model("toy_ae", version="0.1.0", registry=registry, cache=cache)


def test_a_cache_cannot_vouch_for_itself(tmp_path):
    """Rewriting the cached manifest to match tampered weights is still caught."""
    registry, cache = _publish(tmp_path)
    ml.save_model_artifact(_trained(seed=7)[2].artifact, tmp_path / "evil", version="0.1.0")
    shutil.rmtree(cache / "toy_ae" / "0.1.0")
    shutil.copytree(tmp_path / "evil", cache / "toy_ae" / "0.1.0")
    with pytest.raises(ml.ModelResolutionError, match="differs from the registry"):
        ml.resolve_model("toy_ae", version="0.1.0", registry=registry, cache=cache)


def test_resolution_refuses_an_edited_registry_manifest(tmp_path):
    registry, cache = _publish(tmp_path)
    manifest = registry / "models" / "toy_ae" / "versions" / "0.1.0" / "manifest.json"
    manifest.write_text(manifest.read_text().replace('"validation_loss_best"', '"validation_loss_best "'))
    with pytest.raises(ml.ModelResolutionError, match="pins"):
        ml.resolve_model("toy_ae", version="0.1.0", registry=registry, cache=cache)


def test_resolution_errors_name_what_exists(tmp_path):
    registry, cache = _publish(tmp_path, stages={"production": "0.2.0"}, statuses={"0.2.0": "deprecated"})
    kw = dict(registry=registry, cache=cache)
    with pytest.raises(ml.ModelResolutionError, match="known: \\['toy_ae'\\]"):
        ml.resolve_model("ire_detector", version="0.1.0", **kw)
    with pytest.raises(ml.ModelResolutionError, match="versions: \\['0.1.0', '0.2.0'\\]"):
        ml.resolve_model("toy_ae", version="9.9.9", **kw)
    with pytest.raises(ml.ModelResolutionError, match="no stage 'staging'"):
        ml.resolve_model("toy_ae", stage="staging", **kw)
    with pytest.raises(ml.ModelResolutionError, match="deprecated"):
        ml.resolve_model("toy_ae", stage="production", **kw)
    # an explicit deprecated version stays reproducible
    assert ml.resolve_model("toy_ae", version="0.2.0", **kw).status == "deprecated"
    with pytest.raises(ml.ModelContractError, match="exactly one"):
        ml.resolve_model("toy_ae", **kw)
    with pytest.raises(ml.ModelResolutionError, match="VAFT_NN_HOME"):
        ml.resolve_model("toy_ae", version="0.1.0", env={})


# ---------------------------------------------------------------------------
# Import weight
# ---------------------------------------------------------------------------


def test_importing_the_ml_package_loads_no_ml_framework():
    code = (
        "import sys, vaft.process.ml, vaft.process as p; p.ml; "
        "print(','.join(m for m in ('torch', 'onnx', 'onnxruntime', 'sklearn', 'yaml') if m in sys.modules))"
    )
    loaded = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout.strip()
    assert loaded == ""
