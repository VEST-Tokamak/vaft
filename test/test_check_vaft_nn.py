"""install/check_vaft_nn.py: the vaft-nn registry, reported layer by layer (#669)."""

from __future__ import annotations

import importlib.util
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import yaml

from vaft.process import ml

INSTALL = Path(__file__).resolve().parents[1] / "install"


def _checker():
    spec = importlib.util.spec_from_file_location("check_vaft_nn", INSTALL / "check_vaft_nn.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _registry(tmp_path, cached=True):
    registry, cache = tmp_path / "vaft-nn", tmp_path / "cache"
    for sub in ("models", "schemas", "scripts"):
        (registry / sub).mkdir(parents=True)
    (registry / "scripts" / "check_registry.py").write_text("")
    x = np.random.default_rng(0).standard_normal((100, 4))
    ds = ml.build_dataset(x, np.repeat(np.arange(10), 10), spec=ml.DatasetSpec(name="d", task="t"))
    model = ml.train_model(ds, ml.split_groups(ds),
                           ml.ModelSpec(architecture="pca_autoencoder", task="toy_ae", backend="numpy",
                                        hyperparameters={"n_components": 2})).artifact
    saved = ml.save_model_artifact(model, tmp_path / "build", version="0.1.0")
    target = registry / "models" / "toy_ae" / "versions" / "0.1.0"
    target.mkdir(parents=True)
    shutil.copy(tmp_path / "build" / "manifest.json", target / "manifest.json")
    (registry / "models" / "toy_ae" / "releases.yaml").write_text(yaml.safe_dump({
        "schema_version": 0, "model": "toy_ae",
        "versions": [{"version": "0.1.0", "status": "candidate", "manifest_sha256": saved.identity.manifest_sha256}],
    }))
    if cached:
        shutil.copytree(tmp_path / "build", cache / "toy_ae" / "0.1.0")
    return registry, cache


def _by_name(results):
    return {r.name: r for r in results}


def test_every_layer_passes_on_a_complete_registry(tmp_path):
    checker = _checker()
    registry, cache = _registry(tmp_path)
    results = _by_name(checker.run_checks(registry=str(registry), cache=str(cache), skip_network=True))
    assert results["vaft-nn registry"].status == checker.PASS
    assert results["vaft-nn models"].status == checker.PASS
    assert "toy_ae" in results["vaft-nn models"].detail
    assert results["vaft-nn cache"].status == checker.PASS


def test_an_unfetched_version_warns_with_the_fetch_call(tmp_path):
    checker = _checker()
    registry, cache = _registry(tmp_path, cached=False)
    result = _by_name(checker.run_checks(registry=str(registry), cache=str(cache), skip_network=True))["vaft-nn cache"]
    assert result.status == checker.WARN and "fetch_model" in result.remediation


def test_a_broken_pin_fails_the_models_layer(tmp_path):
    checker = _checker()
    registry, cache = _registry(tmp_path)
    manifest = registry / "models" / "toy_ae" / "versions" / "0.1.0" / "manifest.json"
    manifest.write_text(manifest.read_text() + "\n")
    result = _by_name(checker.run_checks(registry=str(registry), cache=str(cache), skip_network=True))["vaft-nn models"]
    assert result.status == checker.FAIL and "pins" in result.detail


def test_no_registry_fails_with_the_clone_command(tmp_path, monkeypatch):
    checker = _checker()
    monkeypatch.delenv("VAFT_NN_HOME", raising=False)
    result = checker.run_checks(skip_network=True)[0]
    assert result.status == checker.FAIL and "git clone" in result.remediation


def test_the_command_line_runs():
    completed = subprocess.run([sys.executable, str(INSTALL / "check_vaft_nn.py"), "--help"],
                               capture_output=True, text=True, timeout=60, check=True)
    assert "--registry" in completed.stdout and "--skip-network" in completed.stdout
