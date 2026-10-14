"""Graph evidence, optional tools, and immutable model versus run identity."""
import json
import shutil
import sys

import numpy as np
import pytest

from vaft.process import ml
from vaft.process.ml.experiments import experiment_record, save_experiment, load_experiment
from vaft.diagram.ml import GraphNode, GraphEdge, MLGraph, extract_torch_graph, scientific_graph_diagram
from vaft.plot.ml import learning_curves, group_metrics, resource_comparison


def test_computational_graph_preserves_state_and_rng():
    torch = pytest.importorskip("torch")
    net = torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.Dropout(.1), torch.nn.Linear(4, 2))
    flags = [m.training for m in net.modules()]
    x = torch.ones(2, 3)
    rng = torch.get_rng_state().clone()
    result = extract_torch_graph(net, (x,))
    assert result.supported, result.diagnostics
    assert result.graph.to_dot() == extract_torch_graph(net, (x,)).graph.to_dot()
    assert result.graph.parameter_count == sum(p.numel() for p in net.parameters())
    assert sum(n.parameter_count for n in result.graph.nodes) == result.graph.parameter_count
    assert any(t.shape == (2, 4) for n in result.graph.nodes for t in n.tensors)
    assert [m.training for m in net.modules()] == flags
    assert torch.equal(torch.get_rng_state(), rng)
    assert "weight" not in json.dumps(result.graph.provenance)
    if shutil.which("dot"):
        assert "<svg" in result.graph.svg()


def test_dynamic_trace_failure_is_explicit():
    torch = pytest.importorskip("torch")
    class Branch(torch.nn.Module):
        def forward(self, x):
            return x * 2 if x.sum() > 0 else x - 1
    result = extract_torch_graph(Branch(), (torch.ones(2),))
    assert not result.supported and result.graph is None and result.diagnostics


def test_shared_parameters_and_export_shapes():
    torch = pytest.importorskip("torch")
    class Shared(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.a = torch.nn.Linear(2, 2)
            self.b = self.a
        def forward(self, x):
            return self.b(self.a(x))
    net = Shared()
    result = extract_torch_graph(net, (torch.ones(3, 2),))
    assert result.graph.parameter_count == 6
    assert sum(n.parameter_count for n in result.graph.nodes) == 6
    exported = extract_torch_graph(net, (torch.ones(3, 2),), method="export")
    assert exported.supported, exported.diagnostics
    assert any(t.shape == (3, 2) for n in exported.graph.nodes for t in n.tensors)


def test_scientific_graph_evidence_and_scene():
    graph = MLGraph("Scientific inference", (GraphNode("obs", "Measured inputs", unit="[A]"),
        GraphNode("net", "Learned block"), GraphNode("loss", "Physics loss", unit="[Wb]")),
        (GraphEdge("obs", "net", evidence="declared"), GraphEdge("net", "loss", "validation", "conceptual")))
    assert "dashed" in graph.to_dot() and "dotted" in graph.to_dot()
    assert scientific_graph_diagram(graph).scene.role("obs")
    with pytest.raises(ValueError, match="evidence"):
        GraphEdge("a", "b", evidence="guessed")


def test_local_experiment_roundtrip_and_plot_views(tmp_path):
    data = ml.build_dataset(np.arange(12.).reshape(6, 2), np.arange(6), targets=np.arange(6.)[:, None],
        spec=ml.DatasetSpec(name="r", task="regression"))
    result = ml.train_model(data, ml.split_groups(data), ml.ModelSpec(architecture="ridge", backend="numpy", task="ridge"))
    record = experiment_record(result, run_id="local")
    path = save_experiment(record, tmp_path / "run.json")
    assert load_experiment(path) == record
    assert record["model_identity"]["state_sha256"] == result.artifact.state_sha256
    with pytest.raises(FileExistsError):
        save_experiment(record, path)
    curves = learning_curves({"train_loss": [2., 1.], "validation_loss": [3., 2.]})
    assert not curves.series[0].y.flags.writeable
    evaluation = ml.evaluate_model(result.artifact, data, result.split)
    assert group_metrics(evaluation, "mse").series
    assert resource_comparison([record]).series
    assert "mlflow" not in sys.modules
