"""Reproduce local tracking/graph-tool qualification without research data."""
from pathlib import Path
import argparse
import json
import time
import importlib.metadata

import numpy as np
import torch

from vaft.process import ml
from vaft.process.ml.experiments import experiment_record, save_experiment, index_mlflow
from vaft.diagram.ml import GraphNode, GraphEdge, MLGraph, extract_torch_graph, scientific_graph_diagram
from vaft.plot.ml import learning_curves
from vaft.plot.renderers.lines import render_line_series


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(7)
    x = rng.normal(size=(48, 3))
    data = ml.build_dataset(x, np.repeat(np.arange(8), 6), targets=(x[:, :1]*2 + .1), spec=ml.DatasetSpec(name="reference", task="regression"))
    result = ml.train_model(data, ml.split_groups(data), ml.ModelSpec(architecture="mlp", task="demo", hyperparameters={"hidden": [4]}),
                            ml.TrainingConfig(epochs=3, batch_size=8))
    record = experiment_record(result, run_id="qualification-reference")
    started = time.perf_counter()
    save_experiment(record, args.out / "run.json")
    json_seconds = time.perf_counter() - started
    started = time.perf_counter()
    tracking = index_mlflow(record, args.out / "mlflow")
    mlflow_seconds = time.perf_counter() - started
    net, _ = ml._backends.get_backend("torch")._load(result.artifact.model_spec, result.artifact.state)
    samples = torch.zeros(2, 3)
    graphs = {}
    for method in ("fx", "export"):
        extracted = extract_torch_graph(net, (samples,), method=method, name="VAFT MLP")
        graphs[method] = {"supported": extracted.supported, "diagnostics": extracted.diagnostics}
        if extracted.supported:
            extracted.graph.svg(args.out / f"mlp_{method}.svg")
            (args.out / f"mlp_{method}.json").write_text(json.dumps(extracted.graph.to_dict(), indent=2))
    try:
        from torchview import draw_graph
        view = draw_graph(net, input_data=samples, device="cpu", save_graph=False)
        view.visual_graph.render(str(args.out / "mlp_torchview"), format="svg", cleanup=True)
        graphs["torchview"] = {"supported": True, "notes": "External visual graph; not VAFT semantic/provenance schema."}
    except Exception as exc:
        graphs["torchview"] = {"supported": False, "diagnostics": str(exc)}
    semantic = MLGraph("Scientific ML: evidence and units", (
        GraphNode("obs", "Measured diagnostic signals", unit="[A]"),
        GraphNode("pre", "Train-only normalization", convention="Training groups only"),
        GraphNode("net", "Learned inference block"),
        GraphNode("out", "Predicted flux", unit="[Wb]"),
        GraphNode("valid", "Independent scientific validation")), (
        GraphEdge("obs", "pre", evidence="declared"), GraphEdge("pre", "net", evidence="declared"),
        GraphEdge("net", "out", evidence="declared"), GraphEdge("out", "valid", "validation", "conceptual")))
    (args.out / "scientific.svg").write_text(scientific_graph_diagram(semantic).svg)
    import matplotlib
    matplotlib.rcParams["svg.hashsalt"] = "vaft-ml-research"
    fig, _ = render_line_series(learning_curves(result.history, keys=("train_loss", "validation_loss")))
    fig.savefig(args.out / "learning.svg", metadata={"Date": None})
    svg_path = args.out / "learning.svg"
    svg_path.write_text("\n".join(line.rstrip() for line in svg_path.read_text().splitlines()) + "\n")
    from mlflow.tracking import MlflowClient
    client = MlflowClient(tracking_uri=tracking["uri"])
    found = client.get_run(tracking["tracking_run_id"])
    assert found.data.tags["vaft.run_id"] == record["run_id"]
    versions = {k: importlib.metadata.version(k) for k in ("torch", "mlflow", "torchview")}
    report = {"versions": versions, "graph_tools": graphs, "tracking": tracking,
        "json_seconds": json_seconds, "mlflow_seconds": mlflow_seconds,
        "decision": "defer MLflow as default; keep explicit local optional PoC",
        "rationale": "JSON already retains all provenance and is service-free. MLflow adds indexed history/querying, but no requirement for that index justifies a default dependency.",
        "graph_decision": "FX default; export explicit alternate for complex/dynamic operators; torchview remains a comparison tool."}
    (args.out / "tools_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
