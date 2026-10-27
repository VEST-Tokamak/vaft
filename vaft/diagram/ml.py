"""Portable ML computational and scientific graphs.

Extraction is an observation for the supplied examples, never proof of all
runtime paths. No tensors or weights are retained in a graph. Torch, Graphviz
and the existing TeX renderer are optional at the point they are used.
"""
from __future__ import annotations

__all__ = ['TensorDescriptor', 'GraphNode', 'GraphEdge', 'MLGraph', 'GraphExtraction', 'extract_torch_graph', 'scientific_graph_diagram']

import copy
from dataclasses import dataclass, field, asdict
import json
from pathlib import Path
import shutil
import subprocess
from typing import Mapping


@dataclass(frozen=True)
class TensorDescriptor:
    shape: tuple
    dtype: str


@dataclass(frozen=True)
class GraphNode:
    key: str
    label: str
    kind: str = "operation"
    tensors: tuple[TensorDescriptor, ...] = ()
    module: str = ""
    parameter_count: int = 0
    unit: str = ""
    convention: str = ""


@dataclass(frozen=True)
class GraphEdge:
    source: str
    target: str
    coupling: str = "data_flow"
    evidence: str = "executed"

    def __post_init__(self):
        from ._modeling_schema import ModelCoupling
        ModelCoupling(self.source, self.target, self.coupling)
        if self.evidence not in ("executed", "declared", "conceptual"):
            raise ValueError("edge evidence must be executed, declared or conceptual")


@dataclass(frozen=True)
class MLGraph:
    name: str
    nodes: tuple[GraphNode, ...]
    edges: tuple[GraphEdge, ...]
    parameter_count: int = 0
    method: str = "semantic"
    limitations: tuple[str, ...] = ()
    provenance: Mapping = field(default_factory=dict)

    def __post_init__(self):
        keys = [node.key for node in self.nodes]
        if len(keys) != len(set(keys)):
            raise ValueError("graph node keys must be unique")
        if any(edge.source not in keys or edge.target not in keys for edge in self.edges):
            raise ValueError("graph edges must reference existing nodes")
        # Store independent JSON-only provenance; never a framework object.
        object.__setattr__(self, "provenance", json.loads(json.dumps(dict(self.provenance))))

    def to_dict(self):
        return asdict(self)

    def to_dot(self):
        quote = json.dumps
        header = self.name + f" | {self.method} | {self.parameter_count} parameters"
        if self.limitations:
            header += "\n" + "\n".join(self.limitations)
        lines = ["digraph model {", "rankdir=LR;", "graph [labelloc=t, label=" + quote(header) + "];",
                 'node [shape=box, style="rounded,filled", fillcolor="#edf3fa", fontname="Helvetica"];']
        for node in self.nodes:
            label = node.label
            if node.module:
                label += "\n" + node.module
            for tensor in node.tensors:
                label += "\n" + str(tensor.shape) + " " + tensor.dtype
            if node.parameter_count:
                label += f"\n{node.parameter_count} parameters"
            if node.unit:
                label += "\n" + node.unit
            if node.convention:
                label += "\n" + node.convention
            lines.append(f"{quote(node.key)} [label={quote(label)}];")
        for edge in self.edges:
            style = {"executed": "solid", "declared": "dashed", "conceptual": "dotted"}[edge.evidence]
            lines.append(f"{quote(edge.source)} -> {quote(edge.target)} [style={style}, label={quote(edge.coupling + ' / ' + edge.evidence)}];")
        return "\n".join([*lines, "}"])

    def svg(self, path=None):
        """Render through the optional Graphviz executable; no Python binding."""
        dot = shutil.which("dot")
        if dot is None:
            raise RuntimeError("ML graph SVG needs Graphviz 'dot'; install Graphviz or use to_dot()")
        result = subprocess.run([dot, "-Tsvg"], input=self.to_dot(), text=True, capture_output=True, check=True, timeout=60)
        from ._render import normalize_svg
        svg = normalize_svg(result.stdout)
        if path is not None:
            Path(path).write_text(svg, encoding="utf-8")
        return svg

    def _repr_svg_(self):
        return self.svg()


@dataclass(frozen=True)
class GraphExtraction:
    graph: MLGraph | None
    method: str
    diagnostics: tuple[str, ...] = ()

    @property
    def supported(self):
        return self.graph is not None


def _descriptors(value):
    if hasattr(value, "shape") and hasattr(value, "dtype"):
        return (TensorDescriptor(tuple(str(s) if not isinstance(s, int) else s for s in value.shape), str(value.dtype)),)
    if isinstance(value, dict):
        return tuple(t for v in value.values() for t in _descriptors(v))
    if isinstance(value, (tuple, list)):
        return tuple(t for v in value for t in _descriptors(v))
    return ()


def extract_torch_graph(module, args, *, name="model", method="fx", provenance=None):
    """Observe a model copy on example positional inputs (including dictionaries).

    Returns explicit diagnostics on unsupported tracing or shape propagation.
    The caller's train/eval flags, weights, buffers and RNG are preserved.
    Export/fx graphs only describe the supplied example, not arbitrary dynamic
    token counts, branches or unsupported complex/spectral operations.
    """
    if method not in ("fx", "export"):
        raise ValueError("method must be fx or export")
    try:
        import torch
    except ImportError as exc:
        raise ImportError("extract_torch_graph needs optional PyTorch: pip install 'vaft[ml]'") from exc
    modules = dict(module.named_modules())
    count = sum(p.numel() for p in module.parameters())
    cuda_devices = sorted({p.device.index for p in module.parameters() if p.device.type == "cuda"})
    try:
        with torch.random.fork_rng(devices=cuda_devices):
            probe = copy.deepcopy(module).eval()
            examples = copy.deepcopy(tuple(args))
            if method == "fx":
                graph_module = torch.fx.symbolic_trace(probe)
                from torch.fx.passes.shape_prop import ShapeProp
                with torch.no_grad():
                    ShapeProp(graph_module).propagate(*examples)
            else:
                exported = torch.export.export(probe, examples, strict=False)
                with torch.no_grad():
                    torch.testing.assert_close(exported.module()(*examples), probe(*examples), rtol=1e-5, atol=1e-6)
                graph_module = exported.graph_module
            nodes, edges = [], []
            charged = set()
            for node in graph_module.graph.nodes:
                module_name = str(node.target) if node.op == "call_module" else ""
                if not module_name and node.meta.get("nn_module_stack"):
                    module_name = list(node.meta["nn_module_stack"].values())[-1][0]
                owned = modules.get(module_name) if module_name else None
                parameters = 0
                if owned is not None:
                    for p in owned.parameters():
                        if id(p) not in charged:
                            parameters += p.numel()
                            charged.add(id(p))
                tensors = _descriptors(node.meta.get("tensor_meta", node.meta.get("val")))
                label = str(node.target)
                if node.op == "call_function":
                    label = getattr(node.target, "__module__", "") + "." + getattr(node.target, "__name__", type(node.target).__name__)
                nodes.append(GraphNode(node.name, label, node.op, tensors, module_name, parameters))
                edges.extend(GraphEdge(parent.name, node.name) for parent in node.all_input_nodes)
        record = {"torch": torch.__version__, "input_tensors": [asdict(t) for t in _descriptors(args)],
                  **dict(provenance or {})}
        graph = MLGraph(name, tuple(nodes), tuple(edges), count, method,
                        ("Example-specific trace; dynamic paths and other shapes are not certified.",), record)
        return GraphExtraction(graph, method)
    except Exception as exc:
        return GraphExtraction(None, method, (f"{type(exc).__name__}: {str(exc)[:1000]}",
                                               "No complete computational graph was extracted."))


def scientific_graph_diagram(graph):
    """Render a declared scientific graph using VAFT's existing scene/TikZ API."""
    from ._concept import box, connector
    from ._scene import Scene, Label
    from ._render import Diagram
    from ._concept import escape_latex
    boxes, items = {}, []
    # A vertical layout is predictable for small explanatory scientific graphs;
    # edges and evidence are explicit, including feedback and cross connections.
    for i, node in enumerate(graph.nodes):
        label = node.label
        if node.unit:
            label += "\n" + node.unit
        if node.convention:
            label += "\n" + node.convention
        b = box(0, -i * 2.5, 7, 1.6, label, role=node.key)
        boxes[node.key] = b
        items.extend(b.items)
    for edge in graph.edges:
        items.append(connector(boxes[edge.source], boxes[edge.target], role=f"{edge.source}:{edge.target}"))
        a, b = boxes[edge.source].center, boxes[edge.target].center
        items.append(Label((4.5, (a[1]+b[1])/2), escape_latex(edge.coupling + " / " + edge.evidence), "concept text"))
    items.append(Label((0, 2), escape_latex(graph.name), "concept text"))
    return Diagram("ml_scientific_graph", Scene(tuple(items)), model=graph)
