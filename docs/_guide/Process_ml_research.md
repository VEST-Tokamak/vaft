---
title: "Scientific ML: structured observations, training and model graphs"
author: VEST team
date: "2026-10-14 21:00:00 +0900"
category: guide
layout: post
permalink: /reference/process/ml/research/
related:
  api: [process]
---

VAFT extends its existing ML lifecycle with mask-aware named observations,
research training options and portable architecture graphs. The NeurIPS
research adapter is a reference consumer; competition-specific models and
physics stay outside VAFT core. No full challenge training is required.

## Model computation and scientific meaning

A computational graph describes the supplied example execution. It carries
shapes, dtypes, module hierarchy, parameter counts and extraction provenance.
A static trace cannot prove behavior for arbitrary dynamic branches or shapes.
FX is the default; an explicit export attempt can support additional operators.
A failed attempt returns diagnostics rather than a misleading complete graph.

![Automatically extracted VAFT MLP, with shape and parameter counts]({{ '/assets/ml-research/mlp_fx.svg' | relative_url }})

Scientific diagrams use VAFT's existing renderer and typed coupling vocabulary.
Executed, declared and conceptual relations are stated independently of model
performance. Physical units and preprocessing conventions belong in the graph.

![Scientific flow with explicit units and declared/conceptual evidence]({{ '/assets/ml-research/scientific.svg' | relative_url }})

## Training metrics

Loss histories, independent-group evaluation and measured resources adapt to
immutable `vaft.plot.models.LineSeries`. Plot renderers do not receive model
weights or a raw Torch module. Select related metrics with compatible units.

![Small CPU training history]({{ '/assets/ml-research/learning.svg' | relative_url }})

## Structured source and checkpoint example

```python
from vaft.process import ml

# source[i] returns StructuredSample(inputs, targets, target_masks).
data = ml.build_structured_dataset(
    source, groups, spec=ml.DatasetSpec(name="observations", task="regression"),
    input_specs={
        "tokens": ml.TensorSpec((None, 8), mask="valid", unit="[mixed declared features]"),
        "valid": ml.TensorSpec((None,), dtype="bool", normalize=False),
    },
    target_specs={"flux": ml.TensorSpec((17, 17), unit="[Wb/rad]")},
)
split = ml.split_groups(data)
# Register a task factory returning a module: named inputs -> named targets.
config = ml.TrainingConfig(
    optimizer="adamw", scheduler="onecycle", max_grad_norm=1.0,
    epochs=4, checkpoint_path="training.pt", max_epochs_this_run=2,
)
result = ml.train_model(data, split, registered_model_spec, config)
```

True means valid, including generated padding masks. Missing measurements are
not physical zeros. Train-only moments exclude masked elements and validation
samples. Target masks are separate from observation masks. Structured models
use manifest schema 1; existing dense models keep schema 0 and their APIs.

Resume uses the same total epoch budget and rejects a changed model, dataset,
split, normalization or training configuration. A warm-start loads only matching
weights. Training checkpoints and published model artifacts remain distinct.

## Qualified behavior and limits

Six NeurIPS network families passed small real DIII-D GPU checks (4/1/1 shots,
8 frames per shot, 17×17 grids, two epochs): PCA-MLP, CNN, FNO and Set
PCA/current/current+PCA. MAST checks used label-free inputs only. FX succeeded
for PCA-MLP/CNN/Set PCA; FNO/current decoders reported FX limitations and
extracted through the explicit export path. AMP forward relative L2 was below
1% on the small fixture; this is functionality evidence, not a scientific
performance or full-training memory claim.

JSON retains canonical local run provenance. A server-free MLflow PoC adds
optional querying; it is deferred as the default dependency. No raw data or
weights are uploaded by the tracking adapter. Model identity still belongs to
VAFT artifacts and `vaft-nn`.

Full consistency-proxy curricula, domain preprocessing/physical decoding and
stochastic geometry subdivision stay in the research implementation. Structured
ONNX export is unqualified; the native artifact remains available. GPU
determinism is not guaranteed across devices or software revisions.

Reproduce the gallery with `workflow/ml_research/qualify_tools.py`; the offline
CPU notebook is `notebooks/scientific_ml_lifecycle.ipynb`. The capability audit,
source/fixture hashes, tool decision and GPU report are in `workflow/ml_research/`.
