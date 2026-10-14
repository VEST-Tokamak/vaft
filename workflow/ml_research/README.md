# VAFT research ML qualification (#2016 / #2015)

This extends the #669 lifecycle; no second trainer, model registry, tuner or
scientific-data schema is introduced. The reference integration is opt-in in
`vaft-NeurIPSequilibrium.models.research.vaft_adapter`.

## Capability audit

| Capability | Existing contract | Confirmed gap and extension |
|---|---|---|
| Dense models | NumPy/sklearn/Torch, grouped splitting, train-only moments | Preserved; old artifact schema 0 still written/read |
| Multiple input tensors and ragged sets | Single ndarray; factory receives one input/output shape | `StructuredDatasetArtifact`, `TensorSpec`, lazy `StructuredSample` source; registered factory receives named shapes |
| Observation validity | Sample-level `available` | Boolean tensor masks; generated `<name>__valid` padding masks, True means valid |
| Missing supervision | Paired dense targets | Separate target masks; named supervised heads, finite-element validation |
| Large datasets | Full normalized arrays and eager device copies | Source access and masked streaming moments; only a collated batch moves to the device |
| Research optimizer | Adam only | Shared dense/structured engine: AdamW, OneCycle, clipping, fp32/bf16/fp16 |
| Physical losses | Registered factory returning scalar | Retained; structured factory receives named targets and `__target_masks__`; composite mapping includes differentiable `loss` and logged components |
| Warm-start / resume | No training checkpoint | Strict weights warm-start; epoch-boundary resume with optimizer/scheduler/scaler/RNG/best state and dataset/split/config binding |
| Run history | Artifact provenance | Local JSON run index and explicit local SQLite MLflow PoC; identities remain distinct |
| Architecture graphs | Scientific diagram primitives only | FX/export extraction into portable graph records; explicit failures and evidence, optional Graphviz SVG |

## Structured contract

A source implements `__len__`/`__getitem__` and returns `StructuredSample`.
`TensorSpec((None, features), mask="valid")` declares a ragged observation set.
Masks are True for real valid observations, distinct from physical zero.
Boolean/integer tensors bypass normalization; floats pool samples and a ragged
axis while keeping feature axes. Input names containing `__` are reserved for
framework context. Target shapes are fixed; missing elements use target masks.
Unobserved normalized channels fail explicitly. Padded/masked floats become zero
only after normalization and always travel with their masks.

Registered structured factories receive `(ModelSpec, input_shapes, target_shapes)`
and return a module mapping a named input dictionary to named target tensors.
NumPy/sklearn refuse this contract rather than coerce a dictionary into arrays.
Existing dense factories/losses/augmentations retain their signatures.

`train_model`, `predict`, and `evaluate_model` dispatch on the dataset contract.
Structured manifests use schema 1 and named dtype/shape/unit/normalization;
legacy dense manifests remain schema 0. Both use the same hashes, bundles and
`vaft-nn` resolver. Structured ONNX export and structured threshold calibration are explicitly unqualified; domain adapters own calibration.

Sources are stable content providers: fingerprints stream every sample once,
without retaining a corpus. Spawn workers require a picklable source; the
framework supplies seeded worker initialization and collation. The NeurIPS
research source owns unpickleable model/preprocessing objects and uses workers=0;
this does not restrict generic picklable sources. `fitted_transform_groups` in
DatasetSpec metadata declares any already-fitted preprocessing population and
must be a subset of the chosen training groups.

## Training and checkpoints

Defaults retain Adam, fp32, no scheduler/clipping, and workers=0. Advanced
options are Torch-only. `scheduler="onecycle"` uses the declared total epoch
budget and every optimizer step, including a final partial batch. A registered
loss may return a scalar or `{ "loss": total, "component": scalar, ... }`.
Nonfinite losses/gradients fail with an explicit error; nothing silently skips.
FP16 requires CUDA; BF16 availability is checked on CUDA. Loss reductions use
FP32 outputs; domain code retains responsibility for physical units/operators.
CPU determinism is qualified for fixed software/hardware; CUDA requests
warn-only deterministic operations and does not promise bitwise portability.

`checkpoint_path` writes an atomic epoch checkpoint. For a bounded pause, keep
`epochs` as the final budget and set `max_epochs_this_run`; resume with the same
config plus `resume_from`. Changing data fingerprint, split, normalization,
model or training settings rejects resume. `warm_start_from` loads strict
matching weights without optimizer/RNG continuation. Checkpoints are separate
from immutable deployment weights. Resume equality is tested at epoch boundaries;
mid-epoch recovery, topology changes and stochastic external source mutation
are unsupported.

## Tracking and tool qualification

Run from the repository with the package on PYTHONPATH:

```sh
pip install 'vaft[ml,tracking]' torchview
PYTHONPATH="$PWD" python workflow/ml_research/qualify_tools.py --out /tmp/ml-tools-unique
```

JSON is canonical for local experiment reports and retains arbitrary legacy JSON
fields. Explicit `index_mlflow(record, local_directory)` indexes scalar histories
in local SQLite, with no server/account, weights or raw data upload. Base imports
never require MLflow. Run UUIDs and MLflow IDs never become model identities.
MLflow is deferred as the default: the PoC adds querying but does not justify a
mandatory operational/dependency cost. Negative adoption is a valid outcome.

FX is the default extractor; export is an explicit alternate. Torchview is
compared, not adopted as the schema/renderer. A trace certifies only its example
inputs. Graphs retain shape/dtype, hierarchy, unique parameter counts and
provenance, never weights. Scientific graphs use typed couplings and distinguish
executed/declared/conceptual edges. Failures include diagnostics, not a fake graph.
Learning histories/group metrics/resource comparisons adapt to `vaft.plot`'s
immutable `LineSeries`; no raw module reaches the plot renderers.

## Integration and scientific boundaries

PCA-MLP/CNN/FNO/Set PCA/current/current+PCA network adapters are qualified against
original forward prediction, loss/gradient and immutable artifact restoration.
PCA/scalers/geometry/physical decoding stay in the research repository and its
pickle-free reference release, with a preprocessing digest and train groups.
Full consistency-proxy curricula and stochastic geometry subdivision remain
explicit extension boundaries in the research trainer. The adapter rejects an
active consistency-proxy curriculum rather than silently dropping its loss.
The frozen challenge path and existing release format/pinned VAFT CI are unchanged.

GPU qualification uses the separate NeurIPS `scripts/qualify_vaft_ml.sbatch`:
1 GPU, 4 CPUs, 16 GB, 30 min, 17x17 grids, 8 frames per shot, 4/1/1 independent
DIII-D train/validation/test shots, 2 epochs/model. It compares fp32/bf16,
clipping/scheduler/masked losses, hashes, restore parity, memory and latency.
Only label-free MAST inference is allowed. These are functionality checks, not
new scientific generalization or performance claims; no full retraining/HPO.
Scientific acceptance remains with the domain/`vaft.validation`, not tracking.

## Recorded qualification

Slurm job **785019** passed on an RTX PRO 6000 Blackwell (Torch 2.7.1+cu128).
All six families passed original/adapter FP32 prediction and artifact restore
checks, eight optimizer steps per run in fp32/bf16, and label-free MAST inference.
AMP forward relative L2 was 0.16–0.90%; peak device allocation was 18–23 MB for
these deliberately small fixtures. This is not a full-training memory forecast.
`gpu_qualification.json` records dataset hashes, selected frame indices, group
roles, software/source manifest identity, numerical metrics and limitations.
The source manifest hash identifies the qualified uploaded source tree; revision
fields identify the base commits, not a claim that uncommitted edits were absent.

FX extracted PCA-MLP, CNN and Set PCA. FNO and current decoders failed FX because
of symbolic shape/control-flow limitations; all six extracted with explicit
`torch.export`. These failures remain visible and never produce a substitute
conceptual graph labelled as executed.

Sample-level unavailable structured inputs are excluded from moments and model execution and return NaN outputs. A generated lazy source is tested against identical eager content for fingerprint, history and model-state equality.

Masked composite losses may additionally return `__aggregation__`, mapping every non-total component name to `(valid_element_count, total_coefficient)`. The trainer computes epoch component means using these counts and reconstructs the monitored total from the coefficients. Counts may be zero; coefficients must stay constant. Without this metadata existing losses retain sample-weighted aggregation. Built-in named masked MSE and the NeurIPS composite adapter supply this metadata.
