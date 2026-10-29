"""PyTorch backend: the first production NN backend.

``torch`` is imported when this backend is first *used*, never when
``vaft.process.ml`` is imported.  Training is seeded -- initialisation,
shuffling and augmentation each from ``TrainingConfig.seed`` -- and runs under
``torch.use_deterministic_algorithms``; on the CPU two runs with one seed
produce the same weights on one machine.  On an accelerator
(``TrainingConfig.device``) determinism is requested but not guaranteed.

Built-in architectures flatten each sample: ``mlp`` (supervised) and
``autoencoder`` (reconstruction).  A registered architecture receives the
sample shapes and keeps them (a sequence or image model).
"""

from __future__ import annotations

import contextlib
import io
import math

import numpy as np

from .._types import BackendUnavailableError, ModelContractError
from . import LOSSES, Backend

_INSTALL_HINT = "pip install 'vaft[ml]'"


def _torch():
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - exercised without torch
        raise BackendUnavailableError(
            f"the 'torch' ML backend needs PyTorch, which is not installed ({_INSTALL_HINT})"
        ) from exc
    return torch


def _activation(nn, name: str):
    table = {"relu": nn.ReLU, "gelu": nn.GELU, "tanh": nn.Tanh, "silu": nn.SiLU}
    if name not in table:
        raise ModelContractError(f"unknown activation {name!r}; known: {sorted(table)}")
    return table[name]


def _dense(spec, widths):
    nn = _torch().nn
    hp = dict(spec.hyperparameters)
    act = _activation(nn, str(hp.get("activation", "relu")))
    dropout = float(hp.get("dropout", 0.0))
    layers = []
    for i, (a, b) in enumerate(zip(widths[:-1], widths[1:])):
        layers.append(nn.Linear(a, b))
        if i < len(widths) - 2:
            layers.append(act())
            if dropout:
                layers.append(nn.Dropout(dropout))
    return nn.Sequential(*layers)


def _flattening(net, output_shape):
    torch = _torch()

    class Flattening(torch.nn.Module):
        """Flatten each sample, run a dense net, restore the output shape."""

        def __init__(self):
            super().__init__()
            self.net = net

        def forward(self, x):
            return self.net(torch.flatten(x, start_dim=1)).reshape((-1, *output_shape))

    return Flattening()


def _mlp(spec, input_shape, output_shape):
    hidden = [int(w) for w in spec.hyperparameters.get("hidden", (64, 64))]
    return _flattening(_dense(spec, [math.prod(input_shape), *hidden, math.prod(output_shape)]), output_shape)


def _autoencoder(spec, input_shape, output_shape):
    width = math.prod(input_shape)
    hidden = [int(w) for w in spec.hyperparameters.get("hidden_dims", (128, 64, 32))]
    if not hidden:
        raise ModelContractError("an autoencoder needs at least one hidden layer")
    if hidden[-1] >= width and not spec.hyperparameters.get("allow_overcomplete", False):
        raise ModelContractError(
            f"latent width {hidden[-1]} is not smaller than the {width} input features, so the "
            "autoencoder can learn the identity; narrow hidden_dims or set allow_overcomplete=True"
        )
    return _flattening(_dense(spec, [width, *hidden, *reversed(hidden[:-1]), width]), output_shape)


_BUILTIN = {"mlp": _mlp, "autoencoder": _autoencoder}


def _mse_factory(spec, normalization):
    torch = _torch()
    return lambda prediction, target, inputs: torch.mean((prediction - target) ** 2)


@contextlib.contextmanager
def _deterministic(torch, strict: bool):
    previous = torch.are_deterministic_algorithms_enabled()
    previous_warn = torch.is_deterministic_algorithms_warn_only_enabled()
    torch.use_deterministic_algorithms(True, warn_only=not strict)
    try:
        yield
    finally:
        torch.use_deterministic_algorithms(previous, warn_only=previous_warn)


class TorchBackend(Backend):
    name = "torch"
    inference_format = "torch-state-dict"
    weights_suffix = ".pt"
    builtin = {"mlp": "supervised", "autoencoder": "reconstruction"}
    iterative = True
    extensible = True
    trainable_kinds = ("supervised", "reconstruction")

    def runtime_versions(self):
        return {"torch": _torch().__version__}

    def build(self, spec, input_shape, output_shape):
        arch = self.architecture(spec.architecture)
        factory = arch.factory or _BUILTIN[spec.architecture]
        return factory(spec, tuple(input_shape), tuple(output_shape))

    def _load(self, spec, state: bytes):
        torch = _torch()
        payload = torch.load(io.BytesIO(state), map_location="cpu", weights_only=True)
        net = (self.build_structured(spec, payload["tensor_schema"]) if "tensor_schema" in payload
               else self.build(spec, tuple(payload["input_shape"]), tuple(payload["output_shape"])))
        net.load_state_dict(payload["state_dict"])
        net.eval()
        return net, payload

    def _loss(self, spec, normalization):
        name = str(spec.hyperparameters.get("loss", "mse"))
        if name == "mse":
            return _mse_factory(spec, normalization)
        if name not in LOSSES:
            raise ModelContractError(f"loss {name!r} is not registered; known: {['mse', *sorted(LOSSES)]}")
        return LOSSES[name](spec, dict(normalization))

    def fit(self, spec, config, x_train, y_train, x_val, y_val, *, output_shape, normalization, augment=None, context=None):
        kind = self.kind(spec.architecture)
        if kind == "score":
            raise ModelContractError("the torch backend trains supervised or reconstruction networks")
        torch = _torch()
        device = torch.device(config.device)
        # Everything that draws random numbers -- initialisation, dropout --
        # runs inside a forked RNG, so training is reproducible from the seed
        # alone and never reseeds the caller's generator.
        with torch.random.fork_rng(devices=[device] if device.type == "cuda" else []):
            torch.manual_seed(config.seed)
            return self._fit(spec, config, x_train, y_train, x_val, y_val, output_shape,
                             normalization, augment, device, context)

    def _fit(self, spec, config, x_train, y_train, x_val, y_val, output_shape, normalization, augment, device, context=None):
        from ._torch_engine import DenseSamples, TrainingHistory, run_training
        torch = _torch()
        input_shape = tuple(x_train.shape[1:])
        net = self.build(spec, input_shape, output_shape).to(device)
        def samples(x, y):
            return DenseSamples(np.asarray(x, np.float32), None if y is None else np.asarray(y, np.float32))
        def epoch_data(rng):
            x, y = augment(x_train, y_train, rng)
            return samples(x, y)
        train = samples(x_train, y_train)
        val = samples(x_val, y_val) if x_val is not None and len(x_val) else None
        history, best_epoch, resources = run_training(net, spec, config, train, val, self._loss(spec, normalization),
            normalization, context or {}, epoch_data=epoch_data if augment else None)
        buffer = io.BytesIO()
        torch.save({"state_dict": {k: v.detach().cpu() for k, v in net.state_dict().items()},
                    "input_shape": list(input_shape), "output_shape": list(output_shape)}, buffer)
        return buffer.getvalue(), TrainingHistory(history, resources), best_epoch, None

    def build_structured(self, spec, schema):
        arch = self.architecture(spec.architecture)
        if arch.factory is None:
            raise ModelContractError("structured data requires a registered architecture factory")
        return arch.factory(spec, {k: tuple(v["shape"]) for k, v in schema["inputs"].items()},
                            {k: tuple(v["shape"]) for k, v in schema["targets"].items()})

    def fit_structured(self, spec, config, dataset, train_indices, val_indices, normalization, context):
        from ._torch_engine import StructuredSamples, StructuredCollator, run_training
        torch = _torch()
        device = torch.device(config.device)
        schema = dataset.schema()
        def masked_mse(prediction, target, inputs):
            terms, aggregation = {}, {}
            for name, values in target.items():
                mask = inputs["__target_masks__"][name]
                count = int(mask.sum())
                component = "target/loss" if name == "loss" else name
                terms[component] = ((prediction[name] - values)[mask] ** 2).mean() if count else prediction[name].sum() * 0
                aggregation[component] = (count, 1.0)
            if not any(count for count, _ in aggregation.values()):
                raise ModelContractError("batch has no supervised target elements")
            return {"loss": sum(terms.values()), **terms, "__aggregation__": aggregation}
        loss = masked_mse if spec.hyperparameters.get("loss", "mse") == "mse" else self._loss(spec, normalization)
        with torch.random.fork_rng(devices=[device] if device.type == "cuda" else []):
            torch.manual_seed(config.seed)
            net = self.build_structured(spec, schema).to(device)
            history, best_epoch, resources = run_training(net, spec, config,
                StructuredSamples(dataset, train_indices, normalization),
                StructuredSamples(dataset, val_indices, normalization) if len(val_indices) else None,
                loss, normalization, context, structured=True, collate=StructuredCollator(dataset.input_specs))
        buffer = io.BytesIO()
        torch.save({"state_dict": {k: v.detach().cpu() for k, v in net.state_dict().items()},
                    "tensor_schema": schema}, buffer)
        return buffer.getvalue(), history, best_epoch, resources

    def predict_structured(self, spec, state, dataset, normalization, batch_size=64):
        from ._torch_engine import StructuredSamples, StructuredCollator
        torch = _torch()
        net, payload = self._load(spec, state)
        loader = torch.utils.data.DataLoader(StructuredSamples(dataset, range(dataset.n_samples), normalization),
            batch_size=batch_size, collate_fn=StructuredCollator(dataset.input_specs))
        values = {name: [] for name in dataset.target_specs}
        with torch.no_grad():
            for inputs, targets, masks in loader:
                output = net(inputs)
                if not isinstance(output, dict) or set(output) != set(values):
                    raise ModelContractError("structured predictions differ from target names")
                for name, a in output.items():
                    if tuple(a.shape) != tuple(targets[name].shape):
                        raise ModelContractError(f"prediction {name!r} has an unexpected shape")
                    a = a.float().numpy().astype(float)
                    values[name].append(a * normalization[f"target.{name}.std"] + normalization[f"target.{name}.mean"])
        return {name: np.concatenate(arrays) for name, arrays in values.items()}

    def predict(self, spec, state, x, output_shape):
        torch = _torch()
        net, _ = self._load(spec, state)
        with torch.no_grad():
            out = net(torch.as_tensor(np.asarray(x, dtype=np.float32))).numpy().astype(np.float64)
        return out.reshape((len(x), *output_shape))

    def export(self, spec, state, input_shape, output_shape, path, fmt):
        if fmt != "onnx":
            return super().export(spec, state, input_shape, output_shape, path, fmt)
        torch = _torch()
        try:
            import onnx  # noqa: F401  (the exporter needs it)
            import onnxruntime
        except ImportError as exc:
            raise BackendUnavailableError(
                f"ONNX export needs the onnx and onnxruntime packages ({_INSTALL_HINT})"
            ) from exc
        model, _ = self._load(spec, state)
        # The TorchScript exporter (dynamo=False): it needs only `onnx`, where the
        # torch.export-based default also needs `onnxscript`. It is deprecated
        # since torch 2.9 and warns; moving to the torch.export exporter is a follow-up.
        opset = 17
        example = torch.zeros((1, *input_shape), dtype=torch.float32)
        torch.onnx.export(
            model,
            (example,),
            str(path),
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={"input": {0: "n"}, "output": {0: "n"}},
            opset_version=opset,
            dynamo=False,
        )
        probe = np.random.default_rng(0).standard_normal((4, *input_shape)).astype(np.float32)
        with torch.no_grad():
            reference = model(torch.as_tensor(probe)).numpy()
        session = onnxruntime.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        exported = session.run(["output"], {"input": probe})[0]
        max_abs = float(np.max(np.abs(exported - reference)))
        tolerance = 1.0e-5 * max(1.0, float(np.max(np.abs(reference))))
        if not max_abs <= tolerance:
            raise ModelContractError(
                f"ONNX export disagrees with the torch model by {max_abs:.3g} > {tolerance:.3g}"
            )
        return {
            "format": "onnx",
            "opset": opset,
            "parity_max_abs_diff": max_abs,
            "parity_tolerance": tolerance,
            "onnxruntime": onnxruntime.__version__,
            "torch": torch.__version__,
        }


BACKEND = TorchBackend()
