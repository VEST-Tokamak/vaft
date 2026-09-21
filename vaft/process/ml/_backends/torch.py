"""PyTorch backend: the first production NN backend (#669 phase 2).

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
import copy
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
        net = self.build(spec, tuple(payload["input_shape"]), tuple(payload["output_shape"]))
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

    def fit(self, spec, config, x_train, y_train, x_val, y_val, *, output_shape, normalization, augment=None):
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
                             normalization, augment, device)

    def _fit(self, spec, config, x_train, y_train, x_val, y_val, output_shape, normalization, augment, device):
        torch = _torch()
        input_shape = tuple(x_train.shape[1:])
        net = self.build(spec, input_shape, output_shape).to(device)
        optimiser = torch.optim.Adam(net.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
        loss_fn = self._loss(spec, normalization)

        def tensor(a):
            return torch.as_tensor(np.asarray(a, dtype=np.float32), device=device)

        supervised = y_train is not None
        has_val = x_val is not None and len(x_val) > 0
        if has_val:
            xv = tensor(x_val)
            yv = tensor(y_val) if supervised else xv
        generator = torch.Generator().manual_seed(config.seed)
        rng = np.random.default_rng(config.seed)
        batch = max(1, int(config.batch_size))
        history: dict[str, list[float]] = {"train_loss": []}
        if has_val:
            history["validation_loss"] = []
        best_loss, best_epoch, best_state, stale = math.inf, None, None, 0
        xt_fixed = tensor(x_train)
        yt_fixed = tensor(y_train) if supervised else xt_fixed
        with _deterministic(torch, strict=device.type == "cpu"):
            for epoch in range(int(config.epochs)):
                if augment is not None:
                    xa, ya = augment(x_train, y_train, rng)
                    xt = tensor(xa)
                    yt = tensor(ya) if supervised else xt
                else:
                    xt, yt = xt_fixed, yt_fixed
                net.train()
                order = torch.randperm(len(xt), generator=generator)
                total = 0.0
                for start in range(0, len(xt), batch):
                    idx = order[start:start + batch].to(device)
                    optimiser.zero_grad()
                    loss = loss_fn(net(xt[idx]), yt[idx], xt[idx])
                    loss.backward()
                    optimiser.step()
                    total += float(loss.detach()) * len(idx)
                history["train_loss"].append(total / len(xt))
                net.eval()
                if has_val:
                    with torch.no_grad():
                        monitored = float(loss_fn(net(xv), yv, xv))
                    history["validation_loss"].append(monitored)
                else:
                    monitored = history["train_loss"][-1]
                if monitored < best_loss:
                    best_loss, best_epoch, stale = monitored, epoch, 0
                    best_state = copy.deepcopy(net.state_dict())
                else:
                    stale += 1
                    patience = config.early_stopping_patience
                    if patience is not None and stale >= patience:
                        break
        if best_state is not None:
            net.load_state_dict(best_state)
        buffer = io.BytesIO()
        cpu_state = {k: v.detach().cpu() for k, v in net.state_dict().items()}
        torch.save(
            {"state_dict": cpu_state, "input_shape": list(input_shape), "output_shape": list(output_shape)},
            buffer,
        )
        return buffer.getvalue(), history, best_epoch, None

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
        # since torch 2.9 and warns; moving is a follow-up of #669.
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
