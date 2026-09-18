"""PyTorch backend: the first production NN backend (#669 phase 2).

``torch`` is imported when this backend is first *used*, never when
``vaft.process.ml`` is imported.  Training runs on the CPU with seeded
initialisation, a seeded shuffling generator and
``torch.use_deterministic_algorithms``, so two runs with one seed produce the
same weights on one machine.
"""

from __future__ import annotations

import contextlib
import copy
import io
import math

import numpy as np

from .._types import BackendUnavailableError, ModelContractError
from . import Backend

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


def _build(spec, in_dim: int, out_dim: int):
    """The network for ``spec``: flat ``in_dim`` features to flat ``out_dim``."""
    torch = _torch()
    nn = torch.nn
    hp = dict(spec.hyperparameters)
    act = _activation(nn, str(hp.get("activation", "relu")))
    dropout = float(hp.get("dropout", 0.0))
    if spec.architecture == "mlp":
        widths = [in_dim, *[int(w) for w in hp.get("hidden", (64, 64))], out_dim]
    elif spec.architecture == "autoencoder":
        hidden = [int(w) for w in hp.get("hidden_dims", (128, 64, 32))]
        if not hidden:
            raise ModelContractError("an autoencoder needs at least one hidden layer")
        widths = [in_dim, *hidden, *reversed(hidden[:-1]), out_dim]
    else:
        raise ModelContractError(f"torch backend has no architecture {spec.architecture!r}")
    layers = []
    for i, (a, b) in enumerate(zip(widths[:-1], widths[1:])):
        layers.append(nn.Linear(a, b))
        if i < len(widths) - 2:
            layers.append(act())
            if dropout:
                layers.append(nn.Dropout(dropout))
    return nn.Sequential(*layers)


class _Shaped:
    """Wrap a flat network so it maps ``(n, *input_shape)`` to ``(n, *output_shape)``."""

    @staticmethod
    def make(net, output_shape):
        torch = _torch()

        class Shaped(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.net = net

            def forward(self, x):
                y = self.net(torch.flatten(x, start_dim=1))
                return y.reshape((-1, *output_shape))

        return Shaped()


@contextlib.contextmanager
def _deterministic(torch):
    previous = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)
    try:
        yield
    finally:
        torch.use_deterministic_algorithms(previous)


class TorchBackend(Backend):
    name = "torch"
    inference_format = "torch-state-dict"
    weights_suffix = ".pt"
    architectures = {"mlp": True, "autoencoder": False}

    def runtime_versions(self):
        return {"torch": _torch().__version__}

    def _load(self, spec, state: bytes):
        torch = _torch()
        payload = torch.load(io.BytesIO(state), map_location="cpu", weights_only=True)
        net = _build(spec, int(payload["in_dim"]), int(payload["out_dim"]))
        net.load_state_dict(payload["state_dict"])
        net.eval()
        return net, payload

    def fit(self, spec, config, x_train, y_train, x_val, y_val):
        self.is_supervised(spec.architecture)
        torch = _torch()
        in_dim = int(np.prod(x_train.shape[1:]))
        out_dim = int(np.prod(y_train.shape[1:]))
        torch.manual_seed(config.seed)
        net = _build(spec, in_dim, out_dim)
        optimiser = torch.optim.Adam(
            net.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
        )
        loss_fn = torch.nn.MSELoss()
        xt = torch.as_tensor(x_train.reshape(len(x_train), -1), dtype=torch.float32)
        yt = torch.as_tensor(y_train.reshape(len(y_train), -1), dtype=torch.float32)
        has_val = x_val is not None and len(x_val) > 0
        if has_val:
            xv = torch.as_tensor(x_val.reshape(len(x_val), -1), dtype=torch.float32)
            yv = torch.as_tensor(y_val.reshape(len(y_val), -1), dtype=torch.float32)
        generator = torch.Generator().manual_seed(config.seed)
        batch = max(1, int(config.batch_size))
        history: dict[str, list[float]] = {"train_loss": []}
        if has_val:
            history["validation_loss"] = []
        best_loss, best_epoch, best_state, stale = math.inf, None, None, 0
        with _deterministic(torch):
            for epoch in range(int(config.epochs)):
                net.train()
                order = torch.randperm(len(xt), generator=generator)
                total = 0.0
                for start in range(0, len(xt), batch):
                    idx = order[start:start + batch]
                    optimiser.zero_grad()
                    loss = loss_fn(net(xt[idx]), yt[idx])
                    loss.backward()
                    optimiser.step()
                    total += float(loss.detach()) * len(idx)
                history["train_loss"].append(total / len(xt))
                net.eval()
                if has_val:
                    with torch.no_grad():
                        monitored = float(loss_fn(net(xv), yv))
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
        torch.save({"state_dict": net.state_dict(), "in_dim": in_dim, "out_dim": out_dim}, buffer)
        return buffer.getvalue(), history, best_epoch

    def predict(self, spec, state, x, output_shape):
        torch = _torch()
        net, _ = self._load(spec, state)
        with torch.no_grad():
            flat = torch.as_tensor(np.asarray(x, dtype=np.float32).reshape(len(x), -1))
            out = net(flat).numpy().astype(np.float64)
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
        net, _ = self._load(spec, state)
        model = _Shaped.make(net, tuple(output_shape)).eval()
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
