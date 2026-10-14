"""One optional Torch training loop for dense and structured observations."""
from __future__ import annotations

import copy
import hashlib
import os
from pathlib import Path
import random
import tempfile
import time

import numpy as np

from .._types import ModelContractError, canonical_json
from ..structured import valid_mask


class TrainingHistory(dict):
    """Internal history mapping with per-fit resource measurements."""
    def __init__(self, values, resources):
        super().__init__(values)
        self.resources = resources


def seed_worker(worker_id):
    import torch
    seed = torch.initial_seed() % (2**32)
    np.random.seed(seed)
    random.seed(seed)


class DenseSamples:
    def __init__(self, x, y):
        self.x, self.y = x, x if y is None else y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        return self.x[index], self.y[index]


class StructuredSamples:
    def __init__(self, data, indices, moments):
        self.data, self.indices, self.moments = data, tuple(int(i) for i in indices), dict(moments)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        sample = self.data.sample(self.indices[index])
        inputs, targets = {}, {}
        for role, values, schemas, out in (("input", sample.inputs, self.data.input_specs, inputs),
                                           ("target", sample.targets, self.data.target_specs, targets)):
            for name, spec in schemas.items():
                a = values[name]
                mask = valid_mask(sample.inputs.get(spec.mask) if role == "input" and spec.mask else
                                  sample.target_masks[name] if role == "target" else None, a.shape)
                if np.dtype(spec.dtype).kind == "f":
                    mean, std = self.moments[f"{role}.{name}.mean"], self.moments[f"{role}.{name}.std"]
                    a = (a - mean) / std
                out[name] = np.asarray(np.where(mask, a, 0), dtype=spec.dtype)
        return inputs, targets, sample.target_masks


class StructuredCollator:
    def __init__(self, input_specs):
        self.input_specs = dict(input_specs)

    def __call__(self, rows):
        import torch
        inputs, targets, masks = {}, {}, {}
        for name, spec in self.input_specs.items():
            arrays = [r[0][name] for r in rows]
            if None in spec.shape:
                length = max(len(a) for a in arrays)
                if length == 0:
                    raise ModelContractError(f"empty ragged batch for {name}; adapters must handle no observations explicitly")
                pad = np.zeros((len(rows), length, *spec.shape[1:]), dtype=spec.dtype)
                valid = np.zeros((len(rows), length), bool)
                for i, a in enumerate(arrays):
                    pad[i, :len(a)] = a
                    valid[i, :len(a)] = True
                inputs[name] = torch.from_numpy(pad)
                inputs[f"{name}__valid"] = torch.from_numpy(valid)
            else:
                inputs[name] = torch.from_numpy(np.stack(arrays))
        for name in rows[0][1]:
            targets[name] = torch.from_numpy(np.stack([r[1][name] for r in rows]))
            masks[name] = torch.from_numpy(np.stack([r[2][name] for r in rows]))
        return inputs, targets, masks


def move(value, device):
    if isinstance(value, dict):
        return {k: move(v, device) for k, v in value.items()}
    return value.to(device)


def checkpoint_binding(spec, config, normalization, context):
    cfg = config.to_dict()
    for name in ("checkpoint_path", "resume_from", "warm_start_from", "max_epochs_this_run"):
        cfg.pop(name, None)
    digest = hashlib.sha256(canonical_json({"model": spec.to_dict(), "config": cfg, "context": context}).encode())
    for key, a in sorted(normalization.items()):
        digest.update(key.encode())
        digest.update(np.ascontiguousarray(a).tobytes())
    return digest.hexdigest()


def _save_checkpoint(torch, path, record):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(prefix=path.name, dir=path.parent)
    os.close(fd)
    try:
        torch.save(record, temp)
        os.replace(temp, path)
    finally:
        if Path(temp).exists():
            Path(temp).unlink()


def run_training(net, spec, config, train_data, val_data, loss_fn, normalization, context,
                 *, structured=False, collate=None, epoch_data=None):
    import torch
    from .torch import _deterministic
    device = torch.device(config.device)
    if config.precision == "fp16" and device.type != "cuda":
        raise ModelContractError("fp16 training requires CUDA; select fp32 or CPU bf16")
    if config.precision == "bf16" and device.type == "cuda" and not torch.cuda.is_bf16_supported():
        raise ModelContractError("this CUDA device does not support bf16")
    dtype = torch.bfloat16 if config.precision == "bf16" else torch.float16
    enabled = config.precision != "fp32"
    scaler = torch.amp.GradScaler("cuda", enabled=enabled and config.precision == "fp16")
    optimizer_cls = torch.optim.AdamW if config.optimizer == "adamw" else torch.optim.Adam
    optimizer = optimizer_cls(net.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    generator = torch.Generator().manual_seed(config.seed)
    rng = np.random.default_rng(config.seed)
    binding = checkpoint_binding(spec, config, normalization, context)
    steps = (len(train_data) + config.batch_size - 1) // config.batch_size
    scheduler = None if config.scheduler is None else torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=config.learning_rate, total_steps=config.epochs * steps)
    best_loss, best_epoch, best_state, stale, start_epoch, step = float("inf"), None, None, 0, 0, 0
    history = {"train_loss": []}
    has_val = val_data is not None and len(val_data) > 0
    if has_val:
        history["validation_loss"] = []
    if config.warm_start_from:
        saved = torch.load(config.warm_start_from, map_location=device, weights_only=True)
        weights = saved.get("best_state") or saved.get("state_dict")
        if weights is None:
            raise ModelContractError("warm-start file must contain state_dict or best_state")
        net.load_state_dict(weights, strict=True)
    if config.resume_from:
        saved = torch.load(config.resume_from, map_location="cpu", weights_only=True)
        if saved.get("binding") != binding or saved.get("checkpoint_schema") != 1:
            raise ModelContractError("resume checkpoint differs in model, dataset, split, normalization or training config")
        net.load_state_dict(saved["state_dict"])
        optimizer.load_state_dict(saved["optimizer"])
        if scheduler is not None:
            scheduler.load_state_dict(saved["scheduler"])
        scaler.load_state_dict(saved["scaler"])
        torch.set_rng_state(saved["torch_rng"])
        if device.type == "cuda":
            torch.cuda.set_rng_state_all(saved["cuda_rng"])
        generator.set_state(saved["loader_rng"])
        rng.bit_generator.state = saved["numpy_rng"]
        history, start_epoch, step = saved["history"], saved["next_epoch"], saved["step"]
        best_loss, best_epoch, best_state, stale = saved["best_loss"], saved["best_epoch"], saved["best_state"], saved["stale"]

    def loader(data, shuffle):
        try:
            return torch.utils.data.DataLoader(data, batch_size=config.batch_size, shuffle=shuffle,
                num_workers=config.loader_workers, pin_memory=config.loader_pin_memory,
                worker_init_fn=seed_worker, generator=generator, collate_fn=collate,
                **({"multiprocessing_context": "spawn"} if config.loader_workers else {}))
        except (TypeError, ValueError) as exc:
            raise ModelContractError("loader source and collator must be picklable for worker processes") from exc

    def measured(batch):
        batch = move(batch, device) if isinstance(batch, dict) else [move(v, device) for v in batch]
        x, target = batch[:2]
        with torch.autocast(device_type=device.type, dtype=dtype, enabled=enabled):
            prediction = net(x)
        if structured:
            if not isinstance(prediction, dict) or set(prediction) != set(target):
                raise ModelContractError("structured architecture must return declared named target tensors")
            for name in target:
                if prediction[name].shape != target[name].shape:
                    raise ModelContractError(f"output {name!r} shape differs from declared target")
            prediction = {k: v.float() for k, v in prediction.items()}
            context_inputs = {**x, "__target_masks__": batch[2]}
            result = loss_fn(prediction, target, context_inputs)
            n = len(next(iter(target.values())))
        else:
            result = loss_fn(prediction.float(), target, x)
            n = len(x)
        components = dict(result) if isinstance(result, dict) else {"loss": result}
        aggregation = components.pop("__aggregation__", None)
        if aggregation is not None:
            if set(aggregation) != set(components) - {"loss"}:
                raise ModelContractError("loss aggregation must describe every non-total component")
            import math
            for weight, coefficient in aggregation.values():
                if not math.isfinite(weight) or weight < 0 or not math.isfinite(coefficient):
                    raise ModelContractError("loss aggregation needs finite nonnegative counts and finite coefficients")
        if "loss" not in components:
            raise ModelContractError("composite loss mapping needs a differentiable 'loss' total")
        for name, value in components.items():
            if not torch.is_tensor(value) or value.ndim != 0 or not torch.isfinite(value):
                raise ModelContractError(f"loss component {name!r} must be a finite scalar tensor")
        return components, n, aggregation

    def accumulate(totals, counts, coefficients, components, n, aggregation):
        for name, value in components.items():
            if aggregation is not None and name == "loss":
                continue
            weight, coefficient = (n, None) if aggregation is None else aggregation[name]
            if name in coefficients and coefficients[name] != coefficient:
                raise ModelContractError("loss aggregation coefficients must remain constant across batches")
            coefficients[name] = coefficient
            totals[name] = totals.get(name, 0.0) + float(value.detach()) * weight
            counts[name] = counts.get(name, 0.0) + weight

    def record(prefix, totals, counts, coefficients):
        values = {name: value / counts[name] if counts[name] else 0.0 for name, value in totals.items()}
        if "loss" not in values:
            values["loss"] = sum(values[name] * coefficients[name] for name in values)
        for name, value in values.items():
            history.setdefault(prefix + name, []).append(value)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    end_epoch = min(config.epochs, start_epoch + (config.max_epochs_this_run or config.epochs))
    with _deterministic(torch, strict=device.type == "cpu"):
        for epoch in range(start_epoch, end_epoch):
            if config.early_stopping_patience is not None and stale >= config.early_stopping_patience:
                break
            net.train()
            totals, counts, coefficients = {}, {}, {}
            current = epoch_data(rng) if epoch_data is not None else train_data
            for batch in loader(current, True):
                optimizer.zero_grad()
                components, n, aggregation = measured(batch)
                scaler.scale(components["loss"]).backward()
                scaler.unscale_(optimizer)
                for parameter in net.parameters():
                    if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
                        raise ModelContractError(f"nonfinite gradient at epoch {epoch}, step {step}")
                if config.max_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(net.parameters(), config.max_grad_norm, error_if_nonfinite=True)
                scaler.step(optimizer)
                scaler.update()
                if scheduler is not None:
                    scheduler.step()
                step += 1
                accumulate(totals, counts, coefficients, components, n, aggregation)
            record("train_", totals, counts, coefficients)
            history.setdefault("learning_rate", []).append(float(optimizer.param_groups[0]["lr"]))
            monitored = history["train_loss"][-1]
            net.eval()
            if has_val:
                totals, counts, coefficients = {}, {}, {}
                with torch.no_grad():
                    for batch in loader(val_data, False):
                        components, n, aggregation = measured(batch)
                        accumulate(totals, counts, coefficients, components, n, aggregation)
                record("validation_", totals, counts, coefficients)
                monitored = history["validation_loss"][-1]
            if monitored < best_loss:
                best_loss, best_epoch, stale = monitored, epoch, 0
                best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
            else:
                stale += 1
            if config.checkpoint_path:
                _save_checkpoint(torch, config.checkpoint_path, {"checkpoint_schema": 1, "binding": binding,
                    "state_dict": {k: v.detach().cpu() for k, v in net.state_dict().items()},
                    "optimizer": optimizer.state_dict(), "scheduler": None if scheduler is None else scheduler.state_dict(),
                    "scaler": scaler.state_dict(), "torch_rng": torch.get_rng_state(),
                    "cuda_rng": torch.cuda.get_rng_state_all() if device.type == "cuda" else [],
                    "loader_rng": generator.get_state(), "numpy_rng": rng.bit_generator.state,
                    "history": copy.deepcopy(history), "next_epoch": epoch + 1, "step": step,
                    "best_loss": best_loss, "best_epoch": best_epoch, "best_state": best_state, "stale": stale})
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    resources = {"fit_seconds": time.perf_counter() - started, "device": str(device),
        "precision": config.precision, "optimizer_steps": step,
        "peak_device_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None}
    if best_state is None:
        raise ModelContractError("no finite best training state was produced")
    net.load_state_dict(best_state)
    try:
        import resource
        import sys
        resources["process_peak_rss_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == "darwin" else 1024)
    except ImportError:
        resources["process_peak_rss_bytes"] = None
    return history, best_epoch, resources
