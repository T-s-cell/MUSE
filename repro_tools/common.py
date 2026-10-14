"""Shared helpers: env config, RNG save/restore, hashing, fingerprints."""

import hashlib
import json
import os
import random

import numpy as np
import torch


def _parse_steps(text):
    if not text:
        return set()
    return {int(x) for x in text.split(",") if x.strip()}


class ReproConfig:
    def __init__(self):
        self.enable = os.environ.get("MUSE_REPRO_ENABLE", "") == "1"
        self.out_dir = os.environ.get(
            "MUSE_REPRO_OUT_DIR", os.path.join("repro_out", "run")
        )
        self.max_steps = int(os.environ.get("MUSE_REPRO_MAX_STEPS", "0") or 0)
        self.val_at = _parse_steps(os.environ.get("MUSE_REPRO_VAL_AT", ""))
        self.audit_steps = _parse_steps(
            os.environ.get("MUSE_REPRO_AUDIT_STEPS", "")
        )
        self.log_every = max(1, int(os.environ.get("MUSE_REPRO_LOG_EVERY", "1") or 1))
        self.save_full = os.environ.get("MUSE_REPRO_SAVE_FULL", "1") == "1"
        self.input_hash = os.environ.get("MUSE_REPRO_INPUT_HASH", "") == "1"
        self.warmup_steps = int(os.environ.get("MUSE_REPRO_WARMUP", "10") or 10)
        self.dump_preds = os.environ.get("MUSE_REPRO_DUMP_PREDS", "1") == "1"
        self.param_sha = os.environ.get("MUSE_REPRO_PARAM_SHA", "") == "1"
        self.force_deterministic = (
            os.environ.get("MUSE_REPRO_FORCE_DETERMINISTIC", "") == "1"
        )


def save_rng():
    state = {
        "torch": torch.get_rng_state(),
        "numpy": np.random.get_state(),
        "python": random.getstate(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng(state):
    torch.set_rng_state(state["torch"].cpu())
    np.random.set_state(state["numpy"])
    random.setstate(state["python"])
    if "cuda" in state:
        torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda"]])


def tensor_sha256(tensor):
    t = tensor.detach()
    if t.device.type != "cpu":
        t = t.cpu()
    t = t.contiguous()
    h = hashlib.sha256()
    try:
        h.update(t.flatten().view(torch.uint8).numpy().tobytes())
    except RuntimeError:
        h.update(t.flatten().to(torch.float64).numpy().tobytes())
    return h.hexdigest()


def module_fingerprint(module):
    fp = {}
    for name, p in module.named_parameters():
        fp["param:" + name] = tensor_sha256(p)
    for name, b in module.named_buffers():
        fp["buffer:" + name] = tensor_sha256(b)
    return fp


def fingerprint_diff(fp_before, fp_after):
    changed = [k for k in fp_before if fp_before[k] != fp_after.get(k)]
    missing = [k for k in fp_before if k not in fp_after]
    added = [k for k in fp_after if k not in fp_before]
    return {"changed": changed, "missing": missing, "added": added}


def trainable_checksum(model):
    h = hashlib.sha256()
    for name, p in model.named_parameters():
        if p.requires_grad:
            h.update(name.encode())
            h.update(tensor_sha256(p).encode())
    return h.hexdigest()


def batch_sha256(tensor):
    return tensor_sha256(tensor)


def grad_snapshot(model):
    return {
        name: (p.grad.detach().to("cpu", copy=True) if p.grad is not None else None)
        for name, p in model.named_parameters()
        if p.requires_grad
    }


def grad_max_abs(snapshot):
    out = {}
    for name, g in snapshot.items():
        out[name] = 0.0 if g is None else float(g.abs().max().item())
    return out


def json_safe(obj):
    if isinstance(obj, dict):
        return {str(k): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    if isinstance(obj, (torch.Tensor,)):
        return tensor_sha256(obj)
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return repr(obj)


def write_json(path, obj):
    with open(path, "w") as f:
        json.dump(json_safe(obj), f, indent=2, sort_keys=True, default=str)
