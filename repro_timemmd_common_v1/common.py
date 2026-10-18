"""Shared utilities: atomic json, deterministic env, digests, comparators,
guards. Comparators implement the user-corrected policy: predictions are
checked with atol/rtol (no cross-batch bitwise gate); identity/truth/weight
hashes remain exact.
"""
import hashlib
import json
import os
import platform
import random
import sys
import tempfile
from pathlib import Path

import numpy as np


def die(msg, code=1):
    print(msg, flush=True)
    sys.exit(code)


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def md5_of(path):
    h = hashlib.md5()
    h.update(Path(path).read_bytes())
    return h.hexdigest()


def dump_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    with os.fdopen(fd, "w") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)


def load_json(path):
    with open(path) as f:
        return json.load(f)


def setup_deterministic_env():
    """TF32 off, cudnn deterministic; CUBLAS workspace + HF offline must be
    set before CUDA/HF first use (wrapper also exports them)."""
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    import torch
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    import torch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def env_info():
    import torch
    info = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "hf_hub_offline": os.environ.get("HF_HUB_OFFLINE"),
        "numpy": np.__version__,
    }
    try:
        import timm
        info["timm"] = timm.__version__
    except Exception:
        info["timm"] = None
    try:
        import einops
        info["einops"] = einops.__version__
    except Exception:
        info["einops"] = None
    if torch.cuda.is_available():
        info["gpu"] = torch.cuda.get_device_name(0)
    return info


def model_state_digest(model):
    """Digest of ALL named_parameters (trainable + frozen backbone) —
    coarse whole-model identity."""
    h = hashlib.sha256()
    for name, prm in sorted(model.named_parameters()):
        h.update(name.encode())
        h.update(prm.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def backbone_digest(model):
    """Digest over vision_model.* parameters ONLY. The MAE backbone must be
    bit-identical to the loaded weights at every stage of this experiment
    (freeze forensics; must equal the digest taken right after load)."""
    h = hashlib.sha256()
    for name, prm in sorted(model.named_parameters()):
        if not name.startswith("vision_model."):
            continue
        h.update(name.encode())
        h.update(prm.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def named_tensor_sha(model):
    """Per-parameter sha256 map {name: hex} — fine-grained freeze/update
    forensics (which exact tensors moved after an update)."""
    out = {}
    for name, prm in sorted(model.named_parameters()):
        h = hashlib.sha256()
        h.update(prm.detach().cpu().numpy().tobytes())
        out[name] = h.hexdigest()
    return out


def buffer_digest(model):
    """Digest over sorted named_buffers (config-derived state; must be
    unchanged by training steps)."""
    h = hashlib.sha256()
    for name, buf in sorted(model.named_buffers()):
        h.update(name.encode())
        h.update(buf.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def all_finite(arr):
    a = np.asarray(arr)
    return bool(np.isfinite(a.astype(np.float64)).all())


def pred_close(reload_arr, base_arr, atol, rtol):
    """Elementwise |a-b| <= atol + rtol*|b|. Returns (close, max_abs,
    max_violation)."""
    a = reload_arr.astype(np.float64)
    b = base_arr.astype(np.float64)
    d = np.abs(a - b)
    tol = atol + rtol * np.abs(b)
    viol = d - tol
    return bool((viol <= 0).all()), float(d.max()), float(viol.max())


def identity_equal(a, b):
    """Exact equality for identity arrays (ints and strings)."""
    if a.shape != b.shape or a.dtype.kind != b.dtype.kind:
        return False
    return bool(np.array_equal(a, b))


def bitwise_equal(a, b):
    if a.shape != b.shape or a.dtype != b.dtype:
        return False
    if a.dtype.kind in "US":
        return bool(np.array_equal(a, b))
    return bool(np.array_equal(a.view(np.uint32), b.view(np.uint32)))


def core_guards(pinfo, batch, selected_p_sha=None, preflight_sha=None):
    """Tier-1 identity of the run: protocol pins, MAE weights, code md5s."""
    from repro_timemmd_common_v1 import protocol as P
    return {
        "protocol_name": P.PROTOCOL_NAME,
        "protocol_json_sha256": P.PROTOCOL_JSON_SHA256,
        "splits_json_sha256": P.SPLITS_JSON_SHA256,
        "scaler_json_sha256": P.SCALER_JSON_SHA256,
        "weights_sha256": pinfo["weights_sha256"],
        "code_md5": P.package_code_md5s(),
        "repo_commit": P.repo_commit(),
        "seed": P.SEED,
        "batch": batch,
        "selected_p_sha256": selected_p_sha,
        "preflight_report_sha256": preflight_sha,
        "env": {k: env_info()[k] for k in
                ("torch", "timm", "einops", "numpy", "python", "gpu")},
    }


def guards_equal(a, b):
    """Every tier-1 key must match exactly. Returns (equal, differing_keys)."""
    diff = []
    for k in sorted(set(a) | set(b)):
        if a.get(k) != b.get(k):
            diff.append(k)
    return (not diff), diff
