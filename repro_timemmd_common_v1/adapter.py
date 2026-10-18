"""Single entry point to the MUSE model for EXP-029.

The muse package is loaded through an isolated importlib namespace
("muse_core_isolated") so that none of the ts_benchmark __init__ chain
(sklearn / matplotlib / reformer_pytorch / TFB) is pulled in. The model
class, its geometry (update_config), freezing, and forward paths are all
accessed only through this module.
"""
import importlib.util
import sys
import types

import numpy as np
import torch

from repro_timemmd_common_v1 import protocol as P
from repro_timemmd_common_v1.common import backbone_digest

_CORE = "muse_core_isolated"
_muse_cls_cache = None


def load_muse_class():
    """Load ts_benchmark/baselines/muse/models/model.py::MUSE under an
    isolated synthetic package (relative imports intact, ts_benchmark
    never imported). Cached per process."""
    global _muse_cls_cache
    if _muse_cls_cache is not None:
        return _muse_cls_cache
    base = P.REPO_ROOT / "ts_benchmark" / "baselines" / "muse"
    pkg = types.ModuleType(_CORE)
    pkg.__path__ = [str(base)]
    sys.modules[_CORE] = pkg
    for nm, sub in (("models", base / "models"),
                    ("layers", base / "layers"),
                    ("utils", base / "utils")):
        m = types.ModuleType(_CORE + "." + nm)
        m.__path__ = [str(sub)]
        sys.modules[_CORE + "." + nm] = m
        setattr(pkg, nm, m)
    spec = importlib.util.spec_from_file_location(
        _CORE + ".models.model", str(base / "models" / "model.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    _muse_cls_cache = mod.MUSE
    return _muse_cls_cache


def build_muse(load_ckpt=True, device=None):
    """Fresh MUSE with frozen protocol constants. vision_model is strict-
    loaded from the pinned MAE checkpoint and frozen inside __init__.
    Geometry is NOT set here (update_config must happen exactly once,
    via configure_muse, before .to(device)/optimizer construction)."""
    cls = load_muse_class()
    model = cls(arch=P.ARCH, ckpt_path=str(P.mae_weights_path()),
                load_ckpt=load_ckpt, num_latents=1, latent_dim=192,
                adapter_num_heads=4, channel_depth=1)
    model.eval()
    if device is not None:
        model.to(device)
    return model


def configure_raw(model, context_len, pred_len, periodicity):
    """update_config exactly once per model instance. A counting shim
    stays attached forever: any second call (which would rebuild TPR and
    reset tp_residual_logits) is a hard error."""
    if getattr(model, "_exp029_config_count", None) is None:
        bound = model.update_config

        def guarded(*a, **k):
            model._exp029_config_count = model._exp029_config_count + 1
            if model._exp029_config_count > 1:
                raise RuntimeError(
                    "update_config called more than once: TPR re-init "
                    "hazard (tp_residual_logits reset)")
            return bound(*a, **k)

        model._exp029_config_count = 0
        model.update_config = guarded
    model.update_config(
        context_len=context_len, pred_len=int(pred_len),
        periodicity=periodicity, norm_const=P.NORM_CONST,
        align_const=P.ALIGN_CONST, interpolation=P.INTERPOLATION)
    return model


def configure_muse(model, domain, h):
    return configure_raw(model, P.TABLE[domain]["L"], int(h),
                         P.FIXED_P[domain])


def trainable_inventory(model):
    """Assert the frozen trainable inventory: 111 tensors, expected
    buffers (7), backbone fully frozen. Returns the name lists."""
    trainable = sorted(name for name, p in model.named_parameters()
                       if p.requires_grad)
    frozen = sorted(name for name, p in model.named_parameters()
                    if not p.requires_grad)
    buffers = sorted(name for name, _ in model.named_buffers())
    if len(trainable) != P.EXPECTED_TRAINABLE_TENSORS:
        raise RuntimeError("trainable tensors %d != %d"
                           % (len(trainable), P.EXPECTED_TRAINABLE_TENSORS))
    if len(buffers) != P.EXPECTED_BUFFERS:
        raise RuntimeError("buffers %d != %d"
                           % (len(buffers), P.EXPECTED_BUFFERS))
    if any(not n.startswith("vision_model.") for n in frozen):
        raise RuntimeError("non-backbone tensor frozen: %s"
                           % [n for n in frozen
                              if not n.startswith("vision_model.")])
    if any(n.startswith("vision_model.") for n in trainable):
        raise RuntimeError("backbone tensor trainable: %s"
                           % [n for n in trainable
                              if n.startswith("vision_model.")])
    for dead in P.DEAD_TENSORS:
        if dead not in trainable:
            raise RuntimeError("dead tensor missing from trainable "
                               "inventory: %s" % dead)
    n_scalars = sum(int(p.numel()) for n, p in model.named_parameters()
                    if p.requires_grad)
    return {"trainable_names": trainable, "frozen_names": frozen,
            "buffer_names": buffers, "n_trainable_tensors": len(trainable),
            "n_trainable_scalars": n_scalars}


def freeze_check_muse(model, inv=None):
    """Post-.to(device)/pre-optimizer gate: inventory + backbone digest
    binding (returned for comparison against ckpt-time digests)."""
    inv = inv or trainable_inventory(model)
    return {"inventory": inv,
            "backbone_sha256": backbone_digest(model)}


def forward_fused(model, x, fp64=False):
    """Eval fused prediction. x: torch [B, L, C] on model device."""
    was_training = model.training
    model.eval()
    try:
        with torch.inference_mode():
            out = model(x, export_image=False, fp64=fp64,
                        use_variable_chunk=P.USE_VARIABLE_CHUNK,
                        return_branches=False)
    finally:
        if was_training:
            model.train()
    return out


def forward_branches(model, x, fp64=False):
    """Grad-path forward returning {output, channel, tp} (all
    denormalized). Keeps caller-controlled mode; use under train()."""
    return model(x, export_image=False, fp64=fp64,
                 use_variable_chunk=P.USE_VARIABLE_CHUNK,
                 return_branches=True)


def predict_split(model, X, batch=None, fp64=False):
    """Full-split fused inference over numpy windows X [N, L, C] f32.
    Returns numpy preds [N, H, C] f32. Sequential slicing, manifest
    order preserved."""
    batch = int(batch or P.INFER_BATCH)
    n = X.shape[0]
    preds = []
    model.eval()
    with torch.inference_mode():
        for lo, hi in [(i, min(i + batch, n))
                       for i in range(0, n, batch)]:
            xb = torch.from_numpy(X[lo:hi]).to(
                next(model.parameters()).device)
            out = model(xb, export_image=False, fp64=fp64,
                        use_variable_chunk=P.USE_VARIABLE_CHUNK,
                        return_branches=False)
            preds.append(out.detach().to("cpu", torch.float32).numpy())
    return np.concatenate(preds, axis=0)
