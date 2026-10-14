"""Checkpoint bundle save/load for restore verification."""

import os

import numpy as np
import torch

from .common import write_json


def clone_full_state(model):
    return {
        k: v.detach().to("cpu", copy=True) for k, v in model.state_dict().items()
    }


def adapter_state(core):
    return {
        k: v.detach().to("cpu", copy=True)
        for k, v in core.state_dict().items()
        if not k.startswith("vision_model.")
    }


def save_val_point(out_dir, model_base, step, val_mse, preds=None, targets=None,
                   window_border=None):
    """Save an independent clone of the current model state for one val point."""
    import json

    core = model_base._core_model()
    tag = f"step{step}"
    full = clone_full_state(model_base.model)
    full_path = os.path.join(out_dir, f"full_model_{tag}.pt")
    torch.save(full, full_path)
    entry = {
        "step": step,
        "val_mse_fused": float(val_mse),
        "full_state_path": full_path,
        "full_state_bytes": os.path.getsize(full_path),
        "adapter_state": adapter_state(core),
        "scaler_mean": model_base.scaler.mean_.tolist(),
        "scaler_scale": model_base.scaler.scale_.tolist(),
    }
    if preds is not None:
        npz_path = os.path.join(out_dir, f"val_preds_{tag}.npz")
        np.savez_compressed(
            npz_path,
            preds=preds,
            targets=targets,
            window_border=window_border,
        )
        entry["preds_path"] = npz_path
        entry["preds_shape"] = list(preds.shape)
    meta = {
        k: v for k, v in entry.items() if k != "adapter_state"
    }
    with open(os.path.join(out_dir, f"val_point_{tag}.json"), "w") as f:
        json.dump(meta, f, indent=2, sort_keys=True)
    return entry


def write_manifest(out_dir, manifest):
    write_json(os.path.join(out_dir, "manifest.json"), manifest)


def load_full_state(path):
    return torch.load(path, map_location="cpu", weights_only=True)
