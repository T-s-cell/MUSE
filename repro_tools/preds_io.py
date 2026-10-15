"""Sharded save/load for prediction arrays (validation AND test dumps).

Large dumps (long horizons: N x H x C arrays for preds+targets) are split
into contiguous window shards so every shard fits the audit-package budget;
the loader reassembles them and asserts full window coverage (no gaps, no
duplicates). Single-file npz stays the common case for small dumps.
"""

import hashlib
import json
import os

import numpy as np

SHARD_THRESHOLD_BYTES = 40 * 1024 * 1024
SHARDS_SCHEMA = "muse-repro/preds-shards-v1"


def _shard_bytes(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def save_preds_sharded(out_dir, stem, preds, targets, window_border,
                       threshold_bytes=SHARD_THRESHOLD_BYTES):
    """Save preds/targets/window_border as stem.npz, or shard by windows.

    Returns (primary_path, index_dict_or_None). primary_path is the single
    npz, or the *_shards.json index when sharded.
    """
    preds = np.asarray(preds)
    targets = np.asarray(targets)
    borders = np.asarray(window_border, dtype=np.int64)
    n = int(len(borders))
    per_row = preds[0].nbytes + targets[0].nbytes + borders.itemsize * 2
    os.makedirs(out_dir, exist_ok=True)

    if n * per_row <= threshold_bytes:
        path = os.path.join(out_dir, f"{stem}.npz")
        np.savez_compressed(path, preds=preds, targets=targets,
                            window_border=borders)
        return path, None

    rows_per_shard = max(1, threshold_bytes // per_row)
    shards = []
    k_total = (n + rows_per_shard - 1) // rows_per_shard
    width = max(3, len(str(k_total)))
    for k in range(k_total):
        lo = k * rows_per_shard
        hi = min(n, lo + rows_per_shard)
        fname = f"{stem}.shard{k + 1:0{width}d}of{k_total}.npz"
        fpath = os.path.join(out_dir, fname)
        np.savez_compressed(
            fpath, preds=preds[lo:hi], targets=targets[lo:hi],
            window_border=borders[lo:hi],
        )
        shards.append({
            "file": fname,
            "first_row": lo,
            "n_windows": hi - lo,
            "border_first": int(borders[lo]),
            "border_last": int(borders[hi - 1]),
            "sha256": _shard_bytes(fpath),
        })
    index = {
        "schema": SHARDS_SCHEMA,
        "stem": stem,
        "n_windows_total": n,
        "preds_shape_full": list(preds.shape),
        "targets_shape_full": list(targets.shape),
        "global_border_first": int(borders[0]),
        "global_border_last": int(borders[-1]),
        "threshold_bytes": int(threshold_bytes),
        "shards": shards,
    }
    ipath = os.path.join(out_dir, f"{stem}_shards.json")
    with open(ipath, "w") as f:
        json.dump(index, f, indent=2, sort_keys=True)
    return ipath, index


def check_coverage(borders):
    """Contiguity + strict monotonicity of a border array (stride-1 grids)."""
    b = np.asarray(borders, dtype=np.int64)
    if len(b) < 1:
        return False, "empty"
    if not np.all(np.diff(b) >= 1):
        return False, "not strictly increasing"
    contiguous = int(b[-1]) - int(b[0]) + 1 == len(b)
    return contiguous, ("contiguous" if contiguous else "gaps present")


def load_preds_any(location, stem):
    """Load preds/targets/window_border saved by save_preds_sharded.

    location: directory containing either <stem>.npz or <stem>_shards.json.
    Returns dict(preds, targets, window_border, sharded, coverage_ok,
    coverage_detail, index_or_None). Raises if neither file exists.
    """
    single = os.path.join(location, f"{stem}.npz")
    index_path = os.path.join(location, f"{stem}_shards.json")
    if os.path.exists(single):
        with np.load(single) as z:
            out = {
                "preds": z["preds"], "targets": z["targets"],
                "window_border": z["window_border"],
                "sharded": False, "index": None,
            }
    elif os.path.exists(index_path):
        with open(index_path) as f:
            index = json.load(f)
        if index.get("schema") != SHARDS_SCHEMA:
            raise ValueError(f"unknown shard schema in {index_path}")
        parts_p, parts_t, parts_b = [], [], []
        for s in sorted(index["shards"], key=lambda s: s["first_row"]):
            with np.load(os.path.join(location, s["file"])) as z:
                parts_p.append(z["preds"])
                parts_t.append(z["targets"])
                parts_b.append(z["window_border"])
        out = {
            "preds": np.concatenate(parts_p),
            "targets": np.concatenate(parts_t),
            "window_border": np.concatenate(parts_b),
            "sharded": True, "index": index,
        }
        if len(out["window_border"]) != index["n_windows_total"]:
            raise ValueError(
                f"shard row total {len(out['window_border'])} != "
                f"index n_windows_total {index['n_windows_total']}"
            )
    else:
        raise FileNotFoundError(
            f"no {stem}.npz or {stem}_shards.json under {location}"
        )
    ok, detail = check_coverage(out["window_border"])
    out["coverage_ok"] = ok
    out["coverage_detail"] = detail
    return out
