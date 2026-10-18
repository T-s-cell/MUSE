"""Independent-process strict restore probe (EXP-029 preflight B4).

Builds a fresh MUSE, configures the requested geometry, restores the
given ckpt strictly (with MAE backbone backfill), predicts the first 8
val windows, and writes pred npz + sidecar json carrying the live
model's packed_sha256 (must equal the ckpt meta's packed_sha256 —
restored params bitwise == ckpt) and the ckpt npz sha.
"""
import argparse

import numpy as np
import torch

from repro_timemmd_common_v1 import protocol as P
from repro_timemmd_common_v1.adapter import build_muse, configure_muse, \
    forward_fused
from repro_timemmd_common_v1.common import dump_json, load_json, \
    setup_deterministic_env, sha256_of
from repro_timemmd_common_v1.data_timemmd import DomainData
from repro_timemmd_common_v1.muse_train import npz_key, packed_sha, \
    restore_muse_ckpt_strict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dom", required=True)
    ap.add_argument("--h", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--meta", required=True)
    args = ap.parse_args()

    setup_deterministic_env()
    P.verify_frozen()
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = build_muse(load_ckpt=True)
    configure_muse(model, args.dom, args.h)
    model = model.to(device)
    meta = restore_muse_ckpt_strict(model, args.ckpt,
                                    load_json(args.meta))
    live = {npz_key(n): t.detach().cpu().numpy()
            for n, t in list(model.named_parameters())
            + list(model.named_buffers())}
    live_packed = packed_sha(live)
    import json
    splits = json.load(open(P.protocol_dir() / "splits.json"))
    dd = DomainData(P.protocol_dir(), splits, args.dom)
    spv = dd.load_split(args.h, "val")
    xb = torch.from_numpy(np.ascontiguousarray(spv["X"][:8])).to(device)
    pred = forward_fused(model, xb).cpu().numpy()
    np.savez(args.out, pred=pred.astype(np.float32))
    dump_json(str(args.out) + ".json", {
        "gid": meta["gid"], "live_packed_sha256": live_packed,
        "ckpt_packed_sha256": meta["packed_sha256"],
        "ckpt_npz_sha256": meta["npz_sha256"],
        "npz_sha_of_out": sha256_of(args.out),
        "pred_shape": list(pred.shape)})


if __name__ == "__main__":
    main()
