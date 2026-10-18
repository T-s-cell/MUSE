"""Per-group TEST inference as an independent process: strict ckpt
restore + MAE backbone backfill -> batch=64 original-order test
inference -> test npz (pred, truth, s, orig_row, label_dates,
window_index) + metrics json.

Driver: run_infer.py (re-execs this module per group). Metrics json is
sha-pinned into test_status.json by the driver and re-checked by
verify_cache.py. No digest_pre != digest_post requirement: acceptance is
restored params bitwise-equal to the ckpt (inside
restore_muse_ckpt_strict).
"""
import argparse
import time

import numpy as np

from repro_timemmd_common_v1 import protocol as P
from repro_timemmd_common_v1.adapter import build_muse, configure_muse, \
    predict_split
from repro_timemmd_common_v1.common import dump_json, load_json, \
    setup_deterministic_env, sha256_of
from repro_timemmd_common_v1.data_timemmd import DomainData
from repro_timemmd_common_v1.muse_train import restore_muse_ckpt_strict


def metrics(pred, truth):
    d = pred.astype(np.float64) - truth.astype(np.float64)
    return float(np.mean(d * d)), float(np.mean(np.abs(d)))


def run_group(dom, h):
    setup_deterministic_env()
    pinfo = P.verify_frozen()
    out_dir = P.out_root()
    gid = P.group_id(dom, h)
    gdir = out_dir / "runs" / gid
    gdir.mkdir(parents=True, exist_ok=True)
    npz_path = gdir / P.test_npz_name(dom, h)
    mjson = gdir / ("metrics__%s.json" % gid)

    model = build_muse(load_ckpt=True).to("cuda:0")
    configure_muse(model, dom, h)
    meta = restore_muse_ckpt_strict(
        model, out_dir / P.ckpt_npz_name(dom, h),
        load_json(out_dir / P.ckpt_meta_name(dom, h)))

    dd = DomainData(pinfo["protocol_dir"], pinfo["splits"], dom)
    sp = dd.load_split(h, "test")
    t0 = time.time()
    pred = predict_split(model, sp["X"], batch=P.INFER_BATCH)
    elapsed = time.time() - t0
    mse, mae = metrics(pred, sp["Y"])
    np.savez(str(npz_path), pred=pred.astype(np.float32),
             truth=sp["Y"].astype(np.float32),
             s=sp["s"], orig_row=sp["orig_row"],
             label_dates=sp["label_dates"],
             window_index=sp["window_index"])
    dump_json(mjson, {
        "gid": gid, "domain": dom, "h": int(h),
        "p": int(P.FIXED_P[dom]), "evaluated_windows": int(sp["X"].shape[0]),
        "mse": mse, "mae": mae,
        "npz_sha256": sha256_of(npz_path),
        "ckpt_npz_sha256": meta["npz_sha256"],
        "packed_sha256": meta["packed_sha256"],
        "backbone_sha256": meta["backbone_sha256"],
        "weights_sha256": pinfo["weights_sha256"],
        "elapsed_s": round(elapsed, 3)})
    print("INFER OK %s mse=%.10f mae=%.10f (%d windows, %.1fs)"
          % (gid, mse, mae, sp["X"].shape[0], elapsed))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dom", required=True)
    ap.add_argument("--h", type=int, required=True)
    args = ap.parse_args()
    run_group(args.dom, args.h)
