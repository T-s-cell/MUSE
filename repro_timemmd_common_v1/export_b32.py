"""Export batch=32 prediction arrays for all 36 groups (test AND
best-epoch val) from the EXISTING best weights, keeping the original
batch=64 files untouched. Per group: strict restore from the full
server ckpt (restore_muse_ckpt_strict, canonical weights) -> copy the
cached batch=64 npz dict and replace ONLY "pred" with the batch=32
prediction (identity arrays preserved bitwise) -> save under
out_root/b32_export/ + per-group metrics json (f32 main + f64 audit,
both splits, evaluate.metrics_from convention) with b32-vs-b64 array
deltas and pre/post sha snapshots proving the original files were never
written. Fresh model per group; verification independence is provided
by replay_b32.py (one subprocess per group from the delivered pack).
Exit 2 on any inconsistency.
"""
import argparse
from pathlib import Path

import numpy as np
import torch

from repro_timemmd_common_v1 import protocol as P
from repro_timemmd_common_v1.adapter import build_muse, configure_muse, \
    predict_split
from repro_timemmd_common_v1.common import bitwise_equal, dump_json, \
    load_json, setup_deterministic_env, sha256_of
from repro_timemmd_common_v1.data_timemmd import DomainData
from repro_timemmd_common_v1.evaluate import metrics_from
from repro_timemmd_common_v1.muse_train import restore_muse_ckpt_strict


def _2d(a):
    return a[:, :, 0] if a.ndim == 3 else a


def run_group(dom, h, pinfo, out_root):
    gid = P.group_id(dom, h)
    bdir = out_root / "b32_export"
    bdir.mkdir(parents=True, exist_ok=True)
    old_test = out_root / "runs" / gid / P.test_npz_name(dom, h)
    old_val = out_root / P.val_npz_name(dom, h)
    new_test = bdir / P.test_npz_name(dom, h)
    new_val = bdir / P.val_npz_name(dom, h)
    for f in (old_test, old_val):
        if not f.exists():
            raise SystemExit("[FATAL] cached b64 array missing: %s" % f)
    pre = {"test_b64_sha256": sha256_of(old_test),
           "val_b64_sha256": sha256_of(old_val)}
    if new_test.exists() or new_val.exists():
        raise SystemExit("[FATAL] b32 export already exists for %s — "
                         "remove out_root/b32_export first" % gid)

    model = build_muse(load_ckpt=True)
    configure_muse(model, dom, h)
    model = model.to("cuda:0")
    meta = load_json(out_root / P.ckpt_meta_name(dom, h))
    restore_muse_ckpt_strict(
        model, out_root / P.ckpt_npz_name(dom, h), meta)

    dd = DomainData(pinfo["protocol_dir"], pinfo["splits"], dom)
    base_test = dict(np.load(str(old_test)))
    base_val = dict(np.load(str(old_val)))
    sp = dd.load_split(h, "test")
    sv = dd.load_split(h, "val")
    for a, b in ((sp, base_test), (sv, base_val)):
        if not (np.array_equal(a["window_index"], b["window_index"])
                and bitwise_equal(a["Y"], b["truth"])):
            raise SystemExit("[FATAL] fresh split vs cached identity "
                             "mismatch %s" % gid)

    variants = {}
    for tag, X, base, newf in (("test", sp["X"], base_test, new_test),
                               ("val", sv["X"], base_val, new_val)):
        pred = predict_split(model, X, batch=P.BATCH)
        arr = dict(base)
        arr["pred"] = pred.reshape(base["pred"].shape).astype(np.float32)
        np.savez(str(newf), **arr)
        p2, t2 = _2d(pred), _2d(base["truth"])
        m = metrics_from(p2, t2)
        p0 = _2d(base["pred"])
        d = np.abs(p2.astype(np.float64) - p0.astype(np.float64))
        m0 = metrics_from(p0, t2)
        variants[tag] = {
            "batch": int(P.BATCH), "windows": int(p2.shape[0]),
            "npz_sha256": sha256_of(newf),
            "mse": m["mse"], "mae": m["mae"],
            "mse_f64": m["mse_f64"], "mae_f64": m["mae_f64"],
            "vs_b64": {
                "bitwise": bool(np.array_equal(
                    _2d(arr["pred"]), p0)),
                "max_abs_diff": float(d.max()),
                "mse_delta_f32": m["mse"] - m0["mse"],
                "mae_delta_f32": m["mae"] - m0["mae"]}}

    post = {"test_b64_sha256": sha256_of(old_test),
            "val_b64_sha256": sha256_of(old_val)}
    immut = pre == post
    rec = {"gid": gid, "domain": dom, "h": int(h),
           "batch": int(P.BATCH), "variants": variants,
           "originals_immutable": immut,
           "originals_sha256": post,
           "ckpt_npz_sha256": meta["npz_sha256"],
           "all_pass": bool(immut)}
    dump_json(bdir / ("metrics_b32__%s.json" % gid), rec)
    print("EXPORT_B32 %s %s test_bitwise=%s val_bitwise=%s "
          "max_abs=%.3e/%.3e"
          % (gid, "PASS" if immut else "FAIL",
             variants["test"]["vs_b64"]["bitwise"],
             variants["val"]["vs_b64"]["bitwise"],
             variants["test"]["vs_b64"]["max_abs_diff"],
             variants["val"]["vs_b64"]["max_abs_diff"]), flush=True)
    del model
    torch.cuda.empty_cache()
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--protocol_dir", default=None)
    ap.add_argument("--out_root", default=None)
    args = ap.parse_args()

    setup_deterministic_env()
    pinfo = P.verify_frozen(args.protocol_dir)
    if args.out_root:
        import os
        os.environ["MUSE_TIMEMMD_S2021_OUT"] = args.out_root
    out_root = P.out_root()

    recs = {}
    n_fail = 0
    for dom, h in P.all_groups():
        rec = run_group(dom, h, pinfo, out_root)
        recs[rec["gid"]] = rec
        if not rec["all_pass"]:
            n_fail += 1

    max_abs = max(v["vs_b64"]["max_abs_diff"]
                  for r in recs.values() for v in r["variants"].values())
    max_dmse = max(abs(v["vs_b64"]["mse_delta_f32"])
                   for r in recs.values() for v in r["variants"].values())
    n_bitwise = sum(1 for r in recs.values() for v in r["variants"].values()
                    if v["vs_b64"]["bitwise"])
    summary = {
        "n_groups": len(recs), "n_pass": len(recs) - n_fail,
        "batch": int(P.BATCH),
        "n_variants_bitwise_vs_b64": n_bitwise,
        "max_abs_diff_vs_b64": max_abs,
        "max_abs_mse_delta_vs_b64": max_dmse,
        "note": "batch=32 export of test + best-epoch val from the "
                "existing best weights; original batch=64 files kept "
                "and proven untouched (pre/post sha256); identity "
                "arrays copied bitwise from the b64 export",
        "groups": recs}
    dump_json(out_root / "b32_export" / "export_b32_summary.json",
              summary)
    print("EXPORT_B32: %d/%d pass, bitwise_vs_b64=%d/72, "
          "max_abs=%.3e, max_|dmse|=%.3e"
          % (summary["n_pass"], len(recs), n_bitwise, max_abs, max_dmse))
    if n_fail:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
