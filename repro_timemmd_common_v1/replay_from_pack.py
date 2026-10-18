"""Replay all 36 groups FROM THE DELIVERED AUDIT PACK: strict
adapter-only restore (restore_muse_ckpt_pack, fixed MAE backfill) ->
recompute test AND best-epoch val predictions at the export batch
(P.INFER_BATCH=64) AND at the protocol batch (32) -> element-wise
comparison against the cached original arrays at atol/rtol -> metrics
recomputed separately per variant (f32 main + f64 audit). Original
files are never written; only reports land in out_root/replay_pack/,
and the cached npz hashes are snapshotted before/after.

Driver: extracts the ckpt pack tars first (see README_rerun); this
module only reads the extracted stage. Exit 2 on any mismatch.
"""
import argparse
from pathlib import Path

import numpy as np

from repro_timemmd_common_v1 import protocol as P
from repro_timemmd_common_v1.adapter import build_muse, configure_muse, \
    predict_split
from repro_timemmd_common_v1.common import bitwise_equal, dump_json, \
    identity_equal, load_json, pred_close, setup_deterministic_env, \
    sha256_of
from repro_timemmd_common_v1.data_timemmd import DomainData
from repro_timemmd_common_v1.muse_train import restore_muse_ckpt_pack


def variant_metrics(pred, truth):
    d32 = (pred - truth).astype(np.float32)
    p64, t64 = pred.astype(np.float64), truth.astype(np.float64)
    d64 = p64 - t64
    return {"mse32": float(np.mean(d32 ** 2)),
            "mae32": float(np.mean(np.abs(d32))),
            "mse64": float(np.mean(d64 ** 2)),
            "mae64": float(np.mean(np.abs(d64)))}


def run_group(dom, h, pack_roots, out_dir, atol, rtol, batches):
    setup_deterministic_env()
    pinfo = P.verify_frozen()
    out_root = P.out_root()
    gid = P.group_id(dom, h)

    npz_path = meta_path = lite_path = None
    for root in pack_roots:
        c1 = root / P.ckpt_npz_name(dom, h)
        c2 = root / P.ckpt_meta_name(dom, h)
        c3 = root / ("adapter_meta__%s.json" % gid)
        if c1.exists() and c2.exists() and c3.exists():
            npz_path, meta_path, lite_path = c1, c2, c3
            break
    if npz_path is None:
        raise SystemExit("[FATAL] pack files missing for %s in %s"
                         % (gid, [str(r) for r in pack_roots]))

    test_path = out_root / "runs" / gid / P.test_npz_name(dom, h)
    val_path = out_root / P.val_npz_name(dom, h)
    sha_test_pre, sha_val_pre = sha256_of(test_path), sha256_of(val_path)

    model = build_muse(load_ckpt=True)
    configure_muse(model, dom, h)
    model = model.to("cuda:0")
    meta = load_json(meta_path)
    lite = load_json(lite_path)
    restore_muse_ckpt_pack(model, npz_path, meta, lite)

    base_test = dict(np.load(str(test_path)))
    base_val = dict(np.load(str(val_path)))
    dd = DomainData(pinfo["protocol_dir"], pinfo["splits"], dom)
    sp = dd.load_split(h, "test")
    sv = dd.load_split(h, "val")

    ident = all(
        identity_equal(a["window_index"], b["window_index"])
        and bitwise_equal(a["truth"], b["truth"])
        and bitwise_equal(a["s"], b["s"])
        and bitwise_equal(a["orig_row"], b["orig_row"])
        and bitwise_equal(a["label_dates"], b["label_dates"])
        for a, b in ((sp, base_test), (sv, base_val)))

    variants = {}
    for tag, X, cached in (("test", sp["X"], base_test),
                           ("val", sv["X"], base_val)):
        for b in batches:
            pred = predict_split(model, X, batch=b)
            close, max_abs, max_viol = pred_close(pred, cached["pred"],
                                                  atol, rtol)
            m = variant_metrics(pred, cached["truth"].reshape(
                pred.shape))
            m.update({"batch": int(b), "close": bool(close),
                      "max_abs_diff": max_abs,
                      "max_tol_violation": max_viol,
                      "bitwise_pred": bool(np.array_equal(
                          pred, cached["pred"].reshape(pred.shape)))})
            variants["%s_b%d" % (tag, b)] = m

    immut = (sha_test_pre == sha256_of(test_path)
             and sha_val_pre == sha256_of(val_path))
    all_close = all(v["close"] for v in variants.values())
    report = {
        "gid": gid, "domain": dom, "h": int(h),
        "identity_ok": bool(ident), "immutability_ok": bool(immut),
        "adapter_npz_sha256": lite["adapter_npz_sha256"],
        "packed_sha256_adapter": lite["packed_sha256"],
        "atol": atol, "rtol": rtol, "variants": variants,
        "all_pass": bool(all_close and ident and immut),
    }
    dump_json(out_dir / ("replay_pack__%s.json" % gid), report)
    worst = max(v["max_abs_diff"] for v in variants.values())
    print("PACK_REPLAY %s %s ident=%s immut=%s worst=%.3e b64_bitwise="
          "%s/%s b32=%s/%s"
          % (gid, "PASS" if report["all_pass"] else "FAIL", ident,
             immut, worst,
             variants["test_b64"]["bitwise_pred"],
             variants["val_b64"]["bitwise_pred"],
             variants["test_b32"]["close"],
             variants["val_b32"]["close"]), flush=True)
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack_roots", required=True,
                    help="comma-separated extracted pack dirs")
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--protocol_dir", default=None)
    ap.add_argument("--out_root", default=None)
    ap.add_argument("--batches", default="%d,%d" % (P.INFER_BATCH,
                                                    P.BATCH))
    ap.add_argument("--groups", default=None)
    args = ap.parse_args()

    setup_deterministic_env()
    P.verify_frozen(args.protocol_dir)
    if args.out_root:
        import os
        os.environ["MUSE_TIMEMMD_S2021_OUT"] = args.out_root
    out_root = P.out_root()
    out_dir = out_root / "replay_pack"
    out_dir.mkdir(parents=True, exist_ok=True)
    roots = [r for r in (Path(x) for x in args.pack_roots.split(","))]

    batches = [int(x) for x in args.batches.split(",")]
    groups = P.all_groups()
    if args.groups:
        want = set(args.groups.split(","))
        groups = [g for g in groups if P.group_id(*g) in want]

    reports = {}
    n_fail = 0
    for dom, h in groups:
        rep = run_group(dom, h, roots, out_dir, P.ATOL, P.RTOL, batches)
        reports[rep["gid"]] = rep
        if not rep["all_pass"]:
            n_fail += 1

    n_b32_close = sum(1 for r in reports.values()
                      if r["variants"]["test_b32"]["close"]
                      and r["variants"]["val_b32"]["close"])
    n_b64_bitwise = sum(1 for r in reports.values()
                        if r["variants"]["test_b64"]["bitwise_pred"]
                        and r["variants"]["val_b64"]["bitwise_pred"])
    summary = {
        "n_groups": len(groups), "n_pass": len(groups) - n_fail,
        "n_b64_bitwise": n_b64_bitwise, "n_b32_close": n_b32_close,
        "batches": batches,
        "note": "original export batch=%d; batch=%d merged per review; "
                "all compared vs cached originals at atol=%g/rtol=%g; "
                "metrics recomputed per variant (f32 main + f64 audit)"
                % (P.INFER_BATCH, P.BATCH, P.ATOL, P.RTOL),
        "reports": reports,
    }
    dump_json(out_dir / "replay_pack_summary.json", summary)
    print("PACK_REPLAY: %d/%d pass, b64_bitwise=%d, b32_close=%d"
          % (summary["n_pass"], len(groups), n_b64_bitwise, n_b32_close))
    if n_fail:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
