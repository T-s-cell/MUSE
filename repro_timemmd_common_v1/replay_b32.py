"""Independent batch=32 acceptance replay FROM THE DELIVERED AUDIT PACK.

Driver spawns ONE SUBPROCESS PER GROUP (fresh interpreter, fresh MAE
load); each worker strict-restores the adapter-only checkpoint
(restore_muse_ckpt_pack + fixed MAE backbone backfill), recomputes test
AND best-epoch val predictions at batch=32 (the same batch as the
export), and compares against the batch=32 export arrays: bitwise flag
+ pred_close(atol, rtol) + identity/truth bitwise + metric
recomputation from the replayed arrays (<=1e-9 vs the export metrics)
+ sha immutability of the export arrays. Reports land in
out_root/replay_b32/; the batch=64 originals are never touched.
Exit 2 on any failure.
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

from repro_timemmd_common_v1 import protocol as P
from repro_timemmd_common_v1.adapter import build_muse, configure_muse, \
    predict_split
from repro_timemmd_common_v1.common import bitwise_equal, dump_json, \
    identity_equal, load_json, pred_close, setup_deterministic_env, \
    sha256_of
from repro_timemmd_common_v1.data_timemmd import DomainData
from repro_timemmd_common_v1.evaluate import metrics_from
from repro_timemmd_common_v1.muse_train import restore_muse_ckpt_pack


def _2d(a):
    return a[:, :, 0] if a.ndim == 3 else a


def run_group(dom, h, roots, out_dir, out_root, atol, rtol):
    setup_deterministic_env()
    pinfo = P.verify_frozen()
    gid = P.group_id(dom, h)
    bdir = out_root / "b32_export"
    test_path = bdir / P.test_npz_name(dom, h)
    val_path = bdir / P.val_npz_name(dom, h)
    mj_path = bdir / ("metrics_b32__%s.json" % gid)
    for f in (test_path, val_path, mj_path):
        if not f.exists():
            raise SystemExit("[FATAL] b32 export missing: %s" % f)
    sha_pre = {"test_b32": sha256_of(test_path),
               "val_b32": sha256_of(val_path)}

    npz_path = meta_path = lite_path = None
    for root in roots:
        c1 = root / P.ckpt_npz_name(dom, h)
        c2 = root / P.ckpt_meta_name(dom, h)
        c3 = root / ("adapter_meta__%s.json" % gid)
        if c1.exists() and c2.exists() and c3.exists():
            npz_path, meta_path, lite_path = c1, c2, c3
            break
    if npz_path is None:
        raise SystemExit("[FATAL] pack files missing for %s in %s"
                         % (gid, [str(r) for r in roots]))

    model = build_muse(load_ckpt=True)
    configure_muse(model, dom, h)
    model = model.to("cuda:0")
    meta = load_json(meta_path)
    lite = load_json(lite_path)
    restore_muse_ckpt_pack(model, npz_path, meta, lite)

    base_test = dict(np.load(str(test_path)))
    base_val = dict(np.load(str(val_path)))
    mj = load_json(mj_path)
    dd = DomainData(pinfo["protocol_dir"], pinfo["splits"], dom)
    sp = dd.load_split(h, "test")
    sv = dd.load_split(h, "val")

    ident = all(
        identity_equal(a["window_index"], b["window_index"])
        and bitwise_equal(a["Y"], b["truth"])
        and bitwise_equal(a["s"], b["s"])
        and bitwise_equal(a["orig_row"], b["orig_row"])
        and bitwise_equal(a["label_dates"], b["label_dates"])
        for a, b in ((sp, base_test), (sv, base_val)))

    variants = {}
    for tag, X, cached in (("test", sp["X"], base_test),
                           ("val", sv["X"], base_val)):
        pred = predict_split(model, X, batch=P.BATCH)
        p2 = _2d(pred)
        close, max_abs, max_viol = pred_close(
            p2, _2d(cached["pred"]), atol, rtol)
        m = metrics_from(p2, _2d(cached["truth"]))
        mref = mj["variants"][tag]
        metrics_match = (abs(m["mse"] - mref["mse"]) <= 1e-9
                         and abs(m["mae"] - mref["mae"]) <= 1e-9
                         and abs(m["mse_f64"] - mref["mse_f64"]) <= 1e-9)
        variants[tag] = {
            "batch": int(P.BATCH), "close": bool(close),
            "bitwise_pred": bool(np.array_equal(
                p2, _2d(cached["pred"]))),
            "max_abs_diff": max_abs, "max_tol_violation": max_viol,
            "metrics_match": bool(metrics_match),
            "mse": m["mse"], "mae": m["mae"]}

    sha_post = {"test_b32": sha256_of(test_path),
                "val_b32": sha256_of(val_path)}
    immut = sha_pre == sha_post
    all_close = all(v["close"] and v["metrics_match"]
                    for v in variants.values())
    report = {
        "gid": gid, "domain": dom, "h": int(h),
        "identity_ok": bool(ident), "immutability_ok": bool(immut),
        "adapter_npz_sha256": lite["adapter_npz_sha256"],
        "packed_sha256_adapter": lite["packed_sha256"],
        "atol": atol, "rtol": rtol, "variants": variants,
        "all_pass": bool(all_close and ident and immut)}
    dump_json(out_dir / ("replay_b32__%s.json" % gid), report)
    worst = max(v["max_abs_diff"] for v in variants.values())
    print("REPLAY_B32 %s %s ident=%s immut=%s bitwise=%s/%s "
          "metrics_match=%s/%s worst=%.3e"
          % (gid, "PASS" if report["all_pass"] else "FAIL", ident,
             immut, variants["test"]["bitwise_pred"],
             variants["val"]["bitwise_pred"],
             variants["test"]["metrics_match"],
             variants["val"]["metrics_match"], worst), flush=True)
    return report


def single(dom, h, pack_roots, out_root=None):
    setup_deterministic_env()
    if out_root:
        os.environ["MUSE_TIMEMMD_S2021_OUT"] = out_root
    P.verify_frozen()
    oroot = Path(out_root) if out_root else P.out_root()
    out_dir = oroot / "replay_b32"
    out_dir.mkdir(parents=True, exist_ok=True)
    roots = [Path(x) for x in pack_roots.split(",")]
    rep = run_group(dom, h, roots, out_dir, oroot, P.ATOL, P.RTOL)
    if not rep["all_pass"]:
        raise SystemExit(2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack_roots", required=True,
                    help="comma-separated extracted pack dirs")
    ap.add_argument("--single", action="store_true",
                    help="worker mode: process one group and exit")
    ap.add_argument("--dom", default=None)
    ap.add_argument("--h", type=int, default=None)
    ap.add_argument("--protocol_dir", default=None)
    ap.add_argument("--out_root", default=None)
    args = ap.parse_args()

    if args.single:
        single(args.dom, args.h, args.pack_roots, args.out_root)
        return

    setup_deterministic_env()
    P.verify_frozen(args.protocol_dir)
    if args.out_root:
        os.environ["MUSE_TIMEMMD_S2021_OUT"] = args.out_root
    out_root = P.out_root()
    out_dir = out_root / "replay_b32"
    out_dir.mkdir(parents=True, exist_ok=True)
    roots = [Path(x) for x in args.pack_roots.split(",")]

    n_fail = 0
    gids = []
    for dom, h in P.all_groups():
        cmd = [sys.executable, "-m",
               "repro_timemmd_common_v1.replay_b32", "--single",
               "--dom", dom, "--h", str(h),
               "--pack_roots", args.pack_roots]
        if args.out_root:
            cmd += ["--out_root", args.out_root]
        rc = subprocess.run(cmd).returncode
        gid = P.group_id(dom, h)
        gids.append(gid)
        if rc != 0:
            n_fail += 1
            print("DRIVER: %s worker rc=%d" % (gid, rc), flush=True)

    reports = {g: load_json(out_dir / ("replay_b32__%s.json" % g))
               for g in gids}
    n_bitwise = sum(1 for r in reports.values()
                    for v in r["variants"].values()
                    if v["bitwise_pred"])
    summary = {
        "n_groups": len(gids), "n_pass": len(gids) - n_fail,
        "n_variants_bitwise": n_bitwise,
        "batch": int(P.BATCH), "atol": P.ATOL, "rtol": P.RTOL,
        "note": "same-batch (32) acceptance: pack restore vs b32 "
                "export arrays; one subprocess per group",
        "reports": reports}
    dump_json(out_dir / "replay_b32_summary.json", summary)
    print("REPLAY_B32: %d/%d pass, bitwise=%d/72"
          % (summary["n_pass"], len(gids), n_bitwise))
    if n_fail:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
