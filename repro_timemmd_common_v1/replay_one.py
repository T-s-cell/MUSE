"""Per-group REPLAY as an independent process: strict ckpt restore + MAE
backbone backfill -> recompute the whole test split AND the whole val
split -> compare against the cached test npz and the training-time val
npz (pred atol/rtol + identity exact + truth bitwise). Also proves
immutability of the ckpt npz/meta, the cached test npz, and the val npz
by hashing before/after.
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
from repro_timemmd_common_v1.muse_train import restore_muse_ckpt_strict

NPZ_KEYS = ["pred", "truth", "s", "orig_row", "label_dates", "window_index"]


def output_paths(out_dir, dom, h, nc_tag):
    """Negative-control replays must never write into the formal
    runs/<gid>/ directory or overwrite out_root val npz; with nc_tag set
    everything lands in negative_control/ under a variant-specific
    name."""
    gid = P.group_id(dom, h)
    if nc_tag:
        ncdir = out_dir / "negative_control"
        return ncdir, ncdir / ("replay__%s__%s.npz" % (nc_tag, gid)), \
            ncdir / ("replay_report__%s__%s.json" % (nc_tag, gid)), \
            ncdir / ("replay_val__%s__%s.npz" % (nc_tag, gid))
    gdir = out_dir / "runs" / gid
    return gdir, gdir / P.replay_npz_name(dom, h), \
        gdir / ("replay_report__%s.json" % gid), \
        gdir / ("replay_val__%s.npz" % gid)


def run_group(dom, h, atol=P.ATOL, rtol=P.RTOL, ckpt_override=None,
              meta_override=None, nc_tag=None):
    setup_deterministic_env()
    pinfo = P.verify_frozen()
    out_dir = P.out_root()
    gid = P.group_id(dom, h)
    gdir = out_dir / "runs" / gid
    test_path = gdir / P.test_npz_name(dom, h)
    val_path = out_dir / P.val_npz_name(dom, h)
    out_gdir, replay_path, report_path, replay_val_path = \
        output_paths(out_dir, dom, h, nc_tag)
    if nc_tag:
        out_gdir.mkdir(parents=True, exist_ok=True)

    sha_test_pre = sha256_of(test_path)
    sha_val_pre = sha256_of(val_path)
    ckpt_path = Path(ckpt_override) if ckpt_override else \
        out_dir / P.ckpt_npz_name(dom, h)
    meta_path = Path(meta_override) if meta_override else \
        out_dir / P.ckpt_meta_name(dom, h)
    sha_ckpt_pre = sha256_of(ckpt_path)

    model = build_muse(load_ckpt=True)
    configure_muse(model, dom, h)
    model = model.to("cuda:0")
    meta = restore_muse_ckpt_strict(model, ckpt_path,
                                    load_json(meta_path))

    base = dict(np.load(str(test_path)))
    base_val = dict(np.load(str(val_path)))
    dd = DomainData(pinfo["protocol_dir"], pinfo["splits"], dom)
    sp = dd.load_split(h, "test")
    sv = dd.load_split(h, "val")
    pred = predict_split(model, sp["X"], batch=P.INFER_BATCH)
    pred_val = predict_split(model, sv["X"], batch=P.INFER_BATCH)
    np.savez(str(replay_path), pred=pred.astype(np.float32),
             truth=sp["Y"].astype(np.float32),
             s=sp["s"], orig_row=sp["orig_row"],
             label_dates=sp["label_dates"],
             window_index=sp["window_index"])
    np.savez(str(replay_val_path), pred=pred_val.astype(np.float32),
             truth=sv["Y"].astype(np.float32),
             s=sv["s"], orig_row=sv["orig_row"],
             label_dates=sv["label_dates"],
             window_index=sv["window_index"])

    close, max_abs, max_viol = pred_close(pred, base["pred"], atol, rtol)
    close_v, max_abs_v, max_viol_v = pred_close(pred_val,
                                                base_val["pred"],
                                                atol, rtol)
    ident = (identity_equal(sp["s"], base["s"])
             and identity_equal(sp["orig_row"], base["orig_row"])
             and identity_equal(sp["label_dates"], base["label_dates"])
             and identity_equal(sp["window_index"], base["window_index"]))
    ident_v = (identity_equal(sv["s"], base_val["s"])
               and identity_equal(sv["orig_row"], base_val["orig_row"])
               and identity_equal(sv["label_dates"],
                                  base_val["label_dates"])
               and identity_equal(sv["window_index"],
                                  base_val["window_index"]))
    truth_ok = bitwise_equal(sp["Y"], base["truth"])
    truth_ok_v = bitwise_equal(sv["Y"], base_val["truth"])
    immut = (sha_test_pre == sha256_of(test_path)
             and sha_ckpt_pre == sha256_of(ckpt_path)
             and sha_val_pre == sha256_of(val_path))
    ok = close and ident and truth_ok and close_v and ident_v \
        and truth_ok_v
    report = {
        "gid": gid, "domain": dom, "h": int(h), "nc_tag": nc_tag,
        "windows": int(sp["X"].shape[0]),
        "windows_val": int(sv["X"].shape[0]),
        "close": close, "close_val": close_v,
        "max_abs_diff": max_abs, "max_tol_violation": max_viol,
        "max_abs_diff_val": max_abs_v, "max_tol_violation_val": max_viol_v,
        "atol": atol, "rtol": rtol,
        "identity_ok": bool(ident), "truth_bitwise_ok": bool(truth_ok),
        "identity_ok_val": bool(ident_v),
        "truth_bitwise_ok_val": bool(truth_ok_v),
        "ckpt_npz_sha256": meta["npz_sha256"],
        "sha_test_before": sha_test_pre, "sha_ckpt_before": sha_ckpt_pre,
        "sha_val_before": sha_val_pre,
        "sha_test_after": sha256_of(test_path),
        "sha_ckpt_after": sha256_of(ckpt_path),
        "sha_val_after": sha256_of(val_path),
        "immutability_ok": bool(immut),
        "all_pass": bool(ok and immut),
    }
    dump_json(report_path, report)
    print("REPLAY %s %s close=%s/%s max_abs=%.3e/%.3e"
          % (report["gid"], "PASS" if report["all_pass"] else "FAIL",
             close, close_v, max_abs, max_abs_v))
    if not report["all_pass"]:
        raise SystemExit(2)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dom", required=True)
    ap.add_argument("--h", type=int, required=True)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--meta", default=None)
    ap.add_argument("--nc_tag", default=None)
    args = ap.parse_args()
    run_group(args.dom, args.h, ckpt_override=args.ckpt,
              meta_override=args.meta, nc_tag=args.nc_tag)
