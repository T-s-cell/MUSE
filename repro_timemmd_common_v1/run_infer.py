"""36-group test inference driver (EXP-029): re-executes infer_one.py as
an independent process per group (strict ckpt restore + MAE backbone
backfill each time).

Gates: preflight all_pass (live), guards.json equal to run_train's tier-1
set, training status all-done, ckpt content hashes bound in status.
Skip rule: state==done AND test artifacts verify (keys/count/permutation/
finiteness/metrics-json sha); else quarantine runs/<gid>/ and redo.
"""
import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from repro_timemmd_common_v1 import protocol as P
from repro_timemmd_common_v1.common import all_finite, die, dump_json, \
    guards_equal, load_json, setup_deterministic_env, sha256_of
from repro_timemmd_common_v1.muse_train import full_guards


def group_artifacts_ok(out_dir, dom, h, want_windows):
    gdir = out_dir / "runs" / P.group_id(dom, h)
    npz = gdir / P.test_npz_name(dom, h)
    if not npz.exists():
        return False
    try:
        arr = np.load(str(npz))
        for k in ("pred", "truth", "s", "orig_row", "label_dates",
                  "window_index"):
            if k not in arr.files:
                return False
        if arr["pred"].shape[0] != want_windows:
            return False
        if not np.array_equal(np.sort(arr["window_index"]),
                              np.arange(want_windows)):
            return False
        if not all_finite(arr["pred"]):
            return False
    except Exception:
        return False
    mj = gdir / ("metrics__%s.json" % P.group_id(dom, h))
    if not mj.exists():
        return False
    m = load_json(mj)
    return (m.get("evaluated_windows") == want_windows
            and m.get("npz_sha256") == sha256_of(npz))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--protocol_dir", default=None)
    ap.add_argument("--out_root", default=None)
    ap.add_argument("--groups", default=None)
    args = ap.parse_args()

    setup_deterministic_env()
    pinfo = P.verify_frozen(args.protocol_dir)
    out_dir = Path(args.out_root or P.out_root())

    pf_path = out_dir / "preflight" / "preflight_report.json"
    pf = load_json(pf_path) if pf_path.exists() else None
    if pf is None or not pf.get("all_pass") or pf.get("errors"):
        die("[TIER-1] preflight report missing or not all_pass")

    guards = full_guards(pinfo, pinfo["selected_p_sha256"])
    gpath = out_dir / "guards.json"
    if gpath.exists():
        ok, diff = guards_equal(load_json(gpath), guards)
        if not ok:
            die("[TIER-1] guards mismatch (%s); refusing to infer."
                % ",".join(diff))
    else:
        die("[TIER-1] guards.json missing; run run_train first")

    train_status_path = out_dir / "status.json"
    if not train_status_path.exists():
        die("[TIER-1] training status.json missing; run run_train first")
    tstat = load_json(train_status_path)
    ckpt_sha = {}
    for gid, ent in tstat.get("groups", {}).items():
        if ent.get("state") != "done":
            die("[TIER-1] group %s not done in training status" % gid)
        ckpt_sha[gid] = ent.get("ckpt_npz_sha256")

    splits = pinfo["splits"]
    groups = P.all_groups()
    if args.groups:
        want = set(args.groups.split(","))
        groups = [g for g in groups if P.group_id(*g) in want]

    ts_path = out_dir / "test_status.json"
    tstat_infer = load_json(ts_path) if ts_path.exists() else {"groups": {}}

    n_fail = 0
    for dom, h in groups:
        gid = P.group_id(dom, h)
        want = int(splits[dom]["window_counts"][str(h)]["test"])
        ent = tstat_infer["groups"].get(gid, {})
        if ent.get("state") == "done" and \
                group_artifacts_ok(out_dir, dom, h, want):
            print("SKIP %s (done, artifacts ok)" % gid, flush=True)
            continue
        gdir = out_dir / "runs" / gid
        if gdir.exists():
            qdir = out_dir / "quarantine" / ("infer_" + gid)
            qdir.parent.mkdir(parents=True, exist_ok=True)
            if qdir.exists():
                shutil.rmtree(qdir)
            shutil.move(str(gdir), str(qdir))
            print("QUARANTINED old infer %s" % gid, flush=True)
        tstat_infer["groups"][gid] = {"state": "running"}
        dump_json(ts_path, tstat_infer)
        t0 = time.time()
        cmd = [sys.executable, "-m",
               "repro_timemmd_common_v1.infer_one",
               "--dom", dom, "--h", str(h)]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            tstat_infer["groups"][gid] = {"state": "failed",
                                          "stderr": r.stderr[-4000:]}
            n_fail += 1
            dump_json(ts_path, tstat_infer)
            print("FAIL %s rc=%d" % (gid, r.returncode), flush=True)
            continue
        mj = load_json(gdir / ("metrics__%s.json" % gid))
        if mj.get("ckpt_npz_sha256") != ckpt_sha.get(gid):
            tstat_infer["groups"][gid] = {
                "state": "failed",
                "error": "inference used a different ckpt than training"}
            n_fail += 1
            dump_json(ts_path, tstat_infer)
            continue
        tstat_infer["groups"][gid] = {
            "state": "done", "mse": mj["mse"], "mae": mj["mae"],
            "npz_sha256": mj["npz_sha256"],
            "metrics_sha256": sha256_of(gdir /
                                        ("metrics__%s.json" % gid)),
            "wall_s": round(time.time() - t0, 1)}
        dump_json(ts_path, tstat_infer)
        print("DONE %s mse=%.6f (%.1fs incl. process)"
              % (gid, mj["mse"], time.time() - t0), flush=True)

    done = sum(1 for g in tstat_infer["groups"].values()
               if g.get("state") == "done")
    print("RUN_INFER EXIT: done=%d failed=%d" % (done, n_fail))
    if n_fail:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
