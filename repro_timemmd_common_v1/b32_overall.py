"""Recompute per-group/domain/overall metrics from the batch=32 export
arrays and compare against the ORIGINAL batch=64 export arrays with the
identical convention (evaluate.metrics_from, f32-pairwise main +
f64 audit). Run under the SAME env that produced evaluation.json
(visionts_mmd) so the f32 means are directly comparable; no torch.
Writes b32_export/b32_overall.json. Exit 2 if any per-group identity
guard fails (window_index exact + truth bitwise between the b64 and b32
arrays).
"""
import argparse

import numpy as np

from repro_timemmd_common_v1 import protocol as P
from repro_timemmd_common_v1.common import bitwise_equal, dump_json, \
    load_json, setup_deterministic_env
from repro_timemmd_common_v1.evaluate import metrics_from


def _2d(a):
    return a[:, :, 0] if a.ndim == 3 else a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--protocol_dir", default=None)
    ap.add_argument("--out_root", default=None)
    args = ap.parse_args()

    setup_deterministic_env()
    P.verify_frozen(args.protocol_dir)
    if args.out_root:
        import os
        os.environ["MUSE_TIMEMMD_S2021_OUT"] = args.out_root
    out_root = P.out_root()

    groups, by_domain = {}, {}
    for dom, h in P.all_groups():
        gid = P.group_id(dom, h)
        a64 = dict(np.load(str(out_root / "runs" / gid
                                / P.test_npz_name(dom, h))))
        a32 = dict(np.load(str(out_root / "b32_export"
                                / P.test_npz_name(dom, h))))
        if not (np.array_equal(a64["window_index"], a32["window_index"])
                and bitwise_equal(a64["truth"], a32["truth"])
                and bitwise_equal(a64["s"], a32["s"])
                and bitwise_equal(a64["orig_row"], a32["orig_row"])
                and bitwise_equal(a64["label_dates"],
                                  a32["label_dates"])):
            raise SystemExit("[FATAL] identity guard b64 vs b32 %s" % gid)
        p0, t = _2d(a64["pred"]), _2d(a64["truth"])
        p1 = _2d(a32["pred"])
        m0, m1 = metrics_from(p0, t), metrics_from(p1, t)
        groups[gid] = {
            "windows": int(p1.shape[0]),
            "b64_mse": m0["mse"], "b64_mae": m0["mae"],
            "b32_mse": m1["mse"], "b32_mae": m1["mae"],
            "delta_mse": m1["mse"] - m0["mse"],
            "delta_mae": m1["mae"] - m0["mae"],
            "abs_delta_mse": abs(m1["mse"] - m0["mse"]),
            "abs_delta_mae": abs(m1["mae"] - m0["mae"]),
            "bitwise_pred": bool(np.array_equal(p0, p1))}
        by_domain.setdefault(dom, []).append(
            {"mse": m1["mse"], "mae": m1["mae"],
             "b64_mse": m0["mse"], "b64_mae": m0["mae"]})

    # per-domain = mean over the 4 horizon GROUPS (f64); 9-domain
    # overall = mean over the 9 domains (f64). Same convention as
    # evaluate.py (group scores f32, aggregation f64).
    by_domain = {dom: {k: float(np.mean([r[k] for r in rows]))
                       for k in ("mse", "mae", "b64_mse", "b64_mae")}
                 for dom, rows in by_domain.items()}

    def overall(key):
        return float(np.mean([by_domain[d][key] for d in P.ORDER]))

    b32_overall9 = {"mse": overall("mse"), "mae": overall("mae")}
    b64_overall9 = {"mse": overall("b64_mse"), "mae": overall("b64_mae")}
    g = list(groups.values())
    b32_o36 = {"mse": float(np.mean([r["b32_mse"] for r in g])),
               "mae": float(np.mean([r["b32_mae"] for r in g]))}
    b64_o36 = {"mse": float(np.mean([r["b64_mse"] for r in g])),
               "mae": float(np.mean([r["b64_mae"] for r in g]))}

    # the fixed per-domain aggregation must reproduce the FROZEN
    # evaluation.json on the b64 side (same env, same convention)
    eval9 = load_json(out_root / "evaluation.json")["overall_9domain_equal"]
    if abs(b64_overall9["mse"] - eval9["mse"]) > 1e-12 or \
            abs(b64_overall9["mae"] - eval9["mae"]) > 1e-12:
        raise SystemExit(
            "[FATAL] b64 9-domain aggregation mismatch vs "
            "evaluation.json: %.16f/%.16f vs %.16f/%.16f"
            % (b64_overall9["mse"], b64_overall9["mae"],
               eval9["mse"], eval9["mae"]))

    result = {
        "env_note": "computed under the same env as evaluation.json "
                    "(visionts_mmd); metrics_from f32-pairwise main; "
                    "aggregation f64",
        "aggregation_check": {
            "b64_9domain_matches_evaluation_json": True,
            "b32_9domain_minus_36group_mse":
                b32_overall9["mse"] - b32_o36["mse"],
            "b32_9domain_minus_36group_mae":
                b32_overall9["mae"] - b32_o36["mae"]},
        "groups": groups,
        "overall_9domain_equal": {
            "b64": b64_overall9, "b32": b32_overall9,
            "delta_mse": b32_overall9["mse"] - b64_overall9["mse"],
            "delta_mae": b32_overall9["mae"] - b64_overall9["mae"]},
        "overall_36group_equal": {
            "b64": b64_o36, "b32": b32_o36,
            "delta_mse": b32_o36["mse"] - b64_o36["mse"],
            "delta_mae": b32_o36["mae"] - b64_o36["mae"]},
        "bounds": {
            "max_abs_group_delta_mse": max(r["abs_delta_mse"]
                                           for r in g),
            "max_abs_group_delta_mae": max(r["abs_delta_mae"]
                                           for r in g),
            "n_bitwise_groups": sum(1 for r in g
                                    if r["bitwise_pred"])}}
    dump_json(out_root / "b32_export" / "b32_overall.json", result)
    print("B32_OVERALL 9-domain: b64 MSE %.9f -> b32 MSE %.9f "
          "(delta %+.3e); MAE %.9f -> %.9f (delta %+.3e)"
          % (b64_overall9["mse"], b32_overall9["mse"],
             result["overall_9domain_equal"]["delta_mse"],
             b64_overall9["mae"], b32_overall9["mae"],
             result["overall_9domain_equal"]["delta_mae"]))
    print("B32_OVERALL 36-group: b64 MSE %.9f -> b32 MSE %.9f "
          "(delta %+.3e); MAE %.9f -> %.9f (delta %+.3e)"
          % (b64_o36["mse"], b32_o36["mse"],
             result["overall_36group_equal"]["delta_mse"],
             b64_o36["mae"], b32_o36["mae"],
             result["overall_36group_equal"]["delta_mae"]))
    print("B32_BOUNDS: max |dMSE|=%.3e max |dMAE|=%.3e "
          "bitwise_groups=%d/36"
          % (result["bounds"]["max_abs_group_delta_mse"],
             result["bounds"]["max_abs_group_delta_mae"],
             result["bounds"]["n_bitwise_groups"]))


if __name__ == "__main__":
    main()
