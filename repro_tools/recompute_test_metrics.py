"""Independent recomputation of official test metrics from saved arrays.

Loads test_preds(.npz | sharded) + test_report.json, rebuilds the eval
StandardScaler from the saved mean/scale, and replays the official
aggregation: per window, transform actual+predicted then compute MSE/MAE
over the whole window; final metric = mean over windows. Compares against
the official mse_norm/mae_norm(/rmse_norm cross-check) with |diff| <= 1e-6.

Exit codes: 0 pass, 1 mismatch/failure.
"""

import argparse
import json
import os
import sys

import numpy as np

from .preds_io import load_preds_any


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--tol", type=float, default=1e-6)
    args = ap.parse_args()

    with open(os.path.join(args.out_dir, "test_report.json")) as f:
        report = json.load(f)

    saved = load_preds_any(args.out_dir, "test_preds")
    preds = saved["preds"].astype(np.float64)
    targets = saved["targets"].astype(np.float64)
    borders = np.asarray(saved["window_border"], dtype=np.int64)

    stats = report["eval_scaler"]
    mean = np.asarray(stats["mean"], dtype=np.float64)
    scale = np.asarray(stats["scale"], dtype=np.float64)

    n = preds.shape[0]
    assert targets.shape == preds.shape
    mse_w = np.empty(n, dtype=np.float64)
    mae_w = np.empty(n, dtype=np.float64)
    for i in range(n):
        e = (targets[i] - mean) / scale - (preds[i] - mean) / scale
        mse_w[i] = np.square(e).mean()
        mae_w[i] = np.abs(e).mean()
    mse_re = float(mse_w.mean())
    mae_re = float(mae_w.mean())

    mean_metrics = list(report["mean_metrics"])
    mse_off = float(mean_metrics[0])
    mae_off = float(mean_metrics[1])
    rmse_re = float(np.sqrt(mse_re))
    checks = {
        "mse_absdiff": abs(mse_re - mse_off),
        "mae_absdiff": abs(mae_re - mae_off),
    }
    rmse_off = (
        float(mean_metrics[2]) if len(mean_metrics) > 2 else None
    )
    if rmse_off is not None:
        checks["rmse_absdiff"] = abs(rmse_re - rmse_off)

    out = {
        "out_dir": args.out_dir,
        "n_windows": n,
        "sharded": saved["sharded"],
        "coverage_ok": saved["coverage_ok"],
        "coverage_detail": saved["coverage_detail"],
        "border_first": int(borders[0]),
        "border_last": int(borders[-1]),
        "preds_shape": list(preds.shape),
        "mse_recomputed": mse_re,
        "mae_recomputed": mae_re,
        "rmse_recomputed": rmse_re,
        "mse_official": mse_off,
        "mae_official": mae_off,
        "rmse_official": rmse_off,
        "absdiff": checks,
        "tol": args.tol,
        "pass": (
            saved["coverage_ok"]
            and all(v <= args.tol for v in checks.values())
        ),
    }
    path = os.path.join(args.out_dir, "recompute_test_metrics.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2, sort_keys=True)
    print(json.dumps(out, indent=2, sort_keys=True))
    if not out["pass"]:
        print(f"[recompute] FAIL (tol={args.tol})")
        sys.exit(1)
    print(f"[recompute] PASS -> {path}")


if __name__ == "__main__":
    main()
