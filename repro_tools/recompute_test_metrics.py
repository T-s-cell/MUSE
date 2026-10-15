"""Independent recomputation of official test metrics from saved arrays.

Loads test_preds(.npz | sharded) + test_report.json, rebuilds the eval
StandardScaler from the saved mean/scale, and replays the official
aggregation (regression_metrics._error_norm + rolling_forecast: per window,
transform actual+predicted then compute the metrics; final metric = mean
over windows).

Official mean_metrics layout (canonical, = regression_metrics.__all__ via
Evaluator.metric_names, 16 entries):
  [mae, mse, rmse, mape, smape, mase, wape, msmape,
   mae_norm, mse_norm, rmse_norm, mape_norm, smape_norm, mase_norm,
   wape_norm, msmape_norm]
The paper reports mse_norm/mae_norm (indices 9/8). Legacy 3-entry reports
([mse, mae, rmse]) are still accepted. rmse_norm is the MEAN of per-window
sqrt(mse), not sqrt of the mean — computed accordingly.

Gated checks: mse_norm/mae_norm/rmse_norm |diff| <= tol (default 1e-6).
Raw-scale metrics are recomputed informationally (ungated; float32 arrays
make 1e-6 unrealistic at raw magnitude).

Exit codes: 0 pass, 1 mismatch/failure.
"""

import argparse
import json
import os
import sys

import numpy as np

from .preds_io import load_preds_any

# Canonical RollingForecast evaluator metric order (regression_metrics.__all__).
METRIC_NAMES = [
    "mae", "mse", "rmse", "mape", "smape", "mase", "wape", "msmape",
    "mae_norm", "mse_norm", "rmse_norm", "mape_norm", "smape_norm",
    "mase_norm", "wape_norm", "msmape_norm",
]
LEGACY_3 = ["mse", "mae", "rmse"]


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
    # official _error_norm per window: scaler.transform(actual) -
    # scaler.transform(predicted), i.e. (actual - predicted) / scale
    err = ((targets - preds) / scale).reshape(n, -1)
    mse_w = np.square(err).mean(axis=1)
    mae_w = np.abs(err).mean(axis=1)
    mse_re = float(mse_w.mean())
    mae_re = float(mae_w.mean())
    rmse_re = float(np.sqrt(mse_w).mean())

    raw = (targets - preds).reshape(n, -1)
    info = {
        "mae": float(np.abs(raw).mean()),
        "mse": float(np.square(raw).mean()),
        "rmse": float(np.sqrt(np.square(raw).mean())),
    }

    mm = list(report["mean_metrics"])
    if len(mm) == len(METRIC_NAMES):
        names = list(report.get("metric_names") or METRIC_NAMES)
        if names != METRIC_NAMES:
            print(f"[recompute] unexpected metric_names {names}")
            sys.exit(1)
        layout = "canonical-16"
        mse_off, mae_off, rmse_off = mm[9], mm[8], mm[10]
        metrics_all = {METRIC_NAMES[i]: mm[i] for i in range(len(mm))}
    elif len(mm) == 3:
        layout = "legacy-3"
        mse_off, mae_off, rmse_off = mm[0], mm[1], mm[2]
        metrics_all = dict(zip(LEGACY_3, mm))
    else:
        print(f"[recompute] unknown mean_metrics layout: {len(mm)} entries")
        sys.exit(1)

    checks = {
        "mse_absdiff": abs(mse_re - float(mse_off)),
        "mae_absdiff": abs(mae_re - float(mae_off)),
        "rmse_absdiff": abs(rmse_re - float(rmse_off)),
    }

    out = {
        "out_dir": args.out_dir,
        "n_windows": n,
        "sharded": saved["sharded"],
        "coverage_ok": saved["coverage_ok"],
        "coverage_detail": saved["coverage_detail"],
        "border_first": int(borders[0]),
        "border_last": int(borders[-1]),
        "preds_shape": list(preds.shape),
        "metric_layout": layout,
        "mse_recomputed": mse_re,
        "mae_recomputed": mae_re,
        "rmse_recomputed": rmse_re,
        "mse_official": mse_off,
        "mae_official": mae_off,
        "rmse_official": rmse_off,
        "official_metrics_all": metrics_all,
        "raw_scale_info": info,
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
    print(json.dumps({k: out[k] for k in (
        "metric_layout", "mse_recomputed", "mae_recomputed", "rmse_recomputed",
        "mse_official", "mae_official", "rmse_official", "absdiff", "pass")},
        indent=2, sort_keys=True))
    if not out["pass"]:
        print(f"[recompute] FAIL (tol={args.tol})")
        sys.exit(1)
    print(f"[recompute] PASS -> {path}")


if __name__ == "__main__":
    main()
