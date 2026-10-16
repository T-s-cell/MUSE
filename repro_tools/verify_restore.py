"""New-process restore verification for a saved val point.

Run on the training machine from the repo root:
    python -m repro_tools.verify_restore --out-dir <bundle_dir> [--step N|--best]

Configuration is read from <bundle_dir>/run_config.json (written by the
orchestrator). The --legacy flag is reserved for the frozen ETTh1/H96
directory, which predates run manifests; new runs without run_config.json
are an error.

Checks:
  1. data/ckpt sha256 recomputed and compared (vs --expect-* args when given);
  2. data pipeline rebuilt with real code from the SAVED configuration;
     window/batch counts asserted against the recorded train_info;
  3. scaler vs saved point meta: allclose atol=1e-10;
  4. full state_dict loaded strict=True;
  5. official MUSE.validate re-run on the rebuilt valid loader:
     |recomputed - recorded| <= 1e-6;
  6. element-wise comparison of re-run predictions/targets vs saved npz
     (single or sharded), max abs errors reported; targets must match ~0;
     window identity asserted against the split formula;
  7. independent float64 MSE recomputation from saved preds/targets vs the
     recorded official value.

Exit codes: 0 all pass, 1 any failure.
"""

import argparse
import hashlib
import json
import os
import sys

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from ts_benchmark.baselines.muse.muse import MUSE
from ts_benchmark.baselines.utils import DatasetForTransformer, train_val_split
from ts_benchmark.common.constant import FORECASTING_DATASET_PATH
from ts_benchmark.data.data_source import LocalForecastingDataSource
from ts_benchmark.evaluation.strategy.rolling_forecast import RollingForecast
from ts_benchmark.utils.data_processing import split_time

from .preds_io import load_preds_any
from .run_config import load_run_config

# Exact frozen ETTh1/H96 constants (--legacy ONLY; never for new runs).
LEGACY_RUN = {
    "series": "ETTh1.csv",
    "horizon": 96,
    "tv_ratio": 0.8,
    "train_ratio_in_tv": 0.75,
    "seq_len": 1440,
    "batch_size": 32,
    "num_workers": 0,
}
LEGACY_HYPER_PARAMS = {
    "batch_size": 32,
    "horizon": 96,
    "loss": "MSE",
    "lr": 0.00001,
    "lradj": "type3",
    "norm": True,
    "num_epochs": 10,
    "patience": 2,
    "seq_len": 1440,
}
# Immutable pretrained MAE backbone used by every run of this campaign
# (recorded as env.mae_sha in all orchestrator-era run_config.json files;
# same file path and content for the frozen stage-1 run).
LEGACY_EXPECTED_MAE_SHA = (
    "c551c37b63a221a3e6d7d7f49e96f34078c9b183f788bedc0387493521bade0e"
)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_json(path):
    with open(path) as f:
        return json.load(f)


def resolve_run_params(args):
    cfg = load_run_config(args.out_dir)
    if cfg is not None:
        run = {
            "series": cfg["dataset_file"],
            "horizon": int(cfg["protocol"]["horizon"]),
            "tv_ratio": float(cfg["protocol"]["tv_ratio"]),
            "train_ratio_in_tv": float(cfg["protocol"]["train_ratio_in_tv"]),
            "seq_len": int(cfg["loader"]["seq_len"]),
            "batch_size": int(cfg["loader"]["batch_size"]),
            "num_workers": int(cfg["loader"].get("num_workers", 0)),
        }
        hyper = dict(cfg["hyper_params_eff"])
        return run, hyper, cfg
    if args.legacy:
        return dict(LEGACY_RUN), dict(LEGACY_HYPER_PARAMS), None
    raise SystemExit(
        f"run_config.json missing in {args.out_dir}; --legacy is reserved "
        f"for the frozen ETTh1/H96 directory"
    )


def preds_stem(preds_path):
    base = os.path.basename(preds_path)
    for suffix in ("_shards.json", ".npz"):
        if base.endswith(suffix):
            return base[: -len(suffix)]
    raise ValueError(f"unrecognised preds path: {preds_path}")


def pick_record(manifest, args):
    records = [
        r for r in (manifest.get("val_records", [])
                    + manifest.get("best_records", []))
        if r.get("saved", {}).get("full_state_path")
    ]
    if not records:
        raise SystemExit("no val record with a saved full state found")
    if args.step is not None:
        rec = next((r for r in records if r["step"] == args.step), None)
        if rec is None:
            raise SystemExit(f"no saved val point at step {args.step}")
    else:
        rec = min(records, key=lambda r: r["val_mse_fused"])
    return rec


def rebuild_pipeline(run, hyper_params):
    csv_path = os.path.join(FORECASTING_DATASET_PATH, run["series"])
    src = LocalForecastingDataSource()
    src.load_series_list([run["series"]])
    series = src.dataset.get_series(run["series"])
    meta = src.dataset.get_series_meta_info(run["series"])

    rf = RollingForecast.__new__(RollingForecast)
    train_length, _ = rf._get_split_lens(series, meta, run["tv_ratio"])
    tv_data, _ = split_time(series, train_length)

    m = MUSE(**dict(hyper_params))
    m.config.seq_len = run["seq_len"]
    MUSE.multi_forecasting_hyper_param_tune(m, tv_data)

    train_data, valid_data = train_val_split(
        tv_data, run["train_ratio_in_tv"], run["seq_len"]
    )
    scaler = StandardScaler().fit(train_data.values)
    valid_scaled = pd.DataFrame(
        scaler.transform(valid_data.values),
        columns=valid_data.columns,
        index=valid_data.index,
    )
    valid_ds = DatasetForTransformer(
        dataset=valid_scaled,
        history_len=run["seq_len"],
        prediction_len=run["horizon"],
        label_len=m.config.label_len,
        timeenc=1,
        freq=m.config.freq,
    )
    loader = DataLoader(
        valid_ds,
        batch_size=run["batch_size"],
        shuffle=False,
        num_workers=run["num_workers"],
        drop_last=False,
    )
    return {
        "muse_adapter": m,
        "scaler": scaler,
        "valid_ds": valid_ds,
        "valid_loader": loader,
        "csv_path": csv_path,
        "valid_data": valid_data,
        "train_data": train_data,
    }


def dump_predictions(muse_adapter, loader):
    """Mirror hook._dump_val_predictions == MUSE.validate accumulation order."""
    config = muse_adapter.config
    sd = muse_adapter._series_dim
    horizon = config.horizon
    vloader = DataLoader(
        loader.dataset,
        batch_size=loader.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        drop_last=False,
    )
    device = next(muse_adapter.model.parameters()).device
    muse_adapter.model.eval()
    preds, tgts = [], []
    with torch.no_grad():
        for input, target, input_mark, target_mark in vloader:
            input, target = input.to(device), target.to(device)
            output = muse_adapter.model(
                input,
                fp64=config.fp64,
                use_variable_chunk=config.use_variable_chunk,
            )
            target = target[:, -horizon:, :sd]
            output = output[:, -horizon:, :sd]
            output, target = muse_adapter._post_process(output, target)
            preds.append(output.cpu().numpy())
            tgts.append(target.cpu().numpy())
    muse_adapter.model.train()
    return np.concatenate(preds), np.concatenate(tgts)


def main():
    if os.environ.get("MUSE_REPRO_FORCE_DETERMINISTIC", "") == "1":
        # match the forcing used by the audited run so kernels are selected
        # identically (CUBLAS_WORKSPACE_CONFIG must come from the environment)
        torch.use_deterministic_algorithms(True)
    # replicate the strategy's official environment (fix_all_random_seed also
    # flips cudnn flags: deterministic=True, benchmark=False, enabled=False)
    from ts_benchmark.utils.random_utils import fix_all_random_seed

    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--step", type=int, default=None)
    ap.add_argument("--best", action="store_true")
    ap.add_argument("--legacy", action="store_true",
                    help="frozen ETTh1/H96 dir only (pre-manifest); new runs "
                         "must carry run_config.json")
    ap.add_argument("--mae-sha", default=None,
                    help="expected sha256 of the MAE checkpoint")
    ap.add_argument("--mae-path", default=None,
                    help="MAE checkpoint path if not recorded in the manifest")
    ap.add_argument("--data-sha", default=None,
                    help="expected sha256 of the dataset CSV")
    args = ap.parse_args()

    if args.step is None and not args.best:
        args.best = True

    run, hyper_params, run_cfg = resolve_run_params(args)
    seed = 2021 if run_cfg is None else int(run_cfg["protocol"]["seed"])
    fix_all_random_seed(seed)

    checks = []
    failures = []

    def check(name, ok, detail=None):
        checks.append({"name": name, "ok": bool(ok), "detail": detail})
        if not ok:
            failures.append(name)
        print(f"[verify] {'PASS' if ok else 'FAIL'} {name}"
              + (f" :: {detail}" if detail else ""))

    manifest = load_json(os.path.join(args.out_dir, "manifest.json"))
    rec = pick_record(manifest, args)
    point_meta = load_json(rec["saved"]["point_meta_path"])
    report = {
        "out_dir": args.out_dir,
        "record": {k: rec[k] for k in ("tag", "step", "val_mse_fused")},
        "run_params": run,
        "legacy": run_cfg is None,
        "checks": checks,
    }

    # 1. hashes ----------------------------------------------------------
    # Forced comparison: a missing/unverifiable hash must FAIL, never
    # vacuously PASS. Expected values come from run_config.env (recorded at
    # training time) and dataset_sync.json; CLI args only override.
    def _fmt(h):
        return "MISSING" if not h else h[:16] + "..."

    csv_path = os.path.join(FORECASTING_DATASET_PATH, run["series"])
    csv_sha = sha256_file(csv_path) if os.path.exists(csv_path) else None
    mae_path = (manifest.get("mae_ckpt_path") or args.mae_path
                or (run_cfg.get("hyper_params_eff") or {}).get("checkpoint_path")
                if run_cfg is not None
                else (manifest.get("mae_ckpt_path") or args.mae_path))
    mae_sha = (sha256_file(mae_path)
               if mae_path and os.path.exists(mae_path) else None)

    expected_csv = args.data_sha
    expected_mae = args.mae_sha
    if run_cfg is not None:
        expected_mae = expected_mae or (run_cfg.get("env") or {}).get("mae_sha")
    if expected_csv is None:
        sync_path = os.path.join(FORECASTING_DATASET_PATH, "dataset_sync.json")
        if os.path.exists(sync_path):
            entry = (load_json(sync_path).get("files") or {}).get(run["series"])
            expected_csv = entry.get("dst_sha256") if entry else None
    if expected_mae is None and run_cfg is None:
        expected_mae = LEGACY_EXPECTED_MAE_SHA

    check("mae_sha_computed", mae_sha is not None,
          f"computed={_fmt(mae_sha)} path={mae_path}")
    check("mae_sha_matches_expected",
          mae_sha is not None and expected_mae is not None
          and mae_sha == expected_mae,
          f"expected={_fmt(expected_mae)} computed={_fmt(mae_sha)}")
    check("data_sha_computed", csv_sha is not None,
          f"computed={_fmt(csv_sha)} path={csv_path}")
    check("data_sha_matches_expected",
          csv_sha is not None and expected_csv is not None
          and csv_sha == expected_csv,
          f"expected={_fmt(expected_csv)} computed={_fmt(csv_sha)}")
    report["hashes"] = {
        "csv": csv_sha, "mae": mae_sha,
        "expected_csv": expected_csv, "expected_mae": expected_mae,
    }
    report["metadata_hashes"] = {
        os.path.basename(p): sha256_file(p)
        for p in [os.path.join(args.out_dir, "run_config.json"),
                  os.path.join(args.out_dir, "manifest.json"),
                  rec["saved"].get("point_meta_path")]
        if p and os.path.exists(p)
    }

    # 2. pipeline rebuild (from SAVED config) ------------------------------
    hyper = dict(hyper_params)
    if mae_path:
        hyper["checkpoint_path"] = mae_path
    pipe = rebuild_pipeline(run, hyper)
    m = pipe["muse_adapter"]
    m._series_dim = pipe["valid_ds"].dataset.shape[-1]
    ti = manifest.get("train_info", {})
    check("valid_windows_count",
          len(pipe["valid_ds"]) == ti.get("valid_windows"),
          f"{len(pipe['valid_ds'])} vs {ti.get('valid_windows')}")
    check("valid_batches_count",
          len(pipe["valid_loader"]) == ti.get("valid_batches"),
          f"{len(pipe['valid_loader'])} vs {ti.get('valid_batches')}")

    # 3. scaler -------------------------------------------------------------
    sm = np.asarray(point_meta["scaler_mean"])
    ss = np.asarray(point_meta["scaler_scale"])
    check("scaler_mean_allclose_1e-10",
          np.allclose(pipe["scaler"].mean_, sm, atol=1e-10, rtol=0),
          f"maxdiff={np.abs(pipe['scaler'].mean_ - sm).max():.3e}")
    check("scaler_scale_allclose_1e-10",
          np.allclose(pipe["scaler"].scale_, ss, atol=1e-10, rtol=0),
          f"maxdiff={np.abs(pipe['scaler'].scale_ - ss).max():.3e}")
    m.scaler = pipe["scaler"]

    # 4. state load -----------------------------------------------------------
    full_state = torch.load(
        rec["saved"]["full_state_path"], map_location="cpu", weights_only=True
    )
    m.model = m._init_model()
    m.model.load_state_dict(full_state, strict=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    m.model.to(device)

    # 5. official validate -----------------------------------------------------
    criterion = torch.nn.MSELoss()
    val_re = float(
        m.validate(pipe["valid_loader"], m._series_dim, criterion)
    )
    dv = abs(val_re - rec["val_mse_fused"])
    check("validate_absdiff_le_1e-6", dv <= 1e-6,
          f"recomputed={val_re:.10f} recorded={rec['val_mse_fused']:.10f} "
          f"absdiff={dv:.3e}")

    # 6. element-wise predictions ------------------------------------------------
    saved = load_preds_any(args.out_dir, preds_stem(rec["saved"]["preds_path"]))
    check("preds_coverage", saved["coverage_ok"], saved["coverage_detail"])
    wb = saved["window_border"]
    if wb.shape[-1] == len(pipe["valid_ds"]):
        valid_start_abs = len(pipe["train_data"]) - run["seq_len"]
        expected_border = valid_start_abs + np.arange(
            len(pipe["valid_ds"]), dtype=np.int64
        ) + run["seq_len"]
        got = np.asarray(wb, dtype=np.int64)
        check("window_border_matches",
              np.array_equal(got, expected_border),
              f"first={int(got[0])} last={int(got[-1])} "
              f"expected_first={int(expected_border[0])}")
    else:
        check("window_border_matches", False,
              f"border length {wb.shape[-1]} != windows "
              f"{len(pipe['valid_ds'])}")
    preds_re, tgts_re = dump_predictions(m, pipe["valid_loader"])
    err_p = float(np.abs(preds_re - saved["preds"]).max())
    err_t = float(np.abs(tgts_re - saved["targets"]).max())
    check("preds_elementwise_maxerr_le_1e-6", err_p <= 1e-6,
          f"maxerr={err_p:.3e} shape={list(preds_re.shape)}")
    check("targets_elementwise_maxerr_le_1e-10", err_t <= 1e-10,
          f"maxerr={err_t:.3e}")

    # 7. independent MSE from saved arrays ----------------------------------------
    horizon = run["horizon"]
    p = saved["preds"][:, -horizon:, :m._series_dim].astype(np.float64)
    t = saved["targets"][:, -horizon:, :m._series_dim].astype(np.float64)
    mse_ind = float(((p - t) ** 2).sum() / p.size)
    dind = abs(mse_ind - rec["val_mse_fused"])
    check("independent_mse_absdiff_le_1e-6", dind <= 1e-6,
          f"independent={mse_ind:.10f} recorded={rec['val_mse_fused']:.10f} "
          f"absdiff={dind:.3e}")

    report["val_recomputed"] = val_re
    report["preds_max_err"] = err_p
    report["targets_max_err"] = err_t
    report["independent_mse"] = mse_ind

    out_path = os.path.join(
        args.out_dir, f"verify_report_step{rec['step']}.json"
    )
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2, sort_keys=True, default=str)

    if failures:
        print(f"[verify] FAILED: {failures}")
        sys.exit(1)
    print(f"[verify] ALL PASS (step {rec['step']}, report -> {out_path})")


if __name__ == "__main__":
    main()
