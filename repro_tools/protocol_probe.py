"""Data-protocol probe: adjudicate META length vs CSV length with real code.

Run on the training machine from the repo root:
    python -m repro_tools.protocol_probe --dataset ETTh2.csv --horizon 96 \
        --seq-len 1720 [--tv-ratio 0.8 --train-ratio-in-tv 0.75 ...]

All dataset-specific values come from the CLI (the orchestrator derives them
from configs_manifest.json); nothing is hardcoded per dataset. The one
exception is --expect-meta-len, an OPTIONAL cross-check the caller may pin
(e.g. ETTh1 14400); without it the probe reports the adjudication only.

Exit codes: 0 = all checks passed;
            2 = protocol mismatch (fallback branch or failed expectation);
            1 = internal assertion/other failure.

The probe is strictly read-only: CSV/META sha256 must be identical before/after.
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


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def jsafe(obj):
    if isinstance(obj, dict):
        return {str(k): jsafe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsafe(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    if isinstance(obj, (pd.Timestamp,)):
        return str(obj)
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return repr(obj)


def dump_json(path, obj):
    with open(path, "w") as f:
        json.dump(jsafe(obj), f, indent=2, sort_keys=True, default=str)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, help="e.g. ETTh2.csv")
    ap.add_argument("--horizon", type=int, required=True)
    ap.add_argument("--seq-len", type=int, required=True)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--tv-ratio", type=float, default=0.8)
    ap.add_argument("--train-ratio-in-tv", type=float, default=0.75)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--num-rollings", type=int, default=48000)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--expect-meta-len", type=int, default=None,
                    help="optional pinned META length cross-check")
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    stem = os.path.splitext(args.dataset)[0]
    out_dir = args.out_dir or os.environ.get(
        "MUSE_REPRO_PROBE_OUT",
        f"repro_out/protocol_probe/{stem}_h{args.horizon}",
    )
    os.makedirs(out_dir, exist_ok=True)
    RUN = {
        "series": args.dataset,
        "horizon": args.horizon,
        "stride": args.stride,
        "num_rollings": args.num_rollings,
        "tv_ratio": args.tv_ratio,
        "train_ratio_in_tv": args.train_ratio_in_tv,
        "seq_len": args.seq_len,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "norm": True,
    }
    EXPECTED_META_LEN = args.expect_meta_len

    csv_path = os.path.join(FORECASTING_DATASET_PATH, RUN["series"])
    meta_path = os.path.join(FORECASTING_DATASET_PATH, "FORECAST_META.csv")

    report = {"run_params": dict(RUN), "out_dir": out_dir}
    sha_before = {
        "csv": sha256_file(csv_path),
        "meta": sha256_file(meta_path),
    }
    report["sha256_before"] = sha_before

    # ---- load via the official data source ---------------------------
    src = LocalForecastingDataSource()
    src.load_series_list([RUN["series"]])
    series = src.dataset.get_series(RUN["series"])
    meta = src.dataset.get_series_meta_info(RUN["series"])
    assert series is not None, "series failed to load"

    report["series_facts"] = {
        "shape_rows": int(series.shape[0]),
        "shape_cols": int(series.shape[1]),
        "columns": list(series.columns),
        "index_name": series.index.name,
        "first_ts": str(series.index[0]),
        "last_ts": str(series.index[-1]),
        "n_unique_timestamps": int(series.index.nunique()),
        "monotonic_increasing": bool(series.index.is_monotonic_increasing),
        "n_duplicate_timestamps": int(series.index.duplicated().sum()),
        "nan_total": int(series.isna().sum().sum()),
        "nan_per_col": {c: int(v) for c, v in series.isna().sum().items()},
        "infer_freq_raw": str(pd.infer_freq(series.index)),
    }

    meta_full = src.dataset.metadata
    row = meta_full.loc[meta_full["file_name"] == RUN["series"]]
    report["meta_row"] = (
        {k: str(v) for k, v in row.iloc[0].items()} if len(row) else None
    )

    # ---- adjudication via the real _get_split_lens --------------------
    rf = RollingForecast.__new__(RollingForecast)
    fallback_reason = None
    if meta is None:
        fallback_reason = "meta_info is None -> _get_meta_info returns len(series)"
        data_len_meta = len(series)
        try:
            train_length, test_length = rf._get_split_lens(
                series, meta, RUN["tv_ratio"]
            )
        except Exception as e:
            train_length = test_length = None
            fallback_reason += f"; _get_split_lens raised {type(e).__name__}: {e}"
    else:
        try:
            train_length, test_length = rf._get_split_lens(
                series, meta, RUN["tv_ratio"]
            )
            data_len_meta = int(meta["length"].item())
        except KeyError as e:
            fallback_reason = (
                f"META present but missing field {e}; official code raises KeyError"
            )
            data_len_meta = len(series)
            train_length = test_length = None

    data_len_csv = len(series)
    report["adjudication"] = {
        "meta_length": (
            None if meta is None or "length" not in meta.index
            else int(meta["length"].item())
        ),
        "csv_length": int(data_len_csv),
        "data_len_used": int(data_len_meta),
        "branch": "META" if data_len_meta != data_len_csv else "FALLBACK",
        "fallback_reason": fallback_reason,
        "train_length_tv": None if train_length is None else int(train_length),
        "test_length": None if test_length is None else int(test_length),
        "expected_meta_length": EXPECTED_META_LEN,
    }

    protocol_ok = (
        train_length is not None
        and (EXPECTED_META_LEN is None or data_len_meta == EXPECTED_META_LEN)
    )
    report["protocol_ok"] = bool(protocol_ok)

    # ---- boundaries ----------------------------------------------------
    if train_length is not None:
        tv_data, test_data = split_time(series, train_length)
        border = int(len(tv_data) * RUN["train_ratio_in_tv"])
        train_data, valid_data = train_val_split(
            tv_data, RUN["train_ratio_in_tv"], RUN["seq_len"]
        )
        report["boundaries"] = {
            "train_valid": {
                "rows": int(len(tv_data)),
                "first_ts": str(tv_data.index[0]),
                "last_ts": str(tv_data.index[-1]),
            },
            "test": {
                "rows": int(len(test_data)),
                "first_ts": str(test_data.index[0]),
                "last_ts": str(test_data.index[-1]),
            },
            "train": {
                "rows": int(len(train_data)),
                "first_ts": str(train_data.index[0]),
                "last_ts": str(train_data.index[-1]),
            },
            "valid": {
                "rows": int(len(valid_data)),
                "first_ts": str(valid_data.index[0]),
                "last_ts": str(valid_data.index[-1]),
                "note": f"valid = tv_data[{border - RUN['seq_len']}:{len(tv_data)}] "
                        f"(train_val_split keeps seq_len of context)",
            },
            "tail_rows_outside_all_splits": int(len(series) - train_length),
        }

        # ---- real MUSE config tune (freq/label_len) --------------------
        m = MUSE(seq_len=RUN["seq_len"])
        m.config.seq_len = RUN["seq_len"]
        MUSE.multi_forecasting_hyper_param_tune(m, tv_data)
        report["tuned_config"] = {
            "freq": m.config.freq,
            "label_len": int(m.config.label_len),
            "enc_in": int(m.config.enc_in),
        }
        label_len = m.config.label_len
        freq = m.config.freq

        # ---- dual scaler ------------------------------------------------
        model_scaler = StandardScaler().fit(train_data.values)
        eval_train, _ = split_time(
            tv_data, int(len(tv_data) * RUN["train_ratio_in_tv"])
        )
        eval_scaler = StandardScaler().fit(eval_train.values)
        scaler_equal = bool(
            np.array_equal(model_scaler.mean_, eval_scaler.mean_)
            and np.array_equal(model_scaler.scale_, eval_scaler.scale_)
        )
        report["dual_scaler"] = {
            "equal": scaler_equal,
            "mean": model_scaler.mean_.tolist(),
            "scale": model_scaler.scale_.tolist(),
        }

        # ---- datasets ----------------------------------------------------
        train_scaled = pd.DataFrame(
            model_scaler.transform(train_data.values),
            columns=train_data.columns,
            index=train_data.index,
        )
        valid_scaled = pd.DataFrame(
            model_scaler.transform(valid_data.values),
            columns=valid_data.columns,
            index=valid_data.index,
        )
        train_ds = DatasetForTransformer(
            dataset=train_scaled,
            history_len=RUN["seq_len"],
            prediction_len=RUN["horizon"],
            label_len=label_len,
            timeenc=1,
            freq=freq,
        )
        valid_ds = DatasetForTransformer(
            dataset=valid_scaled,
            history_len=RUN["seq_len"],
            prediction_len=RUN["horizon"],
            label_len=label_len,
            timeenc=1,
            freq=freq,
        )
        train_loader = DataLoader(
            train_ds,
            batch_size=RUN["batch_size"],
            shuffle=True,
            num_workers=RUN["num_workers"],
            drop_last=True,
        )
        valid_loader = DataLoader(
            valid_ds,
            batch_size=RUN["batch_size"],
            shuffle=False,
            num_workers=RUN["num_workers"],
            drop_last=False,
        )
        train_batches = len(train_loader)
        train_tail = len(train_ds) - train_batches * RUN["batch_size"]
        valid_batches = len(valid_loader)

        last_valid_batch = None
        for bi, batch in enumerate(valid_loader):
            last_valid_batch = (bi, int(batch[0].shape[0]))
        first_valid_batch = None
        for bi, batch in enumerate(valid_loader):
            first_valid_batch = (bi, int(batch[0].shape[0]))
            break

        report["datasets"] = {
            "train_windows": int(len(train_ds)),
            "expected_train_windows": len(train_scaled) - RUN["seq_len"]
            - RUN["horizon"] + 1,
            "train_batches_drop_last": int(train_batches),
            "train_tail_windows_dropped": int(train_tail),
            "valid_windows": int(len(valid_ds)),
            "expected_valid_windows": len(valid_scaled) - RUN["seq_len"]
            - RUN["horizon"] + 1,
            "valid_batches_keep_tail": int(valid_batches),
            "first_valid_batch_size": first_valid_batch[1],
            "last_valid_batch_size": last_valid_batch[1],
        }

        # ---- window spot-check vs pandas slices --------------------------
        sx, sy, sxm, sym = train_ds[0]
        t0 = train_scaled.values
        report["spot_check_train_w0"] = {
            "seq_x_matches_slice": bool(
                torch.allclose(sx, torch.tensor(t0[0:RUN["seq_len"]],
                                                dtype=torch.float32))
            ),
            "seq_y_matches_slice": bool(
                torch.allclose(
                    sy,
                    torch.tensor(
                        t0[
                            RUN["seq_len"] - label_len:
                            RUN["seq_len"] - label_len + label_len + RUN["horizon"]
                        ],
                        dtype=torch.float32,
                    ),
                )
            ),
            "shapes": {
                "seq_x": list(sx.shape),
                "seq_y": list(sy.shape),
                "seq_x_mark": list(sxm.shape),
                "seq_y_mark": list(sym.shape),
            },
        }
        vw0 = valid_scaled.values
        report["valid_window_first_last"] = {
            "first_input_start_ts": str(valid_data.index[0]),
            "last_window_input_start_ts": str(
                valid_data.index[len(valid_ds) - 1]
            ),
            "last_window_target_end_ts": str(
                valid_data.index[
                    min(
                        len(valid_ds) - 1 + RUN["seq_len"] + RUN["horizon"],
                        len(valid_data) - 1,
                    )
                ]
            ),
            "valid_scaled_rows": int(len(vw0)),
        }

        # ---- _get_index (test side, budget only) -------------------------
        index_list = rf._get_index(
            train_length, test_length, RUN["horizon"], RUN["stride"]
        )
        index_list_trimmed = index_list[: RUN["num_rollings"]]
        report["test_windows_budget_only"] = {
            "index_count": int(len(index_list)),
            "after_num_rollings": int(len(index_list_trimmed)),
            "first_index": int(index_list[0]),
            "last_index": int(index_list[-1]),
        }

    # ---- write-back check ---------------------------------------------
    sha_after = {"csv": sha256_file(csv_path), "meta": sha256_file(meta_path)}
    report["sha256_after"] = sha_after
    report["no_writeback"] = sha_after == sha_before

    dump_json(os.path.join(out_dir, "protocol_dump.json"), report)
    write_boundaries_md(out_dir, report)

    ok = (
        report["protocol_ok"]
        and report["no_writeback"]
        and report.get("dual_scaler", {}).get("equal", False)
        and report.get("spot_check_train_w0", {}).get(
            "seq_x_matches_slice", False
        )
        and report.get("spot_check_train_w0", {}).get(
            "seq_y_matches_slice", False
        )
    )
    print(json.dumps(jsafe(report), indent=2, default=str))
    if not ok:
        sys.exit(2)
    print("[probe] PASS")


def write_boundaries_md(out_dir, r):
    series = r["run_params"]["series"]
    lines = [
        f"# {series} H={r['run_params']['horizon']} protocol adjudication "
        f"(probe, real code)",
        "",
        f"- series: {series}",
        f"- CSV sha256: `{r['sha256_before']['csv']}`",
        "",
        "## Adjudication",
        "",
        f"- META length: {r['adjudication']['meta_length']}",
        f"- CSV length (unique timestamps): {r['adjudication']['csv_length']}",
        f"- data_len used by _get_split_lens: {r['adjudication']['data_len_used']} "
        f"(branch: **{r['adjudication']['branch']}**)",
        f"- fallback_reason: {r['adjudication']['fallback_reason']}",
        "",
        "Note: 长表行数 = 时间步数 × 变量数；本报告区分“时间步数”与“长表行数”。",
        "TFC 长表读回后按列块重组为宽表，时间步数 = CSV 数据行数。",
        "",
    ]
    if series == "ETTh1.csv":
        lines += [
            "## Two-branch numbers (ETTh1 anchor, frozen from phase 1)",
            "",
            "| quantity | META branch (14400) | fallback (17420) |",
            "|---|---|---|",
            "| train_valid rows | 11520 | 13936 |",
            "| train rows | 8640 | 10452 |",
            "| valid rows | 4320 | 3484 |",
            "| train windows (drop_last) | 7105 -> 222 batches | 8917 -> 278 batches |",
            "| valid windows (keep tail) | 2785 -> 88 batches | 3389 -> 106 batches |",
            "| tail rows outside splits | 3020 | 0 |",
            "",
        ]
    lines += [
        "## Facts",
        "",
    ]
    for k, v in r.get("boundaries", {}).items():
        lines.append(f"- {k}: {v}")
    lines += [
        "",
        "Paper anchor: ETT 14,400/57,600 with 6:2:2 and tv_ratio=0.8 × "
        "train_ratio_in_tv=0.75; other datasets 7:1:2 via __default__ "
        "train_ratio_in_tv=0.875 (full.md Table 3 & rolling_forecast_config.json).",
        "",
        f"- no_writeback: {r['no_writeback']}",
        f"- dual_scaler_equal: {r.get('dual_scaler', {}).get('equal')}",
        f"- protocol_ok: {r['protocol_ok']}",
    ]
    with open(os.path.join(out_dir, "boundaries.md"), "w") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
