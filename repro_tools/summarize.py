"""Step-6 rollup: run-dir artifacts -> results/budget/status tables.

Reads queue.json + every referenced out_dir's artifacts (run_config.json,
summary.json, test_report.json, recompute_test_metrics.json,
verify_report_step*.json), plus the frozen legacy ETTh1/H96 dir, and emits:

  rollup.json        full machine-readable state per run
  ett16_results.md   16-group ETT table vs paper Table 4 ("MUSE (ours)")
  budget_table.md    measured (formal) / probe-extrapolated (preflight) budget
  status_summary.md  verified / running / pending / failed classification

Missing artifacts are tolerated (in-progress runs roll up as partial).
Pure stdlib; run anywhere the artifacts are visible:
    python -m repro_tools.summarize --logs-root /dev_data/wlt/MUSE_logs
"""

import argparse
import glob
import json
import os
import re

# Paper Table 4 (Appendix C.2), row "MUSE (ours)"; papers/MUSE/full.md L481.
# (mse, mae), 3-decimal paper precision. ZatNoo/CzLan are the paper's
# spellings of ZafNoo/CzeLan.
PAPER_TABLE4 = {
    "ETTh1": {96: (0.338, 0.371), 192: (0.388, 0.403),
              336: (0.405, 0.417), 720: (0.407, 0.431)},
    "ETTh2": {96: (0.264, 0.325), 192: (0.320, 0.363),
              336: (0.342, 0.385), 720: (0.373, 0.416)},
    "ETTm1": {96: (0.285, 0.330), 192: (0.317, 0.353),
              336: (0.338, 0.369), 720: (0.379, 0.397)},
    "ETTm2": {96: (0.162, 0.252), 192: (0.222, 0.293),
              336: (0.270, 0.321), 720: (0.327, 0.371)},
    "Electricity": {96: (0.129, 0.221), 192: (0.146, 0.240),
                    336: (0.164, 0.260), 720: (0.190, 0.283)},
    "Wind": {96: (0.864, 0.643), 192: (0.998, 0.727),
             336: (1.074, 0.769), 720: (1.135, 0.800)},
    "Weather": {96: (0.142, 0.197), 192: (0.188, 0.242),
                336: (0.233, 0.279), 720: (0.299, 0.326)},
    "AQWan": {96: (0.739, 0.465), 192: (0.791, 0.487),
              336: (0.809, 0.497), 720: (0.883, 0.526)},
    "ZafNoo": {96: (0.423, 0.392), 192: (0.475, 0.422),
               336: (0.513, 0.446), 720: (0.564, 0.468)},
    "CzeLan": {96: (0.170, 0.224), 192: (0.203, 0.255),
               336: (0.234, 0.283), 720: (0.238, 0.295)},
}

RUN_ID_RE = re.compile(
    r"^(?P<dataset>[A-Za-z]+\d*)_h(?P<horizon>\d+)_seed\d+"
    r"(?P<suffix>_probe|_g3|_r\d+|_probe_r\d+|_g3_r\d+)?$"
)


def load_json(path):
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except ValueError:
        return None


def parse_run_id(run_id):
    m = RUN_ID_RE.match(run_id)
    if not m:
        return None, None, ""
    return m.group("dataset"), int(m.group("horizon")), m.group("suffix") or ""


def config_id_of(run_id):
    ds, h, _ = parse_run_id(run_id)
    return f"{ds}_h{h}" if ds else None


def verify_ok(out_dir):
    reps = sorted(glob.glob(os.path.join(out_dir, "verify_report_step*.json")))
    if not reps:
        return None, 0, 0
    rep = load_json(reps[-1])
    if rep is None:
        return None, 0, len(reps)
    checks = rep.get("checks", [])
    ok = bool(checks) and all(bool(c.get("ok")) for c in checks)
    return ok, sum(bool(c.get("ok")) for c in checks), len(checks)


def pick_timing(timing, prefix):
    """Prefer the post-warmup range key over _all."""
    cands = {k: v for k, v in timing.items()
             if k.startswith(prefix) and isinstance(v, dict) and v.get("n")}
    if not cands:
        return None
    for k, v in sorted(cands.items()):
        if not k.endswith("_all"):
            return k, v
    return sorted(cands.items())[0][0], sorted(cands.values())[0]


def rollup_run(run_id, status, attempt, out_dir):
    r = {
        "run_id": run_id, "status": status, "attempt": attempt,
        "out_dir": out_dir, "exists": os.path.isdir(out_dir),
    }
    if not r["exists"]:
        return r
    rc = load_json(os.path.join(out_dir, "run_config.json"))
    s = load_json(os.path.join(out_dir, "summary.json"))
    tr = load_json(os.path.join(out_dir, "test_report.json"))
    rec = load_json(os.path.join(out_dir, "recompute_test_metrics.json"))
    r["kind"] = rc.get("kind") if rc else None
    r["config_id"] = config_id_of(run_id)

    if rc:
        r["dataset_file"] = rc.get("dataset_file")
        r["protocol"] = rc.get("protocol")
        r["adjustments"] = rc.get("adjustments")
    if s:
        r["global_step"] = s.get("global_step")
        r["stop_reason"] = s.get("stop_reason")
        r["fit_wall_s"] = s.get("fit_wall_s")
        r["peak_after_warmup"] = s.get("peak_after_warmup")
        r["finite_losses"] = s.get("finite_losses")
        r["loss_decomposition_max_absdiff"] = s.get(
            "loss_decomposition_max_absdiff")
        fp = s.get("frozen_fingerprint") or {}
        r["frozen_unchanged"] = not any(
            fp.get(k) for k in ("added", "changed", "missing"))
        r["n_epoch_validations"] = sum(
            1 for v in (s.get("val_records") or [])
            if v.get("tag") == "epoch_end")
        best = s.get("best") or {}
        r["best_val_mse"] = best.get("val_mse_fused")
        r["best_tag"] = best.get("tag")
        ti = s.get("train_info") or {}
        r["train_info"] = ti
        cfg = s.get("config_effective") or {}
        r["num_epochs"] = cfg.get("num_epochs")
        r["patience"] = cfg.get("patience")
        r["batch_size"] = cfg.get("batch_size")
        r["seq_len"] = cfg.get("seq_len")
        r["lr"] = cfg.get("lr")
        wt = pick_timing(s.get("timing") or {}, "wall_ms")
        ct = pick_timing(s.get("timing") or {}, "cuda_ms")
        r["wall_ms_stats_key"], r["wall_ms_stats"] = wt or (None, None)
        r["cuda_ms_stats_key"], r["cuda_ms_stats"] = ct or (None, None)
        r["official_epoch_validate_time_s"] = s.get(
            "official_epoch_validate_time_s")
        r["best_dump_time_s"] = s.get("best_dump_time_s")
        r["best_save_time_s"] = s.get("best_save_time_s")
        capped = [v for v in (s.get("val_records") or [])
                  if v.get("val_capped")]
        if capped:
            v = capped[-1]
            r["preflight_val"] = {
                "batches_timed": v.get("batches_timed"),
                "n_batches_full": v.get("n_batches_full"),
                "batch_ms_mean": v.get("batch_ms_mean"),
                "batch_ms_median": v.get("batch_ms_median"),
                "batch_ms_p95": v.get("batch_ms_p95"),
                "elapsed_s": v.get("elapsed_s"),
                "tail_batch": v.get("tail_batch"),
                "partial": v.get("partial"),
                "feeds_early_stop": v.get("feeds_early_stop"),
                "val_peak_alloc_mb": v.get("peak_alloc_mb"),
                "val_peak_reserved_mb": v.get("peak_reserved_mb"),
            }
        v_epoch = [v for v in (s.get("val_records") or [])
                   if v.get("tag") == "epoch_end"]
        if v_epoch:
            last = v_epoch[-1]
            r["last_epoch_val"] = {
                "val_mse_fused": last.get("val_mse_fused"),
                "elapsed_s": last.get("elapsed_s"),
                "train_peak_alloc_mb": last.get("train_peak_alloc_mb"),
                "train_peak_reserved_mb": last.get("train_peak_reserved_mb"),
                "val_peak_alloc_mb": last.get("peak_alloc_mb"),
            }
    if tr:
        mm = tr.get("mean_metrics") or []
        # canonical-16 layout: [mae,mse,rmse,...,mae_norm,mse_norm,rmse_norm,...]
        # paper metrics are the *_norm entries; legacy-3 is [mse, mae, rmse].
        if len(mm) == 16:
            r["mse_official"], r["mae_official"] = mm[9], mm[8]
            r["rmse_official"] = mm[10]
        elif len(mm) == 3:
            r["mse_official"], r["mae_official"] = mm[0], mm[1]
            r["rmse_official"] = mm[2]
        r["n_test_windows"] = tr.get("n_windows")
        r["preds_shape"] = tr.get("preds_shape")
        r["official_fit_s"] = tr.get("official_fit_s")
        r["official_total_inference_s"] = tr.get("official_total_inference_s")
    if rec:
        r["mse_recomputed"] = rec.get("mse_recomputed")
        r["mae_recomputed"] = rec.get("mae_recomputed")
        r["recompute_absdiff"] = rec.get("absdiff")
        r["recompute_pass"] = rec.get("pass")
        r["coverage_ok"] = rec.get("coverage_ok")
        if rec.get("mse_official") is not None:
            r["mse_official"] = rec.get("mse_official")
            r["mae_official"] = rec.get("mae_official")
    vok, npass, ntot = verify_ok(out_dir)
    r["verify_all_ok"] = vok
    r["verify_checks"] = f"{npass}/{ntot}"
    r["failure"] = load_json(os.path.join(out_dir, "failure.json"))

    ds, h, suffix = parse_run_id(run_id)
    r["paper"] = PAPER_TABLE4.get(ds, {}).get(h)
    if r.get("mse_official") is not None and r["paper"]:
        r["mse_vs_paper"] = r["mse_official"] - r["paper"][0]
        r["mae_vs_paper"] = r["mae_official"] - r["paper"][1]
    return r


def gates_for(r):
    """Kind-aware acceptance gates.

    formal: natural end + finite losses + frozen unchanged + verify ALL
    PASS + recompute <=1e-6 + test arrays present.
    preflight probe: stopped at cap + finite losses + frozen unchanged +
    NO test artifacts + capped val flagged partial/non-intrusive.
    controlled_stop (G3): stopped at cap + no test artifacts.
    """
    rid = r.get("run_id", "")
    kind = r.get("kind")
    if kind == "controlled_stop" or "_g3" in rid:
        return {
            "stop_max_steps": r.get("stop_reason") == "max_steps",
            "no_test_artifacts": not r.get("preds_shape"),
        }
    if kind == "preflight" or rid.endswith("_probe"):
        pv = r.get("preflight_val") or {}
        return {
            "stop_max_steps": r.get("stop_reason") == "max_steps",
            "finite_losses": r.get("finite_losses"),
            "frozen_unchanged": r.get("frozen_unchanged"),
            "no_test_artifacts": not r.get("preds_shape"),
            "capped_val_partial": bool(pv)
            and pv.get("partial") is True
            and pv.get("feeds_early_stop") is False,
        }
    return {
        "trained": r.get("stop_reason") in ("natural_end", None)
        and r.get("global_step"),
        "finite_losses": r.get("finite_losses"),
        "frozen_unchanged": r.get("frozen_unchanged"),
        "verify": r.get("verify_all_ok"),
        "recompute": r.get("recompute_pass"),
        "test_arrays": bool(r.get("preds_shape")),
    }


def gates_ok(r):
    return all(gates_for(r).values())


def classify(r):
    st = r.get("status")
    if st in ("done", "done_preflight"):
        ok = gates_ok(r)
        rid = r.get("run_id", "")
        if r.get("kind") == "controlled_stop" or "_g3" in rid:
            return "g3-ok" if ok else "g3-FAIL"
        if r.get("kind") == "preflight" or rid.endswith("_probe"):
            return "probe-verified" if ok else "probe-FAIL"
        return "verified" if ok else "done-gates-incomplete"
    if st == "failed":
        return "failed"
    if st in ("running", "postverify"):
        return "in-progress"
    if st == "pending":
        return "queued"
    return st or "unknown"


def fmt(x, nd=6):
    if x is None:
        return "-"
    if isinstance(x, float):
        return f"{x:.{nd}f}"
    return str(x)


def ms_table(rows, headers):
    out = ["| " + " | ".join(headers) + " |",
           "|" + "|".join(["---"] * len(headers)) + "|"]
    out.extend(rows)
    return "\n".join(out)


def ett16_table(rollups):
    rows = []
    for r in sorted(
        (x for x in rollups if x["config_id"] and
         x["config_id"].startswith("ETT")),
        key=lambda x: (x["config_id"].split("_")[0],
                       int(x["config_id"].rsplit("_h", 1)[1]),
                       x["attempt"]),
    ):
        p = r.get("paper")
        note = "复用(冻结)" if r.get("legacy") else classify(r)
        rows.append("| " + " | ".join([
            r["config_id"], str(r["attempt"]),
            fmt(r.get("mse_official")), fmt(r.get("mae_official")),
            fmt((r.get("recompute_absdiff") or {}).get("mse_absdiff"), 2),
            fmt(r.get("verify_all_ok")), fmt(r.get("best_val_mse")),
            fmt(r.get("n_epoch_validations")),
            fmt(p[0]) if p else "-", fmt(p[1]) if p else "-",
            fmt(r.get("mse_vs_paper"), 3), fmt(r.get("mae_vs_paper"), 3),
            note,
        ]) + " |")
    header = ("| config | att | test MSE | test MAE | recompute_dMSE | "
              "verify | best val | epochs | 论文MSE | 论文MAE | "
              "ΔMSE | ΔMAE | 状态 |")
    return (f"论文值精度 3 位小数(Table 4, 'MUSE (ours)');Δ = 复现 − 论文。\n"
            f"{ms_table(rows, header.split('|')[1:-1])}")


def budget_rows(rollups):
    rows = []
    for r in rollups:
        if not r.get("train_info") or r.get("config_id") is None:
            continue
        ti = r["train_info"]
        wm = r.get("wall_ms_stats") or {}
        wmean = wm.get("mean")
        bs = r.get("batch_size")
        tib = ti.get("train_batches_full")
        ep = r.get("num_epochs")
        src = ("实测" if r.get("kind") == "formal" and r.get("fit_wall_s")
               else ("预检外推" if r.get("kind") == "preflight" else "待测"))
        epoch_train_s = None
        if wmean and tib:
            epoch_train_s = tib * wmean / 1000.0
        est_total = None
        if epoch_train_s and ep:
            est_total = epoch_train_s * ep
        pv = r.get("preflight_val") or {}
        ev = r.get("last_epoch_val") or {}
        peak = (r.get("peak_after_warmup") or {}).get("peak_alloc_mb")
        val_peak = ev.get("val_peak_alloc_mb") or pv.get("val_peak_alloc_mb")
        rows.append("| " + " | ".join([
            r["run_id"], classify(r), src,
            fmt(ti.get("train_windows")), fmt(tib),
            fmt(ti.get("train_tail_samples")),
            fmt(ep), fmt(bs), fmt(r.get("seq_len")),
            fmt(wmean, 1), fmt(wm.get("median"), 1), fmt(wm.get("p95"), 1),
            fmt(epoch_train_s, 1),
            fmt(r.get("official_epoch_validate_time_s"), 1),
            fmt(pv.get("batch_ms_mean"), 1),
            fmt(pv.get("n_batches_full")),
            fmt(peak / 1024.0, 1) if peak else "-",
            fmt(val_peak / 1024.0, 1) if val_peak else "-",
            fmt(r.get("fit_wall_s"), 0),
            fmt(est_total / 3600.0, 2) if est_total else "-",
        ]) + " |")
    header = ("| run_id | 状态 | 来源 | train窗 | 满批 | 尾样 | max_ep | "
              "bs | L | wall_ms均值 | 中位 | P95 | 训练s/ep | "
              "验证s(ep) | 预检val_ms/b | 预检满批数 | 峰值GiB(warmup后) | "
              "验证峰GiB | 实测fit_s | 估上限h |")
    return ms_table(rows, header.split("|")[1:-1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs-root", default="/dev_data/wlt/MUSE_logs")
    ap.add_argument("--queue", default=None,
                    help="default <logs-root>/orchestration/queue.json")
    ap.add_argument("--legacy-etth1-h96", default=None,
                    help="frozen full_e10 dir -> ETTh1_h96 复用 row")
    ap.add_argument("--out-dir", default=None,
                    help="default <logs-root>/summary")
    args = ap.parse_args()
    queue_path = args.queue or os.path.join(
        args.logs_root, "orchestration", "queue.json")
    out_dir = args.out_dir or os.path.join(args.logs_root, "summary")
    os.makedirs(out_dir, exist_ok=True)

    rollups = []
    queue = load_json(queue_path) or {"entries": []}
    seen = set()
    for e in queue["entries"]:
        rid = e["run_id"]
        seen.add(rid)
        rollups.append(rollup_run(
            rid, e.get("status"), e.get("attempt"), e.get("out_dir")))
    if args.legacy_etth1_h96:
        r = rollup_run("ETTh1_h96_seed2021", "done", 1,
                       args.legacy_etth1_h96)
        r["legacy"] = True
        rollups.append(r)

    for r in rollups:
        r["classification"] = classify(r)
        if r.get("status") in ("done", "done_preflight"):
            gates, _ = gates_for(r), gates_ok(r)
            r["gates"] = gates
            r["gates_all_ok"] = all(gates.values())

    rollup_path = os.path.join(out_dir, "rollup.json")
    with open(rollup_path, "w") as f:
        json.dump({"queue": queue_path, "n_runs": len(rollups),
                   "runs": rollups}, f, indent=2, sort_keys=True)

    ett = [r for r in rollups
           if (r.get("config_id") or "").startswith("ETT")
           and "_g3" not in r["run_id"]
           and r.get("kind") in (None, "formal")]
    with open(os.path.join(out_dir, "ett16_results.md"), "w") as f:
        f.write("# ETT 16 组结果 vs 论文 Table 4\n\n" + ett16_table(ett) + "\n")
    with open(os.path.join(out_dir, "budget_table.md"), "w") as f:
        f.write("# 预算表(实测/预检外推)\n\n"
                "上限=满轮数×实测单轮;早停可提前,不确定性单列;"
                "ETTh1 数字不外推。\n\n" + budget_rows(rollups) + "\n")

    counts = {}
    for r in rollups:
        counts[r["classification"]] = counts.get(r["classification"], 0) + 1
    with open(os.path.join(out_dir, "status_summary.md"), "w") as f:
        lines = ["# 状态汇总", "",
                 f"queue: {queue_path}", ""]
        for k in sorted(counts):
            lines.append(f"- {k}: {counts[k]}")
        lines.append("")
        for r in rollups:
            gates = r.get("gates") or {}
            bad = [k for k, v in gates.items() if not v] \
                if gates else None
            lines.append(
                f"- `{r['run_id']}` att{r['attempt']} "
                f"[{r['classification']}] {r['status']}"
                + (f" FAIL-GATES:{','.join(bad)}" if bad else ""))
        f.write("\n".join(lines) + "\n")

    print(f"runs: {len(rollups)}; classification: {counts}")
    print(f"-> {rollup_path}")
    print(f"-> {out_dir}/{{ett16_results,budget_table,status_summary}}.md")
    fails = [r["run_id"] for r in rollups
             if "FAIL" in r["classification"]
             or r["classification"] in ("failed", "done-gates-incomplete")]
    if fails:
        print(f"attention: {fails}")


if __name__ == "__main__":
    main()
