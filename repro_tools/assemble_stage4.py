"""Assemble the stage-4 (non-ETT H96 seed2021) audit package on iota.

Reads-only from the four run dirs; writes summary/verify_budget.md,
summary/artifacts_inventory.json, STAGE4_DELIVERY.md, then tars the
staging dir. Weights and test npz shards STAY on the server; the package
carries their paths + sizes + SHA256 + shard coverage only.
"""
import hashlib
import json
import os
import subprocess
import tarfile

LOGS = "/dev_data/wlt/MUSE_logs/stage4_h96"
REPO = "/dev_data/wlt/MUSE"
RUNS = ["Wind_h96_seed2021", "ZafNoo_h96_seed2021",
        "AQWan_h96_seed2021", "CzeLan_h96_seed2021"]
PAPER = {"Wind_h96": (0.864, 0.643), "ZafNoo_h96": (0.423, 0.392),
         "AQWan_h96": (0.739, 0.465), "CzeLan_h96": (0.170, 0.224)}
STAGE = os.path.join(LOGS, "stage4_audit_stage")
OUT_TAR = "/dev_data/wlt/MUSE_logs/stage4_h96/MUSE_stage4_nonett_h96_s2021.tar.gz"

FROZEN_SHA = "33eb9852a4f561c499bf1d59e5e5c991a032de31"
HEAD = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO,
                      capture_output=True, text=True).stdout.strip()


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load(path):
    with open(path) as f:
        return json.load(f)


os.makedirs(STAGE, exist_ok=True)
for d in ("summary", "queue", "probe", "runs", "code"):
    os.makedirs(os.path.join(STAGE, d), exist_ok=True)

# ---- per-run evidence + inventory ----
inventory = {"schema": "muse-repro/stage4-artifacts-v1", "arrays_note":
             "数组完整落盘服务器;本包只含路径+大小+SHA256+分片覆盖;"
             "用户验收状态:待用户核验", "runs": {}}
budget_lines = [
    "# 恢复/复验实测预算(stage4_h96,2026-10-17)",
    "",
    "口径:postverify_timing.json 逐子进程单调时钟时长+ISO 起止+退出码;"
    "verify_restore=验证预测恢复检查(新进程,best 强制哈希);"
    "recompute=测试指标复算。两者独立计时,非日志 mtime 推算。",
    "",
    "| run | restore s | restore 起-止 | restore rc | recompute s | "
    "recompute 起-止 | recompute rc |",
    "|---|---|---|---|---|---|---|",
]
total_restore = total_recompute = 0.0
for run in RUNS:
    src = os.path.join(LOGS, "repro", run)
    rdir = os.path.join(STAGE, "runs", run)
    os.makedirs(rdir, exist_ok=True)
    tim = load(os.path.join(src, "postverify_timing.json"))
    rec = load(os.path.join(src, "recompute_test_metrics.json"))
    vreps = sorted(f for f in os.listdir(src)
                   if f.startswith("verify_report_step"))
    vrep = load(os.path.join(src, vreps[-1]))
    summary = load(os.path.join(src, "summary.json"))
    rcfg = load(os.path.join(src, "run_config.json"))
    shards = load(os.path.join(src, "test_preds_shards.json"))
    best_rows = [ln.split(",")[0] for ln in
                 open(os.path.join(src, "best.csv")).read().splitlines()[1:]]
    best_epoch = best_rows[-1] if best_rows else None

    for f in ("run_config.json", "summary.json", "recompute_test_metrics.json",
              "postverify_timing.json", "test_preds_shards.json",
              "best.csv", "val.csv", "steps.csv", "audit_steps.json",
              "manifest.json", "env_snapshot.json",
              "postverify_verify_restore.log", "postverify_recompute.log",
              "train_stdout.log", "best_point_%s.json" % best_epoch,
              "best_preds_%s.npz" % best_epoch) + tuple(vreps):
        p = os.path.join(src, f)
        if os.path.exists(p):
            subprocess.run(["cp", p, os.path.join(rdir, f)], check=True)

    inv = {"out_dir": src, "files": {}}
    keep = ["best_model_%s.pt" % best_epoch, "best_preds_%s.npz" % best_epoch,
            "test_preds_shards.json"]
    keep += [s["file"] for s in shards.get("shards", [])]
    for f in keep:
        p = os.path.join(src, f)
        if os.path.exists(p):
            inv["files"][f] = {"bytes": os.path.getsize(p),
                               "sha256": sha256(p)}
    inv["shard_coverage"] = {
        "n_windows_total": shards.get("n_windows_total"),
        "preds_shape_full": shards.get("preds_shape_full"),
        "global_border": [shards.get("global_border_first"),
                          shards.get("global_border_last")],
        "shards": [{"file": s["file"], "n_windows": s["n_windows"],
                    "border": [s["border_first"], s["border_last"]]}
                   for s in shards.get("shards", [])]}
    inventory["runs"][run] = inv

    t = {x["step"]: x for x in tim}
    vr, rc_ = t["verify_restore"], t["recompute"]
    total_restore += vr["duration_s"]
    total_recompute += rc_["duration_s"]
    budget_lines.append(
        "| %s | %.1f | %s - %s | %d | %.1f | %s - %s | %d |" % (
            run, vr["duration_s"], vr["started_at"][11:], vr["finished_at"][11:],
            vr["exit_code"], rc_["duration_s"], rc_["started_at"][11:],
            rc_["finished_at"][11:], rc_["exit_code"]))

budget_lines += ["", "**四组合计: restore %.1f min + recompute %.1f s**"
                 % (total_restore / 60.0, total_recompute)]
with open(os.path.join(STAGE, "summary", "verify_budget.md"), "w") as f:
    f.write("\n".join(budget_lines) + "\n")
with open(os.path.join(STAGE, "summary", "artifacts_inventory.json"),
          "w") as f:
    json.dump(inventory, f, indent=2, ensure_ascii=False)

# ---- summaries + queue + probes + configs ----
for f in ("nonett_results.md", "ett16_results.md", "budget_table.md",
          "status_summary.md", "rollup.json"):
    p = os.path.join(LOGS, "summary", f)
    if os.path.exists(p):
        subprocess.run(["cp", p, os.path.join(STAGE, "summary", f)],
                       check=True)
subprocess.run(["cp", os.path.join(LOGS, "orchestration", "queue.json"),
                os.path.join(STAGE, "queue", "queue.json")], check=True)
subprocess.run(["cp", os.path.join(LOGS, "run_all.log"),
                os.path.join(STAGE, "queue", "run_all.log")], check=True)
for ds in ("Wind", "ZafNoo", "AQWan", "CzeLan"):
    pd = os.path.join(LOGS, "probe", ds)
    for f in os.listdir(pd):
        subprocess.run(["cp", os.path.join(pd, f),
                        os.path.join(STAGE, "probe", "%s__%s" % (ds, f))],
                       check=True)
subprocess.run(["cp", os.path.join(LOGS, "configs_manifest.json"),
                os.path.join(STAGE, "configs_manifest.json")], check=True)

# ---- code versions ----
diff = subprocess.run(
    ["git", "diff", FROZEN_SHA, HEAD, "--", "repro_tools/"],
    cwd=REPO, capture_output=True, text=True).stdout
with open(os.path.join(STAGE, "code", "stage4_vs_frozen.diff"), "w") as f:
    f.write(diff)
with open(os.path.join(STAGE, "code", "VERSIONS.txt"), "w") as f:
    f.write("frozen_base: %s (tag muse-etth16-frozen)\n"
            "stage4_head: %s (branch repro/nonett-h96-s2021)\n"
            "delta_commits: %s\n" % (
                FROZEN_SHA, HEAD, subprocess.run(
                    ["git", "log", "--oneline",
                     FROZEN_SHA + ".." + HEAD], cwd=REPO,
                    capture_output=True, text=True).stdout))

with open(os.path.join(STAGE, "STAGE4_DELIVERY.md"), "w") as f:
    f.write("""# MUSE 阶段4交付:非 ETT 四数据集 H96 seed2021 正式复现

- 基线: 冻结 commit %s (tag muse-etth16-frozen);阶段4分支头 %s
- 协议: 串行 训练→最佳验证权重测试→新进程恢复→指标复算;四组均 att1 一次成功
- 结果: 四组 test MSE/MAE 与论文 Table 4 H96 全部三位小数吻合(ΔMSE=ΔMAE=-0.000),
  见 summary/nonett_results.md;检查范围分开记录:
  verify_restore=验证预测恢复检查(恢复 ALL PASS,强制哈希),
  recompute=测试指标复算(|Δ|<=1e-6 全过,详见各 recompute_test_metrics.json)
- 成本: 见 summary/budget_table.md(训练/验证/测试分列,全程峰 vs warmup 读);
  恢复/复验实测见 summary/verify_budget.md
- 数组: 完整落盘服务器 iota(路径+大小+SHA256+分片覆盖见
  summary/artifacts_inventory.json);**用户验收状态:待用户核验**
- 失败证据: 无(四组均无 failed attempt;stage4_h96/repro/ 仅 4 个 att1 目录)
- 范围: Electricity 正式暂缓(成本)、Weather 维持 OOM 阻塞、
  H192/336/720/其他种子/消融均范围外,未执行
""" % (FROZEN_SHA, HEAD))

if os.path.exists(OUT_TAR):
    os.remove(OUT_TAR)
with tarfile.open(OUT_TAR, "w:gz") as tar:
    tar.add(STAGE, arcname="MUSE_stage4_nonett_h96_s2021")
print("tar:", OUT_TAR, os.path.getsize(OUT_TAR), "bytes")
print("sha256:", sha256(OUT_TAR))
subprocess.run(["rm", "-rf", STAGE], check=True)
