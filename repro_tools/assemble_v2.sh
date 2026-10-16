#!/bin/bash
# Assemble the MUSE review-fix v2 main package on iota.
# Usage: bash assemble_v2.sh <reverified_run_count> <reverified_pass_count>
set -euo pipefail
N_RUNS="${1:-?}"
N_PASS="${2:-?}"
REPRO=/dev_data/wlt/MUSE_logs/repro
P=/dev_data/wlt/MUSE_logs/packages/muse_v2
SUM=/tmp/sum_final
rm -rf "$P" "$SUM"
mkdir -p "$P/verify_reports" "$P/summary"

# 1. regenerated summary tables (post-sweep, gates final)
cd /dev_data/wlt/MUSE
/dev_data/wlt/conda/envs/muse/bin/python -m repro_tools.summarize \
  --legacy-etth1-h96 /dev_data/wlt/MUSE_logs/full_e10 \
  --out-dir "$SUM" >/dev/null
cp "$SUM"/ett16_results.md "$SUM"/budget_table.md "$SUM"/status_summary.md \
   "$P/summary/" 2>/dev/null || true
cp "$SUM"/*.json "$P/summary/" 2>/dev/null || true

# 2. new forced-hash verify reports (one per run, incl. frozen legacy)
for d in "$REPRO"/*/ ; do
  name=$(basename "$d")
  [ "$name" = "full_e10" ] && d=/dev_data/wlt/MUSE_logs/full_e10 && name=full_e10
  for f in "$d"/verify_report_step*.json; do
    [ -e "$f" ] || continue
    # keep only the newest report per run (sweep overwrote old buggy ones)
    newest=$(ls -t "$d"/verify_report_step*.json 2>/dev/null | head -1)
    cp "$newest" "$P/verify_reports/${name}__$(basename "$newest")"
    break
  done
done
legacy_newest=$(ls -t /dev_data/wlt/MUSE_logs/full_e10/verify_report_step*.json 2>/dev/null | head -1)
[ -n "$legacy_newest" ] && cp "$legacy_newest" "$P/verify_reports/full_e10__$(basename "$legacy_newest")"

# 3. per-run recompute + small configs for cross-checking
mkdir -p "$P/recompute"
for d in "$REPRO"/*/ ; do
  name=$(basename "$d")
  for f in recompute_test_metrics.json test_report.json; do
    [ -f "$d/$f" ] && cp "$d/$f" "$P/recompute/${name}__$f"
  done
done
cp /dev_data/wlt/MUSE/dataset/forecasting/dataset_sync.json "$P/"
cp /dev_data/wlt/MUSE_logs/orchestration/queue.json "$P/"
cp /dev_data/wlt/MUSE_logs/gates/chunk_probe_r5/chunk_probe_report.json \
   "$P/chunk_probe_report.json"

cat > "$P/REVIEW_FIXES.md" <<EOF
# MUSE 复现 评审修正 v2 主包

对应评审五项修正;数组包 arrays_v2(57 part)与证据包 evidence_v2 单独交付。
分支 repro/etth1-h96 @ $(cd /dev_data/wlt/MUSE && git rev-parse HEAD)。

- R1 哈希强校验: verify_restore 重写为强制比较(run_config.env.mae_sha
  与 dataset_sync.json dst_sha256 为期望值,缺失即 FAIL,永不空过);
  报告新增 hashes + metadata_hashes(run_config/manifest/point_meta 的
  SHA256)。全部 16 组(15 新 + 冻结 full_e10 --legacy)重验:
  **$N_RUNS 组重验,$N_PASS 组 ALL PASS**(见 verify_reports/)。
- R2 分块等价: chunk_probe r5 pass=true, vcr_nonzero=true,
  strict_pass=false(仅 1 元素 Adam 噪声底豁免,exempt_summary 全记录);
  三损失分别比较全部 <=4.8e-7;VCR 直接检查修正张量
  (init 精确 0 为零初始化构造,更新后 0.0063/0.0154/0.0268 非零,
  去零偏移 0.137-0.240 >> atol)。见 chunk_probe_report.json。
- R3 预测数组: arrays_v2 57 个 <50MB tar.gz,含 15 组测试
  preds/targets/window_border + 最终最佳验证 preds(按 argmin
  val_mse_fused 选轮,不重复交中间最佳点),MANIFEST.json 逐文件 SHA256。
- R4 重试与失败证据: evidence_v2.tar.gz(AQWan/CzeLan att1 失败 +
  att2 成功目录、Weather 两次 OOM 原始日志、queue.json、探针 r3/r4/r5)。
- R5 预算口径: budget_table.md 重写——验证s/轮=累计/轮数(预控行按
  限批吞吐外推并标注);上限h(训练+验证);测试h(推理+导出)单列;
  导出累计s 单列;全程峰GiB(训练) 与 warmup读GiB 分列;恢复/复验
  预算见 summary/verify_budget.md。Electricity/H96 上限 =
  75.17(训练) + 5.63(验证) = 80.80h,测试/导出/恢复另计——75h 不再
  作为完整预算上限。

已知偏差: 测试数组 ~2.1GB 超单包限制,按 <50MB 分 57 包交付(计划内);
权重与 npz 留 iota /dev_data/wlt/MUSE_logs/repro/(路径+SHA256 见 v1 清单)。
EOF

cd /dev_data/wlt/MUSE_logs/packages
tar czf muse_v2_main.tar.gz muse_v2
sha256sum muse_v2_main.tar.gz
du -sh muse_v2_main.tar.gz
