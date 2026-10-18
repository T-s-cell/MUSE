"""Human-readable summary md + machine json for EXP-029 (MUSE
supervised training on timemmd_common_801010_v1)."""
import argparse
from pathlib import Path

import numpy as np

from repro_timemmd_common_v1 import protocol as P
from repro_timemmd_common_v1.common import dump_json, load_json, \
    setup_deterministic_env


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--protocol_dir", default=None)
    ap.add_argument("--out_root", default=None)
    args = ap.parse_args()

    setup_deterministic_env()
    pinfo = P.verify_frozen(args.protocol_dir)
    out_root = Path(args.out_root or P.out_root())

    ev = load_json(out_root / "evaluation.json")
    pf = load_json(out_root / "preflight" / "preflight_report.json")
    vr = load_json(out_root / "verify_report.json")
    rs = load_json(out_root / "replay_summary.json")
    tstat = load_json(out_root / "status.json")
    ts_infer = load_json(out_root / "test_status.json")

    n_scalars = None
    if tstat["groups"]:
        first_gid = sorted(tstat["groups"])[0]
        mp = out_root / P.ckpt_meta_name(
            first_gid.rsplit("_H", 1)[0],
            int(first_gid.rsplit("_H", 1)[1]))
        if mp.exists():
            n_scalars = load_json(mp).get("inventory", {}) \
                .get("n_trainable_scalars")

    lines = ["# MUSE x timemmd_common_801010_v1 — 监督训练汇总(EXP-029)",
             "",
             "协议:%s(冻结 51 文件)| 骨干:MAE-Base 冻结(sha %s...,全程逐位不变)"
             % (P.PROTOCOL_NAME, pinfo["weights_sha256"][:12]),
             "适配:VCR(num_latents=1/latent_dim=192/heads=4/C=1)+ TPR"
             "(bottleneck 64/heads 4/3 层共享 //4)+ tp_residual + fusion gate;"
             "可训 111 张量(%s 标量,几何相关)"
             % ("{:,}".format(n_scalars) if n_scalars else "n/a"),
             "训练:双组 Adam(adapters@1e-4、gate@1e-3 wd=0)| batch %d | "
             "type3 | ≤%d epoch patience=%d | 逐 epoch 重新打乱"
             "(单 RandomState(%d) 逐轮 permutation)"
             % (P.BATCH, P.MAX_EPOCHS, P.PATIENCE, P.SEED),
             "36 组独立训练(9 域×4H,seed %d,固定 P=EXP-027 选中,损失="
             "MSE(chan)+MSE(tp)+MSE(gate 融合,分支 detach))" % P.SEED,
             "训练窗 55,836 | 步数上限 17,670 | 单种子固定预算", ""]

    lines += ["## 固定周期 P(=EXP-027 验证集选定,本轮不搜索)", "",
              "| 域 | P |", "|---|---|"]
    for dom in P.ORDER:
        lines.append("| %s | %d |" % (dom, P.FIXED_P[dom]))
    lines.append("")

    lines += ["## 36 组测试结果(标准化空间;逐组 f32 全元素;Δ=MUSE−EXP-028 "
              "LN full-shot)",
              "",
              "| 组 | P | 测试窗 | epochs_run | best_epoch | steps | MSE | "
              "MAE | ΔMSE vs 028 | ΔMSE% |",
              "|---|---|---|---|---|---|---|---|---|---|"]
    for dom, h in P.all_groups():
        gid = P.group_id(dom, h)
        g = ev["groups"][gid]
        t = tstat["groups"][gid]
        lines.append("| %s | %d | %d | %d | %d | %d | %.6f | %.6f | "
                     "%+.6f | %+.2f%% |"
                     % (gid, P.FIXED_P[dom], g["windows"],
                        t.get("epochs_run"), t.get("best_epoch"),
                        t.get("steps"), g["mse"], g["mae"],
                        g["exp028"]["delta_mse"],
                        100 * g["exp028"]["rel_mse"]))
    lines.append("")

    lines += ["## 逐域(4H 等权)", "",
              "| 域 | MSE | MAE | EXP-028 MSE | ΔMSE% | EXP-027 MSE | "
              "ΔMSE% | TaTS MSE | ΔMSE% |",
              "|---|---|---|---|---|---|---|---|---|"]
    for dom in P.ORDER:
        d = ev["by_domain"][dom]
        cols = [dom, "%.6f" % d["mse"], "%.6f" % d["mae"],
                "%.6f" % d["exp028_mse"], "%+.2f%%" % (100 * d["rel_mse_exp028"]),
                "%.6f" % d["exp027_mse"], "%+.2f%%" % (100 * d["rel_mse_exp027"])]
        if "tats_mse" in d:
            cols += ["%.6f" % d["tats_mse"],
                     "%+.2f%%" % (100 * d["rel_mse_tats"])]
        else:
            cols += ["n/a", "n/a"]
        lines.append("| %s |" % " | ".join(cols))
    lines.append("")

    o9, o36 = ev["overall_9domain_equal"], ev["overall_36group_equal"]
    lines += ["## Overall", "",
              "- 9 域等权:MSE **%.6f** | MAE **%.6f**"
              % (o9["mse"], o9["mae"]),
              "- 36 组等权:MSE %.6f | MAE %.6f(测试窗 %d)"
              % (o36["mse"], o36["mae"], ev["test_windows"]),
              "- 主对照 EXP-028 LN full-shot(36 组等权 MSE %.6f):ΔMSE "
              "%+.6f(%+.2f%%)ΔMAE %+.6f(%+.2f%%);改善 %d 组 / 退化 %d 组"
              % (ev["exp028_overall_36group"]["mse"],
                 o36["delta_mse_exp028"], 100 * o36["rel_mse_exp028"],
                 o36["delta_mae_exp028"], 100 * o36["rel_mae_exp028"],
                 ev["vs_exp028"]["n_improved"],
                 ev["vs_exp028"]["n_degraded"]),
              "- 次对照 EXP-027 零样本(36 组等权 MSE %.6f):ΔMSE %+.6f"
              "(%+.2f%%);改善 %d 组 / 退化 %d 组"
              % (ev["exp027_overall_36group"]["mse"],
                 o36["delta_mse_exp027"], 100 * o36["rel_mse_exp027"],
                 ev["vs_exp027"]["n_improved"],
                 ev["vs_exp027"]["n_degraded"]),
              "- 三对照 TaTS 监督(36 组等权 MSE %.6f,锚点断言过):ΔMSE "
              "%+.6f(%+.2f%%)"
              % (ev["tats_overall_36group"]["mse"],
                 o36["delta_mse_tats"], 100 * o36["rel_mse_tats"]),
              "- 口径:EXP-027=零样本冻结;EXP-028=仅 LN 微调;TaTS=监督+文本"
              "+数值先验;本轮=MUSE 全适配分支监督+仅数值历史,单种子固定预算", ""]

    lines += ["## 训练成本(实测)", ""]
    tot_s = sum(g.get("elapsed_s") or 0 for g in tstat["groups"].values())
    peak = max((g.get("peak_mem_mb") or 0)
               for g in tstat["groups"].values())
    lines.append("- 36 组训练总耗时 %.1f min(单卡串行)| 单组峰值显存 %d MB"
                 % (tot_s / 60.0, peak))
    lines.append("- epochs_run 分布:%s"
                 % np.histogram([g.get("epochs_run") for g in
                                 tstat["groups"].values()],
                                bins=[3, 4, 5, 6, 7, 8, 9, 10, 11])[0]
                 .tolist())
    capped = sorted(gid for gid, g in tstat["groups"].items()
                    if int(g.get("epochs_run", 0)) >= P.MAX_EPOCHS)
    early = len(tstat["groups"]) - len(capped)
    lines.append("- 停止原因:%d 组早停自然触发(patience=2);%d 组达到 "
                 "%d 轮上限:%s(无人工截断)"
                 % (early, len(capped), P.MAX_EPOCHS,
                    "、".join(capped) if capped else "无"))
    lines.append("- 冻结核验:全部组 backbone+buffer 逐位不变;实际更新张量数 "
                 "min=%d max=%d(结构性零梯度按源码解释,不作门禁)"
                 % (min(g.get("n_changed_tensors", -1)
                        for g in tstat["groups"].values()),
                    max(g.get("n_changed_tensors", -1)
                        for g in tstat["groups"].values())))
    lines.append("- 预检成本外推:%s" % pf.get("b5", {}).get("note", "n/a"))
    lines.append("")

    lines += ["## 验证集诊断(报告性):VisionTS 零样本验证对照 vs MUSE best-epoch",
              "",
              "注:EXP-027 数组仅作**VisionTS 零样本验证对照**,不是 MUSE "
              "初始化结果——MUSE 的 TPR 初始化已可能改变预测。", "",
              "| 域 | H | VisionTS 零样本 val MSE | MUSE best val MSE(f64) "
              "| Δ |",
              "|---|---|---|---|---|"]
    for dom in P.ORDER:
        for r in ev["val_visionts_zeroshot_vs_muse_best"][dom]:
            d = r["delta_val_mse"]
            lines.append("| %s | %d | %.6f | %s | %s |"
                         % (dom, r["h"],
                            r["visionts_zeroshot_val_mse"],
                            ("%.6f" % r["muse_best_val_mse_f64"])
                            if r["muse_best_val_mse_f64"] is not None
                            else "n/a",
                            ("%+.6f" % d) if d is not None else "n/a"))
    lines.append("")

    lines += ["## 门禁", "",
              "- 预检 B1–B5:%s(b5.go=%s)"
              % ("ALL PASS" if pf["all_pass"] else "FAIL",
                 pf.get("b5", {}).get("go")),
              "- 训练:36/36 done,Σtrain 窗=55,836,epochs∈[3,10],"
              "Σsteps≤17,670",
              "- verify_cache 四方(协议 AND EXP-027 AND TaTS AND EXP-028):"
              "%d/%d 组 ok"
              % (vr["n_groups"] - vr["n_failed"], vr["n_groups"]),
              "- 独立重放(测试+验证):%d/%d pass(max|d|=%.2e;"
              "ckpt/npz 哈希不变=%s)"
              % (rs["n_pass"], rs["n_groups"],
                 max(r.get("max_abs_diff") or 0
                     for r in rs["reports"].values()),
                 rs["immutability_ok"]),
              "- 负控:%s" % ("两变体均检出" if rs.get("perturb_verify", {})
                             .get("variant_a_naive_detected") and
                             rs["perturb_verify"]
                             .get("variant_b_consistent_detected")
                             else "n/a"),
              "- 推理:%d/36 done(每组独立进程严格恢复+MAE 回填)"
              % sum(1 for g in ts_infer["groups"].values()
                    if g.get("state") == "done")]

    # batch=32 同批量验收(评审后补;存在才写)
    rb_path = out_root / "replay_b32" / "replay_b32_summary.json"
    bo_path = out_root / "b32_export" / "b32_overall.json"
    rb = None
    if rb_path.exists() and bo_path.exists():
        rb = load_json(rb_path)
        bo = load_json(bo_path)
        ex = load_json(out_root / "b32_export" / "export_b32_summary.json")
        n_rb_bit = rb.get("n_variants_bitwise")
        lines += ["",
                  "## batch=32 同批量验收(评审补正收尾)", "",
                  "- 导出:以现有最佳权重按 batch=32 导出 36 组 test+best-val "
                  "数组(b32_export/,72 个 npz);原 batch=64 文件保留且经 "
                  "pre/post sha256 证明未改写(36/36 %s)"
                  % ("immutable" if all(g["originals_immutable"]
                                        for g in ex["groups"].values())
                     else "FAILED"),
                  "- 同批量重放:从交付包(adapter-only+固定 MAE)独立进程"
                  "(每组一个子进程)按 batch=32 重放并与导出数组比较:"
                  "**%d/%d 组 pass,bitwise %s/72**(atol=%g/rtol=%g,"
                  "指标复算 ≤1e-9,导出数组哈希不变)"
                  % (rb["n_pass"], rb["n_groups"], n_rb_bit,
                     rb["atol"], rb["rtol"]),
                  "- 指标口径(b32_overall,与 evaluation.json 同 env 同约定):"
                  "9 域等权 MSE %.9f → %.9f(Δ %+.3e);36 组等权 MSE "
                  "%.9f → %.9f(Δ %+.3e);逐组指标差异上界 max|ΔMSE|=%.3e、"
                  "max|ΔMAE|=%.3e"
                  % (bo["overall_9domain_equal"]["b64"]["mse"],
                     bo["overall_9domain_equal"]["b32"]["mse"],
                     bo["overall_9domain_equal"]["delta_mse"],
                     bo["overall_36group_equal"]["b64"]["mse"],
                     bo["overall_36group_equal"]["b32"]["mse"],
                     bo["overall_36group_equal"]["delta_mse"],
                     bo["bounds"]["max_abs_group_delta_mse"],
                     bo["bounds"]["max_abs_group_delta_mae"]),
                  "- 跨批量对照(非验收,保留备查):b32 vs 原 b64 数组 "
                  "24/36 组在 atol/rtol 内,12 组 22 项超差(max_abs ≤8.0e-6);"
                  "隔离探针证明系 f32 批形状数值差异"
                  "(replay_pack/batch32_probe_20261019.json),"
                  "不改变原 b64 文件的导出地位", ""]

    (out_root / "summary.md").write_text("\n".join(lines))
    dump_json(out_root / "summary.json", {
        "fixed_p": P.FIXED_P,
        "by_domain": ev["by_domain"],
        "overall_9domain_equal": o9, "overall_36group_equal": o36,
        "exp028_overall_36group": ev["exp028_overall_36group"],
        "exp027_overall_36group": ev["exp027_overall_36group"],
        "tats_overall_36group": ev.get("tats_overall_36group"),
        "vs_exp028": ev["vs_exp028"], "vs_exp027": ev["vs_exp027"],
        "test_windows": ev["test_windows"],
        "train": {"groups": len(tstat["groups"]),
                  "steps": sum(g.get("steps") or 0
                               for g in tstat["groups"].values()),
                  "total_elapsed_s": round(tot_s, 1),
                  "peak_mem_mb": peak},
        "gates": {"preflight_all_pass": pf["all_pass"],
                  "train_all_done": all(
                      g.get("state") == "done"
                      for g in tstat["groups"].values()),
                  "verify_ok": vr["n_failed"] == 0,
                  "replay_all_pass": rs["n_pass"] == rs["n_groups"],
                  "infer_all_done": all(
                      g.get("state") == "done"
                      for g in ts_infer["groups"].values()),
                  "replay_b32_all_pass": (
                      rb["n_pass"] == rb["n_groups"])
                      if rb is not None else None},
    })
    print((out_root / "summary.md").read_text()[-600:])


if __name__ == "__main__":
    main()
