# FROZEN — MUSE ETT 复现 16 组 (四数据集 × H{96,192,336,720} × seed2021)

冻结批准: 2026-10-17 评审。冻结代码状态 = 本 commit
(repro/etth1-h96 @ 13e8fb3 + 本文件)。ETTh1_h96 复用阶段 1 冻结 run
(full_e10);新增 15 组为阶段 3 正式串行训练结果。

## 冻结判据 (每组全部满足)

自然早停结束 + 损失全有限 + 冻结指纹不变 + 新进程 verify_restore
ALL PASS(强制哈希: 数据 CSV vs dataset_sync.json、MAE 权重 vs
run_config.env.mae_sha,缺失必 FAIL)+ 独立重算 MSE |Δ| ≤ 1e-6。
16/16 组重验 ALL PASS;复算 |Δ| 实测 ≤ 2.4e-09。

## 冻结结果 (test, rolling_forecast 官方协议, vs 论文 Table 4 "MUSE (ours)")

论文值三位小数全部一致 (Δ=0.000);表内 MSE/MAE 为完整精度舍入显示。

| config | att | test MSE | test MAE | recompute |dMSE| | verify | 状态 |
|---|---|---|---|---|---|---|
| ETTh1_h192 | 1 | 0.387714 | 0.402916 | 3.57e-10 | PASS | verified |
| ETTh1_h336 | 1 | 0.405083 | 0.416787 | 1.42e-10 | PASS | verified |
| ETTh1_h720 | 1 | 0.406953 | 0.431275 | 4.45e-10 | PASS | verified |
| ETTh1_h96 | 1 | 0.338021 | 0.370871 | 3.63e-10 | PASS | 复用(冻结) |
| ETTh2_h96 | 1 | 0.264171 | 0.324819 | 5.01e-10 | PASS | verified |
| ETTh2_h192 | 1 | 0.319748 | 0.363070 | 8.09e-10 | PASS | verified |
| ETTh2_h336 | 1 | 0.341535 | 0.385416 | 1.03e-09 | PASS | verified |
| ETTh2_h720 | 1 | 0.373265 | 0.416423 | 2.41e-09 | PASS | verified |
| ETTm1_h96 | 1 | 0.285132 | 0.329860 | 2.77e-10 | PASS | verified |
| ETTm1_h192 | 1 | 0.317486 | 0.353116 | 7.64e-11 | PASS | verified |
| ETTm1_h336 | 1 | 0.338107 | 0.368657 | 2.6e-10 | PASS | verified |
| ETTm1_h720 | 1 | 0.379304 | 0.397418 | 3.39e-10 | PASS | verified |
| ETTm2_h96 | 1 | 0.161880 | 0.251997 | 1.71e-10 | PASS | verified |
| ETTm2_h192 | 1 | 0.221813 | 0.293143 | 1.13e-10 | PASS | verified |
| ETTm2_h336 | 1 | 0.269970 | 0.321466 | 6.71e-10 | PASS | verified |
| ETTm2_h720 | 1 | 0.326794 | 0.370901 | 4.63e-10 | PASS | verified |

## 冻结产物指针

- 运行目录与权重/数组: iota `/dev_data/wlt/MUSE_logs/repro/<run_id>/`
  (15 组) 与 `/dev_data/wlt/MUSE_logs/full_e10/` (ETTh1_h96 冻结);
  SHA256 清单见交付包。
- 交付包 (temp/): `muse_v2_main.tar.gz` sha256 a830443ba3b683028730dba3eb5a6f8bd087bbd4871963ba0bbce63ab758f7f3;
  `evidence_v2.tar.gz` sha256 0fe2968aebeae3d884b06dcc3e3a7926d2d5f5dd9ba4057176355e9b66ac3252;
  `arrays_v2_part01..57.tar.gz` + `arrays_v2_MANIFEST.json` (逐文件 SHA256,
  57/57 已验, 2.14GB: 15 组测试 preds/targets/window_border + 最终最佳验证 preds)。
- Electricity 分块等价探针 (r6): pass=true, vcr_nonzero=true,
  strict_pass=false (保留; 唯一超限更新量 1.22e-5 由噪声底梯度经 Adam
  更新解释, 豁免判据=两分支各自独立 <= 噪声底); 不属本冻结结果表。
- 预算口径: 上限h(训练+验证), 测试/导出/恢复单列; Electricity/H96
  = 75.17 + 5.63 = 80.80h。见 muse_v2_main summary/budget_table.md
  与 verify_budget.md。

## 范围外

六数据集 (Weather/Electricity/AQWan/CzeLan/Wind/ZafNoo) 24 组 formal
占位维持 pending, 不在本冻结范围, 须另行评审批准; Weather 现状
OOM-blocked (两次失败原始日志在 evidence 包)。
