"""Build the traceable 40-group config manifest from the official scripts.

Parses every invocation in scripts/multivariate_forecast/muse/*.sh, records
the RAW command verbatim, and derives the EFFECTIVE configuration where the
only permitted adjustments are machine-specific paths (MAE checkpoint,
save-path). Emits configs_manifest.json with a cross-check against the
review-pinned ETT parameter table.

Pure stdlib; run locally or on the server:
    python -m repro_tools.configs_manifest \
        --scripts-dir scripts/multivariate_forecast/muse \
        --config-json config/rolling_forecast_config.json \
        --mae-path /dev_data/wlt/dmmv_rebuild/mae/mae_visualize_vit_base.pth \
        --out configs_manifest.json
"""

import argparse
import glob
import json
import os
import re
import shlex

# Review-pinned cross-check table (from the phase directive), used ONLY to
# validate the parsed manifest — never as a runtime configuration source.
CHECK_ETT = {
    "ETTh1.csv": {"seq_len": 1440, "lr": [1e-5, 2e-6, 1e-5, 1e-5]},
    "ETTh2.csv": {"seq_len": 1728, "lr": [1e-5, 1e-5, 1e-5, 1e-5]},
    "ETTm1.csv": {"seq_len": 2304, "lr": [1e-5, 5e-6, 1e-5, 1e-5]},
    "ETTm2.csv": {"seq_len": 4032, "lr": [1e-4, 5e-6, 5e-6, 5e-6]},
}
HORIZONS = [96, 192, 336, 720]
CHECK_ELEC = {
    "seq_len": 2304, "lr": 1e-4, "use_variable_chunk": True,
    "batch_size": {96: 16, 192: 16, 336: 32, 720: 16},
    "num_epochs": {96: 6, 192: 6, 336: 8, 720: 6},
}


def parse_line(line):
    toks = shlex.split(line)
    flag = None
    out = {"argv_flags": {}}
    for tok in toks:
        if tok.startswith("--"):
            flag = tok[2:]
            out["argv_flags"][flag] = True
        elif flag is not None:
            out["argv_flags"][flag] = tok
            flag = None
    hp = json.loads(out["argv_flags"]["model-hyper-params"])
    sa = json.loads(out["argv_flags"]["strategy-args"])
    out["model_hyper_params"] = hp
    out["strategy_args"] = sa
    return out


def machine_adjustments(hp, save_path, mae_path, out_save_path):
    adj = []
    eff_hp = dict(hp)
    if hp.get("checkpoint_path") and hp["checkpoint_path"] != mae_path:
        adj.append({
            "field": "checkpoint_path",
            "raw": hp["checkpoint_path"],
            "effective": mae_path,
            "reason": "machine-specific pretrained-weight path",
        })
        eff_hp["checkpoint_path"] = mae_path
    if save_path != out_save_path:
        adj.append({
            "field": "save_path",
            "raw": save_path,
            "effective": out_save_path,
            "reason": "per-run output isolation on this machine",
        })
    return eff_hp, adj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scripts-dir", default="scripts/multivariate_forecast/muse")
    ap.add_argument("--config-json", default="config/rolling_forecast_config.json")
    ap.add_argument("--mae-path", required=True)
    ap.add_argument("--out", default="configs_manifest.json")
    args = ap.parse_args()

    with open(args.config_json) as f:
        strat_cfg = json.load(f)
    eval_cfg = strat_cfg["evaluation_config"]
    sa_cfg = eval_cfg["strategy_args"]
    tri_tv = sa_cfg["train_ratio_in_tv"]
    if not isinstance(tri_tv, dict):
        tri_tv = {"__default__": tri_tv}

    entries = []
    for sh in sorted(glob.glob(os.path.join(args.scripts_dir, "*.sh"))):
        with open(sh) as f:
            lines = [ln.strip() for ln in f if ln.strip().startswith("python")]
        for ln_no, line in enumerate(lines, 1):
            p = parse_line(line)
            fl = p["argv_flags"]
            hp = p["model_hyper_params"]
            dataset = fl["data-name-list"]
            horizon = int(p["strategy_args"]["horizon"])
            stem = os.path.splitext(dataset)[0]
            save_path = fl.get("save-path", "")
            out_save_path = f"repro/{stem}_h{horizon}_seed2021"
            eff_hp, adj = machine_adjustments(
                hp, save_path, args.mae_path, out_save_path
            )
            entries.append({
                "config_id": f"{stem}_h{horizon}",
                "script_ref": f"{sh}#invocation{ln_no}",
                "dataset_file": dataset,
                "horizon": horizon,
                "train_ratio_in_tv_resolved": tri_tv.get(
                    dataset, tri_tv["__default__"]
                ),
                "train_ratio_in_tv_source": (
                    "per-dataset" if dataset in tri_tv else "__default__"
                ),
                "tv_ratio": sa_cfg.get("tv_ratio"),
                "stride": sa_cfg.get("stride"),
                "num_rollings": sa_cfg.get("num_rollings"),
                "seed": sa_cfg.get("seed", 2021),
                "raw": {
                    "command": line,
                    "model_hyper_params": hp,
                    "deterministic": fl.get("deterministic"),
                    "gpus": fl.get("gpus"),
                    "num_workers_flag": fl.get("num-workers"),
                    "has_num_workers_flag": "num-workers" in fl,
                    "timeout": fl.get("timeout"),
                    "save_path": save_path,
                },
                "eff": {
                    "model_hyper_params": eff_hp,
                    "save_path": out_save_path,
                },
                "adjustments": adj,
            })

    # ---- cross-check vs review-pinned tables -----------------------------
    problems = []
    by_id = {e["config_id"]: e for e in entries}
    for ds, chk in CHECK_ETT.items():
        for h, lr in zip(HORIZONS, chk["lr"]):
            e = by_id.get(f"{os.path.splitext(ds)[0]}_h{h}")
            if e is None:
                problems.append(f"missing {ds} h{h}")
                continue
            hp = e["eff"]["model_hyper_params"]
            if hp["seq_len"] != chk["seq_len"]:
                problems.append(f"{ds} h{h} seq_len {hp['seq_len']}")
            if abs(hp["lr"] - lr) > 0 and hp["lr"] != lr:
                problems.append(f"{ds} h{h} lr {hp['lr']} != {lr}")
            if hp["batch_size"] != 32 or hp["num_epochs"] != 10 \
                    or hp["patience"] != 2:
                problems.append(f"{ds} h{h} batch/epochs/patience")
            if hp.get("use_variable_chunk"):
                problems.append(f"{ds} h{h} unexpected use_variable_chunk")
    for h in HORIZONS:
        e = by_id.get(f"Electricity_h{h}")
        if e is None:
            problems.append(f"missing Electricity h{h}")
            continue
        hp = e["eff"]["model_hyper_params"]
        if hp["seq_len"] != CHECK_ELEC["seq_len"] or hp["lr"] != CHECK_ELEC["lr"]:
            problems.append(f"Electricity h{h} seq_len/lr")
        if hp["batch_size"] != CHECK_ELEC["batch_size"][h] \
                or hp["num_epochs"] != CHECK_ELEC["num_epochs"][h]:
            problems.append(f"Electricity h{h} batch/epochs")
        if not hp.get("use_variable_chunk"):
            problems.append(f"Electricity h{h} missing use_variable_chunk")

    manifest = {
        "n_configs": len(entries),
        "check_problems": problems,
        "check_passed": not problems,
        "strategy_config_digest": {
            "tv_ratio": sa_cfg.get("tv_ratio"),
            "train_ratio_in_tv": tri_tv,
            "stride": sa_cfg.get("stride"),
            "num_rollings": sa_cfg.get("num_rollings"),
            "seed": sa_cfg.get("seed", 2021),
            "deterministic_cli": "full",
        },
        "configs": entries,
    }
    with open(args.out, "w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
    print(f"configs: {len(entries)}; check_problems: {problems or 'NONE'}")
    datasets = sorted({e["dataset_file"] for e in entries})
    print("datasets:", datasets)
    if problems or len(entries) != 40:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
