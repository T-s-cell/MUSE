"""Run protocol_probe for every manifest config of the given datasets.

Sequential, CPU-only, read-only (safe alongside a running training queue):
one protocol_probe subprocess per (dataset, horizon), arguments derived from
configs_manifest.json so nothing is retyped per dataset. Each probe's stdout
is captured to <out-root>/<config_id>/probe_stdout.log and its
protocol_dump.json is collected into sweep_report.json + sweep_table.md.
One failing probe never stops the sweep.

    python -m repro_tools.protocol_sweep \
        --manifest configs_manifest.json --repo /dev_data/wlt/MUSE \
        --datasets Weather.csv Electricity.csv AQWan.csv CzeLan.csv Wind.csv \
                   ZafNoo.csv \
        --out-root /dev_data/wlt/MUSE_logs/protocol_sweep
"""

import argparse
import json
import os
import subprocess
import sys

from .sync_datasets import EXPECTED_META_LEN


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="configs_manifest.json")
    ap.add_argument("--repo", default="/dev_data/wlt/MUSE")
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--datasets", nargs="*", required=True)
    ap.add_argument("--out-root", default="repro_out/protocol_sweep")
    args = ap.parse_args()

    with open(args.manifest) as f:
        manifest = json.load(f)
    wanted = set(args.datasets)
    cfgs = [c for c in manifest["configs"]
            if c["dataset_file"] in wanted]
    os.makedirs(args.out_root, exist_ok=True)

    results = []
    for cfg in cfgs:
        cid = cfg["config_id"]
        out_dir = os.path.join(args.out_root, cid)
        os.makedirs(out_dir, exist_ok=True)
        hp = cfg["eff"]["model_hyper_params"]
        argv = [
            args.python, "-m", "repro_tools.protocol_probe",
            "--dataset", cfg["dataset_file"],
            "--horizon", str(cfg["horizon"]),
            "--seq-len", str(hp["seq_len"]),
            "--batch-size", str(hp["batch_size"]),
            "--train-ratio-in-tv", str(cfg["train_ratio_in_tv_resolved"]),
            "--expect-meta-len", str(EXPECTED_META_LEN[cfg["dataset_file"]]),
            "--out-dir", out_dir,
        ]
        if cfg.get("tv_ratio") is not None:
            argv += ["--tv-ratio", str(cfg["tv_ratio"])]
        if cfg.get("stride") is not None:
            argv += ["--stride", str(cfg["stride"])]
        if cfg.get("num_rollings") is not None:
            argv += ["--num-rollings", str(cfg["num_rollings"])]
        log_path = os.path.join(out_dir, "probe_stdout.log")
        print(f"[sweep] {cid}: {' '.join(argv)}")
        with open(log_path, "w") as log:
            rc = subprocess.call(argv, cwd=args.repo, stdout=log,
                                 stderr=subprocess.STDOUT)
        dump = None
        dump_path = os.path.join(out_dir, "protocol_dump.json")
        if os.path.exists(dump_path):
            with open(dump_path) as f:
                dump = json.load(f)
        results.append({
            "config_id": cid,
            "dataset_file": cfg["dataset_file"],
            "horizon": cfg["horizon"],
            "exit_code": rc,
            "protocol_ok": (dump or {}).get("protocol_ok"),
            "no_writeback": (dump or {}).get("no_writeback"),
            "out_dir": out_dir,
            "dump": dump,
        })
        print(f"[sweep] {cid}: exit={rc} "
              f"protocol_ok={(dump or {}).get('protocol_ok')}")

    bad = [r for r in results if r["exit_code"] != 0]
    report = {
        "manifest": args.manifest,
        "datasets": args.datasets,
        "n_probes": len(results),
        "n_bad": len(bad),
        "all_ok": not bad,
        "results": results,
    }
    with open(os.path.join(args.out_root, "sweep_report.json"), "w") as f:
        json.dump(report, f, indent=2, sort_keys=True)

    lines = [
        "| config | dataset | H | exit | protocol_ok | no_writeback |",
        "|---|---|---|---|---|---|",
    ]
    for r in results:
        lines.append(
            f"| {r['config_id']} | {r['dataset_file']} | {r['horizon']} | "
            f"{r['exit_code']} | {r['protocol_ok']} | {r['no_writeback']} |")
    with open(os.path.join(args.out_root, "sweep_table.md"), "w") as f:
        f.write("\n".join(lines) + "\n")

    print(f"[sweep] {len(results)} probes, {len(bad)} bad "
          f"-> {args.out_root}/sweep_report.json")
    if bad:
        for r in bad:
            print(f"[sweep] BAD {r['config_id']}: exit={r['exit_code']} "
                  f"(log {r['out_dir']}/probe_stdout.log)")
        sys.exit(1)


if __name__ == "__main__":
    main()
