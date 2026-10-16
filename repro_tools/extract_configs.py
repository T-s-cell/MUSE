"""Extract stage-4 configs (non-ETT H96 seed2021) from the 40-group manifest.

Two modes:
  extract (default) — filter the requested config_ids, rewrite
      eff.save_path to an ABSOLUTE per-stage path (child cwd is --repo,
      so stage-3's relative "repro/<id>_seed2021" would collide), record
      the rewrite in adjustments, keep top-level provenance keys, and
      write the stage-4 manifest.
  --preview — reuse orchestrate.build_argv + the strategy_args backfill
      to print each config's effective protocol and exact argv, with
      hard assertions seed=2021 / stride=1 / num_rollings=48000 (and
      horizon=96). Run this BEFORE launching; orchestrate --dry-run
      only shows queue transitions, not argv or run_config.
"""

import argparse
import json
import sys

from .orchestrate import build_argv

STAGE4_KEYS = ("seed", "stride", "num_rollings")


def backfilled_sa(cfg):
    sa = cfg.get("strategy_args_resolved")
    if not sa:
        sa = {k: cfg[k] for k in STAGE4_KEYS if k in cfg}
    return sa


def extract(manifest, config_ids, save_path_root):
    by_id = {c["config_id"]: c for c in manifest["configs"]}
    missing = [cid for cid in config_ids if cid not in by_id]
    if missing:
        raise SystemExit(f"config_ids not in manifest: {missing}")
    configs = []
    for cid in config_ids:
        cfg = by_id[cid]
        old_sp = cfg["eff"]["save_path"]
        new_sp = f"{save_path_root.rstrip('/')}/{cid}_seed2021"
        cfg = json.loads(json.dumps(cfg))
        cfg["eff"]["save_path"] = new_sp
        cfg["adjustments"] = list(cfg.get("adjustments", [])) + [
            f"stage4: eff.save_path '{old_sp}' -> '{new_sp}' "
            "(absolute, per-stage logs root; child cwd is repo so the "
            "stage-3 relative path would collide)"
        ]
        for k in STAGE4_KEYS:
            if k not in cfg:
                raise SystemExit(f"{cid}: missing top-level '{k}'")
        configs.append(cfg)
    out = {
        "check_passed": manifest["check_passed"],
        "check_problems": manifest["check_problems"],
        "strategy_config_digest": manifest.get("strategy_config_digest"),
        "n_configs": len(configs),
        "configs": configs,
    }
    return out


def preview(configs, python_exe):
    failed = False
    for cfg in configs:
        cid = cfg["config_id"]
        sa = backfilled_sa(cfg)
        hp = cfg["eff"]["model_hyper_params"]
        checks = {
            "seed": (sa.get("seed"), 2021),
            "stride": (sa.get("stride"), 1),
            "num_rollings": (sa.get("num_rollings"), 48000),
            "horizon": (cfg["horizon"], 96),
            "seq_len": (int(hp["seq_len"]), int(hp["seq_len"])),
        }
        ok = True
        for k, (got, want) in checks.items():
            if k in ("seed", "stride", "num_rollings", "horizon") and got != want:
                print(f"[preview] FAIL {cid}: {k}={got!r} expected {want!r}")
                ok = False
        if cfg["eff"]["save_path"].endswith(f"/{cid}_seed2021") is False:
            print(f"[preview] FAIL {cid}: save_path={cfg['eff']['save_path']!r}")
            ok = False
        argv = build_argv(cfg, python_exe)
        print(f"[preview] {cid}: protocol seed={sa.get('seed')} "
              f"stride={sa.get('stride')} num_rollings={sa.get('num_rollings')} "
              f"horizon={cfg['horizon']} seq_len={hp['seq_len']} "
              f"batch_size={hp['batch_size']} num_epochs={hp['num_epochs']} "
              f"lr={hp.get('lr')} save_path={cfg['eff']['save_path']}")
        print(f"[preview] {cid} CMD: {' '.join(argv)}")
        failed |= not ok
    if failed:
        raise SystemExit(2)
    print(f"[preview] ALL {len(configs)} configs PASS "
          "(seed=2021, stride=1, num_rollings=48000, horizon=96)")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--config-ids", nargs="+", required=True)
    ap.add_argument("--save-path-root",
                    default="/dev_data/wlt/MUSE_logs/stage4_h96/repro")
    ap.add_argument("--out", default=None,
                    help="write extracted manifest here (extract mode)")
    ap.add_argument("--preview", action="store_true")
    ap.add_argument("--python", default="/dev_data/wlt/conda/envs/muse/bin/python",
                    help="python exe shown in preview argv")
    args = ap.parse_args()

    with open(args.manifest) as f:
        manifest = json.load(f)
    configs = extract(manifest, args.config_ids, args.save_path_root)
    if args.preview:
        preview(configs["configs"], args.python)
        return
    if not args.out:
        raise SystemExit("extract mode requires --out")
    with open(args.out, "w") as f:
        json.dump(configs, f, indent=2, ensure_ascii=False)
    print(f"[extract] wrote {args.out} ({configs['n_configs']} configs)")


if __name__ == "__main__":
    main()
