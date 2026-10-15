"""Per-run configuration manifest: written by the orchestrator BEFORE launch.

Every new run directory must contain run_config.json; the hook asserts its
presence when MUSE_REPRO_REQUIRE_RUN_CONFIG=1, and verify_restore rebuilds
the pipeline from it (no hardcoded dataset constants). The only consumer
allowed to run without it is the frozen ETTh1/H96 directory via --legacy.
"""

import json
import os

KINDS = ("formal", "preflight", "controlled_stop", "chunk_probe")


def write_run_config(out_dir, cfg):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "run_config.json")
    with open(path, "w") as f:
        json.dump(cfg, f, indent=2, sort_keys=True, default=str)
    return path


def load_run_config(out_dir):
    path = os.path.join(out_dir, "run_config.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def require_run_config(out_dir):
    cfg = load_run_config(out_dir)
    if cfg is None:
        raise SystemExit(
            f"run_config.json missing in {out_dir}; refusing to guess the "
            f"configuration (--legacy is reserved for the frozen ETTh1/H96 dir)"
        )
    if cfg.get("kind") not in KINDS:
        raise SystemExit(f"run_config.json has invalid kind: {cfg.get('kind')!r}")
    return cfg


def validate(cfg):
    for key in ("run_id", "kind", "dataset_file", "protocol",
                "hyper_params_raw", "hyper_params_eff", "loader"):
        if key not in cfg:
            raise SystemExit(f"run_config.json missing field: {key}")
    for key in ("horizon", "seed", "tv_ratio", "train_ratio_in_tv",
                "stride", "num_rollings", "deterministic"):
        if key not in cfg["protocol"]:
            raise SystemExit(f"run_config.json protocol missing: {key}")
    for key in ("seq_len", "batch_size", "num_workers", "norm"):
        if key not in cfg["loader"]:
            raise SystemExit(f"run_config.json loader missing: {key}")
    return cfg
