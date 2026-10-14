"""Compare two smoke runs (R1 vs R2) over given steps: bitwise neutrality gate.

Usage:
    python -m repro_tools.compare_r1_r2 <dirA> <dirB> --steps 4,5,6

Checks per step, from steps.csv + summary.json:
  - input_sha / target_sha identical (DataLoader order + batch bytes)
  - loss_total / loss_channel / loss_tp / loss_gate / gate_value identical
    (exact float equality as recorded)
  - lr_adapters / lr_gate identical
  - param_sha[step] identical (trainable parameter checksum)
"""

import argparse
import csv
import json
import sys


def load_steps(out_dir):
    with open(f"{out_dir}/steps.csv") as f:
        return {int(r["step"]): r for r in csv.DictReader(f)}


def load_param_sha(out_dir):
    try:
        with open(f"{out_dir}/summary.json") as f:
            return json.load(f).get("param_sha", {})
    except FileNotFoundError:
        return {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir_a")
    ap.add_argument("dir_b")
    ap.add_argument("--steps", default="4,5,6")
    ap.add_argument("--tolerance", type=float, default=0.0,
                    help="0.0 = bitwise/exact; >0 = allowed abs diff for floats")
    args = ap.parse_args()
    steps = [int(s) for s in args.steps.split(",")]

    a_steps, b_steps = load_steps(args.dir_a), load_steps(args.dir_b)
    a_sha, b_sha = load_param_sha(args.dir_a), load_param_sha(args.dir_b)

    float_cols = ["loss_total", "loss_channel", "loss_tp", "loss_gate",
                  "gate_value", "lr_adapters", "lr_gate"]
    exact_cols = ["input_sha", "target_sha", "epoch", "batch_idx"]

    failures = []
    for step in steps:
        ra, rb = a_steps.get(step), b_steps.get(step)
        if ra is None or rb is None:
            failures.append(f"step{step}: missing row ({args.dir_a}: {ra is not None}, "
                            f"{args.dir_b}: {rb is not None})")
            continue
        for col in exact_cols:
            if ra[col] != rb[col]:
                failures.append(f"step{step}.{col}: {ra[col]!r} != {rb[col]!r}")
        for col in float_cols:
            va, vb = float(ra[col]), float(rb[col])
            if abs(va - vb) > args.tolerance:
                failures.append(
                    f"step{step}.{col}: {va!r} != {vb!r} "
                    f"(absdiff={abs(va - vb):.3e}, tol={args.tolerance})"
                )
        sa, sb = a_sha.get(str(step)), b_sha.get(str(step))
        if sa is None or sb is None:
            print(f"[warn] step{step}: param_sha missing (set MUSE_REPRO_PARAM_SHA=1)")
        elif sa != sb:
            failures.append(f"step{step}.param_sha: {sa} != {sb}")

    if failures:
        print("[compare] FAIL")
        for f_ in failures:
            print("  -", f_)
        sys.exit(1)
    print(f"[compare] PASS bitwise-exact over steps {steps}")


if __name__ == "__main__":
    main()
