"""Serial run orchestrator with a persistent queue (phase-3 Step 2).

State machine per entry: pending -> running -> postverify -> done (formal)
/ done_preflight (preflight); any failure -> failed (isolated, queue
continues). queue.json is updated atomically after every transition.

Restart semantics (never fake-resume):
  - done / done_preflight are skipped;
  - a stale `running` entry (pid dead) is marked failed(interrupted) and a
    NEW directory `<run_id>_r<n>` is started from the pretrained weights +
    fixed seed;
  - a stale `postverify` entry re-enters postverify ONLY if the training
    artifacts prove a natural completion (summary.json + manifest.json);
    otherwise it is treated like an interrupted run.

Every entry launches the OFFICIAL command rebuilt from configs_manifest.json
with the only permitted machine adjustments (MAE checkpoint path, per-run
save-path), inside a fresh out_dir carrying run_config.json. Training and
post-verification are strictly serial; nvidia-smi must show no compute
processes before each launch.

Dry-run (--dry-run) prints planned transitions and commands without
executing; --recover-only performs stale-recovery bookkeeping then exits
(G2.5 drill).
"""

import argparse
import datetime as _dt
import hashlib
import json
import os
import shlex
import signal
import subprocess
import sys
import tempfile

from .run_config import write_run_config, validate as validate_run_config

STATUSES = ("pending", "running", "postverify", "done", "done_preflight",
            "failed", "skipped")


def now():
    return _dt.datetime.now().isoformat(timespec="seconds")


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_queue(path):
    if not os.path.exists(path):
        return {"entries": []}
    with open(path) as f:
        return json.load(f)


def save_queue(path, queue):
    d = os.path.dirname(path) or "."
    fd, tmp = tempfile.mkstemp(dir=d, prefix=".queue.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(queue, f, indent=2, sort_keys=True)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def pid_alive(pid):
    if not pid:
        return False
    try:
        os.kill(int(pid), 0)
        return True
    except (ProcessLookupError, ValueError):
        return False


def gpu_busy():
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None  # cannot assert -> caller decides
    pids = [ln.strip() for ln in out.stdout.splitlines() if ln.strip()]
    return pids or None


def git_head(repo):
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo,
                              capture_output=True, text=True,
                              timeout=30).stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        return None


def build_argv(entry_cfg, python_exe):
    argv = shlex.split(entry_cfg["raw"]["command"])
    eff_hp = entry_cfg["eff"]["model_hyper_params"]
    out = []
    flag = None
    for tok in argv:
        if flag == "model-hyper-params":
            out.append(json.dumps(eff_hp))
            flag = None
        elif flag == "save-path":
            out.append(entry_cfg["eff"]["save_path"])
            flag = None
        elif tok.startswith("--"):
            flag = tok[2:].strip()
            out.append(tok)
        else:
            out.append(tok)
    out[0] = python_exe
    return out


def env_overrides_for(kind, args):
    ov = {
        "MUSE_REPRO_ENABLE": "1",
        "MUSE_REPRO_REQUIRE_RUN_CONFIG": "1",
    }
    if kind == "formal":
        ov["MUSE_REPRO_AUDIT_STEPS"] = str(args.audit_steps)
    elif kind == "preflight":
        ov["MUSE_REPRO_MAX_STEPS"] = str(args.preflight_steps)
        ov["MUSE_REPRO_VAL_AT"] = str(args.preflight_steps)
        ov["MUSE_REPRO_VAL_MAX_BATCHES"] = str(args.preflight_val_batches)
    elif kind == "controlled_stop":
        ov["MUSE_REPRO_MAX_STEPS"] = str(args.stop_steps)
    for kv in args.env or []:
        k, _, v = kv.partition("=")
        ov[k] = v
    return ov


def make_entry(cfg, kind, args, logs_root):
    suffix = {"preflight": "_probe", "controlled_stop": "_g3"}.get(kind, "")
    run_id = cfg["config_id"] + "_seed2021" + suffix
    return {
        "run_id": run_id,
        "config_id": cfg["config_id"],
        "kind": kind,
        "attempt": 1,
        "out_dir": os.path.join(logs_root, "repro", run_id),
        "status": "pending",
        "queued_at": now(),
        "started_at": None,
        "finished_at": None,
        "pid": None,
        "git_head": None,
        "host": os.uname().nodename,
        "note": None,
    }


def check_natural_completion(out_dir):
    """Stale postverify may re-enter only with proof of finished training."""
    summary = os.path.join(out_dir, "summary.json")
    manifest = os.path.join(out_dir, "manifest.json")
    if not (os.path.exists(summary) and os.path.exists(manifest)):
        return False, "missing summary.json/manifest.json"
    try:
        with open(summary) as f:
            s = json.load(f)
        reason = str(s.get("stop_reason", s.get("reason", "")))
    except (OSError, ValueError):
        return False, "summary.json unreadable"
    if "max_steps" in reason or "interrupt" in reason:
        return False, f"summary stop_reason={reason!r}"
    return True, f"stop_reason={reason!r}"


def check_controlled_stop(out_dir):
    """G3 criteria: summary says max_steps, no test entry, no test_preds."""
    summary_path = os.path.join(out_dir, "summary.json")
    if not os.path.exists(summary_path):
        return False, "summary.json missing"
    with open(summary_path) as f:
        s = json.load(f)
    if s.get("stop_reason") != "max_steps":
        return False, f"stop_reason={s.get('stop_reason')!r} != max_steps"
    if os.path.exists(os.path.join(out_dir, "test_report.json")):
        return False, "test_report.json exists (reached test phase)"
    for name in ("test_preds.npz", "test_preds_shards.json"):
        if os.path.exists(os.path.join(out_dir, name)):
            return False, f"{name} exists"
    return True, f"max_steps stop at step {s.get('global_step')}"


def recover_stale(queue, dry):
    """Mark dead running entries failed(interrupted) or re-queue postverify."""
    changed = []
    for e in queue["entries"]:
        if e["status"] == "running" and not pid_alive(e.get("pid")):
            e["status"] = "failed"
            e["note"] = "interrupted (stale running, pid dead)"
            e["finished_at"] = now()
            changed.append((e["run_id"], "running->failed(interrupted)"))
            nxt = dict(e)
            nxt["attempt"] = e["attempt"] + 1
            nxt["out_dir"] = f"{e['out_dir'].rsplit('_r', 1)[0]}_r{nxt['attempt']}" \
                if f"_r{e['attempt']}" in e["out_dir"] else f"{e['out_dir']}_r{nxt['attempt']}"
            nxt.update({"status": "pending", "pid": None, "started_at": None,
                        "finished_at": None, "note": "re-attempt in NEW dir "
                        "from pretrained weights + seed", "queued_at": now()})
            queue["entries"].append(nxt)
            changed.append((nxt["run_id"], f"re-queued attempt {nxt['attempt']}"))
        elif e["status"] == "postverify" and not pid_alive(e.get("pid")):
            if e.get("postverify_recovered"):
                continue
            ok, detail = check_natural_completion(e["out_dir"])
            if ok:
                e["pid"] = None
                e["postverify_recovered"] = True
                e["note"] = f"stale postverify -> re-enter postverify ({detail})"
                changed.append((e["run_id"], "postverify re-entered"))
            else:
                e["status"] = "failed"
                e["note"] = f"interrupted during postverify ({detail})"
                e["finished_at"] = now()
                changed.append((e["run_id"], "postverify->failed(interrupted)"))
                nxt = dict(e)
                nxt["attempt"] = e["attempt"] + 1
                nxt["out_dir"] = f"{e['out_dir']}_r{nxt['attempt']}"
                nxt.update({"status": "pending", "pid": None,
                            "started_at": None, "finished_at": None,
                            "note": "re-attempt in NEW dir",
                            "queued_at": now()})
                queue["entries"].append(nxt)
                changed.append((nxt["run_id"], f"re-queued attempt {nxt['attempt']}"))
    return changed


def write_failure(out_dir, entry, stage, detail, log_path=None):
    rec = {
        "run_id": entry["run_id"], "config_id": entry["config_id"],
        "stage": stage, "detail": detail, "time": now(),
        "out_dir": out_dir, "git_head": entry.get("git_head"),
    }
    if log_path and os.path.exists(log_path):
        with open(log_path, errors="replace") as f:
            rec["log_tail"] = f.readlines()[-100:]
    with open(os.path.join(out_dir, "failure.json"), "w") as f:
        json.dump(rec, f, indent=2, sort_keys=True)


def run_entry(entry, cfg, args, queue_path, queue):
    out_dir = entry["out_dir"]
    os.makedirs(out_dir, exist_ok=True)
    entry["status"] = "running"
    entry["started_at"] = now()
    entry["git_head"] = git_head(args.repo)
    save_queue(queue_path, queue)

    env = dict(os.environ)
    env.update(env_overrides_for(entry["kind"], args))
    env["MUSE_REPRO_OUT_DIR"] = out_dir

    patch = os.path.join(args.repo, "muse_patch.diff")
    mae = args.mae_path
    hp = cfg["eff"]["model_hyper_params"]
    hp.setdefault("checkpoint_path", mae)
    loader = {
        "seq_len": int(hp["seq_len"]),
        "batch_size": int(hp["batch_size"]),
        "num_workers": 0,
        "norm": True,
    }
    sa = cfg.get("strategy_args_resolved", {})
    run_cfg = {
        "run_id": entry["run_id"],
        "kind": entry["kind"],
        "dataset_file": cfg["dataset_file"],
        "protocol": {
            "horizon": cfg["horizon"],
            "seed": sa.get("seed", 2021),
            "tv_ratio": cfg.get("tv_ratio"),
            "train_ratio_in_tv": cfg["train_ratio_in_tv_resolved"],
            "stride": sa.get("stride"),
            "num_rollings": sa.get("num_rollings"),
            "deterministic": cfg["raw"].get("deterministic"),
        },
        "hyper_params_raw": cfg["raw"]["model_hyper_params"],
        "hyper_params_eff": hp,
        "adjustments": cfg["adjustments"],
        "loader": loader,
        "script_ref": cfg["script_ref"],
        "raw_command": cfg["raw"]["command"],
        "command_argv": build_argv(cfg, args.python),
        "env": {
            "git_head": entry["git_head"],
            "patch_sha": sha256_file(patch) if os.path.exists(patch) else None,
            "mae_sha": sha256_file(mae) if os.path.exists(mae) else None,
            "muse_repro": {k: env[k] for k in sorted(env) if k.startswith("MUSE_REPRO_")},
        },
        "attempt": entry["attempt"],
        "status": "queued",
        "queued_at": now(),
    }
    validate_run_config(run_cfg)
    write_run_config(out_dir, run_cfg)

    log_path = os.path.join(out_dir, "train_stdout.log")
    argv = run_cfg["command_argv"]
    print(f"[orch] START {entry['run_id']} (attempt {entry['attempt']}) -> {out_dir}")
    print(f"[orch] CMD {' '.join(shlex.quote(a) for a in argv)}")
    if args.dry_run:
        entry["status"] = "pending"
        entry["started_at"] = None
        print(f"[orch] DRY-RUN would execute above; not running")
        return None
    with open(log_path, "w") as log:
        proc = subprocess.Popen(argv, cwd=args.repo, env=env, stdout=log,
                                stderr=subprocess.STDOUT, start_new_session=True)
        entry["pid"] = proc.pid
        save_queue(queue_path, queue)
        try:
            rc = proc.wait()
        except KeyboardInterrupt:
            os.killpg(proc.pid, signal.SIGINT)
            rc = proc.wait()
            entry["status"] = "failed"
            entry["note"] = "interrupted by signal (child terminated)"
            entry["finished_at"] = now()
            write_failure(out_dir, entry, "launch", "keyboard-interrupt; child terminated", log_path)
            return False
    if rc != 0:
        if entry["kind"] == "controlled_stop":
            # SystemExit from the controlled stop is the EXPECTED outcome;
            # postverify checks the real criteria (stop_reason, no test).
            print(f"[orch] controlled stop exit {rc} (expected)")
            return True
        entry["status"] = "failed"
        entry["finished_at"] = now()
        entry["note"] = f"training exit code {rc}"
        write_failure(out_dir, entry, "train", f"exit code {rc}", log_path)
        print(f"[orch] FAIL {entry['run_id']}: training exit {rc}")
        return False
    return True


def postverify_entry(entry, args, queue_path, queue):
    entry["status"] = "postverify"
    save_queue(queue_path, queue)
    out_dir = entry["out_dir"]

    if entry["kind"] == "controlled_stop":
        ok, detail = check_controlled_stop(out_dir)
        entry["status"] = "done_preflight" if ok else "failed"
        entry["finished_at"] = now()
        entry["note"] = detail
        print(f"[orch] {'DONE' if ok else 'FAIL'} {entry['run_id']}: {detail}")
        if not ok:
            write_failure(out_dir, entry, "controlled_stop", detail)
        return ok

    steps = [
        ["verify_restore", [args.python, "-m", "repro_tools.verify_restore",
                            "--out-dir", out_dir]
         + (["--best"] if entry["kind"] == "preflight" else [])],
        ["recompute", [args.python, "-m", "repro_tools.recompute_test_metrics",
                       "--out-dir", out_dir]],
    ]
    for name, argv in steps:
        print(f"[orch] POSTVERIFY {entry['run_id']}: {name}")
        if args.dry_run:
            print(f"[orch] DRY-RUN would run: {' '.join(argv)}")
            continue
        log_path = os.path.join(out_dir, f"postverify_{name}.log")
        with open(log_path, "w") as log:
            rc = subprocess.call(argv, cwd=args.repo, stdout=log,
                                 stderr=subprocess.STDOUT)
        if rc != 0:
            entry["status"] = "failed"
            entry["finished_at"] = now()
            entry["note"] = f"postverify/{name} exit {rc}"
            write_failure(out_dir, entry, f"postverify/{name}",
                          f"exit code {rc}", log_path)
            print(f"[orch] FAIL {entry['run_id']}: postverify/{name} exit {rc}")
            return False
    if args.dry_run:
        return None
    entry["status"] = "done_preflight" if entry["kind"] == "preflight" else "done"
    entry["finished_at"] = now()
    print(f"[orch] DONE {entry['run_id']} -> {entry['status']}")
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="configs_manifest.json")
    ap.add_argument("--repo", default="/dev_data/wlt/MUSE")
    ap.add_argument("--python", default="/dev_data/wlt/conda/envs/muse/bin/python")
    ap.add_argument("--mae-path",
                    default="/dev_data/wlt/dmmv_rebuild/mae/mae_visualize_vit_base.pth")
    ap.add_argument("--logs-root", default="/dev_data/wlt/MUSE_logs")
    ap.add_argument("--queue-dir", default=None,
                    help="default <logs-root>/orchestration")
    ap.add_argument("--filter", default=None, help="substring on config_id")
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--kind", choices=("formal", "preflight", "controlled_stop"),
                    default="formal")
    ap.add_argument("--audit-steps", default=1)
    ap.add_argument("--stop-steps", default=2,
                    help="MAX_STEPS for --kind controlled_stop")
    ap.add_argument("--preflight-steps", default=50)
    ap.add_argument("--preflight-val-batches", default=20)
    ap.add_argument("--include-etth1-h96", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--recover-only", action="store_true",
                    help="perform stale recovery bookkeeping then exit (G2.5)")
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--allow-shared-gpu", action="store_true",
                    help="warn instead of fail when compute processes exist")
    ap.add_argument("--env", nargs="*", default=None,
                    help="extra KEY=VALUE env overrides for the child")
    args = ap.parse_args()
    queue_dir = args.queue_dir or os.path.join(args.logs_root, "orchestration")
    os.makedirs(queue_dir, exist_ok=True)
    queue_path = os.path.join(queue_dir, "queue.json")

    with open(args.manifest) as f:
        manifest = json.load(f)
    if not manifest.get("check_passed"):
        raise SystemExit("configs_manifest.json has check_problems; refusing")

    queue = load_queue(queue_path)
    selected = []
    if not (args.status or args.recover_only):
        known = {(e["run_id"], e["attempt"]) for e in queue["entries"]}
        for cfg in manifest["configs"]:
            cid = cfg["config_id"]
            if cid == "ETTh1_h96" and not args.include_etth1_h96:
                continue
            if args.filter and args.filter not in cid:
                continue
            if args.only and cid not in args.only:
                continue
            e = make_entry(cfg, args.kind, args, args.logs_root)
            if (e["run_id"], e["attempt"]) not in known:
                queue["entries"].append(e)
            selected.append(e["run_id"])
        save_queue(queue_path, queue)

    changed = recover_stale(queue, args.dry_run)
    if changed:
        save_queue(queue_path, queue)
    for rid, what in changed:
        print(f"[orch] RECOVER {rid}: {what}")

    if args.status:
        for e in queue["entries"]:
            print(f"{e['status']:>15}  {e['run_id']}  attempt={e['attempt']}  {e['out_dir']}")
        return
    if args.recover_only or args.dry_run:
        print(f"[orch] queue at {queue_path}; {len(queue['entries'])} entries "
              f"(dry/recover-only, no execution)")
        return

    if not args.allow_shared_gpu:
        busy = gpu_busy()
        if busy:
            raise SystemExit(f"nvidia-smi shows compute processes {busy}; "
                             f"refusing to start (--allow-shared-gpu to override)")

    n_done = n_fail = 0
    for e in list(queue["entries"]):
        if e["run_id"] not in selected or \
                e["status"] in ("done", "done_preflight", "failed", "skipped"):
            continue
        if e["status"] == "running" and pid_alive(e.get("pid")):
            raise SystemExit(f"another orchestrator owns {e['run_id']} (pid {e['pid']})")
        cfg = next(c for c in manifest["configs"] if c["config_id"] == e["config_id"])
        ok = run_entry(e, cfg, args, queue_path, queue)
        if ok is False:
            n_fail += 1
            continue
        if ok is None:  # dry-run
            continue
        pv = postverify_entry(e, args, queue_path, queue)
        if pv is False:
            n_fail += 1
        elif pv is True:
            n_done += 1
        save_queue(queue_path, queue)
    print(f"[orch] session summary: done={n_done} failed={n_fail}")
    if n_fail:
        sys.exit(1)


if __name__ == "__main__":
    main()
