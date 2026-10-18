"""Build 6 audit packs (each <50MB, LIMIT hard-fail) with sha256
sidecars for EXP-029.

  code_logs (x1)      : package + muse module source + code_state.json
                        (md5 manifest, commit, MAE weights location/
                        bytes/sha), frozen protocol (51 files),
                        selected_p.json copy, all top-level json/md,
                        preflight/ + negative_control/ jsons,
                        runs/<gid>/*.json, README_rerun.txt
  pred_npz (x1)       : runs/ wholesale (36 test npz + 36 replay npz +
                        36 replay_val npz + per-group jsons) +
                        36 training-time val npz + replay_summary.json
  ckpts_1..4 (x4)     : 9 groups per pack — ckpt npz (118 arrays) +
                        meta json. ~39MB/pack.

MAE weights NEVER ship (path/bytes/sha in code_state.json).
--tags selects which packs to (re)build; omitted packs keep their
existing files (and sha256 sidecars) untouched.
"""
import argparse
import datetime
import shutil
import tarfile
from pathlib import Path

from repro_timemmd_common_v1 import protocol as P
from repro_timemmd_common_v1.common import dump_json, \
    setup_deterministic_env, sha256_of

LIMIT = 50 * 1024 * 1024
CKPT_GROUPS_PER_PACK = 9


def build_pack(out_dir, tag, stage_dir, extra_files):
    date = datetime.date.today().strftime("%Y%m%d")
    name = "MUSE_timemmd_common_s2021_%s_%s" % (tag, date)
    tpath = Path(out_dir) / (name + ".tar.gz")
    with tarfile.open(tpath, "w:gz") as tar:
        root = Path(stage_dir)
        for p in sorted(root.rglob("*")):
            if p.is_file():
                tar.add(p, arcname=str(Path(name) / p.relative_to(root)))
        for f in extra_files:
            f = Path(f)
            if not f.exists():
                raise SystemExit("[FATAL] pack source missing: %s" % f)
            if f.is_dir():
                for p in sorted(f.rglob("*")):
                    if p.is_file():
                        tar.add(p, arcname=str(Path(name) /
                                               f.name /
                                               p.relative_to(f)))
            else:
                tar.add(f, arcname=str(Path(name) / f.name))
    size = tpath.stat().st_size
    if size > LIMIT:
        tpath.unlink()
        raise SystemExit("[FATAL] pack %s is %.1fMB > 50MB — split it"
                         % (tpath.name, size / 2**20))
    sha = sha256_of(tpath)
    (Path(out_dir) / (tpath.name + ".sha256")).write_text(
        "%s  %s\n" % (sha, tpath.name))
    print("PACK %s  %.2fMB  sha256=%s" % (tpath.name, size / 2**20, sha))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--protocol_dir", default=None)
    ap.add_argument("--out_root", default=None)
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--tags", default="code_logs,pred_npz,ckpts_1,"
                    "ckpts_2,ckpts_3,ckpts_4")
    args = ap.parse_args()

    setup_deterministic_env()
    pinfo = P.verify_frozen(args.protocol_dir)
    out_root = Path(args.out_root or P.out_root())
    default_local = Path("/home/wlt/MMTS/temp")
    out_dir = Path(args.out_dir) if args.out_dir else (
        default_local if default_local.exists() else out_root / "packs")
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = {"repo_commit": P.repo_commit(),
                "code_md5": P.package_code_md5s(),
                "weights": {"path": str(P.mae_weights_path()),
                            "bytes": P.WEIGHTS_BYTES,
                            "sha256": pinfo["weights_sha256"]},
                "protocol_dir": str(pinfo["protocol_dir"]),
                "protocol_name": P.PROTOCOL_NAME,
                "exp027_selected_p": {
                    "path": str(P.exp027_out() / "selected_p.json"),
                    "sha256": P.EXP027_SELECTED_P_SHA256},
                "exp028_out": str(P.exp028_out()),
                "note": "MAE-Base weights stay on the compute server; "
                        "only MUSE adapter ckpts (~4.6MB x 36) ship."}
    dump_json(out_root / "code_state.json", manifest)

    stage = out_root / "packs_stage"
    if stage.exists():
        shutil.rmtree(stage)

    tags = set(args.tags.split(","))
    groups = P.all_groups()

    if "code_logs" in tags:
        code_stage = stage / "code"
        code_stage.mkdir(parents=True, exist_ok=True)
        for rel in manifest["code_md5"]:
            src = P.REPO_ROOT / rel
            dst = code_stage / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_bytes(src.read_bytes())
        (code_stage / "code_state.json").write_text(
            (out_root / "code_state.json").read_text())
        logs_stage = stage / "logs"
        (logs_stage / "runs").mkdir(parents=True)
        for pat in ("*.json", "*.md", "*.txt", "*.csv", "*.log"):
            for jf in sorted(out_root.glob(pat)):
                (logs_stage / jf.name).write_text(jf.read_text())
        for sub in ("preflight", "negative_control"):
            sdir = out_root / sub
            if sdir.is_dir():
                dst = logs_stage / sub
                dst.mkdir(parents=True, exist_ok=True)
                for jf in sorted(sdir.rglob("*.json")):
                    rel = jf.relative_to(sdir)
                    dst_j = dst / rel
                    dst_j.parent.mkdir(parents=True, exist_ok=True)
                    dst_j.write_text(jf.read_text())
        for gdir in sorted((out_root / "runs").iterdir()):
            if gdir.is_dir():
                dst = logs_stage / "runs" / gdir.name
                dst.mkdir(parents=True, exist_ok=True)
                for jf in sorted(gdir.glob("*.json")):
                    (dst / jf.name).write_text(jf.read_text())
        build_pack(out_dir, "code_logs", code_stage, [
            pinfo["protocol_dir"],
            out_root / "selected_p.json",
            logs_stage,
        ])

    if "pred_npz" in tags:
        val_stage = stage / "val_npz"
        val_stage.mkdir(parents=True, exist_ok=True)
        for dom, h in groups:
            f = out_root / P.val_npz_name(dom, h)
            if not f.exists():
                raise SystemExit("[FATAL] val npz missing: %s" % f)
            shutil.copyfile(str(f), str(val_stage / f.name))
        build_pack(out_dir, "pred_npz", out_root / "runs", [
            out_root / "replay_summary.json",
            val_stage,
        ])

    if any(t.startswith("ckpts_") for t in tags):
        n_packs = (len(groups) + CKPT_GROUPS_PER_PACK - 1) \
            // CKPT_GROUPS_PER_PACK
        for ci in range(n_packs):
            tag = "ckpts_%d" % (ci + 1)
            if tag not in tags:
                continue
            ck_stage = stage / tag
            ck_stage.mkdir(parents=True, exist_ok=True)
            chunk = groups[ci * CKPT_GROUPS_PER_PACK:
                           (ci + 1) * CKPT_GROUPS_PER_PACK]
            for dom, h in chunk:
                for f in (out_root / P.ckpt_npz_name(dom, h),
                          out_root / P.ckpt_meta_name(dom, h)):
                    if not f.exists():
                        raise SystemExit("[FATAL] ckpt missing: %s" % f)
                    shutil.copyfile(str(f), str(ck_stage / f.name))
            build_pack(out_dir, tag, ck_stage, [])

    shutil.rmtree(stage)
    print("AUDIT PACKS DONE (%s) -> %s"
          % (",".join(sorted(tags)), out_dir))


if __name__ == "__main__":
    main()
