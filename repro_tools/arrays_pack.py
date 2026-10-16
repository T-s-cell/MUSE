"""Package reviewer-requested prediction arrays into <50MB tar.gz parts.

Per formal run (dirs without _probe/_g3/full_e10 and holding test preds):
  - all test_preds*.npz + test_preds_shards.json (when sharded);
  - the FINAL best validation dump best_preds_epoch<N>* where N is the
    epoch with the lowest val_mse_fused among best_point_epoch*.json
    (intermediate best points are intentionally NOT packaged);
  - test_report.json + recompute_test_metrics.json for cross-checking.
window_border inside each npz is the window index array; the shards
json asserts full coverage of the official window list.

Greedy-packs files into tar.gz parts (raw-size budget, npz barely
compresses), then verifies every part is < 50MB on disk and writes
MANIFEST.json (per-part + per-file SHA256) and README.md.

Usage (from the repo root, on the training machine):
    python -m repro_tools.arrays_pack \
        --repro-root /dev_data/wlt/MUSE_logs/repro \
        --out-dir /dev_data/wlt/MUSE_logs/packages/arrays_v2
"""

import argparse
import glob
import hashlib
import json
import os
import tarfile

MAX_PART_DISK_BYTES = 50 * 1000 * 1000


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_json(path):
    with open(path) as f:
        return json.load(f)


def select_best_epoch(run_dir):
    points = []
    for j in sorted(glob.glob(os.path.join(run_dir, "best_point_epoch*.json"))):
        r = load_json(j)
        v = r.get("val_mse_fused")
        if v is not None:
            points.append((v, j))
    if not points:
        return None, None
    v, j = min(points)
    epoch = os.path.basename(j).replace(
        "best_point_epoch", "").replace(".json", "")
    return epoch, v


def collect_runs(repro_root):
    runs = {}
    for name in sorted(os.listdir(repro_root)):
        run_dir = os.path.join(repro_root, name)
        if not os.path.isdir(run_dir):
            continue
        if ("_probe" in name or "_g3" in name or name == "full_e10"):
            continue
        test_files = (sorted(glob.glob(
            os.path.join(run_dir, "test_preds*.npz")))
            + sorted(glob.glob(
                os.path.join(run_dir, "test_preds_shards.json"))))
        if not test_files:
            continue
        runs[name] = run_dir
    return runs


def build_file_list(repro_root, runs):
    items = []  # (arcname, src_path, raw_bytes)
    info = {}
    for name, run_dir in runs.items():
        epoch, val = select_best_epoch(run_dir)
        assert epoch is not None, f"{name}: no best point found"
        files = sorted(glob.glob(os.path.join(run_dir, "test_preds*.npz")))
        files += sorted(glob.glob(
            os.path.join(run_dir, "test_preds_shards.json")))
        files += sorted(glob.glob(os.path.join(
            run_dir, "best_preds_epoch%s*" % epoch)))
        for extra in ("test_report.json", "recompute_test_metrics.json"):
            p = os.path.join(run_dir, extra)
            if os.path.exists(p):
                files.append(p)
        n = 0
        for f in sorted(set(files)):
            arc = os.path.join(name, os.path.basename(f))
            items.append((arc, f, os.path.getsize(f)))
            n += 1
        info[name] = {
            "best_epoch": epoch, "best_val_mse_fused": val,
            "n_files": n,
            "best_files": [os.path.basename(f) for f in files
                           if "best_preds_epoch" in f],
        }
    return items, info


def pack(items, out_dir, max_bytes):
    parts = []
    cur, cur_bytes, idx = [], 0, 1

    def flush():
        nonlocal cur, cur_bytes, idx
        if not cur:
            return
        name = f"arrays_v2_part{idx:02d}.tar.gz"
        path = os.path.join(out_dir, name)
        with tarfile.open(path, "w:gz") as tf:
            for arc, src in cur:
                tf.add(src, arcname=arc, recursive=False)
        disk = os.path.getsize(path)
        assert disk < MAX_PART_DISK_BYTES, f"{name}={disk} >= 50MB"
        parts.append({"name": name, "path": path,
                      "bytes": disk,
                      "files": [{"arcname": a, "path": s}
                                for a, s in cur]})
        cur, cur_bytes = [], 0
        idx += 1

    for arc, src, size in items:
        if cur and cur_bytes + size > max_bytes:
            flush()
        cur.append((arc, src))
        cur_bytes += size
    flush()
    return parts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repro-root",
                    default="/dev_data/wlt/MUSE_logs/repro")
    ap.add_argument("--out-dir",
                    default="/dev_data/wlt/MUSE_logs/packages/arrays_v2")
    ap.add_argument("--max-bytes", type=int, default=44_000_000)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    runs = collect_runs(args.repro_root)
    assert runs, "no formal runs with test preds found"
    items, info = build_file_list(args.repro_root, runs)
    parts = pack(items, args.out_dir, args.max_bytes)

    manifest = {
        "runs": info,
        "parts": [],
        "total_raw_bytes": sum(s for _, _, s in items),
        "selection": {
            "test": "all test_preds*.npz (+ test_preds_shards.json)",
            "best_val": "best_preds_epoch<N>* for N = argmin "
                        "val_mse_fused over best_point_epoch*.json "
                        "(final best only; intermediate best points "
                        "not packaged)",
            "window_index": "window_border array inside every npz; "
                            "shards json asserts full coverage",
        },
    }
    for p in parts:
        entry = {"name": p["name"], "bytes": p["bytes"],
                 "sha256": sha256_file(p["path"]),
                 "files": []}
        for f in p["files"]:
            entry["files"].append({
                "arcname": f["arcname"],
                "bytes": os.path.getsize(f["path"]),
                "sha256": sha256_file(f["path"]),
            })
        manifest["parts"].append(entry)

    with open(os.path.join(args.out_dir, "MANIFEST.json"), "w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
    readme = (
        "# MUSE arrays v2 (review fix R3)\n\n"
        "Contents per formal run: test predictions, targets and window\n"
        "index (window_border inside each npz; shards json asserts full\n"
        "coverage), the FINAL best validation prediction dump, and the\n"
        "small test_report.json / recompute_test_metrics.json for\n"
        "cross-checking. Intermediate best points are NOT included\n"
        "(per review). Weights stay on the server.\n\n"
        f"- runs: {len(runs)}\n"
        f"- parts: {len(parts)} (each < 50MB on disk)\n"
        f"- raw bytes: {manifest['total_raw_bytes']}\n\n"
        "Verify: MANIFEST.json lists SHA256 for every part and every\n"
        "file inside it.\n")
    with open(os.path.join(args.out_dir, "README.md"), "w") as f:
        f.write(readme)
    print(f"[arrays_pack] {len(runs)} runs, {len(parts)} parts, "
          f"{manifest['total_raw_bytes']/1e6:.0f}MB raw -> {args.out_dir}")
    for p in manifest["parts"]:
        print(f"  {p['name']} {p['bytes']/1e6:.1f}MB {p['sha256'][:16]}...")


if __name__ == "__main__":
    main()
