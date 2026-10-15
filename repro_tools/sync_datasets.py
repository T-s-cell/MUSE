"""Sync the 9 missing forecasting CSVs from the Aurora source to the MUSE
dataset directory (copy, never symlink), with full provenance.

For each dataset: locate exactly one source file under --source-root, verify
sha256 after copy matches the source, and confirm the FORECAST_META.csv row
(length) against the review-pinned table. Existing identical files are
skipped; existing DIFFERENT content is a hard error (never silently
overwrite). Emits dataset_sync.json next to the dataset META file.

Stdlib only; run on the training machine:
    python -m repro_tools.sync_datasets \
        --source-root /dev_data/wlt/Aurora/benchmark_datasets \
        --dst-dir /dev_data/wlt/MUSE/dataset/forecasting
"""

import argparse
import csv
import hashlib
import json
import os
import shutil
import sys

# Review-pinned META lengths (from the phase-3 recon on iota).
EXPECTED_META_LEN = {
    "ETTh1.csv": 14400,
    "ETTh2.csv": 14400,
    "ETTm1.csv": 57600,
    "ETTm2.csv": 57600,
    "Weather.csv": 52696,
    "Electricity.csv": 26304,
    "AQWan.csv": 35064,
    "Wind.csv": 48673,
    "ZafNoo.csv": 19225,
    "CzeLan.csv": 19934,
}


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def find_source(source_root, name):
    hits = []
    for dirpath, _dirnames, filenames in os.walk(source_root):
        if name in filenames:
            hits.append(os.path.join(dirpath, name))
    return hits


def read_meta(dst_dir):
    meta_path = os.path.join(dst_dir, "FORECAST_META.csv")
    if not os.path.exists(meta_path):
        return None, meta_path
    with open(meta_path, newline="") as f:
        rows = list(csv.DictReader(f))
    return {r["file_name"]: r for r in rows}, meta_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-root", default="/dev_data/wlt/Aurora/benchmark_datasets")
    ap.add_argument("--dst-dir",
                    default="/dev_data/wlt/MUSE/dataset/forecasting")
    ap.add_argument("--datasets", nargs="*", default=sorted(EXPECTED_META_LEN))
    args = ap.parse_args()

    meta, meta_path = read_meta(args.dst_dir)
    report = {"source_root": args.source_root, "dst_dir": args.dst_dir,
              "meta_path": meta_path, "files": {}, "problems": []}
    hard_fail = False

    for name in args.datasets:
        entry = {"expected_meta_len": EXPECTED_META_LEN.get(name)}
        report["files"][name] = entry

        hits = find_source(args.source_root, name)
        if len(hits) != 1:
            entry["error"] = f"source matches: {hits}"
            report["problems"].append(f"{name}: source lookup failed")
            hard_fail = True
            continue
        entry["source"] = hits[0]
        entry["source_sha256"] = sha256_file(hits[0])

        dst = os.path.join(args.dst_dir, name)
        entry["dst"] = dst
        if os.path.exists(dst):
            dst_sha = sha256_file(dst)
            entry["dst_sha256"] = dst_sha
            if dst_sha == entry["source_sha256"]:
                entry["action"] = "already-present"
            else:
                entry["error"] = "dst exists with DIFFERENT content; refusing to overwrite"
                report["problems"].append(f"{name}: content conflict")
                hard_fail = True
                continue
        else:
            shutil.copyfile(hits[0], dst)
            dst_sha = sha256_file(dst)
            entry["dst_sha256"] = dst_sha
            if dst_sha != entry["source_sha256"]:
                entry["error"] = "post-copy sha mismatch"
                report["problems"].append(f"{name}: copy verification failed")
                hard_fail = True
                continue
            entry["action"] = "copied"

        if meta is None:
            entry["meta_error"] = "FORECAST_META.csv missing"
            report["problems"].append(f"{name}: META missing")
            hard_fail = True
        elif name not in meta:
            entry["meta_error"] = "no META row"
            report["problems"].append(f"{name}: no META row")
            hard_fail = True
        else:
            row = meta[name]
            entry["meta_row"] = {k: str(v) for k, v in row.items()}
            try:
                got_len = int(float(row["length"]))
            except (KeyError, ValueError):
                got_len = None
            entry["meta_len"] = got_len
            if entry["expected_meta_len"] is not None \
                    and got_len != entry["expected_meta_len"]:
                entry["meta_error"] = (
                    f"META length {got_len} != expected "
                    f"{entry['expected_meta_len']}"
                )
                report["problems"].append(f"{name}: META length mismatch")
                hard_fail = True

    report["ok"] = not hard_fail
    out_path = os.path.join(args.dst_dir, "dataset_sync.json")
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2, sort_keys=True)
    print(json.dumps({k: report[k] for k in ("ok", "problems", "files")},
                     indent=2, sort_keys=True))
    print(f"[sync] report -> {out_path}")
    if hard_fail:
        sys.exit(1)


if __name__ == "__main__":
    main()
