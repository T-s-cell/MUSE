"""Protocol data access: windows manifests + domain CSVs, stdlib csv only.

Rules (frozen protocol):
  - OT parsed with python float() (strtod) -> float64 standardization with
    frozen scaler.json mean/std -> float32 series -> slice
  - input rows [s-L, s), label rows [s, s+H)
  - windows manifests are the ONLY split source; manifest order preserved
  - gates: count == splits.json, segment boundaries, finiteness
Only OT enters the model; fact/prior/preds columns are never read.
"""
import csv
import json
from pathlib import Path

import numpy as np

from repro_timemmd_common_v1.common import all_finite

_SEGMENTS = {
    "train": ("n_train", None),
    "val": ("n_train", "val_end"),
    "test": ("val_end", "n_test_end"),
}


class DomainData:
    def __init__(self, pdir, splits, domain):
        self.pdir = Path(pdir)
        self.domain = domain
        spec = splits[domain]
        self.L = int(spec["L"])
        self.n_train = int(spec["n_train"])
        self.val_end = int(spec["val_end"])
        self.n_test = int(spec["n_test"])
        self.window_counts = spec["window_counts"]
        fmap = spec["field_map"]
        self.date_col = fmap["date"]
        self.target_col = fmap["target"]

        scaler = json.load(open(self.pdir / "scaler.json"))[domain]
        self.mean = float(scaler["mean"])
        self.std = float(scaler["std"])

        rows, dates = [], []
        with open(self.pdir / ("%s.csv" % domain), newline="") as f:
            for rec in csv.DictReader(f):
                rows.append(float(rec[self.target_col]))  # strtod parse
                dates.append(rec[self.date_col])
        ot64 = np.asarray(rows, dtype=np.float64)
        if not all_finite(ot64):
            raise ValueError("non-finite OT in %s" % domain)
        self.series32 = ((ot64 - self.mean) / self.std).astype(np.float32)
        self.dates = dates
        self.N = len(self.series32)
        if self.N != int(spec["N"]):
            raise ValueError("%s rows %d != splits N %d"
                             % (domain, self.N, spec["N"]))

    def segment_bounds(self, split):
        if split == "train":
            return 0, self.n_train
        if split == "val":
            return self.n_train, self.val_end
        if split == "test":
            return self.val_end, self.val_end + self.n_test
        raise ValueError(split)

    def load_split(self, h, split):
        """Returns dict(X, Y, s, orig_row, label_dates, window_index) with
        manifest order preserved and all gates enforced."""
        h = int(h)
        wpath = self.pdir / "windows" / ("%s_H%d.csv" % (self.domain, h))
        rows = []
        with open(wpath, newline="") as f:
            for rec in csv.DictReader(f):
                if rec["split"] == split:
                    rows.append(rec)
        want = int(self.window_counts[str(h)][split])
        if len(rows) != want:
            raise ValueError("%s H%d %s: %d windows != splits.json %d"
                             % (self.domain, h, split, len(rows), want))

        s_arr = np.asarray([int(r["s"]) for r in rows], dtype=np.int64)
        orig_arr = np.asarray([int(r["orig_row"]) for r in rows],
                              dtype=np.int64)
        lab = [json.loads(r["label_dates"]) for r in rows]
        if any(len(x) != h for x in lab):
            raise ValueError("label_dates length != H in %s H%d"
                             % (self.domain, h))
        lab_arr = np.asarray(lab, dtype="U10")

        seg_lo, seg_hi = self.segment_bounds(split)
        for i, s in enumerate(s_arr):
            if s - self.L < 0 or s < seg_lo or s + h > seg_hi:
                raise ValueError(
                    "%s H%d %s win %d: s=%d violates segment [%d,%d) "
                    "or input bounds" % (self.domain, h, split, i, int(s),
                                         seg_lo, seg_hi))

        idx_in = s_arr[:, None] - self.L + np.arange(self.L)[None, :]
        idx_lab = s_arr[:, None] + np.arange(h)[None, :]
        X = self.series32[idx_in]
        Y = self.series32[idx_lab]
        if not (all_finite(X) and all_finite(Y)):
            raise ValueError("non-finite windows %s H%d" % (self.domain, h))
        return {"X": X, "Y": Y[:, :, None].copy(),
                "s": s_arr, "orig_row": orig_arr, "label_dates": lab_arr,
                "window_index": np.arange(len(rows), dtype=np.int64),
                "input_date_range": [r["input_date_range"] for r in rows]}


def infer_batches(n, batch):
    """Sequential slicing batches (no DataLoader, order preserved)."""
    return [(i, min(i + batch, n)) for i in range(0, n, batch)]
