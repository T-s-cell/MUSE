"""Preflight gates B1-B5 for EXP-029 (MUSE supervised training).

Runs before ANY formal training; artifacts live in out_root/preflight/.
run_train.py live-checks this report's all_pass AND b5.go (the report
sha is NOT pinned in guards — EXP-027 lesson).

  B1 protocol/data : verify_frozen (pins, 51 files, MAE weights,
                     selected_p, totals, 1,767 steps/epoch) + per-group
                     train/val/test loads (count/segment/finiteness
                     gates inside load_split) + FOUR-WAY test identity/
                     truth bitwise (protocol recompute AND EXP-027 npz
                     AND TaTS npz AND EXP-028 npz).
  B2 entry equiva. : all 36 geometries x first 32 val windows: grad-path
                     branch forward (model's own fused output) vs eval
                     fused (inference_mode) vs validate-style fused
                     (no_grad), atol=1e-6/rtol=1e-5; two eval forwards
                     bitwise (zero-RNG proof); naive
                     g*chan+(1-g)*tp fusion diff recorded as an
                     associativity DIAGNOSTIC (model fuses in normalized
                     space — not bitwise comparable, not a gate); shapes
                     [B,L,1]->[B,H,1] f32; batch=1 + H=6 + 4-visible-
                     column probe on Traffic/H6 (the real geometry with
                     num_patch_input == 4) end-to-end.
  B3 real steps    : PRECHECK groups (+ Traffic/H6): >=3 consecutive
                     real zero_grad->forward->backward->step cycles on
                     shuffled real batches with the dual-group Adam;
                     grad None only for the 2 pinned dead tensors,
                     non-finite fatal; structural zero-grad pattern
                     recorded as DIAGNOSTIC ONLY; backbone/buffers
                     bitwise unchanged (freeze forensics); real ragged
                     tail batch (Security/H12 == 6 windows) as one real
                     step; batch=1 step on Traffic/H6. Health/H48 model
                     kept for B4.
  B4 ckpt restore  : save the B3-updated Health/H48 model -> fresh
                     subprocess (restore_probe.py) strict restore + MAE
                     backfill -> first 8 val windows pred_close(1e-6,
                     1e-5) + restored packed_sha == ckpt packed_sha in
                     BOTH processes (params bitwise == ckpt; no
                     digest_pre != digest_post requirement; cross-
                     process npz hashes NOT required equal).
  B5 cost timing   : CUDA-synced 50 real steps per PRECHECK geometry
                     (drop first 10) -> s/step; extrapolation to
                     1,767 steps/epoch x 10 epochs + inference costs
                     REPORTED ONLY (no budget stop condition). b5.go is
                     set iff all_pass.

Exit 0 iff all_pass; exit 2 otherwise (report written either way).
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

from repro_timemmd_common_v1 import protocol as P
from repro_timemmd_common_v1.adapter import build_muse, configure_muse, \
    forward_branches, forward_fused
from repro_timemmd_common_v1.common import backbone_digest, \
    bitwise_equal, buffer_digest, dump_json, identity_equal, load_json, \
    named_tensor_sha, pred_close, seed_everything, setup_deterministic_env
from repro_timemmd_common_v1.data_timemmd import DomainData
from repro_timemmd_common_v1.muse_train import build_optimizer, \
    gradient_gate, npz_key, packed_sha, save_muse_ckpt, train_config

DEVICE = "cuda:0"
ATOL, RTOL = 1e-6, 1e-5
PROBE_N = 32
B3_STEPS = 3
B5_STEPS = 50
B5_WARMUP = 10


def _tats_npz(gid):
    ref = P.tats_ref_dir()
    for cand in (ref / gid / "test_arrays.npz",
                 ref / ("%s_seed2021" % gid) / "test_arrays.npz"):
        if cand.exists():
            return cand
    return None


def _ref_test_npz(dom, h, exp_out):
    return exp_out / "runs" / P.group_id(dom, h) / P.test_npz_name(dom, h)


def _cmp_ref(errs, gid, tag, sp, ref_path):
    if not ref_path.exists():
        errs.append("%s: %s test npz missing" % (gid, tag))
        return
    a = dict(np.load(str(ref_path)))
    for k in ("s", "orig_row", "label_dates", "window_index"):
        if k not in a or not identity_equal(sp[k], a[k]):
            errs.append("%s: identity %s != %s" % (gid, k, tag))
    if "truth" not in a or not bitwise_equal(sp["Y"], a["truth"]):
        errs.append("%s: truth != %s" % (gid, tag))


def _cmp_tats(errs, gid, sp):
    tats = _tats_npz(gid)
    if tats is None:
        errs.append("%s: TaTS reference npz missing" % gid)
        return
    t = dict(np.load(str(tats)))
    for k in ("truth", "s", "orig_row", "label_dates"):
        if k not in t:
            errs.append("%s: TaTS npz missing %s" % (gid, k))
            continue
        a = sp["Y"] if k == "truth" else sp[k]
        b = t[k]
        if a.ndim == 3 and b.ndim == 2:
            a = a[:, :, 0]
        if b.ndim == 3 and a.ndim == 2:
            b = b[:, :, 0]
        if not bitwise_equal(np.asarray(a), np.asarray(b)):
            errs.append("%s: %s != TaTS reference" % (gid, k))


def b1_protocol_data(pinfo):
    errs, details = [], {}
    domcache = {}
    for dom, h in P.all_groups():
        gid = P.group_id(dom, h)
        if dom not in domcache:
            domcache[dom] = DomainData(pinfo["protocol_dir"],
                                       pinfo["splits"], dom)
        dd = domcache[dom]
        sp = dd.load_split(h, "test")
        spv = dd.load_split(h, "val")
        spr = dd.load_split(h, "train")
        d = {"train_windows": int(spr["X"].shape[0]),
             "val_windows": int(spv["X"].shape[0]),
             "test_windows": int(sp["X"].shape[0])}
        _cmp_ref(errs, gid, "EXP-027", sp,
                 _ref_test_npz(dom, h, P.exp027_out()))
        _cmp_ref(errs, gid, "EXP-028", sp,
                 _ref_test_npz(dom, h, P.exp028_out()))
        _cmp_tats(errs, gid, sp)
        details[gid] = d
    return errs, details


def _new_model(dom, h, load_ckpt=True):
    seed_everything(P.SEED)
    m = build_muse(load_ckpt=load_ckpt)
    configure_muse(m, dom, h)
    return m.to(DEVICE)


def _drop(m):
    import torch
    del m
    torch.cuda.empty_cache()


def b2_entry_equivalence(pinfo):
    import torch
    errs, rows = [], []
    domcache = {}
    for dom, h in P.all_groups():
        gid = P.group_id(dom, h)
        if dom not in domcache:
            domcache[dom] = DomainData(pinfo["protocol_dir"],
                                       pinfo["splits"], dom)
        spv = domcache[dom].load_split(h, "val")
        n = min(PROBE_N, spv["X"].shape[0])
        m = _new_model(dom, h)
        npi = int(m.num_patch_input)
        xb = torch.from_numpy(np.ascontiguousarray(spv["X"][:n])) \
            .to(DEVICE)
        assert xb.shape == (n, P.TABLE[dom]["L"], 1)
        m.train()
        res = forward_branches(m, xb)
        y_grad = res["output"].detach().cpu().numpy()
        assert y_grad.shape == (n, h, 1) and y_grad.dtype == np.float32
        g = torch.sigmoid(m.fusion_logit).detach()
        naive = (g * res["channel"].detach()
                 + (1 - g) * res["tp"].detach()).cpu().numpy()
        y_inf1 = forward_fused(m, xb).cpu().numpy()
        y_inf2 = forward_fused(m, xb).cpu().numpy()
        with torch.no_grad():
            y_val = m(xb, export_image=False, fp64=False,
                      use_variable_chunk=P.USE_VARIABLE_CHUNK,
                      return_branches=False).cpu().numpy()
        c1 = pred_close(y_grad, y_inf1, ATOL, RTOL)
        c2 = pred_close(y_grad, y_val, ATOL, RTOL)
        bw = bitwise_equal(y_inf1, y_inf2)
        assoc = float(np.abs(naive - y_grad).max())
        rows.append({"gid": gid, "n": int(n),
                     "num_patch_input": npi,
                     "grad_vs_eval_fused": {"close": c1[0],
                                            "max_abs": c1[1],
                                            "max_viol": c1[2]},
                     "grad_vs_validate_fused": {"close": c2[0],
                                                "max_abs": c2[1],
                                                "max_viol": c2[2]},
                     "eval_twice_bitwise": bool(bw),
                     "naive_fusion_max_abs_diag": assoc})
        if not (c1[0] and c2[0] and bw):
            errs.append("%s: entry mismatch grad/eval=%s val=%s "
                        "bitwise=%s" % (gid, c1[0], c2[0], bw))
        print("B2 %s ok=%s max_abs=%.2e assoc_diag=%.2e"
              % (gid, c1[0] and c2[0] and bw,
                 max(c1[1], c2[1]), assoc), flush=True)
        _drop(m)

    # batch=1 + 4-visible-column probe: Traffic/H6 is the real geometry
    # with num_patch_input == 4 (int(0.888*14*0.4)); smallest H end-to-end
    dd = DomainData(pinfo["protocol_dir"], pinfo["splits"], "Traffic")
    spv = dd.load_split(6, "val")
    m = _new_model("Traffic", 6)
    npi = int(m.num_patch_input)
    if npi != 4:
        errs.append("4-visible probe: Traffic/H6 num_patch_input=%d != 4"
                    % npi)
    xb = torch.from_numpy(np.ascontiguousarray(spv["X"][:1])).to(DEVICE)
    m.train()
    y_grad = forward_branches(m, xb)["output"].detach().cpu().numpy()
    y_inf = forward_fused(m, xb).cpu().numpy()
    if y_grad.shape != (1, 6, 1):
        errs.append("Traffic/H6 batch=1 bad shape %s"
                    % (y_grad.shape,))
    c = pred_close(y_grad, y_inf, ATOL, RTOL)
    row_b1 = {"gid": "Traffic_H6__batch1", "close": c[0],
              "max_abs": c[1], "num_patch_input": npi}
    if not c[0]:
        errs.append("Traffic/H6 batch=1 mismatch viol=%.3e" % c[2])
    _drop(m)

    row_4col = {"geometry": "Traffic_H6", "num_patch_input": npi}
    return errs, {"groups": rows, "batch1": row_b1, "probe_4col": row_4col}


def _train_step(m, opt, xb, yb):
    opt.zero_grad()
    res = forward_branches(m, xb)
    chan, tp = res["channel"], res["tp"]
    g = torch.sigmoid(m.fusion_logit)
    gate_pred = g * chan.detach() + (1 - g) * tp.detach()
    chan_loss = torch.mean((chan - yb) ** 2)
    tp_loss = torch.mean((tp - yb) ** 2)
    gate_loss = torch.mean((gate_pred - yb) ** 2)
    total = chan_loss + tp_loss + gate_loss
    total.backward()
    zero_names, none_names = gradient_gate(m)
    opt.step()
    return (res, (float(chan_loss.item()), float(tp_loss.item()),
                  float(gate_loss.item()), float(total.item())),
            zero_names, none_names, float(g.item()))


def _forensics(m, pre):
    backbone_ok = backbone_digest(m) == pre["backbone"]
    buffer_ok = buffer_digest(m) == pre["buffer"]
    post = named_tensor_sha(m)
    moved = sorted(n for n in post if post[n] != pre["named"][n])
    return backbone_ok, buffer_ok, moved


def b3_real_steps(pinfo, pd):
    import torch
    errs, rows = [], {}
    plan = list(P.PRECHECK) + [("Traffic", 6)]
    for dom, h in plan:
        gid = P.group_id(dom, h)
        dd = DomainData(pinfo["protocol_dir"], pinfo["splits"], dom)
        sp = dd.load_split(h, "train")
        X, Y = sp["X"], sp["Y"]
        n_train = int(X.shape[0])
        m = _new_model(dom, h)
        pre = {"backbone": backbone_digest(m),
               "buffer": buffer_digest(m),
               "named": named_tensor_sha(m)}
        opt = build_optimizer(m, P.LR)
        m.train()
        perm = np.random.RandomState(P.SEED).permutation(n_train)
        steps_log, step1_pattern = [], None
        n_steps = min(B3_STEPS, P.expected_steps(n_train))
        for si in range(n_steps):
            idx = perm[si * P.BATCH:(si + 1) * P.BATCH]
            xb = torch.from_numpy(np.ascontiguousarray(X[idx])).to(DEVICE)
            yb = torch.from_numpy(np.ascontiguousarray(Y[idx])).to(DEVICE)
            res, losses, zero_names, none_names, gv = _train_step(
                m, opt, xb, yb)
            if si == 0:
                step1_pattern = {"n_none": len(none_names),
                                 "none": sorted(none_names),
                                 "n_zero": len(zero_names),
                                 "zero": sorted(zero_names),
                                 "n_nonzero": 111 - len(none_names)
                                 - len(zero_names)}
            steps_log.append({"step": si + 1, "losses": losses,
                              "gate": gv,
                              "n_zero_grad": len(zero_names)})
        bb_ok, buf_ok, moved = _forensics(m, pre)
        row = {"gid": gid, "n_train": n_train, "steps": steps_log,
               "step1_gradient_pattern_diag": step1_pattern,
               "backbone_unchanged": bb_ok, "buffer_unchanged": buf_ok,
               "n_moved_tensors": len(moved)}
        if not bb_ok:
            errs.append("%s: backbone moved during B3" % gid)
        if not buf_ok:
            errs.append("%s: buffers moved during B3" % gid)
        if n_steps < 3 and (dom, h) != ("Security", 12):
            errs.append("%s: only %d real steps available (<3)"
                        % (gid, n_steps))
        row["steps_run"] = n_steps
        print("B3 %s loss0=%.6f loss%d=%.6f moved=%d zero1=%d none1=%d"
              % (gid, steps_log[0]["losses"][3], n_steps,
                 steps_log[-1]["losses"][3], len(moved),
                 step1_pattern["n_zero"], step1_pattern["n_none"]),
              flush=True)

        # real ragged tail batch as one more real step
        tail_n = n_train % P.BATCH or P.BATCH
        idx_tail = perm[-tail_n:]
        xb = torch.from_numpy(np.ascontiguousarray(X[idx_tail])).to(DEVICE)
        yb = torch.from_numpy(np.ascontiguousarray(Y[idx_tail])).to(DEVICE)
        _, losses_t, _, _, _ = _train_step(m, opt, xb, yb)
        row["tail_batch"] = {"n": int(tail_n), "losses": losses_t}
        if (dom, h) == ("Security", 12) and tail_n != 6:
            errs.append("Security/H12 tail batch %d != 6" % tail_n)

        if (dom, h) == ("Health", 48):
            meta = save_muse_ckpt(m, {
                "gid": gid + "__preflight", "domain": dom, "h": int(h),
                "p": int(P.FIXED_P[dom]), "L": int(P.TABLE[dom]["L"]),
                "num_patch_input": int(m.num_patch_input),
                "n_train_windows": n_train,
                "n_val_windows": int(dd.load_split(h, "val")["X"].shape[0]),
                "steps_per_epoch": P.expected_steps(n_train),
                "seed": P.SEED,
                "protocol_json_sha256": P.PROTOCOL_JSON_SHA256,
                "splits_json_sha256": P.SPLITS_JSON_SHA256,
                "scaler_json_sha256": P.SCALER_JSON_SHA256,
                "weights_sha256": P.WEIGHTS_SHA256,
                "train_config": train_config(),
                "epochs_run": B3_STEPS, "best_epoch": B3_STEPS,
                "steps": B3_STEPS + 1,
                "note": "preflight B3 model after %d real steps + tail "
                        "batch" % B3_STEPS,
            }, pd / "b4_health_H48.npz", pd / "b4_health_H48.meta.json")
            row["b4_ckpt"] = {"npz_sha256": meta["npz_sha256"],
                              "packed_sha256": meta["packed_sha256"]}
        rows[gid] = row
        _drop(m)
    return errs, rows


def b4_restore(pinfo, pd):
    import torch
    probe_npz = pd / "b4_probe_pred.npz"
    cmd = [sys.executable, "-m",
           "repro_timemmd_common_v1.restore_probe",
           "--dom", "Health", "--h", "48", "--out", str(probe_npz),
           "--ckpt", str(pd / "b4_health_H48.npz"),
           "--meta", str(pd / "b4_health_H48.meta.json")]
    r = subprocess.run(cmd, capture_output=True, text=True)
    out = {"cmd": " ".join(cmd), "returncode": r.returncode,
           "stdout": r.stdout[-2000:], "stderr": r.stderr[-2000:],
           "errors": []}
    if r.returncode != 0:
        out["errors"].append("restore_probe subprocess failed rc=%d"
                             % r.returncode)
        return out
    probe_meta = load_json(str(probe_npz) + ".json")
    parent_meta = load_json(pd / "b4_health_H48.meta.json")
    if probe_meta["live_packed_sha256"] != parent_meta["packed_sha256"]:
        out["errors"].append(
            "subprocess restored params != ckpt (packed_sha)")
    if probe_meta["ckpt_npz_sha256"] != parent_meta["npz_sha256"]:
        out["errors"].append("ckpt npz sha mismatch across processes")
    # parent side: same strict restore, same probe inputs
    m = _new_model("Health", 48)
    from repro_timemmd_common_v1.muse_train import \
        restore_muse_ckpt_strict
    restore_muse_ckpt_strict(m, pd / "b4_health_H48.npz", parent_meta)
    live = {npz_key(n): t.detach().cpu().numpy()
            for n, t in list(m.named_parameters())
            + list(m.named_buffers())}
    if packed_sha(live) != parent_meta["packed_sha256"]:
        out["errors"].append(
            "parent restored params != ckpt (packed_sha)")
    dd = DomainData(pinfo["protocol_dir"], pinfo["splits"], "Health")
    spv = dd.load_split(48, "val")
    xb = torch.from_numpy(np.ascontiguousarray(spv["X"][:8])).to(DEVICE)
    y_parent = forward_fused(m, xb).cpu().numpy()
    y_sub = dict(np.load(str(probe_npz)))["pred"]
    c = pred_close(y_sub, y_parent, ATOL, RTOL)
    out["pred_close"] = {"close": c[0], "max_abs": c[1], "max_viol": c[2]}
    if not c[0]:
        out["errors"].append("probe pred not close: viol %.3e" % c[2])
    _drop(m)
    return out


def b5_timing(pinfo):
    import torch
    rows = {}
    for dom, h in P.PRECHECK:
        gid = P.group_id(dom, h)
        dd = DomainData(pinfo["protocol_dir"], pinfo["splits"], dom)
        sp = dd.load_split(h, "train")
        X, Y = sp["X"], sp["Y"]
        n_train = int(X.shape[0])
        m = _new_model(dom, h)
        opt = build_optimizer(m, P.LR)
        m.train()
        perm = np.random.RandomState(P.SEED).permutation(n_train)
        torch.cuda.synchronize()
        times = []
        n_full = max(1, n_train // P.BATCH)
        for si in range(B5_STEPS):
            b = (si % n_full) * P.BATCH
            idx = perm[b:min(b + P.BATCH, n_train)]
            xb = torch.from_numpy(np.ascontiguousarray(X[idx])).to(DEVICE)
            yb = torch.from_numpy(np.ascontiguousarray(Y[idx])).to(DEVICE)
            t0 = time.time()
            _train_step(m, opt, xb, yb)
            torch.cuda.synchronize()
            times.append(time.time() - t0)
        peak = int(torch.cuda.max_memory_allocated() >> 20)
        torch.cuda.reset_peak_memory_stats()
        rate = float(np.mean(times[B5_WARMUP:]))
        steps_ep = P.expected_steps(n_train)
        rows[gid] = {"s_per_step": round(rate, 4),
                     "steps_per_epoch": steps_ep,
                     "est_epoch_s": round(rate * steps_ep, 1),
                     "est_10epoch_s": round(rate * steps_ep * 10, 1),
                     "peak_mem_mb": peak,
                     "val_windows": int(dd.load_split(h, "val")["X"]
                                        .shape[0])}
        print("B5 %s %.4fs/step epoch~%.1fs peak=%dMB"
              % (gid, rate, rate * steps_ep, peak), flush=True)
        _drop(m)
    splits = load_json(Path(P.protocol_dir()) / "splits.json")
    total_steps = sum(P.expected_steps(
        int(splits[d]["window_counts"][str(h)]["train"]))
        for d in P.ORDER for h in P.TABLE[d]["H"])
    rates = [r["s_per_step"] for r in rows.values()]
    return {"groups": rows, "steps_one_epoch": total_steps,
            "upper_bound_steps_10epoch": P.TRAIN_STEPS_TEN_EPOCH_UPPER,
            "mean_s_per_step": round(float(np.mean(rates)), 4),
            "crude_full_train_upper_s": round(
                max(rates) * total_steps * 10, 1),
            "note": "REPORT ONLY — no budget stop condition; upper bound "
                    "uses worst measured rate x 1,767 steps x 10 epochs",
            "go": True}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--protocol_dir", default=None)
    ap.add_argument("--out_root", default=None)
    args = ap.parse_args()

    setup_deterministic_env()
    pd = (Path(args.out_root) if args.out_root else P.out_root()) \
        / "preflight"
    pd.mkdir(parents=True, exist_ok=True)
    report = {"device": DEVICE, "probe_n": PROBE_N, "atol": ATOL,
              "rtol": RTOL, "errors": []}
    t0 = time.time()
    try:
        pinfo = P.verify_frozen(args.protocol_dir)
        report["verify_frozen"] = {
            k: pinfo[k] for k in ("weights_sha256", "selected_p_sha256",
                                  "train_windows", "val_windows",
                                  "test_windows", "steps_total")}
    except SystemExit as e:
        report["errors"].append("verify_frozen: %s" % e)
        dump_json(pd / "preflight_report.json", report)
        raise

    for stage, fn in (("b1", lambda: b1_protocol_data(pinfo)),
                      ("b2", lambda: b2_entry_equivalence(pinfo)),
                      ("b3", lambda: b3_real_steps(pinfo, pd))):
        try:
            if stage == "b1":
                e, d = fn()
                report[stage] = {"errors": e, "groups": d}
            elif stage == "b2":
                e, d = fn()
                report[stage] = {"errors": e, "rows": d}
            else:
                e, d = fn()
                report[stage] = {"errors": e, "groups": d}
                report["b4"] = b4_restore(pinfo, pd)
                report["b5"] = b5_timing(pinfo)
        except Exception as ex:  # record and fail, report must land
            import traceback
            report[stage] = {"errors": ["exception: %r" % ex],
                             "traceback": traceback.format_exc()[-4000:]}
        print("%s done, %s" % (
            stage.upper(),
            len(report[stage].get("errors", [])) if stage in report
            else "crashed"), flush=True)

    all_errors = list(report.get("b1", {}).get("errors", [])) \
        + list(report.get("b2", {}).get("errors", [])) \
        + list(report.get("b3", {}).get("errors", [])) \
        + list(report.get("b4", {}).get("errors", []))
    report["errors"] = all_errors
    report["all_pass"] = not all_errors
    if "b5" in report:
        report["b5"]["go"] = report["all_pass"]
    report["elapsed_s"] = round(time.time() - t0, 1)
    dump_json(pd / "preflight_report.json", report)
    print("PREFLIGHT %s (%.1fs, %d errors)"
          % ("ALL PASS" if report["all_pass"] else "FAILED",
             report["elapsed_s"], len(all_errors)))
    if not report["all_pass"]:
        for e in all_errors[:20]:
            print("  - %s" % e)
        sys.exit(2)


if __name__ == "__main__":
    main()
