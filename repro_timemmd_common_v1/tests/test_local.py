"""Local CPU unit tests for EXP-029 (run with ~/.venvs/vts/bin/python).
All model mechanics run with load_ckpt=False (no MAE file locally);
restore tests use backfill=False (formal paths always backfill).
"""
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

from repro_timemmd_common_v1 import protocol as P
from repro_timemmd_common_v1.adapter import build_muse, configure_muse, \
    configure_raw, forward_branches, forward_fused, trainable_inventory
from repro_timemmd_common_v1.common import backbone_digest, \
    buffer_digest, pred_close, seed_everything
from repro_timemmd_common_v1.muse_train import EarlyStopping, \
    build_optimizer, epoch_val_f64, gradient_gate, group_fingerprint, \
    npz_key, packed_sha, restore_muse_ckpt_strict, save_muse_ckpt, \
    set_type3_lr, train_config, weighted_batch_mean

PASS = []


def check(name, cond, extra=""):
    if not cond:
        print("FAIL %s %s" % (name, extra))
        sys.exit(1)
    PASS.append(name)
    print("ok %s" % name)


def fresh(dom="Traffic", h=6):
    seed_everything(P.SEED)
    m = build_muse(load_ckpt=False)
    configure_muse(m, dom, h)
    return m


def test_inventory():
    m = fresh()
    inv = trainable_inventory(m)
    check("inventory_111", inv["n_trainable_tensors"] == 111,
          str(inv["n_trainable_tensors"]))
    check("buffers_7", len(inv["buffer_names"]) == 7,
          str(inv["buffer_names"]))
    dead = [n for n in inv["trainable_names"]
            if n in P.DEAD_TENSORS]
    check("dead_in_trainable", sorted(dead) == sorted(P.DEAD_TENSORS))
    check("backbone_frozen_count",
          len(inv["frozen_names"]) > 100
          and all(n.startswith("vision_model.")
                  for n in inv["frozen_names"]))
    print("  n_trainable_scalars(C=1) =", inv["n_trainable_scalars"])


def test_configure_once():
    m = fresh()
    try:
        configure_muse(m, "Traffic", 6)
        check("configure_once_shim", False, "second call allowed")
    except RuntimeError:
        check("configure_once_shim", True)
    m2 = build_muse(load_ckpt=False)
    configure_raw(m2, 6, 96, 1)
    check("npi_floor_1", int(m2.num_patch_input) == 1,
          str(int(m2.num_patch_input)))
    m3 = build_muse(load_ckpt=False)
    configure_muse(m3, "Traffic", 6)
    check("probe_4col_traffic_h6", int(m3.num_patch_input) == 4,
          str(int(m3.num_patch_input)))


def _rand_batch(m, n=8):
    L = int(m.context_len)
    x = np.random.RandomState(0).randn(n, L, 1).astype(np.float32)
    return torch.from_numpy(x)


def test_gradient_pattern():
    m = fresh()
    m.train()
    opt = build_optimizer(m, P.LR)
    xb = _rand_batch(m)
    yb = torch.randn_like(xb[:, :int(m.pred_len), :])
    patterns = []
    for _ in range(2):
        opt.zero_grad()
        res = forward_branches(m, xb)
        g = torch.sigmoid(m.fusion_logit)
        gate_pred = g * res["channel"].detach() \
            + (1 - g) * res["tp"].detach()
        total = torch.mean((res["channel"] - yb) ** 2) \
            + torch.mean((res["tp"] - yb) ** 2) \
            + torch.mean((gate_pred - yb) ** 2)
        total.backward()
        zero_names, none_names = gradient_gate(m)
        opt.step()
        nz = sum(1 for _, p in m.named_parameters()
                 if p.requires_grad and p.grad is not None
                 and p.grad.abs().sum().item() > 0)
        patterns.append((sorted(none_names), len(zero_names), nz))
    check("step1_none_whitelist",
          patterns[0][0] == sorted(P.DEAD_TENSORS), str(patterns[0][0]))
    check("step1_pattern_2_27_82",
          patterns[0][1] == 27 and patterns[0][2] == 82,
          str(patterns[0][1:]))
    check("step2_zero_collapse", patterns[1][1] < 27,
          str(patterns[1][1]))
    check("gate_moved",
          float(m.fusion_logit.item()) != 0.0)


def test_fused_equivalence():
    m = fresh()
    m.eval()
    xb = _rand_batch(m)
    with torch.no_grad():
        res = forward_branches(m, xb)
    g = torch.sigmoid(m.fusion_logit).detach()
    naive = (g * res["channel"].detach()
             + (1 - g) * res["tp"].detach()).numpy()
    out = res["output"].detach().numpy()
    close, max_abs, _ = pred_close(naive, out, 1e-4, 1e-4)
    check("branch_vs_output_close", close, "max_abs=%.2e" % max_abs)
    fused = forward_fused(m, xb).numpy()
    c2, m2, _ = pred_close(fused, out, P.ATOL, P.RTOL)
    check("fused_entry_matches_output", c2, "max_abs=%.2e" % m2)


def test_optimizer_and_type3():
    m = fresh()
    opt = build_optimizer(m, P.LR)
    groups = {g["parameter_group"]: g for g in opt.param_groups}
    check("dual_groups", set(groups) == {"adapters", "gate"})
    check("gate_lr_10x",
          groups["gate"]["lr"] == 10 * groups["adapters"]["lr"])
    n_gate = len(groups["gate"]["params"])
    n_adapter = len(groups["adapters"]["params"])
    check("group_sizes", (n_adapter, n_gate) == (110, 1),
          "%d/%d" % (n_adapter, n_gate))
    traj_a, traj_g = [], []
    for ep in range(1, 11):
        lr = set_type3_lr(opt, ep, P.LR)
        traj_a.append(lr)
        traj_g.append(groups["gate"]["lr"])
    want = [P.LR if e < 3 else P.LR * 0.9 ** (e - 3)
            for e in range(1, 11)]
    check("type3_trajectory",
          all(abs(a - w) < 1e-15 for a, w in zip(traj_a, want))
          and all(g == 10 * a for a, g in zip(traj_a, traj_g)),
          str(traj_a[:5]))


def test_early_stopping():
    class M:
        pass
    es = EarlyStopping(2, 0)
    m = M()
    check("es_first_improves", es(1.0, m) is True)
    check("es_tie_improves", es(1.0, m) is True and es.counter == 0)
    check("es_worse_counter", es(1.1, m) is False and es.counter == 1)
    stop = es(1.2, m)
    check("es_stops", stop is False and es.early_stop is True)


def test_val_f64_accum():
    m = fresh()
    m.eval()
    n, L, H = 11, int(m.context_len), int(m.pred_len)
    X = np.random.RandomState(1).randn(n, L, 1).astype(np.float32)
    Y = np.random.RandomState(2).randn(n, H, 1).astype(np.float32)
    got = epoch_val_f64(m, X, Y, batch=4)
    # same batch structure (f32 forward is batch-size dependent by
    # design — comparisons across batch sizes are NOT bitwise); the
    # invariant under test is the f64 accumulation itself
    ssum, cnt = 0.0, 0
    with torch.no_grad():
        for lo in range(0, n, 4):
            out = forward_fused(
                m, torch.from_numpy(X[lo:min(lo + 4, n)]))
            d = (out.numpy()
                 - Y[lo:min(lo + 4, n)]).astype(np.float64)
            ssum += float((d ** 2).sum())
            cnt += d.size
    want = ssum / cnt
    check("val_f64_accum_identity",
          abs(got - want) <= 1e-12 * max(abs(want), 1e-30),
          "got %.16f want %.16f" % (got, want))


def _ckpt_roundtrip_setup(tmp):
    m = fresh("Health", 48)
    m.train()
    opt = build_optimizer(m, P.LR)
    xb = _rand_batch(m, 4)
    yb = torch.randn_like(xb[:, :48, :])
    opt.zero_grad()
    res = forward_branches(m, xb)
    g = torch.sigmoid(m.fusion_logit)
    total = torch.mean((res["channel"] - yb) ** 2) \
        + torch.mean((res["tp"] - yb) ** 2) \
        + torch.mean((g * res["channel"].detach()
                      + (1 - g) * res["tp"].detach() - yb) ** 2)
    total.backward()
    gradient_gate(m)
    opt.step()
    meta = save_muse_ckpt(m, {
        "gid": "Health_H48", "domain": "Health", "h": 48,
        "p": P.FIXED_P["Health"], "L": P.TABLE["Health"]["L"],
        "num_patch_input": int(m.num_patch_input),
        "seed": P.SEED,
        "weights_sha256": None,
        "train_config": train_config(),
        "epochs_run": 3, "best_epoch": 3, "steps": 3,
    }, Path(tmp) / "ck.npz", Path(tmp) / "ck.meta.json")
    return m, meta


def test_ckpt_roundtrip():
    tmp = tempfile.mkdtemp()
    try:
        m, meta = _ckpt_roundtrip_setup(tmp)
        seed_everything(P.SEED)
        m2 = build_muse(load_ckpt=False)
        configure_muse(m2, "Health", 48)
        restore_muse_ckpt_strict(
            m2, Path(tmp) / "ck.npz", meta, backfill=False)
        live = {npz_key(n): t.detach().cpu().numpy()
                for n, t in list(m2.named_parameters())
                + list(m2.named_buffers())}
        check("restore_packed_equal",
              packed_sha(live) == meta["packed_sha256"])
        check("restore_backbone_equal",
              backbone_digest(m2) == meta["backbone_sha256"])
        check("restore_buffer_equal",
              buffer_digest(m2) == meta["buffer_sha256"])
        m.eval()
        m2.eval()
        xb = _rand_batch(m, 3)
        pa = forward_fused(m, xb).numpy()
        pb = forward_fused(m2, xb).numpy()
        c, mx, _ = pred_close(pb, pa, P.ATOL, P.RTOL)
        check("restore_pred_close", c, "max_abs=%.2e" % mx)
        return meta, Path(tmp) / "ck.npz"
    finally:
        pass


def test_tamper_and_binding():
    tmp = tempfile.mkdtemp()
    try:
        m, meta = _ckpt_roundtrip_setup(tmp)
        ck = Path(tmp) / "ck.npz"
        # tamper bytes
        bad = Path(tmp) / "bad.npz"
        raw = bytearray(ck.read_bytes())
        raw[len(raw) // 2] ^= 0xFF
        bad.write_bytes(bytes(raw))
        seed_everything(P.SEED)
        m2 = build_muse(load_ckpt=False)
        configure_muse(m2, "Health", 48)
        try:
            restore_muse_ckpt_strict(m2, bad, meta, backfill=False)
            check("tamper_rejected", False)
        except SystemExit:
            check("tamper_rejected", True)
        # cross-group binding
        m3 = build_muse(load_ckpt=False)
        configure_muse(m3, "Traffic", 6)
        try:
            restore_muse_ckpt_strict(m3, ck, meta, backfill=False)
            check("binding_rejected", False)
        except SystemExit:
            check("binding_rejected", True)
        # missing buffer key
        z = dict(np.load(str(ck)))
        dropped = {k: v for k, v in z.items()
                   if not k.startswith("mask")}
        ck2 = Path(tmp) / "nobuffer.npz"
        np.savez(str(ck2), **dropped)
        m4 = build_muse(load_ckpt=False)
        configure_muse(m4, "Health", 48)
        try:
            restore_muse_ckpt_strict(m4, ck2, meta, backfill=False)
            check("missing_buffer_rejected", False)
        except SystemExit:
            check("missing_buffer_rejected", True)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_misc():
    check("expected_steps", P.expected_steps(100, 32) == 4
          and P.expected_steps(96, 32) == 3)
    check("npz_key", npz_key("a.b.c") == "a__b__c")
    check("weighted_mean",
          abs(weighted_batch_mean([1.0, 2.0], [3, 1]) - 1.25) < 1e-12)
    g1 = group_fingerprint("A_H1", {"x": 1})
    g2 = group_fingerprint("A_H1", {"x": 1})
    g3 = group_fingerprint("A_H2", {"x": 1})
    check("fingerprint", g1 == g2 and g1 != g3)
    perms = []
    rs = np.random.RandomState(P.SEED)
    for _ in range(2):
        perms.append(rs.permutation(100))
    rs2 = np.random.RandomState(P.SEED)
    p0 = rs2.permutation(100)
    check("perm_reproducible_and_distinct",
          np.array_equal(perms[0], p0)
          and not np.array_equal(perms[0], perms[1]))
    tc = train_config()
    check("train_config_pins",
          tc["lr"] == 1e-4 and tc["gate_lr_mult"] == 10
          and tc["lradj"] == "type3" and tc["patience"] == 2
          and tc["batch"] == 32 and tc["seed"] == 2021)


def test_import_isolation():
    import subprocess
    code = (
        "import sys; import repro_timemmd_common_v1.adapter;"
        " import repro_timemmd_common_v1.muse_train;"
        " bad = [m for m in sys.modules if m.startswith(('ts_benchmark',"
        " 'repro_timemmd_common_ln_v1', 'sklearn', 'matplotlib'))];"
        " print('BAD:' + ','.join(bad) if bad else 'CLEAN')")
    r = subprocess.run([sys.executable, "-c", code],
                       capture_output=True, text=True,
                       cwd=str(P.REPO_ROOT))
    check("import_isolation", "CLEAN" in r.stdout,
          r.stdout[-200:] + r.stderr[-200:])


if __name__ == "__main__":
    test_inventory()
    test_configure_once()
    test_gradient_pattern()
    test_fused_equivalence()
    test_optimizer_and_type3()
    test_early_stopping()
    test_val_f64_accum()
    test_ckpt_roundtrip()
    test_tamper_and_binding()
    test_misc()
    test_import_isolation()
    print("ALL %d LOCAL TESTS PASS" % len(PASS))
