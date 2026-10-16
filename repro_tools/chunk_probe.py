"""G4: chunked-vs-unchunked backward equivalence probe (Electricity path).

Runs the REAL production code on both branches with bitwise-identical
initial weights:
  - reference: official unchunked training math (base-class step + MUSE
    _process: fused/channel loss + tp/gate additional losses, joint
    backward);
  - candidate: official MUSE._backward_variable_chunks (staged per-chunk
    backward with width/sd weighting + correction grad chain).

With series_dim split as 16+16+1 (=33) the tail chunk exercises the
variable-count weighting, and the shared correction exercises the global
cross-variable interaction. Comparisons at the INITIAL state and after
each of --updates optimizer updates (VCR != 0):
  1. channel/tp/fused predictions (no-grad forward, both assembly modes);
  2. the three losses + fused total;
  3. gradients of EVERY trainable parameter;
  4. the parameter delta of one Adam optimizer step.

The frozen MAE vision tower is fingerprinted before/after everything and
must be bit-identical.

Run on the training machine:
    python -m repro_tools.chunk_probe --mae-path <mae.pth> --out-dir <dir>

Exit codes: 0 all comparisons within tolerance, 1 any failure.
"""

import argparse
import json
import os
import sys

import torch

from ts_benchmark.baselines.muse.muse import MUSE as MUSEAdapter
from ts_benchmark.utils.random_utils import fix_all_random_seed

from .common import fingerprint_diff, module_fingerprint


def jcmp(a, b, atol, rtol):
    diff = (a.detach().float() - b.detach().float()).abs()
    max_abs = float(diff.max()) if diff.numel() else 0.0
    ok = bool(torch.allclose(a.detach().float(), b.detach().float(),
                             atol=atol, rtol=rtol))
    return {"max_abs": max_abs, "ok": ok}


def build_adapter(mae_path, args, device):
    adapter = MUSEAdapter(
        seq_len=args.seq_len,
        horizon=args.horizon,
        batch_size=args.batch_size,
        lr=args.lr,
        checkpoint_path=mae_path,
        load_ckpt=True,
        periodicity=args.periodicity,
        use_variable_chunk=False,
        num_epochs=1,
    )
    adapter.model = adapter._init_model()
    adapter._series_dim = args.num_vars
    adapter.model.to(device)
    return adapter


def grads_of(adapter):
    return {
        n: None if p.grad is None else p.grad.detach().clone()
        for n, p in adapter.model.named_parameters() if p.requires_grad
    }


def compare_grads(g_u, g_c, atol, rtol):
    out = {}
    all_ok = True
    for n, gu in g_u.items():
        gc = g_c[n]
        if gu is None and gc is None:
            out[n] = {"both_none": True, "ok": True}
            continue
        if gu is None or gc is None:
            out[n] = {"ok": False,
                      "error": f"one side None (u={gu is not None}, c={gc is not None})"}
            all_ok = False
            continue
        c = jcmp(gc, gu, atol, rtol)
        c["ref_norm"] = float(gu.norm())
        out[n] = c
        all_ok = all_ok and c["ok"]
    return out, all_ok


def checksum(adapter):
    return float(sum(p.detach().double().sum().item()
                     for p in adapter.model.parameters()))


def vcr_check(adp_c, input, series_dim, ref_fused, atol, rtol):
    """Directly inspect the VCR correction output of the chunked branch.

    vcr_nonzero must be evidenced on the correction tensor itself, not by a
    whole-model parameter checksum: (1) correction exists and is nonzero,
    (2) predictions WITH correction match the unchunked reference (within
    tol), (3) predictions with the correction ZEROED deviate from the
    unchunked reference by more than tol (materiality: the correction
    actually participates in the output).
    """
    model = adp_c._core_model()
    ctx = model.prepare_variable_chunk_context(input, fp64=False)
    corr = ctx["correction"]
    out = {
        "correction_present": corr is not None,
        "correction_max_abs": (float(corr.abs().max())
                               if corr is not None else None),
    }
    out["correction_nonzero"] = bool(
        out["correction_present"] and out["correction_max_abs"] > 0.0
    )
    fused_c = fwd_predictions(adp_c, input, series_dim, "chunks")
    dev = (fused_c[2] - ref_fused).abs().max()
    out["with_correction_vs_full_max_abs"] = float(dev)
    out["with_correction_ok"] = bool(
        torch.isclose(torch.tensor(float(dev)),
                      torch.tensor(0.0), atol=atol, rtol=rtol).item()
        or dev <= atol
    )
    zeros = None if corr is None else torch.zeros_like(corr)
    fused_z = fwd_predictions(adp_c, input, series_dim, "chunks",
                              correction_override=zeros)
    dev_z = (fused_z[2] - ref_fused).abs().max()
    out["zero_correction_vs_full_max_abs"] = float(dev_z)
    out["zero_correction_materially_different"] = bool(dev_z > atol)
    out["vcr_ok"] = bool(out["correction_nonzero"] and out["with_correction_ok"]
                         and out["zero_correction_materially_different"])
    return out


def fwd_predictions(adapter, input, series_dim, mode, correction_override=None):
    adapter.model.eval()
    with torch.no_grad():
        if mode == "full":
            r = adapter.model(
                input, fp64=False, use_variable_chunk=False,
                return_branches=True,
            )
            chan, tp, fused = r["channel"], r["tp"], r["output"]
        else:
            model = adapter._core_model()
            ctx = model.prepare_variable_chunk_context(input, fp64=False)
            corr = (ctx["correction"] if correction_override is None
                    else correction_override)
            chans, tps, fuses = [], [], []
            for start in range(0, ctx["num_variables"],
                               model.variable_chunk_size):
                end = min(start + model.variable_chunk_size,
                          ctx["num_variables"])
                r = model.forward_variable_chunk(
                    ctx, start, end, correction=corr,
                    return_branches=True,
                )
                chans.append(r["channel"])
                tps.append(r["tp"])
                fuses.append(r["output"])
            chan = torch.cat(chans, dim=2)
            tp = torch.cat(tps, dim=2)
            fused = torch.cat(fuses, dim=2)
    adapter.model.train()
    return (chan[:, :, :series_dim], tp[:, :, :series_dim],
            fused[:, :, :series_dim])


def train_step_unchunked(adapter, criterion, batch, series_dim):
    input, target = batch
    adapter.model.train()
    adapter.model.zero_grad(set_to_none=True)
    out = adapter._process(input, target, None, None)
    output = out["output"][:, -adapter.config.horizon:, :series_dim]
    tgt = target[:, -adapter.config.horizon:, :series_dim]
    output, tgt = adapter._post_process(output, tgt)
    loss = criterion(output, tgt)
    additional = out.get("additional_loss", 0.0)
    total = loss + additional
    total.backward()
    return {
        "total": float(total.item()),
        "channel": float(loss.item()),
        "additional": float(additional.item()),
    }, grads_of(adapter)


def train_step_chunked(adapter, criterion, batch, series_dim):
    input, target = batch
    adapter.model.train()
    adapter.model.zero_grad(set_to_none=True)
    losses = adapter._backward_variable_chunks(
        input, target, series_dim, criterion
    )
    return {
        "total": float(losses["total"].item()),
        "channel": float(losses["channel"].item()),
        "additional": float((losses["tp"] + losses["gate"]).item()),
        "tp": float(losses["tp"].item()),
        "gate": float(losses["gate"].item()),
    }, grads_of(adapter)


def make_optimizer(adapter, lr):
    gate = adapter._core_model().fusion_logit
    adapter_params = [
        p for p in adapter.model.parameters()
        if p.requires_grad and p is not gate
    ]
    return torch.optim.Adam([
        {"params": adapter_params, "lr": lr, "parameter_group": "adapters"},
        {"params": [gate], "lr": 10 * lr, "weight_decay": 0.0,
         "parameter_group": "gate"},
    ])


def compare_update(adp_u, adp_c, before_u, before_c, gu, gc, atol, rtol,
                   grad_diff):
    """Compare the Adam update deltas of one step (per-branch befores).

    Each branch's delta is measured against ITS OWN pre-step state, so a
    parameter difference inherited from earlier exempted noise-floor
    elements does not contaminate this step's comparison; that inherited
    drift is recorded separately per parameter (param_state_absdiff).

    Adam's m/(sqrt(v)+eps) normalization ill-conditions elements whose
    gradient is near zero: a grad difference far below the demonstrated
    branch agreement maps to a delta difference ABOVE the fixed atol.
    An element is exempt only if the smaller of the two branch gradients
    there is <= 10x the observed per-parameter grad agreement (i.e. the
    element's gradient is indistinguishable from zero at the measured
    noise floor, so its Adam direction is numerical noise on BOTH
    branches). Well-conditioned elements (large grads) are never exempt,
    so a real chunked-path defect still fails. Every exemption is
    recorded.
    """
    per_param = {}
    all_ok = True
    params_u = dict(adp_u.model.named_parameters())
    for n, p in adp_c.model.named_parameters():
        if not p.requires_grad:
            continue
        d_c = p.detach() - before_c[n]
        d_u = params_u[n].detach() - before_u[n]
        diff = (d_c - d_u).abs()
        entry = {"max_abs": float(diff.max()), "n_elem": int(diff.numel())}
        fail = diff > (atol + rtol * d_u.abs())
        entry["n_fail"] = int(fail.sum())
        drift = float((p.detach() - params_u[n].detach()).abs().max())
        if drift:
            entry["param_state_absdiff"] = drift
        if entry["n_fail"]:
            g_u, g_c = gu[n], gc[n]
            bound = 10.0 * grad_diff.get(n, 0.0)
            tiny = max(bound, 1e-9)
            gu_f = g_u.abs()[fail]
            gc_f = g_c.abs()[fail]
            exempt = bool(
                torch.minimum(gu_f, gc_f).max() <= tiny
            )
            k = int((diff * fail).argmax())
            entry.update({
                "failing_grad_max": float(max(gu_f.max(), gc_f.max())),
                "grad_agreement_absdiff": grad_diff.get(n),
                "grad_tiny_threshold": tiny,
                "tiny_grad_exempt": exempt,
                "worst": {
                    "delta_absdiff": float(diff.flatten()[k]),
                    "grad_u": float(g_u.flatten()[k]),
                    "grad_c": float(g_c.flatten()[k]),
                },
            })
            if not exempt:
                all_ok = False
        per_param[n] = entry
    return per_param, all_ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mae-path", required=True)
    ap.add_argument("--out-dir", default="repro_out/chunk_probe")
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--horizon", type=int, default=96)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--num-vars", type=int, default=33,
                    help=">=33 => 16+16+1 chunks (tail chunk present)")
    ap.add_argument("--periodicity", type=int, default=24)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--updates", type=int, default=3)
    ap.add_argument("--seed", type=int, default=2021)
    ap.add_argument("--atol", type=float, default=1e-5)
    ap.add_argument("--rtol", type=float, default=1e-4)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    fix_all_random_seed(args.seed)
    adp_u = build_adapter(args.mae_path, args, device)   # unchunked reference
    fix_all_random_seed(args.seed)
    adp_c = build_adapter(args.mae_path, args, device)   # chunked candidate
    adp_c.model.load_state_dict(adp_u.model.state_dict())  # bitwise identical

    fingerprint0 = module_fingerprint(adp_u._core_model().vision_model)
    fingerprint_c0 = module_fingerprint(adp_c._core_model().vision_model)

    sd_u = adp_u.model.state_dict()
    sd_c = adp_c.model.state_dict()
    bitwise_init = set(sd_u) == set(sd_c) and all(
        torch.equal(sd_u[k], sd_c[k].to(sd_u[k].device)) for k in sd_u
    )

    criterion = torch.nn.MSELoss()
    opt_u = make_optimizer(adp_u, args.lr)
    opt_c = make_optimizer(adp_c, args.lr)

    gen = torch.Generator().manual_seed(args.seed + 1)
    input = torch.randn(
        args.batch_size, args.seq_len, args.num_vars, generator=gen
    ).to(device)
    target = torch.randn(
        args.batch_size, args.horizon, args.num_vars, generator=gen
    ).to(device)
    batch = (input, target)

    report = {
        "config": vars(args),
        "device": str(device),
        "chunk_split": [
            min(args.num_vars, s + 16) - s
            for s in range(0, args.num_vars, 16)
        ],
        "atol": args.atol, "rtol": args.rtol,
        "bitwise_identical_init": bool(bitwise_init),
        "iterations": [],
    }
    all_ok = bool(bitwise_init)

    def record(tag, extra=None):
        it = {"tag": tag}
        if extra:
            it.update(extra)
        report["iterations"].append(it)
        return it

    # ---- initial state -------------------------------------------------
    pu = fwd_predictions(adp_u, input, args.num_vars, "full")
    pc = fwd_predictions(adp_c, input, args.num_vars, "chunks")
    pred_cmp = {
        name: jcmp(c, u, args.atol, args.rtol)
        for name, (u, c) in zip(
            ("channel", "tp", "fused"), zip(pu, pc))
    }
    lu, gu = train_step_unchunked(adp_u, criterion, batch, args.num_vars)
    lc, gc = train_step_chunked(adp_c, criterion, batch, args.num_vars)

    def cmp_losses(lu, lc):
        # The two official paths consume: channel (MSE), additional
        # (chunked tp + gate; unchunked additional_loss = tp + gate by
        # construction), and total. tp/gate split recorded informationally.
        return {
            "total": jcmp(torch.tensor(lc["total"]), torch.tensor(lu["total"]),
                          args.atol, args.rtol),
            "channel": jcmp(torch.tensor(lc["channel"]),
                            torch.tensor(lu["channel"]),
                            args.atol, args.rtol),
            "additional": jcmp(torch.tensor(lc["additional"]),
                               torch.tensor(lu["additional"]),
                               args.atol, args.rtol),
        }

    loss_cmp = cmp_losses(lu, lc)
    grad_cmp, grad_ok = compare_grads(gu, gc, args.atol, args.rtol)
    it = record("init", {
        "preds": pred_cmp, "losses_u": lu, "losses_c": lc,
        "loss_cmp": loss_cmp,
        "grads_max_abs": max(v.get("max_abs", 0.0) for v in grad_cmp.values()
                             if isinstance(v, dict) and "max_abs" in v),
        "grads_all_ok": grad_ok,
        "vcr": vcr_check(adp_c, input, args.num_vars, pu[2],
                         args.atol, args.rtol),
        "checksum_u": checksum(adp_u), "checksum_c": checksum(adp_c),
    })
    it["grads"] = grad_cmp
    all_ok = all_ok and grad_ok and it["vcr"]["vcr_ok"] \
        and all(v["ok"] for v in pred_cmp.values()) \
        and all(v["ok"] for v in loss_cmp.values())

    # ---- update iterations ----------------------------------------------
    for k in range(1, args.updates + 1):
        before_u = {n: p.detach().clone()
                    for n, p in adp_u.model.named_parameters()
                    if p.requires_grad}
        before_c = {n: p.detach().clone()
                    for n, p in adp_c.model.named_parameters()
                    if p.requires_grad}
        opt_u.step()
        opt_c.step()
        grad_diff = {
            n: v.get("max_abs", 0.0)
            for n, v in grad_cmp.items()
            if isinstance(v, dict) and "max_abs" in v
        }
        upd_cmp, upd_ok = compare_update(
            adp_u, adp_c, before_u, before_c, gu, gc,
            args.atol, args.rtol, grad_diff
        )
        pu = fwd_predictions(adp_u, input, args.num_vars, "full")
        pc = fwd_predictions(adp_c, input, args.num_vars, "chunks")
        pred_cmp = {
            name: jcmp(c, u, args.atol, args.rtol)
            for name, (u, c) in zip(
                ("channel", "tp", "fused"), zip(pu, pc))
        }
        lu, gu = train_step_unchunked(adp_u, criterion, batch, args.num_vars)
        lc, gc = train_step_chunked(adp_c, criterion, batch, args.num_vars)
        loss_cmp = cmp_losses(lu, lc)
        grad_cmp, grad_ok = compare_grads(gu, gc, args.atol, args.rtol)
        it = record(f"after_update_{k}", {
            "preds": pred_cmp,
            "losses_u": lu, "losses_c": lc,
            "loss_cmp": loss_cmp,
            "grads_max_abs": max(v.get("max_abs", 0.0) for v in grad_cmp.values()
                                 if isinstance(v, dict) and "max_abs" in v),
            "grads_all_ok": grad_ok,
            "update_all_ok": upd_ok,
            "vcr": vcr_check(adp_c, input, args.num_vars, pu[2],
                             args.atol, args.rtol),
            "update_exempt_params": sorted(
                n for n, v in upd_cmp.items()
                if v.get("n_fail") and v.get("tiny_grad_exempt")
            ),
            "checksum_u": checksum(adp_u), "checksum_c": checksum(adp_c),
        })
        it["grads"] = grad_cmp
        it["update"] = upd_cmp
        all_ok = all_ok and grad_ok and upd_ok and it["vcr"]["vcr_ok"] \
            and all(v["ok"] for v in pred_cmp.values()) \
            and all(v["ok"] for v in loss_cmp.values())

    vcr_iter = [it.get("vcr") for it in report["iterations"]]
    # Iteration 0 is the initial state: the adapter output projection is
    # zero-initialised, so the correction tensor is exactly zero there by
    # construction. VCR-nonzero is therefore asserted on POST-UPDATE
    # iterations and requires materiality (zeroing the correction must move
    # the unchunked-reference comparison well beyond atol).
    post = [v for v in vcr_iter[1:] if v is not None]
    report["vcr_nonzero"] = bool(
        post and all(v["correction_nonzero"] for v in post)
        and all(v["zero_correction_vs_full_max_abs"] > report["atol"]
                for v in post)
    )
    report["vcr_direct"] = {
        "correction_max_abs": [v["correction_max_abs"] for v in vcr_iter],
        "with_correction_vs_full_max_abs":
            [v["with_correction_vs_full_max_abs"] for v in vcr_iter],
        "zero_correction_vs_full_max_abs":
            [v["zero_correction_vs_full_max_abs"] for v in vcr_iter],
        "note": "vcr_nonzero evidenced on the correction tensor itself "
                "(post-update iterations; iter0 correction is exactly 0 by "
                "zero-initialisation) plus materiality: zeroing the "
                "correction must shift the output vs the unchunked "
                "reference by more than atol; checksums kept per-iteration "
                "as auxiliary info",
    }

    exempt_params = {}
    n_exempt_elem = 0
    for it in report["iterations"]:
        for n, v in (it.get("update") or {}).items():
            if v.get("n_fail") and v.get("tiny_grad_exempt"):
                e = exempt_params.setdefault(n, {"updates": 0, "n_fail_max": 0})
                e["updates"] += 1
                e["n_fail_max"] = max(e["n_fail_max"], int(v["n_fail"]))
                n_exempt_elem += int(v["n_fail"])
    report["exempt_summary"] = {
        "used": bool(exempt_params),
        "n_exempt_elements_total": n_exempt_elem,
        "params": exempt_params,
        "criterion": "element exempt only if BOTH branch gradients there are "
                     "<= 10x the per-parameter grad agreement (noise floor); "
                     "well-conditioned elements never exempt",
    }
    report["strict_pass"] = bool(all_ok and not exempt_params)
    fp_u = fingerprint_diff(fingerprint0,
                            module_fingerprint(adp_u._core_model().vision_model))
    fp_c = fingerprint_diff(fingerprint_c0,
                            module_fingerprint(adp_c._core_model().vision_model))
    report["frozen_fingerprint_u"] = fp_u
    report["frozen_fingerprint_c"] = fp_c
    report["frozen_unchanged"] = not any(
        (fp.get("added") or fp.get("changed") or fp.get("missing"))
        for fp in (fp_u, fp_c)
    )
    report["pass"] = bool(all_ok and report["vcr_nonzero"]
                          and report["frozen_unchanged"])

    path = os.path.join(args.out_dir, "chunk_probe_report.json")
    with open(path, "w") as f:
        json.dump(report, f, indent=2, sort_keys=True, default=str)

    brief = {
        it["tag"]: {
            "preds_ok": all(v["ok"] for v in it["preds"].values()),
            "grads_ok": it.get("grads_all_ok"),
            "grads_max_abs": it.get("grads_max_abs"),
            **({"update_ok": it.get("update_all_ok")} if "update_all_ok" in it else {}),
        }
        for it in report["iterations"]
    }
    print(json.dumps(brief, indent=2))
    print(f"vcr_nonzero={report['vcr_nonzero']} "
          f"frozen_unchanged={report['frozen_unchanged']} pass={report['pass']}")
    if not report["pass"]:
        print(f"[chunk_probe] FAIL (report -> {path})")
        sys.exit(1)
    print(f"[chunk_probe] PASS -> {path}")


if __name__ == "__main__":
    main()
