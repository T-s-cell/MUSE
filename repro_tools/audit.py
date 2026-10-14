"""Gradient/loss audits run at selected steps (never in performance runs).

Replays are verification-only: they run forwards/backwards on the live model
after the official optimizer step, restore RNG state afterwards, and leave
.grd cleared (set_to_none), which matches the official optimizer.zero_grad()
at the start of the next step.
"""

import torch

from .common import grad_max_abs, grad_snapshot, restore_rng, save_rng


def _loss_parts(muse_adapter, batch):
    input, target, input_mark, target_mark = batch
    core = muse_adapter._core_model()
    horizon = muse_adapter.config.horizon
    sd = muse_adapter._series_dim
    out = core(
        input,
        fp64=muse_adapter.config.fp64,
        use_variable_chunk=muse_adapter.config.use_variable_chunk,
        return_branches=True,
    )
    tgt = target[:, -horizon:, :sd]
    channel, processed_target = muse_adapter._post_process(
        out["channel"][:, :, :sd], tgt
    )
    tp, _ = muse_adapter._post_process(out["tp"][:, :, :sd], tgt)
    channel_loss = torch.mean((channel - processed_target) ** 2)
    tp_loss = torch.mean((tp - processed_target) ** 2)
    gate = torch.sigmoid(core.fusion_logit)
    gate_prediction = gate * channel.detach() + (1 - gate) * tp.detach()
    gate_loss = torch.mean((gate_prediction - processed_target) ** 2)
    return core, processed_target, channel, tp, channel_loss, tp_loss, gate_loss


def run_step_audit(muse_adapter, batch, step, out_dir, results):
    model = muse_adapter.model
    core = muse_adapter._core_model()

    rng = save_rng()
    official = grad_snapshot(core)
    trainables = [n for n, p in core.named_parameters() if p.requires_grad]
    adapter_names = [n for n in trainables if n != "fusion_logit"]

    def norms(snap, names):
        return {n: norms_one(snap.get(n)) for n in names}

    def norms_one(g):
        if g is None:
            return None
        return float(g.norm().item())

    audit = {"step": step, "trainable_param_count": len(trainables)}

    # (a) branch losses only: fusion_logit must receive None/zero grad
    model.zero_grad(set_to_none=True)
    core_ref, tgt, channel, tp, channel_loss, tp_loss, gate = _loss_parts(
        muse_adapter, batch
    )
    (channel_loss + tp_loss).backward()
    audit["branch_backward"] = {
        "fusion_logit_grad": norms_one(core.fusion_logit.grad),
        "adapter_grad_norms": norms(grad_snapshot(core), adapter_names),
        "channel_loss": float(channel_loss.item()),
        "tp_loss": float(tp_loss.item()),
        "gate_value": float(gate.item()),
        "shapes": {
            "channel": list(channel.shape),
            "tp": list(tp.shape),
        },
        "finite": bool(
            torch.isfinite(channel).all()
            and torch.isfinite(tp).all()
            and torch.isfinite(core.fusion_logit).all()
        ),
    }

    # (b) fusion loss only: every adapter/injection param must get None/zero grad
    model.zero_grad(set_to_none=True)
    _, _, _, _, _, _, gate_loss = _loss_parts(muse_adapter, batch)
    gate_loss.backward()
    fused_snap = grad_snapshot(core)
    non_gate = {
        n: g for n, g in fused_snap.items() if n != "fusion_logit"
    }
    max_nongate_grad = 0.0
    for g in non_gate.values():
        if g is not None:
            max_nongate_grad = max(max_nongate_grad, float(g.abs().max().item()))
    official_gate_grad = official.get("fusion_logit")
    audit["fusion_backward"] = {
        "fusion_logit_grad": norms_one(core.fusion_logit.grad),
        "max_abs_nongate_grad": max_nongate_grad,
        "gate_loss": float(gate_loss.item()),
        "official_fusion_logit_grad_absmax": (
            None
            if official_gate_grad is None
            else float(official_gate_grad.abs().max().item())
        ),
        "fusion_vs_official_absdiff": (
            None
            if official_gate_grad is None or core.fusion_logit.grad is None
            else float(
                (core.fusion_logit.grad.cpu() - official_gate_grad)
                .abs()
                .max()
                .item()
            )
        ),
    }

    # classification of official grads at this step
    off = grad_max_abs(official)
    audit["official_grad_classification"] = {
        "grad_none": sorted(n for n in trainables if off.get(n) is None),
        "grad_zero": sorted(
            n for n in trainables if off.get(n) is not None and off[n] == 0.0
        ),
        "grad_nonzero_count": sum(
            1 for n in trainables if off.get(n) is not None and off[n] > 0.0
        ),
    }
    if step == results.get("_first_audited_step"):
        audit["official_grad_absmax"] = off

    model.zero_grad(set_to_none=True)
    restore_rng(rng)

    # persistent classification across audited steps is aggregated by the hook
    for name in trainables:
        v = off.get(name)
        key = "none" if v is None else ("zero" if v == 0.0 else "nonzero")
        results.setdefault("_param_states", {}).setdefault(name, set()).add(key)

    return audit
