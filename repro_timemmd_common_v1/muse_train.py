"""MUSE supervised training core for EXP-029 (optimizer, type3+gate 10x
schedule, EarlyStopping verbatim, per-epoch reshuffle from one
RandomState(2021), 118-key checkpoint save/strict-restore with MAE
backbone backfill).

Semantics replicated from ts_benchmark/baselines/muse/muse.py +
deep_forecasting_model_base.py forecast_fit (read, not imported):
  loss = MSE(channel, y) + MSE(tp, y)
         + MSE(sigmoid(fusion_logit)*channel.detach()
               + (1-sigmoid(fusion_logit))*tp.detach(), y)
  validate = full-split fused forward, squared errors accumulated in f64
  early stop: patience=2 delta=0, tie counts as improvement, break BEFORE
  the epoch-end lr adjustment
"""
import copy
import hashlib
import time

import numpy as np
import torch

from repro_timemmd_common_v1 import protocol as P
from repro_timemmd_common_v1.adapter import build_muse, configure_muse, \
    forward_branches, predict_split, trainable_inventory
from repro_timemmd_common_v1.common import backbone_digest, buffer_digest, \
    dump_json, named_tensor_sha, seed_everything, setup_deterministic_env, \
    sha256_of
from repro_timemmd_common_v1.data_timemmd import DomainData, infer_batches


# ---------------------------------------------------------------- utils

def npz_key(name):
    return name.replace(".", "__")


def _array_bytes(a):
    return np.ascontiguousarray(a).tobytes()


def packed_sha(arrays):
    """sha256 over {name: ndarray} — name, shape, dtype, bytes, sorted."""
    h = hashlib.sha256()
    for name in sorted(arrays):
        a = np.asarray(arrays[name])
        h.update(name.encode())
        h.update(str(a.shape).encode())
        h.update(str(a.dtype).encode())
        h.update(_array_bytes(a))
    return h.hexdigest()


def weighted_batch_mean(losses, sizes):
    if not losses:
        return None
    total = float(sum(l * s for l, s in zip(losses, sizes)))
    n = int(sum(sizes))
    return total / n if n else None


def train_config():
    return {
        "lr": P.LR, "gate_lr_mult": P.GATE_LR_MULT, "betas": list(P.BETAS),
        "eps": P.EPS, "weight_decay": P.WD, "max_epochs": P.MAX_EPOCHS,
        "patience": P.PATIENCE, "stop_delta": P.DELTA, "lradj": P.LRADJ,
        "batch": P.BATCH, "seed": P.SEED, "fp64": P.FP64,
        "use_variable_chunk": P.USE_VARIABLE_CHUNK,
        "norm_const": P.NORM_CONST, "align_const": P.ALIGN_CONST,
        "interpolation": P.INTERPOLATION, "arch": P.ARCH,
        "num_latents": 1, "latent_dim": 192, "adapter_num_heads": 4,
        "channel_depth": 1,
    }


class EarlyStopping:
    """Verbatim copy of ts_benchmark/baselines/utils.py::EarlyStopping
    (tie counts as improvement; counter reset on improvement)."""

    def __init__(self, patience=7, delta=0):
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta

    def __call__(self, val_loss, model):
        score = -val_loss
        improved = False
        if self.best_score is None:
            self.best_score = score
            improved = True
            print(
                f"Validation loss decreased ({self.val_loss_min:.6f} --> "
                f"{val_loss:.6f}).  Saving model ..."
            )
            self.val_loss_min = val_loss
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f"EarlyStopping counter: {self.counter} "
                  f"out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            improved = True
            print(
                f"Validation loss decreased ({self.val_loss_min:.6f} --> "
                f"{val_loss:.6f}).  Saving model ..."
            )
            self.val_loss_min = val_loss
            self.counter = 0
        return improved


# ------------------------------------------------------- optimizer / lr

def build_optimizer(model, lr):
    """Dual-group Adam, replicating muse.py::_init_criterion_and_optimizer:
    adapters @ lr (Adam default wd=0), fusion_logit @ 10*lr wd=0."""
    gate = model.fusion_logit
    adapter_parameters = [
        p for p in model.parameters()
        if p.requires_grad and p is not gate
    ]
    if len(adapter_parameters) != P.EXPECTED_TRAINABLE_TENSORS - 1:
        raise RuntimeError("adapter group size %d != %d"
                           % (len(adapter_parameters),
                              P.EXPECTED_TRAINABLE_TENSORS - 1))
    return torch.optim.Adam([
        {"params": adapter_parameters, "lr": lr,
         "parameter_group": "adapters"},
        {"params": [gate], "lr": P.GATE_LR_MULT * lr, "weight_decay": 0.0,
         "parameter_group": "gate"},
    ], betas=P.BETAS, eps=P.EPS)


def set_type3_lr(optimizer, epoch, lr):
    """type3 base adjust (epoch = completed epochs) + muse gate 10x
    re-apply. Returns the adapters-group lr."""
    new_lr = lr if epoch < 3 else lr * (0.9 ** ((epoch - 3) // 1))
    for group in optimizer.param_groups:
        if group["parameter_group"] == "adapters":
            group["lr"] = new_lr
    for group in optimizer.param_groups:
        if group["parameter_group"] == "gate":
            group["lr"] = 10 * new_lr
    return new_lr


def epoch_val_f64(model, valX, valY, batch=None):
    """Replicates muse.py::validate: eval, no_grad, fused forward
    (fp64=False per protocol pin), squared errors accumulated in f64."""
    batch = int(batch or P.BATCH)
    n = valX.shape[0]
    model.eval()
    square_sum, count = 0.0, 0
    with torch.no_grad():
        for lo, hi in infer_batches(n, batch):
            xb = torch.from_numpy(valX[lo:hi]).to(
                next(model.parameters()).device)
            yb = torch.from_numpy(valY[lo:hi]).to(xb.device)
            out = model(xb, export_image=False, fp64=False,
                        use_variable_chunk=P.USE_VARIABLE_CHUNK,
                        return_branches=False)
            err = (out - yb).double()
            square_sum += err.square().sum().item()
            count += err.numel()
    model.train()
    return square_sum / count


# ------------------------------------------------------------ gradients

def gradient_gate(model):
    """None only for the pinned dead tensors; non-finite is fatal.
    Zero-grad names are diagnostic-only (patch_up zero-init upstream).
    Returns (zero_names, none_names)."""
    zero_names, none_names = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.grad is None:
            none_names.append(name)
            continue
        if not torch.isfinite(p.grad).all():
            raise RuntimeError("non-finite gradient: %s" % name)
        if not bool(p.grad.abs().sum().item() > 0):
            zero_names.append(name)
    unexpected = [n for n in none_names if n not in P.DEAD_TENSORS]
    if unexpected:
        raise RuntimeError("unexpected grad=None: %s" % unexpected)
    missing = [n for n in P.DEAD_TENSORS if n not in none_names]
    if missing:
        raise RuntimeError("pinned dead tensors now have grads: %s"
                           % missing)
    return zero_names, none_names


# ------------------------------------------------------------ ckpt I/O

def save_muse_ckpt(model, meta_extra, out_npz, out_meta):
    """118 arrays (111 params + 7 buffers, npz_key dots->__) +
    ckpt_names string array of ORIGINAL names; sidecar meta json."""
    arrays, names = {}, []
    for name, p in model.named_parameters():
        arrays[npz_key(name)] = p.detach().cpu().numpy()
        names.append(name)
    for name, b in model.named_buffers():
        arrays[npz_key(name)] = b.detach().cpu().numpy()
        names.append(name)
    arrays["ckpt_names"] = np.asarray(names)
    psha = packed_sha({k: v for k, v in arrays.items()
                       if k != "ckpt_names"})
    np.savez(out_npz, **arrays)
    meta = dict(meta_extra)
    meta.update({
        "n_arrays": len(names),
        "packed_sha256": psha,
        "npz_sha256": sha256_of(out_npz),
        "backbone_sha256": backbone_digest(model),
        "buffer_sha256": buffer_digest(model),
        "code_md5": P.package_code_md5s(),
        "repo_commit": P.repo_commit(),
    })
    dump_json(out_meta, meta)
    return meta


def restore_muse_ckpt_strict(model, npz_path, meta, backfill=True):
    """Strict restore into a live, already-configured MUSE:
    file sha -> bindings -> exact key set (118) -> shape/dtype ->
    packed_sha recompute -> copy_ -> live packed_sha equal -> MAE
    backbone backfill with strict=True full load -> backbone/buffer
    digests equal meta. Acceptance = restored params bitwise equal to
    ckpt (no digest_pre != digest_post requirement). backfill=False is
    for CPU unit tests only (no MAE file); the backbone digest check
    still runs, so the live backbone must equal the saved one."""
    from repro_timemmd_common_v1.common import die
    if sha256_of(npz_path) != meta["npz_sha256"]:
        die("[TIER-1] ckpt npz sha mismatch: %s" % npz_path)
    if int(getattr(model, "context_len")) != int(meta["L"]) \
            or int(getattr(model, "pred_len")) != int(meta["h"]) \
            or int(getattr(model, "periodicity")) != int(meta["p"]):
        die("[TIER-1] ckpt geometry binding mismatch for %s" % meta["gid"])

    z = np.load(npz_path, allow_pickle=False)
    stored_names = [str(x) for x in z["ckpt_names"]]
    stored_keys = sorted(k for k in z.files if k != "ckpt_names")
    if sorted(npz_key(n) for n in stored_names) != stored_keys:
        die("[TIER-1] ckpt_names vs npz keys mismatch: %s" % npz_path)

    live = {}
    for name, t in list(model.named_parameters()) + \
            list(model.named_buffers()):
        live[name] = t
    if set(stored_names) != set(live):
        die("[TIER-1] ckpt key set mismatch: missing %s extra %s"
            % (sorted(set(stored_names) - set(live))[:4],
               sorted(set(live) - set(stored_names))[:4]))

    packed = {}
    for name in stored_names:
        arr = z[npz_key(name)]
        t = live[name]
        if arr.shape != tuple(t.shape) or \
                arr.dtype != np.dtype(t.detach().cpu().numpy().dtype):
            die("[TIER-1] ckpt tensor mismatch %s: %s/%s vs %s/%s"
                % (name, arr.shape, arr.dtype, tuple(t.shape),
                   t.detach().cpu().numpy().dtype))
        packed[npz_key(name)] = arr
    if packed_sha(packed) != meta["packed_sha256"]:
        die("[TIER-1] ckpt packed_sha mismatch: %s" % npz_path)

    with torch.no_grad():
        for name in stored_names:
            live[name].copy_(torch.from_numpy(packed[npz_key(name)]))
    live_packed = {npz_key(name): live[name].detach().cpu().numpy()
                   for name in stored_names}
    if packed_sha(live_packed) != meta["packed_sha256"]:
        die("[TIER-1] restored model != ckpt (packed_sha): %s"
            % npz_path)

    # gold-standard backbone backfill from the pinned MAE file
    if backfill:
        wpath = P.mae_weights_path()
        if sha256_of(wpath) != P.WEIGHTS_SHA256 \
                or meta.get("weights_sha256") not in (None,
                                                      P.WEIGHTS_SHA256):
            die("[TIER-1] MAE weights sha mismatch during restore")
        ck = torch.load(wpath, map_location="cpu")
        sd = model.state_dict()
        for k, v in ck["model"].items():
            fk = "vision_model." + k
            if fk not in sd or tuple(sd[fk].shape) != tuple(v.shape):
                die("[TIER-1] MAE backfill key/shape mismatch: %s" % fk)
            sd[fk] = v.to(sd[fk].dtype)
        model.load_state_dict(sd, strict=True)

    if backbone_digest(model) != meta["backbone_sha256"]:
        die("[TIER-1] backbone digest mismatch after restore: %s"
            % npz_path)
    if buffer_digest(model) != meta["buffer_sha256"]:
        die("[TIER-1] buffer digest mismatch after restore: %s"
            % npz_path)
    return {"ok": True, "gid": meta["gid"],
            "npz_sha256": meta["npz_sha256"],
            "packed_sha256": meta["packed_sha256"]}


# ---------------------------------------------------------- train loop

def group_fingerprint(gid, guards):
    import json as _json
    blob = _json.dumps({"gid": gid, "guards": guards,
                        "train_config": train_config()},
                       sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()


def full_guards(pinfo, selected_p_sha=None, preflight_sha=None):
    from repro_timemmd_common_v1.common import core_guards
    g = core_guards(pinfo, P.BATCH, selected_p_sha=selected_p_sha,
                    preflight_sha=preflight_sha)
    g["train_config"] = train_config()
    g["expected_trainable_tensors"] = P.EXPECTED_TRAINABLE_TENSORS
    g["expected_buffers"] = P.EXPECTED_BUFFERS
    g["dead_tensors"] = list(P.DEAD_TENSORS)
    g["anchors"] = {
        "exp027_overall": [P.EXP027_OVERALL_MSE, P.EXP027_OVERALL_MAE],
        "exp028_overall": [P.EXP028_OVERALL_MSE, P.EXP028_OVERALL_MAE],
        "tats_overall": [P.TATS_OVERALL_MSE, P.TATS_OVERALL_MAE],
    }
    return g


def train_one_group(dom, h, pinfo, out_root, device):
    gid = P.group_id(dom, h)
    data = DomainData(pinfo["protocol_dir"], pinfo["splits"], dom)
    train = data.load_split(h, "train")
    val = data.load_split(h, "val")
    n_train = int(train["X"].shape[0])
    steps_per_epoch = P.expected_steps(n_train)

    setup_deterministic_env()
    seed_everything(P.SEED)
    model = build_muse(load_ckpt=True)
    configure_muse(model, dom, h)
    inv = trainable_inventory(model)
    digests_pre = {"backbone_sha256": backbone_digest(model),
                   "buffer_sha256": buffer_digest(model),
                   "named_sha": named_tensor_sha(model)}
    model.to(device)
    if backbone_digest(model) != digests_pre["backbone_sha256"]:
        raise RuntimeError("backbone changed on .to(device)")
    optimizer = build_optimizer(model, P.LR)

    rs = np.random.RandomState(P.SEED)
    early = EarlyStopping(P.PATIENCE, P.DELTA)
    best_state, best_epoch, epochs_run, steps = None, None, 0, 0
    history = []
    t0 = time.time()

    for epoch in range(P.MAX_EPOCHS):
        perm = rs.permutation(n_train)
        perm_sha = hashlib.sha256(_array_bytes(perm)).hexdigest()
        losses = {"chan": [], "tp": [], "gate": [], "total": []}
        sizes = []
        zero_seen = set()
        for b0 in range(0, n_train, P.BATCH):
            idx = perm[b0:b0 + P.BATCH]
            xb = torch.from_numpy(train["X"][idx]).to(device)
            yb = torch.from_numpy(train["Y"][idx]).to(device)
            optimizer.zero_grad()
            res = forward_branches(model, xb)
            chan, tp = res["channel"], res["tp"]
            g = torch.sigmoid(model.fusion_logit)
            gate_pred = g * chan.detach() + (1 - g) * tp.detach()
            chan_loss = torch.mean((chan - yb) ** 2)
            tp_loss = torch.mean((tp - yb) ** 2)
            gate_loss = torch.mean((gate_pred - yb) ** 2)
            total = chan_loss + tp_loss + gate_loss
            total.backward()
            zero_names, _ = gradient_gate(model)
            zero_seen.update(zero_names)
            optimizer.step()
            bs = len(idx)
            losses["chan"].append(float(chan_loss.item()))
            losses["tp"].append(float(tp_loss.item()))
            losses["gate"].append(float(gate_loss.item()))
            losses["total"].append(float(total.item()))
            sizes.append(bs)
            steps += 1
        val_mse = epoch_val_f64(model, val["X"], val["Y"])
        improved = early(val_mse, model)
        if improved:
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch + 1
        epochs_run = epoch + 1
        lr_now = next(gr["lr"] for gr in optimizer.param_groups
                      if gr["parameter_group"] == "adapters")
        history.append({
            "epoch": epoch + 1,
            "perm_sha256": perm_sha,
            "loss_weighted": {k: weighted_batch_mean(v, sizes)
                              for k, v in losses.items()},
            "losses": losses, "batch_sizes": sizes,
            "val_mse_f64": val_mse,
            "improved": improved,
            "gate_value": float(torch.sigmoid(
                model.fusion_logit).item()),
            "adapters_lr": float(lr_now),
            "zero_grad_tensors": sorted(zero_seen),
        })
        if early.early_stop:
            break
        set_type3_lr(optimizer, epoch + 1, P.LR)

    elapsed = time.time() - t0
    if best_state is None:
        raise RuntimeError("%s: no best state (early stopping never "
                           "improved)" % gid)
    if not (3 <= epochs_run <= P.MAX_EPOCHS):
        raise RuntimeError("%s: epochs_run %d outside [3,%d]"
                           % (gid, epochs_run, P.MAX_EPOCHS))
    if steps != epochs_run * steps_per_epoch:
        raise RuntimeError("%s: steps %d != epochs_run %d x %d"
                           % (gid, steps, epochs_run, steps_per_epoch))

    model.load_state_dict(best_state, strict=True)
    model.eval()
    named_post = named_tensor_sha(model)
    changed = sorted(n for n in named_post
                     if named_post[n] != digests_pre["named_sha"].get(n))
    if backbone_digest(model) != digests_pre["backbone_sha256"]:
        raise RuntimeError("%s: backbone moved during training" % gid)
    if buffer_digest(model) != digests_pre["buffer_sha256"]:
        raise RuntimeError("%s: buffers moved during training" % gid)

    peak_mem = 0
    if device != "cpu":
        peak_mem = int(torch.cuda.max_memory_allocated() / (1 << 20))
        torch.cuda.reset_peak_memory_stats()

    ck_npz = out_root / P.ckpt_npz_name(dom, h)
    ck_meta = out_root / P.ckpt_meta_name(dom, h)
    meta = save_muse_ckpt(model, {
        "gid": gid, "domain": dom, "h": int(h), "p": P.FIXED_P[dom],
        "L": P.TABLE[dom]["L"],
        "num_patch_input": int(model.num_patch_input),
        "n_train_windows": n_train,
        "n_val_windows": int(val["X"].shape[0]),
        "steps_per_epoch": steps_per_epoch,
        "seed": P.SEED,
        "protocol_json_sha256": P.PROTOCOL_JSON_SHA256,
        "splits_json_sha256": P.SPLITS_JSON_SHA256,
        "scaler_json_sha256": P.SCALER_JSON_SHA256,
        "weights_sha256": P.WEIGHTS_SHA256,
        "selected_p_sha256": pinfo["selected_p_sha256"],
        "train_config": train_config(),
        "perm_sha256": [r["perm_sha256"] for r in history],
        "epochs_run": epochs_run, "best_epoch": best_epoch,
        "steps": steps, "elapsed_s": round(elapsed, 1),
        "peak_mem_mb": peak_mem,
        "inventory": {"n_trainable_tensors": inv["n_trainable_tensors"],
                      "n_trainable_scalars": inv["n_trainable_scalars"],
                      "buffer_names": inv["buffer_names"]},
        "changed_tensors": changed,
        "n_changed_tensors": len(changed),
    }, ck_npz, ck_meta)

    val_pred = predict_split(model, val["X"])
    val_npz = out_root / P.val_npz_name(dom, h)
    np.savez(val_npz,
             pred=val_pred.astype(np.float32),
             truth=val["Y"].astype(np.float32),
             s=val["s"], orig_row=val["orig_row"],
             label_dates=val["label_dates"],
             window_index=val["window_index"])

    dump_json(out_root / P.train_log_name(dom, h), {
        "gid": gid, "domain": dom, "h": int(h), "p": P.FIXED_P[dom],
        "n_train_windows": n_train,
        "n_val_windows": int(val["X"].shape[0]),
        "steps_per_epoch": steps_per_epoch,
        "epochs_run": epochs_run, "best_epoch": best_epoch,
        "steps": steps,
        "mean_train_loss_weighted": weighted_batch_mean(
            [x for r in history for x in r["losses"]["total"]],
            [s for r in history for s in r["batch_sizes"]]),
        "history": history,
        "changed_tensors": changed,
        "elapsed_s": round(elapsed, 1), "peak_mem_mb": peak_mem,
        "ckpt_meta": {k: meta[k] for k in
                      ("packed_sha256", "npz_sha256",
                       "backbone_sha256", "buffer_sha256")},
    })
    return {"gid": gid, "state": "done", "epochs_run": epochs_run,
            "best_epoch": best_epoch, "steps": steps,
            "elapsed_s": round(elapsed, 1), "peak_mem_mb": peak_mem,
            "n_train_windows": n_train,
            "final_val_mse_f64": history[-1]["val_mse_f64"],
            "best_val_mse_f64": min(r["val_mse_f64"]
                                    for r in history),
            "n_changed_tensors": len(changed),
            "ckpt_npz_sha256": meta["npz_sha256"],
            "perm_sha256": [r["perm_sha256"] for r in history]}
