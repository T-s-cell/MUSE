"""Environment-variable-gated hook bound onto the official training loop.

Insertion points (deep_forecasting_model_base.forecast_fit):
  on_data_ready   after train/valid loaders are built
  on_train_ready  after criterion/optimizer are created
  on_fit_start    after model.to(device), before epoch loop
  on_step_begin   before forward, per batch
  on_loss         after total_loss = loss + additional_loss
  on_step_end     after optimizer.step()
  timed_validate  wraps the official epoch-end validate call
  on_fit_end      after the epoch loop

RNG neutrality: validation/audit sections save and restore the full RNG state
(torch/cpu, cuda, numpy, python) plus the model train/eval flag, so subsequent
training steps are unaffected. No restore happens on plain steps.
"""

import csv
import os
import time

import torch

from .audit import run_step_audit
from .bundle import save_val_point, write_manifest
from .common import (
    ReproConfig,
    batch_sha256,
    fingerprint_diff,
    module_fingerprint,
    restore_rng,
    save_rng,
    trainable_checksum,
    write_json,
)
from .timing import StepTimer, read_peak_mem, reset_peak_mem, step_stats


class ReproHook:
    def __init__(self, cfg):
        self.cfg = cfg
        os.makedirs(cfg.out_dir, exist_ok=True)
        self.global_step = 0
        self.stop = False
        self.rows = []
        self.val_records = []
        self.audit_records = []
        self.best = None
        self.loss_cache = None
        self.loss_cache_parts = None
        self.fingerprint = None
        self.train_info = {}
        self.fit_t0 = None
        self.timer = StepTimer()
        self.cur_batch = None
        self._last_train_batch = None
        self.model_base = None
        self.optimizer = None
        self.criterion = None
        self.valid_loader = None
        self.series_dim = None
        self._first_audited = None
        self._param_states = {}
        self._param_sha = {}

    @classmethod
    def from_env(cls):
        return cls(ReproConfig())

    # ---- registration of context -------------------------------------
    def on_data_ready(self, model_base, train_dataset, train_loader,
                      valid_loader, series_dim):
        bs = model_base.config.batch_size
        n_train = len(train_dataset)
        self.train_info = {
            "train_windows": n_train,
            "train_batches_full": n_train // bs,
            "train_drop_last": train_loader.drop_last,
            "train_tail_samples": n_train % bs,
            "valid_windows": None if valid_loader is None else len(valid_loader.dataset),
            "valid_batches": None if valid_loader is None else len(valid_loader),
            "series_dim": series_dim,
            "seq_len": model_base.config.seq_len,
            "horizon": model_base.config.horizon,
            "label_len": getattr(model_base.config, "label_len", None),
            "freq": getattr(model_base.config, "freq", None),
            "periodicity": getattr(model_base.config, "periodicity", None),
        }
        self.valid_loader = valid_loader
        self.series_dim = series_dim
        self._steps_csv = open(
            os.path.join(self.cfg.out_dir, "steps.csv"), "w", newline=""
        )
        writer = csv.writer(self._steps_csv)
        writer.writerow([
            "step", "epoch", "batch_idx", "loss_total", "loss_channel",
            "loss_tp", "loss_gate", "gate_value", "lr_adapters", "lr_gate",
            "cuda_ms", "wall_ms", "input_sha", "target_sha",
        ])
        self._steps_writer = writer

    def on_train_ready(self, model_base, optimizer, criterion):
        self.optimizer = optimizer
        self.criterion = criterion

    def on_fit_start(self, model_base):
        self.model_base = model_base
        core = model_base._core_model()
        self.fingerprint = module_fingerprint(core.vision_model)
        self.fit_t0 = time.perf_counter()
        write_json(os.path.join(self.cfg.out_dir, "env_snapshot.json"), {
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "cudnn_version": torch.backends.cudnn.version(),
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "cudnn_enabled": torch.backends.cudnn.enabled,
            "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
            "pythonhashseed_env": os.environ.get("PYTHONHASHSEED"),
            "use_deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "gpu_name": (
                torch.cuda.get_device_name(0)
                if torch.cuda.is_available()
                else "cpu"
            ),
            "gpu_capability": (
                str(torch.cuda.get_device_capability(0))
                if torch.cuda.is_available()
                else None
            ),
            "config": {k: v for k, v in vars(model_base.config).items()},
            "train_info": self.train_info,
            "trainable_params": [
                [n, int(p.numel())]
                for n, p in core.named_parameters()
                if p.requires_grad
            ],
            "trainable_checksum_initial": trainable_checksum(core),
        })

    # ---- per-step -----------------------------------------------------
    def on_step_begin(self, model_base, input, target, input_mark, target_mark):
        self._last_train_batch = (input, target, input_mark, target_mark)
        self.cur_batch = self._last_train_batch if self.cfg.input_hash else None
        self.timer.begin()

    def on_loss(self, model_base, loss, additional_loss):
        self.loss_cache = (loss, additional_loss)

    def note_loss_components(self, channel_loss, tp_loss, gate_loss, gate_value,
                             shapes=None):
        finite = bool(
            torch.isfinite(channel_loss)
            and torch.isfinite(tp_loss)
            and torch.isfinite(gate_loss)
        )
        self.loss_cache_parts = {
            "channel": float(channel_loss.detach().item()),
            "tp": float(tp_loss.detach().item()),
            "gate": float(gate_loss.detach().item()),
            "gate_value": float(gate_value.detach().item()),
            "shapes": shapes or {},
            "finite": finite,
        }

    def on_step_end(self, model_base, optimizer, epoch, i):
        self.global_step += 1
        step = self.global_step
        cuda_ms, wall_ms = self.timer.end()

        loss_t, addl_t = self.loss_cache
        total = float((loss_t + addl_t).detach().item())
        parts = self.loss_cache_parts or {}

        lrs = {}
        for g in optimizer.param_groups:
            lrs[g.get("parameter_group", "default")] = g["lr"]

        input_sha = target_sha = ""
        if self.cur_batch is not None and self.cfg.input_hash:
            input_sha = batch_sha256(self.cur_batch[0])
            target_sha = batch_sha256(self.cur_batch[1])

        row = {
            "step": step,
            "epoch": epoch,
            "batch_idx": i,
            "loss_total": total,
            "loss_channel": parts.get("channel"),
            "loss_tp": parts.get("tp"),
            "loss_gate": parts.get("gate"),
            "gate_value": parts.get("gate_value"),
            "lr_adapters": lrs.get("adapters"),
            "lr_gate": lrs.get("gate"),
            "cuda_ms": cuda_ms,
            "wall_ms": wall_ms,
            "input_sha": input_sha,
            "target_sha": target_sha,
        }
        self.rows.append(row)
        if self.cfg.param_sha:
            self._param_sha[step] = trainable_checksum(model_base._core_model())
        if step % self.cfg.log_every == 0:
            self._steps_writer.writerow([
                row[k] for k in [
                    "step", "epoch", "batch_idx", "loss_total", "loss_channel",
                    "loss_tp", "loss_gate", "gate_value", "lr_adapters",
                    "lr_gate", "cuda_ms", "wall_ms", "input_sha", "target_sha",
                ]
            ])
            self._steps_csv.flush()
        self.loss_cache = None
        self.loss_cache_parts = None
        self.cur_batch = None

        if step in self.cfg.val_at:
            self._run_val(model_base)

        if step in self.cfg.audit_steps and self._last_train_batch is not None:
            if self._first_audited is None:
                self._first_audited = step
            rec = run_step_audit(
                model_base, self._last_train_batch, step, self.cfg.out_dir,
                {"_first_audited_step": self._first_audited,
                 "_param_states": self._param_states},
            )
            self.audit_records.append(rec)
            write_json(
                os.path.join(self.cfg.out_dir, "audit_steps.json"),
                self.audit_records,
            )
        self._last_train_batch = None

        if self.cfg.max_steps and step >= self.cfg.max_steps:
            self.stop = True
        return self.stop

    def should_stop(self):
        return self.stop

    # ---- validation ----------------------------------------------------
    def _rng_guarded(self, fn):
        rng = save_rng()
        try:
            return fn()
        finally:
            restore_rng(rng)

    def timed_validate(self, model_base, valid_data_loader, series_dim,
                       criterion, tag="epoch_end"):
        def run():
            reset_peak_mem()
            t0 = time.perf_counter()
            val = model_base.validate(valid_data_loader, series_dim, criterion)
            elapsed = time.perf_counter() - t0
            mem = read_peak_mem()
            return val, elapsed, mem

        was_training = model_base.model.training
        val, elapsed, mem = self._rng_guarded(run)
        if was_training:
            model_base.model.train()
        self.val_records.append({
            "tag": tag,
            "step": self.global_step,
            "val_mse_fused": float(val),
            "elapsed_s": elapsed,
            **mem,
        })
        self._write_val_csv()
        return val

    def _run_val(self, model_base):
        step = self.global_step
        loader = self.valid_loader
        sd = self.series_dim
        if loader is None:
            return

        reset_peak_mem()
        rng = save_rng()
        t0 = time.perf_counter()
        val = model_base.validate(loader, sd, self.criterion)
        elapsed = time.perf_counter() - t0
        mem = read_peak_mem()
        restore_rng(rng)
        model_base.model.train()

        rec = {
            "tag": f"step{step}",
            "step": step,
            "val_mse_fused": float(val),
            "elapsed_s": elapsed,
            **mem,
        }

        if self.cfg.dump_preds:
            rng = save_rng()
            preds, targets = self._dump_val_predictions(model_base, loader, sd)
            restore_rng(rng)
            model_base.model.train()
        else:
            preds = targets = None

        if self.cfg.save_full:
            entry = save_val_point(
                self.cfg.out_dir, model_base, step, val,
                preds=preds, targets=targets,
                window_border=None,
            )
            rec["saved"] = {
                "full_state_path": entry["full_state_path"],
                "preds_path": entry.get("preds_path"),
                "point_meta_path": os.path.join(
                    self.cfg.out_dir, f"val_point_step{step}.json"
                ),
                "scaler_mean": entry["scaler_mean"],
                "scaler_scale": entry["scaler_scale"],
                "preds_shape": entry.get("preds_shape"),
            }

        if self.best is None or val < self.best["val_mse_fused"]:
            self.best = rec

        self.val_records.append(rec)
        self._write_val_csv()

    def _dump_val_predictions(self, model_base, loader, sd):
        """Verification-only pass mirroring MUSE.validate to collect outputs.

        Rebuilds the loader exactly like MUSE.validate (shuffle=False,
        drop_last=False) so dumped windows match the metric pass 1:1.
        """
        from torch.utils.data import DataLoader

        import numpy as np

        config = model_base.config
        horizon = config.horizon
        vloader = DataLoader(
            loader.dataset,
            batch_size=loader.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            drop_last=False,
        )
        device = next(model_base.model.parameters()).device
        model_base.model.eval()
        preds, tgts = [], []
        with torch.no_grad():
            for input, target, input_mark, target_mark in vloader:
                input, target = input.to(device), target.to(device)
                output = model_base.model(
                    input,
                    fp64=config.fp64,
                    use_variable_chunk=config.use_variable_chunk,
                )
                target = target[:, -horizon:, :sd]
                output = output[:, -horizon:, :sd]
                output, target = model_base._post_process(output, target)
                preds.append(output.cpu().numpy())
                tgts.append(target.cpu().numpy())
        model_base.model.train()
        return np.concatenate(preds), np.concatenate(tgts)

    def _write_val_csv(self):
        path = os.path.join(self.cfg.out_dir, "val.csv")
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "tag", "step", "val_mse_fused", "elapsed_s",
                "peak_alloc_mb", "peak_reserved_mb", "current_alloc_mb",
            ])
            for r in self.val_records:
                writer.writerow([
                    r["tag"], r["step"], r["val_mse_fused"], r["elapsed_s"],
                    r.get("peak_alloc_mb"), r.get("peak_reserved_mb"),
                    r.get("current_alloc_mb"),
                ])

    # ---- fit end --------------------------------------------------------
    def on_fit_end(self, model_base):
        try:
            self._steps_csv.close()
        except Exception:
            pass
        core = model_base._core_model()
        after = module_fingerprint(core.vision_model)
        fp_report = fingerprint_diff(self.fingerprint, after)
        elapsed_fit = time.perf_counter() - self.fit_t0

        warmup = self.cfg.warmup_steps
        stats = {}
        for key in ("cuda_ms", "wall_ms"):
            stats[key + "_steps11_100"] = step_stats(
                self.rows, key, warmup + 1, self.global_step
            )
            stats[key + "_all"] = step_stats(self.rows, key, 1, self.global_step)

        decomp = [
            abs((r["loss_channel"] + r["loss_tp"] + r["loss_gate"]) - r["loss_total"])
            for r in self.rows
            if r.get("loss_channel") is not None
        ]
        summary = {
            "global_step": self.global_step,
            "stop_reason": "max_steps" if self.stop else "natural_end",
            "fit_wall_s": elapsed_fit,
            "frozen_fingerprint": fp_report,
            "train_info": self.train_info,
            "timing": stats,
            "warmup_steps": warmup,
            "val_records": self.val_records,
            "best": self.best,
            "loss_decomposition_max_absdiff": max(decomp) if decomp else None,
            "finite_losses": all(
                r["loss_total"] == r["loss_total"]
                and abs(r["loss_total"]) != float("inf")
                for r in self.rows
            ),
            "param_grad_states": {
                name: sorted(states)
                for name, states in sorted(self._param_states.items())
            },
            "audit_steps": [r["step"] for r in self.audit_records],
            "param_sha": {str(k): v for k, v in sorted(self._param_sha.items())},
            "config_effective": dict(vars(model_base.config)),
        }
        write_json(os.path.join(self.cfg.out_dir, "summary.json"), summary)

        manifest = {
            "out_dir": self.cfg.out_dir,
            "global_step": self.global_step,
            "best": self.best,
            "val_records": self.val_records,
            "train_info": self.train_info,
            "muse_patch_diff_sha": os.environ.get("MUSE_REPRO_DIFF_SHA"),
            "data_sha": os.environ.get("MUSE_REPRO_DATA_SHA"),
            "meta_sha": os.environ.get("MUSE_REPRO_META_SHA"),
            "mae_ckpt_sha": os.environ.get("MUSE_REPRO_MAE_SHA"),
            "mae_ckpt_path": os.environ.get("MUSE_REPRO_MAE_PATH"),
            "torch": torch.__version__,
        }
        write_manifest(self.cfg.out_dir, manifest)

        if self.stop:
            raise SystemExit(
                f"[repro] controlled stop after step {self.global_step} "
                f"(MUSE_REPRO_MAX_STEPS={self.cfg.max_steps}); bundle saved to "
                f"{self.cfg.out_dir}"
            )
