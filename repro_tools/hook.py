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

import numpy as np
import torch

from .audit import run_step_audit
from .bundle import save_best_point, save_val_point, write_manifest
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
        self.best_records = []
        self.epoch_records = []
        self.cur_epoch = -1
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
        self.valid_start_abs = None
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
        # absolute row of valid_data's start in the original series:
        # train rows = n_train + seq_len + horizon - 1, and valid starts
        # seq_len rows earlier (history overlap of train/valid split)
        horizon = model_base.config.horizon
        seq_len = model_base.config.seq_len
        self.valid_start_abs = n_train + horizon - 1
        self.train_info = {
            "train_windows": n_train,
            "train_batches_full": n_train // bs,
            "train_drop_last": train_loader.drop_last,
            "train_tail_samples": n_train % bs,
            "train_rows": n_train + seq_len + horizon - 1,
            "valid_rows": (
                None
                if valid_loader is None
                else len(valid_loader.dataset) + seq_len + horizon - 1
            ),
            "valid_data_start_abs": self.valid_start_abs,
            "valid_windows": None if valid_loader is None else len(valid_loader.dataset),
            "valid_batches": None if valid_loader is None else len(valid_loader),
            "series_dim": series_dim,
            "seq_len": seq_len,
            "horizon": horizon,
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
        if self.cfg.force_deterministic:
            # comparison-runs only: not part of the official perf configuration
            torch.use_deterministic_algorithms(True)
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
        self.cur_epoch = epoch
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
        """Official epoch-end validate: official RNG behaviour preserved.

        No RNG save/restore and no train-flag restore here — the official
        loop consumes whatever RNG this validate consumes and leaves the
        model in eval until the next epoch's .train(). The training-segment
        peak memory is read BEFORE reset clears it.
        """
        train_peak = read_peak_mem()
        reset_peak_mem()
        t0 = time.perf_counter()
        val = model_base.validate(valid_data_loader, series_dim, criterion)
        elapsed = time.perf_counter() - t0
        mem = read_peak_mem()
        rec = {
            "tag": tag,
            "epoch": self.cur_epoch,
            "step": self.global_step,
            "val_mse_fused": float(val),
            "elapsed_s": elapsed,
            "rng_guarded": False,
            "train_peak_alloc_mb": train_peak.get("peak_alloc_mb"),
            "train_peak_reserved_mb": train_peak.get("peak_reserved_mb"),
            **mem,
        }
        self.val_records.append(rec)
        self._write_val_csv()
        return val

    def on_best(self, model_base, valid_loss):
        """Official improved=True: persist best checkpoint bundle.

        The prediction dump is an EXTRA pass and stays RNG-guarded; the
        official best deepcopy (model_base.check_point) is what gets saved.
        Save/dump elapsed times are recorded separately from the official
        validation time.
        """
        epoch = self.cur_epoch
        preds = targets = None
        border_meta = None
        dump_elapsed = 0.0
        if self.cfg.dump_preds and self.valid_loader is not None:
            rng = save_rng()
            try:
                t0 = time.perf_counter()
                preds, targets = self._dump_val_predictions(
                    model_base, self.valid_loader, self.series_dim
                )
                border_meta = self._valid_window_borders(self.valid_loader)
                dump_elapsed = time.perf_counter() - t0
            finally:
                restore_rng(rng)
                # official state at this point is eval (validate just ran);
                # keep it — the next epoch's .train() is official behaviour
                model_base.model.eval()

        t1 = time.perf_counter()
        borders_arr = (
            border_meta["pred_start"] if border_meta is not None else None
        )
        entry = save_best_point(
            self.cfg.out_dir,
            model_base.check_point,
            model_base,
            epoch=epoch,
            step=self.global_step,
            val_mse=float(valid_loss),
            val_history=self.val_records,
            preds=preds,
            targets=targets,
            window_border=borders_arr,
            border_meta=border_meta,
        )
        save_elapsed = time.perf_counter() - t1

        rec = {
            "tag": entry["tag"],
            "epoch": epoch,
            "step": self.global_step,
            "val_mse_fused": float(valid_loss),
            "dump_elapsed_s": dump_elapsed,
            "save_elapsed_s": save_elapsed,
            "rng_guarded_dump": True,
            "saved": {
                "full_state_path": entry["full_state_path"],
                "preds_path": entry.get("preds_path"),
                "point_meta_path": os.path.join(
                    self.cfg.out_dir, f"best_point_epoch{epoch}.json"
                ),
                "scaler_mean": entry["scaler_mean"],
                "scaler_scale": entry["scaler_scale"],
                "preds_shape": entry.get("preds_shape"),
            },
        }
        self.best_records.append(rec)
        self.best = rec
        if self.val_records:
            self.val_records[-1]["improved"] = True
            self.val_records[-1]["best_saved"] = entry["tag"]
            self.val_records[-1]["dump_elapsed_s"] = dump_elapsed
            self.val_records[-1]["save_elapsed_s"] = save_elapsed
        self._write_val_csv()
        self._write_best_csv()
        self._write_manifest()
        return True

    def _valid_window_borders(self, loader):
        """Absolute window identity of the valid loader windows.

        Window index i of DatasetForTransformer: input = rows [i, i+seq_len)
        of valid_data, prediction target = the horizon rows starting at
        valid_data_start_abs + i + seq_len. The absolute offset comes from
        the split formula (n_train_windows + horizon - 1), NOT from the
        DataFrame index (which is a DatetimeIndex in this pipeline).
        """
        ds = loader.dataset
        vdf = ds.dataset
        start_abs = self.valid_start_abs
        hist = int(ds.history_length)
        pred_start = start_abs + np.arange(len(ds), dtype=np.int64) + hist
        return {
            "pred_start": pred_start,
            "valid_data_start_abs": start_abs,
            "valid_data_end_abs_exclusive": start_abs + len(vdf),
            "history_length": hist,
            "prediction_length": int(ds.prediction_length),
            "label_length": int(ds.label_length),
            "n_windows": int(len(ds)),
            "first_pred_row_ts": str(vdf.index[hist]),
            "last_pred_row_ts": str(vdf.index[hist + len(ds) - 1]),
        }

    # ---- test ----------------------------------------------------------
    def on_test_ready(self, all_predicts, targets, index_list, eval_scaler,
                      mean_metrics, total_inference_time, fit_time):
        """Official batch-forecast test finished: persist preds/targets.

        all_predicts: [N, horizon, C] post-inverse predictions;
        index_list: absolute split index per window (test window = rows
        [index, index+horizon) of the full series).
        """
        t0 = time.perf_counter()
        preds = np.asarray(all_predicts)
        tgts = np.asarray(targets)
        borders = np.asarray(index_list, dtype=np.int64)
        npz_path = os.path.join(self.cfg.out_dir, "test_preds.npz")
        np.savez_compressed(
            npz_path, preds=preds, targets=tgts, window_border=borders
        )
        dump_elapsed = time.perf_counter() - t0
        eval_scaler_stats = None
        if eval_scaler is not None and hasattr(eval_scaler, "mean_"):
            eval_scaler_stats = {
                "mean": np.asarray(eval_scaler.mean_).tolist(),
                "scale": np.asarray(eval_scaler.scale_).tolist(),
            }
        report = {
            "preds_shape": list(preds.shape),
            "targets_shape": list(tgts.shape),
            "n_windows": int(len(borders)),
            "window_border_first": int(borders[0]),
            "window_border_last": int(borders[-1]),
            "preds_path": npz_path,
            "eval_scaler": eval_scaler_stats,
            "mean_metrics": list(mean_metrics),
            "official_total_inference_s": float(total_inference_time),
            "official_fit_s": float(fit_time),
            "dump_elapsed_s": dump_elapsed,
        }
        write_json(os.path.join(self.cfg.out_dir, "test_report.json"), report)
        return report

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
            border_meta = self._valid_window_borders(loader)
        else:
            preds = targets = None
            border_meta = None

        if self.cfg.save_full:
            entry = save_val_point(
                self.cfg.out_dir, model_base, step, val,
                preds=preds, targets=targets,
                window_border=(
                    border_meta["pred_start"] if border_meta is not None else None
                ),
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
        cols = [
            "tag", "epoch", "step", "val_mse_fused", "elapsed_s",
            "rng_guarded", "improved", "best_saved",
            "train_peak_alloc_mb", "train_peak_reserved_mb",
            "peak_alloc_mb", "peak_reserved_mb", "current_alloc_mb",
            "dump_elapsed_s", "save_elapsed_s",
        ]
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(cols)
            for r in self.val_records:
                writer.writerow([r.get(c) for c in cols])

    def _write_best_csv(self):
        path = os.path.join(self.cfg.out_dir, "best.csv")
        cols = [
            "tag", "epoch", "step", "val_mse_fused",
            "dump_elapsed_s", "save_elapsed_s",
            "full_state_path", "preds_path",
        ]
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(cols)
            for r in self.best_records:
                writer.writerow([
                    r.get("tag"), r.get("epoch"), r.get("step"),
                    r.get("val_mse_fused"),
                    r.get("dump_elapsed_s"), r.get("save_elapsed_s"),
                    r["saved"]["full_state_path"], r["saved"]["preds_path"],
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
            stats[f"{key}_steps{warmup + 1}_{self.global_step}"] = step_stats(
                self.rows, key, warmup + 1, self.global_step
            )
            stats[key + "_all"] = step_stats(self.rows, key, 1, self.global_step)

        decomp = [
            abs((r["loss_channel"] + r["loss_tp"] + r["loss_gate"]) - r["loss_total"])
            for r in self.rows
            if r.get("loss_channel") is not None
        ]
        val_time_total = sum(
            r.get("elapsed_s", 0.0) for r in self.val_records
            if r.get("tag") == "epoch_end"
        )
        best_dump_time = sum(r.get("dump_elapsed_s", 0.0) for r in self.best_records)
        best_save_time = sum(r.get("save_elapsed_s", 0.0) for r in self.best_records)
        summary = {
            "global_step": self.global_step,
            "stop_reason": "max_steps" if self.stop else "natural_end",
            "fit_wall_s": elapsed_fit,
            "frozen_fingerprint": fp_report,
            "train_info": self.train_info,
            "timing": stats,
            "warmup_steps": warmup,
            "official_epoch_validate_time_s": val_time_total,
            "best_dump_time_s": best_dump_time,
            "best_save_time_s": best_save_time,
            "timing_semantics": {
                "epoch_end_validate_rng_guarded": False,
                "best_dump_rng_guarded": True,
                "train_peak_read_before_val_reset": True,
                "step_wall_after_cuda_sync": True,
            },
            "val_records": self.val_records,
            "best": self.best,
            "best_records": self.best_records,
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
        self._write_manifest()

        if self.stop:
            raise SystemExit(
                f"[repro] controlled stop after step {self.global_step} "
                f"(MUSE_REPRO_MAX_STEPS={self.cfg.max_steps}); bundle saved to "
                f"{self.cfg.out_dir}"
            )

    def _write_manifest(self):
        manifest = {
            "out_dir": self.cfg.out_dir,
            "global_step": self.global_step,
            "cur_epoch": self.cur_epoch,
            "best": self.best,
            "best_records": self.best_records,
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
