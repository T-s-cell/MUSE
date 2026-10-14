"""CUDA-event step timing and peak-memory helpers."""

import time

import torch


class StepTimer:
    def __init__(self):
        self.start_event = None
        self.end_event = None
        self.wall_t0 = None
        if torch.cuda.is_available():
            self.start_event = torch.cuda.Event(enable_timing=True)
            self.end_event = torch.cuda.Event(enable_timing=True)

    def begin(self):
        if self.start_event is not None:
            self.start_event.record()
        self.wall_t0 = time.perf_counter()

    def end(self):
        cuda_ms = None
        if self.end_event is not None:
            self.end_event.record()
            torch.cuda.synchronize()
            cuda_ms = self.start_event.elapsed_time(self.end_event)
        elif torch.cuda.is_available():
            torch.cuda.synchronize()
        wall_ms = (time.perf_counter() - self.wall_t0) * 1000.0
        return cuda_ms, wall_ms


def reset_peak_mem():
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()


def read_peak_mem():
    if not torch.cuda.is_available():
        return {}
    torch.cuda.synchronize()
    return {
        "peak_alloc_mb": torch.cuda.max_memory_allocated() / 1024 ** 2,
        "peak_reserved_mb": torch.cuda.max_memory_reserved() / 1024 ** 2,
        "current_alloc_mb": torch.cuda.memory_allocated() / 1024 ** 2,
    }


def step_stats(rows, key, lo, hi):
    vals = [r[key] for r in rows if lo <= r["step"] <= hi and r.get(key) is not None]
    n = len(vals)
    if n == 0:
        return {"n": 0}
    mean = sum(vals) / n
    srt = sorted(vals)
    median = srt[n // 2] if n % 2 else 0.5 * (srt[n // 2 - 1] + srt[n // 2])
    p95 = srt[min(n - 1, int(round(0.95 * (n - 1))))]
    var = sum((v - mean) ** 2 for v in vals) / n
    cv = (var ** 0.5) / mean if mean else float("nan")
    return {
        "n": n,
        "mean": mean,
        "median": median,
        "p95": p95,
        "min": srt[0],
        "max": srt[-1],
        "cv": cv,
    }
