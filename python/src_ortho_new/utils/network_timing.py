import contextlib
import time

import numpy as np
import torch


_enabled = False
_data_enabled = False
_sections = {}
_generated_frames = []
_codebook_steps = []


def set_enabled(enabled=True):
    global _enabled
    _enabled = bool(enabled)


def set_data_enabled(enabled=True):
    global _data_enabled
    _data_enabled = bool(enabled)


def is_enabled():
    return _enabled


def is_data_enabled():
    return _data_enabled


def reset():
    _sections.clear()
    _generated_frames.clear()
    _codebook_steps.clear()


def _sync_cuda():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


@contextlib.contextmanager
def section(name):
    if not _enabled:
        yield
        return

    _sync_cuda()
    start = time.perf_counter()
    try:
        yield
    finally:
        _sync_cuda()
        elapsed = time.perf_counter() - start
        _sections.setdefault(name, []).append(elapsed)


def start_data_timer():
    if not _data_enabled:
        return None
    _sync_cuda()
    return time.perf_counter()


def end_data_timer(name, start):
    if start is None or not _data_enabled:
        return
    _sync_cuda()
    elapsed = time.perf_counter() - start
    _sections.setdefault(name, []).append(elapsed)


def record_generated_frames(num_frames):
    if _enabled:
        _generated_frames.append(int(num_frames))


def record_codebook_steps(num_steps):
    if _enabled:
        _codebook_steps.append(int(num_steps))


def _stats(values):
    arr = np.asarray(values, dtype=np.float64)
    return {
        "count": int(arr.size),
        "total_s": float(arr.sum()),
        "mean_s": float(arr.mean()) if arr.size else 0.0,
        "std_s": float(arr.std()) if arr.size else 0.0,
        "min_s": float(arr.min()) if arr.size else 0.0,
        "max_s": float(arr.max()) if arr.size else 0.0,
    }


def summary():
    sections = {name: _stats(values) for name, values in _sections.items()}
    total_generated_frames = int(sum(_generated_frames))
    total_codebook_steps = int(sum(_codebook_steps))
    total_timed_s = float(sum(stats["total_s"] for stats in sections.values()))
    prior_decoder_s = float(
        sections.get("prior", {}).get("total_s", 0.0)
        + sections.get("decoder", {}).get("total_s", 0.0)
    )

    return {
        "sections": sections,
        "total_generated_frames": total_generated_frames,
        "total_codebook_steps": total_codebook_steps,
        "total_timed_s": total_timed_s,
        "prior_decoder_s": prior_decoder_s,
        "network_fps": (
            total_generated_frames / total_timed_s
            if total_timed_s > 0.0 else 0.0
        ),
        "prior_decoder_fps": (
            total_generated_frames / prior_decoder_s
            if prior_decoder_s > 0.0 else 0.0
        ),
        "codebook_steps_per_s": (
            total_codebook_steps / sections.get("prior", {}).get("total_s", 0.0)
            if sections.get("prior", {}).get("total_s", 0.0) > 0.0 else 0.0
        ),
    }


def print_summary(animation_fps=30.0):
    stats = summary()
    print("\nNetwork timing report")
    print("===================================================")
    print(f"Generated motion frames: {stats['total_generated_frames']}")
    print(f"Generated codebook steps: {stats['total_codebook_steps']}")
    print(f"Prior + decoder time: {stats['prior_decoder_s']:.6f} s")
    print(f"Prior + decoder FPS: {stats['prior_decoder_fps']:.2f} frames/s")
    if animation_fps > 0:
        print(f"Real-time factor @ {animation_fps:.1f} FPS: {stats['prior_decoder_fps'] / animation_fps:.2f}x")
    print(f"All timed sections total: {stats['total_timed_s']:.6f} s")
    print(f"All timed sections FPS: {stats['network_fps']:.2f} frames/s")
    print(f"Prior throughput: {stats['codebook_steps_per_s']:.2f} codebook steps/s")

    for name, section_stats in stats["sections"].items():
        print(
            f"{name}: total={section_stats['total_s']:.6f}s, "
            f"mean={section_stats['mean_s'] * 1000.0:.3f}ms, "
            f"std={section_stats['std_s'] * 1000.0:.3f}ms, "
            f"calls={section_stats['count']}"
        )
