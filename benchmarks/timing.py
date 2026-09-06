"""Shared measurement primitives for every benchmark in this directory.

Why this exists rather than scattering `time.time()` calls through the
scripts: the project already had *macro* timings (ADR-006's 1.70s BM25 index
build, ADR-008's 264.4s encode) captured ad hoc inside experiment scripts.
What was missing was a per-stage, per-query substrate that reports the same
shape of number for every stage and every dataset, so a MIND table and an
EB-NeRD table are directly comparable and neither can quietly use a
different definition of "throughput". See ADR-014.

Three deliberate choices:

- **`time.perf_counter`, not `time.time`.** `perf_counter` is monotonic and
  is the highest-resolution clock Python exposes; `time.time` is wall-clock
  and can jump (NTP). At the microsecond stage times measured here that
  matters.
- **Wall time, not CPU time.** The question being asked is "how long does a
  query take", which includes BLAS threads and memory stalls. `process_time`
  would over-count multi-threaded stages (numpy/OpenBLAS matvec, LightGBM
  predict) by summing across cores and under-count nothing useful.
- **A warmup pass, always.** First-call costs (lazy imports, BLAS thread-pool
  spin-up, page faults on a freshly-mmapped embedding matrix, LightGBM's
  first predict) are real but one-time. Folding them into a per-query mean
  would misattribute a fixed cost to a marginal one. Warmup iterations are
  recorded in the output so their exclusion is auditable, never silent.

Percentiles are reported alongside the mean because per-query latency here is
strongly right-skewed (a user with a 1,459-article history costs far more to
tokenize than the median user); a mean alone would hide that.
"""
from __future__ import annotations

import json
import platform
import subprocess
import time
from contextlib import contextmanager
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Iterator

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = Path(__file__).resolve().parent / "results"


@dataclass
class StageStats:
    """Per-stage timing summary. All times in milliseconds unless named otherwise."""

    stage: str
    n_calls: int
    total_s: float
    mean_ms: float
    p50_ms: float
    p90_ms: float
    p99_ms: float
    max_ms: float
    throughput_per_s: float  # calls/second *for this stage in isolation*
    pct_of_total: float = 0.0  # filled in by StageTimer.summary()


class StageTimer:
    """Accumulates per-call wall times per named stage.

    Usage:
        timer = StageTimer()
        with timer("bm25_score_all"):
            scores = score_all(index, tokens)

    Samples are kept as a plain list of floats (not a running mean) because
    percentiles need the distribution. At the sample counts used here
    (thousands, not millions) the memory cost is negligible and the honesty
    gain — being able to report p99 and max, not just a mean that hides the
    tail — is the entire point.
    """

    def __init__(self) -> None:
        self._samples: dict[str, list[float]] = {}
        self._order: list[str] = []

    @contextmanager
    def __call__(self, stage: str) -> Iterator[None]:
        if stage not in self._samples:
            self._samples[stage] = []
            self._order.append(stage)
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self._samples[stage].append(time.perf_counter() - t0)

    def record(self, stage: str, seconds: float) -> None:
        """Record a stage duration measured elsewhere (e.g. inside a cProfile
        run, or a one-shot setup cost timed by the caller)."""
        if stage not in self._samples:
            self._samples[stage] = []
            self._order.append(stage)
        self._samples[stage].append(seconds)

    def drop(self, stage: str) -> None:
        """Discard a stage entirely — used to throw away warmup measurements."""
        self._samples.pop(stage, None)
        if stage in self._order:
            self._order.remove(stage)

    def reset(self) -> None:
        self._samples.clear()
        self._order.clear()

    def total_s(self, stages: list[str] | None = None) -> float:
        keys = stages if stages is not None else self._order
        return float(sum(sum(self._samples.get(k, [])) for k in keys))

    def summary(self, denominator_stages: list[str] | None = None) -> list[StageStats]:
        """One `StageStats` per stage, with `pct_of_total` computed against
        `denominator_stages` (default: every recorded stage).

        The denominator is explicit because "% of total per-query time" is
        only meaningful over stages that actually compose one query path.
        Mixing a one-time index build into that denominator would inflate it
        and understate every per-query stage — the exact kind of quietly-wrong
        percentage this parameter exists to prevent.
        """
        denom = self.total_s(denominator_stages)
        out: list[StageStats] = []
        for stage in self._order:
            samples = self._samples[stage]
            if not samples:
                continue
            arr = np.asarray(samples, dtype=np.float64)
            total = float(arr.sum())
            out.append(
                StageStats(
                    stage=stage,
                    n_calls=len(arr),
                    total_s=total,
                    mean_ms=float(arr.mean() * 1e3),
                    p50_ms=float(np.percentile(arr, 50) * 1e3),
                    p90_ms=float(np.percentile(arr, 90) * 1e3),
                    p99_ms=float(np.percentile(arr, 99) * 1e3),
                    max_ms=float(arr.max() * 1e3),
                    throughput_per_s=(len(arr) / total) if total > 0 else float("inf"),
                    pct_of_total=(100.0 * total / denom) if denom > 0 else 0.0,
                )
            )
        return out


def hardware_spec() -> dict:
    """Machine facts, captured at run time rather than hardcoded.

    Every benchmark JSON carries this block, because the whole point of the
    per-hardware comparison is that the same script produces numbers labelled
    with the machine that produced them. A results file without a spec block
    cannot be safely compared against another one.
    """
    spec: dict = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor() or platform.machine(),
        "python": platform.python_version(),
    }

    try:  # macOS
        spec["cpu_model"] = subprocess.check_output(
            ["sysctl", "-n", "machdep.cpu.brand_string"], text=True, stderr=subprocess.DEVNULL
        ).strip()
        spec["hw_model"] = subprocess.check_output(
            ["sysctl", "-n", "hw.model"], text=True, stderr=subprocess.DEVNULL
        ).strip()
        # GiB, not GB. This machine is referred to throughout the project as
        # "the 8GB machine"; hw.memsize is 8,589,934,592 bytes, which is 8.0
        # GiB but 8.6 GB. Reporting the decimal figure would make every
        # results file disagree with PROJECT_STATE's own risk register.
        spec["ram_gb"] = round(
            int(subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True)) / 2**30, 1
        )
        spec["physical_cores"] = int(
            subprocess.check_output(["sysctl", "-n", "hw.physicalcpu"], text=True)
        )
    except Exception:
        pass

    if "cpu_model" not in spec:  # Linux (Kaggle / Ada)
        try:
            with open("/proc/cpuinfo") as fh:
                for line in fh:
                    if line.startswith("model name"):
                        spec["cpu_model"] = line.split(":", 1)[1].strip()
                        break
            with open("/proc/meminfo") as fh:
                for line in fh:
                    if line.startswith("MemTotal"):  # kB -> GiB, same convention as macOS above
                        spec["ram_gb"] = round(int(line.split()[1]) / 2**20, 1)
                        break
            import os

            spec["physical_cores"] = os.cpu_count()
        except Exception:
            pass

    try:
        import torch

        spec["torch"] = torch.__version__
        spec["cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            spec["gpu_name"] = torch.cuda.get_device_name(0)
            spec["gpu_memory_gb"] = round(
                torch.cuda.get_device_properties(0).total_memory / 1e9, 1
            )
        spec["mps_available"] = bool(
            getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
        )
    except Exception:
        spec["torch"] = None

    try:
        import numpy as _np

        spec["numpy"] = _np.__version__
    except Exception:
        pass

    return spec


def _short_cpu(spec: dict) -> str:
    cpu = spec.get("hw_model") or spec.get("cpu_model") or spec.get("machine", "unknown")
    # Trim the marketing noise Intel puts in `model name` so the label stays a
    # label ("Xeon E5-2640 v4", not "Intel(R) Xeon(R) CPU E5-2640 v4 @ 2.40GHz").
    for junk in ("Intel(R) ", "(R)", "(TM)", " CPU"):
        cpu = cpu.replace(junk, "")
    return cpu.split("@")[0].strip()


def hardware_label(spec: dict | None = None) -> str:
    """Short, stable name for the MACHINE — the key every per-hardware table in
    ADR-014 and PERFORMANCE_LOG.md is grouped on.

    Names the CPU *and* the GPU when one is present, rather than the GPU alone.
    The earlier GPU-only version mislabelled real runs: a `--device cpu` profile
    executed on an Ada GPU node was recorded as "NVIDIA GeForce RTX 2080 Ti",
    which reads as a GPU measurement and is exactly backwards. It also labelled
    a job that requested no GPU by whatever card happened to be visible on the
    node.

    This identifies the machine; which device the work actually ran on is a
    separate field in each results file (`config.device` / `setup.nrms_device`),
    because they are genuinely different facts and conflating them produced a
    wrong label once already.
    """
    spec = spec or hardware_spec()
    cpu = _short_cpu(spec)
    ram = spec.get("ram_gb")
    label = f"{cpu}{f' / {ram:.0f}GiB' if ram else ''}"
    gpu = spec.get("gpu_name")
    if gpu:
        label += f" + {gpu.replace('NVIDIA ', '').replace('Tesla ', '').strip()}"
    return label


def peak_rss_gb() -> float:
    """Peak resident set size of this process, in GB.

    `resource.getrusage` reports `ru_maxrss` in *bytes* on macOS and in
    *kilobytes* on Linux — an inconsistency that silently produces a
    1000x-wrong memory number if unhandled, which matters here because
    CLAUDE.md's Memory Estimation clause makes these figures load-bearing for
    go/no-go decisions on an 8GB machine.
    """
    import resource
    import sys

    raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return raw / 1e9 if sys.platform == "darwin" else raw / 1e6


def git_rev() -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "--short", "HEAD"],
            text=True, stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unknown"


def git_dirty() -> bool:
    try:
        out = subprocess.check_output(
            ["git", "-C", str(REPO_ROOT), "status", "--porcelain"],
            text=True, stderr=subprocess.DEVNULL,
        )
        return bool(out.strip())
    except Exception:
        return False


def write_results(name: str, payload: dict, out_dir: Path | None = None) -> Path:
    """Write a timestamped JSON result, mirroring how `experiments/` already
    captures config+results for accuracy work (ADR-014's "same shape as the
    accuracy trail" requirement)."""
    out_dir = out_dir or RESULTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y-%m-%dT%H-%M-%S")
    path = out_dir / f"{stamp}_{name}.json"
    path.write_text(json.dumps(payload, indent=2, default=str))
    return path


def stats_to_rows(stats: list[StageStats]) -> list[dict]:
    return [asdict(s) for s in stats]


def markdown_table(stats: list[StageStats], unit: str = "query") -> str:
    """Render a stage table the way ADR-014 and PERFORMANCE_LOG.md present it."""
    head = (
        f"| Stage | Calls | Mean (ms) | p50 | p90 | p99 | Total (s) | % of total | "
        f"Throughput ({unit}/s) |\n"
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|\n"
    )
    body = "".join(
        f"| `{s.stage}` | {s.n_calls:,} | {s.mean_ms:.3f} | {s.p50_ms:.3f} | "
        f"{s.p90_ms:.3f} | {s.p99_ms:.3f} | {s.total_s:.2f} | {s.pct_of_total:.1f}% | "
        f"{s.throughput_per_s:,.0f} |\n"
        for s in stats
    )
    return head + body
