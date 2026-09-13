#!/usr/bin/env python3
"""A2 Q4.3: cost per 1,000 queries at a target SLA, from measured latencies.

Reads the newest `profile_mind` / `profile_ebnerd` result JSONs and turns their
measured numbers into the three things Q4 asks for that a latency table alone
does not answer:

  1. Does the system meet a p99 < 100 ms SLA for a single user request?
  2. What QPS can one instance sustain while meeting it?
  3. What does 1,000 queries cost at that QPS?

Everything here is arithmetic over measurements plus **four stated
assumptions**. They are printed with the result, because a cost figure whose
assumptions are invisible is worse than no cost figure:

  A1. Price basis. AWS EC2 on-demand list prices, us-east-1, as of 2026-09-12:
      g4dn.xlarge (1x T4, 4 vCPU) $0.526/h; c6i.2xlarge (8 vCPU) $0.340/h.
      List price, not negotiated or spot, and not measured by this project.
  A2. Hardware mapping. Latency was measured on an Apple M3 (local) and an
      RTX 2080 Ti (Ada). Neither is the priced instance. The 2080 Ti is the
      closer analogue of a T4 (same generation family, 2080 Ti is faster), so
      GPU-path costs derived here are, if anything, optimistic.
  A3. Worker scaling. QPS per instance = per-worker QPS x vCPU count, i.e.
      perfect linear scaling. This is optimistic: these stages are
      memory-bandwidth bound (ADR-014 measured a 2.5x slowdown from a single
      concurrent job on one machine), so real scaling will be sublinear. The
      per-worker number is the measured one; the multiplier is not.
  A4. Serial requests per worker. Each worker handles one request at a time,
      so the SLA check is against the measured single-request p99.

Usage:
    poetry run python benchmarks/cost_qps.py
    poetry run python benchmarks/cost_qps.py --sla-ms 100 --gpu-price 0.526
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_RESULTS = _HERE / "results"

# A1: list prices, us-east-1 on-demand, captured 2026-09-12.
PRICES = {
    "g4dn.xlarge": {"usd_per_hour": 0.526, "vcpu": 4, "gpu": "1x T4",
                    "source": "AWS EC2 on-demand list price, us-east-1, 2026-09-12"},
    "c6i.2xlarge": {"usd_per_hour": 0.340, "vcpu": 8, "gpu": None,
                    "source": "AWS EC2 on-demand list price, us-east-1, 2026-09-12"},
}


def newest(pattern: str, exclude_tag: str | None = None, device: str | None = None) -> dict | None:
    """Most recent results JSON matching `pattern`, ignoring smoke runs.

    `device` ("cpu"/"cuda"), when given, filters to profiles whose
    `setup.nrms_device` matches. This is load-bearing, not cosmetic: MIND has
    both a CPU- and a GPU-measured profile on disk, and picking "the newest
    one" for both the CPU-instance and GPU-instance pricing rows would price
    GPU latency on a c6i.2xlarge (no GPU to run NRMS on) or CPU latency on a
    g4dn.xlarge -- exactly the error this function exists to prevent, caught
    once already on the g4dn.xlarge side (excluded rather than reported) and
    now on the c6i.2xlarge side too once a GPU profile became the newest file.
    """
    best, best_ts = None, ""
    for path in sorted(_RESULTS.glob(pattern)):
        try:
            data = json.loads(path.read_text())
        except json.JSONDecodeError:
            continue
        if exclude_tag and exclude_tag in (data.get("tag") or ""):
            continue
        if "per_request" not in data:
            continue
        if device and data.get("setup", {}).get("nrms_device") != device:
            continue
        ts = data.get("timestamp", "")
        if ts >= best_ts:
            best, best_ts = data, ts
    return best


def analyse(profile: dict, sla_ms: float, instance: str) -> dict:
    """Per-worker and per-instance QPS, SLA verdict, and cost per 1k queries."""
    pr = profile["per_request"]["request_total_ms"]
    price = PRICES[instance]
    per_worker_qps = 1000.0 / pr["mean"]
    instance_qps = per_worker_qps * price["vcpu"]          # A3
    usd_per_1k = price["usd_per_hour"] / (instance_qps * 3600.0) * 1000.0
    headroom = sla_ms / pr["p99"] if pr["p99"] else float("inf")
    return {
        "dataset": profile.get("dataset"),
        "arm": profile.get("arm"),
        "hardware_measured": profile.get("hardware_label"),
        "n_requests_timed": profile["per_request"]["n_requests"],
        "request_mean_ms": round(pr["mean"], 3),
        "request_p99_ms": round(pr["p99"], 3),
        "sla_ms": sla_ms,
        "meets_sla": bool(pr["p99"] < sla_ms),
        "sla_headroom_x": round(headroom, 2),
        "per_worker_qps": round(per_worker_qps, 1),
        "instance": instance,
        "instance_vcpu": price["vcpu"],
        "instance_qps_optimistic": round(instance_qps, 1),
        "usd_per_hour": price["usd_per_hour"],
        "usd_per_1k_queries": round(usd_per_1k, 6),
        "batched_impressions_per_s": profile.get("end_to_end_impressions_per_s"),
        "index_memory": profile.get("index_memory"),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sla-ms", type=float, default=100.0, help="A2 Q4.3 target: p99 < this")
    ap.add_argument("--gpu-instance", default="g4dn.xlarge")
    ap.add_argument("--cpu-instance", default="c6i.2xlarge")
    ap.add_argument("--exclude-tag", default="smoke",
                    help="ignore result files whose tag contains this")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    mind_cpu = newest("*mind_*_profile.json", a.exclude_tag, device="cpu")
    mind_gpu = newest("*mind_*_profile.json", a.exclude_tag, device="cuda")
    ebnerd = newest("*ebnerd_*_profile.json", a.exclude_tag)
    if mind_cpu is None and mind_gpu is None and ebnerd is None:
        raise SystemExit("no profile JSON with a per_request block found — run the "
                         "profilers with --per-request first")

    rows = []
    if mind_cpu is not None:
        rows.append(analyse(mind_cpu, a.sla_ms, a.cpu_instance))
    else:
        print(f"NOTE: no CPU-measured MIND profile found; skipping {a.cpu_instance} row")
    if mind_gpu is not None:
        rows.append(analyse(mind_gpu, a.sla_ms, a.gpu_instance))
    else:
        print(f"NOTE: no GPU-measured MIND profile found; skipping {a.gpu_instance} row")
    if ebnerd is not None:
        rows.append(analyse(ebnerd, a.sla_ms, a.cpu_instance))

    payload = {
        "analysis": "cost_qps",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "sla_ms": a.sla_ms,
        "prices": PRICES,
        "assumptions": {
            "A1_price_basis": "AWS on-demand list prices, us-east-1, 2026-09-12; not spot, "
                              "not negotiated, not measured here",
            "A2_hardware_mapping": "latency measured on Apple M3 / RTX 2080 Ti, not on the "
                                   "priced instances; the 2080 Ti is faster than a T4, so "
                                   "GPU-path costs here are optimistic",
            "A3_worker_scaling": "instance QPS = per-worker QPS x vCPU, i.e. perfect linear "
                                 "scaling. Optimistic: ADR-014 measured a 2.5x slowdown from "
                                 "one concurrent job, so real scaling is sublinear",
            "A4_serial_requests": "one in-flight request per worker; the SLA is checked "
                                  "against the measured single-request p99",
        },
        "sources": {
            "mind_profile_cpu": mind_cpu.get("timestamp") if mind_cpu else None,
            "mind_profile_gpu": mind_gpu.get("timestamp") if mind_gpu else None,
            "ebnerd_profile": ebnerd.get("timestamp") if ebnerd else None,
        },
        "rows": rows,
    }

    out = Path(a.out) if a.out else _RESULTS / f"{time.strftime('%Y-%m-%dT%H-%M-%S')}_cost_qps.json"
    out.write_text(json.dumps(payload, indent=1))

    hdr = (f"| dataset | arm | instance | req mean | req p99 | SLA {a.sla_ms:.0f}ms | "
           f"headroom | QPS/worker | QPS/instance | $/1k queries |")
    print(hdr)
    print("|---|---|---|---:|---:|:--:|---:|---:|---:|---:|")
    for r in rows:
        print(f"| {r['dataset']} | {r['arm'] or '-'} | {r['instance']} | "
              f"{r['request_mean_ms']:.2f} ms | {r['request_p99_ms']:.2f} ms | "
              f"{'PASS' if r['meets_sla'] else 'FAIL'} | {r['sla_headroom_x']:.1f}x | "
              f"{r['per_worker_qps']:.1f} | {r['instance_qps_optimistic']:.1f} | "
              f"${r['usd_per_1k_queries']:.4f} |")
    print(f"\nwritten: {out}")
    print("Assumptions A1-A4 are recorded in the JSON; A3 (linear scaling) is the weakest.")


if __name__ == "__main__":
    main()
