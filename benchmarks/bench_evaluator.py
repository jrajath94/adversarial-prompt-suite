"""Benchmark: async batch evaluation throughput.

Measures evaluations/second using the MockLLMClient across varying
batch sizes and concurrency levels.

Run with:
    python benchmarks/bench_evaluator.py

Expected output:
    Batch=10,  concurrency=5:  ~XXXX evals/sec, avg_latency=XXXms
    Batch=50,  concurrency=10: ~XXXX evals/sec, avg_latency=XXXms
    Batch=100, concurrency=20: ~XXXX evals/sec, avg_latency=XXXms
"""

from __future__ import annotations

import asyncio
import statistics
import sys
import time
from pathlib import Path

# Allow running as a script without installation.
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from adversarial_prompt_suite.attacks import (
    load_extraction_templates,
    load_injection_templates,
    load_jailbreak_templates,
)
from adversarial_prompt_suite.evaluator import Evaluator, MockLLMClient
from adversarial_prompt_suite.models import AttackTemplate

# Pre-load all built-in templates once.
ALL_TEMPLATES: list[AttackTemplate] = (
    load_jailbreak_templates()
    + load_injection_templates()
    + load_extraction_templates()
)


async def run_batch_benchmark(
    batch_size: int,
    concurrency: int,
    iterations: int = 3,
) -> dict:
    """Run batch evaluation N times and collect throughput statistics.

    Args:
        batch_size: Number of templates per evaluation batch.
        concurrency: Max concurrent requests (semaphore limit).
        iterations: Number of timed runs to average over.

    Returns:
        Dict with throughput and latency statistics.
    """
    # Cycle through available templates if batch_size > len(ALL_TEMPLATES).
    templates = [ALL_TEMPLATES[i % len(ALL_TEMPLATES)] for i in range(batch_size)]
    client = MockLLMClient()
    evaluator = Evaluator(client=client, concurrency=concurrency)

    wall_times: list[float] = []

    for _ in range(iterations):
        start = time.perf_counter()
        results = await evaluator.evaluate_batch(templates)
        elapsed = time.perf_counter() - start
        wall_times.append(elapsed)

    total_evals = batch_size * iterations
    total_time = sum(wall_times)
    evals_per_sec = total_evals / total_time

    latencies = [r.latency_ms for r in results]

    return {
        "batch_size": batch_size,
        "concurrency": concurrency,
        "iterations": iterations,
        "evals_per_sec": evals_per_sec,
        "avg_wall_time_s": statistics.mean(wall_times),
        "p50_latency_ms": statistics.median(latencies),
        "p99_latency_ms": (
            sorted(latencies)[int(len(latencies) * 0.99)] if latencies else 0
        ),
        "total_calls": client.call_count,
    }


def _print_row(result: dict) -> None:
    """Print a single benchmark result row."""
    print(
        f"  batch={result['batch_size']:>4d} | "
        f"concurrency={result['concurrency']:>3d} | "
        f"throughput={result['evals_per_sec']:>8.1f} evals/sec | "
        f"avg_wall={result['avg_wall_time_s']*1000:>7.1f}ms | "
        f"p50_lat={result['p50_latency_ms']:>5.1f}ms | "
        f"p99_lat={result['p99_latency_ms']:>5.1f}ms"
    )


async def main() -> None:
    """Run all benchmark configurations and print a formatted report."""
    configs = [
        (10, 5),
        (25, 10),
        (50, 10),
        (50, 25),
        (100, 20),
        (100, 50),
    ]

    print("\n" + "=" * 90)
    print("adversarial-prompt-suite — Batch Evaluation Throughput Benchmark")
    print(f"Using MockLLMClient | Templates available: {len(ALL_TEMPLATES)}")
    print("=" * 90)

    results = []
    for batch_size, concurrency in configs:
        result = await run_batch_benchmark(batch_size, concurrency, iterations=5)
        _print_row(result)
        results.append(result)

    print("-" * 90)
    best = max(results, key=lambda r: r["evals_per_sec"])
    print(
        f"\nPeak throughput: {best['evals_per_sec']:.1f} evals/sec "
        f"(batch={best['batch_size']}, concurrency={best['concurrency']})"
    )
    print(
        f"Min p99 latency: "
        f"{min(r['p99_latency_ms'] for r in results):.1f}ms"
    )
    print("=" * 90 + "\n")

    # Return best throughput so callers can assert on it.
    return best["evals_per_sec"]


if __name__ == "__main__":
    asyncio.run(main())
