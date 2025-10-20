"""Interactive benchmarking and visualization for WarpDB GPU vs CPU execution.

This script demonstrates how WarpDB's GPU-accelerated query execution compares to
CPU-bound processing by running example SQL-like expressions against CSV/JSON
inputs.  When CUDA hardware or the Python bindings are not available it falls
back to curated sample metrics so the visualization still communicates the
performance profile of the engine.
"""

from __future__ import annotations

import argparse
import statistics
import sys
import textwrap
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

try:  # Pandas is required only for live benchmarking.
    import pandas as pd
except Exception as exc:  # pragma: no cover - optional dependency
    pd = None  # type: ignore
    _PANDAS_IMPORT_ERROR = exc
else:
    _PANDAS_IMPORT_ERROR = None

try:  # Optional dependency used for GPU execution.
    import pywarpdb  # type: ignore
except Exception:  # pragma: no cover - environment without bindings
    pywarpdb = None

try:
    import matplotlib.pyplot as plt
except Exception as exc:  # pragma: no cover - matplotlib is optional
    plt = None
    _MATPLOTLIB_IMPORT_ERROR = exc
else:
    _MATPLOTLIB_IMPORT_ERROR = None


@dataclass
class Metric:
    label: str
    execution_ms: float
    memory_throughput_gb_s: float
    gpu_utilization_pct: float
    gpus: int

    def as_dict(self) -> Dict[str, float]:
        return {
            "execution_ms": self.execution_ms,
            "memory_throughput_gb_s": self.memory_throughput_gb_s,
            "gpu_utilization_pct": self.gpu_utilization_pct,
            "gpus": self.gpus,
        }


SAMPLE_BENCHMARKS: Dict[str, List[Metric]] = {
    "price * quantity": [
        Metric("CPU", 42.7, 6.4, 0.0, 0),
        Metric("GPU x1", 6.1, 44.5, 68.0, 1),
        Metric("GPU x4", 2.2, 126.0, 92.0, 4),
    ],
    "price * quantity WHERE price > 10": [
        Metric("CPU", 58.4, 4.9, 0.0, 0),
        Metric("GPU x1", 9.8, 29.7, 51.0, 1),
        Metric("GPU x4", 3.5, 83.2, 88.0, 4),
    ],
    "(price * 0.9) + shipping_cost": [
        Metric("CPU", 71.2, 4.2, 0.0, 0),
        Metric("GPU x1", 10.4, 28.6, 48.0, 1),
        Metric("GPU x4", 3.6, 82.7, 86.0, 4),
    ],
}


def parse_query_parts(query: str) -> Dict[str, str]:
    upper = query.upper()
    where_pos = upper.find(" WHERE ")
    if where_pos == -1:
        return {"expr": query.strip(), "where": ""}
    return {
        "expr": query[:where_pos].strip(),
        "where": query[where_pos + len(" WHERE "):].strip(),
    }


def compute_dataset_size_bytes(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def run_cpu_expression(df: pd.DataFrame, expr: str, where: str) -> pd.Series:
    if where:
        filtered = df.query(where, engine="python")
    else:
        filtered = df
    return filtered.eval(expr, engine="python")


def measure_cpu(df: pd.DataFrame, query: str, *, repeats: int, data_size_gb: float,
                theoretical_bandwidth: float) -> Metric:
    parts = parse_query_parts(query)
    timings: List[float] = []
    for _ in range(repeats):
        start = time.perf_counter()
        run_cpu_expression(df, parts["expr"], parts["where"])
        timings.append((time.perf_counter() - start) * 1000.0)
    execution_ms = statistics.mean(timings)
    throughput = 0.0
    if execution_ms > 0:
        throughput = data_size_gb / (execution_ms / 1000.0)
    utilization = min(100.0, (throughput / theoretical_bandwidth) * 100.0) if theoretical_bandwidth else 0.0
    return Metric("CPU", execution_ms, throughput, utilization, gpus=0)


def measure_gpu(csv_path: Path, query: str, *, repeats: int, data_size_gb: float,
                theoretical_bandwidth: float, use_multi_gpu: bool) -> Optional[Metric]:
    if pywarpdb is None:
        return None

    timings: List[float] = []
    db = pywarpdb.WarpDB(str(csv_path))
    for _ in range(repeats):
        start = time.perf_counter()
        if use_multi_gpu:
            db.query_multi_gpu(query)
        else:
            db.query(query)
        timings.append((time.perf_counter() - start) * 1000.0)
    execution_ms = statistics.mean(timings)
    throughput = 0.0
    if execution_ms > 0:
        throughput = data_size_gb / (execution_ms / 1000.0)
    utilization = min(100.0, (throughput / theoretical_bandwidth) * 100.0) if theoretical_bandwidth else 0.0
    label = "GPU multi" if use_multi_gpu else "GPU x1"
    gpu_count = detect_gpu_count()
    return Metric(label, execution_ms, throughput, utilization, gpus=gpu_count)


def detect_gpu_count() -> int:
    if pywarpdb is None:
        return 0

    attr = getattr(pywarpdb, "get_device_count", None)
    if callable(attr):
        try:
            return max(1, int(attr()))
        except Exception:
            return 1

    attr = getattr(pywarpdb, "device_count", None)
    if callable(attr):
        try:
            return max(1, int(attr()))
        except Exception:
            return 1
    if attr is not None:
        try:
            return max(1, int(attr))
        except Exception:
            return 1
    return 1


def make_plot(query: str, metrics: List[Metric], *, output: Path, show: bool) -> None:
    if plt is None:  # pragma: no cover - depends on environment
        print(
            "[warning] matplotlib is unavailable; skipping plot generation for "
            f"'{query}'. Error: {_MATPLOTLIB_IMPORT_ERROR}"
        )
        return

    labels = [m.label for m in metrics]
    exec_values = [m.execution_ms for m in metrics]
    throughput_values = [m.memory_throughput_gb_s for m in metrics]
    util_values = [m.gpu_utilization_pct for m in metrics]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(f"WarpDB GPU vs CPU Benchmark — {query}")

    bars = axes[0].bar(labels, exec_values, color=["#6c757d" if m.gpus == 0 else "#0d6efd" for m in metrics])
    axes[0].set_ylabel("Execution time (ms)")
    axes[0].invert_yaxis()  # shorter bars are faster
    for bar, value in zip(bars, exec_values):
        axes[0].text(bar.get_x() + bar.get_width() / 2.0, bar.get_height(), f"{value:.1f}",
                     ha="center", va="bottom")

    bars = axes[1].bar(labels, throughput_values,
                       color=["#6c757d" if m.gpus == 0 else "#198754" for m in metrics])
    axes[1].set_ylabel("Memory throughput (GB/s)")
    for bar, value in zip(bars, throughput_values):
        axes[1].text(bar.get_x() + bar.get_width() / 2.0, bar.get_height(), f"{value:.1f}",
                     ha="center", va="bottom")

    bars = axes[2].bar(labels, util_values,
                       color=["#adb5bd" if m.gpus == 0 else "#fd7e14" for m in metrics])
    axes[2].set_ylabel("GPU utilization (%)")
    axes[2].set_ylim(0, 100)
    for bar, value in zip(bars, util_values):
        axes[2].text(bar.get_x() + bar.get_width() / 2.0, bar.get_height(), f"{value:.0f}",
                     ha="center", va="bottom")

    for ax in axes:
        ax.set_xticklabels(labels, rotation=15, ha="right")
        ax.grid(axis="y", linestyle="--", alpha=0.3)

    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig(output, dpi=150)
    if show:  # pragma: no cover - requires interactive backend
        plt.show()
    plt.close(fig)


def gather_sample_metrics(query: str) -> List[Metric]:
    if query not in SAMPLE_BENCHMARKS:
        raise KeyError(
            f"No sample metrics available for query '{query}'. Available queries:"
            f" {', '.join(SAMPLE_BENCHMARKS)}"
        )
    return SAMPLE_BENCHMARKS[query]


def format_metric_table(metrics: List[Metric]) -> str:
    header = f"{'Configuration':<15} | {'Exec (ms)':>10} | {'Throughput (GB/s)':>18} | {'GPU Util (%)':>12}"
    rows = [header, "-" * len(header)]
    for m in metrics:
        rows.append(
            f"{m.label:<15} | {m.execution_ms:>10.1f} | {m.memory_throughput_gb_s:>18.1f} | {m.gpu_utilization_pct:>12.1f}"
        )
    return "\n".join(rows)


def run_live_benchmark(args: argparse.Namespace) -> None:
    if pd is None:
        raise RuntimeError(
            "pandas is required for live benchmarking but could not be imported:"
            f" {_PANDAS_IMPORT_ERROR}"
        )

    csv_path = Path(args.dataset)
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset '{csv_path}' does not exist")

    df = pd.read_csv(csv_path)
    data_size_gb = compute_dataset_size_bytes(csv_path) / (1024 ** 3)

    for query in args.queries:
        metrics: List[Metric] = []
        metrics.append(
            measure_cpu(
                df,
                query,
                repeats=args.repeats,
                data_size_gb=data_size_gb,
                theoretical_bandwidth=args.theoretical_cpu_bandwidth,
            )
        )
        if pywarpdb is not None:
            gpu_metric = measure_gpu(
                csv_path,
                query,
                repeats=args.repeats,
                data_size_gb=data_size_gb,
                theoretical_bandwidth=args.theoretical_gpu_bandwidth,
                use_multi_gpu=False,
            )
            if gpu_metric is not None:
                metrics.append(gpu_metric)
            if args.enable_multi_gpu:
                multi = measure_gpu(
                    csv_path,
                    query,
                    repeats=args.repeats,
                    data_size_gb=data_size_gb,
                    theoretical_bandwidth=args.theoretical_gpu_bandwidth,
                    use_multi_gpu=True,
                )
                if multi is not None and multi.gpus > 1:
                    multi.label = f"GPU x{multi.gpus}"
                    metrics.append(multi)
        else:
            print(
                "[warning] pywarpdb bindings are unavailable. Only CPU metrics will be collected."
            )

        output = Path(args.output_dir) / f"{sanitize_filename(query)}.png"
        output.parent.mkdir(parents=True, exist_ok=True)
        make_plot(query, metrics, output=output, show=args.show)
        print(f"Generated visualization for '{query}' at {output}")
        print(format_metric_table(metrics))
        print()


def sanitize_filename(name: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in name)[:80]


def run_sample_mode(args: argparse.Namespace) -> None:
    for query in args.queries:
        metrics = gather_sample_metrics(query)
        output = Path(args.output_dir) / f"sample_{sanitize_filename(query)}.png"
        output.parent.mkdir(parents=True, exist_ok=True)
        make_plot(query, metrics, output=output, show=args.show)
        print(f"Generated sample visualization for '{query}' at {output}")
        print(format_metric_table(metrics))
        print(
            textwrap.fill(
                "These figures use curated benchmark data to illustrate how WarpDB scales from"
                " CPU execution to single- and multi-GPU runs. Use `--mode live` to run the"
                " benchmark on your own hardware.",
                width=100,
            )
        )
        print()


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize GPU-accelerated WarpDB query performance versus CPU execution."
    )
    parser.add_argument(
        "--mode",
        choices=["sample", "live"],
        default="sample",
        help="Use curated sample metrics or execute live benchmarks (requires CUDA + pywarpdb).",
    )
    parser.add_argument(
        "--dataset",
        default="data/test.csv",
        help="Path to the CSV dataset for live benchmarking.",
    )
    parser.add_argument(
        "--queries",
        nargs="+",
        default=list(SAMPLE_BENCHMARKS.keys()),
        help="Queries to evaluate.",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=5,
        help="Number of timing repetitions for each configuration.",
    )
    parser.add_argument(
        "--output-dir",
        default="visualizations",
        help="Directory to store generated plots.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the plots interactively after generating them.",
    )
    parser.add_argument(
        "--enable-multi-gpu",
        action="store_true",
        help="Attempt to run multi-GPU benchmarks when hardware is available.",
    )
    parser.add_argument(
        "--theoretical-gpu-bandwidth",
        type=float,
        default=900.0,
        help="Theoretical GPU memory bandwidth (GB/s) for utilization estimates.",
    )
    parser.add_argument(
        "--theoretical-cpu-bandwidth",
        type=float,
        default=100.0,
        help="Approximate CPU memory bandwidth (GB/s) for utilization estimates.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = parse_args(argv)
    if args.mode == "live":
        run_live_benchmark(args)
    else:
        run_sample_mode(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
