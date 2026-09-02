"""Benchmark WarpDB's GPU expression path against a pandas CPU baseline.

Every number this script prints or plots is measured on the machine it runs on.
There is no sample/demo mode: if pandas or the ``pywarpdb`` bindings are
missing, the script fails with an explanatory error rather than substituting
figures that were not measured.

What is and is not measured:

* Timings cover query execution only. The CSV is read once up front for both
  engines (``pandas.read_csv`` for the CPU side, the ``WarpDB`` constructor for
  the GPU side), so neither number includes ingest.
* ``--warmup`` iterations run before timing. This matters for WarpDB: the first
  execution of an expression pays for NVRTC compilation, and later executions
  hit the JIT cache. Reported times are therefore warm, cache-hit times.
* Throughput is reported as input CSV bytes divided by execution time. It is a
  size-relative rate for comparing the two engines on the same file -- not a
  measurement of achieved device memory bandwidth.
* GPU utilization is not reported. Sampling it requires NVML, which this script
  does not depend on.
* This benchmarks ``WarpDB.query``, the JIT-compiled GPU expression path.
  ``WarpDB.query_sql`` executes on the host and is not covered here.
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

try:  # Required for the CPU baseline.
    import pandas as pd
except Exception as exc:  # pragma: no cover - optional dependency
    pd = None  # type: ignore
    _PANDAS_IMPORT_ERROR = exc
else:
    _PANDAS_IMPORT_ERROR = None

try:  # Required for the GPU measurements.
    import pywarpdb  # type: ignore
except Exception as exc:  # pragma: no cover - environment without bindings
    pywarpdb = None
    _PYWARPDB_IMPORT_ERROR = exc
else:
    _PYWARPDB_IMPORT_ERROR = None

try:
    import matplotlib.pyplot as plt
except Exception as exc:  # pragma: no cover - matplotlib is optional
    plt = None
    _MATPLOTLIB_IMPORT_ERROR = exc
else:
    _MATPLOTLIB_IMPORT_ERROR = None


DEFAULT_QUERIES = [
    "price * quantity",
    "price * quantity WHERE price > 10",
]


@dataclass
class Metric:
    label: str
    mean_ms: float
    stdev_ms: float
    min_ms: float
    throughput_gb_s: float
    gpus: int

    def as_dict(self) -> Dict[str, float]:
        return {
            "mean_ms": self.mean_ms,
            "stdev_ms": self.stdev_ms,
            "min_ms": self.min_ms,
            "throughput_gb_s": self.throughput_gb_s,
            "gpus": self.gpus,
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


def run_cpu_expression(df: "pd.DataFrame", expr: str, where: str):
    # Use pandas' default engine (numexpr when installed) rather than forcing
    # engine="python". Pinning the slow path would understate the baseline and
    # overstate WarpDB's speedup.
    filtered = df.query(where) if where else df
    return filtered.eval(expr)


def _summarize(label: str, timings: List[float], data_size_gb: float,
               gpus: int) -> Metric:
    mean_ms = statistics.mean(timings)
    stdev_ms = statistics.stdev(timings) if len(timings) > 1 else 0.0
    min_ms = min(timings)
    throughput = data_size_gb / (mean_ms / 1000.0) if mean_ms > 0 else 0.0
    return Metric(label, mean_ms, stdev_ms, min_ms, throughput, gpus)


def measure_cpu(df: "pd.DataFrame", query: str, *, repeats: int, warmup: int,
                data_size_gb: float) -> Metric:
    parts = parse_query_parts(query)
    for _ in range(warmup):
        run_cpu_expression(df, parts["expr"], parts["where"])
    timings: List[float] = []
    for _ in range(repeats):
        start = time.perf_counter()
        run_cpu_expression(df, parts["expr"], parts["where"])
        timings.append((time.perf_counter() - start) * 1000.0)
    return _summarize("CPU (pandas)", timings, data_size_gb, gpus=0)


def measure_gpu(db, query: str, *, repeats: int, warmup: int,
                data_size_gb: float, use_multi_gpu: bool) -> Metric:
    run = db.query_multi_gpu if use_multi_gpu else db.query
    for _ in range(warmup):
        run(query)
    timings: List[float] = []
    for _ in range(repeats):
        start = time.perf_counter()
        run(query)
        timings.append((time.perf_counter() - start) * 1000.0)
    gpu_count = detect_gpu_count()
    label = f"GPU x{gpu_count}" if use_multi_gpu else "GPU x1"
    return _summarize(label, timings, data_size_gb,
                      gpus=gpu_count if use_multi_gpu else 1)


def detect_gpu_count() -> int:
    if pywarpdb is None:
        return 0

    for name in ("get_device_count", "device_count"):
        attr = getattr(pywarpdb, name, None)
        if attr is None:
            continue
        try:
            return max(1, int(attr() if callable(attr) else attr))
        except Exception:
            return 1
    return 1


def make_plot(query: str, metrics: List[Metric], *, output: Path,
              show: bool) -> None:
    if plt is None:  # pragma: no cover - depends on environment
        print(
            "[warning] matplotlib is unavailable; skipping plot generation for "
            f"'{query}'. Error: {_MATPLOTLIB_IMPORT_ERROR}"
        )
        return

    labels = [m.label for m in metrics]
    exec_values = [m.mean_ms for m in metrics]
    errors = [m.stdev_ms for m in metrics]
    throughput_values = [m.throughput_gb_s for m in metrics]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f"WarpDB measured benchmark — {query}")

    colors = ["#6c757d" if m.gpus == 0 else "#0d6efd" for m in metrics]
    bars = axes[0].bar(labels, exec_values, yerr=errors, capsize=4, color=colors)
    axes[0].set_ylabel("Execution time (ms), lower is better")
    for bar, value in zip(bars, exec_values):
        axes[0].text(bar.get_x() + bar.get_width() / 2.0, bar.get_height(),
                     f"{value:.2f}", ha="center", va="bottom")

    colors = ["#6c757d" if m.gpus == 0 else "#198754" for m in metrics]
    bars = axes[1].bar(labels, throughput_values, color=colors)
    axes[1].set_ylabel("Input GB/s (CSV bytes ÷ exec time)")
    for bar, value in zip(bars, throughput_values):
        axes[1].text(bar.get_x() + bar.get_width() / 2.0, bar.get_height(),
                     f"{value:.2f}", ha="center", va="bottom")

    for ax in axes:
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=15, ha="right")
        ax.grid(axis="y", linestyle="--", alpha=0.3)

    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig(output, dpi=150)
    if show:  # pragma: no cover - requires interactive backend
        plt.show()
    plt.close(fig)


def format_metric_table(metrics: List[Metric]) -> str:
    header = (f"{'Configuration':<15} | {'Mean (ms)':>10} | {'Stdev':>8} | "
              f"{'Min (ms)':>10} | {'Input GB/s':>11}")
    rows = [header, "-" * len(header)]
    for m in metrics:
        rows.append(
            f"{m.label:<15} | {m.mean_ms:>10.2f} | {m.stdev_ms:>8.2f} | "
            f"{m.min_ms:>10.2f} | {m.throughput_gb_s:>11.2f}"
        )
    return "\n".join(rows)


def require_dependencies(need_gpu: bool) -> None:
    if pd is None:
        raise RuntimeError(
            "pandas is required for the CPU baseline but could not be imported: "
            f"{_PANDAS_IMPORT_ERROR}"
        )
    if need_gpu and pywarpdb is None:
        raise RuntimeError(
            "pywarpdb is required for GPU measurements but could not be "
            f"imported: {_PYWARPDB_IMPORT_ERROR}. Build the bindings with "
            "-DWARPDB_BUILD_PYTHON=ON, or pass --cpu-only to record just the "
            "pandas baseline."
        )


def run_benchmark(args: argparse.Namespace) -> None:
    require_dependencies(need_gpu=not args.cpu_only)

    csv_path = Path(args.dataset)
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset '{csv_path}' does not exist")

    df = pd.read_csv(csv_path)
    data_size_gb = compute_dataset_size_bytes(csv_path) / (1024 ** 3)
    if len(df) < args.min_rows:
        print(
            f"[warning] '{csv_path}' has {len(df)} rows. Timings below "
            f"{args.min_rows} rows are dominated by fixed per-call overhead "
            "(kernel launch, PCIe transfer, Python dispatch) and say nothing "
            "about throughput. Use a larger dataset for a meaningful result.",
            file=sys.stderr,
        )

    db = None if args.cpu_only else pywarpdb.WarpDB(str(csv_path))

    for query in args.queries:
        metrics: List[Metric] = [
            measure_cpu(df, query, repeats=args.repeats, warmup=args.warmup,
                        data_size_gb=data_size_gb)
        ]

        if db is not None:
            metrics.append(
                measure_gpu(db, query, repeats=args.repeats,
                            warmup=args.warmup, data_size_gb=data_size_gb,
                            use_multi_gpu=False)
            )
            if args.enable_multi_gpu:
                multi = measure_gpu(db, query, repeats=args.repeats,
                                    warmup=args.warmup,
                                    data_size_gb=data_size_gb,
                                    use_multi_gpu=True)
                if multi.gpus > 1:
                    metrics.append(multi)
                else:
                    print("[warning] --enable-multi-gpu requested but only one "
                          "device was detected; skipping the multi-GPU run.",
                          file=sys.stderr)

        output = Path(args.output_dir) / f"{sanitize_filename(query)}.png"
        output.parent.mkdir(parents=True, exist_ok=True)
        make_plot(query, metrics, output=output, show=args.show)
        print(f"Generated visualization for '{query}' at {output}")
        print(format_metric_table(metrics))
        print()


def sanitize_filename(name: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in name)[:80]


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure WarpDB GPU query performance against a pandas "
                    "CPU baseline. All reported figures are measured locally.",
    )
    parser.add_argument(
        "--dataset",
        default="data/test.csv",
        help="Path to the CSV dataset to benchmark.",
    )
    parser.add_argument(
        "--queries",
        nargs="+",
        default=list(DEFAULT_QUERIES),
        help="Expressions to evaluate (WarpDB expression syntax).",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=5,
        help="Number of timed repetitions for each configuration.",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=2,
        help="Untimed warmup iterations run before timing. Needed so WarpDB "
             "measurements reflect JIT cache hits rather than NVRTC compiles.",
    )
    parser.add_argument(
        "--min-rows",
        type=int,
        default=100_000,
        help="Warn when the dataset has fewer rows than this, because the "
             "measurement would be dominated by fixed overhead.",
    )
    parser.add_argument(
        "--cpu-only",
        action="store_true",
        help="Record only the pandas baseline; do not require pywarpdb.",
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
        help="Also time WarpDB.query_multi_gpu when multiple devices exist.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = parse_args(argv)
    run_benchmark(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
