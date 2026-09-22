"""Benchmarking harness: randomized problems, comparison tables and plots."""

from .run_benchmark import (
    BenchmarkRow,
    benchmark_grids,
    random_grid,
    summarize,
)

__all__ = ["BenchmarkRow", "benchmark_grids", "random_grid", "summarize"]
