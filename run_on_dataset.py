"""Run all 4 optimisation levels on a single dataset and print results.

Usage:
    python run_on_dataset.py synth_small
    python run_on_dataset.py synth_medium    (when it exists)

Reports algorithm-only time, total time, and the full ranking. Useful
for sanity-checking a new dataset before adding it to the full benchmark.
"""
from __future__ import annotations

import sys

from create_synthetic_datasets import DATASET_REGISTRY

from src.core.cached import CachedComputer
from src.core.early_termination import EarlyTerminationComputer
from src.core.naive import NaiveComputer
from src.core.parallel import ParallelComputer
from src.core.responsibility import ResponsibilityComputer
from src.db.query_rewriter import rewrite_query
from src.db.sqlite_backend import SQLiteBackend


COMPUTERS: list[type[ResponsibilityComputer]] = [
    NaiveComputer,
    EarlyTerminationComputer,
    CachedComputer,
    ParallelComputer,
]


def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: python run_on_dataset.py <dataset_name>")
        print(f"Available: {', '.join(DATASET_REGISTRY.keys())}")
        sys.exit(1)

    dataset_name = sys.argv[1]
    if dataset_name not in DATASET_REGISTRY:
        print(f"Unknown dataset: {dataset_name}")
        print(f"Available: {', '.join(DATASET_REGISTRY.keys())}")
        sys.exit(1)

    # Rebuild the dataset to make sure it's fresh.
    spec = DATASET_REGISTRY[dataset_name]()

    # One-time setup: add disabled columns.
    setup = SQLiteBackend(spec.db_path)
    setup.add_disabled_columns()
    setup.close()

    rewritten = rewrite_query(spec.sql_query, spec.aliases)

    print(f"Dataset:    {spec.name}")
    print(f"Description: {spec.description}")
    print(f"Query:      {spec.sql_query}")
    print(f"Answer:     {spec.expected_answer}")
    print(f"|D_n|:      {spec.n_endogenous}")
    print(f"|C|:        {len(spec.candidates)}")
    print()

    # Run each level once and report.
    results = {}
    for ComputerClass in COMPUTERS:
        result = ComputerClass().compute(
            db_path=spec.db_path,
            rewritten_query=rewritten,
            expected_answer=spec.expected_answer,
            candidates=spec.candidates,
            endogenous_tuples=spec.endogenous,
        )
        results[ComputerClass.__name__] = result

    # ---- Timings ----
    print("Timings (single run):")
    print(f"{'Level':<30} {'Alg (ms)':>10} {'Total (ms)':>12}")
    print("-" * 56)
    for name, r in results.items():
        print(
            f"{name:<30} {r.algorithm_time * 1000:>10.2f} "
            f"{r.total_time * 1000:>12.2f}"
        )
    print()

    # Speedups (algorithm-only)
    naive_alg = results["NaiveComputer"].algorithm_time
    print("Speedup vs Naive (algorithm time only):")
    for name, r in results.items():
        if name == "NaiveComputer":
            continue
        speedup = naive_alg / r.algorithm_time if r.algorithm_time > 0 else float("inf")
        print(f"  {name:<30} {speedup:>6.2f}×")
    print()

    # ---- Ranking (use Naive's output, the others must match) ----
    print("Ranking (from NaiveComputer):")
    print(f"{'Tuple':<15} {'ρ':<8} {'min |Γ|':<10}")
    print("-" * 35)
    naive_ranking = results["NaiveComputer"].ranking
    for r in naive_ranking.sorted_by_responsibility():
        size_str = str(r.min_contingency_size) if r.min_contingency_size is not None else "—"
        print(f"{str(r.tuple_id):<15} {r.responsibility:<8.3f} {size_str:<10}")


if __name__ == "__main__":
    main()