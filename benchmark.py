"""Final benchmark: all 4 optimisation levels on all 7 datasets.

Produces the timing data used in §6 of the thesis. For each
(dataset, level) pair, runs the computation REPEATS times (configured
per dataset depending on cost), records each run's algorithm time and
total time, and writes the median of each metric to a CSV.

Per-dataset repeat counts:
    Fast datasets (synth_small/medium, synth_join3,
        synth_dense, imdb_burton): 5 repeats — gives stable median.
    synth_large: 3 repeats — Naive is ~22s/run, so 3 runs ≈ 4 minutes
        total per level, manageable.
    synth_xlarge: 1 run — Naive is ~9 minutes/run; a single run is
        sufficient for the headline number and avoids hour-scale waits.

Output: experiments/results/benchmark.csv with columns
    dataset, level, repeats, alg_time_median_ms, alg_time_min_ms,
    alg_time_max_ms, total_time_median_ms

The median is what we report; min and max are kept for sanity-checking
that the median is not an outlier. Algorithm time (excluding setup/
teardown) is the primary metric for speedup analysis.
"""
from __future__ import annotations

import csv
import statistics
import time
from pathlib import Path

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


# Per-dataset repeat counts, chosen by total expected cost.
DATASET_REPEATS: dict[str, int] = {
    "synth_small":  5,
    "synth_medium": 5,
    "synth_large":  5,
    "synth_xlarge": 5,
    "synth_join3":  5,
    "synth_dense":  5,
    "imdb_burton":  5,
}


OUTPUT_CSV = Path("experiments/results/benchmark.csv")


def run_one_dataset(
    dataset_name: str,
    repeats: int,
) -> list[dict]:
    """Run all 4 levels on one dataset, repeated `repeats` times each.

    Returns a list of result dicts, one per (level) row, ready to be
    written to the CSV.
    """
    # Rebuild the dataset to ensure we benchmark on a fresh state.
    spec = DATASET_REGISTRY[dataset_name]()

    # Ensure disabled columns exist on the master file.
    setup = SQLiteBackend(spec.db_path)
    setup.add_disabled_columns()
    setup.close()

    rewritten = rewrite_query(spec.sql_query, spec.aliases)

    rows = []
    for ComputerClass in COMPUTERS:
        alg_times = []
        total_times = []
        for run_idx in range(repeats):
            print(
                f"    {ComputerClass.__name__:<28} "
                f"run {run_idx + 1}/{repeats}...",
                end=" ",
                flush=True,
            )
            t_start = time.perf_counter()
            result = ComputerClass().compute(
                db_path=spec.db_path,
                rewritten_query=rewritten,
                expected_answer=spec.expected_answer,
                candidates=spec.candidates,
                endogenous_tuples=spec.endogenous,
            )
            wall = time.perf_counter() - t_start
            alg_times.append(result.algorithm_time)
            total_times.append(result.total_time)
            print(f"alg={result.algorithm_time * 1000:.1f}ms  (wall {wall:.1f}s)")

        row = {
            "dataset": dataset_name,
            "n_endogenous": spec.n_endogenous,
            "level": ComputerClass.__name__,
            "repeats": repeats,
            "alg_time_median_ms": statistics.median(alg_times) * 1000,
            "alg_time_min_ms": min(alg_times) * 1000,
            "alg_time_max_ms": max(alg_times) * 1000,
            "total_time_median_ms": statistics.median(total_times) * 1000,
        }
        rows.append(row)

    return rows


def main() -> None:
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict] = []
    # Run datasets in increasing cost order so easy ones finish first
    # and you see progress quickly.
    dataset_order = [
        "synth_small",
        "imdb_burton",
        "synth_join3",
        "synth_medium",
        "synth_dense",
        "synth_large",
        "synth_xlarge",
    ]

    grand_start = time.perf_counter()

    for i, dataset_name in enumerate(dataset_order, start=1):
        repeats = DATASET_REPEATS[dataset_name]
        spec = DATASET_REGISTRY[dataset_name]()
        print(
            f"\n[{i}/{len(dataset_order)}] {dataset_name} "
            f"(|D_n|={spec.n_endogenous}, repeats={repeats})"
        )
        rows = run_one_dataset(dataset_name, repeats)
        all_rows.extend(rows)

        # Write CSV incrementally so a crash mid-benchmark still leaves
        # partial results on disk.
        with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
            writer.writeheader()
            writer.writerows(all_rows)

    grand_wall = time.perf_counter() - grand_start

    # Final summary table.
    print("\n" + "=" * 78)
    print(f"Benchmark complete in {grand_wall / 60:.1f} minutes.")
    print(f"Results written to {OUTPUT_CSV}")
    print("=" * 78)
    print()
    print(f"{'Dataset':<14} {'|D_n|':>5} {'Level':<28} {'Median alg (ms)':>16}")
    print("-" * 78)
    for row in all_rows:
        print(
            f"{row['dataset']:<14} {row['n_endogenous']:>5} "
            f"{row['level']:<28} {row['alg_time_median_ms']:>16.2f}"
        )


if __name__ == "__main__":
    main()