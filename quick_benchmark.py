"""Quick speed comparison: all currently implemented levels.

Reports two timings per level:
  - algorithm time only (excludes connection setup / teardown)
  - total wall-clock time (setup + algorithm + teardown)

The algorithm time is the metric used for relative speedup comparisons
in §6 of the thesis, since it isolates the effect of the optimisation
under study from the constant overhead of opening a SQLite connection.
Total time is reported alongside for completeness.

For each level, REPEATS independent runs are timed and the minimum is
reported (standard practice in micro-benchmarking to reduce noise from
background processes).
"""
from __future__ import annotations

from pathlib import Path

from src.core.cached import CachedComputer
from src.core.early_termination import EarlyTerminationComputer
from src.core.naive import NaiveComputer
from src.core.responsibility import ResponsibilityComputer
from src.core.types import ComputeResult, TupleId
from src.db.query_rewriter import rewrite_query
from src.db.sqlite_backend import SQLiteBackend

DB_PATH = Path("data/synthetic/smoke_test.db")
REPEATS = 5


def one_run(
    computer: ResponsibilityComputer,
    db_path: Path,
    rewritten_query: str,
    expected_answer: tuple,
    candidates: list[TupleId],
    endogenous_tuples: list[TupleId],
) -> ComputeResult:
    """Run the computer once and return the full ComputeResult."""
    return computer.compute(
        db_path=db_path,
        rewritten_query=rewritten_query,
        expected_answer=expected_answer,
        candidates=candidates,
        endogenous_tuples=endogenous_tuples,
    )


def main() -> None:
    # One-time setup: add disabled columns to the database.
    setup = SQLiteBackend(DB_PATH)
    setup.add_disabled_columns()
    setup.close()

    sql = "SELECT DISTINCT x FROM R, S WHERE R.y = S.y"
    rewritten = rewrite_query(sql, {"R": "R", "S": "S"})

    all_tuples = [TupleId("R", i) for i in range(1, 5)] + [
        TupleId("S", i) for i in range(1, 4)
    ]

    print(f"Dataset:    smoke_test.db ({len(all_tuples)} tuples)")
    print(f"Query:      {sql}")
    print(f"Answer:     ('a',)")
    print(f"Candidates: {len(all_tuples)}")
    print(f"Repeats:    {REPEATS} per level (best run reported)")
    print()

    # Collect best-of-REPEATS timings for each level.
    algorithm_best: dict[str, float] = {}
    total_best: dict[str, float] = {}

    computers: list[type[ResponsibilityComputer]] = [
        NaiveComputer,
        EarlyTerminationComputer,
        CachedComputer,
    ]

    for ComputerClass in computers:
        algorithm_runs = []
        total_runs = []
        for _ in range(REPEATS):
            result = one_run(
                computer=ComputerClass(),
                db_path=DB_PATH,
                rewritten_query=rewritten,
                expected_answer=("a",),
                candidates=all_tuples,
                endogenous_tuples=all_tuples,
            )
            algorithm_runs.append(result.algorithm_time)
            total_runs.append(result.total_time)

        algorithm_best[ComputerClass.__name__] = min(algorithm_runs)
        total_best[ComputerClass.__name__] = min(total_runs)

    # --- Table 1: algorithm time only -------------------------------------
    print("Table 1: algorithm time only (excludes connection setup/teardown)")
    print("-" * 70)
    print(f"{'Level':<30} {'Time (ms)':>12} {'Speedup vs Naive':>20}")
    print("-" * 70)
    naive_alg = algorithm_best["NaiveComputer"]
    for name, t in algorithm_best.items():
        speedup = naive_alg / t if t > 0 else float("inf")
        print(f"{name:<30} {t * 1000:>12.2f} {speedup:>19.2f}×")
    print()

    # --- Table 2: total time ----------------------------------------------
    print("Table 2: total wall-clock time (setup + algorithm + teardown)")
    print("-" * 70)
    print(f"{'Level':<30} {'Time (ms)':>12} {'Speedup vs Naive':>20}")
    print("-" * 70)
    naive_tot = total_best["NaiveComputer"]
    for name, t in total_best.items():
        speedup = naive_tot / t if t > 0 else float("inf")
        print(f"{name:<30} {t * 1000:>12.2f} {speedup:>19.2f}×")
    print()


if __name__ == "__main__":
    main()