"""Quick speed comparison: Naive vs Early Termination.

This is a small smoke benchmark on the synthetic dataset. The dataset
is too small for the speedup to be dramatic, but it should be visible.

A proper benchmark on larger datasets will live in
experiments/benchmark.py.
"""
from __future__ import annotations

import time
from pathlib import Path

from src.core.early_termination import EarlyTerminationComputer
from src.core.naive import NaiveComputer
from src.core.responsibility import ResponsibilityComputer
from src.core.types import TupleId
from src.db.query_rewriter import rewrite_query
from src.db.sqlite_backend import SQLiteBackend

DB_PATH = Path("data/synthetic/smoke_test.db")
REPEATS = 5


def time_one_run(
    computer: ResponsibilityComputer,
    backend: SQLiteBackend,
    rewritten_query: str,
    expected_answer: tuple,
    candidates: list[TupleId],
    endogenous_tuples: list[TupleId],
) -> float:
    """Run the computer once and return wall-clock seconds."""
    backend.enable_all()
    start = time.perf_counter()
    computer.compute(
        backend=backend,
        rewritten_query=rewritten_query,
        expected_answer=expected_answer,
        candidates=candidates,
        endogenous_tuples=endogenous_tuples,
    )
    return time.perf_counter() - start


def main() -> None:
    backend = SQLiteBackend(DB_PATH)
    backend.add_disabled_columns()

    sql = "SELECT DISTINCT x FROM R, S WHERE R.y = S.y"
    rewritten = rewrite_query(sql, {"R": "R", "S": "S"})

    all_tuples = [TupleId("R", i) for i in range(1, 5)] + [
        TupleId("S", i) for i in range(1, 4)
    ]

    print(f"Dataset: smoke_test.db ({len(all_tuples)} tuples)")
    print(f"Query: {sql}")
    print(f"Answer: ('a',)")
    print(f"Candidates: {len(all_tuples)}")
    print(f"Repeats per level: {REPEATS} (taking the minimum)")
    print("-" * 60)

    results: dict[str, float] = {}

    for ComputerClass in [NaiveComputer, EarlyTerminationComputer]:
        runs = []
        for _ in range(REPEATS):
            elapsed = time_one_run(
                computer=ComputerClass(),
                backend=backend,
                rewritten_query=rewritten,
                expected_answer=("a",),
                candidates=all_tuples,
                endogenous_tuples=all_tuples,
            )
            runs.append(elapsed)
        best = min(runs)
        results[ComputerClass.__name__] = best
        print(f"{ComputerClass.__name__:<30} {best * 1000:>8.2f} ms (best of {REPEATS})")

    print("-" * 60)
    naive_time = results["NaiveComputer"]
    et_time = results["EarlyTerminationComputer"]
    speedup = naive_time / et_time
    print(f"Early Termination speedup over Naive: {speedup:.2f}×")

    backend.close()


if __name__ == "__main__":
    main()