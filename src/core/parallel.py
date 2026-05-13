"""Level 4: parallelisation of the per-candidate contingency search.

Each candidate's contingency search is independent of the others —
no shared state, no ordering constraint — so the candidate loop is
embarrassingly parallel. This level distributes that loop across
worker processes via multiprocessing.Pool.

Implementation notes
--------------------
- Each worker opens its own SQLiteBackend on the same database file.
  Concurrent reads on a single SQLite file are safe; concurrent writes
  (UPDATE _disabled) are serialised by SQLite's per-database lock,
  but since each worker handles only its own candidate, there is no
  contention in practice.
- The worker function is at module level (not a method) so it can be
  pickled by multiprocessing on Windows.
- Caching is intentionally NOT applied inside workers. Sharing a cache
  across processes via Manager() would dominate the runtime; isolating
  one optimisation per level makes the benchmarks in §6 easier to
  interpret.

Timing semantics
----------------
Setup_time here measures Pool startup, not just connection opening.
On Windows this can be 100-500 ms because each worker fully re-imports
the project modules ("spawn" start method). On large workloads this
becomes negligible; on tiny ones (like smoke_test.db) it dominates.
"""
from __future__ import annotations

import os
import time
from itertools import combinations
from multiprocessing import Pool
from pathlib import Path
from typing import Iterable

from src.core.responsibility import ResponsibilityComputer
from src.core.types import (
    ComputeResult,
    ResponsibilityRanking,
    ResponsibilityResult,
    TupleId,
)
from src.db.sqlite_backend import SQLiteBackend


# ---------------------------------------------------------------------
# Module-level worker function (picklable for multiprocessing).
# ---------------------------------------------------------------------

def _worker_one_candidate(args: tuple) -> ResponsibilityResult:
    """Compute responsibility for ONE candidate, in a worker process.

    Each worker process opens its own connection, performs the
    early-termination search for its assigned candidate, and closes.
    Returns a ResponsibilityResult ready to be appended to the ranking
    in the parent process.
    """
    (
        db_path,
        rewritten_query,
        expected_answer,
        candidate,
        endogenous_tuples,
    ) = args

    backend = SQLiteBackend(db_path)
    try:
        min_size = _find_min_contingency_size(
            backend=backend,
            rewritten_query=rewritten_query,
            expected_answer=expected_answer,
            candidate=candidate,
            endogenous_tuples=endogenous_tuples,
        )
    finally:
        backend.close()

    score = ResponsibilityComputer.responsibility_from_contingency_size(
        min_size
    )
    return ResponsibilityResult(
        tuple_id=candidate,
        responsibility=score,
        min_contingency_size=min_size,
    )


def _find_min_contingency_size(
    backend: SQLiteBackend,
    rewritten_query: str,
    expected_answer: tuple,
    candidate: TupleId,
    endogenous_tuples: list[TupleId],
) -> int | None:
    """Early-termination contingency search, replicated at module scope
    so it is importable by worker processes without instantiating a
    computer class.
    """
    other_tuples = [t for t in endogenous_tuples if t != candidate]

    for size in range(len(other_tuples) + 1):
        for gamma in combinations(other_tuples, size):
            backend.enable_all()
            backend.disable_set(gamma)

            if not backend.is_answer(rewritten_query, expected_answer):
                continue

            backend.disable(candidate)
            if backend.is_answer(rewritten_query, expected_answer):
                continue

            return size

    return None


# ---------------------------------------------------------------------
# Public class
# ---------------------------------------------------------------------

class ParallelComputer(ResponsibilityComputer):
    """Level 4: per-candidate parallelisation via multiprocessing.Pool."""

    def __init__(self, num_workers: int | None = None) -> None:
        """Create a parallel computer.

        Parameters
        ----------
        num_workers : int | None
            Number of worker processes. Defaults to os.cpu_count().
        """
        self.num_workers = num_workers or os.cpu_count() or 1

    def compute(
        self,
        db_path: str | Path,
        rewritten_query: str,
        expected_answer: tuple,
        candidates: Iterable[TupleId],
        endogenous_tuples: Iterable[TupleId],
    ) -> ComputeResult:
        candidates_list = list(candidates)
        endogenous_list = list(endogenous_tuples)

        # Pack arguments for each worker invocation.
        work_items = [
            (
                str(db_path),
                rewritten_query,
                expected_answer,
                candidate,
                endogenous_list,
            )
            for candidate in candidates_list
        ]

        # ---- setup phase: ensure disabled columns exist, then start Pool.
        t0 = time.perf_counter()
        prep = SQLiteBackend(db_path)
        prep.add_disabled_columns()
        prep.close()
        pool = Pool(processes=self.num_workers)
        setup_time = time.perf_counter() - t0

        # ---- algorithm phase: parallel candidate processing.
        t0 = time.perf_counter()
        try:
            results = pool.map(_worker_one_candidate, work_items)
            algorithm_time = time.perf_counter() - t0
        finally:
            # ---- teardown phase: shut down workers.
            t0 = time.perf_counter()
            pool.close()
            pool.join()
            teardown_time = time.perf_counter() - t0

        ranking = ResponsibilityRanking(results=list(results))
        return ComputeResult(
            ranking=ranking,
            setup_time=setup_time,
            algorithm_time=algorithm_time,
            teardown_time=teardown_time,
        )