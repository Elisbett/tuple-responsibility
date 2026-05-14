"""Level 4: parallelisation of the per-candidate contingency search.

Each candidate's contingency search is independent of the others — no
shared state, no ordering constraint — so the candidate loop is
embarrassingly parallel. This level distributes that loop across worker
processes via multiprocessing.Pool.

Implementation note: each worker is given its own copy of the database
file. SQLite serialises concurrent writers on a single file, and the
high write rate of the contingency search (millions of UPDATEs on
_disabled per candidate) makes shared-file access impractical even
with WAL mode. Copying the file once before starting the pool removes
all inter-worker contention at the cost of a few megabytes of disk and
one fast file copy per worker; given that the algorithmic work is in
seconds-to-minutes, this overhead is negligible.

The worker function is at module level so it can be pickled by
multiprocessing on Windows.

Timing semantics: setup_time here includes Pool startup and the per-worker
file copy, not just connection opening. On Windows this can be 100-500 ms
because each worker fully re-imports the project modules ("spawn" start
method). On small workloads this dominates; on large ones it is negligible.
"""
from __future__ import annotations

import os
import shutil
import tempfile
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

    Each worker process opens its own (already-copied) database file,
    performs the early-termination search for its assigned candidate,
    and closes the file. No two workers ever touch the same .db file.
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
    """Level 4: per-candidate parallelisation via multiprocessing.Pool.

    Each worker process operates on its own copy of the database file
    to avoid lock contention. Copies are placed in a temporary directory
    and deleted automatically when compute() returns.
    """

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

        # ---- setup phase: make per-worker DB copies, start the Pool.
        t0 = time.perf_counter()

        # Ensure disabled columns exist on the master file first.
        master = SQLiteBackend(db_path)
        master.add_disabled_columns()
        master.close()

        # Create per-worker copies in a fresh temp directory.
        tmpdir = Path(tempfile.mkdtemp(prefix="resp_parallel_"))
        worker_db_paths = []
        for i in range(self.num_workers):
            copy_path = tmpdir / f"worker_{i}.db"
            shutil.copyfile(str(db_path), str(copy_path))
            worker_db_paths.append(copy_path)

        # Round-robin assign each candidate to a worker file.
        # multiprocessing.Pool's chunksize handles scheduling; the file
        # assignment just guarantees each candidate has a private file.
        work_items = []
        for idx, candidate in enumerate(candidates_list):
            assigned_db = worker_db_paths[idx % self.num_workers]
            work_items.append(
                (
                    str(assigned_db),
                    rewritten_query,
                    expected_answer,
                    candidate,
                    endogenous_list,
                )
            )

        pool = Pool(processes=self.num_workers)
        setup_time = time.perf_counter() - t0

        # ---- algorithm phase: parallel candidate processing.
        t0 = time.perf_counter()
        try:
            results = pool.map(_worker_one_candidate, work_items)
            algorithm_time = time.perf_counter() - t0
        finally:
            # ---- teardown phase: stop workers, clean up temp files.
            t0 = time.perf_counter()
            pool.close()
            pool.join()
            shutil.rmtree(tmpdir, ignore_errors=True)
            teardown_time = time.perf_counter() - t0

        ranking = ResponsibilityRanking(results=list(results))
        return ComputeResult(
            ranking=ranking,
            setup_time=setup_time,
            algorithm_time=algorithm_time,
            teardown_time=teardown_time,
        )