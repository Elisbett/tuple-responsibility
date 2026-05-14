r"""Level 1: Naive brute-force responsibility computation.

This is the most direct translation of Definition 2.3 of
Meliou et al. (2010) "The Complexity of Causality and Responsibility
for Query Answers and non-Answers":

    ρ(t) = 1 / (1 + min |Γ|)

where Γ ranges over all contingency sets for t — i.e. sets of
endogenous tuples that, after removal, make t a counterfactual cause
for the answer r.

The naive algorithm explicitly enumerates ALL subsets Γ of (D_n \ {t}),
checks each one, and records the minimum size that works. No early
termination, no caching, no parallelism. This is the slowest possible
implementation; subsequent levels speed it up.

Complexity: O(|C| * 2^|D_n| * Q), where Q is the cost of one query
evaluation. Tractable only for very small D_n (say, |D_n| <= 12).
"""
from __future__ import annotations

import time
from itertools import combinations
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

class NaiveComputer(ResponsibilityComputer):
    """Level 1: brute-force enumeration of all contingency subsets."""

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

        ranking = ResponsibilityRanking()

        # ---- setup phase ----
        t0 = time.perf_counter()
        backend = SQLiteBackend(db_path)
        backend.add_disabled_columns()  # idempotent
        setup_time = time.perf_counter() - t0

        # ---- algorithm phase ----
        t0 = time.perf_counter()
        try:
            for candidate in candidates_list:
                min_size = self._find_min_contingency_size(
                    backend=backend,
                    rewritten_query=rewritten_query,
                    expected_answer=expected_answer,
                    candidate=candidate,
                    endogenous_tuples=endogenous_list,
                )
                score = self.responsibility_from_contingency_size(min_size)

                ranking.results.append(
                    ResponsibilityResult(
                        tuple_id=candidate,
                        responsibility=score,
                        min_contingency_size=min_size,
                    )
                )
            algorithm_time = time.perf_counter() - t0
        finally:
            # ---- teardown phase ----
            t0 = time.perf_counter()
            backend.close()
            teardown_time = time.perf_counter() - t0

        return ComputeResult(
            ranking=ranking,
            setup_time=setup_time,
            algorithm_time=algorithm_time,
            teardown_time=teardown_time,
        )

    def _find_min_contingency_size(
        self,
        backend: SQLiteBackend,
        rewritten_query: str,
        expected_answer: tuple,
        candidate: TupleId,
        endogenous_tuples: list[TupleId],
    ) -> int | None:
        r"""Find the smallest contingency size for `candidate`, or None.

        Returns the size of the smallest set Γ ⊆ (D_n \\ {candidate}) such
        that:
            1. removing Γ alone does NOT eliminate the expected answer;
            2. removing Γ ∪ {candidate} DOES eliminate it.

        Returns None if no such Γ exists (i.e. candidate is not an actual
        cause for the expected answer).

        Naive enumeration semantics
        ---------------------------
        This implementation enumerates subsets by increasing size and
        continues the outer loop through ALL sizes (0..|D_n|-1) rather
        than stopping at the first valid contingency. Because the loop
        order is monotonically increasing, the first valid contingency
        size found is necessarily the minimum, so storing only the
        first hit (via `if best_size is None`) is correct.

        We deliberately do NOT break out of the loop after finding the
        first hit: that would constitute the early-termination
        optimisation of Level 2. Keeping the outer loop intact here
        preserves the textbook "explore every subset" semantics of the
        naive algorithm, at the cost of redundant work that Level 2
        eliminates. This makes the comparison between Levels 1 and 2
        clean: the only difference between them is whether the loop
        terminates early.
        """
        # All other endogenous tuples (excluding the candidate itself).
        other_tuples = [t for t in endogenous_tuples if t != candidate]

        # Reset state before testing this candidate.
        backend.enable_all()

        # Enumerate contingency sizes from 0 (counterfactual) upwards.
        # For naive, we go through ALL sizes; no early termination.
        best_size: int | None = None

        for size in range(len(other_tuples) + 1):
            for gamma in combinations(other_tuples, size):
                if self.is_valid_contingency(
                    backend=backend,
                    rewritten_query=rewritten_query,
                    expected_answer=expected_answer,
                    candidate=candidate,
                    gamma=gamma,
                ):
                    # Record the size of the first valid contingency
                    # of this minimum size, but keep iterating to honour
                    # the "naive" contract (no early termination).
                    if best_size is None:
                        best_size = size

        return best_size