"""Level 2: Brute-force responsibility computation with early termination.

This level applies a simple but powerful optimisation to the Naive
algorithm of src/core/naive.py: as soon as a valid contingency of
size k is found for a candidate tuple, the search stops. Since
subsets are enumerated by increasing size, the first valid contingency
found is necessarily the smallest one.

Best case: when a tuple is a counterfactual cause (size-0 contingency
works), only 1 query is needed instead of 2^|D_n|.

Worst case (tuple not an actual cause): all subsets must still be
examined to confirm no contingency exists, matching Level 1's cost.

The semantics — the responsibility scores returned — are IDENTICAL to
Level 1. Only the runtime differs. This is verified by
tests/test_level_equivalence.py (added later).
"""
from __future__ import annotations

from itertools import combinations
from typing import Iterable

from src.core.responsibility import ResponsibilityComputer
from src.core.types import (
    ResponsibilityRanking,
    ResponsibilityResult,
    TupleId,
)
from src.db.sqlite_backend import SQLiteBackend


class EarlyTerminationComputer(ResponsibilityComputer):
    """Level 2: brute-force enumeration with early termination."""

    def compute(
        self,
        backend: SQLiteBackend,
        rewritten_query: str,
        expected_answer: tuple,
        candidates: Iterable[TupleId],
        endogenous_tuples: Iterable[TupleId],
    ) -> ResponsibilityRanking:
        candidates_list = list(candidates)
        endogenous_list = list(endogenous_tuples)

        ranking = ResponsibilityRanking()

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

        return ranking

    def _find_min_contingency_size(
        self,
        backend: SQLiteBackend,
        rewritten_query: str,
        expected_answer: tuple,
        candidate: TupleId,
        endogenous_tuples: list[TupleId],
    ) -> int | None:
        r"""Find the smallest contingency size for `candidate`, or None.

        Iterates over contingency sizes in increasing order. As soon as
        a valid contingency is found, returns its size immediately. This
        guarantees the result is the minimum (since smaller sizes were
        already exhausted) while skipping all larger sizes.
        """
        other_tuples = [t for t in endogenous_tuples if t != candidate]

        backend.enable_all()

        for size in range(len(other_tuples) + 1):
            for gamma in combinations(other_tuples, size):
                if self._is_valid_contingency(
                    backend=backend,
                    rewritten_query=rewritten_query,
                    expected_answer=expected_answer,
                    candidate=candidate,
                    gamma=gamma,
                ):
                    # Early termination: first valid contingency = minimum.
                    return size

        # No contingency exists at any size: not an actual cause.
        return None

    def _is_valid_contingency(
        self,
        backend: SQLiteBackend,
        rewritten_query: str,
        expected_answer: tuple,
        candidate: TupleId,
        gamma: tuple[TupleId, ...],
    ) -> bool:
        """Check whether `gamma` is a valid contingency for `candidate`.

        Same conditions as in NaiveComputer (Def. 2.1 of Meliou et al. 2010).
        """
        backend.enable_all()
        backend.disable_set(gamma)

        if not backend.is_answer(rewritten_query, expected_answer):
            return False

        backend.disable(candidate)

        if backend.is_answer(rewritten_query, expected_answer):
            return False

        return True