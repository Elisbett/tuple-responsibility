"""Level 3: Brute-force responsibility computation with caching.

Builds on Level 2 (early termination) by caching the result of every
"is the expected answer present given this set of disabled tuples?"
query. Because the same frozenset of disabled tuples often arises
multiple times during enumeration — both within one candidate's
search and across candidates — caching avoids redundant SQL execution.

Cache key: (frozenset of TupleIds currently disabled, expected_answer).
Cache value: bool — whether the expected answer is still in the result.

The semantics are identical to Levels 1 and 2; only the runtime differs.
This is verified by tests/test_level_equivalence.py.

A cache can grow large in principle, but in practice:
  - For small candidate sets (|C| <= 20) and small endogenous sets,
    the number of distinct subsets visited is bounded by 2^|D_n|.
  - We use a plain dict rather than lru_cache so that the cache can be
    inspected/cleared explicitly for benchmarking (see __init__ args).
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


class CachedComputer(ResponsibilityComputer):
    """Level 3: early termination + memoization of query results."""

    def __init__(self) -> None:
        self._cache: dict[frozenset[TupleId], bool] = {}
        self._cache_hits = 0
        self._cache_misses = 0

    # --- public statistics (useful for §6 reporting) -------------------

    @property
    def cache_hits(self) -> int:
        """Number of times a query result was reused from the cache."""
        return self._cache_hits

    @property
    def cache_misses(self) -> int:
        """Number of times a query actually had to be evaluated."""
        return self._cache_misses

    def clear_cache(self) -> None:
        """Reset the cache and statistics (use between independent runs)."""
        self._cache.clear()
        self._cache_hits = 0
        self._cache_misses = 0

    # --- core compute() ------------------------------------------------

    def compute(
        self,
        backend: SQLiteBackend,
        rewritten_query: str,
        expected_answer: tuple,
        candidates: Iterable[TupleId],
        endogenous_tuples: Iterable[TupleId],
    ) -> ResponsibilityRanking:
        # The cache is meaningful only for a single (query, expected_answer)
        # pair, so we clear it on each call to be safe.
        self.clear_cache()

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

    # --- contingency search with caching -------------------------------

    def _find_min_contingency_size(
        self,
        backend: SQLiteBackend,
        rewritten_query: str,
        expected_answer: tuple,
        candidate: TupleId,
        endogenous_tuples: list[TupleId],
    ) -> int | None:
        r"""Find the smallest contingency size for `candidate`, or None.

        Same structure as EarlyTerminationComputer: enumerate subsets
        of (D_n \ {candidate}) by increasing size; return on first hit.
        The difference is the underlying queries are answered via the
        cache when possible.
        """
        other_tuples = [t for t in endogenous_tuples if t != candidate]

        for size in range(len(other_tuples) + 1):
            for gamma in combinations(other_tuples, size):
                if self._is_valid_contingency(
                    backend=backend,
                    rewritten_query=rewritten_query,
                    expected_answer=expected_answer,
                    candidate=candidate,
                    gamma=gamma,
                ):
                    return size

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

        Both is_answer queries are routed through the cache.
        """
        gamma_set = frozenset(gamma)
        gamma_plus_candidate = gamma_set | {candidate}

        # Condition 1: D \ Γ |= q(r)
        if not self._cached_is_answer(
            backend=backend,
            rewritten_query=rewritten_query,
            expected_answer=expected_answer,
            disabled_set=gamma_set,
        ):
            return False

        # Condition 2: D \ Γ \ {t} |/= q(r)
        if self._cached_is_answer(
            backend=backend,
            rewritten_query=rewritten_query,
            expected_answer=expected_answer,
            disabled_set=gamma_plus_candidate,
        ):
            return False

        return True

    def _cached_is_answer(
        self,
        backend: SQLiteBackend,
        rewritten_query: str,
        expected_answer: tuple,
        disabled_set: frozenset[TupleId],
    ) -> bool:
        """Look up or compute is_answer for the given disabled set."""
        if disabled_set in self._cache:
            self._cache_hits += 1
            return self._cache[disabled_set]

        # Set the backend state to match disabled_set exactly, then query.
        backend.enable_all()
        backend.disable_set(disabled_set)

        result = backend.is_answer(rewritten_query, expected_answer)

        self._cache[disabled_set] = result
        self._cache_misses += 1
        return result