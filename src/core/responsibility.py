"""Abstract base class for responsibility computation.

All four optimisation levels (naive, early termination, cached, parallel)
inherit from ResponsibilityComputer and override the compute() method.
This strategy-pattern setup allows experiments to swap implementations
without changing any surrounding code.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterable

from src.core.types import (
    ResponsibilityRanking,
    ResponsibilityResult,
    TupleId,
)
from src.db.sqlite_backend import SQLiteBackend


class ResponsibilityComputer(ABC):
    """Base class for computing causal responsibility scores.

    Subclasses implement the compute() method with progressively more
    aggressive optimisations. The signature is identical across all
    levels, so an experiment can call any level interchangeably:

        for ComputerClass in [NaiveComputer, EarlyTerminationComputer, ...]:
            ranking = ComputerClass().compute(
                backend, rewritten_query, expected_answer, candidates
            )
    """

    @abstractmethod
    def compute(
        self,
        backend: SQLiteBackend,
        rewritten_query: str,
        expected_answer: tuple,
        candidates: Iterable[TupleId],
        endogenous_tuples: Iterable[TupleId],
    ) -> ResponsibilityRanking:
        """Compute responsibility scores for each candidate tuple.

        Parameters
        ----------
        backend : SQLiteBackend
            An open connection to the database, with disabled columns
            already added.
        rewritten_query : str
            The user query, already rewritten to include _disabled=0
            filters on every table (see query_rewriter.rewrite_query).
        expected_answer : tuple
            The query answer whose responsibility is being analysed.
            For example, ("Musical",) for a single-column query result.
        candidates : Iterable[TupleId]
            The set C of tuples to evaluate. Each will receive a
            responsibility score in [0, 1].
        endogenous_tuples : Iterable[TupleId]
            The endogenous tuples D_n. Contingencies are drawn from
            this set (excluding the candidate currently being evaluated).

        Returns
        -------
        ResponsibilityRanking
            A ranking containing one ResponsibilityResult per candidate.
        """
        raise NotImplementedError

    @staticmethod
    def responsibility_from_contingency_size(
        min_size: int | None,
    ) -> float:
        """Convert a minimum contingency size to a responsibility score.

        - If min_size is None: tuple is not an actual cause, ρ = 0.
        - If min_size is 0:    tuple is a counterfactual cause, ρ = 1.
        - Otherwise:           ρ = 1 / (1 + min_size).
        """
        if min_size is None:
            return 0.0
        return 1.0 / (1.0 + min_size)