"""Abstract base class for responsibility computation.

All four optimisation levels (naive, early termination, cached, parallel)
inherit from ResponsibilityComputer and override the compute() method.
This strategy-pattern setup allows experiments to swap implementations
without changing any surrounding code.

Each compute() call is fully self-contained: the computer opens its own
connection to the database, performs the contingency search, and closes
the connection. No long-lived backend objects are shared across levels,
which keeps each measurement isolated and avoids contention on the SQLite
file when multi-process levels (Level 4) are added later.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Iterable

from src.core.types import (
    ComputeResult,
    ResponsibilityRanking,
    ResponsibilityResult,
    TupleId,
)


class ResponsibilityComputer(ABC):
    """Base class for computing causal responsibility scores.

    Subclasses implement the compute() method with progressively more
    aggressive optimisations. The signature is identical across all
    levels, so an experiment can call any level interchangeably:

        for ComputerClass in [NaiveComputer, EarlyTerminationComputer, ...]:
            ranking = ComputerClass().compute(
                db_path, rewritten_query, expected_answer, candidates, ...
            )
    """

    @abstractmethod
    def compute(
        self,
        db_path: str | Path,
        rewritten_query: str,
        expected_answer: tuple,
        candidates: Iterable[TupleId],
        endogenous_tuples: Iterable[TupleId],
    ) -> ComputeResult:
        """Compute responsibility scores for each candidate tuple."""
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