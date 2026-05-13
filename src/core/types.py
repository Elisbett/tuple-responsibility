"""Domain types for causal responsibility computation.

These dataclasses represent the basic entities the algorithm works with:
a database tuple, a candidate set, and a computation result.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class TupleId:
    """Identifier of a single tuple in the database.

    A tuple is identified by the relation name and its primary key value.
    The pair (relation, key) is unique across the database.
    """
    relation: str
    key: int

    def __str__(self) -> str:
        return f"{self.relation}#{self.key}"


@dataclass
class ResponsibilityResult:
    """Result of computing responsibility for a single candidate tuple.

    Attributes
    ----------
    tuple_id : TupleId
        The tuple whose responsibility was computed.
    responsibility : float
        The numerical responsibility score in [0, 1].
        - 1.0 means the tuple is a counterfactual cause.
        - Strictly between 0 and 1 means it is an actual cause.
        - 0.0 means it is not an actual cause at all.
    min_contingency_size : int | None
        The size of the smallest contingency that makes the tuple
        counterfactual. None if the tuple is not an actual cause.
    """
    tuple_id: TupleId
    responsibility: float
    min_contingency_size: int | None


@dataclass
class ResponsibilityRanking:
    """Ranked list of responsibility results for a candidate set."""
    results: list[ResponsibilityResult] = field(default_factory=list)

    def sorted_by_responsibility(self) -> list[ResponsibilityResult]:
        """Return results sorted by descending responsibility."""
        return sorted(
            self.results,
            key=lambda r: r.responsibility,
            reverse=True,
        )
    
@dataclass
class ComputeResult:
    """Result of a single compute() call: the ranking plus timing breakdown."""
    ranking: ResponsibilityRanking
    setup_time: float
    algorithm_time: float
    teardown_time: float

    @property
    def total_time(self) -> float:
        return self.setup_time + self.algorithm_time + self.teardown_time