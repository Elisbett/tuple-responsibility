"""Equivalence tests: all optimisation levels must produce identical results.

The optimisations applied at Levels 2, 3, and 4 do not change the
semantics of responsibility computation — only the runtime. Therefore,
running any level on the same input must produce identical scores and
identical min_contingency_size values for every candidate.

A failure here means an optimisation has introduced a bug.

For now we test Level 1 vs Level 2. As Levels 3 and 4 are added, they
will be appended to the COMPUTERS list and tested by the same harness.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from src.core.cached import CachedComputer
from src.core.early_termination import EarlyTerminationComputer
from src.core.naive import NaiveComputer
from src.core.responsibility import ResponsibilityComputer
from src.core.types import TupleId
from src.db.query_rewriter import rewrite_query
from src.db.sqlite_backend import SQLiteBackend

DB_PATH = Path("data/synthetic/smoke_test.db")

# All implemented computers. Add new levels here as they are created.
COMPUTERS: list[type[ResponsibilityComputer]] = [
    NaiveComputer,
    EarlyTerminationComputer,
    CachedComputer,
]


@pytest.fixture
def backend() -> SQLiteBackend:
    assert DB_PATH.exists(), (
        f"Run `python create_synthetic_dataset.py` first to create {DB_PATH}"
    )
    b = SQLiteBackend(DB_PATH)
    b.add_disabled_columns()
    b.enable_all()
    yield b
    b.close()


@pytest.fixture
def query_setup():
    """The smoke-test query and the seven endogenous tuples."""
    sql = "SELECT DISTINCT x FROM R, S WHERE R.y = S.y"
    rewritten = rewrite_query(sql, {"R": "R", "S": "S"})
    all_tuples = [
        TupleId("R", 1),
        TupleId("R", 2),
        TupleId("R", 3),
        TupleId("R", 4),
        TupleId("S", 1),
        TupleId("S", 2),
        TupleId("S", 3),
    ]
    return rewritten, all_tuples


def test_all_levels_produce_identical_responsibility_scores(
    backend: SQLiteBackend, query_setup
) -> None:
    """Every Level should return the same ρ-score for every candidate."""
    rewritten, all_tuples = query_setup

    # Compute the reference ranking with the first computer (Naive).
    reference_computer = COMPUTERS[0]()
    backend.enable_all()
    reference_ranking = reference_computer.compute(
        backend=backend,
        rewritten_query=rewritten,
        expected_answer=("a",),
        candidates=all_tuples,
        endogenous_tuples=all_tuples,
    )
    reference_scores = {
        r.tuple_id: r.responsibility for r in reference_ranking.results
    }

    # Compare every other computer against the reference.
    for ComputerClass in COMPUTERS[1:]:
        backend.enable_all()
        ranking = ComputerClass().compute(
            backend=backend,
            rewritten_query=rewritten,
            expected_answer=("a",),
            candidates=all_tuples,
            endogenous_tuples=all_tuples,
        )
        scores = {r.tuple_id: r.responsibility for r in ranking.results}

        for tid in reference_scores:
            assert scores[tid] == pytest.approx(reference_scores[tid]), (
                f"{ComputerClass.__name__} differs from "
                f"{COMPUTERS[0].__name__} on tuple {tid}: "
                f"{scores[tid]} vs {reference_scores[tid]}"
            )


def test_all_levels_produce_identical_contingency_sizes(
    backend: SQLiteBackend, query_setup
) -> None:
    """Every Level should return the same min_contingency_size."""
    rewritten, all_tuples = query_setup

    reference_computer = COMPUTERS[0]()
    backend.enable_all()
    reference_ranking = reference_computer.compute(
        backend=backend,
        rewritten_query=rewritten,
        expected_answer=("a",),
        candidates=all_tuples,
        endogenous_tuples=all_tuples,
    )
    reference_sizes = {
        r.tuple_id: r.min_contingency_size
        for r in reference_ranking.results
    }

    for ComputerClass in COMPUTERS[1:]:
        backend.enable_all()
        ranking = ComputerClass().compute(
            backend=backend,
            rewritten_query=rewritten,
            expected_answer=("a",),
            candidates=all_tuples,
            endogenous_tuples=all_tuples,
        )
        sizes = {
            r.tuple_id: r.min_contingency_size for r in ranking.results
        }

        for tid in reference_sizes:
            assert sizes[tid] == reference_sizes[tid], (
                f"{ComputerClass.__name__} differs from "
                f"{COMPUTERS[0].__name__} on tuple {tid}: "
                f"size {sizes[tid]} vs {reference_sizes[tid]}"
            )