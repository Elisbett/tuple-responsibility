"""Correctness tests for NaiveComputer on the smoke-test dataset.

The expected responsibility scores are computed by hand from
Definition 2.3 of Meliou et al. (2010) on the small R/S schema.

See the docstring of create_synthetic_dataset.py for the exact
tuples and the structure of the query.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from src.core.naive import NaiveComputer
from src.core.types import TupleId
from src.db.query_rewriter import rewrite_query
from src.db.sqlite_backend import SQLiteBackend

DB_PATH = Path("data/synthetic/smoke_test.db")


@pytest.fixture
def db_path() -> Path:
    """Ensure the smoke-test database exists and has disabled columns.

    Returns the path; each computer opens its own connection internally.
    """
    assert DB_PATH.exists(), (
        f"Run `python create_synthetic_dataset.py` first to create {DB_PATH}"
    )
    # One-time setup: add the _disabled columns if not yet present.
    setup = SQLiteBackend(DB_PATH)
    setup.add_disabled_columns()
    setup.enable_all()
    setup.close()
    return DB_PATH


def test_naive_computes_expected_scores(db_path: Path) -> None:
    """Verify ρ-scores for all tuples on the smoke-test dataset.

    Schema:
        R = { (a,b)#1, (a,f)#2, (c,b)#3, (c,g)#4 }
        S = { (b)#1, (f)#2, (h)#3 }

    Query: SELECT DISTINCT x FROM R, S WHERE R.y = S.y
    Answer of interest: ("a",)

    Expected scores (computed by hand):
        R(a,b)#1 → 0.5  (contingency {R(a,f)#2})
        R(a,f)#2 → 0.5  (contingency {R(a,b)#1})
        S(b)#1   → 0.5  (contingency {S(f)#2})
        S(f)#2   → 0.5  (contingency {S(b)#1})
        R(c,b)#3 → 0.0  (not relevant to answer "a")
        R(c,g)#4 → 0.0  (not relevant)
        S(h)#3   → 0.0  (not relevant)
    """
    sql = "SELECT DISTINCT x FROM R, S WHERE R.y = S.y"
    rewritten = rewrite_query(sql, {"R": "R", "S": "S"})

    # All seven tuples are candidates and endogenous.
    all_tuples = [
        TupleId("R", 1),
        TupleId("R", 2),
        TupleId("R", 3),
        TupleId("R", 4),
        TupleId("S", 1),
        TupleId("S", 2),
        TupleId("S", 3),
    ]

    computer = NaiveComputer()
    result = computer.compute(
        db_path=db_path,
        rewritten_query=rewritten,
        expected_answer=("a",),
        candidates=all_tuples,
        endogenous_tuples=all_tuples,
    )
    ranking = result.ranking

    # Build a dict for easy lookup.
    scores = {r.tuple_id: r.responsibility for r in ranking.results}

    # Tuples relevant to "a" — each needs one other tuple removed.
    assert scores[TupleId("R", 1)] == pytest.approx(0.5)
    assert scores[TupleId("R", 2)] == pytest.approx(0.5)
    assert scores[TupleId("S", 1)] == pytest.approx(0.5)
    assert scores[TupleId("S", 2)] == pytest.approx(0.5)

    # Tuples irrelevant to "a".
    assert scores[TupleId("R", 3)] == 0.0
    assert scores[TupleId("R", 4)] == 0.0
    assert scores[TupleId("S", 3)] == 0.0


def test_naive_min_contingency_sizes(db_path: Path) -> None:
    """Verify min_contingency_size values explicitly."""
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

    computer = NaiveComputer()
    result = computer.compute(
        db_path=db_path,
        rewritten_query=rewritten,
        expected_answer=("a",),
        candidates=all_tuples,
        endogenous_tuples=all_tuples,
    )
    ranking = result.ranking

    sizes = {r.tuple_id: r.min_contingency_size for r in ranking.results}

    # Actual causes have size 1 (need one other tuple disabled).
    assert sizes[TupleId("R", 1)] == 1
    assert sizes[TupleId("R", 2)] == 1
    assert sizes[TupleId("S", 1)] == 1
    assert sizes[TupleId("S", 2)] == 1

    # Non-causes return None.
    assert sizes[TupleId("R", 3)] is None
    assert sizes[TupleId("R", 4)] is None
    assert sizes[TupleId("S", 3)] is None


def test_ranking_sorted_by_responsibility(db_path: Path) -> None:
    """The sorted_by_responsibility method returns descending order."""
    sql = "SELECT DISTINCT x FROM R, S WHERE R.y = S.y"
    rewritten = rewrite_query(sql, {"R": "R", "S": "S"})

    all_tuples = [
        TupleId("R", 1),
        TupleId("R", 2),
        TupleId("R", 3),
        TupleId("S", 1),
    ]

    computer = NaiveComputer()
    result = computer.compute(
        db_path=db_path,
        rewritten_query=rewritten,
        expected_answer=("a",),
        candidates=all_tuples,
        endogenous_tuples=all_tuples,
    )
    ranking = result.ranking

    sorted_results = ranking.sorted_by_responsibility()
    scores = [r.responsibility for r in sorted_results]

    # Verify descending order.
    assert scores == sorted(scores, reverse=True)