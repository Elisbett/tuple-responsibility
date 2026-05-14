"""Smoke tests for SQLiteBackend and query_rewriter.

These tests verify the disabled-tid trick mechanics on the small
synthetic dataset created by create_synthetic_dataset.py.

Before running, generate the dataset:
    python create_synthetic_dataset.py

Then run the tests:
    pytest tests/test_sqlite_backend.py -v
"""
from __future__ import annotations

from pathlib import Path

import pytest

from src.core.types import TupleId
from src.db.query_rewriter import rewrite_query
from src.db.sqlite_backend import SQLiteBackend

DB_PATH = Path("data/synthetic/smoke_test.db")


@pytest.fixture
def backend() -> SQLiteBackend:
    """Open a fresh backend for each test, with disabled columns added."""
    assert DB_PATH.exists(), (
        f"Run `python create_synthetic_dataset.py` first to create {DB_PATH}"
    )
    b = SQLiteBackend(DB_PATH)
    b.add_disabled_columns()
    b.enable_all()  # reset state between tests
    yield b
    b.close()


def test_disabled_column_added_to_all_tables(backend: SQLiteBackend) -> None:
    """Every table should have the _disabled column after setup."""
    for table in backend.list_tables():
        assert backend.has_disabled_column(table)


def test_initial_query_returns_expected_answers(
    backend: SQLiteBackend,
) -> None:
    """With all tuples enabled, the query should return {a, c}."""
    sql = "SELECT DISTINCT x FROM R, S WHERE R.y = S.y"
    rewritten = rewrite_query(sql, {"R": "R", "S": "S"})

    rows = backend.execute_query(rewritten)
    answers = {row[0] for row in rows}

    assert answers == {"a", "c"}


def test_disabling_tuple_removes_answer(backend: SQLiteBackend) -> None:
    """Disabling R(a,b) AND R(a,f) should remove 'a' from the answer."""
    # R(a,b) is the tuple with rowid=1 in R (first INSERT).
    # R(a,f) is rowid=2.
    backend.disable(TupleId("R", 1))
    backend.disable(TupleId("R", 2))

    sql = "SELECT DISTINCT x FROM R, S WHERE R.y = S.y"
    rewritten = rewrite_query(sql, {"R": "R", "S": "S"})

    rows = backend.execute_query(rewritten)
    answers = {row[0] for row in rows}

    assert "a" not in answers
    assert "c" in answers  # c is still derivable via R(c,b) and S(b)


def test_enable_restores_answer(backend: SQLiteBackend) -> None:
    """After disabling and re-enabling, the original answer should return."""
    backend.disable(TupleId("R", 1))
    backend.disable(TupleId("R", 2))
    backend.enable(TupleId("R", 1))
    backend.enable(TupleId("R", 2))

    sql = "SELECT DISTINCT x FROM R, S WHERE R.y = S.y"
    rewritten = rewrite_query(sql, {"R": "R", "S": "S"})

    rows = backend.execute_query(rewritten)
    answers = {row[0] for row in rows}

    assert answers == {"a", "c"}


def test_is_answer_returns_true_when_present(backend: SQLiteBackend) -> None:
    """is_answer should return True when expected row is in result."""
    sql = "SELECT DISTINCT x FROM R, S WHERE R.y = S.y"
    rewritten = rewrite_query(sql, {"R": "R", "S": "S"})

    assert backend.is_answer(rewritten, ("a",)) is True
    assert backend.is_answer(rewritten, ("c",)) is True
    assert backend.is_answer(rewritten, ("z",)) is False


def test_is_answer_returns_false_after_disabling(
    backend: SQLiteBackend,
) -> None:
    """After disabling supporting tuples, expected row is no longer answer."""
    backend.disable(TupleId("R", 1))
    backend.disable(TupleId("R", 2))

    sql = "SELECT DISTINCT x FROM R, S WHERE R.y = S.y"
    rewritten = rewrite_query(sql, {"R": "R", "S": "S"})

    assert backend.is_answer(rewritten, ("a",)) is False

# ----------------------------------------------------------------------
# Tests for find_tuple / find_tuples helpers
# ----------------------------------------------------------------------

def test_find_tuple_returns_correct_tupleid(backend) -> None:
    """find_tuple resolves a row identified by column values to its rowid."""
    # smoke_test.db has R(x, y) with rows (a,b), (a,f), (c,b), (c,g) at
    # rowids 1, 2, 3, 4. Find R(a, b).
    found = backend.find_tuple("R", {"x": "a", "y": "b"})
    assert found == TupleId("R", 1)


def test_find_tuple_with_partial_unique_match(backend) -> None:
    """A subset of conditions works if it uniquely identifies a row."""
    # Only R(c, g) has y='g'.
    found = backend.find_tuple("R", {"y": "g"})
    assert found == TupleId("R", 4)


def test_find_tuple_raises_on_no_match(backend) -> None:
    """find_tuple raises ValueError when no row matches."""
    with pytest.raises(ValueError, match="No row"):
        backend.find_tuple("R", {"x": "nonexistent"})


def test_find_tuple_raises_on_multiple_matches(backend) -> None:
    """find_tuple raises ValueError when conditions are ambiguous."""
    # R has two rows with x='a': R(a, b) and R(a, f).
    with pytest.raises(ValueError, match="Multiple rows"):
        backend.find_tuple("R", {"x": "a"})


def test_find_tuple_raises_on_empty_where(backend) -> None:
    """find_tuple refuses to run with no conditions."""
    with pytest.raises(ValueError, match="at least one"):
        backend.find_tuple("R", {})


def test_find_tuples_returns_all_matching_rows(backend) -> None:
    """find_tuples returns every row meeting the conditions."""
    # R has two rows with x='a'.
    found = backend.find_tuples("R", {"x": "a"})
    assert found == [TupleId("R", 1), TupleId("R", 2)]


def test_find_tuples_returns_empty_list_on_no_match(backend) -> None:
    """find_tuples returns [] when nothing matches (no exception)."""
    assert backend.find_tuples("R", {"x": "nonexistent"}) == []


def test_find_tuples_without_where_returns_all_rows(backend) -> None:
    """An empty/None `where` yields every row of the table."""
    found = backend.find_tuples("R")
    # smoke_test has 4 R rows.
    assert found == [
        TupleId("R", 1),
        TupleId("R", 2),
        TupleId("R", 3),
        TupleId("R", 4),
    ]