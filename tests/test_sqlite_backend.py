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