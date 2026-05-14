"""Tests for the scope-validation guardrail in rewrite_query.

The rewriter must reject queries that fall outside the supported scope
(monotone conjunctive queries, per Assumption 1 of the thesis §3.2),
because the disabled-tid trick is not guaranteed to yield correct
responsibility scores on such queries.
"""
from __future__ import annotations

import pytest

from src.db.query_rewriter import rewrite_query


def test_simple_cq_query_is_accepted() -> None:
    """A flat conjunctive query passes validation and is rewritten."""
    sql = "SELECT DISTINCT x FROM R, S WHERE R.y = S.y"
    rewritten = rewrite_query(sql, {"R": "R", "S": "S"})
    assert "_disabled = 0" in rewritten


def test_query_with_alias_is_accepted() -> None:
    """Aliases different from table names pass validation."""
    sql = (
        "SELECT g.genre FROM Director d, Genre g "
        "WHERE d.lastName = 'Burton' AND g.mid = 1"
    )
    rewritten = rewrite_query(sql, {"d": "Director", "g": "Genre"})
    assert "d._disabled = 0" in rewritten
    assert "g._disabled = 0" in rewritten


@pytest.mark.parametrize(
    "keyword,sql",
    [
        ("NOT IN", "SELECT x FROM R WHERE R.y NOT IN (1, 2)"),
        ("NOT EXISTS", "SELECT x FROM R WHERE NOT EXISTS (SELECT 1 FROM S)"),
        ("EXCEPT", "SELECT x FROM R EXCEPT SELECT x FROM S"),
        ("UNION", "SELECT x FROM R UNION SELECT x FROM S"),
        ("INTERSECT", "SELECT x FROM R INTERSECT SELECT x FROM S"),
        ("GROUP BY", "SELECT x, COUNT(*) FROM R WHERE R.y = 1 GROUP BY x"),
        ("HAVING", "SELECT x FROM R WHERE y = 1 HAVING COUNT(*) > 1"),
    ],
)
def test_unsupported_keywords_are_rejected(keyword: str, sql: str) -> None:
    """Every unsupported construct yields a clear ValueError."""
    with pytest.raises(ValueError, match=keyword):
        rewrite_query(sql, {"R": "R"})


def test_keyword_inside_string_literal_is_ignored() -> None:
    """A keyword inside a quoted literal is not a false positive."""
    sql = "SELECT x FROM R WHERE R.name = 'UNION HALL'"
    rewritten = rewrite_query(sql, {"R": "R"})
    assert "_disabled = 0" in rewritten


def test_error_message_mentions_thesis_section() -> None:
    """The error message points the user to §3.2 of the thesis."""
    sql = "SELECT x FROM R UNION SELECT x FROM S"
    with pytest.raises(ValueError, match="§3.2"):
        rewrite_query(sql, {"R": "R"})

def test_standalone_not_in_where_is_rejected() -> None:
    """A bare NOT before a comparison is rejected as non-monotone."""
    sql = "SELECT x FROM R WHERE NOT R.y = 1"
    with pytest.raises(ValueError, match="standalone NOT"):
        rewrite_query(sql, {"R": "R"})


def test_standalone_not_combined_with_and_is_rejected() -> None:
    """NOT mixed with other conditions also gets caught."""
    sql = "SELECT x FROM R WHERE R.y = 1 AND NOT R.x = 'foo'"
    with pytest.raises(ValueError, match="standalone NOT"):
        rewrite_query(sql, {"R": "R"})


def test_not_in_still_gets_specific_error_message() -> None:
    """NOT IN is reported with the specific 'NOT IN' message,
    not the generic standalone-NOT message."""
    sql = "SELECT x FROM R WHERE R.y NOT IN (1, 2)"
    with pytest.raises(ValueError, match="NOT IN"):
        rewrite_query(sql, {"R": "R"})


def test_not_exists_still_gets_specific_error_message() -> None:
    """NOT EXISTS is reported with the specific 'NOT EXISTS' message."""
    sql = "SELECT x FROM R WHERE NOT EXISTS (SELECT 1 FROM S)"
    with pytest.raises(ValueError, match="NOT EXISTS"):
        rewrite_query(sql, {"R": "R"})


def test_word_not_inside_string_literal_is_ignored() -> None:
    """The word NOT inside a quoted string is not a false positive."""
    sql = "SELECT x FROM R WHERE R.name = 'WILL NOT WORK'"
    rewritten = rewrite_query(sql, {"R": "R"})
    assert "_disabled = 0" in rewritten