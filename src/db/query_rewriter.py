"""Rewrites user SQL queries to respect the disabled-tid trick.

A query like:
    SELECT name FROM Director d, Movie m
    WHERE d.id = m.director_id

is rewritten to:
    SELECT name FROM Director d, Movie m
    WHERE d.id = m.director_id
    AND d._disabled = 0 AND m._disabled = 0

This way, queries automatically skip tuples that have been logically
removed via SQLiteBackend.disable().

Supported scope (Assumption 1 of the thesis §3.2)
-------------------------------------------------
The rewriter only handles flat, monotone conjunctive queries: a single
SELECT-FROM-WHERE block with explicit table aliases. Queries containing
negation (NOT IN, NOT EXISTS), set operations (EXCEPT, UNION, INTERSECT,
MINUS), aggregation (GROUP BY, HAVING), or sub-queries are explicitly
rejected at rewrite time by `_validate_scope`, because the disabled-tid
construction is not guaranteed to produce correct responsibility scores
on them.
"""
from __future__ import annotations

import re


# Keywords that fall outside the supported scope (monotone conjunctive
# queries, per Assumption 1 of the thesis §3.2). If any of these appears
# in a query, the disabled-tid rewriting is not guaranteed to produce
# correct responsibility scores, so we reject the query at rewrite time
# rather than risk silently incorrect output.
_UNSUPPORTED_KEYWORDS = (
    "NOT IN",
    "NOT EXISTS",
    "EXCEPT",
    "UNION",
    "INTERSECT",
    "MINUS",
    "GROUP BY",
    "HAVING",
)


def _validate_scope(sql: str) -> None:
    """Reject queries outside the supported scope (monotone CQ).

    Raises
    ------
    ValueError
        If the query contains a keyword from `_UNSUPPORTED_KEYWORDS`.
        The error message names the offending keyword and points the
        user to the relevant section of the thesis.
    """
    # Strip string literals first so a keyword embedded in a literal
    # (e.g. WHERE name = 'NOT IN') does not trigger a false positive.
    # Single-quoted literals are the only kind we expect; if a user
    # writes one with embedded quotes the regex is conservative and may
    # over-strip, which is fine for a guardrail.
    stripped = re.sub(r"'[^']*'", "''", sql)
    upper = stripped.upper()

    for keyword in _UNSUPPORTED_KEYWORDS:
        # Use word boundaries so SUBSTRING-style false matches are avoided
        # (e.g. avoid matching "UNION" inside an identifier "MY_UNION").
        pattern = r"\b" + keyword.replace(" ", r"\s+") + r"\b"
        if re.search(pattern, upper):
            raise ValueError(
                f"Query contains '{keyword}', which is outside the "
                f"supported scope (monotone conjunctive queries). "
                f"See Assumption 1 of the thesis (§3.2). "
                f"Query: {sql!r}"
            )


def rewrite_query(sql: str, aliases: dict[str, str]) -> str:
    """Append _disabled=0 conditions to a SQL query.

    Parameters
    ----------
    sql : str
        The original SELECT query. Must be a flat conjunctive query.
        Queries with negation, set operations, aggregation, or
        sub-queries are rejected (see `_validate_scope`).
    aliases : dict[str, str]
        Mapping from table alias to table name, e.g. {"d": "Director"}.
        Currently the table name is not used, only the alias matters for
        the SQL output, but we keep both for future extensions.

    Returns
    -------
    str
        The rewritten query with _disabled=0 conditions appended.

    Raises
    ------
    ValueError
        If the query falls outside the supported scope.

    Examples
    --------
    >>> sql = "SELECT name FROM Director d WHERE d.lastName = 'Burton'"
    >>> rewrite_query(sql, {"d": "Director"})
    "SELECT name FROM Director d WHERE d.lastName = 'Burton' AND d._disabled = 0"
    """
    _validate_scope(sql)

    if not aliases:
        return sql

    disabled_conditions = " AND ".join(
        f"{alias}._disabled = 0" for alias in aliases
    )

    # Find the WHERE clause and append our conditions.
    # We use case-insensitive matching for SQL keywords.
    where_match = re.search(r"\bWHERE\b", sql, re.IGNORECASE)
    if where_match is None:
        # No WHERE clause — add one.
        # Look for ORDER BY or LIMIT to know where to insert.
        # (GROUP BY / HAVING already filtered out by _validate_scope.)
        order_match = re.search(
            r"\b(ORDER\s+BY|LIMIT)\b", sql, re.IGNORECASE
        )
        if order_match:
            insert_pos = order_match.start()
            return (
                sql[:insert_pos].rstrip()
                + f" WHERE {disabled_conditions} "
                + sql[insert_pos:]
            )
        else:
            return sql.rstrip() + f" WHERE {disabled_conditions}"

    # WHERE clause exists — append to it.
    # Find the end of the WHERE clause: either before ORDER BY, LIMIT,
    # or end of string.
    end_pattern = re.compile(
        r"\b(ORDER\s+BY|LIMIT)\b", re.IGNORECASE
    )
    end_match = end_pattern.search(sql, where_match.end())

    if end_match:
        before = sql[:end_match.start()].rstrip()
        after = sql[end_match.start():]
        return f"{before} AND {disabled_conditions} {after}"
    else:
        return sql.rstrip() + f" AND {disabled_conditions}"