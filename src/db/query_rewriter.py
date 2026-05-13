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

The rewriter is intentionally simple: it expects the user to provide
queries that use explicit table aliases (e.g. "FROM Director d, Movie m"),
and it appends _disabled=0 conditions for each alias.
"""
from __future__ import annotations

import re


def rewrite_query(sql: str, aliases: dict[str, str]) -> str:
    """Append _disabled=0 conditions to a SQL query.

    Parameters
    ----------
    sql : str
        The original SELECT query. Must contain a WHERE clause.
    aliases : dict[str, str]
        Mapping from table alias to table name, e.g. {"d": "Director"}.
        Currently the table name is not used, only the alias matters for
        the SQL output, but we keep both for future extensions.

    Returns
    -------
    str
        The rewritten query with _disabled=0 conditions appended.

    Examples
    --------
    >>> sql = "SELECT name FROM Director d WHERE d.lastName = 'Burton'"
    >>> rewrite_query(sql, {"d": "Director"})
    "SELECT name FROM Director d WHERE d.lastName = 'Burton' AND d._disabled = 0"
    """
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
        # Look for ORDER BY, GROUP BY, or end of string to know where to insert.
        order_match = re.search(
            r"\b(ORDER\s+BY|GROUP\s+BY|LIMIT)\b", sql, re.IGNORECASE
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
    # Find the end of the WHERE clause: either before ORDER BY, GROUP BY,
    # LIMIT, or end of string.
    end_pattern = re.compile(
        r"\b(ORDER\s+BY|GROUP\s+BY|LIMIT)\b", re.IGNORECASE
    )
    end_match = end_pattern.search(sql, where_match.end())

    if end_match:
        before = sql[:end_match.start()].rstrip()
        after = sql[end_match.start():]
        return f"{before} AND {disabled_conditions} {after}"
    else:
        return sql.rstrip() + f" AND {disabled_conditions}"