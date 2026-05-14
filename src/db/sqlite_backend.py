"""SQLite backend with disabled-tid trick.

The disabled-tid trick avoids physically deleting and reinserting tuples
during contingency exploration. Instead, every table gets an extra column
_disabled (BOOLEAN, default 0). To "remove" a tuple, we set _disabled=1;
to "restore" it, we set _disabled=0. All queries are rewritten to include
AND _disabled=0 conditions on every table, so disabled tuples are filtered
out at query time.

This makes contingency exploration much cheaper than physical deletion:
toggling a flag is O(1), whereas DELETE + INSERT would be O(log n) at best
and would invalidate query plan caches.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Iterable

from src.core.types import TupleId


class SQLiteBackend:
    """Wraps a SQLite database with disabled-tid functionality.

    Usage:
        backend = SQLiteBackend("data/example.db")
        backend.add_disabled_columns()  # one-time setup
        backend.disable(TupleId("Director", 5))
        is_still_answer = backend.is_answer(rewritten_query, expected_row)
        backend.enable(TupleId("Director", 5))
    """

    DISABLED_COLUMN = "_disabled"

    def __init__(self, db_path: str | Path, timeout: float = 30.0) -> None:
        """Open a connection to the SQLite database at the given path.

        Parameters
        ----------
        db_path : str | Path
            Path to the SQLite file.
        timeout : float
            Seconds to wait for a database lock before raising. Important
            when multiple worker processes share the same file (Level 4).
        """
        self.db_path = Path(db_path)
        self.connection = sqlite3.connect(
            str(self.db_path),
            timeout=timeout,
        )
        self.connection.row_factory = None

        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("PRAGMA synchronous=NORMAL")

    def close(self) -> None:
        """Close the database connection."""
        self.connection.close()

    def __enter__(self) -> SQLiteBackend:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Schema introspection
    # ------------------------------------------------------------------

    def list_tables(self) -> list[str]:
        """Return all user-defined tables in the database."""
        cursor = self.connection.execute(
            "SELECT name FROM sqlite_master "
            "WHERE type='table' AND name NOT LIKE 'sqlite_%' "
            "ORDER BY name"
        )
        return [row[0] for row in cursor.fetchall()]

    def list_columns(self, table: str) -> list[str]:
        """Return all column names of a table."""
        cursor = self.connection.execute(f"PRAGMA table_info({table})")
        return [row[1] for row in cursor.fetchall()]

    def has_disabled_column(self, table: str) -> bool:
        """Check whether a table already has the _disabled column."""
        return self.DISABLED_COLUMN in self.list_columns(table)

    # ------------------------------------------------------------------
    # Disabled-tid setup
    # ------------------------------------------------------------------

    def add_disabled_columns(self) -> None:
        """Add the _disabled column to every table that does not yet have it.

        This is idempotent: calling it twice is safe. It also creates an
        index on _disabled for each table, which speeds up the rewritten
        WHERE clauses on large tables.
        """
        for table in self.list_tables():
            if not self.has_disabled_column(table):
                self.connection.execute(
                    f"ALTER TABLE {table} "
                    f"ADD COLUMN {self.DISABLED_COLUMN} "
                    f"INTEGER NOT NULL DEFAULT 0"
                )
                self.connection.execute(
                    f"CREATE INDEX IF NOT EXISTS "
                    f"idx_{table}_disabled ON {table}({self.DISABLED_COLUMN})"
                )
        self.connection.commit()

    # ------------------------------------------------------------------
    # Enable / disable tuples
    # ------------------------------------------------------------------

    def disable(self, tuple_id: TupleId) -> None:
        """Mark a single tuple as disabled (logically removed)."""
        self.connection.execute(
            f"UPDATE {tuple_id.relation} "
            f"SET {self.DISABLED_COLUMN}=1 "
            f"WHERE rowid=?",
            (tuple_id.key,),
        )

    def enable(self, tuple_id: TupleId) -> None:
        """Mark a single tuple as enabled (logically restored)."""
        self.connection.execute(
            f"UPDATE {tuple_id.relation} "
            f"SET {self.DISABLED_COLUMN}=0 "
            f"WHERE rowid=?",
            (tuple_id.key,),
        )

    def disable_set(self, tuple_ids: Iterable[TupleId]) -> None:
        """Disable a batch of tuples in a single transaction."""
        for tid in tuple_ids:
            self.disable(tid)

    def enable_set(self, tuple_ids: Iterable[TupleId]) -> None:
        """Enable a batch of tuples in a single transaction."""
        for tid in tuple_ids:
            self.enable(tid)

    def enable_all(self) -> None:
        """Reset all tables: mark every tuple as enabled."""
        for table in self.list_tables():
            if self.has_disabled_column(table):
                self.connection.execute(
                    f"UPDATE {table} SET {self.DISABLED_COLUMN}=0"
                )

    # ------------------------------------------------------------------
    # Query execution
    # ------------------------------------------------------------------

    def execute_query(self, sql: str) -> list[tuple]:
        """Execute a SELECT query and return all rows."""
        cursor = self.connection.execute(sql)
        return cursor.fetchall()

    def is_answer(self, rewritten_query: str, expected_row: tuple) -> bool:
        """Check whether expected_row appears in the result of the query.

        The query must already be rewritten to include _disabled=0 filters
        on every table. See query_rewriter.py for the rewriting logic.
        """
        rows = self.execute_query(rewritten_query)
        return expected_row in rows
    
    # ------------------------------------------------------------------
    # Tuple lookup by column values
    # ------------------------------------------------------------------

    def find_tuple(
        self,
        table: str,
        where: dict[str, object],
    ) -> TupleId:
        """Find the TupleId of a row identified by column values.

        Lets callers refer to tuples by content ("the Director row
        with lastName='Burton' and firstName='David'") instead of by
        SQLite's internal rowid. Intended as the main entry point for
        users assembling a candidate set without having to inspect
        rowid values directly.

        Parameters
        ----------
        table : str
            The table to search in (e.g. "Director").
        where : dict[str, object]
            Column-value conditions. Each entry is matched with `=`
            and combined with AND. For example
            ``{"firstName": "David", "lastName": "Burton"}`` is turned
            into ``WHERE firstName = ? AND lastName = ?``.

        Returns
        -------
        TupleId
            The (table, rowid) pair of the unique matching row.

        Raises
        ------
        ValueError
            If zero or more than one row matches the conditions. The
            caller is expected to supply enough conditions to identify
            exactly one row.

        Examples
        --------
        Resolve a director identified by name to a TupleId::

            backend = SQLiteBackend("imdb.db")
            david = backend.find_tuple(
                "Director",
                {"firstName": "David", "lastName": "Burton"},
            )
            # -> TupleId(relation='Director', key=1)
        """
        if not where:
            raise ValueError(
                "find_tuple requires at least one column-value condition "
                "to identify a row; got an empty `where` dict."
            )

        conditions = " AND ".join(f"{col} = ?" for col in where)
        sql = f"SELECT rowid FROM {table} WHERE {conditions}"
        rows = self.connection.execute(sql, tuple(where.values())).fetchall()

        if len(rows) == 0:
            raise ValueError(
                f"No row in table '{table}' matches conditions {where}."
            )
        if len(rows) > 1:
            raise ValueError(
                f"Multiple rows ({len(rows)}) in table '{table}' match "
                f"conditions {where}. Refine the conditions to identify "
                f"exactly one row."
            )

        return TupleId(relation=table, key=rows[0][0])

    def find_tuples(
        self,
        table: str,
        where: dict[str, object] | None = None,
    ) -> list[TupleId]:
        """Find all TupleIds matching the given conditions (zero allowed).

        Convenience method for selecting an entire group of candidates,
        e.g. "every Director row" (empty conditions) or "every Movie
        with year > 2000" (a single condition).

        Parameters
        ----------
        table : str
            The table to search.
        where : dict[str, object] | None
            Column-value equality conditions, ANDed together. If None
            or empty, returns every row of the table.

        Returns
        -------
        list[TupleId]
            One TupleId per matching row, in rowid order. Empty list
            if no rows match (no exception, unlike `find_tuple`).
        """
        if where:
            conditions = " AND ".join(f"{col} = ?" for col in where)
            sql = f"SELECT rowid FROM {table} WHERE {conditions} ORDER BY rowid"
            rows = self.connection.execute(
                sql, tuple(where.values())
            ).fetchall()
        else:
            sql = f"SELECT rowid FROM {table} ORDER BY rowid"
            rows = self.connection.execute(sql).fetchall()

        return [TupleId(relation=table, key=row[0]) for row in rows]