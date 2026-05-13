"""Create a small synthetic SQLite dataset for smoke-testing the backend.

Schema (kept abstract on purpose — the algorithm is general, not IMDB-specific):
    Table R(x, y): 4 tuples
    Table S(y):    3 tuples

Query: SELECT DISTINCT x FROM R, S WHERE R.y = S.y
Answer of interest: 'a'

This is enough to manually verify that:
1. disabling a tuple correctly removes it from query results,
2. enabling it restores it,
3. the rewritten query semantics match the expected semantics.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

OUTPUT_PATH = Path("data/synthetic/smoke_test.db")


def main() -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    if OUTPUT_PATH.exists():
        OUTPUT_PATH.unlink()

    conn = sqlite3.connect(str(OUTPUT_PATH))
    cursor = conn.cursor()

    cursor.execute("CREATE TABLE R (x TEXT, y TEXT)")
    cursor.execute("CREATE TABLE S (y TEXT)")

    r_tuples = [
        ("a", "b"),
        ("a", "f"),
        ("c", "b"),
        ("c", "g"),
    ]
    s_tuples = [
        ("b",),
        ("f",),
        ("h",),
    ]

    cursor.executemany("INSERT INTO R(x, y) VALUES (?, ?)", r_tuples)
    cursor.executemany("INSERT INTO S(y) VALUES (?)", s_tuples)

    conn.commit()
    conn.close()

    print(f"Created {OUTPUT_PATH} with {len(r_tuples)} R-tuples "
          f"and {len(s_tuples)} S-tuples.")


if __name__ == "__main__":
    main()