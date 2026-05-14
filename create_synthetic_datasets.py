"""Generator for synthetic SQLite datasets used in §6 experiments.

Each dataset is a self-contained SQLite file in data/synthetic/ with
a fixed, reproducible schema and tuple set. The accompanying
DATASET_REGISTRY dict pairs each dataset with the query, expected
answer, and candidate set used in benchmarks — so experiments do not
need to encode this metadata separately.

Add new datasets by writing a new build_* function and registering it
in DATASET_REGISTRY at the bottom of the file.
"""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from pathlib import Path

from src.core.types import TupleId

OUTPUT_DIR = Path("data/synthetic")


# ---------------------------------------------------------------------
# Per-dataset metadata
# ---------------------------------------------------------------------

@dataclass
class DatasetSpec:
    """Everything an experiment needs to know about a dataset."""
    name: str
    description: str
    sql_query: str
    aliases: dict[str, str]              # alias -> table name, for rewriter
    expected_answer: tuple
    candidates: list[TupleId]
    endogenous: list[TupleId]
    n_endogenous: int = field(init=False)

    def __post_init__(self) -> None:
        self.n_endogenous = len(self.endogenous)

    @property
    def db_path(self) -> Path:
        return OUTPUT_DIR / f"{self.name}.db"


# ---------------------------------------------------------------------
# Dataset builders
# ---------------------------------------------------------------------

def build_synth_small() -> DatasetSpec:
    """Smallest synthetic dataset: 8 endogenous tuples, single 2-way join.

    Schema:
        R(x, y) -- 5 tuples
        S(y)    -- 3 tuples
    Query: find all distinct x such that R.y matches some S.y.
    Answer of interest: 'a' (has two supporting minterms).
    """
    name = "synth_small"
    db_path = OUTPUT_DIR / f"{name}.db"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if db_path.exists():
        db_path.unlink()

    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()
    cur.execute("CREATE TABLE R (x TEXT, y TEXT)")
    cur.execute("CREATE TABLE S (y TEXT)")

    r_tuples = [
        ("a", "b"),   # rowid 1
        ("a", "f"),   # rowid 2
        ("c", "b"),   # rowid 3
        ("c", "g"),   # rowid 4
        ("e", "f"),   # rowid 5  -- new tuple compared to smoke_test
    ]
    s_tuples = [
        ("b",),  # rowid 1
        ("f",),  # rowid 2
        ("h",),  # rowid 3
    ]
    cur.executemany("INSERT INTO R(x, y) VALUES (?, ?)", r_tuples)
    cur.executemany("INSERT INTO S(y) VALUES (?)", s_tuples)
    conn.commit()
    conn.close()

    all_tuples = [TupleId("R", i) for i in range(1, 6)] + [
        TupleId("S", i) for i in range(1, 4)
    ]

    return DatasetSpec(
        name=name,
        description="8 endogenous tuples; 2-way join; answer 'a' has 2 minterms.",
        sql_query="SELECT DISTINCT x FROM R, S WHERE R.y = S.y",
        aliases={"R": "R", "S": "S"},
        expected_answer=("a",),
        candidates=all_tuples,
        endogenous=all_tuples,
    )

def build_synth_medium() -> DatasetSpec:
    """Medium synthetic dataset: 12 endogenous tuples, single 2-way join.

    Same schema as synth_small (R-S join), more tuples on both sides.
    The answer 'a' has multiple supporting minterms; some tuples appear
    in several minterms, making contingency sizes more varied than in
    synth_small.

    Schema:
        R(x, y) -- 7 tuples
        S(y)    -- 5 tuples
    """
    name = "synth_medium"
    db_path = OUTPUT_DIR / f"{name}.db"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if db_path.exists():
        db_path.unlink()

    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()
    cur.execute("CREATE TABLE R (x TEXT, y TEXT)")
    cur.execute("CREATE TABLE S (y TEXT)")

    r_tuples = [
        ("a", "b"),  # rowid 1
        ("a", "f"),  # rowid 2
        ("a", "h"),  # rowid 3
        ("c", "b"),  # rowid 4
        ("c", "g"),  # rowid 5
        ("e", "f"),  # rowid 6
        ("e", "h"),  # rowid 7
    ]
    s_tuples = [
        ("b",),  # rowid 1
        ("f",),  # rowid 2
        ("g",),  # rowid 3
        ("h",),  # rowid 4
        ("k",),  # rowid 5  -- not joined with any R
    ]
    cur.executemany("INSERT INTO R(x, y) VALUES (?, ?)", r_tuples)
    cur.executemany("INSERT INTO S(y) VALUES (?)", s_tuples)
    conn.commit()
    conn.close()

    all_tuples = [TupleId("R", i) for i in range(1, 8)] + [
        TupleId("S", i) for i in range(1, 6)
    ]

    return DatasetSpec(
        name=name,
        description="12 endogenous tuples; 2-way join; answer 'a' has 3 minterms.",
        sql_query="SELECT DISTINCT x FROM R, S WHERE R.y = S.y",
        aliases={"R": "R", "S": "S"},
        expected_answer=("a",),
        candidates=all_tuples,
        endogenous=all_tuples,
    )

def build_synth_large() -> DatasetSpec:
    """Large synthetic dataset: 16 endogenous tuples, 2-way join.

    Same schema as synth_small/synth_medium (R-S), more tuples to push
    Naive into the seconds range. Cached should now show its full
    advantage from minterm overlap, and Parallel should comfortably
    beat sequential on a multi-core machine.

    Schema:
        R(x, y) -- 10 tuples
        S(y)    --  6 tuples
    """
    name = "synth_large"
    db_path = OUTPUT_DIR / f"{name}.db"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if db_path.exists():
        db_path.unlink()

    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()
    cur.execute("CREATE TABLE R (x TEXT, y TEXT)")
    cur.execute("CREATE TABLE S (y TEXT)")

    r_tuples = [
        ("a", "b"),  # 1
        ("a", "f"),  # 2
        ("a", "h"),  # 3
        ("a", "k"),  # 4
        ("c", "b"),  # 5
        ("c", "g"),  # 6
        ("c", "k"),  # 7
        ("e", "f"),  # 8
        ("e", "h"),  # 9
        ("e", "m"),  # 10
    ]
    s_tuples = [
        ("b",),  # 1
        ("f",),  # 2
        ("g",),  # 3
        ("h",),  # 4
        ("k",),  # 5
        ("n",),  # 6 -- no match
    ]
    cur.executemany("INSERT INTO R(x, y) VALUES (?, ?)", r_tuples)
    cur.executemany("INSERT INTO S(y) VALUES (?)", s_tuples)
    conn.commit()
    conn.close()

    all_tuples = [TupleId("R", i) for i in range(1, 11)] + [
        TupleId("S", i) for i in range(1, 7)
    ]

    return DatasetSpec(
        name=name,
        description="16 endogenous tuples; 2-way join; answer 'a' has 4 minterms.",
        sql_query="SELECT DISTINCT x FROM R, S WHERE R.y = S.y",
        aliases={"R": "R", "S": "S"},
        expected_answer=("a",),
        candidates=all_tuples,
        endogenous=all_tuples,
    )

def build_synth_xlarge() -> DatasetSpec:
    """X-large synthetic dataset: 20 endogenous tuples, 2-way join.

    The size at which Naive becomes impractical (minutes per query),
    while Cached and Parallel remain usable. This dataset is the main
    illustration of the engineering payoff documented in §6.

    Schema:
        R(x, y) -- 13 tuples
        S(y)    --  7 tuples
    """
    name = "synth_xlarge"
    db_path = OUTPUT_DIR / f"{name}.db"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if db_path.exists():
        db_path.unlink()

    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()
    cur.execute("CREATE TABLE R (x TEXT, y TEXT)")
    cur.execute("CREATE TABLE S (y TEXT)")

    r_tuples = [
        ("a", "b"),  # 1
        ("a", "f"),  # 2
        ("a", "h"),  # 3
        ("a", "k"),  # 4
        ("a", "m"),  # 5
        ("c", "b"),  # 6
        ("c", "g"),  # 7
        ("c", "k"),  # 8
        ("c", "p"),  # 9
        ("e", "f"),  # 10
        ("e", "h"),  # 11
        ("e", "m"),  # 12
        ("e", "r"),  # 13
    ]
    s_tuples = [
        ("b",),  # 1
        ("f",),  # 2
        ("g",),  # 3
        ("h",),  # 4
        ("k",),  # 5
        ("m",),  # 6
        ("n",),  # 7 -- no match
    ]
    cur.executemany("INSERT INTO R(x, y) VALUES (?, ?)", r_tuples)
    cur.executemany("INSERT INTO S(y) VALUES (?)", s_tuples)
    conn.commit()
    conn.close()

    all_tuples = [TupleId("R", i) for i in range(1, 14)] + [
        TupleId("S", i) for i in range(1, 8)
    ]

    return DatasetSpec(
        name=name,
        description="20 endogenous tuples; 2-way join; answer 'a' has 5 minterms.",
        sql_query="SELECT DISTINCT x FROM R, S WHERE R.y = S.y",
        aliases={"R": "R", "S": "S"},
        expected_answer=("a",),
        candidates=all_tuples,
        endogenous=all_tuples,
    )

def build_synth_join3() -> DatasetSpec:
    """Three-way join dataset: 12 endogenous tuples, R-S-T schema.

    Same |D_n| as synth_medium, but each minterm now consists of THREE
    supporting tuples instead of two. This changes the algorithmic
    profile: more SQL evaluations per minterm, different caching
    behaviour. Used in §6 to demonstrate that the four optimisations
    behave differently on different query structures.

    Schema:
        R(x, y) -- 5 tuples
        S(y, z) -- 4 tuples
        T(z)    -- 3 tuples

    Query: SELECT DISTINCT x FROM R, S, T WHERE R.y = S.y AND S.z = T.z
    """
    name = "synth_join3"
    db_path = OUTPUT_DIR / f"{name}.db"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if db_path.exists():
        db_path.unlink()

    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()
    cur.execute("CREATE TABLE R (x TEXT, y TEXT)")
    cur.execute("CREATE TABLE S (y TEXT, z TEXT)")
    cur.execute("CREATE TABLE T (z TEXT)")

    r_tuples = [
        ("a", "b"),  # 1
        ("a", "f"),  # 2
        ("c", "b"),  # 3
        ("c", "g"),  # 4
        ("e", "f"),  # 5
    ]
    s_tuples = [
        ("b", "p"),  # 1
        ("b", "q"),  # 2
        ("f", "p"),  # 3
        ("g", "r"),  # 4
    ]
    t_tuples = [
        ("p",),  # 1
        ("q",),  # 2
        ("s",),  # 3 -- no match
    ]
    cur.executemany("INSERT INTO R(x, y) VALUES (?, ?)", r_tuples)
    cur.executemany("INSERT INTO S(y, z) VALUES (?, ?)", s_tuples)
    cur.executemany("INSERT INTO T(z) VALUES (?)", t_tuples)
    conn.commit()
    conn.close()

    all_tuples = (
        [TupleId("R", i) for i in range(1, 6)]
        + [TupleId("S", i) for i in range(1, 5)]
        + [TupleId("T", i) for i in range(1, 4)]
    )

    return DatasetSpec(
        name=name,
        description="12 endogenous tuples; 3-way join; structural variation.",
        sql_query=(
            "SELECT DISTINCT x FROM R, S, T "
            "WHERE R.y = S.y AND S.z = T.z"
        ),
        aliases={"R": "R", "S": "S", "T": "T"},
        expected_answer=("a",),
        candidates=all_tuples,
        endogenous=all_tuples,
    )

# ---------------------------------------------------------------------
# Registry of all datasets
# ---------------------------------------------------------------------

DATASET_REGISTRY: dict[str, callable] = {
    "synth_small": build_synth_small,
    "synth_medium": build_synth_medium,
    "synth_large": build_synth_large,
    "synth_xlarge": build_synth_xlarge,
    "synth_join3": build_synth_join3,
    # later: synth_dense
}


def build_all() -> list[DatasetSpec]:
    """Build all registered datasets. Returns list of specs."""
    specs = []
    for name, builder in DATASET_REGISTRY.items():
        print(f"Building {name}...")
        spec = builder()
        print(f"  -> {spec.db_path} ({spec.n_endogenous} endogenous tuples)")
        specs.append(spec)
    return specs


if __name__ == "__main__":
    build_all()