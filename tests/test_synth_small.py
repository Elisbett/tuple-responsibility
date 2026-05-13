"""Correctness test for the synth_small dataset.

Verifies that NaiveComputer produces the responsibility scores and
contingency sizes that we computed by hand for synth_small. Once Naive
is proven correct on this dataset, the level-equivalence test (in
test_level_equivalence.py) implies the other three levels are correct
too. The expected values below are derived by direct application of
Definition 2.3 of Meliou et al. (2010) to the data.

Schema and expected scores (see create_synthetic_datasets.build_synth_small):
    R = { (a,b)#1, (a,f)#2, (c,b)#3, (c,g)#4, (e,f)#5 }
    S = { (b)#1,   (f)#2,   (h)#3 }
    Query: SELECT DISTINCT x FROM R, S WHERE R.y = S.y
    Answer: ("a",)

    Minterms for "a": { R(a,b), S(b) } and { R(a,f), S(f) }.

    Expected:
        R#1, R#2, S#1, S#2 → ρ = 0.5, min |Γ| = 1
        R#3, R#4, R#5, S#3 → ρ = 0.0, min |Γ| = None
"""
from __future__ import annotations

import pytest

from create_synthetic_datasets import build_synth_small

from src.core.naive import NaiveComputer
from src.core.types import TupleId
from src.db.query_rewriter import rewrite_query
from src.db.sqlite_backend import SQLiteBackend


@pytest.fixture
def spec_and_db():
    """Rebuild synth_small and prepare disabled columns; yield the spec."""
    spec = build_synth_small()
    setup = SQLiteBackend(spec.db_path)
    setup.add_disabled_columns()
    setup.close()
    return spec


def test_naive_scores_on_synth_small(spec_and_db) -> None:
    spec = spec_and_db
    rewritten = rewrite_query(spec.sql_query, spec.aliases)

    result = NaiveComputer().compute(
        db_path=spec.db_path,
        rewritten_query=rewritten,
        expected_answer=spec.expected_answer,
        candidates=spec.candidates,
        endogenous_tuples=spec.endogenous,
    )

    scores = {r.tuple_id: r.responsibility for r in result.ranking.results}

    # Actual causes for answer "a": ρ = 0.5
    assert scores[TupleId("R", 1)] == pytest.approx(0.5)
    assert scores[TupleId("R", 2)] == pytest.approx(0.5)
    assert scores[TupleId("S", 1)] == pytest.approx(0.5)
    assert scores[TupleId("S", 2)] == pytest.approx(0.5)

    # Non-causes: ρ = 0.0
    assert scores[TupleId("R", 3)] == 0.0
    assert scores[TupleId("R", 4)] == 0.0
    assert scores[TupleId("R", 5)] == 0.0  # new tuple R(e,f), unrelated to "a"
    assert scores[TupleId("S", 3)] == 0.0


def test_naive_contingency_sizes_on_synth_small(spec_and_db) -> None:
    spec = spec_and_db
    rewritten = rewrite_query(spec.sql_query, spec.aliases)

    result = NaiveComputer().compute(
        db_path=spec.db_path,
        rewritten_query=rewritten,
        expected_answer=spec.expected_answer,
        candidates=spec.candidates,
        endogenous_tuples=spec.endogenous,
    )

    sizes = {r.tuple_id: r.min_contingency_size for r in result.ranking.results}

    assert sizes[TupleId("R", 1)] == 1
    assert sizes[TupleId("R", 2)] == 1
    assert sizes[TupleId("S", 1)] == 1
    assert sizes[TupleId("S", 2)] == 1
    assert sizes[TupleId("R", 3)] is None
    assert sizes[TupleId("R", 4)] is None
    assert sizes[TupleId("R", 5)] is None
    assert sizes[TupleId("S", 3)] is None