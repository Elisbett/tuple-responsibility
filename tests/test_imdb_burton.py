"""Correctness test for the IMDB Burton case study.

Reproduces Figure 2(b) of Meliou et al. (2010) "Causality in
Databases" (IEEE Data Engineering Bulletin) on the 9-tuple Burton
subset constructed in create_synthetic_datasets.build_imdb_burton.

This is the strongest correctness check in the project: instead of
verifying against hand-computed scores on a synthetic dataset, it
verifies against responsibility scores published in a peer-reviewed
paper. Passing this test demonstrates that the implementation
faithfully reproduces the algorithm's intended behaviour on a
non-trivial realistic example.

Expected scores (Figure 2(b)):
    Top tier  (ρ = 1/3 ≈ 0.333, min |Γ| = 2):
        Director#1 (David Burton)
        Director#2 (Humphrey Burton)
        Director#3 (Tim Burton)
        Movie#6    (Sweeney Todd)
    Middle    (ρ = 1/4 = 0.25, min |Γ| = 3):
        Movie#1 (The Melody Lingers On)
        Movie#2 (Let's Fall in Love)
    Bottom    (ρ = 1/5 = 0.20, min |Γ| = 4):
        Movie#3 (Manon Lescaut)
        Movie#4 (Flight)
        Movie#5 (Candide)
"""
from __future__ import annotations

import pytest

from create_synthetic_datasets import build_imdb_burton

from src.core.naive import NaiveComputer
from src.core.types import TupleId
from src.db.query_rewriter import rewrite_query
from src.db.sqlite_backend import SQLiteBackend


@pytest.fixture
def imdb_spec():
    """Rebuild imdb_burton dataset and prepare disabled columns."""
    spec = build_imdb_burton()
    setup = SQLiteBackend(spec.db_path)
    setup.add_disabled_columns()
    setup.close()
    return spec


def test_imdb_scores_match_paper_figure_2b(imdb_spec) -> None:
    """All 9 responsibility scores match Meliou et al. 2010 Figure 2(b)."""
    rewritten = rewrite_query(imdb_spec.sql_query, imdb_spec.aliases)

    result = NaiveComputer().compute(
        db_path=imdb_spec.db_path,
        rewritten_query=rewritten,
        expected_answer=imdb_spec.expected_answer,
        candidates=imdb_spec.candidates,
        endogenous_tuples=imdb_spec.endogenous,
    )
    scores = {r.tuple_id: r.responsibility for r in result.ranking.results}

    # Top tier: ρ = 1/3, contingency size 2 (Fig. 2b lines 1-4)
    assert scores[TupleId("Director", 1)] == pytest.approx(1.0 / 3.0)  # David
    assert scores[TupleId("Director", 2)] == pytest.approx(1.0 / 3.0)  # Humphrey
    assert scores[TupleId("Director", 3)] == pytest.approx(1.0 / 3.0)  # Tim
    assert scores[TupleId("Movie", 6)] == pytest.approx(1.0 / 3.0)     # Sweeney Todd

    # Middle: ρ = 1/4 (Fig. 2b lines 5-6)
    assert scores[TupleId("Movie", 1)] == pytest.approx(0.25)  # Melody Lingers On
    assert scores[TupleId("Movie", 2)] == pytest.approx(0.25)  # Let's Fall in Love

    # Bottom: ρ = 1/5 (Fig. 2b lines 7-9)
    assert scores[TupleId("Movie", 3)] == pytest.approx(0.20)  # Manon Lescaut
    assert scores[TupleId("Movie", 4)] == pytest.approx(0.20)  # Flight
    assert scores[TupleId("Movie", 5)] == pytest.approx(0.20)  # Candide


def test_imdb_contingency_sizes_match_paper(imdb_spec) -> None:
    """Minimum contingency sizes match the values published in the paper.

    Particularly the Manon Lescaut case is documented explicitly in
    PVLDB footnote 2: "{Movie(Candide), Movie(Flight),
    Director(David Burton), Director(Tim Burton)}" — size 4.
    """
    rewritten = rewrite_query(imdb_spec.sql_query, imdb_spec.aliases)

    result = NaiveComputer().compute(
        db_path=imdb_spec.db_path,
        rewritten_query=rewritten,
        expected_answer=imdb_spec.expected_answer,
        candidates=imdb_spec.candidates,
        endogenous_tuples=imdb_spec.endogenous,
    )
    sizes = {r.tuple_id: r.min_contingency_size for r in result.ranking.results}

    # All three directors and Sweeney Todd: size 2
    assert sizes[TupleId("Director", 1)] == 2
    assert sizes[TupleId("Director", 2)] == 2
    assert sizes[TupleId("Director", 3)] == 2
    assert sizes[TupleId("Movie", 6)] == 2

    # David's two films: size 3
    assert sizes[TupleId("Movie", 1)] == 3
    assert sizes[TupleId("Movie", 2)] == 3

    # Humphrey's three films: size 4
    assert sizes[TupleId("Movie", 3)] == 4
    assert sizes[TupleId("Movie", 4)] == 4
    assert sizes[TupleId("Movie", 5)] == 4