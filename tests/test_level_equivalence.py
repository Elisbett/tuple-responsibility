"""Equivalence tests: all optimisation levels must produce identical results.

The optimisations applied at Levels 2, 3, and 4 do not change the
semantics of responsibility computation — only the runtime. Therefore,
running any level on the same input must produce identical scores and
identical min_contingency_size values for every candidate.

A failure here means an optimisation has introduced a bug.

The same two tests are run against every dataset registered in the
equivalence_setup fixture, so adding a new dataset only requires
extending the fixture's params list.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from create_synthetic_datasets import (
    build_synth_large,
    build_synth_medium, 
    build_synth_small,
)

from src.core.cached import CachedComputer
from src.core.early_termination import EarlyTerminationComputer
from src.core.naive import NaiveComputer
from src.core.parallel import ParallelComputer
from src.core.responsibility import ResponsibilityComputer
from src.core.types import TupleId
from src.db.query_rewriter import rewrite_query
from src.db.sqlite_backend import SQLiteBackend


SMOKE_DB_PATH = Path("data/synthetic/smoke_test.db")

# All implemented computers. Add new levels here as they are created.
COMPUTERS: list[type[ResponsibilityComputer]] = [
    NaiveComputer,
    EarlyTerminationComputer,
    CachedComputer,
    ParallelComputer,
]

@pytest.fixture(
    params=["smoke_test", "synth_small", "synth_medium", "synth_large"],
    ids=["smoke_test", "synth_small", "synth_medium", "synth_large"],
)

def equivalence_setup(request):
    """Yield (db_path, rewritten_query, expected_answer, candidates, endogenous)
    for each dataset we want to check level-equivalence on.
    """
    name = request.param

    if name == "smoke_test":
        assert SMOKE_DB_PATH.exists(), (
            f"Run `python create_synthetic_datasets.py` (or the older "
            f"create_synthetic_dataset.py) first to create {SMOKE_DB_PATH}"
        )
        setup = SQLiteBackend(SMOKE_DB_PATH)
        setup.add_disabled_columns()
        setup.enable_all()
        setup.close()

        sql = "SELECT DISTINCT x FROM R, S WHERE R.y = S.y"
        rewritten = rewrite_query(sql, {"R": "R", "S": "S"})
        all_tuples = [
            TupleId("R", 1), TupleId("R", 2), TupleId("R", 3), TupleId("R", 4),
            TupleId("S", 1), TupleId("S", 2), TupleId("S", 3),
        ]
        return (SMOKE_DB_PATH, rewritten, ("a",), all_tuples, all_tuples)
    elif name == "synth_small":
        spec = build_synth_small()
        setup = SQLiteBackend(spec.db_path)
        setup.add_disabled_columns()
        setup.close()

        rewritten = rewrite_query(spec.sql_query, spec.aliases)
        return (
            spec.db_path,
            rewritten,
            spec.expected_answer,
            spec.candidates,
            spec.endogenous,
        )
    elif name == "synth_medium":
        spec = build_synth_medium()
        setup = SQLiteBackend(spec.db_path)
        setup.add_disabled_columns()
        setup.close()

        rewritten = rewrite_query(spec.sql_query, spec.aliases)
        return (
            spec.db_path,
            rewritten,
            spec.expected_answer,
            spec.candidates,
            spec.endogenous,
        )
    elif name == "synth_large":
        spec = build_synth_large()
        setup = SQLiteBackend(spec.db_path)
        setup.add_disabled_columns()
        setup.close()

        rewritten = rewrite_query(spec.sql_query, spec.aliases)
        return (
            spec.db_path,
            rewritten,
            spec.expected_answer,
            spec.candidates,
            spec.endogenous,
        )
    else:
        raise ValueError(f"Unknown dataset name: {name}")


def test_all_levels_produce_identical_responsibility_scores(
    equivalence_setup,
) -> None:
    """Every Level should return the same ρ-score for every candidate."""
    db_path, rewritten, expected_answer, candidates, endogenous = equivalence_setup

    # Compute the reference ranking with the first computer (Naive).
    reference_result = COMPUTERS[0]().compute(
        db_path=db_path,
        rewritten_query=rewritten,
        expected_answer=expected_answer,
        candidates=candidates,
        endogenous_tuples=endogenous,
    )
    reference_scores = {
        r.tuple_id: r.responsibility
        for r in reference_result.ranking.results
    }

    # Compare every other computer against the reference.
    for ComputerClass in COMPUTERS[1:]:
        result = ComputerClass().compute(
            db_path=db_path,
            rewritten_query=rewritten,
            expected_answer=expected_answer,
            candidates=candidates,
            endogenous_tuples=endogenous,
        )
        scores = {
            r.tuple_id: r.responsibility for r in result.ranking.results
        }
        for tid in reference_scores:
            assert scores[tid] == pytest.approx(reference_scores[tid]), (
                f"{ComputerClass.__name__} differs from "
                f"{COMPUTERS[0].__name__} on tuple {tid}: "
                f"{scores[tid]} vs {reference_scores[tid]}"
            )


def test_all_levels_produce_identical_contingency_sizes(
    equivalence_setup,
) -> None:
    """Every Level should return the same min_contingency_size."""
    db_path, rewritten, expected_answer, candidates, endogenous = equivalence_setup

    reference_result = COMPUTERS[0]().compute(
        db_path=db_path,
        rewritten_query=rewritten,
        expected_answer=expected_answer,
        candidates=candidates,
        endogenous_tuples=endogenous,
    )
    reference_sizes = {
        r.tuple_id: r.min_contingency_size
        for r in reference_result.ranking.results
    }

    for ComputerClass in COMPUTERS[1:]:
        result = ComputerClass().compute(
            db_path=db_path,
            rewritten_query=rewritten,
            expected_answer=expected_answer,
            candidates=candidates,
            endogenous_tuples=endogenous,
        )
        sizes = {
            r.tuple_id: r.min_contingency_size
            for r in result.ranking.results
        }
        for tid in reference_sizes:
            assert sizes[tid] == reference_sizes[tid], (
                f"{ComputerClass.__name__} differs from "
                f"{COMPUTERS[0].__name__} on tuple {tid}: "
                f"size {sizes[tid]} vs {reference_sizes[tid]}"
            )