"""Compare naive 1/N counting against responsibility on IMDB Burton.

Background
----------
A simple, intuitive way to score how "responsible" a tuple is for a
query answer is to count how often it appears in the lineage of that
answer. The lineage of the answer ("Musical",) on the IMDB Burton
dataset is a Boolean formula over endogenous tuples whose minimal
satisfying assignments — the *minterms*, also called witness sets —
are the smallest sets of tuples whose joint presence yields the answer.

Two natural scoring rules from this provenance view:
  - frequency:  freq(t) = #minterms containing t / #minterms
  - 1/N:        contribute 1/|M| for each minterm M containing t,
                then average across minterms.

Both rules ignore *interactions* between tuples — they treat a tuple's
role as additive across the lineage. The responsibility score from
Definition 2.3, by contrast, asks a fundamentally different question:
how small a contingency Γ makes t a counterfactual cause? This script
runs both views on the same dataset and prints a side-by-side
comparison, supporting the discussion in §2.4 and §6.

Usage
-----
    python imdb_1_over_n.py
"""
from __future__ import annotations

from itertools import combinations

from create_synthetic_datasets import DATASET_REGISTRY
from src.core.naive import NaiveComputer
from src.core.types import TupleId
from src.db.query_rewriter import rewrite_query
from src.db.sqlite_backend import SQLiteBackend


# Human-readable names for the IMDB Burton endogenous tuples.
NAMES: dict[str, str] = {
    "Director#1": "David Burton",
    "Director#2": "Humphrey Burton",
    "Director#3": "Tim Burton",
    "Movie#1":    "The Melody Lingers On",
    "Movie#2":    "Let's Fall in Love",
    "Movie#3":    "Manon Lescaut",
    "Movie#4":    "Flight",
    "Movie#5":    "Candide",
    "Movie#6":    "Sweeney Todd",
}


def find_minterms(
    backend: SQLiteBackend,
    rewritten_query: str,
    expected_answer: tuple,
    endogenous: list[TupleId],
) -> list[frozenset[TupleId]]:
    """Enumerate the minimal witness sets for the given answer.

    A minterm is a minimal set M of endogenous tuples such that
    disabling everything in (endogenous \\ M) preserves the answer.
    For |D_n| = 9 the brute-force enumeration (sweep by increasing
    |M|, skip supersets of any minterm already found) is fast enough.
    """
    minterms: list[frozenset[TupleId]] = []
    endo_set = set(endogenous)
    for size in range(1, len(endogenous) + 1):
        for combo in combinations(endogenous, size):
            combo_set = frozenset(combo)
            if any(m <= combo_set for m in minterms):
                continue
            backend.enable_all()
            backend.disable_set(endo_set - combo_set)
            if backend.is_answer(rewritten_query, expected_answer):
                minterms.append(combo_set)
    return minterms


def freq_scores(
    minterms: list[frozenset[TupleId]],
    endogenous: list[TupleId],
) -> dict[TupleId, float]:
    """Lineage frequency: fraction of minterms containing each tuple."""
    n = len(minterms)
    if n == 0:
        return {t: 0.0 for t in endogenous}
    return {
        t: sum(1 for m in minterms if t in m) / n
        for t in endogenous
    }


def one_over_n_scores(
    minterms: list[frozenset[TupleId]],
    endogenous: list[TupleId],
) -> dict[TupleId, float]:
    """For each minterm M, every t in M contributes 1/|M|; then averaged.

    Result is on the same [0, 1] scale as rho and the frequency rule.
    """
    n = len(minterms)
    if n == 0:
        return {t: 0.0 for t in endogenous}
    raw: dict[TupleId, float] = {t: 0.0 for t in endogenous}
    for m in minterms:
        contribution = 1.0 / len(m)
        for t in m:
            raw[t] += contribution
    return {t: v / n for t, v in raw.items()}


def rank_of(scores: dict[TupleId, float], tuple_id: TupleId) -> int:
    """1-based dense rank (ties share rank, next rank is +1)."""
    sorted_unique = sorted(set(scores.values()), reverse=True)
    target = scores[tuple_id]
    return sorted_unique.index(target) + 1


def main() -> None:
    spec = DATASET_REGISTRY["imdb_burton"]()

    backend = SQLiteBackend(spec.db_path)
    backend.add_disabled_columns()
    rewritten = rewrite_query(spec.sql_query, spec.aliases)

    print("Finding minterms for the answer ('Musical',)...")
    minterms = find_minterms(backend, rewritten, spec.expected_answer, spec.endogenous)
    print(f"  found {len(minterms)} minterms")

    freq = freq_scores(minterms, spec.endogenous)
    one_n = one_over_n_scores(minterms, spec.endogenous)

    backend.close()

    print("Computing responsibility (NaiveComputer) ...")
    result = NaiveComputer().compute(
        db_path=spec.db_path,
        rewritten_query=rewritten,
        expected_answer=spec.expected_answer,
        candidates=spec.endogenous,
        endogenous_tuples=spec.endogenous,
    )
    rho: dict[TupleId, float] = {
        r.tuple_id: r.responsibility for r in result.ranking.results
    }

    print()
    print("=" * 90)
    print("Comparison of scoring rules on the IMDB Burton dataset (answer 'Musical')")
    print("=" * 90)
    print(
        f"{'Tuple':<14} {'Name':<24} "
        f"{'rho':>7}  {'rho rk':>6}   "
        f"{'Freq':>6}  {'F rk':>5}   "
        f"{'1/N':>6}  {'1/N rk':>7}"
    )
    print("-" * 90)
    order = sorted(spec.endogenous, key=lambda t: -rho[t])
    for t in order:
        key = str(t)
        name = NAMES.get(key, "")
        print(
            f"{key:<14} {name:<24} "
            f"{rho[t]:>7.3f}  {rank_of(rho, t):>6d}   "
            f"{freq[t]:>6.3f}  {rank_of(freq, t):>5d}   "
            f"{one_n[t]:>6.3f}  {rank_of(one_n, t):>7d}"
        )
    print("=" * 90)

    diverging = [
        t for t in spec.endogenous
        if rank_of(rho, t) != rank_of(freq, t)
        or rank_of(rho, t) != rank_of(one_n, t)
    ]
    print()
    if diverging:
        print(f"{len(diverging)} of {len(spec.endogenous)} tuples are ranked "
              f"differently by rho than by at least one of the simple rules.")
    else:
        print("All scoring rules induce the same ranking on this dataset.")


if __name__ == "__main__":
    main()