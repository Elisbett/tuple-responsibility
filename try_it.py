"""Quick interactive script: see Naive responsibility in action."""
from src.core.naive import NaiveComputer
from src.core.types import TupleId
from src.db.query_rewriter import rewrite_query
from src.db.sqlite_backend import SQLiteBackend

backend = SQLiteBackend("data/synthetic/smoke_test.db")
backend.add_disabled_columns()
backend.enable_all()

sql = "SELECT DISTINCT x FROM R, S WHERE R.y = S.y"
rewritten = rewrite_query(sql, {"R": "R", "S": "S"})

all_tuples = [TupleId("R", i) for i in range(1, 5)] + [
    TupleId("S", i) for i in range(1, 4)
]

ranking = NaiveComputer().compute(
    backend=backend,
    rewritten_query=rewritten,
    expected_answer=("a",),
    candidates=all_tuples,
    endogenous_tuples=all_tuples,
)

print(f"{'Tuple':<15} {'Responsibility':<15} {'Min |Γ|':<10}")
print("-" * 40)
for r in ranking.sorted_by_responsibility():
    size_str = str(r.min_contingency_size) if r.min_contingency_size is not None else "—"
    print(f"{str(r.tuple_id):<15} {r.responsibility:<15.3f} {size_str:<10}")

backend.close()