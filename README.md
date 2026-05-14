# Computing Causal Responsibility of Tuples for Query Answers

This repository accompanies the Bachelor's thesis
*Computing Causal Responsibility of Tuples for Query Answers:
A Practical Implementation with Engineering Optimisations*,
submitted to the University of Tartu, Institute of Computer Science,
in 2026.

The tool computes the **causal responsibility score**

```
ρ(t) = 1 / (1 + min |Γ|)
```

for each candidate tuple `t` in a relational database with respect to a
query answer `r`, following the definition of Meliou, Gatterbauer,
Halpern, Koch, Moore and Suciu (2010). Four implementations of the same
algorithm are provided, each adding one engineering optimisation on top
of the previous level:

| Level | Class | What it adds |
|-------|-------|--------------|
| 1 | `NaiveComputer` | Direct enumeration of every subset of `D_n \ {t}` |
| 2 | `EarlyTerminationComputer` | Return on the first valid contingency |
| 3 | `CachedComputer` | Memoise query evaluations across candidates |
| 4 | `ParallelComputer` | Distribute candidates across `multiprocessing.Pool` |

All four return the same responsibility scores; only the runtime differs.
Chapter §6 of the thesis reports the speedups measured on seven datasets.

## Repository layout

```
tuple-responsibility/
├── src/
│   ├── core/
│   │   ├── types.py              # TupleId, ResponsibilityResult, ComputeResult, ...
│   │   ├── responsibility.py     # Abstract base class + shared helpers
│   │   ├── naive.py              # Level 1
│   │   ├── early_termination.py  # Level 2
│   │   ├── cached.py             # Level 3
│   │   └── parallel.py           # Level 4
│   └── db/
│       ├── sqlite_backend.py     # disabled-tid mechanism + connection handling
│       └── query_rewriter.py     # rewrites SQL + enforces scope (§3.2)
├── tests/                        # 51 parametrised pytest tests
├── data/synthetic/               # generated SQLite files (created on first run)
├── experiments/results/          # benchmark CSV + figures (created by scripts)
├── create_synthetic_datasets.py  # builds all 7 datasets
├── run_on_dataset.py             # runs the four levels on one dataset
├── benchmark.py                  # full evaluation reported in §6
├── plot_results.py               # produces the §6 figures
├── imdb_1_over_n.py              # comparison used in §6.5
└── requirements.txt
```

## Requirements

* Python 3.13
* Dependencies in `requirements.txt` (`matplotlib`, `pytest`)

Everything else is from the Python standard library
(`sqlite3`, `multiprocessing`, `dataclasses`, `itertools`, ...).

## Setup

```bash
git clone https://github.com/Elisbett/tuple-responsibility.git
cd tuple-responsibility

# Create a virtual environment (recommended)
python -m venv venv
venv\Scripts\activate            # Windows
# source venv/bin/activate       # macOS / Linux

pip install -r requirements.txt
```

## Building the datasets

Before running anything else, generate the seven SQLite datasets used
in the experiments:

```bash
python create_synthetic_datasets.py
```

This creates files in `data/synthetic/`:


| File | Size | Description |
|------|---------|-------------|
| `synth_small.db` | 8 | small R⨝S join, hand-verified ground truth |
| `synth_medium.db` | 12 | medium R⨝S join |
| `synth_large.db` | 16 | larger R⨝S join, Naive runs in seconds |
| `synth_xlarge.db` | 20 | size at which Naive becomes impractical |
| `synth_join3.db` | 12 | three-way join, structural variation |
| `synth_dense.db` | 14 | high minterm overlap, stress-tests the cache |
| `imdb_burton.db` | 9 | reproduces the IMDB example from Meliou 2010a |

All datasets are deterministic: re-running the script produces identical
files.

## Running the tests

```bash
pytest tests/
```

This should report **51 passed** in roughly two minutes. The suite covers:

* SQLite backend (schema introspection, `_disabled` mechanism, tuple lookup);
* query rewriter (added `_disabled = 0` clauses, rejection of unsupported keywords);
* hand-computed responsibility scores on the smallest dataset;
* reproduction of the published Meliou et al. 2010a Figure 2(b) on IMDB;
* equivalence of all four levels on every dataset
  (all levels must produce identical responsibility scores and
  minimum contingency sizes).

## Running a single dataset

A quick sanity check before the full benchmark:

```bash
python run_on_dataset.py synth_small
```

Prints the timings, the speedups vs Level 1, and the responsibility
ranking. Replace `synth_small` with any dataset name from the table
above.

## Reproducing the experiments

Three commands produce everything reported in the thesis chapter:

```bash
python benchmark.py        # writes experiments/results/benchmark.csv
python plot_results.py     # writes the four PNG figures and prints a summary
python imdb_1_over_n.py    # prints the comparison table of §6.5
```

`benchmark.py` runs every level on every dataset five times and records
median, min, and max algorithm time. Expected total wall-clock time is
roughly **2.5 hours**, dominated by `synth_xlarge` which is run for
each of the four levels.

`plot_results.py` reads `benchmark.csv` and writes:

* `benchmark_runtime_median.png` — runtime vs |D_n|, log y, with error bars;
* `benchmark_speedup_median.png` — speedup vs Level 1, with the Naive baseline;
* `benchmark_runtime_mean.png` and `benchmark_speedup_mean.png` —
  the same plots using approximate means rather than medians,
  as a sanity check.

It also prints a human-readable summary table to standard output.

`imdb_1_over_n.py` compares the responsibility scores produced by our
implementation against two simpler provenance-based scoring rules
(lineage frequency and 1/N counting). The output supports the
ranking-divergence discussion in §6.5.

## License

MIT. See `LICENSE`.

## Author and supervisor

Elisaveta Serikova, University of Tartu, 2026.
Supervised by Miika Hannula, PhD.
```
