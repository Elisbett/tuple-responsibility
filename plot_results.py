"""Generate runtime and speedup figures for §6 of the thesis.

Reads experiments/results/benchmark.csv and writes four PNGs into the
same directory:
  - benchmark_runtime_median.png : median runtime vs |D_n| (log y, error bars)
  - benchmark_speedup_median.png : speedup vs Naive using medians
  - benchmark_runtime_mean.png   : mean runtime vs |D_n| (log y, error bars)
  - benchmark_speedup_mean.png   : speedup vs Naive using means

The mean variants are computed from min/median/max as
  mean ~= (min + 2*median + max) / 4
to avoid re-running the benchmark; this is exact when the five runs are
approximately symmetric and is a documented approximation in the
implementation notes.

Each level gets a distinct colour and marker so the lines remain
distinguishable. Error bars show the [min, max] range observed across
the 5 repeats of each (dataset, level) pair.

Also prints a readable comparison table to stdout summarising the
benchmark for easy copying into notes or the thesis text.

Usage:
    python plot_results.py
"""
from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt

# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------
CSV_PATH = Path("experiments/results/benchmark.csv")
OUTDIR = Path("experiments/results")

LEVEL_STYLES = {
    "NaiveComputer":            {"label": "Level 1 (Naive)",        "marker": "o", "color": "#444444"},
    "EarlyTerminationComputer": {"label": "Level 2 (Early Term.)",  "marker": "s", "color": "#1f77b4"},
    "CachedComputer":           {"label": "Level 3 (Cached)",       "marker": "^", "color": "#2ca02c"},
    "ParallelComputer":         {"label": "Level 4 (Parallel)",     "marker": "D", "color": "#d62728"},
}

DATASET_ORDER = [
    "synth_small",
    "imdb_burton",
    "synth_medium",
    "synth_join3",
    "synth_dense",
    "synth_large",
    "synth_xlarge",
]


def load_rows() -> list[dict]:
    """Load all CSV rows as plain dicts with numeric fields parsed."""
    rows: list[dict] = []
    with CSV_PATH.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({
                "dataset": row["dataset"],
                "n_endogenous": int(row["n_endogenous"]),
                "level": row["level"],
                "repeats": int(row["repeats"]),
                "median": float(row["alg_time_median_ms"]),
                "min": float(row["alg_time_min_ms"]),
                "max": float(row["alg_time_max_ms"]),
                "total_median": float(row["total_time_median_ms"]),
            })
    return rows


def approximate_mean(row: dict) -> float:
    """Approximate mean from min/median/max.

    The benchmark stored only min, median, and max across 5 repeats —
    not the individual measurements. We approximate the mean as a
    weighted combination that is exact for symmetric distributions and
    has small bias otherwise:

        mean ~= (min + 2 * median + max) / 4

    This is the standard approximation used when only quartile-style
    statistics are available; we record it explicitly so the reader
    knows the mean line is not an independent measurement.
    """
    return (row["min"] + 2 * row["median"] + row["max"]) / 4


def build_series(
    rows: list[dict],
    stat_key: str,
) -> dict[str, list[tuple[int, float, float, float, str]]]:
    """For each level, return a sorted list of (n, stat, min, max, dataset)."""
    by_level: dict[str, list[tuple[int, float, float, float, str]]] = defaultdict(list)
    for row in rows:
        if stat_key == "median":
            stat = row["median"]
        elif stat_key == "mean":
            stat = approximate_mean(row)
        else:
            raise ValueError(f"Unknown stat_key: {stat_key}")
        by_level[row["level"]].append(
            (row["n_endogenous"], stat, row["min"], row["max"], row["dataset"])
        )
    for level in by_level:
        by_level[level].sort(key=lambda r: r[0])
    return by_level


def plot_runtime(
    series: dict[str, list[tuple[int, float, float, float, str]]],
    title_suffix: str,
    out_path: Path,
) -> None:
    """Runtime (log y) vs |D_n|, with [min,max] error bars."""
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    for level in LEVEL_STYLES:
        if level not in series:
            continue
        xs   = [r[0] for r in series[level]]
        ys   = [r[1] for r in series[level]]
        mins = [r[2] for r in series[level]]
        maxs = [r[3] for r in series[level]]
        lower = [y - m for y, m in zip(ys, mins)]
        upper = [M - y for y, M in zip(ys, maxs)]
        style = LEVEL_STYLES[level]
        ax.errorbar(
            xs, ys,
            yerr=[lower, upper],
            marker=style["marker"],
            linestyle="-",
            color=style["color"],
            label=style["label"],
            markersize=7,
            linewidth=2.0,
            markeredgewidth=1.2,
            markerfacecolor=style["color"],
            capsize=3,
            elinewidth=1.0,
        )

    ax.set_yscale("log")
    ax.set_xlabel("|D_n| (number of endogenous tuples)", fontsize=11)
    ax.set_ylabel(f"Algorithm time, {title_suffix} (ms, log scale)", fontsize=11)
    ax.grid(True, which="both", axis="y", linestyle=":", linewidth=0.5, alpha=0.6)
    ax.grid(True, which="major", axis="x", linestyle=":", linewidth=0.5, alpha=0.6)
    ax.legend(loc="upper left", frameon=True, fontsize=10)

    all_ns = sorted({r[0] for level in series for r in series[level]})
    ax.set_xticks(all_ns)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")


def plot_speedup(
    series: dict[str, list[tuple[int, float, float, float, str]]],
    title_suffix: str,
    out_path: Path,
) -> None:
    """Speedup vs |D_n| against the Naive baseline of the same stat."""
    naive_by_dataset: dict[str, float] = {}
    for n, stat, _, _, dataset in series["NaiveComputer"]:
        naive_by_dataset[dataset] = stat

    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    plotted = ["EarlyTerminationComputer", "CachedComputer", "ParallelComputer"]
    for level in plotted:
        if level not in series:
            continue
        xs, ys = [], []
        for n, stat, _, _, dataset in series[level]:
            base = naive_by_dataset.get(dataset)
            if base is None or stat <= 0:
                continue
            xs.append(n)
            ys.append(base / stat)
        style = LEVEL_STYLES[level]
        ax.plot(
            xs, ys,
            marker=style["marker"],
            linestyle="-",
            color=style["color"],
            label=style["label"],
            markersize=7,
            linewidth=2.0,
            markeredgewidth=1.2,
            markerfacecolor=style["color"],
        )

    ax.axhline(1.0, color="#888888", linewidth=0.8, linestyle="--", alpha=0.7)
    ax.text(
        ax.get_xlim()[1], 1.07, "Naive baseline",
        fontsize=8, ha="right", va="bottom", alpha=0.7, color="#666666",
    )

    ax.set_xlabel("|D_n| (number of endogenous tuples)", fontsize=11)
    ax.set_ylabel(f"Speedup vs Level 1 ({title_suffix})", fontsize=11)
    ax.grid(True, axis="y", linestyle=":", linewidth=0.5, alpha=0.6)
    ax.grid(True, axis="x", linestyle=":", linewidth=0.5, alpha=0.6)
    ax.legend(loc="upper left", frameon=True, fontsize=10)

    all_ns = sorted({r[0] for level in series for r in series[level]})
    ax.set_xticks(all_ns)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")


def fmt_ms(v: float) -> str:
    """Pretty-print a millisecond value across many orders of magnitude."""
    if v < 1000:
        return f"{v:>10.1f} ms"
    if v < 60_000:
        return f"{v / 1000:>9.2f} s "
    minutes = v / 60_000
    return f"{minutes:>9.2f} m "


def print_summary_table(rows: list[dict]) -> None:
    """Print a single table that ties everything together per dataset."""
    by_dataset: dict[str, dict[str, dict]] = defaultdict(dict)
    for r in rows:
        by_dataset[r["dataset"]][r["level"]] = r

    print()
    print("=" * 100)
    print("Benchmark summary (5 repeats per cell)")
    print("=" * 100)
    header_row = (
        f"{'Dataset':<14} {'|D_n|':>5}  {'Level':<27} "
        f"{'median':>13} {'mean':>13} "
        f"{'speedup_med':>12} {'speedup_mean':>13}"
    )
    print(header_row)
    print("-" * 100)

    for dataset in DATASET_ORDER:
        if dataset not in by_dataset:
            continue
        levels = by_dataset[dataset]
        naive = levels.get("NaiveComputer")
        if naive is None:
            continue
        naive_med = naive["median"]
        naive_mean = approximate_mean(naive)
        n = naive["n_endogenous"]
        for level_name in ["NaiveComputer", "EarlyTerminationComputer",
                           "CachedComputer", "ParallelComputer"]:
            if level_name not in levels:
                continue
            r = levels[level_name]
            med = r["median"]
            mean = approximate_mean(r)
            if level_name == "NaiveComputer":
                sp_med = "—"
                sp_mean = "—"
            else:
                sp_med = f"{naive_med / med:>10.2f}x" if med > 0 else "—"
                sp_mean = f"{naive_mean / mean:>10.2f}x" if mean > 0 else "—"
            print(
                f"{dataset:<14} {n:>5}  {level_name:<27} "
                f"{fmt_ms(med):>13} {fmt_ms(mean):>13} "
                f"{sp_med:>12} {sp_mean:>13}"
            )
        print()
    print("=" * 100)
    print(
        "Note: mean is approximated from (min + 2*median + max) / 4, since "
        "the benchmark records only those three statistics, not all five runs."
    )


def main() -> None:
    if not CSV_PATH.exists():
        raise SystemExit(f"Cannot find {CSV_PATH}. Run benchmark.py first.")

    print(f"Reading {CSV_PATH}...")
    rows = load_rows()
    print(f"  loaded {len(rows)} rows")

    print("Plotting (median)...")
    series_median = build_series(rows, "median")
    plot_runtime(series_median, "median", OUTDIR / "benchmark_runtime_median.png")
    plot_speedup(series_median, "median", OUTDIR / "benchmark_speedup_median.png")

    print("Plotting (mean, approximated)...")
    series_mean = build_series(rows, "mean")
    plot_runtime(series_mean, "mean", OUTDIR / "benchmark_runtime_mean.png")
    plot_speedup(series_mean, "mean", OUTDIR / "benchmark_speedup_mean.png")

    print_summary_table(rows)
    print("\nDone.")


if __name__ == "__main__":
    main()