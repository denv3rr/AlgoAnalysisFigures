"""

figures.py
---------

Generates publication-ready figures for an Algorithm Analysis paper and (optionally)
runs benchmarks on your computer to produce runtime comparison charts.

Requires: numpy, matplotlib

Figures produced (saved to ./figures/):
  1) fig_growth_rates.png         — log n, n, n log n, n^2 (log–log)
  2) fig_bubble_sort.png          — Bubble Sort comparisons (quadratic)
  3) fig_search.png               — Linear vs Binary Search steps (log–log)
  4) fig_sorting_sim.png          — Simulated sorting runtimes (Bubble, Merge, Timsort)
  5) fig_binary_halving.png       — Binary Search halving of interval (log y)
  6) fig_sorting_bars.png         — (Optional) Real benchmark: bar chart for Bubble/Merge/Timsort
  7) fig_searching_bars.png       — (Optional) Real benchmark: bar chart for Linear vs Binary

Benchmark outputs (if enabled):
  - ./figures/bench_sort.csv
  - ./figures/bench_search.csv

USAGE:
  python figures.py
  (By default, real benchmarks are ENABLED but Bubble Sort is limited to n <= 1024.)

NOTES:
  You can toggle RUN_BENCHMARKS to False if you only want the figures without real timing.

"""

import os
import time
import math
import random
import numpy as np
import matplotlib.pyplot as plt
import textwrap
import csv

# -----------------------------
# SETTINGS
# -----------------------------
RUN_BENCHMARKS = True   # Set to False if you want to skip real machine timing
FIG_DIR = "figures"
DPI = 300

# Make output dir
os.makedirs(FIG_DIR, exist_ok=True)

# -----------------------------
# Helper: annotate with wrapped text
# -----------------------------
def add_wrapped_text(ax, text, xy, xytext, fontsize=9, arrow=True):
    wrapped = "\n".join(textwrap.wrap(text, width=48))
    if arrow:
        ax.annotate(
            wrapped,
            xy=xy,
            xytext=xytext,
            textcoords="data",
            arrowprops=dict(arrowstyle="->"),
            fontsize=fontsize,
            ha="left",
            va="top",
        )
    else:
        ax.text(xytext[0], xytext[1], wrapped, fontsize=fontsize, ha="left", va="top")

# =========================================================
# FIGURE 1: Growth-rate comparison on log–log axes
# =========================================================
def figure_growth_rates():
    x = np.logspace(1, 5, 600)  # n = 10 .. 100000
    f_log = np.log2(x)
    f_n = x
    f_nlogn = x * np.log2(x)
    f_n2 = x**2

    plt.figure()
    plt.loglog(x, f_log, label='log n (base 2)')
    plt.loglog(x, f_n, label='n')
    plt.loglog(x, f_nlogn, label='n log n (base 2)')
    plt.loglog(x, f_n2, label='n^2')
    plt.xlabel('Input size n')
    plt.ylabel('Function value (log scale)')
    plt.title('Figure 1. Asymptotic Growth Rates: log n, n, n log n, n^2')
    plt.legend()
    plt.grid(True, which="both", ls="--")
    ax = plt.gca()
    add_wrapped_text(
        ax,
        "As n grows large, curves separate.\n"
        "Quadratic (n^2) rises far faster than n log n.\n"
        "This is why O(n^2) algorithms become impractical.",
        xy=(1e3, 1e6),
        xytext=(1.5e2, 5e7),
    )
    plt.tight_layout()
    out = os.path.join(FIG_DIR, "fig_growth_rates.png")
    plt.savefig(out, dpi=DPI)
    plt.close()
    print(f"Saved {out}")

# =========================================================
# FIGURE 2: Bubble Sort comparisons (quadratic growth)
# =========================================================
def figure_bubble_sort_quadratic():
    n = np.arange(1, 3001)  # up to 3000 elements
    bubble_comparisons = n * (n - 1) / 2

    plt.figure()
    plt.plot(n, bubble_comparisons, label='Comparisons ≈ n(n-1)/2')
    plt.xlabel('Input size n')
    plt.ylabel('Number of comparisons')
    plt.title('Figure 2. Bubble Sort Worst-Case Comparisons (Quadratic Growth)')
    plt.legend()
    plt.grid(True)
    ax = plt.gca()
    add_wrapped_text(
        ax,
        "Quadratic growth: doubling n roughly quadruples comparisons.\n"
        "This is why Bubble Sort is only suitable for tiny inputs.",
        xy=(2000, bubble_comparisons[1999]),
        xytext=(800, bubble_comparisons.max()*0.55),
    )
    plt.tight_layout()
    out = os.path.join(FIG_DIR, "fig_bubble_sort.png")
    plt.savefig(out, dpi=DPI)
    plt.close()
    print(f"Saved {out}")

# =========================================================
# FIGURE 3: Linear vs Binary Search steps (log–log)
# =========================================================
def figure_search_steps():
    n = np.logspace(2, 6, 120, dtype=int)  # 100 .. 1,000,000
    linear_steps = n / 2.0  # avg case
    binary_steps = np.log2(n)

    plt.figure()
    plt.plot(n, linear_steps, label='Linear Search (~n/2 steps average)')
    plt.plot(n, binary_steps, label='Binary Search (log2 n steps worst-case)')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Array size n (log scale)')
    plt.ylabel('Steps (log scale)')
    plt.title('Figure 3. Linear vs Binary Search: Steps vs Input Size')
    plt.legend()
    plt.grid(True, which="both", ls="--")
    ax = plt.gca()
    add_wrapped_text(
        ax,
        "Binary Search halves the search space each step.\n"
        "At n = 1,000,000 it needs ~20 comparisons,\n"
        "while Linear Search averages ~500,000.",
        xy=(1e6, np.log2(1e6)),
        xytext=(3e3, 5e3),
    )
    plt.tight_layout()
    out = os.path.join(FIG_DIR, "fig_search.png")
    plt.savefig(out, dpi=DPI)
    plt.close()
    print(f"Saved {out}")

# =========================================================
# FIGURE 4: Simulated sorting runtimes (curves)
# =========================================================
def figure_sorting_simulated():
    n = np.logspace(2, 7, 200, dtype=int)  # 100 .. 10,000,000
    # Scale the functions so they appear nicely on the same chart
    bubble_time = (n**2) / (1e8)        # scaled
    merge_time = (n * np.log2(n)) / (5e6)
    timsort_time = (n * np.log2(n)) / (8e6)  # slightly smaller constant

    plt.figure()
    plt.loglog(n, bubble_time, label='Bubble Sort ~ n^2 (scaled)')
    plt.loglog(n, merge_time, label='Merge Sort ~ n log n (scaled)')
    plt.loglog(n, timsort_time, label='Timsort ~ n log n (smaller constant)')
    plt.xlabel('Input size n (log scale)')
    plt.ylabel('Relative runtime (scaled, log scale)')
    plt.title('Figure 4. Sorting Runtime Illustration (Relative, Simulated)')
    plt.legend()
    plt.grid(True, which="both", ls="--")
    ax = plt.gca()
    add_wrapped_text(
        ax,
        "Both Merge Sort and Timsort scale as n log n,\n"
        "but Timsort may be faster due to a smaller constant\n"
        "and real-world optimizations.\n"
        "Bubble Sort (~n^2) blows up rapidly.",
        xy=(2e5, (2e5*np.log2(2e5))/(5e6)),
        xytext=(2.5e2, 2e2),
    )
    plt.tight_layout()
    out = os.path.join(FIG_DIR, "fig_sorting_sim.png")
    plt.savefig(out, dpi=DPI)
    plt.close()
    print(f"Saved {out}")

# =========================================================
# FIGURE 5: Binary Search halving demonstration
# =========================================================
def figure_binary_halving():
    n0 = 1024  # starting interval length
    k = np.arange(0, 21)  # up to 20 iterations
    interval = n0 / (2**k)

    plt.figure()
    plt.plot(k, interval, marker='o', label='Remaining interval = 1024 / 2^k')
    plt.yscale('log')
    plt.xlabel('Iteration k')
    plt.ylabel('Interval length (log scale)')
    plt.title('Figure 5. Binary Search Halves the Interval Each Iteration')
    plt.legend()
    plt.grid(True, which="both", ls="--")
    ax = plt.gca()
    add_wrapped_text(
        ax,
        "After k steps the interval is 1024 / 2^k.\n"
        "Terminate when it falls below 1 → steps ≈ log2(n).",
        xy=(10, n0/(2**10)),
        xytext=(1.5, n0/8),
    )
    plt.tight_layout()
    out = os.path.join(FIG_DIR, "fig_binary_halving.png")
    plt.savefig(out, dpi=DPI)
    plt.close()
    print(f"Saved {out}")

# =========================================================
# OPTIONAL: Real benchmarks (Sorting + Searching)
# =========================================================

def bubble_sort(a):
    n = len(a)
    swapped = True
    while swapped:
        swapped = False
        for i in range(1, n):
            if a[i-1] > a[i]:
                a[i-1], a[i] = a[i], a[i-1]
                swapped = True
        n -= 1
    return a

def merge_sort(a):
    if len(a) <= 1:
        return a
    mid = len(a)//2
    left = merge_sort(a[:mid])
    right = merge_sort(a[mid:])
    i=j=0
    out = []
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            out.append(left[i]); i+=1
        else:
            out.append(right[j]); j+=1
    out.extend(left[i:]); out.extend(right[j:])
    return out

def linear_search(a, key):
    for i, v in enumerate(a):
        if v == key:
            return i
    return -1

def binary_search(a, key):
    lo, hi = 0, len(a)-1
    while lo <= hi:
        mid = lo + (hi - lo)//2
        if a[mid] == key:
            return mid
        elif a[mid] < key:
            lo = mid + 1
        else:
            hi = mid - 1
    return -1

def time_func(fn, *args, repeats=3):
    total = 0.0
    for _ in range(repeats):
        start = time.perf_counter()
        fn(*args)
        total += (time.perf_counter() - start)
    return total / repeats

def figure_sorting_bars_and_csv():
    # Note: Bubble Sort limited to 1024 to avoid super-long runs
    sizes = [256, 512, 1024, 2048]
    rows = []
    for n in sizes:
        base = [random.randint(0, 10_000_000) for _ in range(n)]
        # Bubble: only up to 1024 to keep runtime sane
        if n <= 1024:
            t_bubble = time_func(lambda arr: bubble_sort(arr.copy()), base.copy(), repeats=3)
        else:
            t_bubble = float('nan')

        t_merge = time_func(lambda arr: merge_sort(arr.copy()), base.copy(), repeats=3)
        t_tim = time_func(lambda arr: sorted(arr.copy()), base.copy(), repeats=3)

        rows.append((n, t_bubble, t_merge, t_tim))
        print(f"[sort-bench] n={n}  bubble={t_bubble:.6f}s  merge={t_merge:.6f}s  timsort={t_tim:.6f}s")

    # Save CSV
    csv_path = os.path.join(FIG_DIR, "bench_sort.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["n", "bubble_seconds", "merge_seconds", "timsort_seconds"])
        for r in rows:
            w.writerow(r)
    print(f"Saved {csv_path}")

    # Plot bar chart (grouped per n)
    # One chart per figure: we will show three bars per size (bubble may be NaN for 2048)
    labels = [str(n) for n in sizes]
    bubble_vals = [r[1] for r in rows]
    merge_vals = [r[2] for r in rows]
    tim_vals   = [r[3] for r in rows]

    x = np.arange(len(labels))
    width = 0.25

    plt.figure()
    plt.bar(x - width, bubble_vals, width, label='Bubble (O(n^2))')
    plt.bar(x,         merge_vals, width, label='Merge (O(n log n))')
    plt.bar(x + width, tim_vals,   width, label='Timsort (Python sorted)')
    plt.xticks(x, labels)
    plt.ylabel('Seconds (lower is better)')
    plt.xlabel('Input size n')
    plt.title('Figure 6. Sorting Runtime (Real Benchmark on This Machine)')
    plt.legend()
    plt.grid(axis='y', ls='--', alpha=0.5)
    plt.tight_layout()
    out = os.path.join(FIG_DIR, "fig_sorting_bars.png")
    plt.savefig(out, dpi=DPI)
    plt.close()
    print(f"Saved {out}")

def figure_searching_bars_and_csv():
    sizes = [10_000, 100_000, 1_000_000]
    rows = []
    for n in sizes:
        # Prepare sorted array for binary search
        arr = list(range(n))
        # Random key that may or may not exist (50/50)
        key = random.randint(0, n)
        # Time linear search (average-case)
        t_lin = time_func(lambda a, k: linear_search(a, k), arr, key, repeats=3)
        # Time binary search (worst-case ~ log n)
        t_bin = time_func(lambda a, k: binary_search(a, k), arr, key, repeats=5)
        rows.append((n, t_lin, t_bin))
        print(f"[search-bench] n={n}  linear={t_lin:.6f}s  binary={t_bin:.6f}s")

    # Save CSV
    csv_path = os.path.join(FIG_DIR, "bench_search.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["n", "linear_seconds", "binary_seconds"])
        for r in rows:
            w.writerow(r)
    print(f"Saved {csv_path}")

    # Bar chart
    labels = [f"{n:,}" for n in sizes]
    lin_vals = [r[1] for r in rows]
    bin_vals = [r[2] for r in rows]

    x = np.arange(len(labels))
    width = 0.35

    plt.figure()
    plt.bar(x - width/2, lin_vals, width, label='Linear Search (O(n))')
    plt.bar(x + width/2, bin_vals, width, label='Binary Search (O(log n))')
    plt.xticks(x, labels)
    plt.ylabel('Seconds (lower is better)')
    plt.xlabel('Array size n')
    plt.title('Figure 7. Searching Runtime (Real Benchmark on This Machine)')
    plt.legend()
    plt.grid(axis='y', ls='--', alpha=0.5)
    plt.tight_layout()
    out = os.path.join(FIG_DIR, "fig_searching_bars.png")
    plt.savefig(out, dpi=DPI)
    plt.close()
    print(f"Saved {out}")

# -----------------------------
# MAIN
# -----------------------------
if __name__ == "__main__":
    # Always-generate explanatory figures (no benchmarking needed)
    figure_growth_rates()
    figure_bubble_sort_quadratic()
    figure_search_steps()
    figure_sorting_simulated()
    figure_binary_halving()

    # Optional: run real benchmarks and produce bar charts + CSVs
    if RUN_BENCHMARKS:
        print("\nRunning real benchmarks on this machine (this may take a bit)...")
        figure_sorting_bars_and_csv()
        figure_searching_bars_and_csv()

    print("\nDone. All figures are in the ./figures directory.")
