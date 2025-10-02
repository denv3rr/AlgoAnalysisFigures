# Algorithm Analysis Figures

The following Python script (figures.py) was written to generate graphs and benchmark figures for a Data Structures paper on Algorithm Analysis.

The script requires Python 3.9+ and the following packages:

- numpy
- matplotlib

---

## Usage

- Clone the [AlgoAnalysisFigures/](https://github.com/denv3rr/AlgoAnalysisFigures) repo into any folder.
- Install packages. Run:
  - `pip install -r requirements.txt`
- Run the script: `python figures.py`

> [!Note]
> This will create a `figures/` folder with all the PNG files (and CSVs if you keep benchmarks on).

---

## Expectation

```bash
$ python figures.py
Saved figures\fig_growth_rates.png
Saved figures\fig_bubble_sort.png
Saved figures\fig_search.png
Saved figures\fig_sorting_sim.png
Saved figures\fig_binary_halving.png

Running benchmarks on this machine (this may take a bit)...
[sort-bench] n=256  bubble=0.001999s  merge=0.000227s  timsort=0.000010s
[sort-bench] n=512  bubble=0.006741s  merge=0.000514s  timsort=0.000021s
[sort-bench] n=1024  bubble=0.032419s  merge=0.001078s  timsort=0.000042s
[sort-bench] n=2048  bubble=nans  merge=0.002309s  timsort=0.000101s
Saved figures\bench_sort.csv
Saved figures\fig_sorting_bars.png
[search-bench] n=10000  linear=0.000020s  binary=0.000002s
[search-bench] n=100000  linear=0.000429s  binary=0.000003s
[search-bench] n=1000000  linear=0.029770s  binary=0.000004s
Saved figures\bench_search.csv
Saved figures\fig_searching_bars.png

Done. All figures are in the ./figures directory.

```

---

## Glossary

### Main

- Big-O family overview → `fig_growth_rates.png`

- Little-o vs Big-O intuition → also `fig_growth_rates.png`

- Binary search intuition → `fig_binary_halving.png`

### Efficiency Analysis

- Bubble Sort quadratic proof → `fig_bubble_sort.png`

- Growth-rate separation → `fig_growth_rates.png`

### Benchmarking / Results

- Sorting results (real machine) → `fig_sorting_bars.png`

- Searching results (real machine) → `fig_searching_bars.png`

- You can optionally include the simulated curve figure `fig_sorting_sim.png` next to the bar chart.

---
