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
