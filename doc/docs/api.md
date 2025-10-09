# API Reference

This page documents the `plot_posthoc_test` API, organized by user-friendly aliases and their corresponding functions.

---

## Recommended API (User-Friendly Aliases)

These aliased functions provide a clean, intuitive API for common operations. **Use these for new projects.**

### Core Functions

#### `annotate()`
**Alias for:** `main_run_posthoc_tests_and_get_hue_loc_df()`

Run post-hoc statistical tests and get hue locations for all x-axis ticks.

```python
import plot_posthoc_test as ppt

# Run tests and get results
posthoc_df = ppt.annotate(ax, plot_params, ax, comparisons, test_name='MWU')
```

::: plot_posthoc_test.main_run_posthoc_tests_and_get_hue_loc_df
    options:
      show_root_heading: false
      show_source: false

---

#### `plot_significance()` / `plot_bars()`
**Alias for:** `plot_sig_bars_w_comp_df_tight()`

Plot significance bars on the axis using a comparison DataFrame.

```python
import plot_posthoc_test as ppt

# Plot significance bars for significant results
ppt.plot_significance(ax, posthoc_df[posthoc_df['pvalue'] < 0.05])

# Alternative alias
ppt.plot_bars(ax, posthoc_df[posthoc_df['pvalue'] < 0.05])
```

::: plot_posthoc_test.plot_sig_bars_w_comp_df_tight
    options:
      show_root_heading: false
      show_source: false

---

#### `stars()`
**Alias for:** `convert_pvalue_to_asterisks()`

Convert p-values to asterisk notation for significance levels.

```python
import plot_posthoc_test as ppt

# Convert p-value to stars
stars = ppt.stars(0.001)  # Returns "***"
stars = ppt.stars(0.04)   # Returns "*"
```

::: plot_posthoc_test.convert_pvalue_to_asterisks
    options:
      show_root_heading: false
      show_source: false

---

#### `run_tests()`
**Alias for:** `run_posthoc_tests_on_all_ax_ticks()`

Run post-hoc tests on all x-axis tick groups without location inference.

```python
import plot_posthoc_test as ppt

# Run tests on all ticks
results_df = ppt.run_tests(df, plot_params, comparisons, test_name='MWU')
```

::: plot_posthoc_test.run_posthoc_tests_on_all_ax_ticks
    options:
      show_root_heading: false
      show_source: false

---

## Original Function Names (Backward Compatibility)

These are the original function names that remain available for backward compatibility. **New code should use the aliases above.**

### Main Statistical Functions

::: plot_posthoc_test.plot_stat_annotate
    options:
      show_root_heading: true
      show_source: false
      members:
        - main_run_posthoc_tests_and_get_hue_loc_df
        - run_posthoc_tests_on_all_ax_ticks
        - run_posthoc_test_on_tick_hue_groups
        - plot_sig_bars_w_comp_df_tight
        - convert_pvalue_to_asterisks
        - get_hue_loc_on_axis

---

## Axis Inference Functions

Functions for extracting location information from matplotlib axes.

::: plot_posthoc_test.ax_inference
    options:
      show_root_heading: true
      show_source: false

---

## Utility Modules

### Statistical Tests

::: plot_posthoc_test.utils.stat_tests
    options:
      show_root_heading: true
      show_source: false

### Axis Modifier Functions

::: plot_posthoc_test.utils.ax_modifier_functions
    options:
      show_root_heading: true
      show_source: false

### Helper Functions

::: plot_posthoc_test.utils.get_match_index_in_iterable
    options:
      show_root_heading: true
      show_source: false

---

## Quick Reference Table

| User-Friendly Alias | Original Function Name | Purpose |
|---------------------|------------------------|---------|
| `annotate()` | `main_run_posthoc_tests_and_get_hue_loc_df()` | Run tests and get locations |
| `plot_significance()` / `plot_bars()` | `plot_sig_bars_w_comp_df_tight()` | Plot significance bars |
| `stars()` | `convert_pvalue_to_asterisks()` | Convert p-values to asterisks |
| `run_tests()` | `run_posthoc_tests_on_all_ax_ticks()` | Run tests without location inference |

---

## Complete Import Example

```python
import plot_posthoc_test as ppt
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Your plotting code
fig, ax = plt.subplots()
sns.pointplot(data=df, x='condition', y='value', hue='group', ax=ax)

# Define comparisons
comparisons = [('control', 'treatment'), ('control', 'other')]

# User-friendly API
plot_params = {'data': df, 'x': 'condition', 'y': 'value', 'hue': 'group'}
posthoc_df = ppt.annotate(ax, plot_params, ax, comparisons, test_name='MWU')
ppt.plot_significance(ax, posthoc_df[posthoc_df['pvalue'] < 0.05])

plt.show()
```
