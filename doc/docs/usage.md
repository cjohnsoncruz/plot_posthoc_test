# Usage Guide

This guide demonstrates how to use `plot_posthoc_test` with practical examples using the **user-friendly API** introduced in v0.1.0.

## Installation

```bash
# From GitHub (Development Mode)
git clone https://github.com/cjohnsoncruz/plot_posthoc_test.git
cd plot_posthoc_test
pip install -e .
```

---

## Basic Example

```python
import plot_posthoc_test as ppt
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Sample data
df = pd.DataFrame({
    'condition': ['control', 'control', 'treatment', 'treatment'] * 20,
    'group': ['Group_A', 'Group_B'] * 40,
    'value': np.random.randn(80) + np.repeat([0, 0.5, 1, 1.5], 20)
})

# Create plot
fig, ax = plt.subplots(figsize=(8, 6))
sns.pointplot(data=df, x='condition', y='value', hue='group', ax=ax, errorbar='se')

# Define plot parameters
plot_params = {
    'data': df,
    'x': 'condition',
    'y': 'value',
    'hue': 'group'
}

# Define which groups to compare
comparisons = [('Group_A', 'Group_B')]

# Run statistical tests and get results
posthoc_df = ppt.annotate(
    ax_input=ax,
    plot_params=plot_params,
    plot_obj=ax,
    preset_comparisons=comparisons,
    test_name='MWU',  # Mann-Whitney U test
    detect_error_bar=True
)

# Plot significance bars (only for p < 0.05)
ppt.plot_significance(ax, posthoc_df[posthoc_df['pvalue'] < 0.05])

plt.tight_layout()
plt.show()

# Print results with asterisks
for _, row in posthoc_df.iterrows():
    print(f"{row['group_1']} vs {row['group_2']}: p={row['pvalue']:.4f} {ppt.stars(row['pvalue'])}")
```

---

## User-Friendly API Overview

### Core Functions

#### `ppt.annotate()` - Run Tests and Get Locations

The main function for running statistical tests with automatic hue location detection.

```python
posthoc_df = ppt.annotate(
    ax_input=ax,              # Input axis to analyze
    plot_params=plot_params,  # Dict with 'data', 'x', 'y', 'hue'
    plot_obj=ax,              # Plot object (usually same as ax_input)
    preset_comparisons=comparisons,  # List of tuples: [('A', 'B'), ...]
    test_name='MWU',          # 'MWU', 'cohen_d', 'permutation', 't-test'
    detect_error_bar=True     # Auto-detect error bars for placement
)
```

**Returns:** DataFrame with columns including `group_1`, `group_2`, `pvalue`, `x_loc`, `y_loc`

---

#### `ppt.plot_significance()` - Plot Significance Bars

Plot horizontal bars with significance annotations.

```python
ppt.plot_significance(
    ax,                                          # Axis to plot on
    posthoc_df[posthoc_df['pvalue'] < 0.05],   # Filtered results DataFrame
    bar_height=0.02,                            # Bar height (optional)
    text_offset=0.01                            # Text offset from bar (optional)
)
```

**Alternative alias:** `ppt.plot_bars()` does the same thing.

---

#### `ppt.stars()` - Convert P-values to Asterisks

Convert p-values to standard significance notation.

```python
print(ppt.stars(0.0001))  # "****"
print(ppt.stars(0.001))   # "***"
print(ppt.stars(0.01))    # "**"
print(ppt.stars(0.05))    # "*"
print(ppt.stars(0.1))     # "ns"
```

**Thresholds:**
- p < 0.0001: `****`
- p < 0.001: `***`
- p < 0.01: `**`
- p < 0.05: `*`
- p ≥ 0.05: `ns` (not significant)

---

#### `ppt.run_tests()` - Run Tests Without Location Inference

Run statistical tests without automatic axis location detection.

```python
results_df = ppt.run_tests(
    df=df,
    plot_params=plot_params,
    preset_comparisons=comparisons,
    test_name='MWU'
)
```

**Returns:** DataFrame with statistical results (no `x_loc`, `y_loc` columns)

---

## Available Statistical Tests

### Mann-Whitney U Test (`'MWU'`)

Non-parametric test for independent samples. Use when data may not be normally distributed.

```python
posthoc_df = ppt.annotate(ax, plot_params, ax, comparisons, test_name='MWU')
```

### Cohen's d (`'cohen_d'`)

Effect size measure with confidence intervals. Useful for quantifying magnitude of differences.

```python
posthoc_df = ppt.annotate(ax, plot_params, ax, comparisons, test_name='cohen_d')
```

### Permutation Test (`'permutation'`)

Non-parametric resampling test. Robust alternative to parametric tests.

```python
posthoc_df = ppt.annotate(ax, plot_params, ax, comparisons, test_name='permutation')
```

### Two-Sample T-Test (`'t-test'`)

Parametric test using Welch's t-test (unequal variance). Use for normally distributed data.

```python
posthoc_df = ppt.annotate(ax, plot_params, ax, comparisons, test_name='t-test')
```

---

## Advanced Usage

### Using Utility Functions Directly

Access statistical test utilities for custom analyses:

```python
from plot_posthoc_test.utils import stat_tests

# Cohen's d with robust estimation
result = stat_tests.test_group_mean_cohen_d(
    group1_values, 
    group2_values,
    use_robust_cohen_d=True
)

# Permutation test with custom resamples
result = stat_tests.run_permutation_test_on_diff_of_vector_means(
    group1_values,
    group2_values,
    n_resamples=10000
)
```

### Multiple Comparisons

```python
# Define multiple comparisons
comparisons = [
    ('Group_A', 'Group_B'),
    ('Group_A', 'Group_C'),
    ('Group_B', 'Group_C')
]

# Run all tests
posthoc_df = ppt.annotate(ax, plot_params, ax, comparisons, test_name='MWU')

# Filter and plot only significant results
sig_results = posthoc_df[posthoc_df['pvalue'] < 0.05]
ppt.plot_significance(ax, sig_results)
```

### Custom Plot Adjustments

```python
# Plot with custom bar styling
ppt.plot_significance(
    ax, 
    sig_results,
    bar_height=0.03,      # Taller bars
    text_offset=0.015,    # More space for text
)

# Adjust axis limits after plotting
ax.set_ylim(bottom=0, top=ax.get_ylim()[1] * 1.1)
```

---

## API Comparison: Old vs New

### Before v0.1.0 (Still Works!)

```python
from plot_posthoc_test import (
    main_run_posthoc_tests_and_get_hue_loc_df,
    plot_sig_bars_w_comp_df_tight,
    convert_pvalue_to_asterisks
)

posthoc_df = main_run_posthoc_tests_and_get_hue_loc_df(...)
plot_sig_bars_w_comp_df_tight(ax, posthoc_df)
stars = convert_pvalue_to_asterisks(0.001)
```

### After v0.1.0 (Recommended!)

```python
import plot_posthoc_test as ppt

posthoc_df = ppt.annotate(...)
ppt.plot_significance(ax, posthoc_df)
stars = ppt.stars(0.001)
```

**Both APIs work identically** - choose based on your preference!

---

## Complete Working Example

```python
import plot_posthoc_test as ppt
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Generate sample data
np.random.seed(42)
conditions = ['Early', 'Late']
groups = ['WT', 'KO']

data = []
for cond in conditions:
    for grp in groups:
        base = 0.5 if cond == 'Late' else 0
        shift = 0.3 if grp == 'KO' else 0
        values = np.random.randn(30) + base + shift
        for val in values:
            data.append({'condition': cond, 'group': grp, 'value': val})

df = pd.DataFrame(data)

# Create figure
fig, ax = plt.subplots(figsize=(10, 7))
sns.pointplot(
    data=df, 
    x='condition', 
    y='value', 
    hue='group',
    order=conditions,
    hue_order=groups,
    ax=ax, 
    errorbar='se',
    capsize=0.1
)

# Setup for analysis
plot_params = {
    'data': df,
    'x': 'condition',
    'y': 'value',
    'hue': 'group',
    'order': conditions,
    'hue_order': groups
}

comparisons = [('WT', 'KO')]

# Run statistical analysis
posthoc_df = ppt.annotate(
    ax_input=ax,
    plot_params=plot_params,
    plot_obj=ax,
    preset_comparisons=comparisons,
    test_name='MWU',
    detect_error_bar=True
)

# Plot significance bars
sig_results = posthoc_df[posthoc_df['pvalue'] < 0.05]
ppt.plot_significance(ax, sig_results)

# Format plot
ax.set_title('Post-hoc Analysis: WT vs KO', fontsize=14, fontweight='bold')
ax.set_ylabel('Measured Value', fontsize=12)
ax.set_xlabel('Condition', fontsize=12)
ax.legend(title='Group', frameon=True)

plt.tight_layout()
plt.savefig('posthoc_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

# Print detailed results
print("\n=== Statistical Results ===")
for _, row in posthoc_df.iterrows():
    sig = ppt.stars(row['pvalue'])
    print(f"{row['condition']}: {row['group_1']} vs {row['group_2']}")
    print(f"  p-value: {row['pvalue']:.4f} {sig}")
    if 'statistic' in row:
        print(f"  statistic: {row['statistic']:.4f}")
    print()
```

---

## Tips and Best Practices

### 1. Error Bar Detection
Always set `detect_error_bar=True` when your plot has error bars to ensure proper placement of significance bars.

### 2. Filter Before Plotting
Filter your results before plotting to avoid cluttering your visualization:
```python
ppt.plot_significance(ax, posthoc_df[posthoc_df['pvalue'] < 0.05])
```

### 3. Choose Appropriate Tests
- **Normal data, equal variance**: `'t-test'`
- **Normal data, unequal variance**: `'t-test'` (uses Welch's)
- **Non-normal data**: `'MWU'` or `'permutation'`
- **Effect size needed**: `'cohen_d'`

### 4. Multiple Comparison Correction
Consider correcting for multiple comparisons:
```python
from statsmodels.stats.multitest import multipletests

# Bonferroni correction
rejected, pvals_corrected, _, _ = multipletests(
    posthoc_df['pvalue'], 
    alpha=0.05, 
    method='bonferroni'
)
posthoc_df['pvalue_corrected'] = pvals_corrected
```

---

## Troubleshooting

### Issue: Bars don't appear
- Check that filtered DataFrame is not empty
- Verify p-values are in expected range
- Ensure axis limits accommodate bars

### Issue: Bars positioned incorrectly
- Set `detect_error_bar=True`
- Verify plot was created before running `annotate()`
- Check that plot_params match actual plot

### Issue: Import errors
- Ensure package is installed: `pip install -e .`
- Check Python version: requires ≥ 3.8
- Verify seaborn version: requires < 0.13

---

## See Also

- **[API Reference](api.md)** - Detailed function documentation
- **[Changelog](changelog.md)** - Version history
- **[GitHub Repository](https://github.com/cjohnsoncruz/plot_posthoc_test)** - Source code and issues
