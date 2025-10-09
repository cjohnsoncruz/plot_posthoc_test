# plot_posthoc_test Usage Examples

## Installation
```bash
pip install -e .
```

## Quick Start - New Simplified API

```python
import plot_posthoc_test as ppt
import matplotlib.pyplot as plt
import seaborn as sns

# Create your plot
fig, ax = plt.subplots()
sns.pointplot(data=df, x='condition', y='value', hue='group', ax=ax)

# Define comparisons
comparisons = [('Group_A', 'Group_B'), ('Group_A', 'Group_C')]

# Run posthoc tests and get results (NEW SIMPLE NAME)
posthoc_df = ppt.annotate(
    ax_input=ax,
    plot_params={'data': df, 'x': 'condition', 'y': 'value', 
                 'hue': 'group', 'order': conditions, 'hue_order': groups},
    plot_obj=ax,
    preset_comparisons=comparisons,
    test_name='MWU',
    detect_error_bar=True
)

# Plot significance bars (NEW SIMPLE NAME)
ppt.plot_significance(ax, posthoc_df)

# Convert p-values to asterisks (NEW SIMPLE NAME)
sig_level = ppt.stars(0.001)  # Returns "***"
```

## Comparison: Old vs New API

### Before (Still Works!)
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

### After (Cleaner!)
```python
import plot_posthoc_test as ppt

posthoc_df = ppt.annotate(...)
ppt.plot_significance(ax, posthoc_df)
stars = ppt.stars(0.001)
```

## Available Aliases

| New Alias | Original Function | Description |
|-----------|-------------------|-------------|
| `annotate()` | `main_run_posthoc_tests_and_get_hue_loc_df()` | Run posthoc tests and get locations |
| `plot_significance()` | `plot_sig_bars_w_comp_df_tight()` | Plot significance bars on axes |
| `plot_bars()` | `plot_sig_bars_w_comp_df_tight()` | Alternative alias for plotting |
| `run_tests()` | `run_posthoc_tests_on_all_ax_ticks()` | Run tests on all categories |
| `stars()` | `convert_pvalue_to_asterisks()` | Convert p-value to star notation |

## Complete Example

```python
import plot_posthoc_test as ppt
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Sample data
df = pd.DataFrame({
    'condition': ['Early', 'Early', 'Late', 'Late'] * 20,
    'group': ['WT', 'KO'] * 40,
    'value': np.random.randn(80) + np.repeat([0, 0.5, 1, 1.5], 20)
})

# Create plot
fig, ax = plt.subplots(figsize=(8, 6))
sns.pointplot(data=df, x='condition', y='value', hue='group', 
              ax=ax, errorbar='se')

# Setup parameters
plot_params = {
    'data': df,
    'x': 'condition',
    'y': 'value',
    'hue': 'group',
    'order': ['Early', 'Late'],
    'hue_order': ['WT', 'KO']
}

comparisons = [('WT', 'KO')]

# Run analysis - SIMPLE API
posthoc_df = ppt.annotate(
    ax_input=ax,
    plot_params=plot_params,
    plot_obj=ax,
    preset_comparisons=comparisons,
    test_name='MWU',
    detect_error_bar=True
)

# Plot significance bars - SIMPLE API
ppt.plot_bars(ax, posthoc_df[posthoc_df.pvalue < 0.05])

# Add title
ax.set_title('My Analysis')
plt.tight_layout()
plt.show()

# Print results with stars
for _, row in posthoc_df.iterrows():
    print(f"{row.group_1} vs {row.group_2}: p={row.pvalue:.4f} {ppt.stars(row.pvalue)}")
```

## Advanced: Using Utility Functions

```python
# Access statistical test utilities
from plot_posthoc_test.utils import stat_tests

# Run Cohen's d test
result = stat_tests.test_group_mean_cohen_d(
    group1_values, 
    group2_values,
    use_robust_cohen_d=True
)

# Run permutation test
result = stat_tests.run_permutation_test_on_diff_of_vector_means(
    group1_values,
    group2_values,
    n_resamples=10000
)
```

## Backward Compatibility

**Important**: All original function names still work! The aliases are just convenient shortcuts.

```python
# Both work exactly the same:
ppt.annotate(...)  # NEW
ppt.main_run_posthoc_tests_and_get_hue_loc_df(...)  # ORIGINAL

# Choose based on your preference!
```
