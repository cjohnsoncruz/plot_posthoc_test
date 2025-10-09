# plot_posthoc_test

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A Python package for performing and plotting post-hoc statistical tests on matplotlib/seaborn visualizations.

## Overview

Post-hoc tests are essential for statistical analysis involving multiple groups. This package provides functions for performing and annotating post-hoc tests on pandas DataFrames with matplotlib and seaborn plots.

**Features:**
- ✅ Statistical Tests: Mann-Whitney U, Cohen's d, permutation tests, t-tests
- ✅ Smart Annotations: Automatically places significance bars between groups
- ✅ Hue Support: Handles complex hue groupings in seaborn plots  
- ✅ Error Bar Detection: Adjusts bar placement based on error bars
- ✅ **User-Friendly Aliases**: Simple, clean API for common operations
- ✅ Modular Design: Organized utility functions for stats, plotting, and axis manipulation

## Quick Start

```python
import plot_posthoc_test as ppt
import seaborn as sns
import matplotlib.pyplot as plt

# Create your plot
fig, ax = plt.subplots()
sns.pointplot(data=df, x='condition', y='value', hue='group', ax=ax)

# Define comparisons and run tests
comparisons = [('control', 'treatment')]
plot_params = {'data': df, 'x': 'condition', 'y': 'value', 'hue': 'group'}

# Use the clean, user-friendly API
posthoc_df = ppt.annotate(ax, plot_params, ax, comparisons, test_name='MWU')
ppt.plot_significance(ax, posthoc_df[posthoc_df['pvalue'] < 0.05])

plt.show()
```

## Installation

```bash
# From GitHub (Development Mode)
git clone https://github.com/cjohnsoncruz/plot_posthoc_test.git
cd plot_posthoc_test
pip install -e .
```

## User-Friendly API

Version 0.1.0 introduced clean, intuitive aliases for common operations:

| Alias | Original Function | Purpose |
|-------|-------------------|---------|
| `annotate()` | `main_run_posthoc_tests_and_get_hue_loc_df()` | Run tests and get locations |
| `plot_significance()` | `plot_sig_bars_w_comp_df_tight()` | Plot significance bars |
| `stars()` | `convert_pvalue_to_asterisks()` | Convert p-values to asterisks |
| `run_tests()` | `run_posthoc_tests_on_all_ax_ticks()` | Run tests without location inference |

**All original function names remain available for backward compatibility.**

## Documentation

- **[API Reference](api.md)** - Complete function documentation with examples
- **[Usage Examples](usage.md)** - Detailed examples and use cases
- **[Changelog](changelog.md)** - Version history and updates
- **[License](license.md)** - MIT License details

## Available Statistical Tests

- **Mann-Whitney U (MWU)**: Non-parametric test for independent samples
- **Cohen's d**: Effect size measure with confidence intervals
- **Permutation test**: Non-parametric resampling test
- **Two-sample t-test**: Parametric test (Welch's for unequal variance)
- **Bootstrap standard deviation overlap**: Custom distribution separation test

## Requirements

```
python >= 3.8
numpy
pandas
scipy
matplotlib
seaborn < 0.13
```

**Note:** Currently optimized for seaborn pointplots. Requires seaborn < 0.13 to access hue groups in collections.

## Contributing

Contributions are welcome! Visit our [GitHub repository](https://github.com/cjohnsoncruz/plot_posthoc_test) to:
- Report bugs and request features
- Ask questions in Discussions
- Submit Pull Requests

## Author

Carlos Johnson-Cruz ([@cjohnsoncruz](https://github.com/cjohnsoncruz))

---

**Version**: 0.1.0 | **Status**: Development | **PyPI**: Coming Q3 2025
