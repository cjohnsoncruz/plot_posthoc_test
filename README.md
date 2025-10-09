# plot_posthoc_test

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A Python package for performing and plotting post-hoc statistical tests on matplotlib/seaborn visualizations. Inspired by [statannotations](https://github.com/trevismd/statannotations), but with more direct control over plotting parameters and a streamlined, user-friendly API.

### Description

Post-hoc tests are essential for statistical analysis involving multiple groups ([learn more](https://en.wikipedia.org/wiki/Post_hoc_analysis)). This package provides functions for performing and annotating post-hoc tests on pandas DataFrames with matplotlib and seaborn plots.

**Note:** Currently optimized for seaborn pointplots. Requires seaborn < 0.13 to access hue groups in collections.

### Features

- ✅ **Statistical Tests**: Mann-Whitney U, Cohen's d, permutation tests, t-tests
- ✅ **Smart Annotations**: Automatically places significance bars between groups
- ✅ **Hue Support**: Handles complex hue groupings in seaborn plots  
- ✅ **Error Bar Detection**: Adjusts bar placement based on error bars
- ✅ **User-Friendly Aliases**: Simple API for common operations
- ✅ **Modular Design**: Organized utility functions for stats, plotting, and axis manipulation

## Installation

### From GitHub (Development Mode)

```bash
# Clone the repository
git clone https://github.com/cjohnsoncruz/plot_posthoc_test.git
cd plot_posthoc_test

# Install in editable mode
pip install -e .
```

### Requirements

```
python >= 3.8
numpy
pandas
scipy
matplotlib
seaborn < 0.13
```

## Quick Start

```python
import plot_posthoc_test as ppt
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Create your plot
fig, ax = plt.subplots()
sns.pointplot(data=df, x='condition', y='value', hue='group', ax=ax)

# Define which groups to compare
comparisons = [('control', 'treatment'), ('control', 'other')]

# Run statistical tests and annotate
plot_params = {'data': df, 'x': 'condition', 'y': 'value', 'hue': 'group'}
posthoc_df = ppt.annotate(ax, plot_params, ax, comparisons, test_name='MWU')

# Plot significance bars for p < 0.05
ppt.plot_significance(ax, posthoc_df[posthoc_df['pvalue'] < 0.05])

# Optional: Convert p-values to asterisks
stars = ppt.stars(0.001)  # Returns "***"

plt.show()
```

> **Note:** Original function names like `main_run_posthoc_tests_and_get_hue_loc_df()` and `plot_sig_bars_w_comp_df_tight()` are still available for backward compatibility.

## Documentation

- 📚 [Package Structure](PACKAGE_STRUCTURE.md) - Detailed package organization
- 📖 [Usage Examples](USAGE_EXAMPLES.md) - More code examples and use cases
- 📝 [Changelog](CHANGELOG.md) - Version history and updates
- 🔗 [API Reference](https://cjohnsoncruz.github.io/plot_posthoc_test/api/) - Complete API documentation

## Available Statistical Tests

- **Mann-Whitney U (MWU)**: Non-parametric test for independent samples
- **Cohen's d**: Effect size measure with confidence intervals
- **Permutation test**: Non-parametric resampling test
- **Two-sample t-test**: Parametric test (Welch's for unequal variance)
- **Bootstrap standard deviation overlap**: Custom distribution separation test

## Availability

Currently available via GitHub. PyPI upload scheduled for Q3 2025.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Communication

- 💬 [GitHub Discussions] - Ask questions and share ideas
- 🐛 [GitHub Issues] - Report bugs and request features

## License

MIT License - see LICENSE file for details

## Author

Carlos Johnson-Cruz (@cjohnsoncruz)

---

**Version**: 0.1.0 | **Updated**: October 2025

[GitHub Discussions]: https://github.com/cjohnsoncruz/plot_posthoc_test/discussions
[GitHub Issues]: https://github.com/cjohnsoncruz/plot_posthoc_test/issues
