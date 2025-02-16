# plot_posthoc_test
Repository containing custom code inspired by the [statannotations package](https://github.com/trevismd/statannotations) functionality, which I used quite a bit. I wanted more access to plotting parameters, and so I created this module that provides similar results, but with greatly streamlined functionality and no reliance on statannot classes. Currently, this package only works for seaborn's pointplot, but will have more to come. 
Requires specific seaborn version to allow for returning hue groups in collections, this is no longer possible in SNS .13+.

### Description
Post-hoc tests are a key piece of statistical analysis involving multiple groups (for an overview of the concept refer to [Post hoc analysis](https://en.wikipedia.org/wiki/Post_hoc_analysis)) This package provides functions for performing and plotting posthoc tests on data in pandas dataframes, using matplotlib and seaborn. 

### Features
This module includes utilities for:
- Running posthoc tests on data located on x axis ticks.
- Annotating plots with significance bars/stars bewteen groups with p-values below 0.05.
- Determining hue groups and their locations on plots.
- Detecting and adjusting significance bars based on the presence of error bars in plots.

<details open>
<summary>Sample code with scikit-learn</summary>

```python
import ...
# plot 
posthoc_df = main_run_posthoc_tests_and_get_hue_loc_df(plot_ax, plot_params, ax_object, comparison_list, test_name = 'MWU',ax_var_is_hue=False, detect_error_bar = True)
plot_sig_bars_w_comp_df_tight(plot_ax, posthoc_df[posthoc_df['pvalue'] < 0.05], tight_offset = 0.05) 
``` 
</details>

## Installation

plot_posthoc_test is currently only available via github. Python Package Index upload is expected Q2 2025.

## Communication

- [GitHub Discussions] for questions.
- [GitHub Issues] for bug reports and feature requests.

[GitHub Discussions]: https://github.com/cjohnsoncruz/plot_posthoc_test/discussions
[GitHub issues]: https://github.com/cjohnsoncruz/plot_posthoc_test/issues