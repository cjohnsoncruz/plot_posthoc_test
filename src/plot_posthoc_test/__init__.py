"""
Module for posthoc test plotting functions with matplotlib and seaborn.
"""

# Define the plot_posthoc_test version
__version__ = "0.1.0"

# Import main plotting functions
from .plot_stat_annotate import (
    main_run_posthoc_tests_and_get_hue_loc_df,
    run_posthoc_tests_on_all_ax_ticks,
    run_posthoc_test_on_tick_hue_groups,
    get_hue_loc_on_axis,
    plot_sig_bars_w_comp_df_tight,
    convert_pvalue_to_asterisks,
)

# Import ax inference functions
from .ax_inference import (
    get_x_ticks_as_df,
    get_hue_point_loc_df,
    get_hue_errorbar_loc_dict,
    get_hue_point_loc_dict,
)

# Expose utilities
from . import utils

# User-friendly aliases for common functions
# These provide a cleaner API while maintaining backward compatibility
annotate = main_run_posthoc_tests_and_get_hue_loc_df
plot_significance = plot_sig_bars_w_comp_df_tight
plot_bars = plot_sig_bars_w_comp_df_tight  # Alternative alias
run_tests = run_posthoc_tests_on_all_ax_ticks
stars = convert_pvalue_to_asterisks

__all__ = [
    '__version__',
    # Main functions (original names)
    'main_run_posthoc_tests_and_get_hue_loc_df',
    'run_posthoc_tests_on_all_ax_ticks',
    'run_posthoc_test_on_tick_hue_groups',
    'get_hue_loc_on_axis',
    'plot_sig_bars_w_comp_df_tight',
    'convert_pvalue_to_asterisks',
    # User-friendly aliases (NEW)
    'annotate',
    'plot_significance',
    'plot_bars',
    'run_tests',
    'stars',
    # Ax inference functions
    'get_x_ticks_as_df',
    'get_hue_point_loc_df',
    'get_hue_errorbar_loc_dict',
    'get_hue_point_loc_dict',
    # Submodules
    'utils',
]