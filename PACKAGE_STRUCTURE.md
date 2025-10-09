# Package Structure Reorganization Summary

## Overview
The `plot_posthoc_test` package has been restructured to follow modern Python packaging best practices with proper module organization.

## New Structure

```
plot_posthoc_test/
├── src/
│   └── plot_posthoc_test/          # Main package (already existed)
│       ├── __init__.py              # Updated: Exposes main API
│       ├── plot_stat_annotate.py   # Updated: Fixed imports to use utils
│       ├── ax_inference.py         # Core module (unchanged)
│       └── utils/                   # NEW: Utility modules
│           ├── __init__.py          # Utils package init
│           ├── stat_tests.py        # Moved from Function .py Storage/
│           ├── ax_modifier_functions.py  # Moved from Function .py Storage/
│           └── helpers.py           # NEW: Extracted helper function
├── Function .py Storage/            # Development-only helpers (not in package)
├── pyproject.toml                   # Package metadata
├── setup.py                         # Build configuration
└── requirements.txt                 # Dependencies
```

## What Changed

### 1. Created `utils/` Subdirectory
- **Location**: `src/plot_posthoc_test/utils/`
- **Purpose**: Contains utility modules needed by the package
- **Contents**:
  - `stat_tests.py` - Statistical test functions (MWU, Cohen's d, permutation tests)
  - `ax_modifier_functions.py` - Matplotlib axes modification utilities
  - `helpers.py` - Small helper functions extracted from helper_functions.py

### 2. Fixed Module Imports
- `plot_stat_annotate.py` now imports from `.utils` instead of external modules
- Added proper pandas import
- Uses relative imports for package-internal modules

### 3. Updated Package API (`__init__.py`)
Exposes the following functions directly:
- `main_run_posthoc_tests_and_get_hue_loc_df`
- `run_posthoc_tests_on_all_ax_ticks`
- `run_posthoc_test_on_tick_hue_groups`
- `get_hue_loc_on_axis`
- `plot_sig_bars_w_comp_df_tight`
- `convert_pvalue_to_asterisks`
- `get_x_ticks_as_df`
- `get_hue_point_loc_df`
- `get_hue_errorbar_loc_dict`
- `get_hue_point_loc_dict`

## Usage

### Installing the Package
```bash
# Development install
pip install -e .
```

### Importing in Code

**NEW: Simple Aliases (Recommended)**
```python
import plot_posthoc_test as ppt

# Use simplified function names
posthoc_df = ppt.annotate(ax, plot_params, plot_obj, comparisons)
ppt.plot_significance(ax, posthoc_df)
stars = ppt.stars(0.001)  # "***"
```

**Original Names (Still Supported)**
```python
import plot_posthoc_test

# Use full function names
posthoc_df = plot_posthoc_test.main_run_posthoc_tests_and_get_hue_loc_df(
    ax, plot_params, plot_obj, comparisons
)

# Access utility functions
from plot_posthoc_test.utils import stat_tests
result = stat_tests.test_group_mean_cohen_d(group1, group2)
```

### Notebook Usage
Your notebooks can continue to use `Function .py Storage/` modules for additional helper functions not included in the package:
```python
import sys
import os

# Add Function .py Storage to path for notebook-specific helpers
sys.path.append(os.path.join(os.getcwd(), 'Function .py Storage'))

# Now you can import both package and development helpers
import plot_posthoc_test
from helper_functions import make_folder, save_csv_to_analysis_storage
```

## Why This Structure?

### The `src/` Layout is Best Practice
- **Prevents accidental imports** during development
- **Ensures tests run against installed package** not local files  
- **Used by major projects**: pytest, requests, black, mypy, etc.

### Benefits
1. **Clean separation**: Package code vs. development helpers
2. **Proper imports**: Will work when installed via pip
3. **Extensible**: Easy to add new utility modules
4. **Maintainable**: Clear organization of functionality

## What Stayed in `Function .py Storage/`
These remain as development-only helpers (not part of installed package):
- `custom_module_imports.py`
- `dataframe_annotation_functions.py`
- `dimension_reduce.py`
- `helper_functions.py` (most functions)
- `preprocess_data.py`
- `sns_plotting_config.py`
- `paper_plot.mplstyle`

These are used by your notebooks but aren't core to the package's statistical annotation functionality.

## Next Steps

1. **Test the package**: Run `pip install -e .` to install in development mode
2. **Update notebooks**: Import from `plot_posthoc_test` instead of direct file imports
3. **Verify functionality**: Ensure your existing notebooks still work with the new structure
4. **Consider adding**: More functions to utils if needed for package functionality
