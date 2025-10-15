# Changelog

All notable changes to this project will be documented in this file.


## [Unreleased]

### Added
- **Acknowledgements section** in README and documentation homepage

### Changed
- **Updated API reference documentation** to centralize and highlight user-friendly aliased functions
  - Reorganized `doc/docs/api.md` with "Recommended API" section featuring aliases first
  - Improved documentation structure with code examples for each alias
  - Moved original function names to "Backward Compatibility" section
- **Enhanced documentation site** with comprehensive usage guide and updated homepage
  - Created `doc/docs/usage.md` with detailed examples and best practices
  - Updated `doc/docs/index.md` with quick start and API overview
  - Consolidated changelog: `mkdocs.yml` now references root `CHANGELOG.md` directly (removed duplicate)

## [0.1.0] - 2025-10-08

### Added
- **User-friendly aliases** for main functions:
  - `annotate()` as alias for `main_run_posthoc_tests_and_get_hue_loc_df()`
  - `plot_significance()` / `plot_bars()` as aliases for `plot_sig_bars_w_comp_df_tight()`
  - `run_tests()` as alias for `run_posthoc_tests_on_all_ax_ticks()`
  - `stars()` as alias for `convert_pvalue_to_asterisks()`
- **Utils subpackage** (`plot_posthoc_test.utils`) with organized utility modules:
  - `utils.stat_tests` - Statistical test functions (MWU, Cohen's d, permutation tests)
  - `utils.ax_modifier_functions` - Matplotlib axes modification utilities
  - `utils.helpers` - Helper functions (e.g., `get_match_index_in_iterable`)
- **Package structure documentation** (`PACKAGE_STRUCTURE.md`)
- **Usage examples** (`USAGE_EXAMPLES.md`)
- **Development mode installation** support via `pip install -e .`
- Self-contained `annotator_default` configuration within package

### Changed
- **Reorganized package structure** following Python best practices
- Moved helper modules from external `Function .py Storage/` into package
- Updated imports to use relative imports within package
- Improved `__init__.py` to expose clean public API

### Fixed
- Missing pandas import in `plot_stat_annotate.py`
- Import dependencies now properly contained within package
- Removed dependency on external configuration files


## [0.0.1] - 2024-12-08

### Added
- Initial package structure
- Core plotting functions for posthoc statistical annotations
- Support for seaborn and matplotlib plots
- Statistical test implementations (Mann-Whitney U, Cohen's d)
- Axes inference utilities

---

## Version Guidelines

- **MAJOR** (X.0.0): Breaking changes - users must update their code
- **MINOR** (0.X.0): New features - backward compatible
- **PATCH** (0.0.X): Bug fixes - backward compatible

## Links
- [Unreleased]: https://github.com/cjohnsoncruz/plot_posthoc_test/compare/v0.1.0...HEAD
- [0.1.0]: https://github.com/cjohnsoncruz/plot_posthoc_test/releases/tag/v0.1.0
- [0.0.1]: https://github.com/cjohnsoncruz/plot_posthoc_test/releases/tag/v0.0.1
