"""Helper utility functions for plot_posthoc_test package."""


def get_match_index_in_iterable(iterable, val):
    """Returns index of val in iterable if present."""
    return [idx for idx, x in enumerate(iterable) if x == val]
