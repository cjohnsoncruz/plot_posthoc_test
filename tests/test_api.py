"""Test user-facing API functions.

This module tests the main functions that users interact with:
- stars() - Convert p-values to asterisks
- annotate() - Run statistical tests
- plot_significance() - Add significance bars to plots
"""

import plot_posthoc_test as ppt


def test_stars_highly_significant():
    """Test that very low p-values return four stars."""
    result = ppt.stars(0.00001)
    assert result == "****"


def test_stars_not_significant():
    """Test that high p-values return 'ns'."""
    result = ppt.stars(0.1)
    assert result == "ns"
