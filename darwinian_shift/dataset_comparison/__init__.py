import warnings

try:
    from .location_comparison_plots import cdf_comparison_plot, chi2_comparison_plot, weighted_cdf
    from .root.homtests import homtest_sections, chi2_test_sections, chi2_test_window
except ImportError as e:
    warnings.warn("Install ROOT to use data comparison functions.")