import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from statsmodels.stats.multitest import multipletests
from .root.homtests import chi2_test_window


## CDF plot
def weighted_cdf(section, spectrum=None, position_col='residue', value_col=None, unweighted=False):
    # Weight by the inverse of the signature.
    null = section.null_mutations
    obs = section.observed_mutations
    if spectrum is None:
        spectrum = section.project.spectra[0]  # Use the first spectrum set up for the section

    if value_col is None:
        value_col = position_col

    agg_null = null.groupby([position_col, value_col])[spectrum.rate_column].agg(sum)
    agg_obs = obs.groupby([position_col, value_col]).agg('count')[['chr']]
    agg_obs.columns = ['count']

    muts = pd.merge(agg_null, agg_obs, how='left', left_index=True, right_index=True)
    muts['count'] = muts['count'].fillna(0)
    if unweighted:
        muts['weight'] = 1
    else:
        muts['weight'] = 1 / muts[spectrum.rate_column]

    muts['pdf_value'] = muts['weight'] * muts['count']

    muts = muts.sort_index(level=1)
    muts = muts.groupby(level=1).agg(sum)
    cdf = muts['pdf_value'].cumsum()
    cdf = cdf / cdf.iloc[-1]

    return cdf


def cdf_comparison_plot(section1, section2, spectrum1=None, spectrum2=None, position_col='residue', value_col=None,
                        figsize=(5, 5), ax=None, label1='Section1', label2='Section2', show_unweighted=False,
                        show_weighted=True):
    """
    For comparing the position of mutations in the same gene/transcript/section from two datasets.
    Plots the uncorrected CDF of the positions (residue numbers by default) and the CDF adjusted by the mutational
    spectrum.
    :param section1:
    :param section2:
    :param spectrum1:
    :param spectrum2:
    :param position_col:
    :param figsize:
    :param ax:
    :param label1:
    :param label2:
    :return:
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    if show_unweighted:
        ax.plot(weighted_cdf(section1, spectrum1, position_col, value_col, unweighted=True),
                'C0--', label=label1 + ' Unweighted')
        ax.plot(weighted_cdf(section2, spectrum2, position_col, value_col, unweighted=True),
                'C1--', label=label1 + ' Unweighted')

    if show_weighted:
        ax.plot(weighted_cdf(section1, spectrum1, position_col, value_col, unweighted=False), label=label1)
        ax.plot(weighted_cdf(section2, spectrum2, position_col, value_col, unweighted=False), label=label2)

    if value_col is None:
        value_col = position_col
    ax.set_xlabel(value_col.capitalize())
    ax.set_ylabel('CDF')
    ax.legend()



## Chi2 sliding window plot
def chi2_comparison_plot(section1, section2, window_size=20, window_step=1, spectrum1=None, spectrum2=None,
                         show_legend=True, figsize=(15, 10), legend_args=None, show_plot=False,
                         colour1='C0', colour2='C1', show_residues=False, xlim=None, ylim=None, axes=None,
                         label1='Observed1', label2='Observed2', allow_low_counts=False,
                         multiple_test_correction_method='fdr_bh', include_missing_pvalues=True,
                         return_pvalues=False):
    """
    Compares the mutation counts in each window between the same gene/transcript/section in two datasets.
    Plots the sliding window of mutation counts adjusted by the mutational spectra and the log10 pvalues of the chi2-test
    for each window.

    There is no multiple test correction - the p-values should only be used as an indication of which parts of the
    gene are most significantly different in the two datasets.
    :param section1:
    :param section2:
    :param window_size:
    :param window_step:
    :param spectrum1:
    :param spectrum2:
    :param show_legend:
    :param figsize:
    :param legend_args:
    :param show_plot:
    :param colour1:
    :param colour2:
    :param show_residues:
    :param xlim:
    :param ylim:
    :param axes:
    :param label1:
    :param label2:
    :param allow_low_counts: If False, will not include any points with less then 10 effective counts in each bin for
    the chi2 test.
    :return:
    """
    if axes is None:
        fig, axes = plt.subplots(2, 1, figsize=figsize)
    else:
        fig = plt.gcf()
    ax1, ax2 = axes

    if spectrum1 is None:
        spectrum1 = section1.project.spectra[0]  # Use the first spectrum set up for the section
    if spectrum2 is None:
        spectrum2 = section2.project.spectra[0]  # Use the first spectrum set up for the section

    obs_1 = section1.plot_sliding_window(window_size=window_size, window_step=window_step, spectra=spectrum1,
                                         show_legend=show_legend, figsize=figsize, legend_args=legend_args,
                                         show_plot=False,
                                         colours=[colour1], return_fig=False, show_residues=show_residues, xlim=xlim,
                                         ylim=ylim, ax=ax1, divide_by_expected_rate=True,
                                         observed_label=label1,
                                         return_values=True)[spectrum1.name]
    obs_2 = section2.plot_sliding_window(window_size=window_size, window_step=window_step, spectra=spectrum2,
                                         show_legend=show_legend, figsize=figsize, legend_args=legend_args,
                                         show_plot=False,
                                         colours=[colour2], return_fig=False, show_residues=show_residues, xlim=xlim,
                                         ylim=ylim, ax=ax1, divide_by_expected_rate=True,
                                         observed_label=label2,
                                         return_values=True)[spectrum2.name]

    max_obs = max(obs_1.max(), obs_2.max())
    if ylim is None:
        ylim_low, ylim_high = ax1.get_ylim()
        if ylim_high <= max_obs:
            ylim_high = max_obs*1.1
            ax1.set_ylim(top=ylim_high)

    start, end = section1.null_mutations['residue'].min(), section1.null_mutations['residue'].max()

    starts = np.arange(start, end + 2 - window_size, window_step)
    if len(starts) == 0:
        print("Region too small compared to window size/step.")
        return
    ends = starts + window_size - 1

    p_values = np.full_like(obs_1, np.nan)
    directions = np.full_like(obs_1, np.nan)
    for i, (s, e) in enumerate(zip(starts, ends)):
        try:
            p_values[i] = chi2_test_window(section1, section2, s, e, verbose=False, allow_low_counts=allow_low_counts)
            if obs_1[i] > obs_2[i]:
                directions[i] = -1
            else:
                directions[i] = 1

        except ValueError as e:
            pass


    finite_vals = np.isfinite(p_values)
    if any(finite_vals):
        if multiple_test_correction_method is not None:
            # Run some multiple test correction. Regardless of method, this is probably not sufficiently conservative.
            if include_missing_pvalues:
                # Assume all cases with p-value = np.nan (low numbers) can be given a p-value of 1.
                p_values[~finite_vals] = 1
                p_values = multipletests(p_values, method=multiple_test_correction_method)[1]
            else:
                # Missing cases are ignored, reducing the number of tests that need correcting for.
                q_values = np.full_like(p_values, 1)
                q_values[finite_vals] = multipletests(p_values[finite_vals], method=multiple_test_correction_method)[1]
                p_values = q_values
        else:
            if include_missing_pvalues:
                p_values[~finite_vals] = 1

        plot_values = np.log10(p_values) * directions
        finite_plot_values = plot_values[np.isfinite(plot_values)]
        min_value, max_value = np.nanmin(finite_plot_values), np.nanmax(finite_plot_values)
        ylim = [None, None]
        if min_value < 0:
            ylim[0] = min_value * 1.2
        else:
            ylim[0] = 0
        if max_value > 0:
            ylim[1] = max_value * 1.2
        else:
            ylim[1] = 0

        if np.any(np.isinf(plot_values)):
            ax2.hlines([min_value * 1.1, max_value * 1.1], start, end, color='r', linestyle='--', linewidth=1)

        plot_values[plot_values == np.inf] = max_value * 1.1
        plot_values[plot_values == -np.inf] = min_value * 1.1
        section1._plot_sliding_window_results(starts, window_size, [], ['k'],
                                              [], plot_values, fig, ax2, xlim, start, end, ylim=ylim,
                                              show_legend=False, legend_args=None, show_residues=show_residues,
                                              show_plot=show_plot, ylabel='-log10(p-value)',
                                              observed_label='', divide_by_window_size=False)

        ax2.hlines([np.log10(0.05), 0, -np.log10(0.05)], start, end, color='k', linestyle='--', linewidth=1)
        yticklabels = ax2.get_yticks()
        ax2.set_yticklabels([abs(y) for y in yticklabels])

    if return_pvalues:
        return p_values