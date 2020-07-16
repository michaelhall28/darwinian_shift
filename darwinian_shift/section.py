import numpy as np
import pandas as pd
from collections import Counter
from adjustText import adjust_text
from scipy.stats import fisher_exact
from statsmodels.stats.multitest import multipletests
import matplotlib.pylab as plt
from matplotlib.patches import Arc
from matplotlib.cm import autumn, winter
from matplotlib.colors import Normalize, Colormap
from matplotlib.ticker import MaxNLocator, StrMethodFormatter
import seaborn as sns
import MDAnalysis
import os
from darwinian_shift.statistics import get_median, calculate_repeat_proportion, get_cdf, ChiSquareTest
from darwinian_shift.statistics import bootstrap_cdf_confint_method, get_null_cdf_confint
from darwinian_shift.utils import sort_multiple_arrays_using_one, get_pdb_positions
from darwinian_shift.mutation_spectrum import MutationalSpectrum
from darwinian_shift.additional_plotting_functions import annotate_mutations_on_plot, hide_top_and_right_axes, \
    colour_mutations_by_scores, _plot_single_scatter_category


class NoMutationsError(ValueError): pass


def get_distribution_from_mutational_spectrum(values, mut_rates, num=1000000):
    mut_rates = np.array(mut_rates)
    weights = mut_rates / mut_rates.sum()
    dist = np.repeat(values, np.random.multinomial(num, weights))
    return dist


class Section:
    # Class for the analysis of a single transcript/part of transcript

    def __init__(self, transcript, start=None, end=None, section_id=None, pdb_id=None, pdb_chain=None,
                 excluded_mutation_types=None, included_mutation_types=None, included_residues=None,
                 excluded_residues=None,
                 **kwargs):
        self.project = transcript.project  # The darwinian shift class that can run over multiple sections
        self.transcript = transcript
        self.gene = transcript.gene
        self.transcript_id = transcript.transcript_id
        self.chrom = transcript.chrom

        # included_mutation_types will be used if defined and excluded_mutation_types ignored.
        # If not defined here, will use the definitions from the project
        self.excluded_mutation_types = None
        self.included_mutation_types = None
        if included_mutation_types is not None:
            self.included_mutation_types = included_mutation_types
        elif self.project.included_mutation_types is not None:
            self.included_mutation_types = self.project.included_mutation_types
        elif self.excluded_mutation_types is not None:
            self.excluded_mutation_types = excluded_mutation_types
        elif self.project.excluded_mutation_types is not None:
            self.excluded_mutation_types = self.project.excluded_mutation_types

        if isinstance(self.included_mutation_types, str):
            self.included_mutation_types = [self.included_mutation_types]
        elif isinstance(self.excluded_mutation_types, str):
            self.excluded_mutation_types = [self.excluded_mutation_types]

        self.start = start  # Start residue of the section. If None, will use whole transcript.
        self.end = end  # End residue of the section
        self.included_residues = included_residues  # Alternative to start/end, specify which residues to include
        self.excluded_residues = excluded_residues  # Specify any residues to exclude
        self.pdb_id = pdb_id
        self.pdb_chain = pdb_chain
        if section_id is not None:
            self.section_id = section_id
        elif pdb_id is not None:
            self.section_id = "{}:{}".format(self.pdb_id, self.pdb_chain)
        elif self.start is None:
            self.section_id = self.transcript_id
        else:
            self.section_id = "{}:{}-{}".format(self.transcript_id, self.start, self.end)

        for k, v in kwargs.items():
            setattr(self, k, v)  # Any extra attributes for the lookup class to use or for results table.

        self._scoring_complete = False

        ### Results. Populated during self.run
        self.null_mutations = None
        self.observed_mutations = None
        self.num_mutations = None
        self.null_scores = None
        self.observed_values = None
        self.ref_mismatch_count = None

        self.statistic_results = None

        self.repeat_proportion = None

    def load_section_mutations(self):
        # Make a copy, as overlapping sections might have different scores (e.g. different pdb files)
        self.null_mutations = self.transcript.get_possible_mutations(return_copy=True)
        if self.start is not None:
            self.null_mutations = self.null_mutations[
                (self.null_mutations['residue'] >= self.start) &
                (self.null_mutations['residue'] <= self.end)]

        if self.included_residues is not None:
            self.null_mutations = self.null_mutations[self.null_mutations['residue'].isin(self.included_residues)]

        if self.excluded_residues is not None:
            self.null_mutations = self.null_mutations[~self.null_mutations['residue'].isin(self.excluded_residues)]

        if self.included_mutation_types is not None:
            self.null_mutations = self.null_mutations[
                self.null_mutations['effect'].isin(self.included_mutation_types)]
        elif self.excluded_mutation_types is not None:
            self.null_mutations = self.null_mutations[~
                self.null_mutations['effect'].isin(self.excluded_mutation_types)]

        if self.project.excluded_positions is not None:
            chrom_excluded_pos = self.project.excluded_positions[self.chrom]
            self.null_mutations = self.null_mutations[
                ~self.null_mutations['pos'].isin(chrom_excluded_pos)]

        # Add the mutation rate for each possible mutation under the spectra
        for spectrum in self.project.spectra:
            self.null_mutations = spectrum.apply_spectrum(self, self.null_mutations)

        # Will be further filtered based on the defined null mutations in the get_values function
        self.null_mutations['null_exists'] = 1
        self.observed_mutations = self.transcript.get_observed_mutations(return_copy=True)

        # Check for mismatching reference bases
        self._check_mismatches()

        self.observed_mutations = pd.merge(self.observed_mutations, self.null_mutations, how='left',
                                           on=['pos', 'ref', 'mut'], suffixes=['_input', ''])
        self.observed_mutations = self.observed_mutations[~pd.isnull(self.observed_mutations['null_exists'])]
        self.observed_mutations = self.observed_mutations.drop('null_exists', axis=1)
        self.null_mutations = self.null_mutations.drop('null_exists', axis=1)

        self.num_mutations = len(self.observed_mutations)  # Initial number of observed mutations (may be filtered later)

        self._add_mutation_id_column()

    def run(self, plot_permutations=False, spectra=None, statistics=None):
        self.apply_scores()

        # Compare distributions
        self.statistical_tests(plot_permutations, spectra=spectra, statistics=statistics)

    def get_results_dictionary(self):
        res = {k: getattr(self, k) for k in self.project.result_columns}
        res.update(self.statistic_results)
        return res

    def get_pvalues(self):
        if self.statistic_results is None:
            self.statistical_tests()
        p = {k: v for (k, v) in self.statistic_results.items() if 'pvalue' in k}
        return p

    def apply_scores(self):
        if not self._scoring_complete:
            self.load_section_mutations()

            # Get scores, and make sure all results are numpy arrays of floats for consistency
            self.null_mutations['score'] = np.array(self.project.lookup(self)).astype(float)

            # Exclude all cases without a score
            self.null_mutations = self.null_mutations[~pd.isnull(self.null_mutations['score'])]
            if len(self.null_mutations) == 0:
                raise NoMutationsError('No scores for {} {}'.format(self.section_id, self.gene))
            self.null_scores = self.null_mutations['score'].values

            # Match the observed mutations with the null mutations.
            self.observed_mutations = pd.merge(self.observed_mutations,
                                               self.null_mutations[['pos', 'ref', 'mut', 'score']], on=['pos', 'ref', 'mut'],
                                               how='left', suffixes=["_x", ""])

            # Exclude all cases without a score. These haven't matched against a null mutation.
            self.observed_mutations = self.observed_mutations[~pd.isnull(self.observed_mutations['score'])]

            # Recalculate the number of observed mutations after filtering out those which did not get a score
            self.num_mutations = len(self.observed_mutations)
            if self.num_mutations == 0:
                raise NoMutationsError('No mutations retained for {} {}'.format(self.section_id, self.gene))


            self.observed_values = self.observed_mutations['score']

            self._scoring_complete=True

    def _get_mutation_id(self, row):
        return '{}:{}>{}'.format(row['pos'], row['ref'], row['mut'])

    def _add_mutation_id_column(self):
        if len(self.observed_mutations) > 0:
            self.observed_mutations['ds_mut_id'] = self.observed_mutations.apply(self._get_mutation_id, axis=1)
        if len(self.null_mutations) > 0:
            self.null_mutations['ds_mut_id'] = self.null_mutations.apply(self._get_mutation_id, axis=1)

    def _check_mismatches(self):
        # All mutations are single base substitutions. So the base should be identical if the position is the same.
        null_refs = self.null_mutations[['pos', 'ref']].drop_duplicates()
        merged_df = pd.merge(self.observed_mutations, null_refs, on='pos')
        self.ref_mismatch_count = (merged_df['ref_x'] != merged_df['ref_y']).sum()
        if self.ref_mismatch_count > 0:
            print('Warning: {}/{} mutations do not match reference base in {}'.format(self.ref_mismatch_count,
                                                                                      len(self.observed_mutations),
                                                                                   self.section_id))

    def _get_spectra(self, spectra=None):
        if spectra is None:
            spectra = self.project.spectra
        elif isinstance(spectra, MutationalSpectrum):
            spectra = [spectra]
        return spectra

    def statistical_tests(self, plot=False, spectra=None, statistics=None):
        if self.statistic_results is None:
            self.statistic_results = {}
        spectra = self._get_spectra(spectra)
        if statistics is None:
            statistics = self.project.statistics
        elif not isinstance(statistics, (list, tuple, set)):
            statistics = [statistics]

        self.statistic_results['observed_median'] = self.observed_values.median()
        self.statistic_results['observed_mean'] = self.observed_values.mean()
        for spectrum in spectra:
            self.statistic_results['expected_median_' + spectrum.name] = get_median(self.null_scores,
                                                                                self.null_mutations[spectrum.rate_column])
            self.statistic_results['median_shift_' + spectrum.name] = self.statistic_results['observed_median'] - \
                                                                  self.statistic_results['expected_median_' + spectrum.name]

            self.statistic_results['expected_mean_' + spectrum.name] = np.mean(self.null_scores*self.null_mutations[spectrum.rate_column])
            self.statistic_results['mean_shift_' + spectrum.name] = self.statistic_results['observed_mean'] - \
                                                                self.statistic_results['expected_mean_' + spectrum.name]
            for statistic in statistics:
                self.statistic_results.update(statistic(self, spectrum, plot))

        self.repeat_proportion = calculate_repeat_proportion(self.null_scores)

    def plot(self, spectra=None, violinplot_bw=None, plot_scale=None,
             marker_size_from_count=True, base_marker_size=10, colours=None):
        spectra = self._get_spectra(spectra)
        if self.observed_values is None:
            self.apply_scores()  # Make sure all the values needed for plotting are generated

        self.plot_sliding_window(spectra=spectra, colours=colours, show_plot=True)
        self.plot_sliding_3D_window(spectra=spectra, colours=colours, show_plot=True)
        self.plot_scatter(plot_scale, marker_size_from_count, base_marker_size, show_plot=True)
        self.plot_violinplot(spectra, plot_scale, violinplot_bw, colours=colours, show_plot=True)
        self.plot_boxplot(spectra, plot_scale, colours=colours, show_plot=True)
        self.plot_cdfs(spectra, plot_scale, colours=colours, show_plot=True)
        self.plot_binned_counts(spectra, colours=colours, show_plot=True)
        self.plot_aa_abundance(spectra, show_plot=True)

    def _get_plot_colours(self, colours, spectra):
        if colours is None:
            colours = plt.rcParams['axes.prop_cycle'].by_key()['color']  # Default colour cycle

            if len(spectra) + 1 > len(colours):
                print('Too many spectra to plot with unique default colours.')
                colours = [colours[i%len(colours)] for i in range(len(spectra)+1)]

        elif len(spectra) + 1 > len(colours):
            print('Too many spectra to plot with given colours.')
            colours = [colours[i % len(colours)] for i in range(len(spectra) + 1)]
        return colours

    def _get_seaborn_colour_palette(self, colours, num):
        """
        The first colour in the list is for the observed data.
        Currently the observed data is last in the box and violinplots, so swap the colour order
        :param colours:
        :return:
        """
        return sns.color_palette(list(colours[1:num]) + [colours[0]])

    def _format_xticks(self, ax):
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        formatter = StrMethodFormatter("{x:.0f}")
        ax.xaxis.set_major_formatter(formatter)


    def _show_residues(self, fig, ax):
        ax.tick_params(axis='x', direction='out', pad=12, length=0)
        fig.subplots_adjust(bottom=0.20)
        ax2 = plt.gca().twiny()
        ax2.patch.set_visible(False)
        hide_top_and_right_axes(ax2)
        ax2.xaxis.set_ticks_position('bottom')
        ax2.xaxis.set_label_position('bottom')
        residues = self.null_mutations[['residue', 'aaref']].drop_duplicates()
        ax2.set_xticks(residues['residue'].values)
        ax2.set_xticklabels(residues['aaref'].values)
        ax2.set_xlim(ax.get_xlim())
        ax2.tick_params(axis='x', length=0, pad=1)
        plt.sca(ax)

    def _plot_sliding_window_results(self, starts, window_size, spectra, colours,
                                     expected_counts, observed_counts, fig, ax, xlim, start, end, ylim,
                                     show_legend, legend_args, show_residues, show_plot, ylabel):
        plot_x = (starts + window_size / 2 - 0.5)
        for spectrum, colour in zip(spectra, colours[1:]):
            plt.plot(plot_x, expected_counts[spectrum.name] / window_size,
                     label=spectrum.name, c=colour)

        plt.plot(plot_x, observed_counts / window_size, label='Observed', c=colours[0],
                 linewidth=3)
        if xlim is not None:
            ax.set_xlim(xlim)
        else:
            ax.set_xlim([start, end])
        if ylim is not None:
            ax.set_ylim(ylim)
        else:
            ax.set_ylim(bottom=0)
        plt.ylabel(ylabel)
        plt.xlabel('Residue')

        self._format_xticks(ax)

        if show_legend:
            if legend_args is None:
                legend_args = {}
            plt.legend(**legend_args)

        if show_residues:
            self._show_residues(fig, ax)

        if show_plot:
            plt.show()

    def plot_sliding_window(self, window_size=20, window_step=1,
                            spectra=None, show_legend=True, figsize=(15, 5), legend_args=None, show_plot=False,
                            colours=None, return_fig=False, show_residues=False, xlim=None, ylim=None):
        fig, ax = plt.subplots(figsize=figsize)
        spectra = self._get_spectra(spectra)

        colours = self._get_plot_colours(colours, spectra)

        start, end = self.null_mutations['residue'].min(), self.null_mutations['residue'].max()

        starts = np.arange(start, end + 2 - window_size, window_step)
        ends = starts + window_size - 1

        # Get some normalising factors, so the total mutations in the gene is the same in the observed and expected plots
        normalising_factors = {
            spectrum.name: self.num_mutations / self.null_mutations[spectrum.rate_column].sum() for spectrum in spectra
        }

        expected_counts = {spectrum.name: np.empty(len(starts)) for spectrum in spectra}
        observed_counts = np.empty(len(starts))
        for i, (s, e) in enumerate(zip(starts, ends)):
            obs_window_mutations = self.observed_mutations[(self.observed_mutations['residue'] >= s) &
                                                           (self.observed_mutations['residue'] <= e)]
            observed_counts[i] = len(obs_window_mutations)

            null_window_mutations = self.null_mutations[(self.null_mutations['residue'] >= s) &
                                                        (self.null_mutations['residue'] <= e)]
            for spectrum in spectra:
                total_window_rate = null_window_mutations[spectrum.rate_column].sum()
                normalised_window_rate = total_window_rate * normalising_factors[spectrum.name]
                expected_counts[spectrum.name][i] = normalised_window_rate

        self._plot_sliding_window_results(starts, window_size, spectra, colours,
                                          expected_counts, observed_counts, fig, ax, xlim, start, end, ylim,
                                          show_legend, legend_args, show_residues, show_plot,
                                          ylabel='Mutations per codon')

        if return_fig:
            return fig

    def plot_scatter(self, plot_scale=None, marker_size_from_count=True,
                     base_marker_size=10, show_plot=False, unobserved_mutation_colour='#BBBBBB',
                     missense_mutation_colour='C1', synonymous_mutation_colour='C2', nonsense_mutation_colour='C3',
                     show_legend=True, figsize=(15, 5), legend_args=None, return_fig=False,
                     show_residues=False, xlim=None, unmutated_marker_size=1,
                     unobserved_alpha=1, observed_alpha=1,
                     sections_for_colours=None, score_regions_for_colours=None,
                     score_region_colours=None, colour_unmutated=False, hotspots_in_foreground=False,
                     observed_marker=None, unobserved_marker=None,
                     show_observed_only=False, show_null_only=False
                     ):
        fig, ax = plt.subplots(figsize=figsize)
        if not show_observed_only:
            null_to_plot = self.null_mutations[~self.null_mutations['ds_mut_id'].isin(self.observed_mutations['ds_mut_id'])]
            plt.scatter(null_to_plot['residue'], null_to_plot['score'], s=unmutated_marker_size,
                    alpha=unobserved_alpha, c=unobserved_mutation_colour, label=None, linewidth=0,
                    marker=unobserved_marker)

        if not show_null_only:
            mut_counts = self.observed_mutations['ds_mut_id'].value_counts()
            for effect, col in zip(['synonymous', 'missense', 'nonsense'], [synonymous_mutation_colour,
                                                                        missense_mutation_colour,
                                                                        nonsense_mutation_colour]):

                muts = self.observed_mutations[self.observed_mutations['effect'] == effect]
                if len(muts) > 0:
                    _plot_single_scatter_category(muts, mut_counts, 'residue', 'score', marker_size_from_count,
                                              base_marker_size, col, effect,
                                              observed_alpha, hotspots_in_foreground, marker=observed_marker)

        if sections_for_colours is not None:
            if colour_unmutated and not show_observed_only:
                colour_mutations_by_scores(null_to_plot, None, 'residue', 'score',
                                           sections_for_colours,
                                           score_regions_for_colours, score_region_colours,
                                           marker_size_from_count=False,
                                           base_marker_size=unmutated_marker_size, alpha=unobserved_alpha,
                                           hotspots_in_foreground=False, use_null=True, marker=unobserved_marker)

            if not show_null_only:
                colour_mutations_by_scores(self.observed_mutations, mut_counts, 'residue', 'score', sections_for_colours,
                                       score_regions_for_colours, score_region_colours, marker_size_from_count,
                                       base_marker_size, observed_alpha, hotspots_in_foreground, marker=observed_marker)

        plt.xlabel('Residue')
        try:
            plt.ylabel(self.project.lookup.name)
        except AttributeError as e:
            pass
        if xlim is not None:
            ax.set_xlim(xlim)

        self._format_xticks(ax)

        if plot_scale is not None:
            plt.yscale(plot_scale)
        if show_legend:
            if legend_args is None:
                legend_args = {}
            plt.legend(**legend_args)

        if show_residues:
            self._show_residues(fig, ax)

        if show_plot:
            plt.show()
        if return_fig:
            return fig

    def plot_violinplot(self, spectra=None, plot_scale=None, violinplot_bw=None, show_plot=False, colours=None,
                        figsize=(5, 5), return_fig=False):
        fig = plt.figure(figsize=figsize)
        spectra = self._get_spectra(spectra)
        colours = self._get_plot_colours(colours, spectra)
        if violinplot_bw is None:
            # Must set this otherwise will be different for the obs and exp distributions.
            violinplot_bw = (max(self.null_scores) - min(self.null_scores)) / 200

        data = [get_distribution_from_mutational_spectrum(self.null_scores, self.null_mutations[spectrum.rate_column]) for spectrum in spectra]
        data.append(self.observed_values)
        sns.violinplot(
            data=data, cut=0,
            bw=violinplot_bw, palette=self._get_seaborn_colour_palette(colours, len(spectra) + 1))
        plt.xticks(range(len(data)), ['Expected\n' + spectrum.name for spectrum in spectra]  + ['Observed'])
        try:
            plt.ylabel(self.project.lookup.name)
        except AttributeError as e:
            pass
        if plot_scale is not None:
            plt.yscale(plot_scale)
        if show_plot:
            plt.show()
        if return_fig:
            return fig

    def plot_boxplot(self, spectra=None, plot_scale=None, show_plot=False, colours=None,
                     figsize=(5, 5), return_fig=False):
        fig = plt.figure(figsize=figsize)
        spectra = self._get_spectra(spectra)

        colours = self._get_plot_colours(colours, spectra)
        data = [get_distribution_from_mutational_spectrum(self.null_scores,
                                                          self.null_mutations[spectrum.rate_column])
                for spectrum in spectra]
        data.append(self.observed_values)
        sns.boxplot(
            data=data, palette=self._get_seaborn_colour_palette(colours, len(spectra) + 1))
        plt.xticks(range(len(data)), ['Expected\n' + spectrum.name for spectrum in spectra] + ['Observed'])
        try:
            plt.ylabel(self.project.lookup.name)
        except AttributeError as e:
            pass
        if plot_scale is not None:
            plt.yscale(plot_scale)
        if show_plot:
            plt.show()
        if return_fig:
            return fig

    def plot_cdfs(self, spectra=None, plot_scale=None, show_plot=False, legend_args=None, colours=None,
                  figsize=(5, 5), return_fig=False, show_legend=True,
                  show_CI=False, CI_alpha=0.05, CI_num_samples=10000):
        fig = plt.figure(figsize=figsize)
        spectra = self._get_spectra(spectra)

        colours = self._get_plot_colours(colours, spectra)
        num_obs = len(self.observed_values)
        sorted_null_scores = sorted(self.null_scores)
        xvals = np.linspace(min(sorted_null_scores), max(sorted_null_scores), 10000)
        for spectrum, colour in zip(spectra, colours[1:]):
            cdf_func = get_cdf(self.null_scores, self.null_mutations[spectrum.rate_column])
            plt.plot(xvals, cdf_func(xvals), label='Expected ' + spectrum.name, c=colour)
            if show_CI:
                ci_low, ci_high = get_null_cdf_confint(self.null_scores, self.null_mutations[spectrum.rate_column],
                                                       num_obs, xvals, num_samples=CI_num_samples, alpha=CI_alpha)
                plt.fill_between(xvals, ci_low, ci_high, color=colour, alpha=0.1)

        sorted_obs = sorted(self.observed_values)
        plt.plot(sorted_obs, np.arange(1, len(sorted_obs) + 1) / len(sorted_obs), label='Observed', linewidth=3,
                 c=colours[0])
        if show_CI:
            ci_low, ci_high = bootstrap_cdf_confint_method(sorted_obs, xvals, num_samples=CI_num_samples,
                                                           alpha=CI_alpha)
            plt.fill_between(xvals, ci_low, ci_high, color=colours[0], alpha=0.1)

        try:
            plt.xlabel(self.project.lookup.name)
        except AttributeError as e:
            pass
        plt.ylabel('CDF')
        if show_legend:
            if legend_args is None:
                legend_args = {}
            plt.legend(**legend_args)
        if plot_scale is not None:
            plt.xscale(plot_scale)
        if show_plot:
            plt.show()
        if return_fig:
            return fig

    def plot_binned_counts(self, spectra=None, show_plot=False, figsize=(15, 5), colours=None, return_fig=False,
                           show_legend=True, legend_args=None, chi_tests=None, show_CI=True):
        spectra = self._get_spectra(spectra)

        colours = self._get_plot_colours(colours, spectra)
        if chi_tests is None:
            chi_tests = [t.name for t in self.project.statistics if isinstance(t, ChiSquareTest)]
        else:
            chi_tests = [t.name for t in self.project.statistics if isinstance(t, ChiSquareTest) and
                         (t in chi_tests or t.name in chi_tests)]
        if len(chi_tests) == 0:
            print('No chi square tests to plot')
        else:
            if self.statistic_results is None:
                self.statistical_tests(plot=False)  # The chi-square tests need to be run first

            fig, axes = plt.subplots(nrows=len(chi_tests), ncols=len(spectra), squeeze=False, figsize=figsize)
            legend_items = []
            for i, chi_name in enumerate(chi_tests):
                for j, spectrum in enumerate(spectra):
                    expected_counts = self.statistic_results["_".join([chi_name, spectrum.name,  'expected_counts'])]
                    observed_counts = self.statistic_results["_".join([chi_name, spectrum.name, 'observed_counts'])]
                    bins = self.statistic_results["_".join([chi_name, spectrum.name, 'bins'])]
                    x = np.arange(len(expected_counts)) + 0.1
                    w = 0.4
                    ax = axes[i, j]

                    if show_CI:
                        observed_CI_low = np.array(
                            self.statistic_results["_".join([chi_name, spectrum.name, 'observed_CI_low'])])
                        observed_CI_high = np.array(
                            self.statistic_results["_".join([chi_name, spectrum.name, 'observed_CI_high'])])
                        yerr_obs = [observed_counts - observed_CI_low, observed_CI_high - observed_counts]

                        expected_CI_low = np.array(
                            self.statistic_results["_".join([chi_name, spectrum.name, 'expected_CI_low'])])
                        expected_CI_high = np.array(
                            self.statistic_results["_".join([chi_name, spectrum.name, 'expected_CI_high'])])
                        yerr_exp = [expected_counts - expected_CI_low, expected_CI_high - expected_counts]
                    else:
                        yerr_obs=None
                        yerr_exp = None


                    l1 = ax.bar(x, expected_counts, yerr=yerr_exp, label='Expected counts', width=w, color=colours[j+1])
                    legend_items.append(l1)
                    obs = ax.bar(x + w,
                            observed_counts, yerr=yerr_obs, label='Observed counts', width=w, color=colours[0])
                    xticklabels = []
                    for k in range(len(expected_counts)):
                        if k != len(expected_counts)-1:
                            xticklabels.append("[ {:.2f},  {:.2f} )".format(bins[k], bins[k + 1]))
                        else:
                            xticklabels.append("[ {:.2f},  {:.2f} ]".format(bins[k], bins[k + 1]))
                    ax.set_xticks(x + w / 2)
                    ax.set_xticklabels(xticklabels)
                    ax.tick_params(axis='x', rotation=90)
                    if j == 0:
                        ax.set_ylabel('Number of mutations')
                    ax.set_title('Chi square counts -\n' + spectrum.name)

            if show_legend:
                if legend_args is None:
                    legend_args = dict(bbox_to_anchor=(0.5,-0.13), loc="lower center",
                       bbox_transform=fig.transFigure, ncol=len(spectra) + 1)
                fig.legend(legend_items + [obs],
                       ['Expected\ncounts\n{}'.format(s.name) for s in spectra]+ ['Observed\ncounts'],
                       **legend_args
                       )

            if show_plot:
                plt.show()
            if return_fig:
                return fig

    def plot_binned_counts_common_bins(self, spectra=None, show_plot=False, figsize=(15, 5),
                                       colours=None, return_fig=False, linewidth=0,
                                       hatches=None,
                                       show_legend=True, legend_args=None, chi_tests=None,
                                       show_CI=True):
        spectra = self._get_spectra(spectra)

        colours = self._get_plot_colours(colours, spectra)
        if hatches is None:
            hatches = [None]*len(colours)
        if chi_tests is None:
            chi_tests = [t.name for t in self.project.statistics if isinstance(t, ChiSquareTest)]
        else:
            chi_tests = [t.name for t in self.project.statistics if isinstance(t, ChiSquareTest) and
                         (t in chi_tests or t.name in chi_tests)]
        if len(chi_tests) == 0:
            print('No chi square tests to plot')
        else:
            if self.statistic_results is None:
                self.statistical_tests(plot=False)  # The chi-square tests need to be run first

            fig, axes = plt.subplots(nrows=len(chi_tests), ncols=1, squeeze=False, figsize=figsize)
            legend_items = []
            for i, chi_name in enumerate(chi_tests):
                all_bins = [self.statistic_results["_".join([chi_name, spectrum.name, 'bins'])] for spectrum in spectra]
                bins = all_bins[0]
                if not all([b == bins for b in all_bins[1:]]):
                    print('Cannot plot for {} because the bins vary with spectrum'.format(chi_name))
                else:
                    ax = axes[i, 0]
                    w = 0.8 / (len(spectra) + 1)

                    observed_counts = np.array(
                        self.statistic_results["_".join([chi_name, spectra[0].name, 'observed_counts'])])
                    if show_CI:
                        observed_CI_low = np.array(
                            self.statistic_results["_".join([chi_name, spectra[0].name, 'observed_CI_low'])])
                        observed_CI_high = np.array(
                            self.statistic_results["_".join([chi_name, spectra[0].name, 'observed_CI_high'])])
                        yerr = [observed_counts - observed_CI_low, observed_CI_high - observed_counts]
                    else:
                        yerr=None

                    x = np.arange(0, len(observed_counts)) + w / 2

                    obs = ax.bar(x + w * len(spectra),
                                 observed_counts,
                                 yerr=yerr,
                                 label='Observed counts', width=w, linewidth=linewidth, edgecolor='k',
                                 color=colours[0], hatch=hatches[0])

                    for j, spectrum in enumerate(spectra):
                        expected_counts = np.array(
                            self.statistic_results["_".join([chi_name, spectrum.name, 'expected_counts'])])


                        if show_CI:
                            expected_CI_low = np.array(
                                self.statistic_results["_".join([chi_name, spectrum.name, 'expected_CI_low'])])
                            expected_CI_high = np.array(
                                self.statistic_results["_".join([chi_name, spectrum.name, 'expected_CI_high'])])
                            yerr = [expected_counts - expected_CI_low, expected_CI_high - expected_counts]
                        else:
                            yerr=None

                        l1 = ax.bar(x + w * j, expected_counts,
                                    yerr=yerr,
                                    label='Expected counts {}'.format(spectrum.name),
                                    width=w, color=colours[j + 1], linewidth=linewidth, edgecolor='k',
                                    hatch=hatches[j + 1])
                        legend_items.append(l1)

                    xticklabels = []
                    for k in range(len(expected_counts)):
                        if k != len(expected_counts):
                            xticklabels.append("[ {:.2f},  {:.2f} )".format(bins[k], bins[k + 1]))
                        else:
                            xticklabels.append("[ {:.2f},  {:.2f} ]".format(bins[k], bins[k + 1]))
                    ax.set_xticks(np.arange(0, len(observed_counts)) + 0.4)
                    ax.set_xticklabels(xticklabels)
                    ax.tick_params(axis='x', rotation=90)
                    ax.set_ylabel('Number of mutations')
                    ax.set_title('Chi square counts -\n' + spectrum.name)

            if show_legend:
                if legend_args is None:
                    legend_args = dict(bbox_to_anchor=(0.5, -0.13), loc="lower center",
                                       bbox_transform=fig.transFigure, ncol=len(spectra) + 1)
                fig.legend(legend_items + [obs],
                           ['Expected\ncounts\n{}'.format(s.name) for s in spectra] + ['Observed\ncounts'],
                           **legend_args
                           )

            if show_plot:
                plt.show()
            if return_fig:
                return fig

    def plot_aa_abundance(self, spectra=None, sig_threshold=0.05, use_qval=True, show_plot=False, max_texts=10,
                          figsize=(15, 5), return_fig=False):
        spectra = self._get_spectra(spectra)

        fig, axes = plt.subplots(nrows=1, ncols=len(spectra), squeeze=False, figsize=figsize)

        for i, spectrum in enumerate(spectra):
            ax = axes[0, i]
            total_relative_rate = self.null_mutations[spectrum.rate_column].sum()

            self.null_mutations['norm_relative_rate'] = self.null_mutations[spectrum.rate_column] / total_relative_rate * self.num_mutations

            aa_expected = {}
            for (r, m), rmdf in self.null_mutations.groupby(['aaref', 'aamut']):
                aa_expected[(r, m)] = rmdf['norm_relative_rate'].sum()
            aa_obs = {}
            for (r, m), rmdf in self.observed_mutations.groupby(['aaref', 'aamut']):
                aa_obs[(r, m)] = len(rmdf)

            aachanges = aa_expected.keys()
            exp_counts = [aa_expected[a] for a in aachanges]
            obs_counts = [aa_obs.get(a, 0) for a in aachanges]
            max_count = max(exp_counts + obs_counts)
            ax.scatter(exp_counts, obs_counts, s=5)
            ax.plot([0, max_count * 1.1], [0, max_count * 1.1])
            if i == 0:
                ax.set_ylabel('Observed')
            ax.set_xlabel('Expected')
            total = sum(obs_counts)
            texts = []

            pvals = []
            for e, o in zip(exp_counts, obs_counts):
                f = fisher_exact([[e, total - e], [o, total - o]])
                pvals.append(f[1])

            qvals = multipletests(pvals, method='fdr_bh')[1]

            pvals, qvals, aachanges, exp_counts, obs_counts = sort_multiple_arrays_using_one(pvals, qvals,
                                                                                             list(aachanges),
                                                                                             exp_counts, obs_counts)
            for j, (a, e, o, p, q) in enumerate(zip(aachanges, exp_counts, obs_counts, pvals, qvals)):
                if (q <= sig_threshold and use_qval) or (p <= sig_threshold and not use_qval):
                    texts.append(ax.text(e, o, "{}>{}".format(*a)))
                    ax.scatter([e], [o], s=5, color='r')
                if max_texts is not None and j >= max_texts-1:
                    break
            adjust_text(texts, arrowprops=dict(arrowstyle='->', color='red', alpha=0.4), ax=ax)
            ax.set_xlim(left=0)
            ax.set_ylim(bottom=0)
            ax.set_title('AA changes - \n' + spectrum.name)

        if show_plot:
            plt.show()
        if return_fig:
            return fig

    def plot_sliding_3D_window(self, distance=10, normalise=True, include_chains=None,
                               figsize=(15, 5), spectra=None, colours=None, show_legend=True, legend_args=None,
                               show_arcs=False, arc_scale=0.001, arc_alpha=0.01, min_arc_residue_gap=5,
                               show_plot=False, colourmap=autumn, colourmap_different_chain=winter, return_fig=False,
                               show_residues=False, xlim=None, ylim=None):
        if self.pdb_id is not None:
            try:
                u = MDAnalysis.Universe(os.path.join(self.project.pdb_directory, self.pdb_id.lower() + '.pdb.gz'))
            except FileNotFoundError as e:
                try:
                    u = MDAnalysis.Universe(os.path.join(self.project.pdb_directory, self.pdb_id.lower() + '.pdb'))
                except FileNotFoundError as e:
                    print('Not plotting the 3D window. PDB file not found.')
                    return

            spectra = self._get_spectra(spectra)

            colours = self._get_plot_colours(colours, spectra)

            fig, ax = plt.subplots(figsize=figsize)

            seq_pdb_residues = sorted(get_pdb_positions(self.null_mutations['residue'].unique(),
                                                        self.pdb_id, self.pdb_chain, self.project.sifts_directory,
                                                        self.project.download_sifts)['pdb position'].astype(int))

            # Get some normalising factors, so the total mutations in the gene is the same in the observed and expected plots
            normalising_factors = {
                spectrum.name: self.num_mutations / self.null_mutations[spectrum.rate_column].sum() for spectrum in
            spectra
            }
            colourmap_norm = Normalize(vmin=seq_pdb_residues[0],vmax=seq_pdb_residues[-1])

            if include_chains is None:
                include_chains = {self.pdb_chain}

            expected_counts = {spectrum.name: np.empty(len(seq_pdb_residues)) for spectrum in spectra}
            observed_counts = np.empty(len(seq_pdb_residues))

            for i, res in enumerate(seq_pdb_residues):
                surrounding_res = get_3D_window(u, self.pdb_chain, res, distance)
                surrounding_res = [r for r in surrounding_res if r[0] in include_chains]
                residue_window = [r[1] for r in surrounding_res]
                if normalise:
                    norm_fac = len(residue_window)
                else:
                    norm_fac = 1

                if show_arcs:
                    for r in surrounding_res:
                        if r[0] == self.pdb_chain and abs(r[1] - res) >= min_arc_residue_gap:
                            plot_arc(ax, res, r[1], colourmap, colourmap_norm, arc_scale=arc_scale, alpha=arc_alpha)
                        elif r[0] != self.pdb_chain:
                            plot_arc(ax, res, r[1], colourmap_different_chain, colourmap_norm,
                                     arc_scale=arc_scale, alpha=arc_alpha)

                window_mutations = self.observed_mutations[self.observed_mutations['residue'].isin(residue_window)]
                obs = len(window_mutations)

                null_window_mutations = self.null_mutations[self.null_mutations['residue'].isin(residue_window)]

                for spectrum in spectra:
                    total_window_rate = null_window_mutations[spectrum.rate_column].sum()
                    normalised_window_rate = total_window_rate * normalising_factors[spectrum.name]
                    expected_counts[spectrum.name][i] = normalised_window_rate / norm_fac

                observed_counts[i] = obs / norm_fac

            for spectrum, colour in zip(spectra, colours):
                plt.plot(seq_pdb_residues, expected_counts[spectrum.name],
                         label=spectrum.name, c=colour)
            observed_counts = np.array(observed_counts)
            plt.plot(seq_pdb_residues, observed_counts, c=colours[len(spectra)], label='Observed')
            if xlim is not None:
                ax.set_xlim(xlim)
            else:
                plt.xlim([seq_pdb_residues[0] - 1, seq_pdb_residues[-1] + 1])
            if ylim is not None:
                ax.set_ylim(ylim)
            elif not show_arcs:
                ax.set_ylim(bottom=0)

            if show_arcs:
                # Remove yticks below zero
                ax.set_yticks([t for t in ax.get_yticks() if t >= 0])

            plt.xlabel('Residue')

            self._format_xticks(ax)

            if normalise:
                plt.ylabel('Mutations per codon')
            else:
                plt.ylabel('Mutations')

            if show_legend:
                if legend_args is None:
                    legend_args = {}
                plt.legend(**legend_args)

            if show_residues:
                self._show_residues(fig, ax)

            if show_plot:
                plt.show()
            if return_fig:
                return fig

    def plot_sliding_window_totalled_score(self, window_size=20, window_step=1,
                                          spectra=None, show_legend=True, figsize=(15, 5), legend_args=None,
                                          show_plot=False,
                                          colours=None, return_fig=False, show_residues=False, xlim=None, ylim=None):
        fig, ax = plt.subplots(figsize=figsize)
        spectra = self._get_spectra(spectra)

        colours = self._get_plot_colours(colours, spectra)

        start, end = self.null_mutations['residue'].min(), self.null_mutations['residue'].max()

        starts = np.arange(start, end + 2 - window_size, window_step)
        ends = starts + window_size - 1

        # Get some normalising factors, so the total mutations in the gene is the same in the observed and expected plots
        normalising_factors = {
            spectrum.name: self.num_mutations / self.null_mutations[spectrum.rate_column].sum() for spectrum in spectra
        }

        expected_counts = {spectrum.name: np.empty(len(starts)) for spectrum in spectra}
        observed_counts = np.empty(len(starts))
        for i, (s, e) in enumerate(zip(starts, ends)):
            obs_window_mutations = self.observed_mutations[(self.observed_mutations['residue'] >= s) &
                                                           (self.observed_mutations['residue'] <= e)]
            observed_counts[i] = obs_window_mutations['score'].sum()

            null_window_mutations = self.null_mutations[(self.null_mutations['residue'] >= s) &
                                                        (self.null_mutations['residue'] <= e)]
            for spectrum in spectra:
                total_window_score = (
                            null_window_mutations[spectrum.rate_column] * null_window_mutations['score']).sum()
                mean_window_rate = total_window_score * normalising_factors[spectrum.name]

                expected_counts[spectrum.name][i] = mean_window_rate

        self._plot_sliding_window_results(starts, window_size, spectra, colours,
                                     expected_counts, observed_counts, fig, ax, xlim, start, end, ylim,
                                     show_legend, legend_args, show_residues, show_plot,
                                          ylabel='Total score per residue')

        if return_fig:
            return fig

    def plot_mutation_rate_scatter(self, spectra=None, metric_plot_scale=None, marker_size_from_count=True,
                                   base_marker_size=10, show_plot=False,
                                   unobserved_mutation_colour='C0', missense_mutation_colour='C1',
                                   synonymous_mutation_colour='C2', nonsense_mutation_colour='C3',
                                   show_legend=True, figsize=(15, 5), legend_args=None, return_fig=False,
                                   xlim=None, unmutated_marker_size=1, mut_rate_plot_scale=None,
                                   mutations_to_annotate=None, annotation_column='aachange',
                                   annotation_offset=(0, 0), unobserved_alpha=1, observed_alpha=1,
                                   sections_for_colours=None, score_regions_for_colours=None,
                                   score_region_colours=None, colour_unmutated_by_scores=False, hotspots_in_foreground=False,
                                   observed_marker=None, unobserved_marker=None,
                                   show_observed_only=False, show_null_only=False
                                   ):
        spectra = self._get_spectra(spectra)

        if mutations_to_annotate is not None:
            mutations_to_annotate = mutations_to_annotate.drop_duplicates(subset=['ds_mut_id'])

        fig, axes = plt.subplots(nrows=1, ncols=len(spectra), squeeze=False, figsize=figsize)

        if len(self.observed_mutations) == 0:
            show_null_only = True

        if not show_null_only:
            mut_counts = self.observed_mutations['ds_mut_id'].value_counts()

        if not show_observed_only:
            if len(self.observed_mutations) > 0:
                null_to_plot = self.null_mutations[~
                    self.null_mutations['ds_mut_id'].isin(self.observed_mutations['ds_mut_id'])]
            else:
                null_to_plot = self.null_mutations
        for i, spectrum in enumerate(spectra):
            ax = axes[0, i]
            if not show_observed_only:
                ax.scatter(null_to_plot[spectrum.rate_column], null_to_plot['score'],
                       s=unmutated_marker_size, alpha=unobserved_alpha,
                       c=unobserved_mutation_colour, label=None, marker=unobserved_marker)
            if not show_null_only:
                for effect, col in zip(['synonymous', 'missense', 'nonsense'], [synonymous_mutation_colour,
                                                                    missense_mutation_colour,
                                                                    nonsense_mutation_colour]):
                    muts = self.observed_mutations[self.observed_mutations['effect'] == effect]
                    if len(muts) > 0:
                        _plot_single_scatter_category(muts, mut_counts, spectrum.rate_column, 'score',
                                                      marker_size_from_count,
                                                      base_marker_size, col, effect,
                                                      observed_alpha, hotspots_in_foreground, ax=ax, marker=observed_marker)

            if sections_for_colours is not None:
                if colour_unmutated_by_scores and not show_observed_only:
                    colour_mutations_by_scores(null_to_plot, None, spectrum.rate_column, 'score',
                                               sections_for_colours,
                                               score_regions_for_colours, score_region_colours,
                                               marker_size_from_count=False,
                                               base_marker_size=unmutated_marker_size, alpha=unobserved_alpha,
                                               hotspots_in_foreground=False, use_null=True, marker=unobserved_marker)
                if not show_null_only:
                    colour_mutations_by_scores(self.observed_mutations, mut_counts, spectrum.rate_column, 'score',
                                           sections_for_colours,
                                           score_regions_for_colours, score_region_colours, marker_size_from_count,
                                           base_marker_size, observed_alpha, hotspots_in_foreground, ax=ax,
                                           marker=observed_marker)

            ax.set_xlabel('Mutation rate')
            try:
                ax.set_ylabel(self.project.lookup.name)
            except AttributeError as e:
                pass

            if xlim is not None:
                ax.set_xlim(xlim)
            else:
                ax.set_xlim(left=0)

            if metric_plot_scale is not None:
                ax.set_yscale(metric_plot_scale)
            if mut_rate_plot_scale is not None:
                ax.set_xscale(mut_rate_plot_scale)

            ax.set_title(spectrum.name)

            if show_legend:
                if legend_args is None:
                    legend_args = {}
                ax.legend(**legend_args)

            if mutations_to_annotate is not None:
                annotate_mutations_on_plot(ax, mutations_to_annotate, annotation_column, xcol=spectrum.rate_column,
                                           ycol='score',
                                        annotation_offset=annotation_offset)

        if show_plot:
            plt.show()
        if return_fig:
            return fig

    def plot_bar_observations(self, groupby='residue', figsize=(15, 5), facecolour='k', linewidth=0, edgecolour='k',
                              show_legend=False,legend_args=None, show_plot=False, return_fig=False,
                              show_residues=False, xlim=None, ylim=None,
                              show_lollipop_tops=False, lollipop_top_size=50,
                              binning_regions=None, normalise_by_region_size=True):

        fig, ax = plt.subplots(figsize=figsize)

        if binning_regions is not None:
            x, bins = pd.cut(self.observed_mutations[groupby], binning_regions, retbins=True)
            counts = self.observed_mutations.groupby(x).count()['chr']
            widths = np.diff(bins)
            if normalise_by_region_size:
                counts = counts / widths
            plt.bar(bins[:-1] + widths / 2, counts, color=facecolour, width=widths,
                    linewidth=linewidth, edgecolor=edgecolour)
        else:
            counts = self.observed_mutations.groupby(groupby).count()['chr']  # Pick arbitrary column we know is there.
            plt.bar(counts.index, counts, color=facecolour)
            if show_lollipop_tops:
                plt.scatter(counts.index, counts, color=facecolour, s=lollipop_top_size,
                            linewidth=linewidth, edgecolor=edgecolour)

        if xlim is not None:
            ax.set_xlim(xlim)
        if ylim is not None:
            ax.set_ylim(ylim)
        else:
            ax.set_ylim(bottom=0)

        plt.ylabel('Mutations per {}'.format(groupby))
        plt.xlabel(groupby)

        self._format_xticks(ax)

        if show_legend:
            if legend_args is None:
                legend_args = {}
            plt.legend(**legend_args)

        if show_residues:
            self._show_residues(fig, ax)

        if show_plot:
            plt.show()

        if return_fig:
            return fig

    def get_oncodrive_score_input(self):
        if self.null_scores is None:
            self.apply_scores()
        self.null_mutations['chr'] = self.chrom
        self.null_mutations['element'] = self.section_id
        return self.null_mutations[['chr', 'pos', 'ref', 'mut', 'score', 'element']].sort_values('pos')

    def get_oncodrive_regions(self):
        # A general function that should work with missing bases (e.g. for pdb files with missing residues)
        # Not the most efficient for cases like uses a full transcript.
        if self.null_mutations is None:
            self.load_section_mutations()
        positions = sorted(self.null_mutations['pos'].unique())
        last_p = positions[0]
        starts = [last_p]
        ends = []
        for p in positions[1:]:
            if p != last_p + 1:
                ends.append(last_p)
                starts.append(p)
            last_p = p
        ends.append(last_p)
        regions = pd.DataFrame({'CHROMOSOME': self.chrom, 'START': starts,
                                'END': ends, 'ELEMENT': self.section_id})
        return regions


def plot_chi_bins_y(bins, linestyle='dashed', linewidth=1, alpha=0.5, zorder=0, colour=None):
    # For overlaying on other plots with the metric on the y-axis
    xlim = plt.gca().get_xlim()
    plt.hlines(bins, xlim[0], xlim[1], linestyles=linestyle, linewidth=linewidth, alpha=alpha, zorder=zorder,
               color=colour)
    plt.xlim(xlim)


def plot_chi_bins_x(bins, linestyle='dashed', linewidth=1, alpha=0.5, zorder=0, colour=None):
    # For overlaying on other plots with the metric on the x-axis
    ylim = plt.gca().get_ylim()
    plt.vlines(bins, ylim[0], ylim[1], linestyles=linestyle, linewidth=linewidth, alpha=alpha, zorder=zorder,
               color=colour)
    plt.ylim(ylim)


def get_3D_window(u, chain, residue, distance):
    # Get all atoms with the distance of the residue
    g = u.select_atoms("around {} segid {} and resnum {}".format(distance, chain, residue))
    # Remove any non-protein atoms, e.g. from water
    g = g.select_atoms('protein')

    residues = {(chain, residue)}  # Must add the residue itself
    for a in g:
        residues.add((a.segid, a.resid))
    return residues


def plot_arc(ax, res1, res2, colourmap, colourmap_norm, arc_scale, alpha=0.1):
    x = (res1+res2)/2
    a = abs(res1-res2)
    min_res = min(res1, res2)
    b = a*arc_scale
    if b != 0:
        # Arcs
        if isinstance(colourmap, Colormap):
            c = colourmap(colourmap_norm(min_res))
        else:
            c = colourmap  # Assume this is a str, or rgb tuple rather than a colour map. Do not scale.
        ax.add_patch(Arc((x, 0), a, b, theta1=180.0, theta2=360.0, edgecolor=c, alpha=alpha))
