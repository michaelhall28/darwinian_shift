import sys
import numpy as np
import pandas as pd
from adjustText import adjust_text
from scipy.stats import fisher_exact
from statsmodels.stats.multitest import multipletests
import matplotlib.pylab as plt
from matplotlib import collections
from matplotlib.cm import autumn, winter
from matplotlib.colors import Normalize, Colormap
from matplotlib.ticker import MaxNLocator, StrMethodFormatter
import seaborn as sns
try:
    import MDAnalysis
except ImportError as e:
    pass
import os
from darwinian_shift.statistics import get_median, get_cdf, ChiSquareTest, BinomTest
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
    """
    Class for the analysis of a single transcript/part of transcript

    Stores the null and observed mutations in the regions, applies the metric scores, run the statistical tests
    and functions to plot the results.
    """

    def __init__(self, transcript, start=None, end=None, section_id=None, pdb_id=None, pdb_chain=None,
                 excluded_mutation_types=None, included_mutation_types=None, included_residues=None,
                 excluded_residues=None, lookup=None,
                 **kwargs):
        """

        :param transcript: Transcript object. Used to generate the null mutations.
        :param start: Will exclude residues before this one from the analysis. If None, will start from the first
        residue of the protein.
        :param end: Will exclude residues after this one from the analysis. If None, will end at the last
        residue of the protein.
        :param section_id: A label for the section to be analysed. Can be useful when running multiple analyses and
        collecting results. If None, will be named automatically based on the transcript_id or pdb file and chain.
        :param pdb_id: For analyses that use a protein structure. Four letter ID of the pdb file to use.
        :param pdb_chain: For analyses that use a protein structure. The chain to use for the analysis.
        :param excluded_mutation_types: Can be string or list of strings. Mutation types to exclude from the
        analysis. E.g. ['synonymous', 'nonsense']. If None, will use the excluded_mutation_types of the project.
        :param included_mutation_types: Can be string or list of strings. Mutation types to include in the
        analysis. E.g. ['synonymous', 'nonsense']. If None, will use the included_mutation_types of the project.
        :param included_residues: List or array of integers. The residues to analyse. If None, will analyse all
        residues (except those excluded by other arguments).
        :param excluded_residues: List or array of integers. The residues to exclude from the analysis.
        :param lookup: The class object or function used to score the mutations. If None, will use the lookup of
        the project.
        :param kwargs: Any additional attributes that will be assigned to the Section object created.
        These can be used by the lookup class.
        """
        self.project = transcript.project  # The darwinian shift class that can run over multiple sections
        self.transcript = transcript
        self.gene = transcript.gene
        self.transcript_id = transcript.transcript_id
        self.chrom = transcript.chrom

        self.statistics = self.project.statistics
        if not isinstance(self.statistics, (list, tuple, set)):
            self.statistics = [self.statistics]
        elif isinstance(self.statistics, (list, tuple, set)):
            self.statistics = list(self.statistics)

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

        if lookup is not None:
            self.lookup = lookup
        else:
            self.lookup = self.project.lookup

        self._scoring_complete = False

        ### Results. Populated during self.run
        self.null_mutations = None
        self.observed_mutations = None
        self.num_mutations = None
        self.null_scores = None
        self.observed_values = None
        self.ref_mismatch_count = None

        self.statistical_results = None

        self.repeat_proportion = None

        self.load_section_mutations()

    def change_lookup_inplace(self, lookup):
        self.lookup = lookup
        self._scoring_complete = False  # Reset the mutations. These need scoring again with the new lookup

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
                                           on=['pos', 'ref', 'mut'], suffixes=('_input', ''))
        self.observed_mutations = self.observed_mutations[~pd.isnull(self.observed_mutations['null_exists'])]
        self.observed_mutations = self.observed_mutations.drop('null_exists', axis=1)
        self.null_mutations = self.null_mutations.drop('null_exists', axis=1)
        # Any rows missing in the null will have had nan and made the int columns floats. Convert back.
        int_cols = ['residue', 'base', 'cdspos', 'mut_count_glob_k3', 'seq_count_glob_k3']
        self.observed_mutations = self.observed_mutations.astype({c: int for c in int_cols})

        self.num_mutations = len(self.observed_mutations)  # Initial number of observed mutations (may be filtered later)

        self._add_mutation_id_column()

    def run(self, plot_mc=False, spectra=None, statistics=None):
        """
        Run the statistical analysis of the section.
        :param plot_mc: If True, will plot a histogram of the output of any Monte Carlo tests used.
        :param spectra: The mutational spectrum or list of mutational spectra to use for the statistical tests. If None,
        will run for all of the spectra in the project.
        :param statistics: The statistical tests to run. If None, will run all of the statistical tests in the project.
        :return:
        """
        self.apply_scores()

        # Compare distributions
        self._run_statistical_tests(plot_mc, spectra=spectra, statistics=statistics)

    def get_results_dictionary(self):
        """
        Get a dictionary of results for the section.
        :return: dict
        """
        res = {k: getattr(self, k) for k in self.project.result_columns}
        res.update(self.statistical_results)
        return res

    def get_pvalues(self):
        """
        Get the p-values of any statistical tests run for the section.
        :return:
        """
        if self.statistical_results is None:
            self._run_statistical_tests()
        p = {k: v for (k, v) in self.statistical_results.items() if 'pvalue' in k}
        return p

    def apply_scores(self):
        """
        Uses the lookup to apply scores to the null and observed mutations. These will be placed in the 'score' column
        of the self.null_mutations and self.observed_mutations dataframes.
        :return: None
        """
        if not self._scoring_complete:
            # Get scores, and make sure all results are numpy arrays of floats for consistency
            self.null_mutations['score'] = np.array(self.lookup(self)).astype(float)

            # Exclude all cases without a score
            self.null_mutations = self.null_mutations[~pd.isnull(self.null_mutations['score'])]
            if len(self.null_mutations) == 0:
                raise NoMutationsError('No scores for {} {}'.format(self.section_id, self.gene))
            self.null_scores = self.null_mutations['score'].values

            # Match the observed mutations with the null mutations.
            self.observed_mutations = pd.merge(self.observed_mutations,
                                               self.null_mutations[['pos', 'ref', 'mut', 'score']],
                                               on=['pos', 'ref', 'mut'],
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

    def _run_statistical_tests(self, plot=False, spectra=None, statistics=None):
        if self.statistical_results is None:
            self.statistical_results = {}
        spectra = self._get_spectra(spectra)
        if statistics is None:
            statistics = self.statistics
        else:
            if not isinstance(statistics, (list, tuple, set)):
                statistics = [statistics]
            for s in statistics:
                if s not in self.statistics:
                    self.statistics.append(s)

        self.statistical_results['observed_median'] = self.observed_values.median()
        self.statistical_results['observed_mean'] = self.observed_values.mean()
        for spectrum in spectra:
            self.statistical_results['expected_median_' + spectrum.name] = get_median(self.null_scores,
                                                                                      self.null_mutations[spectrum.rate_column])
            self.statistical_results['median_shift_' + spectrum.name] = self.statistical_results['observed_median'] - \
                                                                        self.statistical_results['expected_median_' + spectrum.name]

            self.statistical_results[
                'expected_mean_' + spectrum.name] = np.sum(self.null_scores *
                                                           self.null_mutations[spectrum.rate_column])\
                                                    /self.null_mutations[spectrum.rate_column].sum()
            self.statistical_results['mean_shift_' + spectrum.name] = self.statistical_results['observed_mean'] - \
                                                                      self.statistical_results['expected_mean_' + spectrum.name]
            for statistic in statistics:
                self.statistical_results.update(statistic(self, spectrum, plot))

        # self.repeat_proportion = calculate_repeat_proportion(self.null_scores)

    def plot(self, spectra=None, plot_scale=None,
             marker_size_from_count=True, base_marker_size=10, colours=None):
        """
        Plots a small selection of plots showing the location and scores of mutations in the section.
        More plots and plotting options are available using the other .plot_ functions
        :param spectra: The mutational spectrum or list of mutational spectra to use.
        :param plot_scale: The scaling for the plot axis showing the metric scores.
        :param marker_size_from_count: If True (default), the size of the scatter plot marker showing a mutation will
        be based on the frequency of that mutation in the data. If False, all markers will have the same size.
        :param base_marker_size: The size of a marker for a mutation occurring once in the data (or size for all
        mutation markers if marker_size_from_count=False).
        :param colours: List of colours for plots. First colour is for the observed data, the subsequent colours are
        for plotting the null distributions from each of the mutational spectra given.
        :return: None
        """
        spectra = self._get_spectra(spectra)
        if self.observed_values is None:
            self.apply_scores()  # Make sure all the values needed for plotting are generated

        self.plot_sliding_window(spectra=spectra, colours=colours, show_plot=True)
        self.plot_scatter(plot_scale, marker_size_from_count, base_marker_size, show_plot=True)
        self.plot_cdfs(spectra, plot_scale, colours=colours, show_plot=True)

    def _get_plot_colours(self, colours, num_lines):
        if colours is None:
            colours = plt.rcParams['axes.prop_cycle'].by_key()['color']  # Default colour cycle

            if num_lines > len(colours):
                print('Too many spectra to plot with unique default colours.')
                colours = [colours[i%len(colours)] for i in range(num_lines)]

        elif num_lines > len(colours):
            print('Too many spectra to plot with given colours.')
            colours = [colours[i % len(colours)] for i in range(num_lines)]
        return colours

    def _get_seaborn_colour_palette(self, colours, num):
        """
        The first colour in the list is for the observed data.
        Currently the observed data is last in the box and violinplots, so swap the colour order
        :param colours:
        :return:
        """
        return sns.color_palette(list(colours[1:num]) + [colours[0]])

    def _set_xticks(self, ax, ticks, labels):
        # Just because this is used a few times
        ax.set_xticks(ticks)
        ax.set_xticklabels(labels)

    def _format_xticks(self, ax):
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        formatter = StrMethodFormatter("{x:.0f}")
        ax.xaxis.set_major_formatter(formatter)


    def _show_residues(self, fig, ax):
        ax.tick_params(axis='x', direction='out', pad=12, length=0)
        fig.subplots_adjust(bottom=0.20)
        ax2 = ax.twiny()
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
                                     show_legend, legend_args, show_residues, show_plot, ylabel,
                                     observed_label='Observed', divide_by_window_size=True, plot_kwargs_obs=None,
                                     plot_kwargs_exp=None, show_expected=True):
        obs_linewidth = 3
        if plot_kwargs_obs is None:
            plot_kwargs_obs = {}
        elif 'linewidth' in plot_kwargs_obs:
            obs_linewidth = plot_kwargs_obs.pop('linewidth')

        if plot_kwargs_exp is None:
            plot_kwargs_exp = {}

        plot_x = (starts + window_size / 2 - 0.5)
        if show_expected:
            for spectrum, colour in zip(spectra, colours[1:]):
                if divide_by_window_size:
                    exp_ = expected_counts[spectrum.name] / window_size
                else:
                    exp_ = expected_counts[spectrum.name]
                ax.plot(plot_x, exp_, label=spectrum.name, c=colour, **plot_kwargs_exp)

        if divide_by_window_size:
            obs_ = observed_counts / window_size
        else:
            obs_ = observed_counts

        ax.plot(plot_x, obs_, label=observed_label, c=colours[0],
                 linewidth=obs_linewidth, **plot_kwargs_obs)
        if xlim is not None:
            ax.set_xlim(xlim)
        else:
            ax.set_xlim([start, end])
        if ylim is not None:
            ax.set_ylim(ylim)
        else:
            ax.set_ylim(bottom=0)
        ax.set_ylabel(ylabel)
        ax.set_xlabel('Residue')

        self._format_xticks(ax)

        if show_legend:
            if legend_args is None:
                legend_args = {}
            ax.legend(**legend_args)

        if show_residues:
            self._show_residues(fig, ax)

        if show_plot:
            plt.show()

    def plot_sliding_window(self, window_size=20, window_step=1,
                            spectra=None, show_legend=True, figsize=(15, 5), legend_args=None, show_plot=False,
                            colours=None, return_fig=False, show_residues=False, xlim=None, ylim=None, ax=None,
                            divide_by_expected_rate=False, observed_label='Observed', return_values=False,
                            yaxis_right=False, plot_kwargs_obs=None, plot_kwargs_exp=None, show_expected=True):
        """
        Plots a sliding window of mutation counts across the residues analysed. By default will show the observed data
        and the expected results under each mutational spectrum given. Alternatively, if divide_by_expected_rate=True,
        will instead show the observed data relative to the expected results.
        :param window_size: Size of the sliding window in residues.
        :param window_step: Size of the step between starts of adjacent windows in residues.
        :param spectra: The mutational spectrum or list of mutational spectra to use.
        :param show_legend: Will show the legend listing the names of the mutational spectra.
        :param figsize: Size of the figure.
        :param legend_args: kwargs to pass to matplotlib for the legend appearance.
        :param show_plot: If True, will call plt.show()
        :param colours: List of colours for plots. First colour is for the observed data, the subsequent colours are
        for plotting the null distributions from each of the mutational spectra given.
        :param return_fig: If True, will return the figure. Used for testing.
        :param show_residues: If True, will show the reference amino acids on the x-axis.
        :param xlim: Limits for the x-axis.
        :param ylim: Limits for the y-axis.
        :param ax: Matplotlib axis to plot on. If None, will create a new figure.
        :param divide_by_expected_rate: If True, will show the observed data relative to the expected results. If False,
        will show the observed data and expected results as separate lines.
        :param observed_label: Label for the observed data in the legend. By default this is 'Observed'.
        :param return_values: If True, will return the values for the lines plotted.
        :param yaxis_right: If True, will show the y-axis on the right of the plot.
        :param plot_kwargs_obs: Additional kwargs for the appearance of the observed line.
        :param plot_kwargs_exp: Additional kwargs for the appearance of the expected lines.
        :param show_expected: Set to False to hide the expected sliding window.
        :return: By default, None.
        If return_values=True, will return a tuple (observed_counts, expected_counts), where observed_counts is a
        numpy array and expected_counts is a dictionary of numpy arrays.
        Else if return_fig=True, will return the figure.
        """
        start, end = self.null_mutations['residue'].min(), self.null_mutations['residue'].max()

        starts = np.arange(start, end + 2 - window_size, window_step)
        if len(starts) == 0:
            print("Region too small compared to window size/step.")
            return
        ends = starts + window_size - 1

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
            if yaxis_right:
                ax.yaxis.tick_right()
        else:
            fig = ax.figure
            if yaxis_right:
                ax = ax.twinx()

        spectra = self._get_spectra(spectra)

        if divide_by_expected_rate:
            num_lines = len(spectra)  # One line for each spectrum
        else:
            num_lines = len(spectra) + 1   # One line for each spectrum plus the observed mutations
        colours = self._get_plot_colours(colours, num_lines)

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

        if divide_by_expected_rate:
            res = {}
            for spectrum, colour in zip(spectra, colours):
                obs_w = observed_counts / expected_counts[spectrum.name]
                res[spectrum.name] = obs_w
                self._plot_sliding_window_results(starts, window_size, [], [colour],
                                                  [], obs_w, fig, ax, xlim, start, end, ylim,
                                                  show_legend, legend_args, show_residues, show_plot,
                                                  ylabel='Mutations relative to expected',
                                                  observed_label=observed_label, divide_by_window_size=False,
                                                  plot_kwargs_obs=plot_kwargs_obs,
                                                  plot_kwargs_exp=plot_kwargs_exp)
            if return_values:
                return res
        else:
            self._plot_sliding_window_results(starts, window_size, spectra, colours,
                                          expected_counts, observed_counts, fig, ax, xlim, start, end, ylim,
                                          show_legend, legend_args, show_residues, show_plot,
                                          ylabel='Mutations per codon', observed_label=observed_label,
                                              plot_kwargs_obs=plot_kwargs_obs,
                                              plot_kwargs_exp=plot_kwargs_exp, show_expected=show_expected)
            if return_values:
                return observed_counts, expected_counts

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
                     show_observed_only=False, show_unobserved_only=False, ax=None, yaxis_right=False,
                     unobserved_zorder=0, missense_zorder=2, synonymous_zorder=1, nonsense_zorder=3):
        """
        Scatter plot of mutation residue vs mutation score.
        :param plot_scale: Scale for the y-axis (mutation scores). Passed to matplotlib. 'log', 'symlog' etc.
        :param marker_size_from_count: If True (default), the size of the marker will be based on the frequency of
        that mutation in the data. If False, all markers will have the same size.
        :param base_marker_size: The size of a marker for a mutation occurring once in the data (or size for all
        mutation markers if marker_size_from_count=False).
        :param show_plot: If True, will call plt.show().
        :param unobserved_mutation_colour: Colour for mutations that are not seen in the data. Hide these mutations
        entirely by setting show_observed_only=True.
        :param missense_mutation_colour: Colour for missense mutations.
        :param synonymous_mutation_colour: Colour for synonymous mutations.
        :param nonsense_mutation_colour: Colour for nonsense mutations.
        :param show_legend: If True (default), show the legend.
        :param figsize: Size of the figure.
        :param legend_args: kwargs to pass to matplotlib for the legend appearance.
        :param return_fig: If True, will return the figure. Used for testing.
        :param show_residues: If True, will show the reference amino acids on the x-axis.
        :param xlim: Limits for the x-axis.
        :param unmutated_marker_size: Size of the markers for mutations not seen in the data.
        :param unobserved_alpha: Alpha for the markers for mutations not seen in the data.
        :param observed_alpha: Alpha for the markers of observed mutations.
        :param sections_for_colours: List of additional Section objects that can be used to colour mutations based on
        further scores.  To be used with score_regions_for_colours, and optionally score_region_colours.
        :param score_regions_for_colours: List of the regions to colour using the sections_for_colours. Each region is a
        tuple of lower and upper score bound. i-th region applies to the i-th section in sections_for_colours.
        :param score_region_colours: List of colours to use for sections_for_colours. i-th colour applies to the
        i-th section in sections_for_colours.
        :param colour_unmutated: If True, the null mutations will also be coloured according to the
        section_for_colours and score_regions_for_colours arguments. If False, null mutations will be plotted with the
        unobserved_mutation_colour.
        :param hotspots_in_foreground: If True, will plot the most frequent mutations (the largest markers)
        in the foreground. If False, will plot the largest markers in the background so other mutations they overlap
        with are visible.
        :param observed_marker: Shape of the marker to use for observed mutations.
        :param unobserved_marker: Shape of the marker to use for unobserved mutations.
        :param show_observed_only: If True, will not show the unobserved mutations.
        :param show_unobserved_only: If True, will only show the unobserved mutations.
        :param ax: Matplotlib axis to plot on. If None, will create a new figure.
        :param yaxis_right: If True, will show the y-axis on the right of the plot.
        :param unobserved_zorder: The zorder for the unobserved mutations.
        :param missense_zorder: The zorder for the observed missense mutations.
        :param synonymous_zorder: The zorder for the observed synonymous mutations.
        :param nonsense_zorder: The zorder for the observed nonsense mutations.
        :return: By default, None. If return_fig=True, will return the figure.
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
            if yaxis_right:
                ax.yaxis.tick_right()
        else:
            fig = ax.figure
            if yaxis_right:
                ax = ax.twinx()

        if not show_observed_only:
            null_to_plot = self.null_mutations[~self.null_mutations['ds_mut_id'].isin(self.observed_mutations['ds_mut_id'])]
            ax.scatter(null_to_plot['residue'], null_to_plot['score'], s=unmutated_marker_size,
                    alpha=unobserved_alpha, c=unobserved_mutation_colour, label=None, linewidth=0,
                    marker=unobserved_marker, zorder=unobserved_zorder)

        if not show_unobserved_only:
            mut_counts = self.observed_mutations['ds_mut_id'].value_counts()
            for effect, col, z in zip(['synonymous', 'missense', 'nonsense'], [synonymous_mutation_colour,
                                                                        missense_mutation_colour,
                                                                        nonsense_mutation_colour],
                                      [synonymous_zorder, missense_zorder, nonsense_zorder]):

                muts = self.observed_mutations[self.observed_mutations['effect'] == effect]
                if len(muts) > 0:
                    _plot_single_scatter_category(muts, mut_counts, 'residue', 'score', marker_size_from_count,
                                              base_marker_size, col, effect,
                                              observed_alpha, hotspots_in_foreground, marker=observed_marker, ax=ax,
                                                  zorder=z)

        if sections_for_colours is not None:
            if colour_unmutated and not show_observed_only:
                colour_mutations_by_scores(null_to_plot, None, 'residue', 'score',
                                           sections_for_colours,
                                           score_regions_for_colours, score_region_colours,
                                           marker_size_from_count=False,
                                           base_marker_size=unmutated_marker_size, alpha=unobserved_alpha,
                                           hotspots_in_foreground=False, use_null=True, marker=unobserved_marker, ax=ax,
                                           zorder=unobserved_zorder)

            if not show_unobserved_only:
                colour_mutations_by_scores(self.observed_mutations, mut_counts, 'residue', 'score', sections_for_colours,
                                           score_regions_for_colours, score_region_colours, marker_size_from_count,
                                           base_marker_size, observed_alpha, hotspots_in_foreground,
                                           marker=observed_marker, ax=ax,
                                           zorder=missense_zorder)

        ax.set_xlabel('Residue')
        try:
            ax.set_ylabel(self.lookup.name)
        except AttributeError as e:
            pass
        if xlim is not None:
            ax.set_xlim(xlim)

        self._format_xticks(ax)

        if plot_scale is not None:
            ax.set_yscale(plot_scale)
        if show_legend:
            if legend_args is None:
                legend_args = {}
            ax.legend(**legend_args)

        if show_residues:
            self._show_residues(fig, ax)

        if show_plot:
            plt.show()
        if return_fig:
            return fig

    def plot_violinplot(self, spectra=None, plot_scale=None, violinplot_bw=None, show_plot=False, colours=None,
                        figsize=(5, 5), return_fig=False, ax=None):
        """
        Violinplot of the expected and observed distributions of mutation scores.
        :param spectra: The mutational spectrum or list of mutational spectra to use.
        :param plot_scale: Scale for the y-axis (mutation scores). Passed to matplotlib. 'log', 'symlog' etc.
        :param violinplot_bw: Bandwidth for the violins.
        :param show_plot: If True, will call plt.show().
        :param colours: List of colours. First colour is for the observed data, the subsequent colours are
        for plotting the null distributions from each of the mutational spectra given.
        :param figsize: Size of the figure.
        :param return_fig: If True, will return the figure. Used for testing.
        :param ax: Matplotlib axis to plot on. If None, will create a new figure.
        :return: By default, None. If return_fig=True, will return the figure.
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.figure
        spectra = self._get_spectra(spectra)
        colours = self._get_plot_colours(colours, len(spectra)+1)
        if violinplot_bw is None:
            # Must set this otherwise will be different for the obs and exp distributions.
            violinplot_bw = (max(self.null_scores) - min(self.null_scores)) / 200

        data = [get_distribution_from_mutational_spectrum(self.null_scores, self.null_mutations[spectrum.rate_column]) for spectrum in spectra]
        data.append(self.observed_values)
        sns.violinplot(
            data=data, cut=0,
            bw=violinplot_bw, palette=self._get_seaborn_colour_palette(colours, len(spectra) + 1), ax=ax)
        self._set_xticks(ax, range(len(data)), ['Expected\n' + spectrum.name for spectrum in spectra]  + ['Observed'])
        try:
            ax.set_ylabel(self.lookup.name)
        except AttributeError as e:
            pass
        if plot_scale is not None:
            ax.set_yscale(plot_scale)
        if show_plot:
            plt.show()
        if return_fig:
            return fig

    def plot_boxplot(self, spectra=None, plot_scale=None, show_plot=False, colours=None,
                     figsize=(5, 5), return_fig=False, ax=None):
        """
        Boxplot of the expected and observed distributions of mutation scores.
        :param spectra: The mutational spectrum or list of mutational spectra to use.
        :param plot_scale:  Scale for the y-axis (mutation scores). Passed to matplotlib. 'log', 'symlog' etc.
        :param show_plot: If True, will call plt.show().
        :param colours: List of colours. First colour is for the observed data, the subsequent colours are
        for plotting the null distributions from each of the mutational spectra given.
        :param figsize: Size of the figure.
        :param return_fig: If True, will return the figure. Used for testing.
        :param ax: Matplotlib axis to plot on. If None, will create a new figure.
        :return: By default, None. If return_fig=True, will return the figure.
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.figure

        spectra = self._get_spectra(spectra)

        colours = self._get_plot_colours(colours, len(spectra)+1)
        data = [get_distribution_from_mutational_spectrum(self.null_scores,
                                                          self.null_mutations[spectrum.rate_column])
                for spectrum in spectra]
        data.append(self.observed_values)
        sns.boxplot(
            data=data, palette=self._get_seaborn_colour_palette(colours, len(spectra) + 1), ax=ax)
        self._set_xticks(ax, range(len(data)), ['Expected\n' + spectrum.name for spectrum in spectra] + ['Observed'])
        try:
            ax.set_ylabel(self.lookup.name)
        except AttributeError as e:
            pass
        if plot_scale is not None:
            ax.set_yscale(plot_scale)
        if show_plot:
            plt.show()
        if return_fig:
            return fig

    def plot_cdfs(self, spectra=None, plot_scale=None, show_plot=False, legend_args=None, colours=None,
                  figsize=(5, 5), return_fig=False, show_legend=True,
                  show_CI=False, CI_alpha=0.05, CI_num_samples=10000, ax=None):
        """
        CDF curves for the expected and observed distribution of mutation scores.
        :param spectra: The mutational spectrum or list of mutational spectra to use.
        :param plot_scale: Scale for the x-axis (mutation scores). Passed to matplotlib. 'log', 'symlog' etc.
        :param show_plot: If True, will call plt.show().
        :param legend_args: kwargs to pass to matplotlib for the legend appearance.
        :param colours: List of colours. First colour is for the observed data, the subsequent colours are
        for plotting the null distributions from each of the mutational spectra given.
        :param figsize: Size of the figure.
        :param return_fig: If True, will return the figure. Used for testing.
        :param show_legend: If True (default), show the legend.
        :param show_CI: Show confidence intervals around the CDFs. The CIs for the expected distributions are calculated
        by randomly drawing from the null score distributions, the CIs for the observed data are calculated by drawing
        with replacement from the observed data.
        :param CI_alpha: The alpha for the interval. The default is 0.05 (95% confidence interval).
        :param CI_num_samples: Number of random samples used to calculate the CIs. Default 10000.
        :param ax: Matplotlib axis to plot on. If None, will create a new figure.
        :return: By default, None. If return_fig=True, will return the figure.
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.figure
        spectra = self._get_spectra(spectra)

        colours = self._get_plot_colours(colours, len(spectra)+1)
        num_obs = len(self.observed_values)
        sorted_null_scores = sorted(self.null_scores)
        xvals = np.linspace(min(sorted_null_scores), max(sorted_null_scores), 10000)
        for spectrum, colour in zip(spectra, colours[1:]):
            cdf_func = get_cdf(self.null_scores, self.null_mutations[spectrum.rate_column])
            ax.plot(xvals, cdf_func(xvals), label='Expected ' + spectrum.name, c=colour)
            if show_CI:
                ci_low, ci_high = get_null_cdf_confint(self.null_scores, self.null_mutations[spectrum.rate_column],
                                                       num_obs, xvals, num_samples=CI_num_samples, alpha=CI_alpha)
                ax.fill_between(xvals, ci_low, ci_high, color=colour, alpha=0.1)

        sorted_obs = sorted(self.observed_values)
        ax.plot(sorted_obs, np.arange(1, len(sorted_obs) + 1) / len(sorted_obs), label='Observed', linewidth=3,
                 c=colours[0])
        if show_CI:
            ci_low, ci_high = bootstrap_cdf_confint_method(sorted_obs, xvals, num_samples=CI_num_samples,
                                                           alpha=CI_alpha)
            ax.fill_between(xvals, ci_low, ci_high, color=colours[0], alpha=0.1)

        try:
            ax.set_xlabel(self.lookup.name)
        except AttributeError as e:
            pass
        ax.set_ylabel('CDF')
        if show_legend:
            if legend_args is None:
                legend_args = {}
            ax.legend(**legend_args)
        if plot_scale is not None:
            ax.set_xscale(plot_scale)
        if show_plot:
            plt.show()
        if return_fig:
            return fig

    def plot_chi_sq_counts(self, spectra=None, show_plot=False, figsize=(15, 5), colours=None, return_fig=False,
                           show_legend=True, legend_args=None, chi_tests=None, show_CI=True):
        """

        :param spectra: The mutational spectrum or list of mutational spectra to use.
        :param show_plot: If True, will call plt.show()
        :param figsize: Size of the figure.
        :param colours: List of colours for plots. First colour is for the observed data, the subsequent colours are
        for plotting the null distributions from each of the mutational spectra given.
        :param return_fig: If True, will return the figure. Used for testing.
        :param show_legend: Will show the legend listing the names of the mutational spectra.
        :param legend_args: kwargs to pass to matplotlib for the legend appearance.
        :param chi_tests: ChiSquareTest of list of ChiSquareTest objects to use. Will create one subplot per test.
        If None, will use any ChiSquareTests already run for this Section.
        :param show_CI: Show 95% confidence intervals
        :return: By default, None. If return_fig=True, will return the figure.
        """
        spectra = self._get_spectra(spectra)

        colours = self._get_plot_colours(colours, len(spectra)+1)
        if chi_tests is None:
            # Use any chi-square tests from the project
            chi_tests = [t for t in self.statistics if isinstance(t, ChiSquareTest)]
        elif isinstance(chi_tests, str):
            # Names of single test given instead. Use only the chi-square test from the project that match this name
            chi_tests = [t for t in self.statistics if isinstance(t, ChiSquareTest) and t.name == chi_tests]
        if isinstance(chi_tests, ChiSquareTest):
            chi_tests = [chi_tests]
            self._run_statistical_tests(plot=False, statistics=chi_tests)  # The chi-square tests need to be run first
        elif all([isinstance(c, str) for c in chi_tests]):
            # Names given instead. Use only the chi-square tests from the project that match these names
            chi_tests = [t for t in self.statistics if isinstance(t, ChiSquareTest) and t.name in chi_tests]
        elif all([isinstance(c, ChiSquareTest) for c in chi_tests]):
            self._run_statistical_tests(plot=False, statistics=chi_tests)  # The chi-square tests need to be run first
        else:
            raise ValueError('chi_tests not recognised')

        if len(chi_tests) == 0:
            print('No chi square tests to plot')
        else:

            fig, axes = plt.subplots(nrows=len(chi_tests), ncols=len(spectra), squeeze=False, figsize=figsize)
            legend_items = []
            for i, chi_test in enumerate(chi_tests):
                chi_name = chi_test.name
                for j, spectrum in enumerate(spectra):
                    expected_counts = self.statistical_results["_".join([chi_name, spectrum.name, 'expected_counts'])]
                    observed_counts = self.statistical_results["_".join([chi_name, spectrum.name, 'observed_counts'])]
                    bins = self.statistical_results["_".join([chi_name, spectrum.name, 'bins'])]
                    x = np.arange(len(expected_counts)) + 0.1
                    w = 0.4
                    ax = axes[i, j]

                    if show_CI:
                        observed_CI_low = np.array(
                            self.statistical_results["_".join([chi_name, spectrum.name, 'observed_CI_low'])])
                        observed_CI_high = np.array(
                            self.statistical_results["_".join([chi_name, spectrum.name, 'observed_CI_high'])])
                        yerr_obs = [observed_counts - observed_CI_low, observed_CI_high - observed_counts]

                        expected_CI_low = np.array(
                            self.statistical_results["_".join([chi_name, spectrum.name, 'expected_CI_low'])])
                        expected_CI_high = np.array(
                            self.statistical_results["_".join([chi_name, spectrum.name, 'expected_CI_high'])])
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

    def plot_binned_counts(self, spectra=None, show_plot=False, figsize=(15, 5), colours=None, return_fig=False,
                           linewidth=0, hatches=None, show_legend=True, legend_args=None, bins=(-1, 0.5, 2),
                           show_CI=True, CI_num_samples=10000, CI_alpha=0.05):
        """
        Bar plot of observed and expected counts of mutations with scores in the given bins.
        :param spectra: The mutational spectrum or list of mutational spectra to use.
        :param show_plot: If True, will call plt.show()
        :param figsize: Size of the figure.
        :param colours: List of colours for plots. First colour is for the observed data, the subsequent colours are
        for plotting the null distributions from each of the mutational spectra given.
        :param return_fig: If True, will return the figure. Used for testing.
        :param linewidth: Linewidth for the bar plot.
        :param hatches: Hatches for the bar plot
        :param show_legend: Will show the legend listing the names of the mutational spectra.
        :param legend_args: kwargs to pass to matplotlib for the legend appearance.
        :param bins: Bins for the scores.
        :param show_CI: Show 95% confidence intervals
        :param CI_num_samples: Number of random samples used to calculate the CIs. Default 10000.
        :param CI_alpha:  The alpha for the CIs. The default is 0.05 (95% confidence intervals).
        :return: By default, None. If return_fig=True, will return the figure.
        """
        spectra = self._get_spectra(spectra)

        colours = self._get_plot_colours(colours, len(spectra)+1)
        if hatches is None:
            hatches = [None]*len(colours)


        chi_name = 'chi_for_plot'
        chi_test = ChiSquareTest(bins=bins, CI_num_samples=CI_num_samples, CI_alpha=CI_alpha, name=chi_name)
        # Run the test using each of the spectra
        statistical_results = {}
        for spectrum in spectra:
            statistical_results.update(chi_test(self, spectrum, plot=False))

        fig, ax = plt.subplots(figsize=figsize)
        legend_items = []

        w = 0.8 / (len(spectra) + 1)

        observed_counts = np.array(statistical_results["_".join([chi_name, spectra[0].name, 'observed_counts'])])

        if show_CI:
            observed_CI_low = np.array(
                statistical_results["_".join([chi_name, spectra[0].name, 'observed_CI_low'])])
            observed_CI_high = np.array(
                statistical_results["_".join([chi_name, spectra[0].name, 'observed_CI_high'])])
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
                statistical_results["_".join([chi_name, spectrum.name, 'expected_counts'])])


            if show_CI:
                expected_CI_low = np.array(
                    statistical_results["_".join([chi_name, spectrum.name, 'expected_CI_low'])])
                expected_CI_high = np.array(
                    statistical_results["_".join([chi_name, spectrum.name, 'expected_CI_high'])])
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

    def plot_binomial(self, spectra=None, show_plot=False, figsize=(2, 3), colours=None, return_fig=False,
                           show_legend=True, legend_args=None, binom_test=None, show_CI=True):
        """
        Bar plot of the expected and observed counts of mutations with scores above the threshold for a BinomTest
        :param spectra: The mutational spectrum or list of mutational spectra to use.
        :param show_plot: If True, will call plt.show()
        :param figsize: Size of the figure.
        :param colours: List of colours for plots. First colour is for the observed data, the subsequent colours are
        for plotting the null distributions from each of the mutational spectra given.
        :param return_fig: If True, will return the figure. Used for testing.
        :param show_legend: Will show the legend listing the names of the mutational spectra.
        :param legend_args: kwargs to pass to matplotlib for the legend appearance.
        :param binom_test: BinomTest or list of BinomTests to use. If None, will use the first BinomTest run on this
        Section (if any).
        :param show_CI: Show 95% confidence intervals
        :return: By default, None. If return_fig=True, will return the figure.
        """
        spectra = self._get_spectra(spectra)

        colours = self._get_plot_colours(colours, len(spectra)+1)
        if binom_test is None:
            # Use the first binomial test run on the Section
            binom_test = [t for t in self.statistics if isinstance(t, BinomTest)][0]
        elif isinstance(binom_test, str):
            # Names of single test given instead. Use only the BinomTest test that matches this name
            binom_test = [t for t in self.statistics if isinstance(t, BinomTest) and t.name == binom_test][0]
        elif isinstance(binom_test, BinomTest):
            self._run_statistical_tests(plot=False, statistics=[binom_test])  # The test needs to be run first
        else:
            raise ValueError('binom_test not recognised')

        if binom_test is None:
            print('No binomial tests to plot')
        else:
            fig, axes = plt.subplots(nrows=len(spectra), squeeze=False, figsize=figsize)
            axes = axes[:, 0]
            legend_items = []
            test_name = binom_test.name
            for j, spectrum in enumerate(spectra):
                expected_count = [self.statistical_results["_".join([test_name, spectrum.name, 'expected_count'])]]
                observed_count = [self.statistical_results["_".join([test_name, spectrum.name, 'observed_count'])]]
                x = np.array([0.1])
                w = 0.4
                ax = axes[j]

                if show_CI:
                    observed_CI_low = np.array(
                        self.statistical_results["_".join([test_name, spectrum.name, 'observed_CI_low'])])
                    observed_CI_high = np.array(
                        self.statistical_results["_".join([test_name, spectrum.name, 'observed_CI_high'])])
                    yerr_obs = [observed_count - observed_CI_low, observed_CI_high - observed_count]

                    expected_CI_low = np.array(
                        self.statistical_results["_".join([test_name, spectrum.name, 'expected_CI_low'])])
                    expected_CI_high = np.array(
                        self.statistical_results["_".join([test_name, spectrum.name, 'expected_CI_high'])])
                    yerr_exp = [expected_count - expected_CI_low, expected_CI_high - expected_count]
                else:
                    yerr_obs=None
                    yerr_exp = None


                l1 = ax.bar(x, expected_count, yerr=yerr_exp, label='Expected counts', width=w, color=colours[j+1])
                legend_items.append(l1)
                obs = ax.bar(x + w,
                        observed_count, yerr=yerr_obs, label='Observed counts', width=w, color=colours[0])
                ax.set_xticks([0.1, 0.5])
                ax.set_xticklabels(['Exp.', 'Obs.'])
                if j == 0:
                    ax.set_ylabel('Number of mutations')
                ax.set_title('Binomial test counts -\n' + spectrum.name)

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

    def plot_aa_abundance(self, spectra=None, sig_threshold=0.05, use_qval=True, show_plot=False, max_texts=10,
                          figsize=(5, 5), return_fig=False):
        """
        Scatter plot of expected and observed amino acid changes.
        :param spectra: The mutational spectrum or list of mutational spectra to use.
        :param sig_threshold: Significance threshold for labelling mutations.
        :param use_qval: Multiple test correct the p-values (True by default).
        :param show_plot: If True, will call plt.show()
        :param max_texts: Max number of labels on the plot. Will start from most significant cases.
        :param figsize: Size of the figure.
        :param return_fig: If True, will return the figure. Used for testing.
        :return: By default, None. If return_fig=True, will return the figure.
        """
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
                               show_residues=False, xlim=None, ylim=None, ax=None, yaxis_right=False,
                               show_expected=True):
        """
        Plot of mutation counts with a given distance from the residue in the protein structure. Requires a pdb_id and
        pdb_chain to be defined for the section.
        :param distance: Distance in Angstroms to use.
        :param normalise: If True, plot mutations per residue. If False, plot raw number of mutations within the
        distance threshold.
        :param include_chains: For structures including multiple chains from the same protein, can specify which chains
        to include here. The pdb_chain will be used to define the 'windows', but mutation counts will be based on all
        of the chains listed in include_chains.
        :param figsize: Size of the figure.
        :param spectra: The mutational spectrum or list of mutational spectra to use.
        :param colours: List of colours for plots. First colour is for the observed data, the subsequent colours are
        for plotting the null distributions from each of the mutational spectra given.
        :param show_legend: Will show the legend listing the names of the mutational spectra.
        :param legend_args: kwargs to pass to matplotlib for the legend appearance.
        :param show_arcs: Show arcs linking residues within the distance threshold of each other.
        :param arc_scale: Scale for the arcs. Likely to need adjusting.
        :param arc_alpha: Alpha for the arcs.
        :param min_arc_residue_gap: Will not show arcs between residues closer than this in the sequence if from the
        same chain.
        :param show_plot: If True, will call plt.show().
        :param colourmap: Colormap for plotting the arcs between residues in the same chain.
        :param colourmap_different_chain: Colormap for plotting the arcs between residues in different chains.
        :param return_fig: If True, will return the figure. Used for testing.
        :param show_residues: If True, will show the reference amino acids on the x-axis.
        :param xlim: Limits for the x-axis.
        :param ylim: Limits for the y-axis.
        :param ax: Matplotlib axis to plot on. If None, will create a new figure.
        :param yaxis_right: If True, will show the y-axis on the right of the plot.
        :param show_expected: Set to False to hide the expected sliding window.
        :return: By default, None. If return_fig=True, will return the figure.
        """
        if "MDAnalysis" not in sys.modules:
            raise ImportError("Must install MDAnalysis package to use this function")
        if self.pdb_id is None:
            raise ValueError("No pdb_id and pdb_chain defined for this Section")
        else:
            try:
                u = MDAnalysis.Universe(os.path.join(self.project.pdb_directory, self.pdb_id.lower() + '.pdb.gz'))
            except FileNotFoundError as e:
                try:
                    u = MDAnalysis.Universe(os.path.join(self.project.pdb_directory, self.pdb_id.lower() + '.pdb'))
                except FileNotFoundError as e:
                    print('Not plotting the 3D window. PDB file not found.')
                    return

            spectra = self._get_spectra(spectra)

            colours = self._get_plot_colours(colours, len(spectra)+1)

            if ax is None:
                fig, ax = plt.subplots(figsize=figsize)
                if yaxis_right:
                    ax.yaxis.tick_right()
            else:
                fig = ax.figure
                if yaxis_right:
                    ax = ax.twinx()

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

            arcs_same_chain = set()
            arcs_different_chain = set()

            expected_counts = {spectrum.name: np.zeros(len(seq_pdb_residues)) for spectrum in spectra}
            observed_counts = np.zeros(len(seq_pdb_residues))
            for i, res in enumerate(seq_pdb_residues):
                surrounding_res = get_3D_window(u, self.pdb_chain, res, distance)
                surrounding_res = [r for r in surrounding_res if r[0] in include_chains]
                residue_window = [r[1] for r in surrounding_res]
                if residue_window:
                    if normalise:
                        norm_fac = len(residue_window)
                    else:
                        norm_fac = 1

                    if show_arcs:
                        for r in surrounding_res:
                            tup = (min(res, r[1]), max((res, r[1])))
                            if r[0] == self.pdb_chain and abs(r[1] - res) >= min_arc_residue_gap and \
                                    tup not in arcs_same_chain:
                                arcs_same_chain.add(tup)
                            elif r[
                                0] != self.pdb_chain and tup not in arcs_different_chain and tup not in arcs_same_chain:
                                arcs_different_chain.add(tup)

                    window_mutations = self.observed_mutations[self.observed_mutations['residue'].isin(residue_window)]
                    obs = len(window_mutations)

                    null_window_mutations = self.null_mutations[self.null_mutations['residue'].isin(residue_window)]

                    for spectrum in spectra:
                        total_window_rate = null_window_mutations[spectrum.rate_column].sum()
                        normalised_window_rate = total_window_rate * normalising_factors[spectrum.name]
                        expected_counts[spectrum.name][i] = normalised_window_rate / norm_fac

                    observed_counts[i] = obs / norm_fac

            if arcs_same_chain:
                plot_arcs(arcs_same_chain, arc_scale, colourmap, colourmap_norm, ax, alpha=arc_alpha)
            if arcs_different_chain:
                plot_arcs(arcs_different_chain, arc_scale, colourmap_different_chain, colourmap_norm, ax,
                          alpha=arc_alpha)

            if show_expected:
                for spectrum, colour in zip(spectra, colours[1:]):
                    ax.plot(seq_pdb_residues, expected_counts[spectrum.name],
                            label=spectrum.name, c=colour)
            observed_counts = np.array(observed_counts)
            ax.plot(seq_pdb_residues, observed_counts, c=colours[0], label='Observed', linewidth=3)
            if xlim is not None:
                ax.set_xlim(xlim)
            else:
                ax.set_xlim([seq_pdb_residues[0] - 1, seq_pdb_residues[-1] + 1])
            if ylim is not None:
                ax.set_ylim(ylim)
            elif not show_arcs:
                ax.set_ylim(bottom=0)

            if show_arcs:
                # Remove yticks below zero
                ax.set_yticks([t for t in ax.get_yticks() if t >= 0])

            ax.set_xlabel('Residue')

            self._format_xticks(ax)

            if normalise:
                ax.set_ylabel('Mutations per codon')
            else:
                ax.ylabel('Mutations')

            if show_legend:
                if legend_args is None:
                    legend_args = {}
                ax.legend(**legend_args)

            if show_residues:
                self._show_residues(fig, ax)

            if show_plot:
                plt.show()
            if return_fig:
                return fig

    def plot_sliding_window_totalled_score(self, window_size=20, window_step=1,
                                          spectra=None, show_legend=True, figsize=(15, 5), legend_args=None,
                                          show_plot=False, colours=None, return_fig=False,
                                           show_residues=False, xlim=None, ylim=None, ax=None, yaxis_right=False,
                                           plot_kwargs_obs=None, plot_kwargs_exp=None):
        """
        A sliding window with the total score of mutations within the window.
        :param window_size: Size of the sliding window in residues.
        :param window_step: Size of the step between starts of adjacent windows in residues.
        :param spectra: The mutational spectrum or list of mutational spectra to use.
        :param show_legend: Will show the legend listing the names of the mutational spectra.
        :param figsize: Size of the figure.
        :param legend_args: kwargs to pass to matplotlib for the legend appearance.
        :param show_plot: If True, will call plt.show()
        :param colours: List of colours for plots. First colour is for the observed data, the subsequent colours are
        for plotting the null distributions from each of the mutational spectra given.
        :param return_fig: If True, will return the figure. Used for testing.
        :param show_residues: If True, will show the reference amino acids on the x-axis.
        :param xlim: Limits for the x-axis.
        :param ylim: Limits for the y-axis.
        :param ax: Matplotlib axis to plot on. If None, will create a new figure.
        :param yaxis_right: If True, will show the y-axis on the right of the plot.
        :param plot_kwargs_obs: Additional kwargs for the appearance of the observed line.
        :param plot_kwargs_exp: Additional kwargs for the appearance of the expected lines.
        :return: By default, None. If return_fig=True, will return the figure.
        """

        start, end = self.null_mutations['residue'].min(), self.null_mutations['residue'].max()

        starts = np.arange(start, end + 2 - window_size, window_step)
        if len(starts) == 0:
            print("Region too small compared to window size/step.")
            return

        ends = starts + window_size - 1

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
            if yaxis_right:
                ax.yaxis.tick_right()
        else:
            fig = ax.figure
            if yaxis_right:
                ax = ax.twinx()

        spectra = self._get_spectra(spectra)

        colours = self._get_plot_colours(colours, len(spectra)+1)

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
                                          ylabel='Total score per residue', plot_kwargs_obs=plot_kwargs_obs,
                                              plot_kwargs_exp=plot_kwargs_exp)

        if return_fig:
            return fig

    def plot_mutation_rate_scatter(self, spectra=None, metric_plot_scale=None, marker_size_from_count=True,
                                   base_marker_size=10, show_plot=False,
                                   unobserved_mutation_colour='#BBBBBB', missense_mutation_colour='C1',
                                   synonymous_mutation_colour='C2', nonsense_mutation_colour='C3',
                                   show_legend=True, figsize=(5, 5), legend_args=None, return_fig=False,
                                   xlim=None, unmutated_marker_size=1, mut_rate_plot_scale=None,
                                   mutations_to_annotate=None, annotation_column='aachange',
                                   annotation_offset=(0, 0), unobserved_alpha=1, observed_alpha=1,
                                   sections_for_colours=None, score_regions_for_colours=None,
                                   score_region_colours=None, colour_unmutated_by_scores=False,
                                   hotspots_in_foreground=False, observed_marker=None, unobserved_marker=None,
                                   show_observed_only=False, show_unobserved_only=False):
        """
        Plots the expected mutation rate of mutations against their metric score.
        :param spectra: The mutational spectrum or list of mutational spectra to use.
        :param metric_plot_scale: Scale for the y-axis (mutation scores). Passed to matplotlib. 'log', 'symlog'
        :param marker_size_from_count: If True (default), the size of the marker will be based on the frequency of
        that mutation in the data. If False, all markers will have the same size.
        :param base_marker_size: The size of a marker for a mutation occurring once in the data (or size for all
        mutation markers if marker_size_from_count=False).
        :param show_plot: If True, will call plt.show().
        :param unobserved_mutation_colour: Colour for mutations that are not seen in the data. Hide these mutations
        entirely by setting show_observed_only=True.
        :param missense_mutation_colour: Colour for missense mutations.
        :param synonymous_mutation_colour: Colour for synonymous mutations.
        :param nonsense_mutation_colour: Colour for nonsense mutations.
        :param show_legend: If True (default), show the legend.
        :param figsize: Size of the figure.
        :param legend_args: kwargs to pass to matplotlib for the legend appearance.
        :param return_fig: If True, will return the figure. Used for testing.
        :param xlim: Limits for the x-axis.
        :param unmutated_marker_size: Size of the markers for mutations not seen in the data.
        :param mut_rate_plot_scale: Scale for the x-axis (mutation scores). Passed to matplotlib. 'log', 'symlog'
        :param mutations_to_annotate: Dataframe of mutations to label on the plot. Should be a sub-dataframe of
        self.null_mutations.
        :param annotation_column: Column to use for the annotation.
        :param annotation_offset: Tuple. Offset for annotation.
        :param unobserved_alpha: Alpha for the markers for mutations not seen in the data.
        :param observed_alpha: Alpha for the markers of observed mutations.
        :param sections_for_colours: List of additional Section objects that can be used to colour mutations based on
        further scores.  To be used with score_regions_for_colours, and optionally score_region_colours.
        :param score_regions_for_colours: List of the regions to colour using the sections_for_colours. Each region is a
        tuple of lower and upper score bound. i-th region applies to the i-th section in sections_for_colours.
        :param score_region_colours: List of colours to use for sections_for_colours. i-th colour applies to the
        i-th section in sections_for_colours.
        :param colour_unmutated_by_scores: If True, the null mutations will also be coloured according to the
        section_for_colours and score_regions_for_colours arguments. If False, null mutations will be plotted with the
        unobserved_mutation_colour.
        :param hotspots_in_foreground: If True, will plot the most frequent mutations (the largest markers)
        in the foreground. If False, will plot the largest markers in the background so other mutations they overlap
        with are visible.
        :param observed_marker: Shape of the marker to use for observed mutations.
        :param unobserved_marker: Shape of the marker to use for unobserved mutations.
        :param show_observed_only: If True, will not show the unobserved mutations.
        :param show_unobserved_only: If True, will only show the unobserved mutations.
        :return: By default, None. If return_fig=True, will return the figure.
        """
        spectra = self._get_spectra(spectra)

        if mutations_to_annotate is not None:
            mutations_to_annotate = mutations_to_annotate.drop_duplicates(subset=['ds_mut_id'])

        fig, axes = plt.subplots(nrows=1, ncols=len(spectra), squeeze=False, figsize=figsize)

        if len(self.observed_mutations) == 0:
            show_unobserved_only = True

        if not show_unobserved_only:
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
                if colour_unmutated_by_scores and not show_observed_only:
                    unplotted_null_muts = colour_mutations_by_scores(null_to_plot, None, spectrum.rate_column, 'score',
                                               sections_for_colours,
                                               score_regions_for_colours, score_region_colours,
                                               marker_size_from_count=False,
                                               base_marker_size=unmutated_marker_size, alpha=unobserved_alpha,
                                               hotspots_in_foreground=False, use_null=True, marker=unobserved_marker,
                                                                     zorder=1)
                else:
                    unplotted_null_muts = null_to_plot
                ax.scatter(unplotted_null_muts[spectrum.rate_column], unplotted_null_muts['score'],
                       s=unmutated_marker_size, alpha=unobserved_alpha,
                       c=unobserved_mutation_colour, label=None, marker=unobserved_marker, zorder=0)
            if not show_unobserved_only:
                if sections_for_colours is not None:
                    unplotted_obs_muts = colour_mutations_by_scores(self.observed_mutations, mut_counts,
                                                                    spectrum.rate_column, 'score',
                                               sections_for_colours,
                                               score_regions_for_colours, score_region_colours, marker_size_from_count,
                                               base_marker_size, observed_alpha, hotspots_in_foreground, ax=ax,
                                               marker=observed_marker, zorder=1)
                else:
                    unplotted_obs_muts = self.observed_mutations

                for effect, col in zip(['synonymous', 'missense', 'nonsense'], [synonymous_mutation_colour,
                                                                    missense_mutation_colour,
                                                                    nonsense_mutation_colour]):
                    muts = unplotted_obs_muts[unplotted_obs_muts['effect'] == effect]
                    if len(muts) > 0:
                        _plot_single_scatter_category(muts, mut_counts, spectrum.rate_column, 'score',
                                                      marker_size_from_count,
                                                      base_marker_size, col, effect,
                                                      observed_alpha, hotspots_in_foreground, ax=ax,
                                                      marker=observed_marker, zorder=0)

            ax.set_xlabel('Mutation rate')
            try:
                ax.set_ylabel(self.lookup.name)
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

    def plot_bar_observations(self, binning_regions, groupby_col='residue', figsize=(15, 5),
                              facecolour='k', linewidth=0, edgecolour='k',
                              show_legend=False, legend_args=None, show_plot=False, return_fig=False,
                              show_residues=False, xlim=None, ylim=None,
                              normalise_by_region_size=True, ax=None):
        """
        Bar plot of the mutation count (or mutation count per residue) in bins of residues across the Section.
        :param binning_regions: List of bins. Passed to pandas.cut.
        :param groupby_col: Column for binning the mutations. Default is 'residue'.
        :param figsize: Size of the figure.
        :param facecolour: Colour or list of colours for the bars.
        :param linewidth: Linewidth for bar plot
        :param edgecolour: Edgecolour or list of edgecolours for the bars.
        :param show_legend: If True (default), show the legend.
        :param legend_args: kwargs to pass to matplotlib for the legend appearance.
        :param show_plot: If True, will call plt.show().
        :param return_fig: If True, will return the figure. Used for testing.
        :param show_residues: If True, will show the reference amino acids on the x-axis.
        Only used if groupby_col='residue'.
        :param xlim: Limits for the x-axis.
        :param ylim: Limits for the y-axis.
        :param normalise_by_region_size: If True, each bar will show the mutation count per residue in that bin.
        If False, will show the raw counts in that bin.
        :param ax: Matplotlib axis to plot on. If None, will create a new figure.
        :return: By default, None. If return_fig=True, will return the figure.
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.figure

        x, bins = pd.cut(self.observed_mutations[groupby_col], binning_regions, retbins=True)
        counts = self.observed_mutations.groupby(x).count()['chr']
        widths = np.diff(bins)
        if normalise_by_region_size:
            counts = counts / widths
        ax.bar(bins[:-1] + widths / 2, counts, color=facecolour, width=widths,
                linewidth=linewidth, edgecolor=edgecolour)

        if xlim is not None:
            ax.set_xlim(xlim)
        if ylim is not None:
            ax.set_ylim(ylim)
        else:
            ax.set_ylim(bottom=0)

        ax.set_ylabel('Mutations per {}'.format(groupby_col))
        ax.set_xlabel(groupby_col)

        self._format_xticks(ax)

        if show_legend:
            if legend_args is None:
                legend_args = {}
            ax.legend(**legend_args)

        if groupby_col == 'residue' and show_residues:
            self._show_residues(fig, ax)

        if show_plot:
            plt.show()

        if return_fig:
            return fig

    def plot_lollipop(self, groupby_col='residue', figsize=(15, 5), linefmt='k-', basefmt='k-', markerfmt='ko',
                      show_plot=False, return_fig=False, show_residues=False, xlim=None, ylim=None, ax=None,
                      yaxis_right=False, label_peaks_min_size=None, label_peaks_column='aachange',
                      label_peaks_fmt_kwargs=None):
        """
        Lollipop plot (Matplotlib stem plot) of mutation counts.
        :param groupby_col: Column for binning the mutations. Default is 'residue'.
        :param figsize: Size of the figure.
        :param linefmt: linefmt for matplotlib stem plot.
        :param basefmt: basefmt for matplotlib stem plot.
        :param markerfmt: markerfmt for matplotlib stem plot.
        :param show_plot: If True, will call plt.show().
        :param return_fig: If True, will return the figure. Used for testing.
        :param show_residues: If True, will show the reference amino acids on the x-axis.
        :param xlim: Limits for the x-axis.
        :param ylim: Limits for the y-axis.
        :param ax: Matplotlib axis to plot on. If None, will create a new figure.
        :param yaxis_right: If True, will show the y-axis on the right of the plot.
        :param label_peaks_min_size: Will add labels to peaks over this size.
        :param label_peaks_column: The column to use for the peak labels. If the peak contains mutations with different
        values in this column, they will all be displayed with a mutation count for each label.
        :param label_peaks_fmt_kwargs: kwargs for the matplotlib annotate for the peak labels.
        :return: By default, None. If return_fig=True, will return the figure.
        """

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
            if yaxis_right:
                ax.yaxis.tick_right()
        else:
            fig = ax.figure
            if yaxis_right:
                ax = ax.twinx()

        counts = self.observed_mutations.groupby(groupby_col).count()['chr']  # Pick arbitrary column we know is there.
        markerline, stemlines, baseline = ax.stem(counts.index, counts,
                                                   linefmt=linefmt, basefmt=basefmt, markerfmt=markerfmt,
                                                   use_line_collection=True)

        if xlim is None:
            xlim = [self.null_mutations['residue'].min() - 1, self.null_mutations['residue'].max() + 1]
        ax.set_xlim(xlim)
        if ylim is not None:
            ax.set_ylim(ylim)
        else:
            ax.set_ylim(bottom=0)

        if label_peaks_min_size is not None:
            if label_peaks_fmt_kwargs is None:  #  Use some default options
                label_peaks_fmt_kwargs = dict(ha='center')
            peaks = counts[counts >= label_peaks_min_size].keys()
            all_peak_obs = self.observed_mutations[self.observed_mutations[groupby_col].isin(peaks)]
            for peak, peak_muts in all_peak_obs.groupby(groupby_col):
                labels = peak_muts[label_peaks_column].value_counts()
                txt = "\n".join(["{} ({})".format(k, v) for k, v in labels.iteritems()])
                ax.annotate(txt, (peak, counts[peak] + 0.5), **label_peaks_fmt_kwargs)

            # If the ylim has not been set manually, increase it here to include all the labels.
            #  Just a rough thing to work most of the time. Will not work in all cases.
            if ylim is None:
                ax.set_ylim(top=ax.get_ylim()[1] * 1.2)

        ax.set_ylabel('Mutations per {}'.format(groupby_col))
        ax.set_xlabel(groupby_col)

        self._format_xticks(ax)

        if groupby_col=='residue' and show_residues:
            self._show_residues(fig, ax)

        if show_plot:
            plt.show()

        if return_fig:
            return fig

    def plot_expected_mutation_rates_for_residues_bar(self, residues, figsize=(5, 5), spectrum=None, show_plot=False,
                                                      return_fig=False, xlim=None, ylim=None, ax=None, yaxis_right=False,
                                                      orientation='horizontal'):
        """
        Bar plot of expected mutation rates for amino acid changes on a given set of residues.
        :param residues: List of residue numbers.
        :param figsize: Size of the figure.
        :param spectrum: Mutational spectrum to use.
        :param show_plot: If True, will call plt.show().
        :param return_fig: If True, will return the figure. Used for testing.
        :param xlim:  Limits for the x-axis.
        :param ylim:  Limits for the y-axis.
        :param ax: Matplotlib axis to plot on. If None, will create a new figure.
        :param yaxis_right: If True, will show the y-axis on the right of the plot.
        :param orientation: 'horizontal' or 'vertical'
        :return: By default, None. If return_fig=True, will return the figure.
        """
        if isinstance(residues, int):
            residues = [residues]
        if spectrum is None:
            spectrum = self.project.spectra[0]

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
            if yaxis_right:
                ax.yaxis.tick_right()
        else:
            fig = ax.figure
            if yaxis_right:
                ax = ax.twinx()

        muts = self.null_mutations[self.null_mutations['residue'].isin(residues)].groupby('aachange').agg(sum).sort_values(
            spectrum.rate_column, ascending=False)

        if orientation.lower() in ['horizontal', 'h']:
            b_f = ax.barh
            ticks_f = ax.set_yticks
            ticklabel_f = ax.set_yticklabels
            rotation=0
            axis_label_f = ax.set_xlabel
            rang = range(len(muts), 0, -1)
        elif orientation.lower() in ['vertical', 'v']:
            b_f = ax.bar
            ticks_f = ax.set_xticks
            ticklabel_f = ax.set_xticklabels
            axis_label_f = ax.set_ylabel
            rotation = 90
            rang = range(len(muts))
        else:
            raise TypeError("orientation must be horizontal/h or vertical/v")

        b_f(rang, muts[spectrum.rate_column])
        ticks_f(rang)
        ticklabel_f(muts.index, rotation=rotation)
        axis_label_f('Expected mutation rate')

        if xlim is not None:
            ax.set_xlim(xlim)
        if ylim is not None:
            ax.set_ylim(ylim)

        if show_plot:
            plt.show()

        if return_fig:
            return fig

    def plot_expected_mutation_rates_vs_observed_for_residues(self, residues, figsize=(5, 5), spectrum=None,
                                                              show_plot=False, return_fig=False, xlim=None, ylim=None,
                                                              ax=None, yaxis_right=False, scatter_plot_kwargs=None,
                                                              mutations_to_annotate=None,
                                                              annotation_mut_rate_threshold=None,
                                                              annotation_obs_count_threshold=None,
                                                              annotation_offset=(0, 0)):
        """
        Scatter plot of expected mutation rates vs observed mutation counts for amino acid changes on a
        given set of residues.
        :param residues: List of residue numbers.
        :param figsize: Size of the figure.
        :param spectrum: Mutational spectrum to use.
        :param show_plot: If True, will call plt.show().
        :param return_fig: If True, will return the figure. Used for testing.
        :param xlim:  Limits for the x-axis.
        :param ylim:  Limits for the y-axis.
        :param ax: Matplotlib axis to plot on. If None, will create a new figure.
        :param yaxis_right:  If True, will show the y-axis on the right of the plot.
        :param scatter_plot_kwargs: kwargs for matplotlib scatter plot.
        :param mutations_to_annotate: Dataframe of mutations (a subset of self.null_mutations) or a list of amino acid
        changes (e.g. ["M1L", "S241F", "G245S"]) to annotate.
        :param annotation_mut_rate_threshold: Will label mutations with a mutation rate higher than this.
        :param annotation_obs_count_threshold: Will label mutations with an observed count higher than this.
        :param annotation_offset: Tuple. Offset for the annotations.
        :return: By default, None. If return_fig=True, will return the figure.
        """
        if isinstance(residues, int):
            residues = [residues]
        if spectrum is None:
            spectrum = self.project.spectra[0]

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
            if yaxis_right:
                ax.yaxis.tick_right()
        else:
            fig = ax.figure
            if yaxis_right:
                ax = ax.twinx()

        muts = self.null_mutations[self.null_mutations['residue'].isin(residues)].groupby('aachange').agg(sum)

        # Use the chr column (arbitrary choice) to count the mutations
        counts = self.observed_mutations[self.observed_mutations['residue'].isin(residues)].groupby(
            'aachange').agg('count')['chr']

        m = pd.merge(muts, counts, how='left', left_index=True, right_index=True)
        m['chr'] = m['chr'].fillna(0)

        if scatter_plot_kwargs is None:
            scatter_plot_kwargs = {}
        ax.scatter(m[spectrum.rate_column], m['chr'], **scatter_plot_kwargs)

        if mutations_to_annotate is not None:
            if not isinstance(mutations_to_annotate, pd.DataFrame):
                mutations_to_annotate = m.loc[mutations_to_annotate].reset_index()
            annotate_mutations_on_plot(ax, mutations_to_annotate, 'aachange', xcol=spectrum.rate_column,
                                       ycol='chr',
                                       annotation_offset=annotation_offset)
        if annotation_mut_rate_threshold is not None:
            mutations_to_annotate = m[m[spectrum.rate_column] >= annotation_mut_rate_threshold].reset_index()
            annotate_mutations_on_plot(ax, mutations_to_annotate, 'aachange', xcol=spectrum.rate_column,
                                       ycol='chr',
                                       annotation_offset=annotation_offset)
        if annotation_obs_count_threshold is not None:
            mutations_to_annotate = m[m['chr'] >= annotation_obs_count_threshold].reset_index()
            annotate_mutations_on_plot(ax, mutations_to_annotate, 'aachange', xcol=spectrum.rate_column,
                                       ycol='chr',
                                       annotation_offset=annotation_offset)

        ax.set_xlabel('Expected mutation rate')
        ax.set_ylabel('Observed mutation count')

        if xlim is not None:
            ax.set_xlim(xlim)
        if ylim is not None:
            ax.set_ylim(ylim)

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
        # Not the most efficient for cases that use a full transcript.
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

    def get_observed_mutations_in_region(self, start=None, end=None, position_col='residue'):
        """
        Return the observed mutations sorted by position and filtered with a start and end point
        :param start:
        :param end:
        :param position_col:
        :return:
        """
        if start is None and end is None:
            return self.observed_mutations.sort_values(position_col)
        elif start is None:
            start = -np.inf
        elif end is None:
            end = np.inf

        obs = self.observed_mutations
        return obs[(obs[position_col] >= start) & (obs[position_col] <= end)].sort_values(position_col)


def plot_chi_bins_y(bins, linestyle='dashed', linewidth=1, alpha=0.5, zorder=0, colour=None, ax=None):
    # For overlaying on other plots with the metric on the y-axis
    if ax is None:
        ax = plt.gca()
    xlim = ax.get_xlim()
    ax.hlines(bins, xlim[0], xlim[1], linestyles=linestyle, linewidth=linewidth, alpha=alpha, zorder=zorder,
               color=colour)
    ax.set_xlim(xlim)


def plot_chi_bins_x(bins, linestyle='dashed', linewidth=1, alpha=0.5, zorder=0, colour=None, ax=None):
    # For overlaying on other plots with the metric on the x-axis
    if ax is None:
        ax = plt.gca()
    ylim = ax.get_ylim()
    ax.vlines(bins, ylim[0], ylim[1], linestyles=linestyle, linewidth=linewidth, alpha=alpha, zorder=zorder,
               color=colour)
    ax.set_ylim(ylim)


def get_3D_window(u, chain, residue, distance):
    # Get all atoms with the distance of the residue
    g = u.select_atoms("around {} segid {} and resnum {}".format(distance, chain, residue))
    # Remove any non-protein atoms, e.g. from water
    g = g.select_atoms('protein')

    residues = {(chain, residue)}  # Must add the residue itself
    for a in g:
        residues.add((a.segid, a.resid))
    return residues


def get_arcs(starts, ends, scale):
    """
    Get the x and y values for plotting arc from the start points to the end points.
    These are plotted below the x-axis.
    The arcs are half-ellipses
    :param starts: Numpy array of numbers or a single number. In practice will be integers.
    :param ends: Numpy array of numbers or a single number. In practice will be integers. Should be larger than the
    start values.
    :param scale: The height of each arc will be scale * arc length / 2
    :return: List of 2D arrays containing the coordinates of each arc.
    It is the correct format to draw the arcs using matplotlib.collections.LineCollection
    """
    # Convert the starts and ends to arrays if required.
    if isinstance(starts, (int, np.number, float)):
        starts = np.array([starts])
    elif isinstance(starts, (list, tuple)):
        starts = np.array(starts)

    if isinstance(ends, (int, np.number, float)):
        ends = np.array([ends])
    elif isinstance(ends, (list, tuple)):
        ends = np.array(ends)

    # Use 100 points for each arc.
    x = np.linspace(starts, ends, 100)
    half_lengths = (ends - starts) / 2
    height = scale * half_lengths

    # Draw the arcs below the x-axis, so have a minus sign
    y = -height / half_lengths * np.sqrt(half_lengths ** 2 - (x - starts - half_lengths) ** 2)

    # Stack the x and y values
    s = np.stack([x.T, y.T])
    # Convert to the format required for matplotlib.collections.LineCollection
    arcs = [s[:, i, :].T for i in range(s.shape[1])]
    return arcs


def plot_arcs(arcs, scale, colourmap, colourmap_norm, ax, alpha=0.5):
    """
    Use matplotlib.collections.LineCollection to plot a set of arcs under the x-axis.
    :param arcs: List/set of tuples with (arc_start, arc_end)
    :param scale: The height of each arc will be scale * arc length / 2
    :param colourmap: Matplotlib colormap or a single colour.
    :param colourmap_norm: A normalisation for the colourmap colours.
    :param ax: Axes to plot on.
    :param alpha: Alpha of the arcs.
    :return:
    """
    # Get the coordinates for the arcs
    starts, ends = list(zip(*arcs))
    arcs = get_arcs(starts, ends, scale)

    # Plot as a LineCollection
    col = collections.LineCollection(arcs)
    ax.add_collection(col, autolim=True)

    # Set the colours and the alpha of the arcs. 
    if isinstance(colourmap, Colormap):
        colors = colourmap(colourmap_norm(starts))
    else:
        colors = colourmap  # Assume this is a str, or rgb tuple rather than a colour map. Do not scale.
    col.set_color(colors)
    col.set_alpha(alpha)
