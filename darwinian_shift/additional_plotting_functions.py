import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle


def _plot_single_scatter_category(muts, mut_counts, xcol, ycol,
                                  marker_size_from_count, base_marker_size, colour, label, alpha,
                                  hotspots_in_foreground, marker=None, ax=None, zorder=None):
    """
    General function for plotting two columns of a mutation dataframe as a scatter plot. Called from a few of the
    scatter plots.

    :param muts: Dataframe of mutations.
    :param mut_counts: The counts associated with each mutation, will usually be the output of
    pandas.Series.value_counts() run on the column that defines the grouping of the mutations. Used for the marker sizes.
    :param xcol: The dataframe column for the x-values of the scatter plot.
    :param ycol: The dataframe column for the y-values of the scatter plot.
    :param marker_size_from_count: If True, the size of each marker will be proportional to the mut_counts values.
    :param base_marker_size: The size of all markers if marker_size_from_count=False or mut_counts=None. Otherwise, the
    marker size is the base_marker_size * the mut_count.
    :param colour: Color for the matplotlib scatter function.
    :param label: Label for matplotlib. Will appear on the legend.
    :param alpha: Alpha for the scatter.
    :param hotspots_in_foreground: If True, the most common mutations (with larger markers) will appear in the
    foreground - this can hide less frequent mutations behind the larger markers. If false, the smaller markers are in
    the foreground.
    :param marker: The marker shape.
    :param ax: Axis to plot on.
    :param zorder: Zorder for matplotlib.
    :return: None.
    """
    if ax is None:
        ax = plt.gca()

    dedup_muts = muts.drop_duplicates(subset='ds_mut_id').copy()  # copying prevents pandas SettingwithCopyWarning
    if mut_counts is not None:
        dedup_muts.loc[:, 'count'] = mut_counts.loc[dedup_muts['ds_mut_id']].values
        if hotspots_in_foreground:  # Will put the most common mutations at the front, but can hide less frequent mutations
            ascending=True
        else:
            ascending=False
        dedup_muts = dedup_muts.sort_values('count', ascending=ascending)

    x = dedup_muts[xcol]
    y = dedup_muts[ycol]
    if marker_size_from_count and mut_counts is not None:
        counts = dedup_muts['count']
        s = base_marker_size * counts
        linewidth = [0 if size < 20 else 0.2 for size in s]  # Only use outlines for the larger markers.
    else:
        s = base_marker_size
        linewidth = 0
    ax.scatter(x, y, s=s, c=colour, edgecolor='k', linewidth=linewidth, label=label, alpha=alpha, marker=marker,
               zorder=zorder)


def colour_mutations_by_scores(mutation_df, mut_counts, xcol, ycol, sections_for_colours, score_regions_for_colours,
                               score_region_colours, marker_size_from_count, base_marker_size, alpha,
                               hotspots_in_foreground, ax=None, use_null=False, marker=None, zorder=None):
    """
    Plots a series of scatter plots in various colours. Uses the sections_for_colours to define the mutations in
    each colour.
    For example, could plot all mutations with a ∆∆G value > 2 kcal/mol in red and mutations on an interface in blue.

    :param mutation_df: Dataframe of mutations to plot. Only those mutations which overlap with the sections_for_colours
    and score_regions_for_colours will be plotted.
    :param mut_counts: pd.Series.value_counts output for the mutation counts. Used for the marker size.
    :param xcol: The dataframe column for the x-values of the scatter plot.
    :param ycol: The dataframe column for the y-values of the scatter plot.
    :param sections_for_colours: List of Section objects with mutations already scored.
    :param score_regions_for_colours: List of tuples of the lower and upper bounds for the coloured region associated
    with each Section in sections_for_colours.
    :param score_region_colours: List of colours for each region.
    :param marker_size_from_count: Boolean. If True, the size of each marker will be proportional to the
    mut_counts values.
    :param base_marker_size: Float. The size of all markers if marker_size_from_count=False or mut_counts=None.
    Otherwise, the marker size is the base_marker_size * the mut_count.
    :param alpha: Alpha for the scatter.
    :param hotspots_in_foreground: If True, the most common mutations (with larger markers) will appear in the
    foreground - this can hide less frequent mutations behind the larger markers. If false, the smaller markers are in
    the foreground.
    :param ax: Axis to plot on.
    :param use_null: Will use the null_mutations to score the regions. Means every scored mutation, observed or not, can
    be shown (what will actually be shown will depend on the contents of the mutation_df).
    :param marker: The marker shape.
    :param zorder: Zorder for matplotlib.
    :return: pandas DataFrame including all mutations that were not plotted.
    """
    mut_df_copy = mutation_df.copy()  # Make sure not to edit the original dataframe
    if score_region_colours is None:
        score_region_colours = plt.rcParams['axes.prop_cycle'].by_key()['color'][4:]  # Default colour cycle

    all_included_mutations = set()

    for i, (sec, region, colour) in enumerate(zip(sections_for_colours, score_regions_for_colours,
                                                  score_region_colours)):
        if use_null:
            sec_muts = sec.null_mutations
        else:
            sec_muts = sec.observed_mutations.drop_duplicates(subset=["ds_mut_id"])
        mut_df_copy = pd.merge(mut_df_copy, sec_muts,
                               on=['pos', 'ref', 'mut', 'effect', 'aachange', 'ds_mut_id'],
                               suffixes=('', '_{}'.format(i + 3)), how='left')
        muts = mut_df_copy[(mut_df_copy['score_{}'.format(i + 3)] >= region[0]) &
                           (mut_df_copy['score_{}'.format(i + 3)] <= region[1])]
        if len(muts) > 0:
            _plot_single_scatter_category(muts, mut_counts, xcol, ycol,
                                          marker_size_from_count, base_marker_size, colour,
                                          sec.lookup.name, alpha, hotspots_in_foreground, ax=ax, marker=marker,
                                          zorder=zorder)
            all_included_mutations.update(muts['ds_mut_id'])

    return mutation_df[~mutation_df['ds_mut_id'].isin(all_included_mutations)]

def plot_scatter_two_scores(section1, section2, sections_for_colours=None, score_regions_for_colours=None,
                            score_region_colours=None, colour_unmutated_by_scores=False,
                            mut_lists_to_colour=None, mut_list_colours=None, mut_list_columns='aachange',
                            mut_list_labels=None,
                            plot_xscale=None, plot_yscale=None, marker_size_from_count=True,
                            base_marker_size=10, show_plot=False,
                            unobserved_mutation_colour='#BBBBBB', missense_mutation_colour='C1',
                            synonymous_mutation_colour='C2', nonsense_mutation_colour='C3',
                            show_legend=True, figsize=(10, 10), legend_args=None, return_fig=False,
                            xlim=None, ylim=None, unmutated_marker_size=1, xlabel=None, ylabel=None,
                            return_dataframes=False, annotate_mutations=False, annotate_xregion=None,
                            annotate_yregion=None, annotate_min_count=1, annotate_column='aachange',
                            annotation_offset=(0, 0), unobserved_alpha=1, observed_alpha=1,
                            hotspots_in_foreground=False, observed_marker=None, unobserved_marker=None,
                            show_observed_only=False, show_null_only=False, ax=None):
    """
    Plot two scores applied to the same mutations in a scatter plot.
    Only mutations that are scored using both methods can be shown. The mutations are matched based on
    position, reference nucleotide and mutant nucleotide.

    :param section1: Section object run with first scoring method.
    :param section2: Section object run with second scoring method.
    :param sections_for_colours: List of additional Section objects that can be used to colour mutations based on
    further scores.  To be used with score_regions_for_colours, and optionally score_region_colours.
    :param score_regions_for_colours: List of the regions to colour using the sections_for_colours. Each region is a
    tuple of lower and upper score bound. i-th region applies to the i-th section in sections_for_colours.
    :param score_region_colours: List of colours to use for sections_for_colours. i-th colour applies to the
    i-th section in sections_for_colours.
    :param colour_unmutated_by_scores: If True, the null mutations will also be coloured according to the
    section_for_colours and score_regions_for_colours arguments. If False, null mutations will be plotted with the
    unobserved_mutation_colour.
    :param mut_lists_to_colour: List of lists mutations to be coloured according to the mut_list_colours.
    To be used with mut_list_colours, and optionally mut_list_columns and mut_list_labels. The lists much match
    against one of the columns in the Section.observed_mutations. By default this is 'aachange'. For example,
    `mut_lists_to_colour=[['A100C', 'V101P'], ['G102R']], mut_list_colours=['r', 'b']`
    will plot the mutations 'A100C', 'V101P' in red and the mutation 'G102R' in blue.
    :param mut_list_colours: To be used with mut_lists_to_colour. See above.
    :param mut_list_columns: String or list of strings. The columns to use to define the mut_lists_to_colour. Can
    be a different column for each mutation list, or a single column to use for all given mutation lists.
    :param mut_list_labels: The labels to appear in the legend for the mut_lists_to_colour.
    :param plot_xscale: Scale for x-axis of the plot. Passed to matplotlib. 'log', 'symlog' etc.
    :param plot_yscale: Scale for y-axis of the plot. Passed to matplotlib. 'log', 'symlog' etc.
    :param marker_size_from_count: If True (default), the size of the marker will be scaled by the observed
    frequency of the mutation. If False, all markers of observed mutation will be the base_marker_size.
    :param base_marker_size: If marker_size_from_count=True, this is the marker size for a mutation observed once.
    If marker_size_from_count=False, this is the marker size of all observed mutations.
    :param show_plot: Will call plt.show() at the end of the function.
    :param unobserved_mutation_colour: Colour for plotting mutations that do not appear in the data.
    :param missense_mutation_colour: Colour for plotting observed missense mutations.
    :param synonymous_mutation_colour: Colour for plotting observed synonymous mutations.
    :param nonsense_mutation_colour: Colour for plotting observed nonsense mutations.
    :param show_legend: Show the plot legend.
    :param figsize: Tuple, figure size.
    :param legend_args: Dictionary of keyword arguments to pass to matplotlib for the legend.
    :param return_fig: If True, the function will return the figure.
    :param xlim: Tuple, limits of the x-axis.
    :param ylim: Tuple, limits of the y-axis.
    :param unmutated_marker_size: Marker size for mutations not in the observed data.
    :param xlabel: Label for the x-axis.
    :param ylabel: Label for the y-axis.
    :param return_dataframes: Will return the dataframes of mutations (unobserved and observed).
    :param annotate_mutations: If True, will label mutations in the region specified by annotate_xregion and
    annotate_yregion
    :param annotate_xregion: Tuple, will label mutations between these x-values.
    :param annotate_yregion: Tuple, will label mutations between these y-values.
    :param annotate_min_count: Minimum observed frequency of a mutation for it to be annotated.
    :param annotate_column: The dataframe column with which to annotate the mutations.
    :param annotation_offset: Offset for the annotation text.
    :param unobserved_alpha: Alpha for the unobserved mutations.
    :param observed_alpha: Alpha for the observed mutations.
    :param hotspots_in_foreground: If True, will plot the most frequent mutations (the largest markers)
    in the foreground. If False, will plot the largest markers in the background so other mutations they overlap with
    are visible.
    :param observed_marker: Shape of the marker to use for observed mutations.
    :param unobserved_marker: Shape of the marker to use for unobserved mutations.
    :param show_observed_only: Will only show mutations that appear in the data.
    :param show_null_only: Will only show mutations that do not appear in the data.
    :param ax: Matplotlib axis to plot on.
    :return:
    """

    # Can only plot mutations that have a score in both sections.
    common_mut_ids = set(section1.null_mutations['ds_mut_id']).intersection(section2.null_mutations['ds_mut_id'])
    s1_null = section1.null_mutations[section1.null_mutations['ds_mut_id'].isin(common_mut_ids)].copy()
    s2_null = section2.null_mutations[section2.null_mutations['ds_mut_id'].isin(common_mut_ids)].copy()
    s1_obs = section1.observed_mutations[section1.observed_mutations['ds_mut_id'].isin(common_mut_ids)].copy()
    s2_obs = section2.observed_mutations[section2.observed_mutations['ds_mut_id'].isin(common_mut_ids)].copy()

    merged_null = pd.merge(s1_null, s2_null, on=['pos', 'ref', 'mut', 'effect',
                                                                                 'aachange', 'ds_mut_id'],
                           suffixes=('_1', '_2'))

    # Need to deduplicate mutations before merging to prevent additional copies of recurrent mutations being created
    obs2 = s2_obs.drop_duplicates(subset='ds_mut_id')
    merged_obs = pd.merge(s1_obs, obs2,
                          on=['pos', 'ref', 'mut', 'effect', 'aachange', 'ds_mut_id'], suffixes=('_1', '_2'))

    if return_dataframes:
        full_null = merged_null.copy()
        full_obs = merged_obs.copy()

    mut_counts = merged_obs['ds_mut_id'].value_counts()

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    if not show_observed_only:
        if len(merged_obs) > 0:
            null_to_plot = merged_null[~merged_null['ds_mut_id'].isin(merged_obs['ds_mut_id'])].copy()
        else:
            null_to_plot = merged_null

        if sections_for_colours is not None:
            if colour_unmutated_by_scores and not show_observed_only:
                null_to_plot['score'] = np.nan
                null_to_plot = colour_mutations_by_scores(null_to_plot, None, 'score_1', 'score_2', sections_for_colours,
                                           score_regions_for_colours, score_region_colours,
                                           marker_size_from_count=False,
                                           base_marker_size=unmutated_marker_size, alpha=unobserved_alpha,
                                           hotspots_in_foreground=False, use_null=True, marker=unobserved_marker,
                                           zorder=1)

        plt.scatter(null_to_plot['score_1'], null_to_plot['score_2'], s=unmutated_marker_size,
                c=unobserved_mutation_colour, alpha=unobserved_alpha, marker=unobserved_marker, zorder=0)

    if not show_null_only:

        if annotate_mutations:
            if annotate_min_count > 1:
                annot_muts = merged_obs[
                    merged_obs['ds_mut_id'].isin(mut_counts[mut_counts >= annotate_min_count].index)]
            else:
                annot_muts = merged_obs
            if annotate_xregion is not None:
                annot_muts = annot_muts[(annot_muts['score_1'] >= annotate_xregion[0]) &
                                        (annot_muts['score_1'] <= annotate_xregion[1])]
            if annotate_yregion is not None:
                annot_muts = annot_muts[(annot_muts['score_2'] >= annotate_yregion[0]) &
                                        (annot_muts['score_2'] <= annotate_yregion[1])]

            if len(annot_muts) > 0:
                if annotate_column not in annot_muts.columns:
                    if annotate_column + '_1' in annot_muts.columns:  # May have added suffix in the merging
                        annotate_column += '_1'
                    else:
                        print('Cannot find requested annotation column in dataframes')
                        annotate_column = None

                if annotate_column is not None:
                    annot_muts = annot_muts.drop_duplicates(subset=[annotate_column, 'score_1', 'score_2'])
                    annotate_mutations_on_plot(ax, annot_muts, annotate_column, 'score_1', 'score_2',
                                               annotation_offset=annotation_offset)

        if mut_lists_to_colour is not None:
            if mut_list_colours is None or mut_list_columns is None:
                raise ValueError('If colouring mutations using mut_lists_to_colour, must provide mut_list_colours')
            if not isinstance(mut_lists_to_colour[0], (list, tuple)):
                mut_lists_to_colour = [mut_lists_to_colour]
            if not isinstance(mut_list_colours, (list, tuple)):
                mut_list_colours = [mut_list_colours] * len(mut_lists_to_colour)
            elif len(mut_list_colours) < len(mut_lists_to_colour):
                msg = 'If colouring mutations using mut_lists_to_colour'
                msg += 'must provide mut_list_colours with colour for each list'
                raise ValueError(msg)

            if not isinstance(mut_list_columns, (list, tuple)):
                mut_list_columns = [mut_list_columns] * len(mut_lists_to_colour)
            if not isinstance(mut_list_labels, (list, tuple)):
                mut_list_labels = [mut_list_labels] * len(mut_lists_to_colour)

            for mlist, mcolour, mcolumn, mlabel in zip(mut_lists_to_colour, mut_list_colours,
                                                       mut_list_columns, mut_list_labels):
                muts = merged_obs[merged_obs[mcolumn].isin(mlist)]
                merged_obs = merged_obs[~merged_obs[mcolumn].isin(mlist)]
                if len(muts) > 0:
                    _plot_single_scatter_category(muts, mut_counts, 'score_1', 'score_2',
                                                  marker_size_from_count, base_marker_size, mcolour,
                                                  mlabel, observed_alpha, hotspots_in_foreground,
                                                  marker=observed_marker, zorder=4)

        if score_region_colours is not None:
            #  Add a dummy score column to make sure new score columns get the merging suffix
            merged_obs['score'] = np.nan
            merged_obs = colour_mutations_by_scores(merged_obs, mut_counts, 'score_1', 'score_2', sections_for_colours,
                                       score_regions_for_colours, score_region_colours, marker_size_from_count,
                                       base_marker_size, observed_alpha, hotspots_in_foreground, marker=observed_marker,
                                       zorder=3)

        # Colour by effect.
        for effect, colour in zip(['synonymous', 'missense', 'nonsense'], [synonymous_mutation_colour,
                                                                        missense_mutation_colour,
                                                                        nonsense_mutation_colour]):



            muts = merged_obs[merged_obs['effect'] == effect]
            if len(muts) > 0:
                _plot_single_scatter_category(muts, mut_counts, 'score_1', 'score_2', marker_size_from_count,
                                          base_marker_size, colour, effect,
                                          observed_alpha, hotspots_in_foreground, marker=observed_marker,
                                              zorder=2)


    if plot_xscale is not None:
        plt.xscale(plot_xscale)
    if plot_yscale is not None:
        plt.yscale(plot_yscale)

    if xlabel is not None:
        plt.xlabel(xlabel)
    else:
        try:
            plt.xlabel(section1.lookup.name)
        except AttributeError as e:
            pass
    if ylabel is not None:
        plt.ylabel(ylabel)
    else:
        try:
            plt.ylabel(section2.lookup.name)
        except AttributeError as e:
            pass

    if xlim is not None:
        plt.xlim(xlim)
    if ylim is not None:
        plt.ylim(ylim)

    if show_legend:
        if legend_args is None:
            legend_args = {}
        plt.legend(**legend_args)

    if show_plot:
        plt.show()

    if return_fig and return_dataframes:
        return fig, full_null, full_obs
    elif return_fig:
        return fig
    elif return_dataframes:
        return full_null, full_obs


def annotate_mutations_on_plot(ax, mutations_to_annotate, annotation_column, xcol, ycol,
                                        annotation_offset=(0, 0)):
    """
    Use matplotlib annotation to add text to plot for the dataframe of mutations given.
    :param ax: The axis to plot on.
    :param mutations_to_annotate: Dataframe of mutations to annotated. Must include the xcol, ycol and annotation_column
    as columns.
    :param annotation_column: The value in this column will be used for the text annotation.
    :param xcol: The column to use for the x-position of the text (+ annotation_offset[0])
    :param ycol: The column to use for the x-position of the text (+ annotation_offset[1])
    :param annotation_offset: Tuple (number, number). x and y offset for the annotation relative to the xcol and ycol
    values. The same offset is used for all mutations.
    :return: None.
    """
    for i, row in mutations_to_annotate.iterrows():
        ax.annotate(row[annotation_column], (row[xcol] + annotation_offset[0],
                                             row[ycol] + annotation_offset[1]))


def plot_domain_structure(bins, colours, height, figsize=(15, 1), end_pos=None, linewidth=4, linecolour='C7',
                          round_edges=True, vertical_offset=0, rounding_size=None, pad=0, ax=None):
    """
    Plots a series of blocks in a horizontal line, intended for plotting domains across protein residues.
    A line will be plotted behind the domains. For a gap between domains, use a height of 0 for that bin.

    :param bins: List of boundary points between the domains/regions. Length=number of regions + 1.
    :param colours: Colour for each region.
    :param height: Float/int or list of floats/ints. If single number, a single height will be used for all regions.
    If list, then one height value for each region.
    :param figsize: Tuple. Figure size.
    :param end_pos: The last position for the line (it will start from zero). If None, will use the last bin.
    :param linewidth: The width of the line behind the domain boxes.
    :param linecolour: The colour for the line behind the domain boxes.
    :param round_edges: If True, will use rounded corners for the boxes.
    :param vertical_offset: Used for the FancyBboxPatch with round edges.
    :param rounding_size: Used to define the rounding of if round_edges=True.
    :param pad: Used for the pad of the rounded boxes.
    :param ax: The axis to plot on.
    :return: None.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    if end_pos is None:
        end_pos = bins[-1]
    ax.plot([0, end_pos], [0, 0], linewidth=linewidth, c=linecolour, zorder=0)

    if isinstance(height, (float, int)):
        heights = [height]*len(colours)
    else:
        heights = height
    max_height = max(heights)

    widths = np.diff(bins)
    if rounding_size is None:
        rounding_size = max_height / 10  # Try value that might work.

    for b, c, w, h in zip(bins, colours, widths, heights):
        if c is None:
            continue
        if round_edges:
            bbox = FancyBboxPatch((b, -h / 2 + vertical_offset),
                                  w, h,
                                  boxstyle="round,pad={},rounding_size={}".format(pad, rounding_size),
                                  ec="k", fc=c,
                                  )
        else:
            bbox = Rectangle((b, -h / 2),
                             w, h,
                             ec="k", fc=c,
                             )

        ax.add_patch(bbox)

    ax.axis('off')
    ax.set_ylim([-max_height, max_height])


def hide_top_and_right_axes(ax=None):
    """
    Convenient function to remove the right and top axes from a plot.

    :param ax: 
    :return:
    """
    if ax is None:
        ax = plt.gca()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')