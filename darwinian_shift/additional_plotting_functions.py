import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle


def _plot_single_scatter_category(muts, mut_counts, xcol, ycol,
                                  marker_size_from_count, base_marker_size, colour, label, alpha,
                                  hotspots_in_foreground, marker=None, ax=None, zorder=None):
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
        linewidth = [0 if size < 20 else 0.2 for size in s]
    else:
        s = base_marker_size
        linewidth = 0
    ax.scatter(x, y, s=s, c=colour, edgecolor='k', linewidth=linewidth, label=label, alpha=alpha, marker=marker,
               zorder=zorder)


def colour_mutations_by_scores(mutation_df, mut_counts, xcol, ycol, sections_for_colours, score_regions_for_colours,
                               score_region_colours, marker_size_from_count, base_marker_size, alpha,
                               hotspots_in_foreground, ax=None, use_null=False, marker=None, zorder=None):
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
                               suffixes=['', '_{}'.format(i + 3)], how='left')
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
                            show_observed_only=False, show_null_only=False):

    # Can only plot mutations that have a score in both sections.
    common_mut_ids = set(section1.null_mutations['ds_mut_id']).intersection(section2.null_mutations['ds_mut_id'])
    s1_null = section1.null_mutations[section1.null_mutations['ds_mut_id'].isin(common_mut_ids)].copy()
    s2_null = section2.null_mutations[section2.null_mutations['ds_mut_id'].isin(common_mut_ids)].copy()
    s1_obs = section1.observed_mutations[section1.observed_mutations['ds_mut_id'].isin(common_mut_ids)].copy()
    s2_obs = section2.observed_mutations[section2.observed_mutations['ds_mut_id'].isin(common_mut_ids)].copy()

    # if sections_for_colours is not None:
    #     for s in sections_for_colours:
    #         assert len(s.null_mutations) == len(s1_null)
    #         assert len(s.observed_mutations) == len(s1_obs)

    merged_null = pd.merge(s1_null, s2_null, on=['pos', 'ref', 'mut', 'effect',
                                                                                 'aachange', 'ds_mut_id'],
                           suffixes=['_1', '_2'])



    # Need to deduplicate mutations before merging to prevent additional copies of recurrent mutations being created
    obs2 = s2_obs.drop_duplicates(subset='ds_mut_id')
    merged_obs = pd.merge(s1_obs, obs2,
                          on=['pos', 'ref', 'mut', 'effect', 'aachange', 'ds_mut_id'], suffixes=['_1', '_2'])

    mut_counts = merged_obs['ds_mut_id'].value_counts()

    if not show_observed_only:
        null_to_plot = merged_null[~merged_null['ds_mut_id'].isin(merged_obs['ds_mut_id'])].copy()

    fig, ax = plt.subplots(figsize=figsize)
    if not show_observed_only:
        plt.scatter(null_to_plot['score_1'], null_to_plot['score_2'], s=unmutated_marker_size,
                c=unobserved_mutation_colour, alpha=unobserved_alpha, marker=unobserved_marker, zorder=0)

    # Colour by effect.
    for effect, colour in zip(['synonymous', 'missense', 'nonsense'], [synonymous_mutation_colour,
                                                                    missense_mutation_colour,
                                                                    nonsense_mutation_colour]):

        if not show_null_only:
            muts = merged_obs[merged_obs['effect'] == effect]
            if len(muts) > 0:
                _plot_single_scatter_category(muts, mut_counts, 'score_1', 'score_2', marker_size_from_count,
                                          base_marker_size, colour, effect,
                                          observed_alpha, hotspots_in_foreground, marker=observed_marker,
                                              zorder=2)

    if sections_for_colours is not None:
        if colour_unmutated_by_scores and not show_observed_only:
            null_to_plot['score'] = np.nan
            colour_mutations_by_scores(null_to_plot, None, 'score_1', 'score_2', sections_for_colours,
                                       score_regions_for_colours, score_region_colours, marker_size_from_count=False,
                                       base_marker_size=unmutated_marker_size, alpha=unobserved_alpha,
                                       hotspots_in_foreground=False, use_null=True, marker=unobserved_marker,
                                       zorder=1)

        if not show_null_only:
            merged_obs['score'] = np.nan  #  Add a dummy score column to make sure new score columns get the merging suffix
            colour_mutations_by_scores(merged_obs, mut_counts,  'score_1', 'score_2', sections_for_colours,
                                   score_regions_for_colours, score_region_colours, marker_size_from_count,
                                   base_marker_size, observed_alpha, hotspots_in_foreground, marker=observed_marker,
                                       zorder=3)

    if mut_lists_to_colour is not None:
        if mut_list_colours is None or mut_list_columns is None:
            raise ValueError('If colouring mutations using mut_lists_to_colour, must provide mut_list_colours')
        if not isinstance(mut_lists_to_colour[0], (list, tuple)):
            mut_lists_to_colour = [mut_lists_to_colour]
        if not isinstance(mut_list_colours, (list, tuple)):
            mut_list_colours = [mut_list_colours]*len(mut_lists_to_colour)
        elif len(mut_list_colours) < len(mut_lists_to_colour):
            raise ValueError('If colouring mutations using mut_lists_to_colour, must provide mut_list_colours with colour for each list')

        if not isinstance(mut_list_columns, (list, tuple)):
            mut_list_columns = [mut_list_columns]*len(mut_lists_to_colour)
        if not isinstance(mut_list_labels, (list, tuple)):
            mut_list_labels = [mut_list_labels]*len(mut_lists_to_colour)

        for mlist, mcolour, mcolumn, mlabel in zip(mut_lists_to_colour, mut_list_colours,
                                                   mut_list_columns, mut_list_labels):
            muts = merged_obs[merged_obs[mcolumn].isin(mlist)]
            if len(muts) > 0:
                _plot_single_scatter_category(muts, mut_counts, 'score_1', 'score_2',
                                              marker_size_from_count, base_marker_size, mcolour,
                                              mlabel, observed_alpha, hotspots_in_foreground, marker=observed_marker)

    if annotate_mutations:
        if annotate_min_count > 1:
            annot_muts = merged_obs[merged_obs['ds_mut_id'].isin(mut_counts[mut_counts >= annotate_min_count].index)]
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
        return fig, merged_null, merged_obs
    elif return_fig:
        return fig
    elif return_dataframes:
        return merged_null, merged_obs


def annotate_mutations_on_plot(ax, mutations_to_annotate, annotation_column, xcol, ycol,
                                        annotation_offset=(0, 0)):
    for i, row in mutations_to_annotate.iterrows():
        ax.annotate(row[annotation_column], (row[xcol] + annotation_offset[0],
                                             row[ycol] + annotation_offset[1]))


def plot_domain_structure(bins, colours, height, figsize=(15, 1), end_pos=None, linewidth=4, linecolour='C7',
                          round_edges=True, vertical_offset=0, rounding_size=None, pad=0, ax=None):
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
    if ax is None:
        ax = plt.gca()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')