"""
A few functions that are useful for exploring mutations in genes using UniProt or PDBe-KB features, and for listing
available PDB structures.
"""

from darwinian_shift.lookup_classes import UniprotLookup, PDBeKBLookup
from darwinian_shift.lookup_classes.uniprot_lookup import UniprotLookupError
from darwinian_shift.general_functions import DarwinianShift
from darwinian_shift.transcript import Transcript, NoTranscriptError, CodingTranscriptError
from darwinian_shift.section import Section
from darwinian_shift.statistics import binned_chisquare, ztest_cdf_sum
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binom_test
from statsmodels.stats.multitest import multipletests
import requests
from .additional_plotting_functions import hide_top_and_right_axes


example_feature_types = ('signal peptide', 'chain', 'topological domain', 'topological domain',
                                  'topological domain', 'topological domain', 'transmembrane region',
                                 'domain', 'repeat', 'region of interest', 'metal ion-binding site',
                                 'site', 'modified residue', 'glycosylation site', 'disulfide bond',
                                 'cross-link', 'splice variant', 'mutagenesis site', 'sequence conflict',
                                  'helix', 'strand')

example_description_contains=(None, None, None, "Extracellular", 'Cytoplasmic', 'Lumenal', None,
                                         None, None, None, None, None, None, None, None,None, None, None,
                                         None, None, None)

def uniprot_exploration(genes=None, transcripts=None, sections=None, ds_object=None, data=None, exon_file=None, fasta_file=None,
                        spectrum=None, output_plot_file_name_template=None, plot=True,
                        **uniprot_kargs):
    """
    Tests for enrichment or depletion of mutations in uniprot features.
    Uses the UniprotLookup class for the annotating of mutations - the options for defining uniprot feature categories
    are therefore the same as for the UniprotLookup class.

    :param genes: Gene (string) or list of genes to run the analysis on.
    :param transcripts:  Alternative to using 'genes', a transcript or list of transcript ids to run the analysis on.
    :param sections: Alternative to using 'genes' or 'transcripts', provide a list of dictionaries defining the sections
    to run analysis on.
    :param ds_object: A DarwinianShift object with the data to analyse. Alternatively, can provide paths to the data,
    exon_file and fasta_file and a new DarwinianShift object will be made.
    :param data: A path to file of mutations or a dataframe containing the mutational data.
    Not required if the ds_object is given.
    :param exon_file: A path to the exon locations file. Not required if the ds_object is given.
    :param fasta_file: A path to the reference genome. Not required if the ds_object is given.
    :param spectrum: A path to the spectrum to use for analysis. If not given, will use the first spectrum in the
    DarwinianShift object used.
    :param output_plot_file_name_template: Name for the output plots. The name of the region analysed will be added with
    .format.
    :param plot: If True, will plot the expected and observed mutation counts for each category.
    :param uniprot_kargs: Any arguments to pass to the UniprotLookup
    :return: A dataframe of the statistical results. Includes q-values from multiple test correction.
    """
    # Only runs for one spectrum at a time
    if genes is None and transcripts is None and sections is None:
        raise TypeError("Must provide either genes, transcripts or sections to run")
    if genes is not None and isinstance(genes, str):
        genes = [genes]
    elif transcripts is not None and isinstance(transcripts, str):
        transcripts = [transcripts]
    elif sections is not None and isinstance(sections, (dict, pd.Series)):
        sections = [sections]

    seq_type = None
    if genes is not None:
        seq_type = 'gene'
        if transcripts is not None or sections is not None:
            print('Running on given genes, any given transcripts and sections ignored.')
        transcripts = [None]*len(genes)
        sections = [None] * len(genes)
    elif transcripts is not None:
        seq_type = 'transcript'
        if sections is not None:
            print('Running on given transcripts, any given sections ignored.')
        genes = [None]*len(transcripts)
        sections = [None] * len(transcripts)
    elif sections is not None:
        seq_type = 'section'
        genes = [None]*len(sections)
        transcripts = [None]*len(sections)

    if ds_object is None:
        if data is None or exon_file is None or fasta_file is None:
            raise TypeError('Either a DarwinianShift object must be provided, or data, exon_file and fasta_file')
        ds_object = DarwinianShift(data=data,
                       exon_file=exon_file,
                       reference_fasta=fasta_file,
                       lookup=None,
                       spectra=spectrum)

    if isinstance(ds_object.lookup, UniprotLookup) and len(uniprot_kargs) == 0:
        # the ds object already has a uniprot lookup so use that
        uniprot_lookup = ds_object.lookup
    else:
        # Make a new lookup if either ds has a different lookup type, or new kwargs have been given.
        uniprot_lookup = UniprotLookup(**uniprot_kargs)

    if spectrum is None:
        spectrum = ds_object.spectra[0]  # Use the first spectrum from the ds object

    results = []
    all_feature_columns = set()
    for gene, t, sec in zip(genes, transcripts, sections):
        try:
            if seq_type == 'section':
                if isinstance(sec, Section):
                    s = sec
                else:
                    s = ds_object.make_section(sec)
            else:
                transcript = Transcript(gene=gene, transcript_id=t, project=ds_object)
                s = Section(transcript)
            expected, observed, feature_columns, total_muts = _get_uniprot_counts(s, spectrum, uniprot_lookup)

            d = _get_binom_pvalues_dict(expected, observed, feature_columns, total_muts)
            if seq_type == 'gene':
                seq_name = gene
            elif seq_type == 'transcript':
                seq_name = t
            else:
                seq_name = s.section_id

            d[seq_type] = seq_name

            for ex, ob, f in zip(expected, observed, feature_columns):
                d[f + '_expected'] = ex
                d[f + '_observed'] = ob
            results.append(d)
            all_feature_columns.update(feature_columns)
            if plot:
                _plot_uniprot_counts(expected, observed, feature_columns, seq_name, output_plot_file_name_template)
        except (CodingTranscriptError, NoTranscriptError, UniprotLookupError) as e:
            print(type(e).__name__, e, '- Unable to run for', gene)


    results_dataframe = pd.DataFrame(results)
    return _multiple_test_correct(results_dataframe, all_feature_columns)


def _get_uniprot_counts(section, spectrum, uniprot_lookup):
    """
    Get the expected and observed number of mutations in each UniProt feature.
    :param section: Section object.
    :param spectrum: The mutational spectrum to use for the expected mutation rates.
    :param uniprot_lookup: UniprotLookup to use for annotating the mutations with the UniProt features.
    :return:
    """

    # Load the mutations in the section and apply the expected mutation rate
    section.load_section_mutations()
    null_mutations = spectrum.apply_spectrum(section, section.null_mutations)

    # Annotate the null mutations (all possible mutations in the region) with the UniProt features.
    annotated_null_mutations, feature_columns = uniprot_lookup.annotate_dataframe(null_mutations,
                                                                                  section.transcript_id)

    # Apply the UniProt annotations to the observed mutations.
    annotated_observed_mutations = pd.merge(section.observed_mutations, annotated_null_mutations,
                                            on=['pos', 'ref', 'mut'], suffixes=["_x", ""],
                                            how='left')

    # Calculate the total expected and observed mutations in each UniProt feature.
    mut_rates = annotated_null_mutations[spectrum.rate_column]
    annotated_null_mutations['norm_mut_rate'] = mut_rates / mut_rates.sum() * len(section.observed_mutations)
    total_mutations = len(annotated_observed_mutations)
    expected_counts = []
    observed_counts = []
    for feature in feature_columns:
        feature_null_muts = annotated_null_mutations[annotated_null_mutations[feature].notnull()]
        feature_rate_sum = feature_null_muts['norm_mut_rate'].sum()
        if feature_rate_sum > total_mutations and np.isclose(feature_rate_sum, total_mutations):
            # All mutations in the region are part of the feature
            # Floating point errors have put the expected count above the total count. Correct it
            feature_rate_sum = total_mutations
        expected_counts.append(feature_rate_sum)
        feature_observed_muts = annotated_observed_mutations[annotated_observed_mutations[feature].notnull()]
        observed_counts.append(len(feature_observed_muts))

    return expected_counts, observed_counts, feature_columns, total_mutations


def _plot_uniprot_counts(expected_counts, observed_counts, feature_columns, title, output_file_name_template=None):
    """
    A bar plot of the expected and observed mutation counts in UniProt features.
    :param expected_counts:
    :param observed_counts:
    :param feature_columns:
    :param title:
    :param output_file_name_template:
    :return:
    """
    width = 0.4
    xe = np.arange(len(expected_counts))
    xo = xe + width
    plt.bar(xe, expected_counts, width=width, label='Expected')
    plt.bar(xo, observed_counts, width=width, label='Observed')
    plt.xticks(xe + width / 2, feature_columns, rotation='vertical')
    plt.title(title)
    plt.ylabel('Frequency')
    plt.legend()
    if output_file_name_template is not None:
        plt.savefig(output_file_name_template.format(title))
    plt.show()


def _get_binom_pvalues_dict(expected, observed, feature_columns, total_muts):
    return {feature + '_pvalue': binom_test(ob, n=total_muts, p=ex/total_muts) for (ex, ob, feature) in zip(expected, observed,
                                                                                               feature_columns)}

def _multiple_test_correct(results_dataframe, feature_columns):
    """
    Run Benjamini-Hochberg multiple test correction on the p-values from the UniProt exploration and add the q-values
    as columns to the dataframe.
    :param results_dataframe:
    :param feature_columns:
    :return:
    """
    p_value_columns = [col+'_pvalue' for col in feature_columns]
    if not results_dataframe[p_value_columns].empty:
        pvals = results_dataframe[p_value_columns].values.ravel()
        non_nan_pos = np.where(~np.isnan(pvals))
        non_nan_pvals = pvals[non_nan_pos]
        q_ = multipletests(non_nan_pvals, method='fdr_bh')[1]
        # Reinsert the nan values to keep the length of q-values the same as the number of p-value positions in the df
        q = np.full(len(pvals), np.nan)
        q[non_nan_pos] = q_
        q_value_columns = [col+'_qvalue' for col in feature_columns]
        for c in q_value_columns:
            results_dataframe[c] = np.nan

        results_dataframe.loc[:, q_value_columns] = q.reshape(len(results_dataframe), int(len(q) / len(results_dataframe)))

        if "gene" in results_dataframe.columns:
            results_dataframe = results_dataframe.set_index('gene')
        elif "transcript" in results_dataframe.columns:
            results_dataframe = results_dataframe.set_index('transcript')
        elif "section" in results_dataframe.columns:
            results_dataframe = results_dataframe.set_index('section')
        results_dataframe = results_dataframe[sorted(results_dataframe.columns)]

    return results_dataframe


def get_bins_for_uniprot_features(uniprot_features_df,
                                  feature_types=None,
                                  start_from_zero=True, min_gap=0, last_residue=None,
                                  ):
    """
    Returns bins to be used with pd.cut on the 'residue' column
    The intervals are (a,b], so for a uniprot feature from residue X to residue Y (including both ends), the
    interval will be (X-1, Y].
    The bins may therefore mark the end of each uniprot feature, not the start.

    This function is mostly used for a bar plot showing mutations per domain
    (function plot_mutation_counts_in_uniprot_features below).

    :param uniprot_features_df: dataframe of uniprot features from UniprotLookup.get_uniprot_data
    :param feature_types: List of the Uniprot feature types to include. For example, ['domain', 'repeat']
    :param start_from_zero: Include a bin prior to the first listed features
    :param min_gap: Setting this to >0 will count residues in small gaps between features as belonging to the next
    feature.
    :param last_residue: The last residue in the protein (or at least the last residue in the plot). Will add a bin from
    the end of the last feature to the last_residue.
    :return: bins array, types list, descriptions list
    """
    if feature_types is not None:
        features = uniprot_features_df[uniprot_features_df['type'].isin(feature_types)]
    else:
        features = uniprot_features_df
    features = features.sort_values('begin_position')
    bins = []
    types = []
    descriptions = []
    if start_from_zero:
        bins.append(0)
        last_end = 0
    else:
        last_end = None
    for i, row in features.iterrows():
        start = row['begin_position'] - 1   # The intervals in pd.cut do not include the left edge, so -1.
        end = row['end_position']
        type_ = row['type']
        desc = row['description']
        if last_end is None or start > last_end + min_gap:  # Does not continue directly from the previous bin.
            bins.append(start)
            types.append(None)
            descriptions.append(None)
        bins.append(end)
        types.append(type_)
        descriptions.append(desc)
        last_end = end

    if last_residue is not None:
        if last_residue > last_end + min_gap + 1:  # Last part not included in a feature, so add it.
            bins.append(last_residue)
            types.append(None)
            descriptions.append(None)
        elif last_residue > last_end:  # There is a gap at the end, but not large enough for a separate domain.
            bins[-1] = last_residue
    return bins, types, descriptions


def plot_mutation_counts_in_uniprot_features(section, uniprot_data, min_gap=1, feature_types=None,
                           colours=None, labels="descriptions", figsize=(10, 3), linewidth=1,
                           normalise_by_region_size=True, return_bins=False):
    """
    Function to help plot mutation counts in regions defined by uniprot.
    Default arguments unlikely to look good, but can help determine the colours and labels to use.
    Features must not overlap, so the input uniprot_data may have to be filtered (using feature_types or otherwise).

    :param section: The Section object for the gene/region
    :param uniprot_data: dataframe of uniprot features from UniprotLookup.get_uniprot_data
    :param min_gap: If gaps between features are smaller than this, the gap residues will be included in the next feature
    :param feature_types: List of the Uniprot feature types to include. For example, ['domain', 'repeat']
    :param colours: List of colours for each feature
    :param labels: Either list of labels for each feature, or "types" or "descriptions" to use the Uniprot feature type
    or description for the plot labels.
    :param figsize: Figure size for matplotlib
    :param linewidth: Linewidth for matplotlib bar plot
    :param normalise_by_region_size: Will divide the number of mutations in a feature by the number of residues, to plot
    the mutation density. Set to false to plot the raw counts in each feature.
    :param return_bins: see return.
    :return: if return_bins=True, returns the bins for plotting and the type and descriptions of the uniprot features.
    Otherwise, does not return anything.
    """
    if section.null_mutations is None:
        section.load_section_mutations()
    last_residue = section.null_mutations['residue'].max()
    bins, types, descriptions = get_bins_for_uniprot_features(uniprot_data, feature_types=feature_types,
                                                          min_gap=min_gap, last_residue=last_residue)
    if colours is None:
        colours = ["C{}".format(i) for i in range(len(descriptions))]

    section.plot_bar_observations(figsize=figsize, binning_regions=bins,
             normalise_by_region_size=normalise_by_region_size, linewidth=linewidth, facecolour=colours)
    plt.xlim([0, last_residue])
    hide_top_and_right_axes(plt.gca())
    if labels == 'descriptions':
        labels = descriptions
    elif labels == 'types':
        labels = types
    plt.xticks(bins[:-1] + np.diff(bins)/2, labels, rotation=90)
    plt.xlabel('')
    if normalise_by_region_size is False:
        plt.ylabel('Mutation count')

    if return_bins:
        return bins, types, descriptions


def annotate_data(ds_object, output_file=None, verbose=False, annotation_separator="|||", remove_spectra_columns=False,
                  transcripts=None, lookup=None):
    """
    Annotate the input mutations for a DarwinianShift object with the lookup values. Will add the score, and any other
    information added by the annotate_dataframe function of the lookup class.

    If you only want lookup annotation, run ds with spectra=EvenMutationalSpectrum() to save time
    Mutations that are not in any transcripts will not appear in the results.
    Mutations in multiple transcripts will appear multiple times in the results

    Will only work for lookup classes which have an annotate_dataframe function.
    :param ds_object: DarwinianShift object.
    :param output_file: File path to output a tab-separated file with the annotated mutations. If None, will return the
    results as a dataframe.
    :param verbose: If True, will print the progress through each transcript being run.
    :param annotation_separator: String to separate multiple annotations.
    :param remove_spectra_columns: If True, will remove columns including expected mutation rates.
    :param transcripts: List of transcripts to run.  If None, will run all transcripts from the DarwinianShift object
    (which will usually by the longest transcripts of any genes with mutations).
    :param lookup: Lookup object. If None, will use the lookup already associated with the DarwinianShift object.
    :return: If output_file is None, pandas DataFrame. If output_file is a file path, returns None.
    """

    all_annotated_mutations = []
    output_columns = list(ds_object.data.columns)
    if transcripts is None:
        transcripts = ds_object.exon_data['Transcript stable ID'].unique()

    if lookup is None:
        lookup = ds_object.lookup

    for transcript_id in transcripts:
        if verbose:
            print(transcript_id)
        try:
            transcript_obj = ds_object.make_transcript(transcript_id=transcript_id)
            transcript_mutations = transcript_obj.get_observed_mutations()
            transcript_mutations = pd.merge(transcript_mutations, transcript_obj.get_possible_mutations(),
                                            on=['pos', 'ref', 'mut'],
                                            how='left')
            annotated_df, new_columns = lookup.annotate_dataframe(transcript_mutations, transcript_id,
                                                                               sep=annotation_separator)
            all_annotated_mutations.append(annotated_df)
            for n in new_columns:
                if n not in output_columns:
                    output_columns.append(n)

        except (NoTranscriptError, CodingTranscriptError):
            pass

    all_annotated_mutations = pd.concat(all_annotated_mutations, sort=False)
    if remove_spectra_columns:
        all_annotated_mutations = all_annotated_mutations.reindex(columns=output_columns)

    if output_file is None:
        return all_annotated_mutations
    else:
        all_annotated_mutations.to_csv(output_file, sep="\t", index=False)




# Functions to get pdb structure details from Uniprot and the PDBe API
# Useful for finding lists of structures to analyse
STATUS_URL = "http://www.ebi.ac.uk/pdbe/api/pdb/entry/status/"
MOLECULE_URL = "https://www.ebi.ac.uk/pdbe/api/pdb/entry/molecules/"
SUMMARY_URL = "https://www.ebi.ac.uk/pdbe/api/pdb/entry/summary/"


def get_pdb_details(transcript_id, uniprot_lookup=None, min_resolution=None, method=None, reject_mutants=True,
                    uniprot_lookup_kwargs=None):
    """
    Uses UniProt to get a list of available protein structures for a protein.

    :param transcript_id: Used to find the UniProt accession
    :param uniprot_lookup: Use this to provide a UniprotLookup object if wanting to use non-default arguments.
    Alternatively can pass these arguments to uniprot_lookup_kwargs.
    :param min_resolution: Will exclude structures with a resolution value above this threshold
    :param method:  Will only keep structures found using this method, e.g. 'X-ray'
    :param reject_mutants: Will filter out structures with the mutation flag.
    :param uniprot_lookup_kwargs: Dictionary of keyword arguments passed to the UniprotLookup
    :return: Dataframe
    """
    if uniprot_lookup is None:
        if uniprot_lookup_kwargs is None:
            uniprot_lookup_kwargs = {}
        uniprot_lookup = UniprotLookup(**uniprot_lookup_kwargs)

    # Get the full list of PDB structures for the protein
    pdbs = uniprot_lookup.get_pdb_structures_for_transcript(transcript_id)

    # Then filter the results
    if min_resolution is not None:
        if 'resolution' in pdbs.columns:
            pdbs = pdbs[pdbs['resolution'] <= min_resolution]
        else:
            return pd.DataFrame()
    if method is not None:
        pdbs = pdbs[pdbs['method'] == method]

    if len(pdbs) == 0:
        return pd.DataFrame()

    # Add more details about each PDB file using the PDBe API
    # Make one line per PDB chain
    details = []
    for i, p in pdbs.iterrows():
        if is_current(p['pdb_id']):  # Exclude structures which are not current
            title = get_pdb_title(p['pdb_id'])
            chain_details = get_chain_details(p['pdb_id'])
            chains = p['chains'].split('/')
            chain_details = chain_details[chain_details['pdb_chain'].isin(chains)]  # Remove other proteins
            if reject_mutants:
                chain_details = chain_details[pd.isnull(chain_details['mutation_flag'])]

            if len(chain_details) > 0:
                chain_details['title'] = title
                for k, v in p.items():
                    chain_details[k] = v
                details.append(chain_details)

    if len(details) > 0:
        df = pd.concat(details)
        df['transcript_id'] = transcript_id
        return df.reset_index()
    else:
        return pd.DataFrame()


def is_current(pdb_id):
    """
    Check if a PDB file is current or has been deprecated.
    :param pdb_id: String
    :return: Boolean
    """
    r = requests.get(STATUS_URL + pdb_id + '?pretty=false')
    j = r.json()
    if j:  #  Removed theoretical models may return {}
        status = j[pdb_id.lower()][0]['status_code']
        if status == 'REL':
            return True
    return False


def get_chain_details(pdb_id):
    """
    Get all chains and their mutation flags in the given PDB structure
    :param pdb_id: String
    :return: pandas DataFrame
    """
    r = requests.get(MOLECULE_URL + pdb_id + '?pretty=false')
    molecules = r.json()[pdb_id.lower()]
    mol_df = []
    for molecule in molecules:
        molecule_type = molecule['molecule_type']
        if molecule_type == 'polypeptide(L)':
            for chain in molecule['in_chains']:
                mutation_flag = molecule['mutation_flag']
                mol_df.append({'pdb_chain': chain, 'mutation_flag': mutation_flag})

    return pd.DataFrame(mol_df)


def get_pdb_title(pdb_id):
    """

    :param pdb_id:
    :return: String
    """
    r = requests.get(SUMMARY_URL + pdb_id + '?pretty=false')
    title = r.json()[pdb_id.lower()][0]['title']
    return title


# Rough function for quick testing with PDBe-KB data
def pdbe_kb_exploration(ds_object, gene=None, transcript_id=None, score_methods=None, data_label='accession',
                        verbose=False, spectrum=None):
    """
    Runs an analysis of the gene mutations against the data in PDBe-KB.
    This database contains various analyses of the protein structure and binding to other molecules.
    See the PDBeKBLookup class for more information.

    This function has not been extensively tested.

    Running outside of the usual system to reduce redundancy of scoring mutations.

    :param ds_object: DarwinianShift object
    :param gene: Gene to analyse. Alternative, use transcript_id.
    :param transcript_id: Ensembl transcript id to analyse. Use instead of gene.
    :param score_methods: A dictionary of the scoring of non-boolean cases e.g. {
          'cath-funsites': 'mean',
          'efoldmine': 'mean',
          'backbone': 'mean',
          'complex_residue_depth': 'mean',
          'monomeric_residue_depth': 'mean'
        }
    Scores not listed here are assumed to be true for listed residues and false for all residues not listed.
    :param data_label:
    :param verbose:
    :param spectrum:
    :return: pandas dataframe
    """

    if not isinstance(ds_object.lookup, PDBeKBLookup):
        raise TypeError("ds_object must have a PDBeKBLookup as its lookup attribute")
    if score_methods is None:
        score_methods = dict(ds_object.lookup.annotation_default_score_methods)
    if spectrum is None:
        spectrum = ds_object.spectra[0]

    section = ds_object.make_section(gene=gene, transcript_id=transcript_id)
    section.load_section_mutations()
    old_cols = section.null_mutations.columns
    annotated_null = ds_object.lookup.annotate_df(transcript_id=section.transcript_id, df=section.null_mutations,
                                                  score_methods=score_methods, data_label=data_label)
    new_cols = [c for c in annotated_null.columns if c not in old_cols]
    annotated_observed = pd.merge(section.observed_mutations,
                                  annotated_null[['pos', 'ref', 'mut'] + new_cols], on=['pos', 'ref', 'mut'],
                                  how='left', suffixes=["_x", ""])

    res = []
    for col in new_cols:
        if verbose:
            print(col)

        null = annotated_null[~pd.isnull(annotated_null[col])]
        obs = annotated_observed[~pd.isnull(annotated_observed[col])]
        if not obs.empty:
            if score_methods.get(col, None) is None:
                stats = binned_chisquare(null_scores=null[col].values.astype(float),
                                         null_mut_rates=null[spectrum.rate_column].values,
                                         observed_values=obs[col].astype(float), bins=[-0.5, 0.5, 1.5])
            else:
                stats = ztest_cdf_sum(null[col], null[spectrum.rate_column].values, obs[col])

            if stats is not None:
                stats[data_label] = col
                res.append(stats)

    res = pd.DataFrame(res)
    if not res.empty:
        res['qvalue'] = multipletests(res['pvalue'], method='fdr_bh')[1]
    return res