from darwinian_shift.lookup_classes import UniprotLookup
from darwinian_shift.general_functions import DarwinianShift
from darwinian_shift.transcript import Transcript, NoTranscriptError, CodingTranscriptError
from darwinian_shift.section import Section
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binom_test
from statsmodels.stats.multitest import multipletests
import requests


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
                        spectrum=None, output_file_name_template=None, plot=True,
                        **uniprot_kargs):
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
    feature_columns = []
    for gene, t, sec in zip(genes, transcripts, sections):
        if seq_type == 'section':
            s = ds_object.make_section(sec)
        else:
            transcript = Transcript(gene=gene, transcript_id=t, project=ds_object)
            s = Section(transcript)
        expected, observed, feature_columns, total_muts = _get_uniprot_counts(s, spectrum, uniprot_lookup)

        d = _get_poisson_pvalues_dict(expected, observed, feature_columns, total_muts)
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
        if plot:
            _plot_uniprot_counts(expected, observed, feature_columns, seq_name, output_file_name_template)

    results_dataframe = pd.DataFrame(results)
    return _multiple_test_correct(results_dataframe, feature_columns)


def _get_uniprot_counts(section, spectrum, uniprot_lookup):

    section.load_section_mutations()
    null_mutations = spectrum.apply_spectrum(section, section.null_mutations)
    annotated_null_mutations, feature_columns = uniprot_lookup.annotate_dataframe(null_mutations,
                                                                                  section.transcript_id)

    annotated_observed_mutations = pd.merge(section.observed_mutations, annotated_null_mutations,
                                            on=['pos', 'ref', 'mut'], suffixes=["_x", ""],
                                            how='left')

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


def _get_poisson_pvalues_dict(expected, observed, feature_columns, total_muts):
    return {feature + '_pvalue': binom_test(ob, n=total_muts, p=ex/total_muts) for (ex, ob, feature) in zip(expected, observed,
                                                                                               feature_columns)}

def _multiple_test_correct(results_dataframe, feature_columns):
    p_value_columns = [col+'_pvalue' for col in feature_columns]
    q = multipletests(results_dataframe[p_value_columns].values.ravel(), method='fdr_bh')[1]
    q_value_columns = [col+'_qvalue' for col in feature_columns]
    for c in q_value_columns:
        results_dataframe[c] = np.nan
    results_dataframe.loc[:, q_value_columns] = q.reshape(len(results_dataframe), int(len(q) / len(results_dataframe)))
    return results_dataframe


def get_bins_for_uniprot_features(uniprot_features_df,
                                  feature_types=('domain', 'repeat', 'transmembrane region'),
                                  start_from_zero=True, min_gap=0, last_residue=None):
    features = uniprot_features_df[uniprot_features_df['type'].isin(feature_types)].sort_values('begin_position')
    bins = []
    types = []
    descriptions = []
    if start_from_zero:
        bins.append(0)
    last_end = None
    for i, row in features.iterrows():
        start = row['begin_position']
        end = row['end_position']
        type_ = row['type']
        desc = row['description']
        if last_end is None or start > last_end + min_gap + 1:  # Does not continue directly from the previous bin.
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



def annotate_data(ds_object, output_file=None, verbose=False, annotation_separator="|||", remove_spectra_columns=False,
                  transcripts=None):
    # Set up a ds_object with a lookup.
    # Annotated the mutations with the lookup information
    # If only want lookup annotation, run ds with spectra=EvenMutationalSpectrum() to save time
    # Mutations that are not in any transcripts will not appear in the results.
    # Mutations in multiple transcripts will appear multiple times in the results
    all_annotated_mutations = []
    output_columns = list(ds_object.data.columns)
    if transcripts is None:
        transcripts = ds_object.exon_data['Transcript stable ID'].unique()

    for transcript_id in transcripts:
        if verbose:
            print(transcript_id)
        try:
            transcript_obj = ds_object.make_transcript(transcript_id=transcript_id)
            transcript_mutations = transcript_obj.get_observed_mutations()
            transcript_mutations = pd.merge(transcript_mutations, transcript_obj.get_possible_mutations(),
                                            on=['pos', 'ref', 'mut'],
                                            how='left')
            annotated_df, new_columns = ds_object.lookup.annotate_dataframe(transcript_mutations, transcript_id,
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


def get_pdb_details(uniprot_lookup, transcript_id, min_resolution=None, method=None, reject_mutants=True):
    pdbs = uniprot_lookup.get_pdb_structures_for_transcript(transcript_id)
    if min_resolution is not None:
        if 'resolution' in pdbs.columns:
            pdbs = pdbs[pdbs['resolution'] <= min_resolution]
        else:
            return pd.DataFrame()
    if method is not None:
        pdbs = pdbs[pdbs['method'] == method]

    if len(pdbs) == 0:
        return pd.DataFrame()
    details = []
    for i, p in pdbs.iterrows():
        if is_current(p['pdb_id']):
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
        return df
    else:
        return pd.DataFrame()


def is_current(pdb_id):
    r = requests.get(STATUS_URL + pdb_id + '?pretty=false')
    j = r.json()
    if j:  #  Removed theoretical models may return {}
        status = j[pdb_id.lower()][0]['status_code']
        if status == 'REL':
            return True
    return False


def get_chain_details(pdb_id):
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
    r = requests.get(SUMMARY_URL + pdb_id + '?pretty=false')
    title = r.json()[pdb_id.lower()][0]['title']
    return title
