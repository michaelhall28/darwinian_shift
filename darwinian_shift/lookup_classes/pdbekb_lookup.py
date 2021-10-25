import pandas as pd
import numpy as np
import json
import gzip
import requests
import os
import collections
from darwinian_shift.utils.util_functions import get_uniprot_acc_from_transcript_id
from .errors import MetricLookupException

class PDBeKBLookupError(MetricLookupException): pass

class PDBeKBLookup:
    """
    Uses the PDBe-KB data to score mutations.
    Some data comes with a score, others are just a list of residues that meet the criteria e.g. on an interface
    For cases using scores, the residues without a score will be marked as null and excluded from statistical tests

    To use this lookup, a dictionary of pdbekb_kwargs is required when running the analysis.
    For example:
    d.run_section({
        'gene': GENESYMBOL,
        'pdbekb_kwargs': {
            'uniprot_subsection': 'annotations',
            'data_accessions': 'monomeric_residue_depth',
            'score_method': 'mean'
        }
    }

    The options available will depend on the structure.
    To use a score (instead of true/false for residue listed or not), define the score_method in pdbekb_kwargs.
    The options are 'mean', 'median', 'min', 'max' or a function that could be applied to a list of values.

    This data is associated with PDB structures, so in some cases may be better accessed elsewhere where the results for
    the full protein may be available (e.g. 14-3-3-pred).
    """
    uniprot_subsections = ('annotations', 'ligand_sites', 'interface_residues')

    # For some data types the score is important, for other data it just lists the residues in the category
    # The data types that use scores are listed here
    # Listed with a method to combine scores for the same residue from multiple structures
    # data not listed assumed to be just true/false for each residue
    # used for dataframe annotation
    annotation_default_score_methods = [('cath-funsites', 'mean'),
                                        ('efoldmine', 'mean'),
                                        ('backbone', 'mean'),
                                        ('complex_residue_depth', 'mean'),
                                        ('monomeric_residue_depth', 'mean'),
                                        ('14-3-3-pred', 'mean')
                                        ]

    def __init__(self, pdbekb_dir='.', verbose=False, force_download=False, store_json=False,
                 base_url="https://www.ebi.ac.uk/pdbe/graph-api/uniprot/", name="PDBe-KB",
                 transcript_uniprot_mapping=None):
        """

        :param pdbekb_dir: Directory where pdbekb json.gz files can be stored.
        :param verbose:
        :param force_download: Will download new PDBeKB files, even if a previous one exists in the pdbekb_dir.
        :param store_json: Will store the PDBeKB json.gz files in the pdbekb_dir after downloading.
        :param base_url: The base url used to obtain the data.
        :param name: Name of the lookup to appear on plot axes.
        :param transcript_uniprot_mapping: Dictionary of mapping from transcript id to Uniprot accession number if not
        using the mapping from Uniprot. Be careful that the residue numbers still matches the Uniprot annotations.
        """
        self.pdbekb_dir = pdbekb_dir
        self.verbose = verbose
        self.store_json = store_json
        self.force_download = force_download
        if base_url[-1] != '/':
            base_url += '/'
        self.base_url = base_url + '{}/{}'
        self.name = name

        self.transcript_uniprot_mapping = None
        if isinstance(transcript_uniprot_mapping, dict):
            self.transcript_uniprot_mapping = transcript_uniprot_mapping
        elif isinstance(transcript_uniprot_mapping, str):
            self.transcript_uniprot_mapping = {}
            with open(transcript_uniprot_mapping) as fh:
                for line in fh:
                    try:
                        transcript, uniprot = line.strip().split()
                    except ValueError as e:
                        # Skip any rows with wrong number of entries (e.g. where uniprot id is missing).
                        continue
                    self.transcript_uniprot_mapping[transcript] = uniprot
        elif transcript_uniprot_mapping is not None:
            raise PDBeKBLookupError('transcript_uniprot_mapping must be dictionary or file path')

    def __call__(self, seq_object):
        return self._get_scores(seq_object.null_mutations, seq_object.transcript_id, seq_object.pdbekb_kwargs)

    def setup_project(self, project):
        if project.verbose and not self.verbose:
            self.verbose = True

    def _load_pdbekb_data(self, transcript_id, pdbekb_kwargs, silent=False):
        uniprot_acc = None
        if self.transcript_uniprot_mapping is not None:
            uniprot_acc = self.transcript_uniprot_mapping.get(transcript_id, None)

        if uniprot_acc is None:
            uniprot_acc = get_uniprot_acc_from_transcript_id(transcript_id)
            if uniprot_acc == '':
                raise ValueError('Could not find uniprot id for transcript_id', transcript_id)
        if self.verbose and not silent:
            print("Using uniprot accession {} for transcript id {}".format(uniprot_acc, transcript_id))
            print("https://www.uniprot.org/uniprot/{}".format(uniprot_acc))

        uniprot_subsection = pdbekb_kwargs['uniprot_subsection']

        # Look to see if the data has already been downloaded
        file_name = os.path.join(self.pdbekb_dir, uniprot_acc + '_' + uniprot_subsection + '.json.gz')
        if os.path.isfile(file_name) and not self.force_download:
            with gzip.open(file_name) as fh:
                data = json.load(fh)
        else:
            # Download the data if it hasn't been found.
            url = self.base_url.format(uniprot_subsection, uniprot_acc)
            data = requests.get(url.format(uniprot_acc)).json()
            if self.store_json:  # Save for future use
                with gzip.open(file_name, 'wt', encoding="ascii") as fh:
                    json.dump(data, fh)

        if data:
            data = data[uniprot_acc]['data']
        return data

    def _process_all_entries(self, data, score_method=None):
        if score_method is None:
            # Using boolean, residue is/is not listed.
            # Just need to return a set of residues
            return self._get_sites_from_all_entries(data)
        else:
            # Getting the raw scores from the data and processing using the score_method
            return self._get_scores_from_all_entries(data, score_method)

    def _get_sites_from_all_entries(self, data):
        residue_numbers = set()
        for d in data:
            for res in d['residues']:
                residue_numbers.update(list(range(res['startIndex'], res['endIndex'] + 1)))
        return residue_numbers

    def _get_scores_from_all_entries(self, data, score_method='mean'):
        residue_data = collections.defaultdict(list)
        for d in data:
            for res in d['residues']:
                if res['startIndex'] == res['endIndex']:
                    residue_data[res['startIndex']].extend([r['additionalData']['rawScore'] for r in res['pdbEntries']])
                else:
                    # Range of residues given.
                    # Assume the values apply to all residues in this range
                    for i in range(res['startIndex'], res['endIndex'] + 1):
                        residue_data[i].extend([r['additionalData']['rawScore'] for r in res['pdbEntries']])

        if score_method == 'mean':
            f = np.mean
        elif score_method == 'max':
            f = np.max
        elif score_method == 'min':
            f = np.min
        elif score_method == 'median':
            f = np.median
        elif isinstance(score_method, collections.Callable):
            f = score_method

        df = pd.DataFrame([(k, f(v)) for k, v in residue_data.items()])
        df.columns = ['residue', 'score']
        return df

    def _process_data(self, data, data_accessions=None, data_names=None, score_method=None):
        """
        If score method is None, will return a set of residues
        If score method is not None, will return a dataframe with a residue and a score column
        """

        if data_accessions is None and data_names is None:
            # Use all the ligand data
            processed_data = self._process_all_entries(data, score_method)
        else:
            if data_names is not None:
                if isinstance(data_names, str):
                    data_names = [data_names]
            if data_accessions is not None:
                if isinstance(data_accessions, str):
                    data_accessions = [data_accessions]

            filtered_data = []
            for d in data:
                if ((data_names is not None and d['name'] in data_names) or
                        (data_accessions is not None and d['accession'] in data_accessions)):
                    filtered_data.append(d)
            processed_data = self._process_all_entries(filtered_data, score_method)

        return processed_data

    def _get_processed_data(self, transcript_id, pdbekb_kwargs):
        """
        If score method is None or is not given, will return a set of residues
        If score method is not None, will return a dataframe with a residue column and a score column
        """
        data = self._load_pdbekb_data(transcript_id, pdbekb_kwargs)
        if data:
            processed_data = self._process_data(data, pdbekb_kwargs.get('data_accessions', None),
                                                pdbekb_kwargs.get('data_names', None),
                                                pdbekb_kwargs.get('score_method', None))
            return processed_data
        else:
            return None

    def _get_scores(self, df, transcript_id, pdbekb_kwargs):
        processed_data = self._get_processed_data(transcript_id, pdbekb_kwargs)
        if processed_data is None:  # No data of the requested type
            return None
        if isinstance(processed_data, set):  # Boolean, residue is/is not listed in the data
            scores = df['residue'].isin(processed_data).astype(float)
        else:  # Scores for each residue used, processed_data is a dataframe
            merged_df = pd.merge(df, processed_data, on='residue', how='left')
            scores = merged_df['score'].values
        return scores

    def get_all_data_types(self, transcript_id, uniprot_subsection=None):
        if uniprot_subsection is not None:
            data = self._load_pdbekb_data(transcript_id, {'uniprot_subsection': uniprot_subsection})
            return {uniprot_subsection: [(d['name'], d['accession']) for d in data]}
        else:
            res = {}
            silent = False
            for us in self.uniprot_subsections:
                data = self._load_pdbekb_data(transcript_id, {'uniprot_subsection': us}, silent=silent)
                res[us] = [{'name': d['name'], 'accession': d['accession']} for d in data]
                silent = True
            return res

    def annotate_df(self, df, transcript_id, score_methods=None, data_label='accession'):
        if score_methods is None:
            score_methods = dict(self.annotation_default_score_methods)
            print(score_methods)

        silent = False
        for us in self.uniprot_subsections:
            data = self._load_pdbekb_data(transcript_id, {'uniprot_subsection': us}, silent=silent)
            for d in data:
                label = d[data_label]
                score_method = score_methods.get(label, None)
                if score_method is None:
                    residues = self._get_sites_from_all_entries([d])
                    df[label] = df['residue'].isin(residues)
                else:
                    scores = self._get_scores_from_all_entries([d], score_method)
                    df[label] = pd.merge(df[['residue']], scores, on='residue', how='left')['score']
            silent = True
        return df



