import pandas as pd
import os
import glob
from darwinian_shift.utils import get_sifts_alignment_for_chain
from .errors import MetricLookupException

class FoldXLookupError(MetricLookupException): pass


class FoldXLookup:
    """
    Scores mutations using pre-calculated ∆∆G values generated using FoldX position scan.

    Requires pdb_id and pdb_chain attributes to be defined for the section being analysed.

    The FoldX results must be in the following directory structure:

    ─── foldx_results_directory
        │
        ├── ABC1    # PDB file ID used as the folder name
        │   │
        │   ├── chainA    # Subfolder for each chain of interest containing the FoldX results.
        │   │   ├── PS_f0_scanning_output.txt
        │   │   ├── PS_f1_scanning_output.txt
        │   │   ...
        │   │
        │   └── chainB
        │       ├── PS_f0_scanning_output.txt
        │       ...
        │
        └── XY2Z
            └── chainD
                ├── PS_f0_scanning_output.txt
                ...

    """
    # Map of 3-letter codes to 1-letter code from http://foldxsuite.crg.eu/allowed-residues
    AA_map = {'GLY': 'G', 'ALA': 'A', 'LEU': 'L', 'VAL': 'V', 'ILE': 'I', 'PRO': 'P', 'ARG': 'R',
              'THR': 'T', 'SER': 'S', 'CYS': 'C', 'MET': 'M', 'LYS': 'K', 'GLU': 'E', 'GLN': 'Q',
              'ASP': 'D', 'ASN': 'N', 'TRP': 'W', 'TYR': 'Y', 'PHE': 'F', 'HIS': 'H',
              'PTR': 'T',  # phoshoporylated threonine
              'TPO': 'Y',  # phosphorylated tyrosine
              'SEP': 'S',  # phosphorylated serine
              'HYP': 'P',  # hydroxiproline
              'TYS': 'Y',  # sulfotyrosine
              'MLZ': 'K',  # monomethylated lysine
              'MLY': 'K',  # dimethylated lysine
              'M3L': 'K',  # trimethylated lysine
              'H1S': 'H',  # charged ND1 histidine
              'H2S': 'H',  # charged NE2 histidine
              'H3S': 'H'  # neutral histidine
              }

    def __init__(self, foldx_results_directory, sifts_directory=None, download_sifts=None, foldx_file_name_start="PS_f*",
                 force_synonymous_zero=True, name='FoldX ∆∆G (kcal/mol)'):
        """

        :param foldx_results_directory: Base directory of the FoldX results output
        :param sifts_directory: File path to directory of SIFTS files for aligning PDB files with protein sequence positions
        :param download_sifts: If True, will attempt to download the SIFTS file if it is not found in the sifts_directory.
        :param foldx_file_name_start: The start of the file names of the FoldX results to use.
        :param force_synonymous_zero: Will force synonymous mutations to have a ∆∆G value of zero.
        :param name: Name of the lookup to appear on plot axes.
        """
        self.foldx_results_directory = foldx_results_directory
        self.foldx_file_name_start = foldx_file_name_start
        self.sifts_directory = sifts_directory
        self.download_sifts = download_sifts
        self.force_synonymous_zero = force_synonymous_zero
        self.name = name  # Will appear on some plot axes

    def setup_project(self, project):
        if self.sifts_directory is None:
            self.sifts_directory = project.sifts_directory
        else:
            project.sifts_directory = self.sifts_directory
        if self.download_sifts is None:
            self.download_sifts = project.download_sifts
        else:
            project.download_sifts = self.download_sifts

    def __call__(self, seq_object):
        if seq_object.pdb_id is None or seq_object.pdb_chain is None:
            raise TypeError("Must define the 'pdb_id' and 'pdb_chain' to use this lookup")
        return self._get_scores(seq_object.pdb_id, seq_object.pdb_chain, seq_object.null_mutations,
                                getattr(seq_object, 'foldx_results_dir', None))

    def _convert_alt(self, row):
        if row['aa_mut'] in 'oe':
            return 'H'
        else:
            return row['aa_mut']

    def _get_info_cols(self, row):
        aachange = row['aachange']
        ref_aa = aachange[:3]
        mut_aa = aachange[-1]
        resnum = int(aachange[4:-1])
        return ref_aa, mut_aa, resnum

    def _get_mutid(self, row):
        return '{}{:.0f}{}'.format(row['converted_ref'], row['resnum'], row['converted_alt'])

    def load_foldx_data(self, foldx_results_dir):
        """

        :param foldx_results_dir:
        :return:
        """
        # Check directory exists first
        if not os.path.isdir(foldx_results_dir):
            raise FoldXLookupError('FoldX: Directory {} does not exist'.format(foldx_results_dir))


        all_results = []
        for f in glob.glob(os.path.join(foldx_results_dir, self.foldx_file_name_start + '*')):
            with open(f) as fh:
                for line in fh:
                    try:
                        change, ddG = line.strip().split()
                        all_results.append({
                            'aachange': change,
                            'ddG': float(ddG)
                        })
                    except Exception as e:
                        print(f, line)
                        raise e

        all_results = pd.DataFrame(all_results)

        if len(all_results) > 0:
            ref_mut_resnums = all_results.apply(self._get_info_cols, axis=1)
            refs, muts, resnums = zip(*ref_mut_resnums)

            all_results['aa_ref'] = refs
            all_results['aa_mut'] = muts
            all_results['resnum'] = resnums

            all_results['converted_ref'] = all_results.apply(lambda x: self.AA_map[x['aa_ref']], axis=1)
            all_results['converted_alt'] = all_results.apply(self._convert_alt, axis=1)
            if self.force_synonymous_zero:
                # For some cases, particularly histidines where the charge can effect ∆∆G,
                # FoldX can predict that a synonymous mutation has non-zero ∆∆G. Change to zero.
                all_results.loc[all_results['converted_ref'] == all_results['converted_alt'], 'ddG'] = 0
            all_results.drop_duplicates(subset=['resnum', 'converted_ref', 'converted_alt'], inplace=True)
        else:
            raise FoldXLookupError('FoldX: Found no results for {}'.format(foldx_results_dir))

        return all_results

    def _get_scores(self, pdb_id, pdb_chain, df, pdb_foldx_results_dir=None):
        sifts = get_sifts_alignment_for_chain(pdb_id, pdb_chain, self.sifts_directory, self.download_sifts)
        if sifts is None:
            scores = None
        elif len(sifts['uniprot id'].unique()) > 1:
            raise FoldXLookupError('SIFTS: Multiple uniprot ids {} for same chain {}:chain{}. May be fusion?'.format(
                sifts['uniprot id'].unique(), pdb_id, pdb_chain))
        else:
            if pdb_foldx_results_dir is None:
                # If not given, try to use the file structure self.foldx_results_directory/PDB_ID/chainX
                pdb_foldx_results_dir = os.path.join(self.foldx_results_directory, pdb_id.upper(), "chain{}".format(pdb_chain))

            foldx_results = self.load_foldx_data(pdb_foldx_results_dir)
            foldx_results = pd.merge(foldx_results, sifts, left_on='resnum', right_on='pdb position', how='left')
            foldx_results.rename(index=str, columns={"resnum": "old_resnum", "uniprot position": "resnum"}, inplace=True)
            foldx_results['aachange'] = foldx_results.apply(self._get_mutid, axis=1)
            merge_df = pd.merge(df, foldx_results, on='aachange', how='left')
            scores = merge_df['ddG'].values
        return scores
