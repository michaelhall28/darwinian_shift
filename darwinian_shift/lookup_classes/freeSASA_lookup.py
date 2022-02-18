import os
import freesasa
import pandas as pd
import numpy as np
from darwinian_shift.utils import get_sifts_alignment_for_chain, download_pdb_file


class FreeSASALookup:
    """
    Calculates the solvent accessible surface area for each residue.

    Can use any of the FreeSASA ResidueArea attributes to score each residue.
    These are:
        "total": Total SASA of residue. Default
        "polar": Polar SASA
        "apolar": Apolar SASA
        "mainChain": Main chain SASA
        "sideChain": Side chain SASA
        "relativeTotal": Relative total SASA
        "relativePolar": Relative polar SASA
        "relativeApolar": Relative Apolar SASA
        "relativeMainChain": Relative main chain SASA
        "relativeSideChain": Relative side chain SASA

    """

    def __init__(self, metric='total', freesasa_parameters=None,
                 pdb_directory='.', sifts_directory='.', download_sifts=None,
                 download_pdb_file=False, name='Solvent Accessibility'):
        """

        :param metric: SASA metric. Default='total'
        :param freesasa_parameters: Change the parameters FreeSASA uses to calculate the solvent accessibility.
        E.g. freesasa.Parameters({'algorithm' : freesasa.LeeRichards, 'n-slices' : 100})
        :param pdb_directory: File path to the directory where pdb files are stored.
        :param sifts_directory: File path to directory of SIFTS files for aligning PDB files with protein sequence positions
        :param download_sifts: If True, will attempt to download the SIFTS file if it is not found in the sifts_directory.
        :param name: Name of the lookup to appear on plot axes.
        """
        self.metric = metric
        self.freesasa_parameters = freesasa_parameters
        self.pdb_directory = pdb_directory
        self.sifts_directory = sifts_directory
        self.download_sifts = download_sifts
        self.download_pdb_file = download_pdb_file
        self.name = name  # Will appear on some plot axes

    def setup_project(self, project):
        if self.pdb_directory is None:
            if project.pdb_directory is not None:
                self.pdb_directory = project.pdb_directory
            else:
                self.pdb_directory = '.'
        else:
            project.pdb_directory = self.pdb_directory

        if self.sifts_directory is None:
            self.sifts_directory = project.sifts_directory
        else:
            project.sifts_directory = self.sifts_directory
        if self.download_sifts is None:
            self.download_sifts = project.download_sifts
        else:
            project.download_sifts = self.download_sifts

    def __call__(self, seq_object):
        return self._get_scores(seq_object.null_mutations, seq_object.pdb_id, seq_object.pdb_chain)

    def _get_scores(self, df, pdb_id, pdb_chain):
        sifts = get_sifts_alignment_for_chain(pdb_id, pdb_chain, self.sifts_directory, self.download_sifts)
        if sifts is None:
            scores = None
        else:
            df = pd.merge(df, sifts, left_on='residue', right_on='uniprot position', how='left')

            pdb_file_path = os.path.join(self.pdb_directory, pdb_id + '.pdb')
            if not os.path.isfile(pdb_file_path):
                # PDB file not already downloaded.
                if self.download_pdb_file:
                    download_pdb_file(pdb_id, self.pdb_directory)
                else:
                    raise LookupError("PDB file {} is not in the pdb_directory {}".format(pdb_id, self.pdb_directory))

            structure = freesasa.Structure(pdb_file_path)
            result = freesasa.calc(structure, self.freesasa_parameters)
            chain_results = result.residueAreas()[pdb_chain]
            scores = np.full(len(df), np.nan)
            for i, residue in enumerate(df['pdb position']):
                if not np.isnan(residue):
                    try:
                        scores[i] = getattr(chain_results[str(int(residue))], self.metric)
                    except KeyError as e:
                        pass

        return scores