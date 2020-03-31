from prody import *
import pandas as pd
import numpy as np
import subprocess
import os
from darwinian_shift.utils import get_sifts_alignment_for_chain


class ProDyLookup:
    # Also DSSP since this can be parsed with Prody.

    _options = (
        'sq_flucts_GNM',  # Squared fluctuations from GNM model
        'sq_flucts_ANM',  # Squared fluctuations from ANM model
        'mean_stiffness',  # Mean mechanical stiffness
        'PRS_effectiveness',  # Effectiveness from Perturbation Response Scanning
        'PRS_sensitivity',  # Sensitivity from Perturbation Response Scanning
        'solvent_accessibility'  # From DSSP
    )

    def __init__(self, metric="sq_flucts_GNM", exclude_ends=0, pdb_directory=None, dssp_directory='.',
                 sifts_directory=None, download_sifts=None, quiet=True, name=None):
        if metric in self._options:
            self.metric = metric
            self.metric_function = getattr(self, "_" + metric)  # Returns a function
        else:
            raise ValueError('Must pick metric from {}'.format(self._options))
        self.exclude_ends = exclude_ends
        if quiet:
            prody.LOGGER.verbosity = 'warning'
        else:
            prody.LOGGER.verbosity = 'none'
        self.pdb_directory = pdb_directory
        self.dssp_directory = dssp_directory
        self.sifts_directory = sifts_directory
        self.download_sifts = download_sifts
        if name is None:
            self.name = 'ProDy ' + self.metric
        else:
            self.name = name  # Will appear on some plot axes


    def setup_project(self, project):
        if self.pdb_directory is None:
            self.pdb_directory = project.pdb_directory
            pathPDBFolder(self.pdb_directory)
        if self.sifts_directory is None:
            self.sifts_directory = project.sifts_directory
        if self.download_sifts is None:
            self.download_sifts = project.download_sifts

    def __call__(self, seq_object):
        return self._get_scores(seq_object.pdb_id, seq_object.pdb_chain, seq_object.null_mutations)

    def _get_anm(self, ca):
        anm = ANM()
        anm.buildHessian(ca)
        anm.calcModes()
        return anm

    def _sq_flucts_GNM(self, pdb_id, pdb_chain, ca):
        # Calculates the squared fluctuation for each residue using an ANM
        gnm = GNM()
        gnm.buildKirchhoff(ca)
        gnm.calcModes()
        return calcSqFlucts(gnm)

    def _sq_flucts_ANM(self, pdb_id, pdb_chain, ca):
        # Calculates the squared fluctuation for each residue using an ANM
        anm = self._get_anm(ca)
        return calcSqFlucts(anm)

    def _mean_stiffness(self, pdb_id, pdb_chain, ca):
        anm = self._get_anm(ca)
        stiffness = calcMechStiff(anm, ca)
        return np.array([np.mean(stiffness, axis=0)])[0]  # The mean stiffness of each residue

    def _PRS(self, ca):
        anm = self._get_anm(ca)
        prs_mat, effectiveness, sensitivity = calcPerturbResponse(anm)
        return effectiveness, sensitivity

    def _PRS_effectiveness(self, pdb_id, pdb_chain, ca):
        return self._PRS(ca)[0]

    def _PRS_sensitivity(self, pdb_id, pdb_chain, ca):
        return self._PRS(ca)[1]

    def _solvent_accessibility(self, pdb_id, pdb_chain, ca):
        pdb = parsePDB(pdb_id)  # Read in the whole structure
        dssp_path = os.path.join(self.dssp_directory, "{}.dssp".format(pdb_id.lower()))
        if not os.path.exists(dssp_path):
            cmd = "wget ftp://ftp.cmbi.ru.nl/pub/molbio/data/dssp/{}.dssp -O {}".format(pdb_id.lower(),
                                                                                        dssp_path)
            subprocess.run(cmd, shell=True, check=True)
        dssp = parseDSSP(dssp_path, pdb)
        return dssp[pdb_chain].ca.getData('dssp_acc')  # The solvent accessibility for each residue

    def _get_scores(self, pdb_id, pdb_chain, df):
        if pdb_id is None:
            raise ValueError('Can only run with a pdb file')
        if pdb_chain is None:
            raise ValueError('Must specify which chain in a pdb file')
        sifts = get_sifts_alignment_for_chain(pdb_id, pdb_chain, self.sifts_directory, self.download_sifts)
        ca = parsePDB(pdb_id, chain=pdb_chain, subset='calpha')  # Read in the alpha carbons
        if sifts is None:
            scores = None
        else:
            values = self.metric_function(pdb_id, pdb_chain, ca)
            resnums = ca.getResnums()  # These are residue numbers according to the PDB file.
            vr = pd.DataFrame({'score': values, 'pdb position': resnums})
            # Use SIFTS to covert PDB positions to uniprot protein positions
            sifts = pd.merge(sifts, vr, on='pdb position', how='left')
            if self.exclude_ends > 0:
                sifts = sifts.iloc[self.exclude_ends:]
                sifts = sifts.iloc[:-self.exclude_ends]
            merge_df = pd.merge(df, sifts, left_on='residue', right_on='uniprot position', how='left')
            scores = merge_df['score'].values
        return scores