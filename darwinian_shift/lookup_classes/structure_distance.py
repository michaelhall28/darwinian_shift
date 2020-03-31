import os
from darwinian_shift.utils import get_pdb_positions
import pandas as pd
import MDAnalysis
from MDAnalysis.lib.distances import distance_array


class StructureDistanceLookup:
    # Use the distance from a given selection of atoms in the pdb file.
    # Takes a selection string as used by MDAnalysis (same as VMD etc), e.g. 'protein and segid A and resid 3 4 5'
    # The selection string must be correct for the pdb file (which may not match the protein residue numbers)
    # Uses the alpha-carbons of the mutated residues for the distance calculations.
    # The targets do not have to be alpha-carbons - depends entirely on the given string.
    # Where alternative locations for an alpha-carbon exist in the pdb file, will use the average position.
    def __init__(self, pdb_directory=None, sifts_directory=None, download_sifts=None, boolean=False,
                 target_key='target_selection', name=None):
        self.pdb_directory = pdb_directory
        self.sifts_directory = sifts_directory
        self.download_sifts = download_sifts
        self.boolean = boolean  # Return True/False for in the target rather than a distance from it
        self.target_key = target_key
        if name is None and self.boolean:
            self.name = 'In target residues'
        elif name is None:
            self.name = '3D distance'
        else:
            self.name = name  # Will appear on some plot axes

    def setup_project(self, project):
        if self.pdb_directory is None:
            self.pdb_directory = project.pdb_directory
        if self.sifts_directory is None:
            self.sifts_directory = project.sifts_directory
        if self.download_sifts is None:
            self.download_sifts = project.download_sifts

    def __call__(self, seq_object):
        target_selection = getattr(seq_object, self.target_key, None)
        if target_selection is None:
            raise ValueError('Target selection {} not defined for section {}'.format(self.target_key,
                                                                                     seq_object.section_id))
        return self._get_distance(seq_object.null_mutations, seq_object.pdb_id, seq_object.pdb_chain,
                                  target_selection)

    def _get_distance(self, df, pdb_id, pdb_chain, target_selection):
        try:
            u = MDAnalysis.Universe(os.path.join(self.pdb_directory, "{}.pdb.gz".format(pdb_id.lower())))
        except FileNotFoundError as e:
            u = MDAnalysis.Universe(os.path.join(self.pdb_directory, "{}.pdb".format(pdb_id.lower())))
        target_residues = u.select_atoms(target_selection)

        # Use SIFTS to align with the null residues
        null_residues_original = sorted(df['residue'].unique())
        null_residues = get_pdb_positions(null_residues_original, pdb_id, pdb_chain, self.sifts_directory,
                                          self.download_sifts)
        null_residues_original = null_residues['residue']  # Just take the cases that are defined in the structure
        pdb_positions = null_residues['pdb position'].values
        null_residues = u.select_atoms('protein and segid {} and name CA and resid {}'.format(pdb_chain,
                                                                                              " ".join(
                                                                                                  ['{:.0f}'.format(r)
                                                                                                   for r in
                                                                                                   pdb_positions])))

        arr = distance_array(target_residues.positions, null_residues.positions)
        distances = arr.min(axis=0)
        distance_df = pd.DataFrame({'pdb_residue': null_residues.resids, 'distance': distances})
        original_resids = pd.DataFrame({'residue': null_residues_original, 'pdb_residue': pdb_positions})
        merge_distance_df = pd.merge(distance_df, original_resids, on='pdb_residue', how='left')
        averaged_distance_df = merge_distance_df.groupby('residue').agg('mean')

        # Apply back to the null
        merge_df = pd.merge(df, averaged_distance_df, on='residue', how='left')
        if self.boolean:
            return (merge_df['distance'] == 0).values.astype(float)
        return merge_df['distance'].values