import os
from darwinian_shift.utils import get_pdb_positions
import pandas as pd
import MDAnalysis
from MDAnalysis.lib.distances import distance_array
from darwinian_shift.utils import download_pdb_file


class StructureDistanceLookup:
    """
    Use the distance from a given selection of atoms in the pdb file.
    Takes a selection string as used by MDAnalysis (same as VMD etc), e.g. 'protein and segid A and resid 3 4 5'
    The selection string must be correct for the pdb file (which may not match the protein residue numbers)
    Uses the alpha-carbons of the mutated residues for the distance calculations if distance_to_alpha_carbons=True.
    Otherwise, finds the shortest distance of any atom in the residue
    The targets do not have to be alpha-carbons - depends entirely on the given string.
    Where alternative locations for an alpha-carbon exist in the pdb file, will use the average position.

    If the pdb file does not already exist in the pdb_directory (current directory by default), will try to download it.
    """
    def __init__(self, pdb_directory=None, sifts_directory=None, download_sifts=None, boolean=False,
                 target_key='target_selection', name=None, distance_to_alpha_carbons=False):
        self.pdb_directory = pdb_directory
        self.sifts_directory = sifts_directory
        self.download_sifts = download_sifts
        self.boolean = boolean  # Return True/False for in the target rather than a distance from it
        self.target_key = target_key
        self.distance_to_alpha_carbons = distance_to_alpha_carbons
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

    def _get_pdb_file(self, pdb_id):
        # First look for the gzipped pdb file
        fp = os.path.join(self.pdb_directory, "{}.pdb.gz".format(pdb_id.lower()))
        if os.path.isfile(fp):
            return fp
        else:
            # If the compressed file doesn't exist, look for the uncompressed pdb file
            fp2 = os.path.join(self.pdb_directory, "{}.pdb".format(pdb_id.lower()))
            if os.path.isfile(fp):
                return fp2
            else:
                # If that doesn't exist, try to download the file
                download_pdb_file(pdb_id, self.pdb_directory, file_type='pdb.gz')
                return fp


    def _get_distance(self, df, pdb_id, pdb_chain, target_selection):
        fp = self._get_pdb_file(pdb_id)
        u = MDAnalysis.Universe(fp)
        target_residues = u.select_atoms(target_selection)
        if len(target_residues) == 0:
            # The requested target was not found in the pdb file
            raise ValueError('Target selection "{}" did not include any atoms in structure {}'.format(target_selection,
                                                                                                      pdb_id))

        # Use SIFTS to align with the null residues
        null_residues_original = sorted(df['residue'].unique())
        null_residues = get_pdb_positions(null_residues_original, pdb_id, pdb_chain, self.sifts_directory,
                                          self.download_sifts)
        null_residues_original = null_residues['residue']  # Just take the cases that are defined in the structure
        pdb_positions = null_residues['pdb position'].values

        null_selection = 'protein and segid {} and resid {}'.format(pdb_chain, " ".join(['{:.0f}'.format(r)
                                                                                         for r in pdb_positions]))
        if self.distance_to_alpha_carbons:
            null_selection += " and name CA"
        null_atoms = u.select_atoms(null_selection)

        arr = distance_array(target_residues.positions, null_atoms.positions)
        distances = arr.min(axis=0)
        # Get the distance from the closest atom in each residue (will just be the alpha-carbon if distance_to_alpha_carbons=True)
        distance_df = pd.DataFrame({'pdb_residue': null_atoms.resids, 'distance': distances}).groupby('pdb_residue').agg(min).reset_index()
        original_resids = pd.DataFrame({'residue': null_residues_original, 'pdb_residue': pdb_positions})
        merge_distance_df = pd.merge(distance_df, original_resids, on='pdb_residue', how='left')

        # Apply back to the null
        merge_df = pd.merge(df, merge_distance_df, on='residue', how='left')
        if self.boolean:
            return (merge_df['distance'] == 0).values.astype(float)
        return merge_df['distance'].values