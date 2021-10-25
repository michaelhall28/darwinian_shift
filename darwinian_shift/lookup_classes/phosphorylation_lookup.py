from .sequence_distance_lookup import SequenceDistanceLookup
import pandas as pd

class PhosphorylationLookup(SequenceDistanceLookup):
    """
    Built to use the Phosphorylation_site_dataset downloaded from www.phosphosite.org,
    Hornbeck PV, Zhang B, Murray B, Kornhauser JM, Latham V, Skrzypek E PhosphoSitePlus, 2014:
        mutations, PTMs and recalibrations. Nucleic Acids Res. 2015 43:D512-20.

    For the file dated 17 Feb 21, may not work if the format has altered since.

    Passes the data to the SequenceDistanceLookup to measure distance of mutations from the phosphorylation sites.
    """

    def __init__(self, phos_file, species, boolean=False, name=None):
        """

        :param phos_file: File path to the downloaded file from www.phosphosite.org
        :param species: String. For selecting the correct organism from the data.
        :param boolean: If true, will test for mutation exactly on the phosphorylation sites. If False (default), will
        test for the distance from the mutations to the nearest phosphorylation site.
        :param name: Name of the lookup to appear on plot axes.
        """
        phos_data = pd.read_csv(phos_file, sep="\t", skiprows=2)
        self.phos_data = phos_data[phos_data['ORGANISM'] == species].copy()
        if len(self.phos_data) == 0:
            print('No phosphorylation data found. Species available: {}'.format(phos_data['ORGANISM'].unique()))
        self.phos_data['residue'] = self.phos_data['MOD_RSD'].apply(lambda x: x.split("-")[0][1:]).astype(int)

        # Attributes for the sequence distance lookup
        self.boolean = boolean  # Return True/False for in the target rather than a distance from it
        self.position_type = 'residue'
        if name is None and not self.boolean:
            self.name = 'Distance to phosphorylation site'
        elif name is None:
            self.name = 'On phosphorylation site'
        else:
            self.name = name

    def __call__(self, seq_object):
        gene_pho = self.phos_data[self.phos_data['GENE'] == seq_object.gene]
        if getattr(seq_object, "protein", None) is not None:
            # Some genes have more than one associated protein.
            # If specified using a kwarg of the Section, then filter using the protein name
            gene_pho = gene_pho[gene_pho['PROTEIN'] == seq_object.protein]

        if len(gene_pho['PROTEIN'].unique()) > 1:
            msg = "More than one protein ({}) associated with gene {}.".format(gene_pho['PROTEIN'].unique(),
                                                                               seq_object.gene)
            msg += " Specify which protein to use by adding .protein attribute to the Section object."
            raise ValueError(msg)

        target_selection = gene_pho['residue']
        return self._get_distance(seq_object.null_mutations, target_selection)