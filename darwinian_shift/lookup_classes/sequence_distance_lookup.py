import numpy as np


class SequenceDistanceLookup:
    """
    Score is the distance from a given selection of positions in the protein or gene sequence.
    Does not consider distance in 3D space, just the sequence
    Can use residue numbers ('residue'), cds positions ('cdspos'), or chromosomal positions ('pos') to calculate the distance.

    Requires the target key attribute to be defined for the section being analysed. The name of the attribute can be
    defined here, by default it is 'target_selection'.

    For example:
    d.run_section({
        'gene': GENESYMBOL,
        'target_selection': [1, 2, 30, 100]
    }

    """

    position_types = ('residue', 'cdspos', 'pos')

    def __init__(self, position_type='residue', boolean=False, target_key='target_selection', name=None):
        """

        :param position_type: Whether to measure distance and define target locations in terms of residues,
        cds positions, or chromosomal positions. The options are 'residue', 'cdspos' or 'pos'.
        :param boolean: If true, will test for mutation exactly on the target sites. If False (default), will
        test for the distance from the mutations to the nearest target site.
        :param target_key: The name of the section attribute which lists the target site.
        :param name: Name of the lookup to appear on plot axes.
        """
        self.boolean = boolean  # Return True/False for in the target rather than a distance from it
        if position_type not in self.position_types:
            raise TypeError('Must pick from {}'.format(self.position_types))
        self.position_type = position_type
        self.target_key = target_key
        if name is None and not self.boolean:
            self.name = self.position_type + ' distance'
        elif name is None:
            self.name = 'In target ' + self.position_type
        else:
            self.name = name  # Will appear on some plot axes

    def __call__(self, seq_object):
        target_selection = getattr(seq_object, self.target_key, None)
        if target_selection is None:
            raise ValueError('Target selection {} not defined for section {}'.format(self.target_key,
                                                                                     seq_object.section_id))
        return self._get_distance(seq_object.null_mutations, target_selection)

    def _get_distance_to_residue(self, target_selection, null_mutation):
        distances = np.abs(target_selection - null_mutation[self.position_type])
        min_distance = distances.min()
        if self.boolean:
            return min_distance == 0
        return min_distance

    def _get_distance(self, df, target_selection):
        target_selection = np.array(list(target_selection))
        return df.apply(lambda x: self._get_distance_to_residue(target_selection, x), axis=1).values.astype(float)