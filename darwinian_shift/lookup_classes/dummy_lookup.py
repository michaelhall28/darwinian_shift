import numpy as np


class DummyValuesPosition:
    """
    Can be useful for testing. May detect selection in cases of hotspots.
    Assigns a random value per position
    """

    def __init__(self, data, random_function=np.random.random, testing_random_seed=None, name='Dummy Values'):
        """

        :param data: pandas dataframe of mutations. Used to generate random values for all locations in the vicinity
        of the data.
        :param random_function: Must be based on a numpy random function and just take the number of values required
        as the argument.
        :param testing_random_seed: A seed for numpy so ensure results are reproducible.
        :param name: Name of the lookup to appear on plot axes.
        """
        if testing_random_seed is not None:
            np.random.seed(testing_random_seed)
        self.chrom_starts = {}
        self.res = {}
        for chrom, chrom_data in data.groupby('chr'):
            chrom = str(chrom)
            # Extra bases for the start and end of the gene
            self.chrom_starts[chrom] = chrom_data['pos'].min() - 10000

            self.res[chrom] = random_function(chrom_data['pos'].max() - self.chrom_starts[chrom] + 10000)
        self.name = name  # Will appear on some plot axes

    def __call__(self, seq_object):
        return self._get_scores(seq_object.chrom, seq_object.null_mutations)

    def _get_scores(self, chrom, df):
        min_pos = df['pos'].min()
        max_pos = df['pos'].max()
        res = self._get_score_for_range(chrom, min_pos, max_pos)
        scores = df.apply(lambda x: res[x['pos'] - min_pos], axis=1)
        return scores

    def _get_score_for_range(self, chrom, min_pos, max_pos):
        return [self.res[chrom][pos - self.chrom_starts[chrom]] for pos in range(min_pos, max_pos + 1)]


class DummyValuesRandom:
    """
    Can be useful for testing. May detect selection in cases of hotspots.
    Random value for each mutation in a section/sequence.
    """

    def __init__(self, random_function=np.random.random, testing_random_seed=None, name='Dummy Values'):
        """

        :param random_function: Must be based on a numpy random function and just take the number of values required
        as the argument. Only works for numpy random numbers.
        :param testing_random_seed: A seed for numpy so ensure results are reproducible.
        :param name: Name of the lookup to appear on plot axes.
        """
        self.f = random_function
        self.testing_random_seed=testing_random_seed
        self.name = name  # Will appear on some plot axes

    def __call__(self, seq_object):
        return self._get_scores(seq_object.null_mutations)

    def _get_scores(self, df):
        if self.testing_random_seed is not None:
            np.random.seed(self.testing_random_seed)
        return self.f(len(df))


class DummyValuesFixed:
    """
    Can be useful for testing.
    All mutations given the same value
    """
    def __init__(self, fixed_value=0, name='Dummy Value'):
        self.value = fixed_value
        self.name = name  # Will appear on some plot axes

    def __call__(self, seq_object):
        return self._get_scores(seq_object.null_mutations)

    def _get_scores(self, df):
        return np.full(len(df), self.value, dtype=float)