import numpy as np

class ORLookup:
    """
    Class to combine other lookups using simple thresholds.
    Will return 1 for a mutation if any thresholds are broken, 0 otherwise.
    """

    def __init__(self, lookups, thresholds, directions, include_missing_values=False, name='OR'):
        """

        :param lookups: List of the Lookup objects
        :param thresholds: List of the thresholds corresponding to each lookup
        :param directions: Direction which breaks the threshold. 1 if above, -1 if below.
        E.g. if testing the case that mutations are either close to an active site
        (looking for distance *below* a threshold) or increase ∆∆G (looking for values *above* a threshold), then use
        directions=[-1, 1]
        :param include_missing_values: Some mutations may not have a metric from one of the lookups
        (e.g. nonsense mutations will not have a ∆∆G value). If included, these will be deemed to have not broken the
        threshold.
        :param name: Name of the lookup to appear on plot axes.
        """
        assert len(lookups) == len(thresholds), "Length of lookups must be the same as the length of the thresholds"
        self.lookups = lookups
        self.thresholds = thresholds
        if directions is not None:
            assert len(lookups) == len(directions), "Length of lookups must be the same as the length of the directions"
            assert all([i in [1, -1] for i in directions]), "Directions should be a list comprising only 1s and -1s"
            self.directions = directions
        else:  # Assume checking for values which exceed the threshold
            self.directions = [1] * len(lookups)
        self.include_missing_values = include_missing_values
        self.name = name  # Will appear on some plot axes

    def __call__(self, seq_object):
        results = np.empty((len(self.lookups), len(seq_object.null_mutations)))
        for i, (lookup, threshold, direction) in enumerate(zip(self.lookups, self.thresholds, self.directions)):
            score = lookup(seq_object)
            # Mutations where any individual lookup does not return a value
            missing_values = np.isnan(score)

            if direction == 1:  # True if above the threshold
                results[i] = score > threshold
            else:  # True if below the threshold
                results[i] = score < threshold

            if self.include_missing_values:
                results[i, missing_values] = 0  # If not given a score, it does not cross the threshold
            else:
                # Replace missing values with nan
                results[i, missing_values] = np.nan

        combined_results = results.any(axis=0).astype(float)

        if not self.include_missing_values:
            # Mutations where any individual lookup does not return a value
            missing_values = np.any(np.isnan(results), axis=0)
            # Replace these with nan
            combined_results[missing_values] = np.nan

        return combined_results


class ANDLookup:
    """
    Class to combine other lookups using simple thresholds.
    Will return 1 for a mutation if all thresholds are broken, 0 otherwise.
    """
    def __init__(self, lookups, thresholds, directions, include_missing_values=False, name='AND'):
        """

        :param lookups: List of the Lookup objects
        :param thresholds: List of the thresholds corresponding to each lookup
        :param directions: Direction which breaks the threshold. 1 if above, -1 if below.
        E.g. if testing the case that mutations are close to an active site
        (looking for distance *below* a threshold) and increase ∆∆G (looking for values *above* a threshold), then use
        directions=[-1, 1]
        :param include_missing_values: Some mutations may not have a metric from one of the lookups
        (e.g. nonsense mutations will not have a ∆∆G value). If included, these will be deemed to have not broken the
        threshold.
        :param name: Name of the lookup to appear on plot axes.
        """
        assert len(lookups) == len(thresholds), "Length of lookups must be the same as the length of the thresholds"
        self.lookups = lookups
        self.thresholds = thresholds
        if directions is not None:
            assert len(lookups) == len(directions), "Length of lookups must be the same as the length of the directions"
            assert all([i in [1, -1] for i in directions]), "Directions should be a list comprising only 1s and -1s"
            self.directions = directions
        else:  # Assume checking for values which exceed the threshold
            self.directions = [1] * len(lookups)
        self.include_missing_values = include_missing_values
        self.name = name  # Will appear on some plot axes

    def __call__(self, seq_object):
        results = np.empty((len(self.lookups), len(seq_object.null_mutations)))
        for i, (lookup, threshold, direction) in enumerate(zip(self.lookups, self.thresholds, self.directions)):
            score = lookup(seq_object)
            # Mutations where any individual lookup does not return a value
            missing_values = np.isnan(score)

            if direction == 1:  # True if above the threshold
                results[i] = score > threshold
            else:  # True if below the threshold
                results[i] = score < threshold

            if self.include_missing_values:
                results[i, missing_values] = 0  # If not given a score, it does not cross the threshold
            else:
                # Replace missing values with nan
                results[i, missing_values] = np.nan

        combined_results = results.all(axis=0).astype(float)
        if not self.include_missing_values:
            # Mutations where any individual lookup does not return a value
            missing_values = np.any(np.isnan(results), axis=0)
            # Replace these with nan
            combined_results[missing_values] = np.nan

        return combined_results


class MutationExclusionLookup:
    """
    Use one lookup to exclude mutations from the test using a second lookup
    For example, test whether mutations not on an interface are destabilising.
    """
    def __init__(self, lookup, exclusion_lookup, exclusion_threshold, exclusion_direction=1, name='Mutation Exclusion'):
        """

        :param lookup: Lookup class to run the statistical tests with.
        :param exclusion_lookup: Exclude any mutations from the statistical test that cross a threshold from this lookup
        :param exclusion_direction: Direction which breaks the threshold. 1 if above, -1 if below.
        E.g. if wanting to exclude mutations with a high ∆∆G, use exclusion_direction=1. If wanting to exclude mutations
        with a low ∆∆G, use exclusion_direction=-1
        :param name: Name of the lookup to appear on plot axes.
        """
        self.lookup = lookup
        self.exclusion_lookup = exclusion_lookup
        self.exclusion_threshold = exclusion_threshold
        self.exclusion_direction = exclusion_direction
        self.name = name  # Will appear on some plot axes

    def __call__(self, seq_object):
        exclusion_scores = self.exclusion_lookup(seq_object)
        missing_values = np.isnan(exclusion_scores)
        if self.exclusion_direction == 1:  # True if above the threshold
            excluded_muts = exclusion_scores > self.exclusion_threshold
        else:  # True if below the threshold
            assert self.exclusion_direction == -1, "exclusion_direction must be either 1 or -1"
            excluded_muts = exclusion_scores < self.exclusion_threshold
        excluded_muts[missing_values] = 0  # Make sure mutations without a score from the exclusion_lookup are not excluded
        lookup_scores = self.lookup(seq_object)
        assert len(exclusion_scores) == len(lookup_scores)
        lookup_scores[excluded_muts] = np.nan

        return lookup_scores