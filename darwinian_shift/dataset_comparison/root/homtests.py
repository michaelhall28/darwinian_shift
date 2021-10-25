"""
This is just a wrapper for the homogeneity tests from Jakub Trusina et al 2020
The tests are written for ROOT, and the code for the tests is available from http://gams.fjfi.cvut.cz/homtests

Install ROOT (https://root.cern/install/) and then run
> root homtests.C+
on the command line to produce the homtests_C.so file used here.
"""
import os
import array
import numpy as np
import pandas as pd
from collections import namedtuple
import ROOT
from darwinian_shift.utils import sort_multiple_arrays_using_one

DIR_PATH = os.path.dirname(os.path.realpath(__file__))

ROOT.gSystem.Load(os.path.join(DIR_PATH, "homtests_C.so"))

HOMTEST_RESULT = namedtuple('HOMTEST_RESULT', ('statistic', 'pvalue'))

def root_homtest(arr1, weights1, arr2, weights2):
    """
    A wrapper for the ROOT code.
    :param arr1:
    :param weights1:
    :param arr2:
    :param weights2:
    :return: dictionary of statistics and pvalues.
    """
    ROOT.gErrorIgnoreLevel = 1001  # Do not print the warnings from the ROOT files
    if len(arr1) <= 2 or len(arr2) <= 2:
        raise ValueError('root_homtest: Each data set must contain contain more than two data points.')
    # The values must be sorted
    arr1, weights1 = sort_multiple_arrays_using_one(arr1, weights1)
    arr2, weights2 = sort_multiple_arrays_using_one(arr2, weights2)

    # Cannot use arrays that are slices/views of a larger array (which is what sort_multiple_arrays_using_one returns).
    # Take copies so the arrays passed to the ROOT function are not pointing to some other array
    arr1 = arr1.copy().astype(float)
    arr2 = arr2.copy().astype(float)
    weights1 = weights1.copy().astype(float)
    weights2 = weights2.copy().astype(float)

    pvalues = ROOT.WHomogeneityTest(len(arr1), arr1, weights1, len(arr2), arr2, weights2, "")  # returns p-values
    pvalues = np.frombuffer(pvalues, count=3)
    stats = ROOT.WHomogeneityTest(len(arr1), arr1, weights1, len(arr2), arr2, weights2, "M")  # returns statistics
    stats = np.frombuffer(stats, count=3)

    tests = ['KS', 'CVM', 'AD']
    return {tests[i]: HOMTEST_RESULT(stats[i], pvalues[i]) for i in range(3)}


def get_positions_and_weights(section, spectrum=None, position_col='residue', value_col=None):
    """
    Get the positions/values and weighting (the inverse of the mutation rate) for each observed mutation.
    For the tests the relative weighting is the important thing, so the mutation rates do not have to be scaled first.
    :param section:
    :param spectrum:
    :param position_col: Used for grouping the mutations. E.g. if it is residue, the weighting for a mutation will be
    based on the total expected mutation rate for the residue.
    :param value_col: If None, the "value" returned for each mutation is its value from the position column. Otherwise,
    it is the value in the value_col.
    :return:
    """
    if spectrum is None:
        spectrum = section.project.spectra[0]  # Use the first spectrum set up for the section
    if value_col is None:
        value_col = position_col
        obs = section.observed_mutations[[position_col]]  #  Reduce columns to make sure spectrum comes from null
    else:
        obs = section.observed_mutations[
            [position_col, value_col]]  #  Reduce columns to make sure spectrum comes from null

    null = section.null_mutations
    spectrum_rate_col = spectrum.rate_column

    agg_null = null.groupby(position_col).agg(sum)[[spectrum.rate_column ]]
    muts = pd.merge(agg_null, obs, how='right', left_index=True, right_on=position_col)

    muts['weight'] = 1 / muts[spectrum_rate_col]
    muts = muts.sort_values(position_col)
    return muts[value_col].values, muts['weight'].values


def homtest_sections(section1, section2, spectrum1=None, spectrum2=None, position_col='residue', use_weights=True):
    """
    Compare the distribution of mutations in two sections for the same protein.
    :param section1:
    :param section2:
    :param spectrum:
    :param position_col:
    :param use_weights:
    :return:
    """
    positions1, weights1 = get_positions_and_weights(section1, spectrum1, position_col)
    positions2, weights2 = get_positions_and_weights(section2, spectrum2, position_col)
    if not use_weights:
        weights1 = np.ones(len(positions1))
        weights2 = np.ones(len(positions2))
    return root_homtest(positions1, weights1, positions2, weights2)


def root_chi2(arr1, weights1, arr2, weights2, bin_boundaries, verbose=False, allow_low_counts=False):
    """

    :param arr1:
    :param weights1:
    :param arr2:
    :param weights2:
    :param bin_boundaries:
    :param verbose:
    :param allow_low_counts: If False, raise a ValueError if not enough effective events in all bins
    :return: float. pvalue
    """
    ROOT.gErrorIgnoreLevel = 1001  # Do not print the warnings from the ROOT files
    bin_boundaries = np.array(bin_boundaries, dtype=float)
    num_bins = len(bin_boundaries) - 1
    h1 = ROOT.TH1F('h1', '1', num_bins, bin_boundaries)
    h2 = ROOT.TH1F('h2', '2', num_bins, bin_boundaries)

    h1.FillN(len(arr1), arr1, weights1)
    h2.FillN(len(arr2), arr2, weights2)
    if verbose:
        h1_uw = ROOT.TH1F('h1_uw', '1', num_bins, bin_boundaries)
        h2_uw = ROOT.TH1F('h2_uw', '2', num_bins, bin_boundaries)
        h1_uw.FillN(len(arr1), arr1, np.ones_like(arr1))
        h2_uw.FillN(len(arr2), arr2, np.ones_like(arr2))
        counts1 = [h1_uw.GetBinContent(n) for n in range(1, num_bins + 1)]
        counts2 = [h2_uw.GetBinContent(n) for n in range(1, num_bins + 1)]

        w_counts1 = [h1.GetBinContent(n) for n in range(1, num_bins + 1)]
        w_counts2 = [h2.GetBinContent(n) for n in range(1, num_bins + 1)]
        print('Counts1', counts1)
        print('Counts2', counts2)
        print('Weighted Counts1', w_counts1)
        print('Weighted Counts2', w_counts2)

    chi = array.array('d', [0])
    ndf = array.array('i', [0])
    igood = array.array('i', [0])
    resids = np.empty(num_bins)
    p = h1.Chi2TestX(h2, chi, ndf, igood, 'WW', resids)
    if verbose:
        print('chi', chi[0])
        print('ndf', ndf[0])
        print('igood', igood[0])
        print('Residuals', np.frombuffer(resids, count=num_bins))
    warning = igood[0]
    if warning > 0 and not allow_low_counts:
        """
        From the ROOT docs
        igood=0 - no problems
        igood=1'There is a bin in the 1st histogram with less then 10 effective number of events'
        igood=2'There is a bin in the 2nd histogram with less then 10 effective number of events'
        igood=3'when the conditions for igood=1 and igood=2 are satisfied'
        """
        if warning == 1:
            msg = 'There is a bin in the 1st histogram with less than 10 effective number of events'
        elif warning == 2:
            msg = 'There is a bin in the 2nd histogram with less than 10 effective number of events'
        elif warning == 3:
            msg = 'There are bins in both histograms with less than 10 effective number of events'
        raise ValueError(msg)
    if ndf[0] == 0:
        """
        Only one bin in the test. Cannot run. To be consistent with the allow_low_counts behaviour, raise an error 
        if allow_low_counts==false.
        If allow_low_counts=True, return p=1 
        """
        msg = 'Only one bin in the histograms has non-zero counts'
        if verbose:
            print(msg)
        if not allow_low_counts:
            raise ValueError(msg)
        p = 1

    return p


def chi2_test_sections(section1, section2, bin_boundaries, spectrum1=None, spectrum2=None,
                       position_col='residue', use_weights=True, verbose=False, allow_low_counts=False,
                       value_col=None):
    """
    Compare the distribution of mutations in two sections for the same protein.
    :param section1:
    :param section2:
    :param spectrum:
    :param position_col:
    :param use_weights:
    :return:
    """
    positions1, weights1 = get_positions_and_weights(section1, spectrum1, position_col, value_col=value_col)
    positions2, weights2 = get_positions_and_weights(section2, spectrum2, position_col, value_col=value_col)
    if not use_weights:
        weights1 = np.ones(len(positions1))
        weights2 = np.ones(len(positions2))
    return root_chi2(positions1, weights1, positions2, weights2, bin_boundaries, verbose=verbose,
                     allow_low_counts=allow_low_counts)


def chi2_test_window(section1, section2, start, end, verbose=True, allow_low_counts=False):
    """
    A shortcut function for testing the relative mutation count in a subsection of a gene.
    I.e. similar to just running chi2_test_sections, but the regions before and after the window are combined
    in a single bin.
    :param section1:
    :param section2:
    :param start:
    :param end:
    :param verbose:
    :param allow_low_counts:
    :return:
    """
    section1.null_mutations['in_window'] = ((section1.null_mutations['residue'] > start) &
                                          (section1.null_mutations['residue'] < end)).astype(float)
    section2.null_mutations['in_window'] = ((section2.null_mutations['residue'] > start) &
                                          (section2.null_mutations['residue'] < end)).astype(float)
    section1.observed_mutations['in_window'] = ((section1.observed_mutations['residue'] > start) &
                                              (section1.observed_mutations['residue'] < end)).astype(float)
    section2.observed_mutations['in_window'] = ((section2.observed_mutations['residue'] > start) &
                                              (section2.observed_mutations['residue'] < end)).astype(float)
    return chi2_test_sections(section1, section2, bin_boundaries=(-1, 0.5, 2),
                              position_col='residue', verbose=verbose, allow_low_counts=allow_low_counts,
                              value_col='in_window')