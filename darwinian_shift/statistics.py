from scipy.stats import kstest, chisquare, binom_test, norm
from bisect import bisect_left, bisect_right
import numpy as np
import pandas as pd
import matplotlib.pylab as plt
from darwinian_shift.utils import sort_multiple_arrays_using_one
import math


class PermutationTest:
    # Runs a permutation test for a given statistic like the median or mean.
    def __init__(self, stat_function=np.mean, num_permutations=10000, name=None,
                 testing_random_seed=None):
        self.stat_function = stat_function
        self.num_permutations = num_permutations
        if name is None:
            self.name = stat_function.__name__ + "_perm"
        else:
            self.name = name
        self.testing_random_seed=testing_random_seed

    def set_testing_random_seed(self, s):
        self.testing_random_seed = s

    def __call__(self, seq_object, spectrum, plot=False, show_plot=True):
        res = permutation_test(seq_object.null_scores, seq_object.null_mutations[spectrum.rate_column].values,
                               seq_object.observed_values,
                               self.stat_function, self.num_permutations, plot=plot,
                               plot_title=" - ".join(['Monte Carlo Test', self.name, spectrum.name]),
                               testing_random_seed=self.testing_random_seed, show_plot=show_plot)
        return {"_".join([self.name, spectrum.name, k]): v for k, v in res.items()}


class CDFPermutationTest:
    # Runs a permutation test using the cdf of the null distribution instead of the raw value
    # May be more robust to outlier values
    def __init__(self, num_permutations=10000, name='CDF_perm', testing_random_seed=None):
        self.num_permutations = num_permutations
        self.name = name
        self.testing_random_seed=testing_random_seed

    def set_testing_random_seed(self, s):
        self.testing_random_seed = s

    def __call__(self, seq_object, spectrum, plot=False, show_plot=True):
        res = permutation_test_cdf_sum(seq_object.null_scores, seq_object.null_mutations[spectrum.rate_column].values,
                                       seq_object.observed_values, self.num_permutations, plot=plot,
                                       plot_title='Monte Carlo Test - CDF sum - ' + spectrum.name,
                                       testing_random_seed=self.testing_random_seed, show_plot=show_plot)
        return {"_".join([self.name, spectrum.name, k]): v for k, v in res.items()}


class CDFZTest:
    # Using the central limit theorem to get a the normal distribution limit of the permutation test.
    def __init__(self, name='CDF_Z'):
        self.name = name

    def __call__(self, seq_object, spectrum, plot=False, show_plot=True):
        res = ztest_cdf_sum(seq_object.null_scores, seq_object.null_mutations[spectrum.rate_column].values,
                            seq_object.observed_values, plot=plot,
                            plot_title='Z-Test - CDF sum - ' + spectrum.name,
                            show_plot=show_plot)
        return {"_".join([self.name, spectrum.name, k]): v for k, v in res.items()}


class ChiSquareTest:
    # A non-directional test for differences in the distribution
    # Differences between the null and the observed distributions may indicate selection or inappropriate null assumptions
    # Selection detected here may not correlate with the chosen metric
    def __init__(self, bins=None, max_bins=10, min_exp_freq=5, name='chi_square', CI_num_samples=10000, CI_alpha=0.05):
        if isinstance(bins, tuple):
            self.bins = list(bins)   #Needs to be mutable
        else:
            self.bins = bins
        self.max_bins = max_bins
        self.min_exp_freq = min_exp_freq
        self.name = name

        # Num samples and alpha for the bootstrapped confidence intervals.
        self.CI_num_samples=CI_num_samples
        self.CI_alpha = CI_alpha

    def __call__(self, seq_object, spectrum, plot=False):
        res = binned_chisquare(
            null_scores=seq_object.null_scores, null_mut_rates=seq_object.null_mutations[spectrum.rate_column].values,
            observed_values=seq_object.observed_values, bins=self.bins,
            max_bins=self.max_bins, min_exp_freq=self.min_exp_freq,
            CI_sample_num=self.CI_num_samples, CI_alpha=self.CI_alpha
        )

        # if len(res['observed_counts']) == 2:
        #     # Run a binomial test too.
        #     # By default will be labelled as chi_square_binom..., but prevents confusion if multiple chi-square are run.
        #     binomial_res = binomial_test(res['expected_counts'], res['observed_counts'])
        #     res.update(binomial_res)

        return {"_".join([self.name, spectrum.name, k]): v for k, v in res.items()}


class BinomTest:
    """
    A wrapper for the Scipy binomial test.
    For cases with two values (e.g. on/not on an interface) or where the metric values can be split using a threshold
    By default, assumes the values are 1 and zero, and the threshold is set at 0.5
    """
    def __init__(self, threshold=0.5, name='binom', CI_num_samples=10000,
                 CI_alpha=0.05):
        self.threshold = threshold
        self.name = name

        # Num samples and alpha for the bootstrapped confidence intervals.
        self.CI_num_samples = CI_num_samples
        self.CI_alpha = CI_alpha

    def __call__(self, seq_object, spectrum, plot=False):
        res = binomial_test(
            null_scores=seq_object.null_scores,
            null_mut_rates=seq_object.null_mutations[spectrum.rate_column].values,
            observed_values=seq_object.observed_values, threshold=self.threshold,
            CI_sample_num=self.CI_num_samples, CI_alpha=self.CI_alpha
        )

        return {"_".join([self.name, spectrum.name, k]): v for k, v in res.items()}


class KSTest:
    # A non-directional test for differences in the distribution
    # Differences between the null and the observed distributions may indicate selection or inappropriate null assumptions
    # Selection detected here may not correlate with the chosen metric
    # Just for continuous distributions. Use repeat_proportion to check if the results may be valid.
    name = 'ks'
    def __call__(self, seq_object, spectrum, plot=False):
        res = ks_test(seq_object.null_scores, seq_object.null_mutations[spectrum.rate_column], seq_object.observed_values)
        return {"_".join([self.name, spectrum.name, k]): v for k, v in res.items()}


def calculate_repeat_proportion(null_values):
    num_unique = len(np.unique(null_values))
    num_total = len(null_values)
    repeat_prop = (num_total-num_unique)/num_total
    return repeat_prop

def get_vectorized_quantile_function(values, mut_rates):
    # Not used anywhere
    mut_rates = np.array(mut_rates)
    weights = mut_rates / mut_rates.sum()
    values, weights = sort_multiple_arrays_using_one(values, weights)
    cumsum = np.cumsum(weights)

    def quantiler(q):
        idx1 = bisect_left(cumsum, q)
        idx2 = bisect_right(cumsum, q)
        if idx1 == len(cumsum):
            q1 = 1
        else:
            q1 = values[idx1]
        if idx2 == 0:
            q2 = 0
        else:
            q2 = values[idx2 - 1]
        return (q1 + q2) / 2

    quantiler = np.vectorize(quantiler)
    return quantiler

def get_median(values, mut_rates):
    # This may be very slightly off in some cases due to floating point errors
    # Generally because the floating point errors in the cumsum do not hit 0.5 exactly. Can catch some cases.
    if len(values) == 1:
        return values[0]
    mut_rates = np.array(mut_rates)
    weights = mut_rates / mut_rates.sum()
    values, weights = sort_multiple_arrays_using_one(values, weights)
    cumsum = np.cumsum(weights)
    idx = bisect_left(cumsum, 0.5)
    if cumsum[idx] == 0.5:
        # 0.5 is in the cumsum.
        # The median (to be consistent with np.median) is the average of the two values around the centre
        return (values[idx] + values[idx+1])/2
    elif idx < len(values) - 1 and not math.isclose(cumsum[idx], 0.5) and math.isclose(cumsum[idx+1], 0.5):
        # Floating point error means bisect has missed the 0.5 value
        idx += 1
        return (values[idx] + values[idx+1])/2
    elif idx > 0 and not math.isclose(cumsum[idx], 0.5) and math.isclose(cumsum[idx-1], 0.5):
        # Floating point error means bisect has missed the 0.5 value
        idx -= 1
        return (values[idx] + values[idx+1])/2
    else:
        return values[idx]

def get_cdf(values, mut_rates):
    # Order from low to high.
    # Weight by the mutational spectrum.
    # Take the cumulative sum
    # Only appropriate if (effectively) continuous data
    mut_rates = np.array(mut_rates)
    weights = mut_rates / mut_rates.sum()
    values, weights = sort_multiple_arrays_using_one(values, weights)
    cumsum = np.cumsum(weights)
    def cdf(e):
        idx = bisect_right(values, e)
        if idx == 0:
            return 0
        return cumsum[idx-1]
    cdf = np.vectorize(cdf)
    return cdf

def ks_test(all_gene_values, mut_rates, observed_values):
    if len(all_gene_values) > 0 and len(observed_values) > 0:
        cdf = get_cdf(all_gene_values, mut_rates)
        k = kstest(observed_values, cdf)
        results = {
            'statistic':k.statistic,
            'pvalue': k.pvalue
        }
    else:
        results = {
            'statistic': np.nan,
            'pvalue': 1
        }
    return results

# Functions for permutation tests
def get_samples_from_mutational_spectrum(values, mut_rates, num_per_sample=1000000, num_samples=1):
    mut_rates = np.array(mut_rates)
    weights = mut_rates / mut_rates.sum()
    samples = np.repeat(values, np.random.multinomial(num_per_sample * num_samples, weights))
    np.random.shuffle(samples)
    return samples.reshape(num_samples, num_per_sample)


def permutation_p_value(num_permutations, perm_metrics, obs_metric, rerr=1e-7):
    """

    :param num_permutations:
    :param perm_metrics:
    :param obs_metric:
    :param rerr: Relative compensation for errors in summing of floating points. The values from the permutations will
    be compared to the observed value * (1±rerr) (the more conservative case for each tail).
    :return:
    """
    num_smaller_or_equal = bisect_right(perm_metrics, obs_metric*(1+rerr)) + 1  # +1 to include the observation itself
    num_larger_or_equal = num_permutations - bisect_left(perm_metrics,
                                                         obs_metric*(1-rerr)) + 1 # +1 to include the observation itself
    pvalue = min(num_smaller_or_equal, num_larger_or_equal) / (num_permutations + 1) * 2  # Two-tailed p-value
    pvalue = min(pvalue, 1)
    return pvalue, num_smaller_or_equal, num_larger_or_equal


def permutation_test(exp_values, mut_rates, observed_values, metric_function, num_permutations, plot=False,
                     num_plot_bins=100, plot_title=None, testing_random_seed=None, show_plot=True, rerr=1e-7):
    """
    Use a chosen metric e.g. np.median, np.mean, np.sum etc for the permutation test.
    :param exp_values:
    :param mut_rates:
    :param observed_values:
    :param metric_function:
    :param num_permutations:
    :param plot:
    :param num_plot_bins:
    :param testing_random_seed:
    :param rerr:  Relative compensation for errors in summing of floating points. The values from the permutations will
    be compared to the observed value * (1±rerr) (the more conservative case for each tail).
    :return:
    """
    if testing_random_seed is not None:
        np.random.seed(testing_random_seed)

    if plot_title is None:
        plot_title = 'Permutation Test - {}'.format(metric_function.__name__)

    num_obs = len(observed_values)
    obs_metric = metric_function(observed_values)
    samples = get_samples_from_mutational_spectrum(exp_values, mut_rates, num_obs, num_permutations)
    perm_metrics = np.sort(metric_function(samples, axis=1))
    if plot:
        bins = np.linspace(min(min(perm_metrics), obs_metric), max(max(perm_metrics), obs_metric), num_plot_bins)
        plt.hist(perm_metrics, bins=bins)
        ylim = plt.gca().get_ylim()
        plt.vlines(obs_metric, 0, ylim[1])
        plt.ylim(ylim)
        plt.title(plot_title)
        plt.xlabel(metric_function.__name__)
        plt.ylabel('Frequency')
        if show_plot:
            plt.show()

    pvalue, num_smaller_or_equal, num_larger_or_equal = permutation_p_value(num_permutations, perm_metrics,
                                                                            obs_metric, rerr)

    results = {
        'observed': obs_metric,
        'null_mean': np.mean(perm_metrics),
        'null_median': np.median(perm_metrics),
        'pvalue': pvalue,
        'num_smaller_or_equal': num_smaller_or_equal,
        'num_larger_or_equal': num_larger_or_equal
    }
    return results


def permutation_test_cdf_sum(exp_values, mut_rates, observed_values, num_permutations, plot=False,
                               num_plot_bins=100, plot_title='CDF sum', testing_random_seed=None, show_plot=True,
                             rerr=1e-7):
    """
    Use the sum of the cdf values for the permutation test.
    For tied values, using the average of the cdf values.
    :param exp_values:
    :param mut_rates:
    :param observed_values:
    :param num_permutations:
    :param plot:
    :param num_plot_bins:
    :param testing_random_seed:
    :param rerr:  Relative compensation for errors in summing of floating points. The values from the permutations will
    be compared to the observed value * (1±rerr) (the more conservative case for each tail).
    :return:
    """
    if testing_random_seed is not None:
        np.random.seed(testing_random_seed)

    num_obs = len(observed_values)
    sorted_exp_values, sorted_mut_rates = sort_multiple_arrays_using_one(exp_values, mut_rates)

    # Reduce to unique values. May reduce length by a lot for discrete distributions
    # Method from https://stackoverflow.com/a/43094244
    sorted_mut_rates = np.array([np.sum(m) for m in np.split(sorted_mut_rates,
                                                             np.cumsum(np.unique(sorted_exp_values,
                                                                                 return_counts=True)[1]))[:-1]])
    sorted_exp_values = np.unique(sorted_exp_values)

    weights = sorted_mut_rates / sorted_mut_rates.sum()
    # For repeated values, use the mid-point of the cdf jump.
    # Will centre the null results on 0.5, even in skewed discrete cases.
    cumsum = np.cumsum(weights) - weights / 2

    observed_cumsum = cumsum[np.searchsorted(sorted_exp_values, observed_values)]

    samples = get_samples_from_mutational_spectrum(cumsum, sorted_mut_rates, num_obs, num_permutations)
    samples = np.concatenate([samples, np.array(observed_cumsum, ndmin=2)])  # Add the observed values at the end
    perm_metrics = samples.sum(axis=1)
    perm_metrics, obs_metric = perm_metrics[:-1], perm_metrics[-1]
    perm_metrics.sort()
    if plot:
        bins = np.linspace(min(min(perm_metrics), obs_metric), max(max(perm_metrics), obs_metric), num_plot_bins)
        plt.hist(perm_metrics, bins=bins, density=True)
        ylim = plt.gca().get_ylim()
        plt.vlines(obs_metric, 0, ylim[1])
        # plt.vlines(len(observed_values)*0.5, 0, ylim[1], linestyles='dashed')
        plt.ylim(ylim)
        plt.title(plot_title)
        plt.xlabel("CDF sum")
        plt.ylabel('Frequency')
        if show_plot:
            plt.show()

    pvalue, num_smaller_or_equal, num_larger_or_equal = permutation_p_value(num_permutations, perm_metrics,
                                                                            obs_metric, rerr)

    results = {
        'num_smaller_or_equal': num_smaller_or_equal,
        'num_larger_or_equal': num_larger_or_equal,
        'pvalue': pvalue, 'cdf_mean': obs_metric/num_obs
    }
    return results


def z_pvalue(observed_value, loc, scale):
    p_low = norm.cdf(observed_value, loc=loc, scale=scale)
    p_high = norm.sf(observed_value, loc=loc, scale=scale)
    return min(p_low, p_high) * 2  # Multiply by two to get two-tailed p-value


def get_cdf_var(cumsum, weights):
    return (cumsum**2*weights).sum() - (cumsum*weights).sum()**2


def ztest_cdf_sum(exp_values, mut_rates, observed_values, plot=False,
                  plot_title='CDF sum', show_plot=True):
    """
    Use the sum of the cdf values
    The central limit theorem to get the normal distribution limit of the permutation test
    For tied values, using the average of the cdf values.
    :param exp_values:
    :param mut_rates:
    :param observed_values:
    :param num_permutations:
    :param plot:
    :param num_plot_bins:
    :param testing_random_seed:
    :return:
    """
    num_obs = len(observed_values)
    sorted_exp_values, sorted_mut_rates = sort_multiple_arrays_using_one(exp_values, mut_rates)

    # Reduce to unique values. May reduce length by a lot for discrete distributions
    # Method from https://stackoverflow.com/a/43094244
    sorted_mut_rates = np.array([np.sum(m) for m in np.split(sorted_mut_rates,
                                                             np.cumsum(np.unique(sorted_exp_values,
                                                                                 return_counts=True)[1]))[:-1]])
    sorted_exp_values = np.unique(sorted_exp_values)

    weights = sorted_mut_rates / sorted_mut_rates.sum()
    # For repeated values, use the mid-point of the cdf jump.
    # Will centre the null results on 0.5, even in skewed discrete cases.
    cumsum = np.cumsum(weights) - weights / 2

    cdf_var = get_cdf_var(cumsum, weights)

    observed_cumsum = cumsum[np.searchsorted(sorted_exp_values, observed_values)]

    obs_metric = observed_cumsum.sum()

    # Parameters for the normal distribution
    loc = num_obs * 0.5
    scale = np.sqrt(num_obs) * np.sqrt(cdf_var)

    if plot:
        x = np.linspace(*norm.interval(0.9999, loc=loc, scale=scale), 1000)
        plt.plot(x, norm.pdf(x, loc=loc, scale=scale), 'r--', label='Expected')
        ylim = plt.gca().get_ylim()
        plt.vlines(obs_metric, 0, ylim[1], label='Observed')
        plt.ylim([0, ylim[1]])
        plt.title(plot_title)
        plt.xlabel("CDF sum")
        plt.ylabel('Frequency')
        plt.legend()
        if show_plot:
            plt.show()

    pvalue = z_pvalue(obs_metric, loc=loc, scale=scale)

    results = {
        'pvalue': pvalue, 'cdf_mean': obs_metric / num_obs
    }
    return results

# Functions for chi-squared test
def get_bins_and_expected_counts(values, mut_rates, max_bins, total_count, min_exp_freq=5):
    """
    Use bin definitions like numpy histogram.
    e.g. bins = [1, 2, 3] means first (and middle) bins half-open like [1, 2), last is closed like [2, 3]

    This will not create the optimum evenly sized categories (if that is important). Will start from the least and
    create the smallest contiguous categories until the end, when there may be a bigger or smaller bin.

    This assumes that there are no conditions on the observed counts for the chi-squared test to be valid.
    """

    mut_rates = np.array(mut_rates)
    weights = mut_rates / mut_rates.sum() * total_count
    values, weights = sort_multiple_arrays_using_one(values, weights)
    d = pd.DataFrame(np.array([values, weights]).T).groupby(0).agg({1: np.sum})

    values, weights = d.index.values, d[1].values

    if len(values) == 1:  # Can't run the test. Only a single unique value.
        bins = [values[0] - 1, values[0] + 1]
        expected_frequencies = [total_count]
    else:
        if len(values) < max_bins:
            # If fewer unique values than max bins, only combine if required to reach minimum frequency
            min_weight_per_bin = min_exp_freq
        else:
            # Use a rough method to get even(ish) bins
            min_weight_per_bin = max(1 / max_bins * total_count, min_exp_freq)  # Get initial average size
            if min_weight_per_bin > min_exp_freq:
                # Some frequent single values can be put in own bin, and reduce required size of other bins to reach average.
                ordered_weights = sorted(weights, reverse=True)
                single_value_bins = 0
                remaining_count = total_count
                for ww in ordered_weights:
                    if ww > min_weight_per_bin:  # Any value that would be a large enough bin on its own
                        remaining_count -= ww
                        single_value_bins += 1
                    else:
                        # Recalculate Weight per multi-value bin:
                        min_weight_per_bin = max(1 / (max_bins - single_value_bins) * remaining_count, min_exp_freq)
                        # Check if any remaining single values are larger than this new min bin weight
                        if ww > min_weight_per_bin:
                            remaining_count -= ww
                            single_value_bins += 1
                        else:
                            break


        min_diff = np.diff(values).min()
        bins = [values[0]]
        expected_frequencies = []
        bin_weight = 0
        for v, w in zip(values, weights):
            if bin_weight >= min_weight_per_bin:  # Do this before adding weight since left bound is closed, right is open
                bins.append(v)
                expected_frequencies.append(bin_weight)
                bin_weight = 0
            bin_weight += w

        if len(expected_frequencies)  == 0:
            # One bin at most.
            bins.append(v)  # Last value is end of the only bin
            expected_frequencies.append(bin_weight)
        else:
            # Have some remaining weight. Either add to previous bin or include in separate bin.
            a = bin_weight + expected_frequencies[-1]  # Weight if added to previous bin
            a_diff = a - min_weight_per_bin  # Difference from ideal weight if added to previous bin
            b_diff = min_weight_per_bin - bin_weight  # Difference from ideal weight if in its own bin.
            if bin_weight >= min_weight_per_bin or (a_diff > b_diff and bin_weight > min_exp_freq):  # add to own bin.
                # If the last value is a bin on its own, have to make sure not used as right bound on previous bin
                if bins[-1] == values[-1]:
                    bins[-1] -= min_diff / 2
                bins.append(v)  # Add the last bound
                expected_frequencies.append(bin_weight)
            else:
                # If the last value is not a bin on its own, make sure it is the last bound and add weight to the last bin
                bins[-1] = v
                expected_frequencies[-1] += bin_weight

    return bins, expected_frequencies


def get_expected_frequency_for_bins(null_scores, mut_rates, total_count, bins):
    bins = bins.copy()   # Make sure original is not edited
    mut_rates = np.array(mut_rates)
    weights = mut_rates / mut_rates.sum() * total_count
    values, weights = sort_multiple_arrays_using_one(null_scores, weights)
    d = pd.DataFrame(np.array([values, weights]).T, columns=['values', 'weights'])
    d = d[d['values'] <= bins[-1]]  # Make sure we exclude values beyond the highest bin.
    bins[-1] += 0.001 # Cheat to make it act like right inclusive.
    d['bin'] = pd.cut(d['values'], bins=bins, right=False, labels=False)
    mutated_bins = d['bin'].unique()
    expected_counts = pd.DataFrame(d).groupby('bin').agg({'weights': np.sum})['weights'].values
    expected_counts_all_bins = np.zeros(len(bins)-1)
    c = 0
    for i in range(len(bins)-1):
        if i in mutated_bins:
            expected_counts_all_bins[i] = expected_counts[c]
            c += 1
    return expected_counts_all_bins


def get_intervals_from_random_sample_bin_counts(samples, bins, num_samples=1000, alpha=0.05):
    sample_bin_counts = np.empty((num_samples, len(bins)-1))
    for i, s in enumerate(samples):
        sample_bin_counts[i] = np.histogram(s, bins=bins)[0]
    CI_high = np.quantile(sample_bin_counts, 1-alpha/2, axis=0)
    CI_low = np.quantile(sample_bin_counts, alpha/2, axis=0)
    return CI_low, CI_high


def bootstrap_binned_confint_method(observed_values, bins, num_samples=10000, alpha=0.05):
    samples = np.random.choice(observed_values, size=(num_samples, len(observed_values)), replace=True)
    return get_intervals_from_random_sample_bin_counts(samples, bins, num_samples, alpha)


def get_null_binned_confint(null_scores, null_mut_rates, num_obs, bins, num_samples=10000, alpha=0.05):
    null_samples = get_samples_from_mutational_spectrum(null_scores, null_mut_rates, num_obs, num_samples)
    return get_intervals_from_random_sample_bin_counts(null_samples, bins, num_samples, alpha)


def binned_chisquare(null_scores, null_mut_rates, observed_values, bins=None, max_bins=10,
                     min_exp_freq=5, CI_sample_num=10000, CI_alpha=0.05):
    num_obs = len(observed_values)
    if bins is None:
        bins, expected_frequencies = get_bins_and_expected_counts(null_scores, null_mut_rates, max_bins, num_obs,
                                                                  min_exp_freq=min_exp_freq)
    else:
        expected_frequencies = get_expected_frequency_for_bins(null_scores, null_mut_rates, num_obs, bins)

    observed_counts = np.histogram(observed_values, bins=bins)[0]
    low_count_warning = False
    if len(observed_counts) < 2:
        low_count_warning = True
        pvalue = 1
        statistic = np.nan
    else:
        if min(expected_frequencies) < 5:
            # print('Counts may be too low for chi square. Min expected = {}'.format(min(expected_frequencies)))
            low_count_warning = True
        chi_results = chisquare(observed_counts, expected_frequencies)
        pvalue = chi_results.pvalue
        statistic = chi_results.statistic

    # Confidence interval calculations
    # For the null hypothesis, take 1000 samples from the null and get a 95% confidence interval for each bin
    expected_CI_low, expected_CI_high = get_null_binned_confint(null_scores, null_mut_rates, num_obs, bins,
                                                                num_samples=CI_sample_num, alpha=CI_alpha)

    # For the observed data, use bootstrapping
    observed_CI_low, observed_CI_high = bootstrap_binned_confint_method(observed_values, bins,
                                                                        num_samples=CI_sample_num, alpha=CI_alpha)

    results = {
        'statistic': statistic, 'pvalue': pvalue, 'low_count_warning': low_count_warning,
        'bins': bins, 'observed_counts': observed_counts, 'expected_counts': expected_frequencies,
        'expected_CI_high': expected_CI_high,
        'expected_CI_low': expected_CI_low,
        'observed_CI_low': observed_CI_low,
        'observed_CI_high': observed_CI_high
    }

    return results


def binomial_test(null_scores, null_mut_rates, observed_values, threshold, CI_sample_num=10000, CI_alpha=0.05):
    total_rate = null_mut_rates.sum()
    rate_high = null_mut_rates[null_scores>threshold].sum()/total_rate   # Expected proportion of mutations in high bin
    count_high = (observed_values > threshold).sum()
    total_count = len(observed_values)

    # Confidence interval calculations
    # For the null hypothesis, take 1000 samples from the null and get a 95% confidence interval for each bin
    bins = [-np.inf, threshold, np.inf]
    expected_CI_low, expected_CI_high = get_null_binned_confint(null_scores, null_mut_rates, len(observed_values), bins,
                                                                num_samples=CI_sample_num, alpha=CI_alpha)

    # For the observed data, use bootstrapping
    observed_CI_low, observed_CI_high = bootstrap_binned_confint_method(observed_values, bins,
                                                                        num_samples=CI_sample_num, alpha=CI_alpha)

    results = {
        'pvalue': binom_test(x=count_high, n=total_count, p=rate_high),
        'expected_proportion': rate_high,
        'observed_proportion': count_high / total_count,
        'expected_count': rate_high*total_count,
        'observed_count': count_high,
        'threshold': threshold,
        'expected_CI_high': expected_CI_high[1],
        'expected_CI_low': expected_CI_low[1],
        'observed_CI_low': observed_CI_low[1],
        'observed_CI_high': observed_CI_high[1]
    }
    return results


# Functions for plotting intervals on CDF plots
def get_cdf_from_sample(sample, xvals):
    sorted_values = sorted(sample)
    xvals_idx = np.searchsorted(sorted_values, xvals, side='right')  # Match to the sample values
    cdf_vals = np.arange(0, len(sorted_values) + 1) / len(sorted_values)
    yvals = cdf_vals[xvals_idx]
    return yvals


def get_intervals_from_sample_cdfs(samples, xvals, num_samples=1000, alpha=0.05):
    sample_cdfs = np.empty((num_samples, len(xvals)))
    for i, s in enumerate(samples):
        sample_cdfs[i] = get_cdf_from_sample(s, xvals)
    CI_high = np.quantile(sample_cdfs, 1-alpha/2, axis=0)
    CI_low = np.quantile(sample_cdfs, alpha/2, axis=0)
    return CI_low, CI_high


def bootstrap_cdf_confint_method(observed_values, xvals, num_samples=10000, alpha=0.05):
    samples = np.random.choice(observed_values, size=(num_samples, len(observed_values)), replace=True)
    return get_intervals_from_sample_cdfs(samples, xvals, num_samples, alpha)


def get_null_cdf_confint(null_scores, null_mut_rates, num_obs, xvals, num_samples=10000, alpha=0.05):
    null_samples = get_samples_from_mutational_spectrum(null_scores, null_mut_rates, num_obs, num_samples)
    return get_intervals_from_sample_cdfs(null_samples, xvals, num_samples, alpha)