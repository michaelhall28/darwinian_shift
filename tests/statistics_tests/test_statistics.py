import pytest
import os
import pickle
from darwinian_shift.statistics import *

FILE_DIR = os.path.dirname(os.path.realpath(__file__))

@pytest.fixture
def exp_values():
    np.random.seed(0)
    return np.random.random(1000)

@pytest.fixture
def mut_rates():
    np.random.seed(1)
    return np.random.random(1000)

@pytest.fixture
def observed_values(exp_values):
    np.random.seed(2)
    return np.random.choice(sorted(exp_values)[100:], size=100)

# Test the functions not the classes

def test_calculate_repeat_proportion():
    values = np.concatenate([
        np.arange(100), np.arange(50), np.arange(20)
    ])
    res = calculate_repeat_proportion(values)
    assert res == 70/170

def test_get_median():
    # Some very simple tests with even weights.
    res = get_median([1], [1])
    assert res == 1

    res = get_median([1, 1], [1, 1])
    assert res == 1

    res = get_median([1, 0], [1, 1])
    assert res == 0.5

    res = get_median([1, 2, 3], [1, 1, 1])
    assert res == 2

    res = get_median([1, 1, 2], [1, 1, 1])
    assert res == 1

    res = get_median([1, 1, 2, 2], [1, 1, 1, 1])
    assert res == 1.5

    np.random.seed(0)
    v = np.random.random(101)
    res1 = get_median(v, np.ones(len(v)))
    assert res1 == np.median(v)

    np.random.seed(0)
    v = np.random.random(100)
    res1 = get_median(v, np.ones(len(v)))
    # known problem that floating point errors mean function does not correctly average two values either side of median
    assert res1 == 0.4663107728563063
    # assert res1 == np.median(v) # Will not work in this case (np.median(v) = 0.46748098725200393)

    # Some tests with simple weights so median easy to calculate
    res = get_median([1, 2], [1, 2])  # Like calculating median of [1, 2, 2]
    assert res == 2

    res = get_median([1, 2, 3], [0.1, 0.2, 0.3])  # Like calculating median of [1, 2, 2, 3, 3, 3]
    assert res == 2.5

    # More difficult tests. Just checking it is consistent.
    np.random.seed(0)
    v = np.random.random(101)
    w = np.random.random(101)
    res1 = get_median(v, w)
    assert res1 == 0.5232480534666997

    np.random.seed(0)
    v = np.random.random(100)
    w = np.random.random(100)
    res1 = get_median(v, w)
    assert res1 == 0.46147936225293185


def test_get_cdf():
    # Some very simple tests with even weights.
    cdf_func = get_cdf([1], [1])
    assert cdf_func(1) == 1

    cdf_func = get_cdf([1, 1], [1, 1])
    np.testing.assert_array_equal(cdf_func([0, 1, 2]), [0, 1, 1])

    cdf_func = get_cdf([1, 0], [1, 1])
    np.testing.assert_array_equal(cdf_func([0, 0.5, 1, 2]), [0.5, 0.5, 1, 1])

    cdf_func = get_cdf([1, 2, 3], [1, 1, 1])
    np.testing.assert_array_equal(cdf_func([1.5, 1, 2, 3]), [1/3, 1/3, 2/3, 1])

    cdf_func = get_cdf([1, 1, 2], [1, 1, 1])
    np.testing.assert_array_equal(cdf_func([1.5, 1, 2, 3]), [2 / 3, 2 / 3, 1, 1])

    cdf_func = get_cdf([1, 1, 2, 2], [1, 1, 1, 1])
    np.testing.assert_array_equal(cdf_func([1.5, 1, 2]), [0.5, 0.5, 1])

    np.random.seed(0)
    v = np.random.random(101)
    cdf_func = get_cdf(v, np.ones(len(v)))
    v2 = [0.1, 0.2, 0.3, 0.4, 0.5, 0, 1]
    np.testing.assert_array_almost_equal(cdf_func(v2), [0.11881188118811879, 0.2475247524752476,
                                                 0.33663366336633677, 0.40594059405940613,
                                                 0.5049504950495052,  0,
                                                 1])

    # Test with simple weights so cdf easy to calculate
    cdf_func = get_cdf([1, 2, 3], [0.1, 0.2, 0.3])  # Like calculating median of [1, 2, 2, 3, 3, 3]
    np.testing.assert_array_almost_equal(cdf_func([1.5, 1, 2, 3]), [1/6, 1/6, 3/6, 1])

    # # More difficult tests. Just checking it is consistent.
    np.random.seed(0)
    v = np.random.random(101)
    w = np.random.random(101)
    cdf_func = get_cdf(v, w)
    v2 = [0.1, 0.2, 0.3, 0.4, 0.5, 0, 1]
    np.testing.assert_array_almost_equal(cdf_func(v2),[0.1164912, 0.25953564, 0.35902817, 0.41183393, 0.4899966,  0, 1])


def test_ks_test(exp_values, mut_rates, observed_values):
    np.random.seed(0)
    k = ks_test(exp_values, mut_rates, observed_values)
    assert k['statistic'] == 0.12733016521331045
    assert k['pvalue'] ==  0.07141968809198758


def test_get_samples_from_mutational_spectrum(exp_values, mut_rates):
    samples = get_samples_from_mutational_spectrum(exp_values, mut_rates, num_per_sample=101, num_samples=11)
    # output new test file. Do not uncomment unless results have changed and confident new results are correct
    # pickle.dump(samples, open(os.path.join(FILE_DIR, 'reference_samples_results.pickle'), 'wb'))

    expected_res = pickle.load(open(os.path.join(FILE_DIR, 'reference_samples_results.pickle'),'rb'))

    np.testing.assert_array_almost_equal(samples, expected_res)


def test_permutation(exp_values, mut_rates, observed_values):
    for metric_function in [np.mean, np.median]:
        res = permutation_test(exp_values, mut_rates, observed_values, metric_function, 1000, plot=False,
                               testing_random_seed=1)

        # output new test file. Do not uncomment unless results have changed and confident new results are correct
        # pickle.dump(res, open(os.path.join(FILE_DIR, 'reference_perm_{}_results.pickle'.format(metric_function.__name__)),
        #                         'wb'))

        expected_res = pickle.load(open(os.path.join(FILE_DIR,
                                                     'reference_perm_{}_results.pickle'.format(metric_function.__name__)),
                                        'rb'))

        assert res == expected_res


def test_permutation_cdf(exp_values, mut_rates, observed_values):
    res = permutation_test_cdf_sum(exp_values, mut_rates, observed_values, 1000, plot=False,
                               testing_random_seed=1)

    # output new test file. Do not uncomment unless results have changed and confident new results are correct
    # pickle.dump(res, open(os.path.join(FILE_DIR, 'reference_perm_cdf_results.pickle'), 'wb'))

    expected_res = pickle.load(open(os.path.join(FILE_DIR, 'reference_perm_cdf_results.pickle'), 'rb'))

    assert res == expected_res

def test_get_bins_and_expected_counts(exp_values, mut_rates):
    bins, expected_frequencies = get_bins_and_expected_counts(exp_values, mut_rates, max_bins=10,
                                                              total_count=101, min_exp_freq=5)

    np.testing.assert_array_almost_equal(bins, [0.0005459648969956543, 0.10204481074802807, 0.2097499496556351,
                                                0.29288856502701466, 0.39725674716804205, 0.47837030703998806,
                                                0.5864101661863267, 0.6964824307014501, 0.7992025873523917,
                                                0.9185464511903499, 0.9998085781169653])

    np.testing.assert_array_almost_equal(expected_frequencies, [10.240758488868384, 10.161631629074323,
                                                                10.16546606683974, 10.102286220224913,
                                                                10.112529230672934, 10.149850518146103,
                                                                10.205877896402528, 10.25128133621094,
                                                                10.176127843005194, 9.434190770554949])


def test_get_expected_frequency_for_bins(exp_values, mut_rates):
    bins =  [0.0005459648969956543, 0.10204481074802807, 0.2097499496556351,
                                                0.29288856502701466, 0.39725674716804205, 0.47837030703998806,
                                                0.5864101661863267, 0.6964824307014501, 0.7992025873523917,
                                                0.9185464511903499, 0.9998085781169653]

    expected_counts = get_expected_frequency_for_bins(exp_values, mut_rates, 101, bins)

    np.testing.assert_array_almost_equal(expected_counts, [10.240758488868384, 10.161631629074323,
                                                                10.16546606683974, 10.102286220224913,
                                                                10.112529230672934, 10.149850518146103,
                                                                10.205877896402528, 10.25128133621094,
                                                                10.176127843005194, 9.434190770554949])

def test_binned_chisquare(exp_values, mut_rates, observed_values):
    res = binned_chisquare(exp_values, mut_rates, observed_values, bins=None, max_bins=10,
                     min_exp_freq=5)

    # output new test file. Do not uncomment unless results have changed and confident new results are correct
    # pickle.dump(res, open(os.path.join(FILE_DIR, 'reference_chi_results1.pickle'), 'wb'))

    expected_res = pickle.load(open(os.path.join(FILE_DIR, 'reference_chi_results1.pickle'), 'rb'))

    assert res.keys() == expected_res.keys()
    for k, v in res.items():
        if isinstance(v, np.ndarray):
            np.testing.assert_array_almost_equal(v, expected_res[k])
        else:
            assert v == expected_res[k]

    bins = [0.0005459648969956543, 0.10204481074802807, 0.2097499496556351,
            0.29288856502701466, 0.39725674716804205, 0.47837030703998806,
            0.5864101661863267, 0.6964824307014501, 0.7992025873523917,
            0.9185464511903499, 0.9998085781169653]
    res = binned_chisquare(exp_values, mut_rates, observed_values, bins=bins)

    # output new test file. Do not uncomment unless results have changed and confident new results are correct
    # pickle.dump(res, open(os.path.join(FILE_DIR, 'reference_chi_results2.pickle'), 'wb'))

    expected_res = pickle.load(open(os.path.join(FILE_DIR, 'reference_chi_results2.pickle'), 'rb'))

    assert res.keys() == expected_res.keys()
    for k, v in res.items():
        if isinstance(v, np.ndarray):
            np.testing.assert_array_almost_equal(v, expected_res[k])
        else:
            assert v == expected_res[k]


def test_binned_chisquare_binom(exp_values, mut_rates, observed_values):
    # Make chi-square with two bins so
    res = binned_chisquare(exp_values, mut_rates, observed_values, bins=None, max_bins=2,
                     min_exp_freq=5)

    # output new test file. Do not uncomment unless results have changed and confident new results are correct
    # pickle.dump(res, open(os.path.join(FILE_DIR, 'reference_chi_results3.pickle'), 'wb'))

    expected_res = pickle.load(open(os.path.join(FILE_DIR, 'reference_chi_results3.pickle'), 'rb'))

    assert res.keys() == expected_res.keys()
    for k, v in res.items():
        if 'binom' in k:
            print(k, v)
        if isinstance(v, np.ndarray):
            np.testing.assert_array_almost_equal(v, expected_res[k])
        else:
            assert v == expected_res[k]


def test_z_cdf(exp_values, mut_rates, observed_values):
    res = ztest_cdf_sum(exp_values, mut_rates, observed_values, plot=False)

    # output new test file. Do not uncomment unless results have changed and confident new results are correct
    # pickle.dump(res, open(os.path.join(FILE_DIR, 'reference_z_cdf_results.pickle'), 'wb'))

    expected_res = pickle.load(open(os.path.join(FILE_DIR, 'reference_z_cdf_results.pickle'), 'rb'))

    assert res == expected_res