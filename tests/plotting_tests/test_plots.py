import pytest
import matplotlib.pyplot as plt
import numpy as np
from darwinian_shift import DarwinianShift, EvenMutationalSpectrum, GlobalKmerSpectrum, TranscriptKmerSpectrum
from darwinian_shift import MonteCarloTest, CDFMonteCarloTest, ChiSquareTest, BinomTest
from darwinian_shift.additional_functions import get_bins_for_uniprot_features
from darwinian_shift.lookup_classes import DummyValuesRandom, UniprotLookup
from darwinian_shift import plot_scatter_two_scores
from tests.conftest import MUTATION_DATA_FILE, EXON_FILE, REFERENCE_FASTA_FILE, TEST_DATA_DIR

# Colours seem a bit messed up, but this will still test if the plots are consistent and can run.
pytestmark = pytest.mark.slow

@pytest.fixture
def proj():
    np.random.seed(0)
    d = DarwinianShift(data=MUTATION_DATA_FILE,
                       exon_file=EXON_FILE,
                       reference_fasta=REFERENCE_FASTA_FILE,
                       lookup=DummyValuesRandom(random_function=np.random.random, testing_random_seed=0),
                       statistics=[CDFMonteCarloTest(testing_random_seed=0), ChiSquareTest()],
                       spectra=(EvenMutationalSpectrum(),
                                TranscriptKmerSpectrum(k=1),
                                GlobalKmerSpectrum(k=3))
                       )
    d.run_all()
    return d

@pytest.fixture
def seq(proj):
    s = proj.make_section({'transcript_id': 'ENST00000263388'})
    s.apply_scores()
    return s

@pytest.fixture
def seq_bool(proj):
    def bool_random(num):
        return np.random.binomial(1, 0.5, num)
    proj_bool = proj.change_lookup(DummyValuesRandom(random_function=bool_random, testing_random_seed=0))
    s = proj_bool.make_section({'transcript_id': 'ENST00000263388'})
    s.apply_scores()
    return s

@pytest.fixture
def seq_pdb():
    np.random.seed(0)
    d = DarwinianShift(data=MUTATION_DATA_FILE,
                       exon_file=EXON_FILE,
                       reference_fasta=REFERENCE_FASTA_FILE,
                       lookup=DummyValuesRandom(random_function=np.random.random, testing_random_seed=0),
                       statistics=[CDFMonteCarloTest(testing_random_seed=0), ChiSquareTest()],
                       spectra=(EvenMutationalSpectrum(),
                                TranscriptKmerSpectrum(k=1),
                                GlobalKmerSpectrum(k=3)),
                       pdb_directory=TEST_DATA_DIR,
                       sifts_directory=TEST_DATA_DIR,
                       download_sifts=False
                       )
    s = d.make_section(dict(transcript_id='ENST00000263388', pdb_id='4ZLP', pdb_chain='A', start=1378, end=1640))
    s.apply_scores()
    return s


## Tests for the multi-gene summary plots
@pytest.mark.mpl_image_compare(filename='volcano.png')
def test_volcano(proj):
    fig = plt.figure()
    proj.volcano_plot(sig_col='CDF_MC_glob_k3_qvalue', shift_col='CDF_MC_glob_k3_cdf_mean', colours=['C0', 'C3'],
                      qcutoff=0.05, shift_cutoff_low=0.6, shift_cutoff_high=0.65)
    return fig

@pytest.mark.mpl_image_compare(filename='volcano_colour_genes.png')
def test_volcano_by_gene(proj):
    fig = plt.figure()
    proj.volcano_plot_colour_by_gene(genes=['NOTCH3', 'KEAP1', 'AKT2'],
                                     sig_col='CDF_MC_glob_k3_qvalue', shift_col='CDF_MC_glob_k3_cdf_mean'
                                     )
    return fig

@pytest.mark.mpl_image_compare(filename='results_distribution.png')
def test_result_distribution(proj):
    fig = plt.figure()
    proj.plot_result_distribution('median_shift_glob_k3')
    return fig


## Tests for the single gene/pdb plots
@pytest.mark.mpl_image_compare(filename='scatter1.png')
def test_scatter1(seq):
    return seq.plot_scatter(show_plot=False, return_fig=True)


@pytest.mark.mpl_image_compare(filename='scatter2.png')
def test_scatter2(seq):
    return seq.plot_scatter('symlog', 20, show_plot=False, return_fig=True)

@pytest.mark.mpl_image_compare(filename='scatter3.png')
def test_scatter3(seq):
    return seq.plot_scatter(show_plot=False, return_fig=True, xlim=(1000, 1040), show_residues=True)

@pytest.mark.mpl_image_compare(filename='violinplot1.png')
def test_violinplot1(seq):
    np.random.seed(0)
    return seq.plot_violinplot(show_plot=False, return_fig=True)

@pytest.mark.mpl_image_compare(filename='violinplot2.png')
def test_violinplot2(seq):
    np.random.seed(0)
    return seq.plot_violinplot(plot_scale='symlog', violinplot_bw=0.1, show_plot=False, return_fig=True)

@pytest.mark.mpl_image_compare(filename='boxplot.png')
def test_boxplot(seq):
    np.random.seed(0)
    return seq.plot_boxplot(show_plot=False, return_fig=True)

@pytest.mark.mpl_image_compare(filename='cdfs.png')
def test_cdfs(seq):
    return seq.plot_cdfs(show_plot=False, return_fig=True)

@pytest.mark.mpl_image_compare(filename='cdfs2.png')
def test_cdfs2(seq):
    return seq.plot_cdfs(show_plot=False, return_fig=True, show_CI=True)

@pytest.mark.mpl_image_compare(filename='chi_sq.png')
def test_chi_sq(seq):
    return seq.plot_chi_sq_counts(show_plot=False, return_fig=True, show_CI=False)

@pytest.mark.mpl_image_compare(filename='chi_sq2.png')
def test_chi_sq2(seq):
    return seq.plot_chi_sq_counts(show_plot=False, return_fig=True, show_CI=True)

@pytest.mark.mpl_image_compare(filename='chi_sq3.png')
def test_chi_sq3(seq_bool):
    return seq_bool.plot_binned_counts(show_plot=False, return_fig=True, show_CI=True)

@pytest.mark.mpl_image_compare(filename='binom.png')
def test_binom(seq_bool):
    return seq_bool.plot_binomial(show_plot=False, return_fig=True, show_CI=True, binom_test=BinomTest(),
                                  figsize=(15, 5))

@pytest.mark.mpl_image_compare(filename='aa.png')
def test_aa(seq):
    return seq.plot_aa_abundance(show_plot=False, return_fig=True, figsize=(15, 5))

@pytest.mark.mpl_image_compare(filename='sliding_window.png')
def test_sliding_window(seq):
    return seq.plot_sliding_window(show_plot=False, return_fig=True)

@pytest.mark.mpl_image_compare(filename='sliding_window2.png')
def test_sliding_window2(seq):
    return seq.plot_sliding_window(show_plot=False, return_fig=True, xlim=(1000, 1040), show_residues=True)

@pytest.mark.mpl_image_compare(filename='sliding_3D_window.png')
def test_3D_sliding_window(seq_pdb):
    return seq_pdb.plot_sliding_3D_window(show_plot=False, show_arcs=True, arc_scale=0.001, return_fig=True)

@pytest.mark.mpl_image_compare(filename='sliding_score_window.png')
def test_sliding_score_window(seq_pdb):
    return seq_pdb.plot_sliding_window_totalled_score(show_plot=False, return_fig=True)


@pytest.mark.mpl_image_compare(filename='mutation_rate_vs_score_scatter.png')
def test_mutation_rate_vs_score_scatter(seq_pdb):
    return seq_pdb.plot_mutation_rate_scatter(show_plot=False, return_fig=True, unmutated_marker_size=30,
                                              base_marker_size=50, figsize=(15, 5),
                                              mutations_to_annotate=seq_pdb.observed_mutations.iloc[:2],
                                              annotation_offset=(0.01, 0.05))

# Plots from the Monte Carlo tests
@pytest.mark.mpl_image_compare(filename='mc.png')
def test_MC(seq):
    fig = plt.figure()
    p = MonteCarloTest(testing_random_seed=1, num_draws=1000)
    p(seq, seq.project.spectra[0], plot=True, show_plot=False)
    return fig

@pytest.mark.mpl_image_compare(filename='cdf_mc.png')
def test_cdf_MC(seq):
    fig = plt.figure()
    p = CDFMonteCarloTest(testing_random_seed=1, num_draws=1000)
    p(seq, seq.project.spectra[0], plot=True, show_plot=False)
    return fig

@pytest.mark.mpl_image_compare(filename='scatter_two_scores.png')
def test_scatter_two_scores(proj, seq):
    d2 = proj.change_lookup(DummyValuesRandom(testing_random_seed=1, name='S2'))
    d3 = proj.change_lookup(DummyValuesRandom(testing_random_seed=2, name='S3'))
    d4 = proj.change_lookup(DummyValuesRandom(testing_random_seed=3, name='S4'))
    s2 = d2.run_transcript('ENST00000263388')
    s3 = d3.run_transcript('ENST00000263388')
    s4 = d4.run_transcript('ENST00000263388')
    return plot_scatter_two_scores(seq, s2, sections_for_colours=[s3, s4],
                                   score_regions_for_colours=[[0, 0.5], [0.8, 199]],
                                    score_region_colours=['C7', 'C8'],
                                   show_plot=False, return_fig=True)


@pytest.mark.mpl_image_compare(filename='scatter_two_scores2.png')
def test_scatter_two_scores2(proj, seq):
    s2 = proj.run_transcript('ENST00000263388', lookup=DummyValuesRandom(testing_random_seed=1, name='S2'))
    s3 = proj.run_transcript('ENST00000263388', lookup=DummyValuesRandom(testing_random_seed=2, name='S3'))
    s4 = proj.run_transcript('ENST00000263388', lookup=DummyValuesRandom(testing_random_seed=3, name='S4'))

    mut_lists = [
        ['S154N', 'A1613T'], ['Y529*', 'A1430A', 'R113*']
    ]
    mut_list_colours = ['C4', 'C5']
    mut_list_labels = ['List1', 'List3']

    return plot_scatter_two_scores(seq, s2, sections_for_colours=[s3, s4],
                                   score_regions_for_colours=[[0, 0.5], [0.8, 199]],
                                    score_region_colours=['C7', 'C8'],
                                   unobserved_alpha=0.1,
                                   show_observed_only=False,
                                   show_plot=False, return_fig=True, annotate_mutations=True,
                                   annotate_xregion=[0.1, 0.25], annotate_yregion=[0.2, 0.5],
                                   mut_lists_to_colour=mut_lists, mut_list_colours=mut_list_colours,
                                   mut_list_labels=mut_list_labels
                                   )


@pytest.mark.mpl_image_compare(filename='lollipop_plot.png')
def test_lollipop_plot(seq):
    return seq.plot_lollipop(xlim=(100, 800), return_fig=True)

@pytest.mark.mpl_image_compare(filename='bar_plot.png')
def test_bar_plot(seq):
    uniprot_lookup = UniprotLookup(uniprot_directory=TEST_DATA_DIR)
    transcript_uniprot = uniprot_lookup.get_uniprot_data('ENST00000263388')
    bins, types, descriptions = get_bins_for_uniprot_features(transcript_uniprot,
                                                              feature_types=('domain', 'repeat'),
                                                    min_gap=0, last_residue=3000)
    colours = []
    for t, d in zip(types, descriptions):
        if t == 'domain':
            if '1' in d:
                colours.append('C0')
            else:
                colours.append('C2')
        elif t == 'repeat':
            colours.append('C3')
        elif t is None and d is None:
            colours.append('C7')
    return seq.plot_bar_observations(binning_regions=bins, facecolour=colours, linewidth=1, return_fig=True)


@pytest.mark.mpl_image_compare(filename='bar_expected_rates.png')
def test_bar_expected_rates(seq):
    return seq.plot_expected_mutation_rates_for_residues_bar(residues=[154, 1613, 1430, 113], return_fig=True)


@pytest.mark.mpl_image_compare(filename='bar_expected_rates_vertical.png')
def test_bar_expected_rates_vertical(seq):
    return seq.plot_expected_mutation_rates_for_residues_bar(residues=[113, 154, 1613, 1430], return_fig=True,
                                                             orientation='vertical')


@pytest.mark.mpl_image_compare(filename='scatter_expected_rates_vs_observed.png')
def test_scatter_expected_rates_vs_observed(seq):
    return seq.plot_expected_mutation_rates_vs_observed_for_residues(residues=[113, 154, 1613, 1430], return_fig=True,
                                                                     mutations_to_annotate=['S154N', 'A1613T'])