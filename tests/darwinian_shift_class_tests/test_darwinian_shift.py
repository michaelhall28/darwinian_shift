import pickle
import numpy as np
import os
import pandas as pd
from pandas.testing import assert_frame_equal, assert_series_equal
from tests.conftest import MUTATION_DATA_FILE, EXON_FILE, REFERENCE_FASTA_FILE, TEST_DATA_DIR, DIR_PATH, sort_dataframe
import pytest
from darwinian_shift import DarwinianShift
from darwinian_shift import GlobalKmerSpectrum, TranscriptKmerSpectrum, EvenMutationalSpectrum
from darwinian_shift.lookup_classes import *
from darwinian_shift import PermutationTest, CDFPermutationTest, ChiSquareTest, KSTest

# Test the class functions of the DarwinianShift class.
# Also test that the results of the full process is consistent
# Test with all input options

FILE_DIR = os.path.dirname(os.path.realpath(__file__))
RESULTS_DIR = os.path.join(FILE_DIR, 'reference_results')

def test_get_overlapped_transcripts(project):
    exon_data = pd.read_csv(EXON_FILE, sep="\t")
    exon_data.loc[:, 'Chromosome/scaffold name'] = exon_data['Chromosome/scaffold name'].astype(str)
    exon_data = exon_data[exon_data['Chromosome/scaffold name'].isin(project.chromosomes)]
    exon_data = exon_data[~pd.isnull(exon_data['Genomic coding start'])]
    exon_data['Genomic coding start'] = exon_data['Genomic coding start'].astype(int)
    exon_data['Genomic coding end'] = exon_data['Genomic coding end'].astype(int)
    res = project.get_overlapped_transcripts(project.data, exon_data)

    # output new test file. Do not uncomment unless results have changed and confident new results are correct
    # pickle.dump(res, open(os.path.join(RESULTS_DIR, "overlapped_transcripts.pickle"), 'wb'))

    expected = pickle.load(open(os.path.join(RESULTS_DIR, "overlapped_transcripts.pickle"), 'rb'))
    assert_frame_equal(sort_dataframe(res), sort_dataframe(expected))


def run_full_process(spectra, lookup, gene_list, transcript_list, deduplicate, excluded_positions,
                      use_longest_transcript_only, exclude_synonymous, exclude_nonsense):
    np.random.seed(0)
    # This will be a very slow test.
    # Checks that the process works from start to finish for all options.
    if gene_list is not None or transcript_list is not None:
        # Results should be same if any of these are defined. Always testing the same transcripts
        gene_list_name = '2genes'
    else:
        gene_list_name = 'None'
    if excluded_positions is not None:
        ep = '1'
    else:
        ep = '0'

    excluded_mutations = []
    if exclude_synonymous:
        excluded_mutations.append('synonymous')
    if exclude_nonsense:
        excluded_mutations.append('nonsense')

    test_name = "_".join([lookup.__class__.__name__, gene_list_name, str(int(deduplicate)), ep,
                          str(int(use_longest_transcript_only)), str(int(exclude_synonymous)),
                          str(int(exclude_nonsense))])

    statistics = [CDFPermutationTest(num_permutations=1000), ChiSquareTest(),
                  PermutationTest(stat_function=np.mean, num_permutations=1000),
                  KSTest()]

    d = DarwinianShift(data=MUTATION_DATA_FILE,
                       exon_file=EXON_FILE,
                       reference_fasta=REFERENCE_FASTA_FILE,
                       lookup=lookup,
                       stats=statistics,
                       spectra=spectra['spectra'],
                       gene_list=gene_list,
                       transcript_list=transcript_list,
                       deduplicate=deduplicate,
                       excluded_positions=excluded_positions,
                       use_longest_transcript_only=use_longest_transcript_only,
                       excluded_mutation_types=excluded_mutations,
                       testing_random_seed=1,
                       verbose=True
                       )
    d.run_all()

    # output new test file. Do not uncomment unless results have changed and confident new results are correct
    # Be careful to only overwrite for the particular test(s) the you want.
    # This function runs with many parameters. Uncommenting and running all will overwrite all cases.
    # pickle.dump(d.scored_data, open(os.path.join(RESULTS_DIR, "scored_data_{}.pickle".format(test_name)), 'wb'))
    # pickle.dump(d.results, open(os.path.join(RESULTS_DIR, "results_{}.pickle".format(test_name)), 'wb'))

    expected = pickle.load(open(os.path.join(RESULTS_DIR, "scored_data_{}.pickle".format(test_name)), 'rb'))
    assert_frame_equal(sort_dataframe(d.scored_data), sort_dataframe(expected), check_dtype=False)

    expected = pickle.load(open(os.path.join(RESULTS_DIR, "results_{}.pickle".format(test_name)), 'rb'))
    assert_frame_equal(sort_dataframe(d.results), sort_dataframe(expected))

@pytest.mark.slow
@pytest.mark.parametrize('option', list(range(8)))
def test_full_process_options_short(option):
    """
    Run through each option for the full_process and the combination of all.
    Makes sure they all work without having to every single combination.
    Specify option to track which option combination a failure is caused by
    :return:
    """
    spectra = {'name': '1', 'spectra': (
        EvenMutationalSpectrum(),
        GlobalKmerSpectrum(k=1),
        GlobalKmerSpectrum(k=3),
        GlobalKmerSpectrum(k=5),
        GlobalKmerSpectrum(k=1, deduplicate_spectrum=True, name='GK1dd'),
        GlobalKmerSpectrum(k=3, ignore_strand=True, name='GK3is'),
        GlobalKmerSpectrum(k=5, missing_value=0.01, name='GK5mv'),
        TranscriptKmerSpectrum(k=1),
        TranscriptKmerSpectrum(k=3),
        TranscriptKmerSpectrum(k=5),
        TranscriptKmerSpectrum(k=1, deduplicate_spectrum=True, name='TK1dd'),
        TranscriptKmerSpectrum(k=3, ignore_strand=True, name='TK3is'),
        TranscriptKmerSpectrum(k=5, missing_value=0.01, name='TK3mv'),
    )}

    if option == 0:
        # With gene list
        run_full_process(spectra=spectra, lookup=DummyValuesRandom(np.random.random, testing_random_seed=0),
                      gene_list=('NOTCH3', 'KEAP1'), transcript_list=None, deduplicate=False, excluded_positions=None,
                      use_longest_transcript_only=True, exclude_synonymous=False, exclude_nonsense=False)

    if option == 1:
        # With transcript list
        run_full_process(spectra=spectra, lookup=DummyValuesRandom(np.random.random, testing_random_seed=0),
                          gene_list=None, transcript_list=('ENST00000263388','ENST00000171111'),
                          deduplicate=False, excluded_positions=None,
                          use_longest_transcript_only=True, exclude_synonymous=False, exclude_nonsense=False)

    if option == 2:
        # Deduplicate
        run_full_process(spectra=spectra, lookup=DummyValuesRandom(np.random.random, testing_random_seed=0),
                          gene_list=None, transcript_list=None, deduplicate=True, excluded_positions=None,
                          use_longest_transcript_only=True, exclude_synonymous=False, exclude_nonsense=False)

    if option == 3:
        # Exclude positions
        run_full_process(spectra=spectra, lookup=DummyValuesRandom(np.random.random, testing_random_seed=0),
                          gene_list=None, transcript_list=None, deduplicate=False,
                          excluded_positions={'19':[40761076, 40761152, 15302303, 15288493]},
                          use_longest_transcript_only=True, exclude_synonymous=False, exclude_nonsense=False)

    if option == 4:
        # All transcripts
        run_full_process(spectra=spectra, lookup=DummyValuesRandom(np.random.random, testing_random_seed=0),
                          gene_list=None, transcript_list=None, deduplicate=False, excluded_positions=None,
                          use_longest_transcript_only=False, exclude_synonymous=False, exclude_nonsense=False)

    if option == 5:
        # exclude synonymous
        run_full_process(spectra=spectra, lookup=DummyValuesRandom(np.random.random, testing_random_seed=0),
                          gene_list=None, transcript_list=None, deduplicate=False, excluded_positions=None,
                          use_longest_transcript_only=True, exclude_synonymous=True, exclude_nonsense=False)

    if option == 6:
        # exclude nonsense
        run_full_process(spectra=spectra, lookup=DummyValuesRandom(np.random.random, testing_random_seed=0),
                          gene_list=None, transcript_list=None, deduplicate=False, excluded_positions=None,
                          use_longest_transcript_only=True, exclude_synonymous=False, exclude_nonsense=True)

    if option == 7:
        # All
        run_full_process(spectra=spectra, lookup=DummyValuesRandom(np.random.random, testing_random_seed=0),
                          gene_list=('NOTCH3', 'KEAP1'), transcript_list=('ENST00000263388','ENST00000171111'),
                          deduplicate=True, excluded_positions={'19':[40761076, 40761152, 15302303, 15288493]},
                          use_longest_transcript_only=False, exclude_synonymous=True, exclude_nonsense=True)

@pytest.mark.parametrize("sections,lookup", [pytest.param(pd.DataFrame({
    'transcript_id': ['ENST00000263388', 'ENST00000601011', 'ENST00000171111'],
    'start': [50, 70, 14],
    'end': [500, 780, 250],
    'section_id': ['s1', 's2', 's3'],
    'extra_kwarg': ['arbitrary', 'extra', 'values']
}), DummyValuesRandom(np.random.random, testing_random_seed=0),  marks=pytest.mark.slow),
    pytest.param(pd.DataFrame({
    'transcript_id': ['ENST00000263388'],
    'pdb_id': ['4zlp'],
    'pdb_chain': ['A'],
    'section_id': ['s1']
}), FoldXLookup(foldx_results_directory=os.path.join(DIR_PATH, 'lookup_tests', 'foldx_data'),
                sifts_directory=TEST_DATA_DIR, download_sifts=False),  marks=pytest.mark.slow)
], ids=['sections1', 'sections2'])
def test_section_input_process(sections, lookup):
    np.random.seed(0)

    d = DarwinianShift(data=MUTATION_DATA_FILE,
                       exon_file=EXON_FILE,
                       reference_fasta=REFERENCE_FASTA_FILE,
                       lookup=lookup,
                       sections=sections,
                       stats=[CDFPermutationTest(), ChiSquareTest()],
                       testing_random_seed=1,
                       verbose=True
                       )
    d.run_all()

    if 'start' in sections.columns:
        test_name = 'sections1'
    else:
        test_name = 'sections2'

    # output new test file. Do not uncomment unless results have changed and confident new results are correct
    # Be careful to only overwrite for the particular test(s) the you want.
    # This function runs with many parameters. Uncommenting and running all will overwrite all cases.
    # pickle.dump(d.scored_data, open(os.path.join(RESULTS_DIR, "scored_data_{}.pickle".format(test_name)), 'wb'))
    # pickle.dump(d.results, open(os.path.join(RESULTS_DIR, "results_{}.pickle".format(test_name)), 'wb'))

    expected = pickle.load(open(os.path.join(RESULTS_DIR, "scored_data_{}.pickle".format(test_name)), 'rb'))
    assert_frame_equal(sort_dataframe(d.scored_data), sort_dataframe(expected), check_dtype=False)

    expected = pickle.load(open(os.path.join(RESULTS_DIR, "results_{}.pickle".format(test_name)), 'rb'))
    assert_frame_equal(sort_dataframe(d.results), sort_dataframe(expected))


def test_incorrect_bases():
    d = DarwinianShift(data=os.path.join(TEST_DATA_DIR, 'test_mutation_data_wrong_bases.tsv'),
                           exon_file=EXON_FILE,
                           reference_fasta=REFERENCE_FASTA_FILE,
                           lookup=lambda x: [1]*len(x.null_mutations),  # Make simple lookup here
                           spectra=[GlobalKmerSpectrum(k=3)],
                           low_mem=False)
    assert d.total_spectrum_ref_mismatch == 87

    s1 = d.run_transcript('ENST00000263388')
    assert s1.ref_mismatch_count == 77


def compare_column_lists(df1, df2):
    cols1 = set(df1.columns)
    cols2 = set(df2.columns)
    only1 = cols1.difference(cols2)
    only2 = cols2.difference(cols1)
    common = cols1.intersection(cols2)
    assert_frame_equal(sort_dataframe(df1[common]), sort_dataframe(df2[common]))
    return only1, only2

