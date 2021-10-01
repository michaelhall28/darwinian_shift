import pytest
import os
from darwinian_shift import DarwinianShift, GlobalKmerSpectrum, TranscriptKmerSpectrum
from darwinian_shift import CDFPermutationTest, ChiSquareTest
from darwinian_shift.section import Section

DIR_PATH = os.path.dirname(os.path.realpath(__file__))
TEST_DATA_DIR = os.path.join(DIR_PATH, 'test_data')
# REFERENCE_RESULTS_DIR =  os.path.join(DIR_PATH, 'reference_results')
EXON_FILE = os.path.join(TEST_DATA_DIR, 'test_exon_data.tsv')
REFERENCE_FASTA_FILE = os.path.join(TEST_DATA_DIR, 'test_fasta.fa.gz')
MUTATION_DATA_FILE = os.path.join(TEST_DATA_DIR, 'test_mutation_data.tsv')

def sort_dataframe(df):
    # Sort the dataframe by the columns and remove the index.
    # For matching dataframes when the order may have changed.
    df = df.sort_index(axis=1)  # Sort columns
    # Ignore un-hashable columns for sorting. Will still be kept in dataframe.
    cols = [c for c in df.columns if (('bins' not in c) and ('count' not in c) and ('_CI_' not in c))]
    df = df.sort_values(cols).reset_index(drop=True)  # Sort rows and reset index
    return df


def compare_sorted_files(f1, f2):
    with open(f1) as fh:
        lines1 =fh.readlines()
    with open(f2) as fh:
        lines2 =fh.readlines()

    assert sorted(lines1) == sorted(lines2)


@pytest.fixture(scope='session')
def project():
    # General project that can be used for many tests but doesn't take too long to load.
    d = DarwinianShift(data=MUTATION_DATA_FILE,
                       exon_file=EXON_FILE,
                       reference_fasta=REFERENCE_FASTA_FILE,
                       lookup=lambda x: [1]*len(x.null_mutations),  # Make simple lookup here
                       # stats=[CDFPermutationTest(testing_random_seed=0), ChiSquareTest()],
                       spectra=[GlobalKmerSpectrum(k=3)],
                       low_mem=False)
    return d

@pytest.fixture(scope='session')
def project_spectrum():
    # A project that can do lots of the spectra
    # Needs to be set up with the spectra in advance to the correct kmers are collected.
    d = DarwinianShift(data=MUTATION_DATA_FILE,
                       exon_file=EXON_FILE,
                       reference_fasta=REFERENCE_FASTA_FILE,
                       lookup=lambda x: [1]*len(x.null_mutations),  # Make simple lookup here
                       # stats=[CDFPermutationTest(testing_random_seed=0), ChiSquareTest()],
                       low_mem=False,
                       spectra=[GlobalKmerSpectrum(deduplicate_spectrum=False,
                                                    k=3,  # Size of kmer nucleotide context. Use 3 for trinucleotides.
                                                    ignore_strand=False,
                                                    missing_value=0,  # To replace missing values. Useful to make non-zero in some cases.
                                                    name='glob_k3'),
                                GlobalKmerSpectrum(deduplicate_spectrum=False,
                                                   k=1,  # Size of kmer nucleotide context. Use 3 for trinucleotides.
                                                   ignore_strand=False,
                                                   missing_value=0,
                                                   # To replace missing values. Useful to make non-zero in some cases.
                                                   name='glob_k1'),
                                GlobalKmerSpectrum(deduplicate_spectrum=False,
                                                       k=5,
                                                       # Size of kmer nucleotide context. Use 3 for trinucleotides.
                                                       ignore_strand=False,
                                                       missing_value=0,
                                                       # To replace missing values. Useful to make non-zero in some cases.
                                                       name='glob_k5'),
                                GlobalKmerSpectrum(deduplicate_spectrum=True,
                                                       k=3,
                                                       # Size of kmer nucleotide context. Use 3 for trinucleotides.
                                                       ignore_strand=False,
                                                       missing_value=0,
                                                       # To replace missing values. Useful to make non-zero in some cases.
                                                       name='glob_k3_dd'),
                                GlobalKmerSpectrum(deduplicate_spectrum=False,
                                                   k=3,  # Size of kmer nucleotide context. Use 3 for trinucleotides.
                                                   ignore_strand=True,
                                                   missing_value=0,
                                                   # To replace missing values. Useful to make non-zero in some cases.
                                                   name='glob_k3_is')

                                ])
    return d

@pytest.fixture(scope='session')
def seq_object(project):
    s = project.make_section({'transcript_id': 'ENST00000263388'})
    s.load_section_mutations()
    return s


@pytest.fixture(scope='session')
def pdb_seq_object(project):
    transcript = project.get_transcript_obj(transcript_id='ENST00000263388')
    s = Section(transcript, pdb_id='4ZLP', pdb_chain='A', start=1378, end=1640)
    s.load_section_mutations()
    return s


def pytest_addoption(parser):
    parser.addoption(
        "--runslow", action="store_true", default=False, help="run slow tests"
    )
    parser.addoption(
        "--runveryslow", action="store_true", default=False, help="run very slow tests"
    )

def pytest_configure(config):
    config.addinivalue_line("markers", "slow: mark test as slow to run")
    config.addinivalue_line("markers", "veryslow: mark test as very slow to run")

def pytest_collection_modifyitems(config, items):
    if config.getoption("--runveryslow"):
        # --runveryslow given in cli: do not skip slow or very slow tests
        return
    skip_veryslow = pytest.mark.skip(reason="need --runveryslow option to run")
    for item in items:
        if "veryslow" in item.keywords:
            item.add_marker(skip_veryslow)
    if config.getoption("--runslow"):
        # --runslow given in cli: do not skip slow tests
        return
    skip_slow = pytest.mark.skip(reason="need --runslow option to run")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)