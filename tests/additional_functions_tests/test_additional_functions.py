from darwinian_shift.additional_functions import uniprot_exploration, get_bins_for_uniprot_features
from darwinian_shift.additional_functions import pdbe_kb_exploration
from darwinian_shift import UniprotLookup, PDBeKBLookup
from pandas.testing import assert_frame_equal
from tests.conftest import sort_dataframe, TEST_DATA_DIR, MUTATION_DATA_FILE, EXON_FILE, REFERENCE_FASTA_FILE
import os
import pickle

FILE_DIR = os.path.dirname(os.path.realpath(__file__))

def test_uniprot_annotation(seq_object):
    uniprot_lookup = UniprotLookup(uniprot_directory=TEST_DATA_DIR)
    annotated_data = uniprot_lookup.annotate_dataframe(seq_object.observed_mutations, seq_object.transcript_id)

    # output new test file. Do not uncomment unless results have changed and confident new results are correct
    # pickle.dump(annotated_data, open(os.path.join(FILE_DIR, "annotated_data.pickle"), 'wb'))

    expected = pickle.load(open(os.path.join(FILE_DIR, "annotated_data.pickle"), 'rb'))
    assert_frame_equal(sort_dataframe(annotated_data), sort_dataframe(expected))


def test_uniprot_exploration():
    res = uniprot_exploration(genes=['NOTCH3'], data=MUTATION_DATA_FILE, exon_file=EXON_FILE,
                              fasta_file=REFERENCE_FASTA_FILE, plot=False, uniprot_directory=TEST_DATA_DIR,
                              match_variant_change=False)

    # output new test file. Do not uncomment unless results have changed and confident new results are correct
    # pickle.dump(res, open(os.path.join(FILE_DIR, "uniprot_exploration.pickle"), 'wb'))

    expected = pickle.load(open(os.path.join(FILE_DIR, "uniprot_exploration.pickle"), 'rb'))
    assert_frame_equal(sort_dataframe(res), sort_dataframe(expected))


def test_uniprot_exploration2():
    res = uniprot_exploration(transcripts=['ENST00000263388'], data=MUTATION_DATA_FILE, exon_file=EXON_FILE,
                              fasta_file=REFERENCE_FASTA_FILE, plot=False, uniprot_directory=TEST_DATA_DIR,
                              match_variant_change=False)

    # output new test file. Do not uncomment unless results have changed and confident new results are correct
    # pickle.dump(res, open(os.path.join(FILE_DIR, "uniprot_exploration2.pickle"), 'wb'))

    expected = pickle.load(open(os.path.join(FILE_DIR, "uniprot_exploration2.pickle"), 'rb'))
    assert_frame_equal(sort_dataframe(res), sort_dataframe(expected))


def test_uniprot_exploration3():
    res = uniprot_exploration(sections=[{'transcript_id': 'ENST00000263388', 'start': 1378, 'end': 1640}],
                              data=MUTATION_DATA_FILE, exon_file=EXON_FILE,
                              fasta_file=REFERENCE_FASTA_FILE, plot=False, uniprot_directory=TEST_DATA_DIR,
                              match_variant_change=False)

    # output new test file. Do not uncomment unless results have changed and confident new results are correct
    # pickle.dump(res, open(os.path.join(FILE_DIR, "uniprot_exploration3.pickle"), 'wb'))

    expected = pickle.load(open(os.path.join(FILE_DIR, "uniprot_exploration3.pickle"), 'rb'))
    assert_frame_equal(sort_dataframe(res), sort_dataframe(expected))

def test_uniprot_exploration4():
    res = uniprot_exploration(genes=['NOTCH3'], data=MUTATION_DATA_FILE, exon_file=EXON_FILE,
                              fasta_file=REFERENCE_FASTA_FILE, plot=False, uniprot_directory=TEST_DATA_DIR,
                              match_variant_change=True)

    # output new test file. Do not uncomment unless results have changed and confident new results are correct
    # pickle.dump(res, open(os.path.join(FILE_DIR, "uniprot_exploration4.pickle"), 'wb'))

    expected = pickle.load(open(os.path.join(FILE_DIR, "uniprot_exploration4.pickle"), 'rb'))
    assert_frame_equal(sort_dataframe(res), sort_dataframe(expected))

def test_get_uniprot_domains():
    uniprot_lookup = UniprotLookup(uniprot_directory=TEST_DATA_DIR)
    transcript_uniprot = uniprot_lookup.get_uniprot_data('ENST00000263388')
    res = get_bins_for_uniprot_features(transcript_uniprot, feature_types=('domain', 'transmembrane region'),
                                                                           min_gap=0, last_residue=3000)

    # output new test file. Do not uncomment unless results have changed and confident new results are correct
    # pickle.dump(res, open(os.path.join(FILE_DIR, "uniprot_bins.pickle"), 'wb'))

    expected = pickle.load(open(os.path.join(FILE_DIR, "uniprot_bins.pickle"), 'rb'))
    assert res == expected

def test_pdbekb_exploration(project):
    p = PDBeKBLookup(pdbekb_dir=TEST_DATA_DIR,
                     transcript_uniprot_mapping={'ENST00000263388': 'ABC123'}  # Map to the fake test data
    )
    d = project.change_lookup(p)
    res = pdbe_kb_exploration(d, transcript_id='ENST00000263388')

    # output new test file. Do not uncomment unless results have changed and confident new results are correct
    # pickle.dump(res, open(os.path.join(FILE_DIR, "pdbekb_exploration.pickle"), 'wb'))

    expected = pickle.load(open(os.path.join(FILE_DIR, "pdbekb_exploration.pickle"), 'rb'))
    assert_frame_equal(sort_dataframe(res), sort_dataframe(expected))