from darwinian_shift.mutation_spectrum import *
from pandas.testing import assert_frame_equal
from tests.conftest import sort_dataframe, compare_sorted_files
import os
import pickle
import filecmp

FILE_DIR = os.path.dirname(os.path.realpath(__file__))

def test_global_kmerA(project_spectrum):
    s = project_spectrum.get_spectrum('glob_k3')

    # output new test file. Do not uncomment unless results have changed and confident new results are correct
    #pickle.dump(s.spectrum, open(os.path.join(FILE_DIR, "glob_kmer_A.pickle"), 'wb'))

    expected = pickle.load(open(os.path.join(FILE_DIR, "glob_kmer_A.pickle"), 'rb'))
    assert_frame_equal(sort_dataframe(s.spectrum), sort_dataframe(expected))

def test_global_kmerB(project_spectrum):
    s = project_spectrum.get_spectrum('glob_k1')

    # output new test file. Do not uncomment unless results have changed and confident new results are correct
    #pickle.dump(s.spectrum, open(os.path.join(FILE_DIR, "glob_kmer_B.pickle"), 'wb'))

    expected = pickle.load(open(os.path.join(FILE_DIR, "glob_kmer_B.pickle"), 'rb'))
    assert_frame_equal(sort_dataframe(s.spectrum), sort_dataframe(expected))


def test_global_kmerC(project_spectrum):
    s = project_spectrum.get_spectrum('glob_k5')

    # output new test file. Do not uncomment unless results have changed and confident new results are correct
    # pickle.dump(s.spectrum, open(os.path.join(FILE_DIR, "glob_kmer_C.pickle"), 'wb'))

    expected = pickle.load(open(os.path.join(FILE_DIR, "glob_kmer_C.pickle"), 'rb'))
    assert_frame_equal(sort_dataframe(s.spectrum), sort_dataframe(expected))

def test_global_kmerD(project_spectrum):
    s = project_spectrum.get_spectrum('glob_k3_dd')

    # output new test file. Do not uncomment unless results have changed and confident new results are correct
    # pickle.dump(s.spectrum, open(os.path.join(FILE_DIR, "glob_kmer_D.pickle"), 'wb'))

    expected = pickle.load(open(os.path.join(FILE_DIR, "glob_kmer_D.pickle"), 'rb'))
    assert_frame_equal(sort_dataframe(s.spectrum), sort_dataframe(expected))


def test_global_kmerE(project_spectrum):
    s = project_spectrum.get_spectrum('glob_k3_is')

    # output new test file. Do not uncomment unless results have changed and confident new results are correct
    # pickle.dump(s.spectrum, open(os.path.join(FILE_DIR, "glob_kmer_E.pickle"), 'wb'))

    expected = pickle.load(open(os.path.join(FILE_DIR, "glob_kmer_E.pickle"), 'rb'))
    assert_frame_equal(sort_dataframe(s.spectrum), sort_dataframe(expected))


def test_transcript_kmerA(project_spectrum):
    s = TranscriptKmerSpectrum(deduplicate_spectrum=False,
                 k=3,  # Size of kmer nucleotide context. Use 3 for trinucleotides.
                 ignore_strand=False,
                 missing_value=0,  # To replace missing values. Useful to make non-zero in some cases.
                 name=None)
    s.set_project(project_spectrum)
    spectrum = s.get_complete_spectrum()

    # output new test file. Do not uncomment unless results have changed and confident new results are correct
    # pickle.dump(s.spectrum, open(os.path.join(FILE_DIR, "transcript_kmer_A.pickle"), 'wb'))

    expected = pickle.load(open(os.path.join(FILE_DIR, "transcript_kmer_A.pickle"), 'rb'))
    assert_frame_equal(sort_dataframe(spectrum), sort_dataframe(expected))

def test_transcript_kmerB(project_spectrum):
    s = TranscriptKmerSpectrum(deduplicate_spectrum=False,
                 k=1,  # Size of kmer nucleotide context. Use 3 for trinucleotides.
                 ignore_strand=False,
                 missing_value=0,  # To replace missing values. Useful to make non-zero in some cases.
                 name=None)
    s.set_project(project_spectrum)
    spectrum = s.get_complete_spectrum()

    # output new test file. Do not uncomment unless results have changed and confident new results are correct
    # pickle.dump(s.spectrum, open(os.path.join(FILE_DIR, "transcript_kmer_B.pickle"), 'wb'))

    expected = pickle.load(open(os.path.join(FILE_DIR, "transcript_kmer_B.pickle"), 'rb'))
    assert_frame_equal(sort_dataframe(spectrum), sort_dataframe(expected))


def test_transcript_kmerC(project_spectrum):
    s = TranscriptKmerSpectrum(deduplicate_spectrum=False,
                           k=5,  # Size of kmer nucleotide context. Use 3 for trinucleotides.
                           ignore_strand=False,
                           missing_value=0,  # To replace missing values. Useful to make non-zero in some cases.
                           name=None)
    s.set_project(project_spectrum)
    spectrum = s.get_complete_spectrum()

    # output new test file. Do not uncomment unless results have changed and confident new results are correct
    # pickle.dump(s.spectrum, open(os.path.join(FILE_DIR, "transcript_kmer_C.pickle"), 'wb'))

    expected = pickle.load(open(os.path.join(FILE_DIR, "transcript_kmer_C.pickle"), 'rb'))
    assert_frame_equal(sort_dataframe(spectrum), sort_dataframe(expected))

def test_transcript_kmerD(project_spectrum):
    s = TranscriptKmerSpectrum(deduplicate_spectrum=True,
                           k=3,  # Size of kmer nucleotide context. Use 3 for trinucleotides.
                           ignore_strand=False,
                           missing_value=0,  # To replace missing values. Useful to make non-zero in some cases.
                           name=None)
    s.set_project(project_spectrum)
    spectrum = s.get_complete_spectrum()

    # output new test file. Do not uncomment unless results have changed and confident new results are correct
    # pickle.dump(s.spectrum, open(os.path.join(FILE_DIR, "transcript_kmer_D.pickle"), 'wb'))

    expected = pickle.load(open(os.path.join(FILE_DIR, "transcript_kmer_D.pickle"), 'rb'))
    assert_frame_equal(sort_dataframe(spectrum), sort_dataframe(expected))


def test_transcript_kmerE(project_spectrum):
    s = TranscriptKmerSpectrum(deduplicate_spectrum=False,
                           k=3,  # Size of kmer nucleotide context. Use 3 for trinucleotides.
                           ignore_strand=True,
                           missing_value=0,  # To replace missing values. Useful to make non-zero in some cases.
                           name=None)
    s.set_project(project_spectrum)
    spectrum = s.get_complete_spectrum()

    # output new test file. Do not uncomment unless results have changed and confident new results are correct
    # pickle.dump(s.spectrum, open(os.path.join(FILE_DIR, "transcript_kmer_E.pickle"), 'wb'))

    expected = pickle.load(open(os.path.join(FILE_DIR, "transcript_kmer_E.pickle"), 'rb'))
    assert_frame_equal(sort_dataframe(spectrum), sort_dataframe(expected))


def test_global_write(project_spectrum):
    s = project_spectrum.get_spectrum('glob_k3')
    s.write_to_file(os.path.join(FILE_DIR, "glob_kmer_A.spectrum_new"))
    compare_sorted_files(os.path.join(FILE_DIR, "glob_kmer_A.spectrum_new"), os.path.join(FILE_DIR, "glob_kmer_A.spectrum"))

def test_global_read():
    s = read_spectrum(os.path.join(FILE_DIR, "glob_kmer_A.spectrum"))
    expected = pickle.load(open(os.path.join(FILE_DIR, "glob_kmer_A.pickle"), 'rb'))
    assert_frame_equal(sort_dataframe(s.spectrum), sort_dataframe(expected))


def test_transcript_write(project_spectrum):
    s = TranscriptKmerSpectrum(deduplicate_spectrum=False,
                               k=3,  # Size of kmer nucleotide context. Use 3 for trinucleotides.
                               ignore_strand=False,
                               missing_value=0,  # To replace missing values. Useful to make non-zero in some cases.
                               name=None)
    s.set_project(project_spectrum)
    s.get_complete_spectrum()
    s.write_to_file(os.path.join(FILE_DIR, "transcript_kmer_A.spectrum_new"))

    compare_sorted_files(os.path.join(FILE_DIR, "transcript_kmer_A.spectrum_new"),
                         os.path.join(FILE_DIR, "transcript_kmer_A.spectrum"))

def test_transcript_read():
    s = read_spectrum(os.path.join(FILE_DIR, "transcript_kmer_A.spectrum"))
    expected = pickle.load(open(os.path.join(FILE_DIR, "transcript_kmer_A.pickle"), 'rb'))
    assert_frame_equal(sort_dataframe(s.spectrum), sort_dataframe(expected))