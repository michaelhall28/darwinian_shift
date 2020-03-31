from darwinian_shift.lookup_classes import *
import os
import numpy as np
from tests.conftest import TEST_DATA_DIR

FILE_DIR = os.path.dirname(os.path.realpath(__file__))

def test_aaindex_lookup1(seq_object):
    aaindex_file = os.path.join(FILE_DIR, 'aaindex_sample.txt')
    aa = AAindexLookup(aaindex_file, 'alpha-CH chemical shifts (Andersen et al., 1992)', abs_value=False)
    res = aa(seq_object)

    # Save reference results if they have been deliberately changed
    # np.save(os.path.join(FILE_DIR, 'reference_aaindex_results1'), res)

    expected = np.load(os.path.join(FILE_DIR, 'reference_aaindex_results1.npy'))

    np.testing.assert_array_equal(expected, res)

def test_aaindex_lookup2(seq_object):
    aaindex_file = os.path.join(FILE_DIR, 'aaindex_sample.txt')
    aa = AAindexLookup(aaindex_file, 'Hydrophobicity index (Argos et al., 1982)', abs_value=True)
    res = aa(seq_object)

    # Save reference results if they have been deliberately changed
    # np.save(os.path.join(FILE_DIR, 'reference_aaindex_results2'), res)

    expected = np.load(os.path.join(FILE_DIR, 'reference_aaindex_results2.npy'))

    np.testing.assert_array_equal(expected, res)

def test_bigwig_lookup(seq_object):
    bw_file = os.path.join(FILE_DIR, 'phylop_bw_section.bw')
    bw = BigwigLookup(bw_file)
    res = bw(seq_object)

    # Save reference results if they have been deliberately changed
    # np.save(os.path.join(FILE_DIR, 'reference_bw_results'), res)

    expected = np.load(os.path.join(FILE_DIR, 'reference_bw_results.npy'))

    np.testing.assert_array_equal(expected, res)


def test_dummy_lookup(seq_object, project):
    np.random.seed(0)
    d = DummyValuesPosition(project.data, np.random.random)
    res = d(seq_object)

    # Save reference results if they have been deliberately changed
    # np.save(os.path.join(FILE_DIR, 'reference_dummy_results'), res)

    expected = np.load(os.path.join(FILE_DIR, 'reference_dummy_results.npy'))

    np.testing.assert_array_equal(expected, res)

def test_foldx_lookup(pdb_seq_object):
    f = FoldXLookup(foldx_results_directory=os.path.join(FILE_DIR, 'foldx_data'),
                    sifts_directory=TEST_DATA_DIR, download_sifts=False)
    res = f(pdb_seq_object)

    # Save reference results if they have been deliberately changed
    # np.save(os.path.join(FILE_DIR, 'reference_foldx_results'), res)

    expected = np.load(os.path.join(FILE_DIR, 'reference_foldx_results.npy'))

    np.testing.assert_array_equal(expected, res)

def test_OR_lookup1(pdb_seq_object):
    # Want to use foldx lookup as one of the inputs so can test include_missing_values
    bw_file = os.path.join(FILE_DIR, 'phylop_bw_section.bw')
    bw = BigwigLookup(bw_file)
    f = FoldXLookup(foldx_results_directory=os.path.join(FILE_DIR, 'foldx_data'),
                    sifts_directory=TEST_DATA_DIR, download_sifts=False)

    orl = ORLookup([bw, f], thresholds=[4, 1], directions=[1, -1])
    res = orl(pdb_seq_object)

    # Save reference results if they have been deliberately changed
    # np.save(os.path.join(FILE_DIR, 'reference_or_results1'), res)

    expected = np.load(os.path.join(FILE_DIR, 'reference_or_results1.npy'))

    np.testing.assert_array_equal(expected, res)

def test_OR_lookup2(pdb_seq_object):
    # Want to use foldx lookup as one of the inputs so can test include_missing_values
    bw_file = os.path.join(FILE_DIR, 'phylop_bw_section.bw')
    bw = BigwigLookup(bw_file)
    f = FoldXLookup(foldx_results_directory=os.path.join(FILE_DIR, 'foldx_data'),
                    sifts_directory=TEST_DATA_DIR, download_sifts=False)

    orl = ORLookup([bw, f], thresholds=[4, 1], directions=[1, -1], include_missing_values=True)
    res = orl(pdb_seq_object)

    # Save reference results if they have been deliberately changed
    #np.save(os.path.join(FILE_DIR, 'reference_or_results2'), res)

    expected = np.load(os.path.join(FILE_DIR, 'reference_or_results2.npy'))

    np.testing.assert_array_equal(expected, res)

def test_AND_lookup1(pdb_seq_object):
    # Want to use foldx lookup as one of the inputs so can test include_missing_values
    bw_file = os.path.join(FILE_DIR, 'phylop_bw_section.bw')
    bw = BigwigLookup(bw_file)
    f = FoldXLookup(foldx_results_directory=os.path.join(FILE_DIR, 'foldx_data'),
                    sifts_directory=TEST_DATA_DIR, download_sifts=False)

    andl = ANDLookup([bw, f], thresholds=[4, 1], directions=[1, -1])
    res = andl(pdb_seq_object)

    # Save reference results if they have been deliberately changed
    #np.save(os.path.join(FILE_DIR, 'reference_and_results1'), res)

    expected = np.load(os.path.join(FILE_DIR, 'reference_and_results1.npy'))

    np.testing.assert_array_equal(expected, res)

def test_MutationExclusion_lookup(pdb_seq_object):
    # Want to use foldx lookup as one of the inputs so can test include_missing_values
    bw_file = os.path.join(FILE_DIR, 'phylop_bw_section.bw')
    bw = BigwigLookup(bw_file)
    f = FoldXLookup(foldx_results_directory=os.path.join(FILE_DIR, 'foldx_data'),
                    sifts_directory=TEST_DATA_DIR, download_sifts=False)

    ml = MutationExclusionLookup(lookup=bw, exclusion_lookup=f, exclusion_threshold=2)
    res = ml(pdb_seq_object)

    # Save reference results if they have been deliberately changed
    # np.save(os.path.join(FILE_DIR, 'reference_mut_exl_results'), res)

    expected = np.load(os.path.join(FILE_DIR, 'reference_mut_exl_results.npy'))

    np.testing.assert_array_equal(expected, res)

def test_AND_lookup2(pdb_seq_object):
    # Want to use foldx lookup as one of the inputs so can test include_missing_values
    bw_file = os.path.join(FILE_DIR, 'phylop_bw_section.bw')
    bw = BigwigLookup(bw_file)
    f = FoldXLookup(foldx_results_directory=os.path.join(FILE_DIR, 'foldx_data'),
                    sifts_directory=TEST_DATA_DIR, download_sifts=False)

    andl = ANDLookup([bw, f], thresholds=[4, 1], directions=[1, -1], include_missing_values=True)
    res = andl(pdb_seq_object)

    # Save reference results if they have been deliberately changed
    #np.save(os.path.join(FILE_DIR, 'reference_and_results2'), res)

    expected = np.load(os.path.join(FILE_DIR, 'reference_and_results2.npy'))

    np.testing.assert_array_equal(expected, res)


def test_iupred2a_lookup(seq_object):
    pred = IUPRED2ALookup(os.path.join(FILE_DIR, 'iupred_sample.txt'))
    res = pred(seq_object)

    # Save reference results if they have been deliberately changed
    # np.save(os.path.join(FILE_DIR, 'reference_iupred2a_results'), res)

    expected = np.load(os.path.join(FILE_DIR, 'reference_iupred2a_results.npy'))

    np.testing.assert_array_equal(expected, res)

def test_prody_lookup(pdb_seq_object):
    for prody_metric in ProDyLookup._options:

        pr = ProDyLookup(metric=prody_metric, exclude_ends=0, pdb_directory=TEST_DATA_DIR,
                         sifts_directory=TEST_DATA_DIR, dssp_directory=TEST_DATA_DIR,
                         download_sifts=False, quiet=True)

        res = pr(pdb_seq_object)

        # Save reference results if they have been deliberately changed
        # np.save(os.path.join(FILE_DIR, 'reference_{}_results'.format(prody_metric)), res)

        expected = np.load(os.path.join(FILE_DIR, 'reference_{}_results.npy'.format(prody_metric)))

        np.testing.assert_almost_equal(expected, res)

    for prody_metric in ProDyLookup._options:

        pr = ProDyLookup(metric=prody_metric, exclude_ends=5, pdb_directory=TEST_DATA_DIR,
                         sifts_directory=TEST_DATA_DIR, dssp_directory=TEST_DATA_DIR,
                         download_sifts=False, quiet=True)

        res = pr(pdb_seq_object)
        # print(res)
        # print(sum([1 for r in res if not np.isnan(r)]))

        # Save reference results if they have been deliberately changed
        # np.save(os.path.join(FILE_DIR, 'reference_{}_exclude_ends_results'.format(prody_metric)), res)

        expected = np.load(os.path.join(FILE_DIR, 'reference_{}_exclude_ends_results.npy'.format(prody_metric)))

        np.testing.assert_almost_equal(expected, res)


def test_residue_distance_lookup(pdb_seq_object):
    dist_lookup = StructureDistanceLookup(pdb_directory=TEST_DATA_DIR,
                                          sifts_directory=TEST_DATA_DIR)
    pdb_seq_object.target_selection = "protein and segid A and resid 1531 1532"

    res = dist_lookup(pdb_seq_object)

    # Save reference results if they have been deliberately changed
    # np.save(os.path.join(FILE_DIR, 'reference_residue_distance_results'), res)

    expected = np.load(os.path.join(FILE_DIR, 'reference_residue_distance_results.npy'))

    np.testing.assert_almost_equal(expected, res)


def test_residue_distance_lookup_bool(pdb_seq_object):
    dist_lookup = StructureDistanceLookup(pdb_directory=TEST_DATA_DIR,
                                          sifts_directory=TEST_DATA_DIR, boolean=True)
    pdb_seq_object.target_selection = "protein and segid A and resid 1531 1532"

    res = dist_lookup(pdb_seq_object)

    # Save reference results if they have been deliberately changed
    np.save(os.path.join(FILE_DIR, 'reference_residue_distance_results2'), res)

    expected = np.load(os.path.join(FILE_DIR, 'reference_residue_distance_results2.npy'))

    np.testing.assert_almost_equal(expected, res)


def test_sequence_distance_lookup(pdb_seq_object):
    dist_lookup = SequenceDistanceLookup()
    pdb_seq_object.target_selection = np.array([1531, 1532])

    res = dist_lookup(pdb_seq_object)

    # Save reference results if they have been deliberately changed
    np.save(os.path.join(FILE_DIR, 'reference_sequence_distance_results'), res)

    expected = np.load(os.path.join(FILE_DIR, 'reference_sequence_distance_results.npy'))

    np.testing.assert_almost_equal(expected, res)


def test_sequence_distance_lookup_bool(pdb_seq_object):
    dist_lookup = SequenceDistanceLookup(boolean=True, target_key='some_test_key')
    pdb_seq_object.some_test_key = np.array([1531, 1532])

    res = dist_lookup(pdb_seq_object)

    # Save reference results if they have been deliberately changed
    np.save(os.path.join(FILE_DIR, 'reference_sequence_distance_results2'), res)

    expected = np.load(os.path.join(FILE_DIR, 'reference_sequence_distance_results2.npy'))

    np.testing.assert_almost_equal(expected, res)


def test_uniprot_lookup1(seq_object):
    # Uses pre-downloaded files
    uniprot_lookup = UniprotLookup(uniprot_directory=TEST_DATA_DIR)

    res = uniprot_lookup(seq_object)

    # Save reference results if they have been deliberately changed
    # np.save(os.path.join(FILE_DIR, 'reference_uniprot_results'), res)

    expected = np.load(os.path.join(FILE_DIR, 'reference_uniprot_results.npy'))

    np.testing.assert_almost_equal(expected, res)


def test_uniprot_lookup2(seq_object):
    # Uses uniprot api
    uniprot_lookup = UniprotLookup()

    res = uniprot_lookup(seq_object)

    # Save reference results if they have been deliberately changed
    # np.save(os.path.join(FILE_DIR, 'reference_uniprot_results'), res)

    expected = np.load(os.path.join(FILE_DIR, 'reference_uniprot_results.npy'))

    np.testing.assert_almost_equal(expected, res)


def test_uniprot_lookup3(seq_object):
    # Uses the uniprot accession
    uniprot_lookup = UniprotLookup(uniprot_directory=TEST_DATA_DIR,
                                   transcript_uniprot_mapping={'ENST00000263388': 'Q9UM47'})

    res = uniprot_lookup(seq_object)

    # Save reference results if they have been deliberately changed
    # np.save(os.path.join(FILE_DIR, 'reference_uniprot_results'), res)

    expected = np.load(os.path.join(FILE_DIR, 'reference_uniprot_results.npy'))

    np.testing.assert_almost_equal(expected, res)


def test_uniprot_lookup4(seq_object):
    # Uses the uniprot accession and api
    uniprot_lookup = UniprotLookup(transcript_uniprot_mapping={'ENST00000263388': 'Q9UM47'})

    res = uniprot_lookup(seq_object)

    # Save reference results if they have been deliberately changed
    # np.save(os.path.join(FILE_DIR, 'reference_uniprot_results'), res)

    expected = np.load(os.path.join(FILE_DIR, 'reference_uniprot_results.npy'))

    np.testing.assert_almost_equal(expected, res)