from darwinian_shift.utils.sifts_functions import *
import filecmp
import pandas as pd

FILE_DIR = os.path.dirname(os.path.realpath(__file__))

def test_read_xml():
    pdbid = "5UK5"
    read_xml(pdbid, FILE_DIR)
    assert filecmp.cmp(os.path.join(FILE_DIR, "{}.txt".format(pdbid.lower())),
                os.path.join(FILE_DIR,"reference_{}.txt".format(pdbid.lower())))


def test_get_sifts_alignment():
    pdbid = "5UK5"
    aln = get_sifts_alignment(pdbid, FILE_DIR, download=False, convert_numeric=True)

    # Make results the first time (or after a deliberate change to the results)
    # aln.to_csv(os.path.join(DIR_PATH, "reference_sifts_aln.csv"), index=False)

    expected = pd.read_csv(os.path.join(FILE_DIR, "reference_sifts_aln.csv"))

    pd.testing.assert_frame_equal(aln, expected)