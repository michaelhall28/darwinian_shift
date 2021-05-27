from darwinian_shift.utils.util_functions import *
from tests.conftest import sort_dataframe, compare_sorted_files, MUTATION_DATA_FILE, TEST_DATA_DIR
from darwinian_shift.reference_data.reference_utils import get_source_genome_reference_file_paths
from pandas.testing import assert_frame_equal
import filecmp
import os

FILE_DIR = os.path.dirname(os.path.realpath(__file__))

def test_reverse_complement():
    seq = "ATGCCTATTGGATCCAAAGAGAGGCCAACATTTTTTGAAATTTTTAAGACACGCTGCAACAAAGCAGATT"
    rev_comp_expected = "AATCTGCTTTGTTGCAGCGTGTCTTAAAAATTTCAAAAAATGTTGGCCTCTCTTTGGATCCAATAGGCAT"

    rev_comp_calc = reverse_complement(seq)

    assert rev_comp_calc == rev_comp_expected


def test_output_transcript_sequences_to_fasta_aa(project, tmpdir):
    output_file = os.path.join(tmpdir, "output_transcript_aa.fa")
    output_transcript_sequences_to_fasta(project, output_file, aa_sequence=True)
    compare_sorted_files(output_file, os.path.join(FILE_DIR, "reference_transcript_aa.fa"))


def test_output_transcript_sequences_to_fasta_nuc(project, tmpdir):
    output_file = os.path.join(tmpdir, "output_transcript_nuc.fa")
    output_transcript_sequences_to_fasta(project, output_file, aa_sequence=False)
    compare_sorted_files(output_file, os.path.join(FILE_DIR, "reference_transcript_nuc.fa"))

def test_sort_multiple_arrays_using_one():
    arr1 = np.array([4, 3, 6, 1, 2.1])
    arr2 = np.arange(5)
    list3 = [4, 5, 6, 7, 8]
    sorted_arrs = sort_multiple_arrays_using_one(arr1, arr2, list3)

    expected = np.array([
        [1, 2.1, 3, 4, 6],
        [3, 4, 1, 0, 2],
        [7, 8, 5, 4, 6]
    ])
    assert np.array_equal(sorted_arrs, expected)


def _get_partial_file_path(full_path, num_dir=5):
    return "/".join(full_path.split("/")[-num_dir:])


def test_reference_file_paths():
    exon_file, reference_file = get_source_genome_reference_file_paths(source_genome='homo_sapiens')

    assert _get_partial_file_path(exon_file) == "darwinian_shift/reference_data/homo_sapiens/ensembl-99/biomart_exons_homo_sapiens.txt"
    assert _get_partial_file_path(
        reference_file) == "darwinian_shift/reference_data/homo_sapiens/ensembl-99/Homo_sapiens.GRCh38.dna.primary_assembly.fa.gz"

    exon_file, reference_file = get_source_genome_reference_file_paths(source_genome='GRch37')

    assert _get_partial_file_path(
        exon_file, 6) == "darwinian_shift/reference_data/homo_sapiens/ensembl-99/GRCh37/biomart_exons_homo_sapiens.txt"
    assert _get_partial_file_path(
        reference_file, 6) == "darwinian_shift/reference_data/homo_sapiens/ensembl-99/GRCh37/Homo_sapiens.GRCh37.dna.primary_assembly.fa.gz"

    exon_file, reference_file = get_source_genome_reference_file_paths(source_genome='mus_musculus', ensembl_release=99)

    assert _get_partial_file_path(
        exon_file) == "darwinian_shift/reference_data/mus_musculus/ensembl-99/biomart_exons_mus_musculus.txt"
    assert _get_partial_file_path(
        reference_file) == "darwinian_shift/reference_data/mus_musculus/ensembl-99/Mus_musculus.GRCm38.dna.primary_assembly.fa.gz"


def _test_get_uniprot_acc_from_transcript_id():
    assert get_uniprot_acc_from_transcript_id("ENST00000263388") == "Q9UM47"

def test_read_vcf():
    tsv_df = pd.read_csv(MUTATION_DATA_FILE, sep="\t")
    bases = ['A', 'C', 'G', 'T']
    tsv_df = tsv_df[(tsv_df['ref'].isin(bases)) & (tsv_df['mut'].isin(bases))]
    vcf_df = read_sbs_from_vcf(os.path.join(MUTATION_DATA_FILE[:-4] + '.vcf'))
    tsv_df['chr'] = tsv_df['chr'].astype(str)

    assert_frame_equal(sort_dataframe(tsv_df[['chr', 'pos', 'ref', 'mut']]), sort_dataframe(vcf_df))
