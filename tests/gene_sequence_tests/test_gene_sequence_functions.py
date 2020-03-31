from darwinian_shift.utils.gene_sequence_functions import *
from tests.conftest import EXON_FILE, REFERENCE_FASTA_FILE
import pickle
import os
import numpy as np
from pandas.testing import assert_frame_equal
import pytest

FILE_DIR = os.path.dirname(os.path.realpath(__file__))

@pytest.mark.parametrize("gene,transcript",[("KEAP1", None), (None, "ENST00000171111")])
def test_genomic_ranges(gene, transcript):
    exon_data = pd.read_csv(EXON_FILE, sep="\t")
    ranges = get_genomic_ranges_for_gene(exon_data, gene=gene, transcript=transcript)

    # output new test file. Do not uncomment unless results have changed and confident new results are correct
    # pickle.dump(ranges, open(os.path.join(FILE_DIR, "genomic_ranges.pickle"), 'wb'))
    expected = pickle.load(open(os.path.join(FILE_DIR, "genomic_ranges.pickle"), 'rb'))
    assert np.array_equal(ranges[0], expected[0])  # the ranges
    assert ranges[1] == expected[1]  # the transcript
    assert ranges[2] == expected[2]  # the strand


def test_get_positions_from_ranges():
    ranges, transcript, strand = pickle.load(open(os.path.join(FILE_DIR, "genomic_ranges.pickle"), 'rb'))
    cds_positions = get_positions_from_ranges(ranges, strand)

    # output new test file. Do not uncomment unless results have changed and confident new results are correct
    # pickle.dump(cds_positions, open(os.path.join(FILE_DIR, "cds_positions.pickle"), 'wb'))

    expected = pickle.load(open(os.path.join(FILE_DIR, "cds_positions.pickle"), 'rb'))
    assert cds_positions == expected


def test_get_gene_kmers_from_exon_ranges():
    ranges, transcript, strand = pickle.load(open(os.path.join(FILE_DIR, "genomic_ranges.pickle"), 'rb'))
    chrom = '19'
    gene_kmers, exonic_gene_sequence = get_gene_kmers_from_exon_ranges(chromosome_sequence=None,
                                                                       chrom_offset=None,
                                                                       exon_ranges=ranges,
                                                                       strand=strand,
                                                                       ks=[3, 5, 7],
                                                                       reference=REFERENCE_FASTA_FILE,
                                                                       chrom=chrom)

    # output new test file. Do not uncomment unless results have changed and confident new results are correct
    # pickle.dump([gene_kmers, exonic_gene_sequence], open(os.path.join(FILE_DIR, "kmers.pickle"), 'wb'))

    expected_gene_kmers, expected_exonic_gene_sequence = pickle.load(open(os.path.join(FILE_DIR, "kmers.pickle"), 'rb'))
    assert gene_kmers == expected_gene_kmers
    assert exonic_gene_sequence == expected_exonic_gene_sequence


def test_get_all_possible_single_nucleotide_mutations():
    gene_kmers, exonic_gene_sequence = pickle.load(open(os.path.join(FILE_DIR, "kmers.pickle"), 'rb'))
    all_possible_mutations = get_all_possible_single_nucleotide_mutations(exonic_gene_sequence, gene_kmers,
                                                                          sort_results=True)

    # output new test file. Do not uncomment unless results have changed and confident new results are correct
    # pickle.dump(all_possible_mutations, open(os.path.join(FILE_DIR, "all_mutations.pickle"), 'wb'))

    expected = pickle.load(open(os.path.join(FILE_DIR, "all_mutations.pickle"), 'rb'))

    assert_frame_equal(all_possible_mutations, expected)