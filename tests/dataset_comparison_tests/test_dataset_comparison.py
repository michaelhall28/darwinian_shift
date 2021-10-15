import os
import pickle
from darwinian_shift import DarwinianShift, homtest_sections

FILE_DIR = os.path.dirname(os.path.realpath(__file__))

def test_neutral():
    exon_file = os.path.join(FILE_DIR, 'gene1_exons.txt')
    reference_fasta = os.path.join(FILE_DIR, 'gene1.fasta')
    d1 = DarwinianShift(data=os.path.join(FILE_DIR, 'data1_neutral.tsv'),
                        exon_file=exon_file, reference_fasta=reference_fasta,
                        spectra=os.path.join(FILE_DIR, 'spectrum1.txt')
                        )

    d2 = DarwinianShift(data=os.path.join(FILE_DIR, 'data2_neutral.tsv'),
                        exon_file=exon_file, reference_fasta=reference_fasta,
                        spectra=os.path.join(FILE_DIR, 'spectrum2.txt')
                        )

    s1 = d1.make_section(gene='gene1')
    s1.load_section_mutations()

    s2 = d2.make_section(gene='gene1')
    s2.load_section_mutations()

    res = homtest_sections(s1, s2, use_weights=True)

    # output new test file. Do not uncomment unless results have changed and confident new results are correct
    # pickle.dump(res, open(os.path.join(FILE_DIR, "res_neutral.pickle"), 'wb'))

    expected = pickle.load(open(os.path.join(FILE_DIR, "res_neutral.pickle"), 'rb'))
    assert res == expected


def test_neutral_no_weights():
    exon_file = os.path.join(FILE_DIR, 'gene1_exons.txt')
    reference_fasta = os.path.join(FILE_DIR, 'gene1.fasta')
    d1 = DarwinianShift(data=os.path.join(FILE_DIR, 'data1_neutral.tsv'),
                        exon_file=exon_file, reference_fasta=reference_fasta,
                        spectra=os.path.join(FILE_DIR, 'spectrum1.txt')
                        )

    d2 = DarwinianShift(data=os.path.join(FILE_DIR, 'data2_neutral.tsv'),
                        exon_file=exon_file, reference_fasta=reference_fasta,
                        spectra=os.path.join(FILE_DIR, 'spectrum2.txt')
                        )

    s1 = d1.make_section(gene='gene1')
    s1.load_section_mutations()

    s2 = d2.make_section(gene='gene1')
    s2.load_section_mutations()

    res = homtest_sections(s1, s2, use_weights=False)

    # output new test file. Do not uncomment unless results have changed and confident new results are correct
    # pickle.dump(res, open(os.path.join(FILE_DIR, "res_neutral_no_weights.pickle"), 'wb'))

    expected = pickle.load(open(os.path.join(FILE_DIR, "res_neutral_no_weights.pickle"), 'rb'))
    assert res == expected