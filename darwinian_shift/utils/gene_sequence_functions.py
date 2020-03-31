import pandas as pd
import numpy as np
from collections import defaultdict
from pysam import FastaFile
from Bio.Seq import Seq
from .util_functions import reverse_complement


def get_genomic_ranges_for_gene(exon_data, gene=None, transcript=None, print_transcript_warning=False):
    """
    If transcript is not specified, use the longest one.
    Biomart data needs Gene name, Transcript stable ID, Exon rank in transcript, Genomic coding start,
    Genomic coding end, Strand
    If transcript is not specified, also needs CDS Length

    Assumes that exons with null "Genomic coding start" have already been removed.
    """
    if gene is None and transcript is None:
        raise TypeError('get_genomic_ranges_for_gene requires either a gene name or transcript id')

    if transcript is None:
        gene_exons = exon_data[(exon_data['Gene name'] == gene)]
        if len(gene_exons) == 0:
            print('No protein coding transcript matching gene name', gene)
            return [], None, None, None
        longest_cds = gene_exons['CDS Length'].max()
        longest_transcripts = gene_exons[gene_exons['CDS Length'] == longest_cds]
        transcript = longest_transcripts.iloc[0]['Transcript stable ID']
        if print_transcript_warning and len(longest_transcripts['Transcript stable ID'].unique()) > 1:
            print('More than one longest transcript for {}, selecting arbitrary one: {}'.format(gene, transcript))

    transcript_exons = exon_data[exon_data['Transcript stable ID'] == transcript].sort_values(
        'Exon rank in transcript')
    exon_ranges = transcript_exons[['Genomic coding start', 'Genomic coding end']].values
    strand = transcript_exons.iloc[0]['Strand']
    if strand == -1:
        # Swap the start and end to match direction of gene
        exon_ranges[:, [0, 1]] = exon_ranges[:, [1, 0]]

    exon_ranges = exon_ranges.astype(int)

    return exon_ranges, transcript, strand, transcript_exons


def get_positions_from_ranges(exon_ranges, strand):
    cds_positions = []
    for start, end in exon_ranges:
        if strand == 1:
            cds_positions.extend(list(range(start, end + 1)))
        else:
            cds_positions.extend(list(range(start, end - 1, -1)))
    return cds_positions


def get_gene_kmers_from_exon_ranges(chromosome_sequence, chrom_offset, exon_ranges, strand, ks,
                                    reference=None, chrom=None):
    assert isinstance(exon_ranges, np.ndarray)

    if len(ks) == 0:
        max_k = 0
    else:
        max_k = max(ks)

    if max_k == 0:
        context = 0
    else:
        context = int((max_k-1)/2)

    gene_start = int(exon_ranges.min() - context)  # Give one extra base for context
    gene_end = int(exon_ranges.max() + context)  # Give one extra base for context

    if chromosome_sequence is not None:
        assert chrom_offset is not None
        gene_sequence = chromosome_sequence[
                        gene_start - chrom_offset:gene_end].upper()
    else:
        f = FastaFile(reference)
        try:
            gene_sequence = f[str(chrom)][
                            gene_start - 1:gene_end].upper()  #  Another -1 to move to zero based coordinates.
        except KeyError as e:
            print('Did not recognize chromosome', chrom)
            return []

    gene_kmers = defaultdict(list)
    exonic_gene_sequence = ""
    for start, end in exon_ranges:

        if strand == 1:
            context_seqs_before = []
            context_seqs_after = []
            s_, e_ = start - gene_start, end - gene_start + 1
            sequence = gene_sequence[s_:e_]  # Exon sequence.
            for i in range(1, context+1):
                context_seqs_before.append(gene_sequence[s_ - i:e_ - i])  # Bases i before
                context_seqs_after.append(gene_sequence[s_ + i:e_ + i])  # Bases 1 after
        else:
            s_, e_ = end - gene_start, start - gene_start + 1  #  Have to swap since the order is in relative to gene read direction
            context_seqs_before = []
            context_seqs_after = []
            sequence = reverse_complement(gene_sequence[s_:e_])
            for i in range(1, context+1):
                context_seqs_after.append(reverse_complement(gene_sequence[s_ - i:e_ - i]))  # Bases i after - reverse strand.
                context_seqs_before.append(reverse_complement(gene_sequence[s_ + i:e_ + i]))  # Bases 1 before - reverse strand

        exonic_gene_sequence += sequence
        for k in ks:
            a = int((k-1)/2)
            zippers = [context_seqs_before[i] for i in range(a)] + [sequence] + [context_seqs_after[i] for i in range(a)]
            gene_kmers[k].extend([''.join(z) for z in zip(*zippers)])

    return gene_kmers, exonic_gene_sequence


def get_all_possible_single_nucleotide_mutations(gene_sequence, gene_kmers, sort_results=True):
    length = len(gene_sequence)
    assert length % 3 == 0, 'Transcript nucleotide sequence not multiple of 3'
    assert gene_kmers is not None
    bases = 'ACGT'
    results = []

    aa_length = int(length / 3)

    # Make a long sequence with all changes
    # For each base positions, change to each base.
    # Can join them all then translate at once.

    # There are 13 repeats of the gene sequence.
    # The first is the WT
    # 2-5 is first base position in codon, 6-9 is 2nd, 10-13 is 3rd
    # 2, 6, 10 are changing to A
    # 3, 7, 11 are changing to C
    # 4, 8, 12 are changing to G
    # 5, 9, 13 are changing to T
    #
    # So the ath codon represents:
    # n = a//aa_length  is the sequence number
    # r = a%aa_length is the residue number (add 1 to get 1-based coordinates)
    # m = (n-1)//4 gives the base position
    # b = (n-1)%4 gives the base change (index of A, C, G, T)
    extended_seq = gene_sequence
    for j in range(3):  # base position
        for b in bases:  # Base change
            new_seq = "".join([s if ii % 3 != j else b for (ii, s) in enumerate(gene_sequence)])
            extended_seq += new_seq

    extended_seq_translated = Seq(extended_seq).translate()

    for i in range(aa_length, aa_length * 13):
        residue = i % aa_length + 1
        n = i // aa_length
        base_position = (n - 1) // 4
        mut = bases[(n - 1) % 4]
        cds_pos = (residue - 1) * 3 + base_position + 1
        aa = extended_seq_translated[residue - 1]
        aa2 = extended_seq_translated[i]
        seq = extended_seq[(residue - 1) * 3:residue * 3]
        seq2 = extended_seq[i * 3:(i + 1) * 3]
        if seq != seq2:
            aachange = '{}{}{}'.format(aa, residue, aa2)
            codon_change = '{}>{}'.format(seq, seq2)
            if aa == aa2:
                effect = 'synonymous'
            elif aa2 == '*':
                effect = 'nonsense'
            else:
                effect = 'missense'
            d = {'residue': residue, 'base': base_position,
                 'ref': seq[base_position], 'mut': seq2[base_position], 'cdspos': cds_pos,
                 'aachange': aachange, 'aaref': str(aa), 'aamut': str(aa2), 'codon_change': codon_change,
                 'effect': effect}
            for k, kmers in gene_kmers.items():
                kmer = kmers[(residue - 1) * 3 + base_position]
                d['{}mer_ref'.format(k)] = kmer
            results.append(d)

    all_possible_mutations = pd.DataFrame(results)
    if sort_results:
        return all_possible_mutations.sort_values(['residue', 'base', 'mut']).reset_index(drop=True)
    else:
        return all_possible_mutations

