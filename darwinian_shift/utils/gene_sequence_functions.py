"""
Functions for converting exon locations to nucleotide sequences, for getting the nucleotide contexts for mutational
spectrum calculations, and for generating all possible coding mutations in a gene.
"""

import pandas as pd
import numpy as np
from collections import defaultdict
from pysam import FastaFile
from Bio.Seq import Seq
from .util_functions import reverse_complement


def get_genomic_ranges_for_gene(exon_data, gene=None, transcript=None, print_transcript_warning=False):
    """
    Given a gene name or an Ensembl transcript id, will select return the start and end of each exon in the reading
    direction of the gene.

    If transcript is not specified, will use the longest one.
    Biomart data needs Gene name, Transcript stable ID, Exon rank in transcript, Genomic coding start,
    Genomic coding end, Strand
    If transcript is not specified, also needs CDS Length

    Assumes that exons with null "Genomic coding start" have already been removed.

    :param exon_data: Dataframe of data from Ensembl Biomart. Needs to include Gene name, Transcript stable ID,
    Exon rank in transcript, Genomic coding start, Genomic coding end, Strand.
    :param gene: The name of the gene. If not given, need to provide transcript
    :param transcript: Ensemble transcript id. If not given, the longest transcript for the given gene will be used.
    :param print_transcript_warning: If True, will print a warning if there are multiple transcripts that are the
    longest for a gene. If this is the case, one transcript will arbitrarily be picked.
    :return Tuple (exon_ranges [array], transcript [string], strand [int, 1 or -1], transcript_exons [pandas dataframe])
    """
    if gene is None and transcript is None:
        raise TypeError('get_genomic_ranges_for_gene requires either a gene name or transcript id')

    if transcript is None:  # Pick the longest transcript for the gene.
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
    if len(transcript_exons) == 0:
        # no data for this transcript
        return [], transcript, None, None
    exon_ranges = transcript_exons[['Genomic coding start', 'Genomic coding end']].values
    strand = transcript_exons.iloc[0]['Strand']
    if strand == -1:
        # Swap the start and end to match direction of gene
        exon_ranges[:, [0, 1]] = exon_ranges[:, [1, 0]]

    exon_ranges = exon_ranges.astype(int)

    return exon_ranges, transcript, strand, transcript_exons


def get_positions_from_ranges(exon_ranges, strand):
    """
    Converts an array of exon starts and ends to a list of all exonic positions.
    :param exon_ranges: 2D Array. One row per exon, [start, end].
    :param strand: 1 or -1.
    :return: List of ints.
    """
    cds_positions = []
    for start, end in exon_ranges:
        if strand == 1:
            cds_positions.extend(list(range(start, end + 1)))
        else:
            # If on the reverse strand, need the positions to be from highest to lowest value.
            cds_positions.extend(list(range(start, end - 1, -1)))
    return cds_positions


def get_gene_kmers_from_exon_ranges(chromosome_sequence, chrom_offset, exon_ranges, strand, ks,
                                    reference=None, chrom=None):
    """
    Return the context bases for every position in the exonic sequence of the gene. These are used to calculate the
    mutational spectrum and assign expected mutation rates to all mutations.
    Multiple k values can be given.
    The context is symmetrical. So for example if k=3, then the context will include the base before and the base after
    the nucleotide of interest.

    :param chromosome_sequence: The nucleotide sequence of the chromosome. Does not have to be the entire chromosome,
    chrom_offset defines the starting position of the sequence.  If None, can supply reference and chrom and read the
    nucleotide sequence from a fasta file.
    :param chrom_offset: The starting position of the chromosome sequence given.
    :param exon_ranges: Array of exon positions. One row per exon, then the start and end of each exon. This is one of
    the outputs of the function get_genomic_ranges_for_gene
    :param strand: The strand of the gene.
    :param ks: List or set of ints (unique values only). The number of context bases to use for the mutational spectra.
    If an empty list, will only run with no context bases.
    :param reference: Path to a (compressed) fasta file containing the reference genome.
    Only required if chromsome_sequence is None.
    :param chrom: The chromosome for the gene. Only required if chromsome_sequence is None.
    :return: Tuple. (Dictionary, key=k, value=list of all k-mers in the exonic sequence of the gene.
    String exonic gene sequence)
    """
    assert isinstance(exon_ranges, np.ndarray)

    # Get the maximum context required.
    if len(ks) == 0:
        max_k = 0
    else:
        max_k = max(ks)

    if max_k == 0:
        context = 0
    else:
        context = int((max_k-1)/2)

    gene_start = int(exon_ranges.min() - context)  # Give extra bases for context
    gene_end = int(exon_ranges.max() + context)  # Give extra bases for context

    if gene_start < 1:
        raise ValueError('gene_start too close to start of chromosome. Cannot include required context bases for spectrum')

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

    # The kmers are put together by collecting sequences offset by 1 base.
    # For k=3, there is the gene sequence, and the gene sequence shifted one-base before and one-base after.
    # These sequences are then zipped together to create the nucleotide k-mers.
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
    """
    For an exonic gene sequence, returns a dataframe containing all possible single nucleotide mutations, including
    the effect (missense, synonymous, nonsense, stop_lost, start_lost) and the nucleotide context.
    :param gene_sequence: String, exonic gene sequence. Second output of the function get_gene_kmers_from_exon_ranges
    :param gene_kmers: Dictionary. First output of the function. get_gene_kmers_from_exon_ranges
    :param sort_results: If True, will return the dataframe sorted by position and the alternate nucleotide.
    :return: pandas dataframe.
    """
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

    for i in range(aa_length, aa_length * 13):  # 13 is not a typo, see comments above
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
            elif aa == '*':
                effect = 'stop_lost'
            elif residue == 1:
                effect = 'start_lost'
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

