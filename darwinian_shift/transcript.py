from collections import Counter, defaultdict
import numpy as np
from darwinian_shift.utils import reverse_complement, get_genomic_ranges_for_gene, get_positions_from_ranges, \
    get_gene_kmers_from_exon_ranges, get_all_possible_single_nucleotide_mutations

class NoTranscriptError(ValueError): pass
class CodingTranscriptError(ValueError): pass

class Transcript:
    """
    A class to calculate and store the transcript nucleotide sequence, kmer counts for mutational spectra, and the
    mutations in the transcript.
    """
    def __init__(self, project, gene=None, transcript_id=None, genomic_sequence_chunk=None, offset=None,
                 region_exons=None, region_mutations=None):
        """

        :param project: DarwinianShift object containing the mutation data and reference data.
        :param gene: Gene associated with the transcript. Required if transcript_id is None.
        :param transcript_id: The Ensembl transcript id. Required if gene is None.
        :param genomic_sequence_chunk: Optional. Can provide a part of the chromosome sequence containing this transcript
        instead of going back to the fasta file. Is more efficient for processing multiple transcripts.
        :param offset: The chromosomal start position of the genomic_sequence_chunk
        :param region_exons: Dataframe of exon locations including the transcript exons.
        :param region_mutations: Dataframe of mutations including mutations overlapping with this transcript.
        """
        self.project = project
        self.transcript_id = transcript_id
        self.gene = gene
        self.genomic_sequence_chunk = genomic_sequence_chunk
        self.offset = offset
        if region_exons is None:
            self.region_exons = self.project.exon_data
        else:
            self.region_exons = region_exons

        self.chrom = None
        self.exon_ranges = None
        self.strand = None
        self.chromosomal_positions = None
        self.kmers = None
        self.nuc_sequence = None
        self.length = None
        self.transcript_ref_kmer_counts = None
        self.transcript_obs_kmer_counts = None
        self.transcript_data_locs = None
        self.transcript_mutations = None
        self.all_possible_sn_mutations = None

        self.mismatches = 0
        self.dedup_mismatches = 0

        self._get_sequences()

        if self.project.aa_mut_input:
            if self.transcript_id is not None and 'transcript_id' in self.project.data.columns:
                self.region_mutations = self.project.data[self.project.data['transcript_id'] == self.transcript_id]
            elif self.gene is not None and 'gene' in self.project.data.columns:
                self.region_mutations = self.project.data[self.project.data['gene'] == self.gene]
            else:
                raise ValueError('Must provide either a gene name or transcript_id')
        elif region_mutations is None:
            self.region_mutations = self.project.data[self.project.data['chr'] == self.chrom]
        else:
            self.region_mutations = region_mutations

    def _get_sequences(self):
        """
        Loads the nucleotide sequence of the transcript and counts the kmers for the mutational spectrum calculations.
        Sets various attributes of the Transcript object.
        :return: None
        """
        if self.gene is not None:
            transcripts = self.project.gene_transcripts_map[self.gene]
            if len(transcripts) == 0:
                self.transcript_id = None
            else:
                self.transcript_id = list(transcripts)[0]
                if len(transcripts) > 1:
                    print('More than one transcript used for gene {}. Using transcript {}'.format(self.gene,
                                                                                                  self.transcript_id))

        if self.gene is None and self.transcript_id is None:
            raise ValueError('Please provide a transcript id or gene name')

        self.exon_ranges, self.transcript_id, self.strand, transcript_exons = get_genomic_ranges_for_gene(self.region_exons,
                                                                                        self.gene,
                                                                         transcript=self.transcript_id)
        if self.transcript_id is None:
            raise NoTranscriptError('No transcript for {}'.format(self.gene, self.transcript_id))
        elif transcript_exons is None:
            if self.gene is not None:
                raise NoTranscriptError('No data for gene/transcript {}/{} in exon_data'.format(self.gene,
                                                                                            self.transcript_id))
            else:
                raise NoTranscriptError('No data for transcript {} in exon_data'.format(self.transcript_id))

        self.chromosomal_positions = get_positions_from_ranges(self.exon_ranges, self.strand)

        self.chrom = transcript_exons.iloc[0]['Chromosome/scaffold name']
        if self.gene is None:
            self.gene = transcript_exons.iloc[0]['Gene name']

        self.kmers, self.nuc_sequence = get_gene_kmers_from_exon_ranges(self.genomic_sequence_chunk,
                                                                        self.offset,
                                                                        self.exon_ranges,
                                                                        self.strand,
                                                                        self.project.ks,
                                                                        reference=self.project.reference_fasta,
                                                                        chrom=self.chrom)
        self.genomic_sequence_chunk = None  # Remove reference so chunk can be deleted from memory

        if len(self.nuc_sequence) % 3 != 0:
            raise CodingTranscriptError('Transcript nucleotide sequence not multiple of 3, {}:{}. Cannot run.'.format(
                self.gene, self.transcript_id))

        self.transcript_ref_kmer_counts = {}
        for k in self.kmers:
            if k == 1:
                self.transcript_ref_kmer_counts[k] = Counter(self.nuc_sequence)
            else:
                self.transcript_ref_kmer_counts[k] = Counter(self.kmers[k])

        self.length = len(self.nuc_sequence)

    def get_possible_mutations(self, return_copy=True):
        """
        Creates a dataframe containing all possible exonic single nucleotide substitutions in the transcript.
        The dataframe is stored in self.all_possible_sn_mutations.

        :param return_copy: Will return a copy of self.all_possible_sn_mutations instead of the original
        :return: Pandas dataframe
        """
        if self.all_possible_sn_mutations is None:
            if self.kmers is None:
                self._get_sequences()
            self.all_possible_sn_mutations = get_all_possible_single_nucleotide_mutations(self.nuc_sequence,
                                                                                          self.kmers)
            self._get_chrom_pos_from_cds_pos()

            # Correct ref and mut for strand. Also store the ref and mut in strand orientation
            self.all_possible_sn_mutations['strand_ref'] = self.all_possible_sn_mutations['ref']
            self.all_possible_sn_mutations['strand_mut'] = self.all_possible_sn_mutations['mut']
            if self.strand == -1:

                self.all_possible_sn_mutations['ref'] = [a for a in reverse_complement(
                    "".join(self.all_possible_sn_mutations['ref']), keep_order=True)]
                self.all_possible_sn_mutations['mut'] = [a for a in reverse_complement(
                    "".join(self.all_possible_sn_mutations['mut']), keep_order=True)]

            self.all_possible_sn_mutations['transcript_id'] = self.transcript_id

        if return_copy:
            return self.all_possible_sn_mutations.copy()
        else:
            return self.all_possible_sn_mutations

    def _get_chrom_pos_from_cds_pos(self):
        """
        Converts CDS positions in the transcript to chromosomal positions for all mutations in the
        self.all_possible_sn_mutations dataframe.
        :return:
        """
        cds_positions = self.all_possible_sn_mutations['cdspos']
        chrom_positions = self.chromosomal_positions
        positions = np.empty(len(self.all_possible_sn_mutations), dtype=int)
        last_c = -1
        last_pos = -1
        for i, c in enumerate(cds_positions):
            if c == last_c:  # Another mutation on the same nucleotide as the previous
                positions[i] = last_pos
            else:
                last_pos = chrom_positions[c - 1]
                positions[i] = last_pos
            last_c = c
        self.all_possible_sn_mutations['pos'] = positions

    def get_observed_mutations(self, return_copy=True):
        """
        Return a dataframe of all exonic mutations overlapping with the transcript.
        :param return_copy:
        :return: Pandas dataframe
        """
        if self.project.aa_mut_input:
            # Assume all mutations labelled with the transcript or gene are those to keep.
            self.transcript_mutations = self.region_mutations
        else:
            if self.transcript_data_locs is None:
                self.transcript_data_locs = self.region_mutations.index[
                    self.region_mutations['pos'].isin(self.chromosomal_positions)]

            self.transcript_mutations = self.region_mutations.loc[self.transcript_data_locs]

        if return_copy:
            return self.transcript_mutations.copy()
        else:
            return self.transcript_mutations

    def _get_kmer_counts_for_observed_mutations(self):
        """
        Count the number of mutations in each category in the mutational spectrum.
        E.g. for a trinucleotide transcript, count the number of AAA>ACA, the number of AAC>ACC etc. for all of the
        different categories in the spectrum.

        :return: Dictionary of dictionaries of collections.Counter.
        Structure is {True/False: {1: {'A>C': 3, .... } ... } ... }
        where the True/False = spectrum.deduplicate_spectrum, the number is spectrum.k, and the innermost part is the
        Counter of mutation types.
        """
        if self.transcript_obs_kmer_counts is None:
            # If lots of spectra with the same k, may be worth refactoring to collect based on k, not for each spectrum
            spectra = self.project.spectra
            dedup_spectra = [s for s in spectra if (not s.precalculated and s.deduplicate_spectrum)]
            non_depdup_spectra = [s for s in spectra if (not s.precalculated and not s.deduplicate_spectrum)]
            observed_mutations = self.get_observed_mutations()
            if self.strand == -1:
                observed_mutations['strand_mut'] = [a for a in reverse_complement(
                    "".join(observed_mutations['mut']), keep_order=True)]
                observed_mutations['strand_ref'] = [a for a in reverse_complement(
                    "".join(observed_mutations['ref']), keep_order=True)]
            else:
                observed_mutations['strand_mut'] = observed_mutations['mut']
                observed_mutations['strand_ref'] = observed_mutations['ref']
            if dedup_spectra:
                dedup_mutations = observed_mutations.drop_duplicates(subset=['chr', 'pos', 'ref', 'mut'])
            else:
                dedup_mutations = None

            chrom_positions = self.chromosomal_positions
            kmers = self.kmers
            self.transcript_obs_kmer_counts = defaultdict(lambda: defaultdict(lambda: Counter()))
            # structure {True/False: {1: {'A>C': 3, ....   where the True/False = s.deduplicate_spectrum

            self.mismatches = 0
            if non_depdup_spectra:
                last_p = -1
                last_ref = None
                last_kmer = None
                ks = {s.k for s in non_depdup_spectra}  # Use a set to only get unique k values
                for p, r, m in observed_mutations[['pos', 'strand_ref', 'strand_mut']].sort_values('pos').values:
                    if p == last_p:
                        if r == last_ref:
                            for k in ks:
                                self.transcript_obs_kmer_counts[False][k]["{}>{}".format(last_kmer[k], m)] += 1
                        else:
                            self.mismatches += 1
                    else:
                        pidx = chrom_positions.index(p)
                        last_kmer = {k: kmers[k][pidx] for k in ks}
                        last_ref = self.nuc_sequence[pidx]

                        if r == last_ref:
                            for k in ks:
                                self.transcript_obs_kmer_counts[False][k]["{}>{}".format(last_kmer[k], m)] += 1
                        else:
                            self.mismatches += 1
                    last_p = p

            self.dedup_mismatches = 0
            if dedup_spectra:
                ks = {s.k for s in dedup_spectra}
                last_p = -1
                last_ref = None
                last_kmer = None
                for p, r, m in dedup_mutations[['pos', 'strand_ref', 'strand_mut']].sort_values('pos').values:
                    if p == last_p:
                        if r == last_ref:
                            for k in ks:
                                self.transcript_obs_kmer_counts[True][k]["{}>{}".format(last_kmer[k], m)] += 1
                        else:
                            self.dedup_mismatches += 1
                    else:
                        pidx = chrom_positions.index(p)
                        last_kmer = {k: kmers[k][pidx] for k in ks}
                        last_ref = self.nuc_sequence[pidx]

                        if r == last_ref:
                            for k in ks:
                                self.transcript_obs_kmer_counts[True][k]["{}>{}".format(last_kmer[k], m)] += 1
                        else:
                            self.dedup_mismatches += 1
                    last_p = p

        return self.transcript_obs_kmer_counts