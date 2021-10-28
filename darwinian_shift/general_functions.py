import numpy as np
import pandas as pd
from collections import defaultdict
from statsmodels.stats.multitest import multipletests
import pybedtools
from adjustText import adjust_text
from copy import deepcopy
from pysam import FastaFile
import matplotlib.pylab as plt
from darwinian_shift.section import Section, NoMutationsError
from darwinian_shift.transcript import Transcript, NoTranscriptError, CodingTranscriptError
from darwinian_shift.mutation_spectrum import MutationalSpectrum, GlobalKmerSpectrum, read_spectrum, \
    EvenMutationalSpectrum
from darwinian_shift.statistics import CDFPermutationTest
from darwinian_shift.reference_data.reference_utils import get_source_genome_reference_file_paths
from darwinian_shift.lookup_classes.errors import MetricLookupException
from darwinian_shift.utils import read_sbs_from_vcf


BED_COLS = ['Chromosome/scaffold name', 'Genomic coding start', 'Genomic coding end',
            'Gene name', 'Transcript stable ID']


class DarwinianShift:
    """
    Class to process and store data for statistical testing of selection
    Calculates the mutation spectrum from the data

    Tests of individual genes/transcripts can run using the run_gene and run_transcript methods.
    """

    def __init__(self,
                 # Input data
                 data,

                 # Reference data.
                 # If supplying the name of the source_genome (e.g. homo_sapiens), it will use pre-downloaded ensembl data from the reference_data directory,
                 # Specify the ensembl release number to use a specific version, or it will use the most recent release that has been downloaded.
                 # Alternatively, provide the paths to an exon file and a reference genome fa.gz file.
                 # This must be compressed with bgzip and with a faidx index file
                 # the exon file or reference file will take precedence over the source_genome/ensembl_release
                 # e.g. can specify source_genome and exon file to use the standard genome with a custom set of exons
                 # Special case, can specify source_genome="GRCh37" to use that human build.
                 source_genome=None, ensembl_release=None,
                 exon_file=None, reference_fasta=None,
                 # Measurements and statistics
                 lookup=None,  # A class which will return metric value. See lookup_classes directory
                 statistics=None,
                 # Options
                 sections=None,  # Tab separated file or dataframe.

                 # PDB file options
                 pdb_directory=None,
                 sifts_directory=None,
                 download_sifts=False,

                 # MutationalSpectrum options
                 spectra=None,  # Predefined spectra

                 gene_list=None, transcript_list=None,
                 deduplicate=False,

                 # If false, will run over all transcripts in exon data file that have a mutation.
                 # If true, will pick the longest transcript for each gene.
                 use_longest_transcript_only=True,

                 # These will exclude from the tests, not the spectrum.
                 excluded_positions=None,  # dict, key=chrom, val=positions. E.g. {'1': [1, 2, 3]}
                 excluded_mutation_types=None,
                 included_mutation_types=None,  # This has priority over excluded mutation_types

                 chunk_size=50000000,  # How much of a chromosome will be processed at once.
                 # Bigger=quicker but more memory
                 low_mem=True,  # Will not store sequence data during spectrum calculation for later reuse.

                 # Options for testing/reproducibility
                 random_seed=None,  # Int. Set this to return consistent results.
                 testing_random_seed=None,  # Int. Set to get same results, even if changing order or spectra/tests.

                 verbose=False
                 ):
        """

        :param data:
        :param source_genome:
        :param ensembl_release:
        :param exon_file:
        :param reference_fasta:
        :param lookup:
        :param statistics:
        :param sections:
        :param pdb_directory:
        :param sifts_directory:
        :param download_sifts:
        :param spectra: Spectrum object or list of Spectrum objects. The mutational spectrum is calculated from the data.
        To skip this process, use EvenMutationalSpectrum or "Even", although this will impact the statistical results.
        Default is a global trinucleotide spectrum.
        :param gene_list:
        :param transcript_list:
        :param deduplicate:
        :param use_longest_transcript_only:
        :param excluded_positions:
        :param excluded_mutation_types:
        :param included_mutation_types:
        :param chunk_size:
        :param low_mem:
        :param random_seed:
        :param testing_random_seed:
        :param verbose:
        """

        self.verbose=verbose
        self.low_mem=low_mem
        if random_seed is not None:
            np.random.seed(random_seed)

        if isinstance(data, pd.DataFrame):
            self.data = data
        elif isinstance(data, str):  # File path given
            if data.endswith('.vcf') or data.endswith('.vcf.gz'):
                self.data = read_sbs_from_vcf(data)
            else:  # Assume the data is in a tab-delimited file including "chr", "pos", "ref" and "mut" columns.
                self.data = pd.read_csv(data, sep="\t")

        # Filter non-snv mutations.
        bases = ['A', 'C', 'G', 'T']
        # Take copy here to deal with pandas SettingWithCopyWarning
        # From here on, want to be editing views of the following copy of the data and ignore previous unfiltered data.
        self.data = self.data[(self.data['ref'].isin(bases)) & (self.data['mut'].isin(bases))].copy()
        self.data.loc[:, 'chr'] = self.data['chr'].astype(str)
        self.data = self.data.reset_index(drop=True)  # Make sure every mutation has a unique index

        if self.verbose:
            # For tracking which mutations are included in used transcripts.
            # Slows process, so only done if verbose=True.
            self.data.loc[:, 'included'] = False

        self.chromosomes = self.data['chr'].unique()

        if deduplicate:
            self._deduplicate_data()
        self.excluded_positions = excluded_positions
        self.excluded_mutation_types = excluded_mutation_types
        self.included_mutation_types = included_mutation_types

        if pdb_directory is None:
            self.pdb_directory = "."
        else:
            self.pdb_directory = pdb_directory
        if sifts_directory is None:
            self.sifts_directory = "."
        else:
            self.sifts_directory = sifts_directory
        self.download_sifts = download_sifts

        self.lookup = lookup
        if hasattr(self.lookup, "setup_project"):
            self.lookup.setup_project(self)

        exon_file, reference_fasta = self._get_reference_data(source_genome, ensembl_release, exon_file, reference_fasta)

        self.exon_data = pd.read_csv(exon_file, sep="\t")
        self.exon_data.loc[:, 'Chromosome/scaffold name'] = self.exon_data['Chromosome/scaffold name'].astype(str)
        # This filter helps to remove obscure scaffolds so they will not be matched.
        self.exon_data = self.exon_data[self.exon_data['Chromosome/scaffold name'].isin(self.chromosomes)]
        # Removes exons without any coding bases
        self.exon_data = self.exon_data[~pd.isnull(self.exon_data['Genomic coding start'])]
        self.exon_data['Genomic coding start'] = self.exon_data['Genomic coding start'].astype(int)
        self.exon_data['Genomic coding end'] = self.exon_data['Genomic coding end'].astype(int)

        # If transcripts are not specified, may be used to check if excluded mutations are in alternative transcripts
        self.unfiltered_exon_data = None

        self.use_longest_transcript_only = use_longest_transcript_only
        self.gene_list = gene_list
        self.transcript_list = transcript_list

        self.transcript_objs = {}
        # If transcripts not specified, will use longest transcript per gene to calculate signature.
        self.signature_transcript_list = None
        self.alternative_transcripts = None

        self.reference_fasta = reference_fasta
        self.transcript_gene_map = {}
        self.gene_transcripts_map = defaultdict(set)

        self.spectra = None
        self._process_spectra(spectra)

        if len(set([s.name for s in self.spectra])) < len(self.spectra):
            raise ValueError('More than one MutationSpectrum with the same name. Provide unique names to each.')

        self.ks = set([getattr(s, 'k', None) for s in self.spectra])   # Kmers to use for the spectra. E.g. 3 for trinucleotides
        self.ks.discard(None)

        self.chunk_size = chunk_size
        self.section_transcripts = None  # Transcripts required for the sections.
        self.sections = None
        additional_results_columns = self._get_sections(sections)

        self._set_up_exon_data()

        self.checked_included = False
        self.total_spectrum_ref_mismatch = 0
        if any([(isinstance(s, GlobalKmerSpectrum) and not s.precalculated) for s in self.spectra]):
            # Collect data for any signatures that need to be calculated from the global data
            if not self.use_longest_transcript_only:
                print('WARNING: Using multiple transcripts per gene may double count mutants in the spectrum')
                print('Can set use_longest_transcript_only=True and save the spectrum, then run using the pre-calculated spectrum.')
            self._calculate_spectra(self.verbose)
            if self.verbose:
                self._check_non_included_mutations()

        if statistics is None:
            # Use the default statistics only
            # Use the cdf permutation test as the default as it is appropriate for a wide range of null distributions.
            self.statistics = [CDFPermutationTest()]
        elif not isinstance(statistics, (list, tuple)):
            self.statistics = [statistics]
        else:
            self.statistics = statistics

        if testing_random_seed is not None:
            for s in self.statistics:
                try:
                    s.set_testing_random_seed(testing_random_seed)
                except AttributeError as e:
                    # Not a test that uses a seed
                    pass

        if len(set([s.name for s in self.statistics])) < len(self.statistics):
            raise ValueError('More than one statistic with the same name. Provide unique names to each.')

        # Results
        self.result_columns = ['gene', 'transcript_id', 'chrom', 'section_id', 'num_mutations']
        self.result_columns.extend(additional_results_columns)

        self.results = None
        self.scored_data = []

    def _get_reference_data(self, source_genome, ensembl_release, exon_file, reference_fasta):
        if source_genome is None and (exon_file is None or reference_fasta is None):
            raise TypeError('Must provide a source_genome or an exon_file and a reference_fasta')
        if source_genome is not None and (exon_file is None or reference_fasta is None):
            try:
                exon_file1, reference_file1 = get_source_genome_reference_file_paths(source_genome, ensembl_release)
            except Exception as e:
                print('Reference files not found for source_genome:{} and ensembl_release:{}'.format(source_genome, ensembl_release))
                print('Try downloading the data using download_reference_data_from_latest_ensembl or download_grch37_reference_data')
                print('Or download manually and specify the file paths using exon_file and reference_fasta')
                raise e
        if exon_file is None:
            exon_file = exon_file1
        if reference_fasta is None:
            reference_fasta = reference_file1

        if self.verbose:
            print('exon_file:', exon_file)
            print('reference_fasta:', reference_fasta)

        return exon_file, reference_fasta

    def _deduplicate_data(self):
        self.data.drop_duplicates(subset=['chr', 'pos', 'ref', 'mut'], inplace=True)

    def _process_spectra(self, spectra):
        if spectra is None:
            self.spectra = [
                GlobalKmerSpectrum()
            ]
        elif isinstance(spectra, (list, tuple, set)):
            try:
                processed_spectra = []
                for s in spectra:
                    if isinstance(s, MutationalSpectrum):
                        processed_spectra.append(s)
                    elif isinstance(s, str):
                        processed_spectra.append(read_spectrum(s))
                    else:
                        raise TypeError(
                            'Each spectrum must be a MutationalSpectrum object or file path to precalculated spectrum, {} given'.format(
                                type(s)))
                self.spectra = processed_spectra
            except TypeError as e:
                raise TypeError(
                    'Each spectrum must be a MutationalSpectrum object or file path to precalculated spectrum, {} given'.format(
                        type(s)))
        elif isinstance(spectra, MutationalSpectrum):
            self.spectra = [spectra]
        elif isinstance(spectra, str):
            if spectra.lower() == 'even':
                self.spectra = [EvenMutationalSpectrum()]
            else:
                # File path
                self.spectra = [read_spectrum(spectra)]
        else:
            raise TypeError(
                'spectra must be a MutationalSpectrum object or list of MutationalSpectrum objects, {} given'.format(
                    type(spectra)))

        for i, s in enumerate(self.spectra):
            s.set_project(self)
            s.reset()  # Makes sure counts start from zero in case using same spectrum object again.
            # Run s.fix_spectrum() for each spectrum to prevent reset.

    def _get_sections(self, sections):
        additional_results_columns = []
        if sections is not None:

            if isinstance(sections, str):
                self.sections = pd.read_csv(sections, sep="\t")
            elif isinstance(sections, pd.DataFrame):
                self.sections = sections
            else:
                raise ValueError('Do not recognize input sections. Should be file path or pandas dataframe')
            print(len(self.sections), 'sections')
            additional_results_columns = [c for c in self.sections.columns if c != 'transcript_id']
            self.section_transcripts = self.sections['transcript_id'].unique()

        return additional_results_columns

    def make_section(self, section_dict=None, transcript_id=None, gene=None, lookup=None, **kwargs):
        """

        :param section_dict: Can be dictionary or pandas Series. Must contain "transcript_id" or "gene" and any
        other information required to define a section (e.g. start/end or pdb_id and pdb_chain)
        :param transcript_id:
        :param gene:
        :param lookup: A Lookup object, if given will override the lookup for the DarwinianShift object. Alternatively,
        this can be provided in the section_dict under the key "lookup"
        :return:
        """
        if section_dict is not None:
            section_dict_copy = section_dict.copy()  # Make sure not to edit the original dictionary/series
            if 'transcript_id' in section_dict_copy:
                transcript_id = section_dict_copy.pop('transcript_id')
            else:
                transcript_id = self.get_transcript_id(section_dict_copy['gene'])
                if isinstance(transcript_id, set):
                    transcript_id = list(transcript_id)[0]
                    print('Multiple transcripts for gene {}. Running {}'.format(section_dict_copy['gene'], transcript_id))
        elif transcript_id is not None:
            section_dict_copy = {}
        elif gene is not None:
            transcript_id = self.get_transcript_id(gene)
            if isinstance(transcript_id, set):
                transcript_id = list(transcript_id)[0]
                print('Multiple transcripts for gene {}. Running {}'.format(gene, transcript_id))
            elif transcript_id is None:
                raise ValueError('No transcript associated with gene {}'.format(gene))
            section_dict_copy = {}
        else:
            raise ValueError('Must provide a section_dict, transcript_id or gene')
        transcript_obj = self.get_transcript_obj(transcript_id)
        if lookup is not None:
            section_dict_copy['lookup'] = lookup
        for k, v in kwargs.items():
            section_dict_copy[k] = v
        sec = Section(transcript_obj, **section_dict_copy)
        return sec

    def get_transcript_obj(self, transcript_id):
        t = self.transcript_objs.get(transcript_id, None)
        if t is None:
            t = self.make_transcript(transcript_id=transcript_id)
        return t

    def get_transcript_id(self, gene):
        t = self.gene_transcripts_map.get(gene, None)
        if t is not None and len(t) == 1:
            return list(t)[0]
        else:
            return t

    def get_gene_name(self, transcript_id):
        return self.transcript_gene_map.get(transcript_id, None)

    def get_gene_list(self):
        """
        If no gene_list or transcript_list has been given for __init__,
        this will be a list of the genes that overlap with the data.
        :return:
        """
        if self.gene_list is not None:
            return self.gene_list
        else:
            return list(self.gene_transcripts_map.keys())

    def get_transcript_list(self):
        """
        If no gene_list or transcript_list has been given for __init__,
        this will be a list of the genes that overlap with the data.
        :return:
        """
        if self.transcript_list is not None:
            return self.transcript_list
        else:
            return list(self.transcript_gene_map.keys())

    def make_transcript(self, gene=None, transcript_id=None, genomic_sequence_chunk=None, offset=None,
                        region_exons=None, region_mutations=None):
        """
        Makes a new transcript object and adds to the gene-transcript maps
        :param gene:
        :param transcript_id:
        :return:
        """
        if gene is None and transcript_id is None:
            raise ValueError('Need to supply gene or transcript_id')
        t = Transcript(self, gene=gene, transcript_id=transcript_id,
                       genomic_sequence_chunk=genomic_sequence_chunk, offset=offset, region_exons=region_exons,
                       region_mutations=region_mutations)
        self.transcript_gene_map[t.transcript_id] = t.gene
        t.get_observed_mutations()
        self.gene_transcripts_map[t.gene].add(t.transcript_id)
        if self.verbose:
            self.data.loc[t.transcript_data_locs, 'included'] = True
        if not self.low_mem:
            self.transcript_objs[t.transcript_id] = t
        return t

    def get_overlapped_transcripts(self, mut_data, exon_data):
        mut_data = mut_data.sort_values(['chr', 'pos'])
        mut_bed = pybedtools.BedTool.from_dataframe(mut_data[['chr', 'pos', 'pos']])
        exon_bed = pybedtools.BedTool.from_dataframe(exon_data[BED_COLS])
        try:
            intersection = exon_bed.intersect(mut_bed).to_dataframe()
        except pd.errors.EmptyDataError as e:
            return None

        if not intersection.empty:
            intersection.rename({'score': 'Transcript stable ID', 'name': 'Gene name'}, inplace=True, axis=1)
            intersection = intersection[['Gene name', 'Transcript stable ID']]
            return intersection.drop_duplicates()
        else:
            return None

    def _set_up_exon_data(self):
        if self.transcript_list is not None:
            if self.section_transcripts:
                transcript_list_total = set(self.transcript_list).union(self.section_transcripts)
            else:
                transcript_list_total = self.transcript_list
            self.exon_data = self.exon_data[self.exon_data['Transcript stable ID'].isin(transcript_list_total)]
            self.exon_data = self.exon_data.sort_values(['Chromosome/scaffold name', 'Genomic coding start'])
            if set(self.transcript_list).difference(self.exon_data['Transcript stable ID'].unique()):
                raise ValueError('Not all requested transcripts found in exon data.')
        elif self.gene_list is not None:  # Given genes. Will use longest transcript for each.
            if self.section_transcripts:
                self.exon_data = self.exon_data[(self.exon_data['Gene name'].isin(self.gene_list)) |
                                                (self.exon_data['Transcript stable ID'].isin(self.section_transcripts))]
            else:
                self.exon_data = self.exon_data[self.exon_data['Gene name'].isin(self.gene_list)]
            if self.use_longest_transcript_only:
                self._remove_unused_transcripts()
            self.exon_data = self.exon_data.sort_values(['Chromosome/scaffold name', 'Genomic coding start'])
            if set(self.gene_list).difference(self.exon_data['Gene name'].unique()):
                raise ValueError('Not all requested genes found in exon data.')
        else:
            overlapped_transcripts = self.get_overlapped_transcripts(self.data, self.exon_data)
            if overlapped_transcripts is None:
                raise ValueError('No transcripts found matching the mutations')
            self.exon_data = self.exon_data[
                self.exon_data['Transcript stable ID'].isin(overlapped_transcripts['Transcript stable ID'])]
            self.exon_data = self.exon_data.sort_values(['Chromosome/scaffold name', 'Genomic coding start'])
            if self.use_longest_transcript_only:
                self._remove_unused_transcripts()
                if self.verbose:
                    print(len(self.exon_data['Gene name'].unique()), 'genes')

        # Record the transcripts that are associated with each gene so they can be looked up
        ed = self.exon_data[['Gene name', 'Transcript stable ID']].drop_duplicates()
        for g, t in zip(ed['Gene name'], ed['Transcript stable ID']):
            self.transcript_gene_map[t] = g
            self.gene_transcripts_map[g].add(t)

    def _remove_unused_transcripts(self):
        if self.verbose:
            self.unfiltered_exon_data = self.exon_data.copy()

        transcripts = set()
        for gene, gene_df in self.exon_data.groupby('Gene name'):
            longest_cds = gene_df['CDS Length'].max()
            transcript_df = gene_df[gene_df['CDS Length'] == longest_cds]
            if len(transcript_df) > 0:
                transcripts.add(transcript_df.iloc[0]['Transcript stable ID'])

        self.signature_transcript_list = list(transcripts)

        if self.section_transcripts is not None:
            transcripts = transcripts.union(self.section_transcripts)

        self.exon_data = self.exon_data[self.exon_data['Transcript stable ID'].isin(transcripts)]

    def _check_non_included_mutations(self):
        # annotated mutations are those included in one of the given transcripts
        if not self.checked_included and self.unfiltered_exon_data is not None:
            non_annotated_mutations = self.data[~self.data['included']]
            overlapped = self.get_overlapped_transcripts(non_annotated_mutations, self.unfiltered_exon_data)
            if overlapped is not None:
                genes = overlapped['Gene name'].unique()
                print(len(non_annotated_mutations), 'mutations not in used transcripts, of which', len(overlapped),
                      'are exonic in an alternative transcript')
                print('Selected transcripts miss exonic mutations in alternative transcripts for:', genes)
                print('Look at self.alternative_transcripts for alternatives')
                self.alternative_transcripts = overlapped
            else:
                print(len(non_annotated_mutations), 'mutations not in used transcripts, of which 0 are exonic '
                                                    'in an alternative transcript')
        self.checked_included = True

    def _get_processing_chunks(self):
        # Divide the genes/transcripts into sections which are nearby in chromosomes
        chunks = []
        for chrom, chrom_df in self.exon_data.groupby('Chromosome/scaffold name'):
            chrom_data = self.data[self.data['chr'] == chrom]
            while len(chrom_df) > 0:
                first_pos = chrom_df['Genomic coding start'].min()
                chunk = chrom_df[chrom_df['Genomic coding start'] <= first_pos + self.chunk_size]
                chunk_transcripts = chunk['Transcript stable ID'].unique()
                chunk = chrom_df[chrom_df['Transcript stable ID'].isin(chunk_transcripts)]  # Do not split up transcripts
                last_pos = chunk['Genomic coding end'].max()
                chunk_data = chrom_data[(chrom_data['pos'] >= first_pos) & (chrom_data['pos'] <= last_pos)]
                chunks.append({
                    'chrom': str(chrom),
                    'start': int(first_pos), 'end': int(last_pos), 'transcripts': chunk_transcripts,
                    'exon_data': chunk, 'mut_data': chunk_data
                })
                # Remove the transcripts in last chunk
                chrom_df = chrom_df[~chrom_df['Transcript stable ID'].isin(chunk_transcripts)]

        return chunks

    def _chunk_iterator(self):
        chunks = self._get_processing_chunks()
        f = FastaFile(self.reference_fasta)
        if len(self.ks) == 0:
            max_k = 0
        else:
            max_k = max(self.ks)
        if max_k == 0:
            context = 0
        else:
            context = int((max_k - 1) / 2)

        for c in chunks:
            offset = c['start'] - context
            try:
                chunk_seq = f[c['chrom']][
                            offset - 1:c['end'] + context].upper()  #  Another -1 to move to zero based coordinates.
            except KeyError as e:
                print('Did not recognize chromosome', c['chrom'])
                continue

            for t in c['transcripts']:
                yield t, chunk_seq, offset, c['exon_data'], c['mut_data']

    def _calculate_spectra(self, verbose=False):

        for transcript_id, chunk_seq, offset, region_exons, region_mutations in self._chunk_iterator():
            if self.signature_transcript_list is not None and transcript_id not in self.signature_transcript_list:
                continue
            try:
                # Get the trinucleotides from the transcript.
                t = self.make_transcript(transcript_id=transcript_id, genomic_sequence_chunk=chunk_seq, offset=offset,
                                         region_exons=region_exons, region_mutations=region_mutations)
                for s in self.spectra:
                    s.add_transcript_muts(t)
                if t.mismatches > 0:
                    self.total_spectrum_ref_mismatch += t.mismatches
                elif t.dedup_mismatches > 0:  # If all signature deduplicate, just count those mismatches.
                    self.total_spectrum_ref_mismatch += t.dedup_mismatches
                if verbose:
                    print('{}:{}'.format(t.gene, t.transcript_id), end=" ")
            except (NoTranscriptError, CodingTranscriptError):
                pass
            except Exception as e:
                print('Failed to collect signature data from {}'.format(transcript_id))
                raise e

        if verbose:
            print()

        for sig in self.spectra:
            sig.get_spectrum()

        if self.total_spectrum_ref_mismatch > 0:
            print('Warning: {} mutations do not match reference base'.format(self.total_spectrum_ref_mismatch))

    def run_gene(self, gene, plot=False, spectra=None, statistics=None, start=None, end=None,
                 excluded_mutation_types=None, included_mutation_types=None,
                 included_residues=None, excluded_residues=None, pdb_id=None, pdb_chain=None, lookup=None,
                 **additional_kwargs):
        """
        Run analysis for a single gene
        :param gene:  Gene name.
        :param plot: Will plot the standard results of the analysis. If False, can still be plotted afterwards from
        the returned class object. Plotting afterwards also enables more plotting options.
        :param spectra: The mutational spectrum or spectra to use for the analysis. If None, will use the spectra
        of the project.
        :param statistics: The statistical tests to run. If None, will use the statistics of the project.
        :param start: Will exclude residues before this one from the analysis. If None, will start from the first
        residue of the protein.
        :param end: Will exclude residues after this one from the analysis. If None, will end at the last
        residue of the protein.
        :param excluded_mutation_types: Can be string or list of strings. Mutation types to exclude from the
        analysis. E.g. ['synonymous', 'nonsense']. If None, will use the excluded_mutation_types of the project.
        :param included_mutation_types: Can be string or list of strings. Mutation types to include in the
        analysis. E.g. ['synonymous', 'nonsense']. If None, will use the included_mutation_types of the project.
        :param included_residues: List or array of integers. The residues to analyse. If None, will analyse all
        residues (except those excluded by other arguments).
        :param excluded_residues: List or array of integers. The residues to exclude from the analysis.
        :param pdb_id: For analyses that use a protein structure. Four letter ID of the pdb file to use.
        :param pdb_chain: For analyses that use a protein structure. The chain to use for the analysis.
        :param lookup: The class object or function used to score the mutations. If None, will use the lookup of
        the project.
        :param additional_kwargs: Any additional attributes that will be assigned to the Section object created.
        These can be used by the lookup class.
        :return: Section object
        """
        gene_transcripts = self.gene_transcripts_map[gene]
        if len(gene_transcripts) == 0:
            transcript_obj = self.make_transcript(gene=gene)
            transcript_id = transcript_obj.transcript_id
        else:
            transcript_id = list(gene_transcripts)[0]
            if len(gene_transcripts) > 1:
                print('Multiple transcripts for gene {}. Running {}'.format(gene, transcript_id))

        return self.run_transcript(transcript_id, plot=plot, spectra=spectra, statistics=statistics,
                                   start=start, end=end,
                                   excluded_mutation_types=excluded_mutation_types,
                                   included_mutation_types=included_mutation_types, included_residues=included_residues,
                                   excluded_residues=excluded_residues, pdb_id=pdb_id, pdb_chain=pdb_chain,
                                   lookup=lookup, **additional_kwargs)

    def run_transcript(self, transcript_id, plot=False, spectra=None, statistics=None, start=None, end=None,
                       excluded_mutation_types=None, included_mutation_types=None,
                       included_residues=None, excluded_residues=None, pdb_id=None, pdb_chain=None, lookup=None,
                       **additional_kwargs):
        """
        Run analysis for a single transcript.
        :param transcript_id: Transcript id.
        :param plot: Will plot the standard results of the analysis. If False, can still be plotted afterwards from
        the returned class object. Plotting afterwards also enables more plotting options.
        :param spectra: The mutational spectrum or spectra to use for the analysis. If None, will use the spectra
        of the project.
        :param statistics: The statistical tests to run. If None, will use the statistics of the project.
        :param start: Will exclude residues before this one from the analysis. If None, will start from the first
        residue of the protein.
        :param end: Will exclude residues after this one from the analysis. If None, will end at the last
        residue of the protein.
        :param excluded_mutation_types: Can be string or list of strings. Mutation types to exclude from the
        analysis. E.g. ['synonymous', 'nonsense']. If None, will use the excluded_mutation_types of the project.
        :param included_mutation_types: Can be string or list of strings. Mutation types to include in the
        analysis. E.g. ['synonymous', 'nonsense']. If None, will use the included_mutation_types of the project.
        :param included_residues: List or array of integers. The residues to analyse. If None, will analyse all
        residues (except those excluded by other arguments).
        :param excluded_residues: List or array of integers. The residues to exclude from the analysis.
        :param pdb_id: For analyses that use a protein structure. Four letter ID of the pdb file to use.
        :param pdb_chain: For analyses that use a protein structure. The chain to use for the analysis.
        :param lookup: The class object or function used to score the mutations. If None, will use the lookup of
        the project.
        :param additional_kwargs: Any additional attributes that will be assigned to the Section object created.
        These can be used by the lookup class.
        :return: Section object
        """
        try:
            section = Section(self.get_transcript_obj(transcript_id), start=start, end=end,
                              pdb_id=pdb_id, pdb_chain=pdb_chain,
                              excluded_mutation_types=excluded_mutation_types,
                              included_mutation_types=included_mutation_types, included_residues=included_residues,
                            excluded_residues=excluded_residues, lookup=lookup, **additional_kwargs)
        except (CodingTranscriptError, NoTranscriptError) as e:
            print(type(e).__name__, e, '- Unable to run for', transcript_id)
            return None
        return self.run_section(section, plot=plot, spectra=spectra, statistics=statistics)

    def run_section(self, section, plot=False, verbose=False, spectra=None, statistics=None, lookup=None):
        """
        Run statistics and optionally plot plots for a section.
        The section can be a Section object, or a dictionary that defines the Section object to be made
        The spectra and statistics can be passed here, but other options for the Section (such as included/excluded
        mutation types) must be defined when the Section object is created or in the dictionary passed to the section arg
        :param section: Section object or dictionary with Section.__init__ kwargs.
        :param plot: Will plot the standard results of the analysis. If False, can still be plotted afterwards from
        the returned class object. Plotting afterwards also enables more plotting options.
        :param verbose: Will print section id and gene name when running.
        :param spectra: The mutational spectrum or spectra to use for the analysis. If None, will use the spectra of
        the project.
        :param statistics: The statistical tests to run. If None, will use the statistics of the project.
        :param lookup: A Lookup object, if given will override the lookup for the DarwinianShift object. Alternatively,
        this can be provided in the section_dict under the key "lookup"
        :return: Section object
        """
        if self.lookup is None and lookup is None:
            # No lookup defined for the project or given as an argument to this function.
            # See if one has been defined for the section

            if isinstance(section, (dict, pd.Series)):
                section_lookup = section.get('lookup', None)
            else:
                section_lookup = getattr(section, 'lookup', None)
            if section_lookup is None:
                raise ValueError('No lookup defined. Define one for the whole project using ' \
                                 'self.change_lookup() or provide one to this function.')
        try:
            if isinstance(section, (dict, pd.Series)):
                # Dictionary/series with attributes to define a new section
                section = self.make_section(section, lookup=lookup)
            elif lookup is not None:
                section.change_lookup_inplace(lookup)

            if verbose:
                print('Running', section.section_id, section.gene)
            section.run(plot_permutations=plot, spectra=spectra, statistics=statistics)
            if plot:
                section.plot()
            return section
        except (NoMutationsError, AssertionError, CodingTranscriptError, NoTranscriptError, MetricLookupException) as e:
            if isinstance(section, Section):
                print(type(e).__name__, e, '- Unable to run for', section.section_id)
            else:
                print(type(e).__name__, e, '- Unable to run for', section)
            return None

    def run_all(self, verbose=None, spectra=None, statistics=None):
        """
        Run analysis over all genes, transcripts or sections defined in the project.
        After running, the results can be seen in the 'results' attribute of the DarwinianShift object.
        The scores of each mutation can also be seen in the 'scored_data' attribute of the DarwinianShift object.
        :param verbose: Will print additional information.
        :param spectra: The mutational spectrum or spectra to use for the analysis. If None, will use the spectra of
        the project.
        :param statistics: The statistical tests to run. If None, will use the statistics of the project.
        :return:
        """
        if verbose is None:
            verbose = self.verbose
        results = []
        scored_data = []

        for t, chunk_seq, offset, region_exons, region_mutations in self._chunk_iterator():
            if self.sections is not None:  # Sections are defined.
                if t not in self.section_transcripts:
                    continue
            transcript_obj = self.transcript_objs.get(t)
            if transcript_obj is None:
                try:
                    transcript_obj = self.make_transcript(transcript_id=t, genomic_sequence_chunk=chunk_seq,
                                                      offset=offset, region_exons=region_exons,
                                                          region_mutations=region_mutations)
                except (NoTranscriptError, CodingTranscriptError) as e:
                    if verbose:
                        print(e)
                    continue

            if self.sections is None:  # If the sections to run are not defined, use the whole transcript as one section
                transcript_sections = [Section(transcript_obj)]
            else:
                transcript_sections_df = self.sections[self.sections['transcript_id'] == transcript_obj.transcript_id]
                transcript_sections = []
                for i, row in transcript_sections_df.iterrows():
                    transcript_sections.append(self.make_section(row))

            for section in transcript_sections:
                res = self.run_section(section, verbose=verbose, spectra=spectra, statistics=statistics)
                if res is not None:
                    scored_data.append(res.observed_mutations)
                    results.append(res.get_results_dictionary())

        if results:
            self.results = pd.DataFrame(results)
            for col in self.results.columns:
                if col.endswith('_pvalue'):
                    self.results[col.replace("pvalue", "qvalue")] = multipletests(self.results[col], method='fdr_bh')[1]

            self.scored_data = pd.concat(scored_data, ignore_index=True)

        if self.verbose:
            self._check_non_included_mutations()

    def _generate_sections(self):
        # Go through the process of getting the scores and sequences.
        # Just don't do the tests.
        # Useful for outputting to oncodrive or other external tools
        chunks = self._get_processing_chunks()
        f = FastaFile(self.reference_fasta)
        max_k = max(self.ks)
        if max_k == 0:
            context = 0
        else:
            context = int((max_k - 1) / 2)
        for c in chunks:
            offset = c['start'] - context
            try:
                chunk_seq = f[c['chrom']][
                            offset - 1:c['end'] + context].upper()  #  Another -1 to move to zero based coordinates.
            except KeyError as e:
                print('Did not recognize chromosome', c['chrom'])
                continue

            for t in set(c['transcripts']).intersection(self.transcript_list):
                transcript_obj = self.transcript_objs.get(t)
                if transcript_obj is None:
                    transcript_obj = self.make_transcript(transcript_id=t, genomic_sequence_chunk=chunk_seq,
                                                          offset=offset)
                    section = Section(transcript_obj)
                    self.sections[t].append(section)


    def get_spectrum(self, spectrum_name):
        for s in self.spectra:
            if s.name == spectrum_name:
                return s
        print('No spectrum with name', spectrum_name)

    def volcano_plot(self, sig_col, shift_col, qcutoff=0.05, shift_cutoff_low=None, shift_cutoff_high=None,
                     show_labels=True, colours=('C0', 'C3'), zero_p_offset=None):
        plt.scatter(self.results[shift_col], -np.log10(self.results[sig_col]), c=colours[0], linewidths=0)
        zero_q = self.results[self.results[sig_col] == 0]
        if len(zero_q) > 0 and zero_p_offset is None:
            zero_p_offset = self.results[self.results[sig_col] >= 0].min()/2
        plt.scatter(zero_q[shift_col], -np.log10(zero_q[sig_col] + zero_p_offset), marker='^', c=colours[0],
                    linewidths=0)

        plt.ylabel('Significance (-log10 {})'.format(sig_col))
        plt.xlabel(shift_col)
        if qcutoff is not None or shift_cutoff_low is not None or shift_cutoff_high is not None:
            sig_res = self.results
            if qcutoff is not None:
                sig_res = sig_res[sig_res[sig_col] < qcutoff]
            if shift_cutoff_low is not None:
                if shift_cutoff_high is not None:
                    sig_res = sig_res[(sig_res[shift_col] >= shift_cutoff_high) |
                                      (sig_res[shift_col] <= shift_cutoff_low)]
                else:
                    sig_res = sig_res[(sig_res[shift_col] <= shift_cutoff_low)]
            elif shift_cutoff_high is not None:
                sig_res = sig_res[(sig_res[shift_col] >= shift_cutoff_high)]

            plt.scatter(sig_res[shift_col], -np.log10(sig_res[sig_col]) , c=colours[1], linewidths=0)
            zero_q = sig_res[sig_res[sig_col] == 0]
            plt.scatter(zero_q[shift_col], -np.log10(zero_q[sig_col] + zero_p_offset), marker='^', c=colours[1],
                        linewidths=0)

            if show_labels:
                texts = []
                non_zero_q = sig_res[sig_res[sig_col] > 0]
                for i, row in non_zero_q.iterrows():
                    texts.append(plt.text(row[shift_col], -np.log10(row[sig_col]), row['gene']))
                for i, row in zero_q.iterrows():
                    texts.append(plt.text(row[shift_col], -np.log10(row[sig_col] + zero_p_offset), row['gene']))
                adjust_text(texts, arrowprops=dict(arrowstyle='->', color='red', alpha=0.4))

    def volcano_plot_colour_by_gene(self, genes, sig_col, shift_col,
            colours=None,   # List of colours if not wanting matplotlib default. First colour is for the other genes if shown.
            show_other_genes=True,  colour_sig_only=False,
            qcutoff=0.05, shift_cutoff_low=None, shift_cutoff_high=None, show_labels=True, zero_p_offset=None):
        colour_idx = 0
        if colours is None:
            colours = ["C{}".format(i%10) for i in range(len(genes)+1)]
            if len(genes) > 10 or (len(genes) > 9 and show_other_genes):
                print('Not enough colours to colour all genes uniquely. Reduce number of genes or provide longer list of colours.')
        if self.results[sig_col].min() == 0 and zero_p_offset is None:
            zero_p_offset = self.results[self.results[sig_col] >= 0].min() / 2
        if show_other_genes:
            plt.scatter(self.results[shift_col], -np.log10(self.results[sig_col]), c=colours[colour_idx], s=5,
                        alpha=0.5, label=None)
            zero_q = self.results[self.results[sig_col] == 0]
            plt.scatter(zero_q[shift_col], -np.log10(zero_q[sig_col] + zero_p_offset), marker='^', s=5,
                        alpha=0.5, c=colours[colour_idx], label=None)
            colour_idx += 1

        plt.ylabel('Significance (-log10 {})'.format(sig_col))
        plt.xlabel('Shift - {}'.format(shift_col))
        if qcutoff is not None or shift_cutoff_low is not None or shift_cutoff_high is not None:
            sig_res = self.results
            if qcutoff is not None:
                sig_res = sig_res[sig_res[sig_col] < qcutoff]
            if shift_cutoff_low is not None:
                if shift_cutoff_high is not None:
                    sig_res = sig_res[(sig_res[shift_col] >= shift_cutoff_high) |
                                      (sig_res[shift_col] <= shift_cutoff_low)]
                else:
                    sig_res = sig_res[(sig_res[shift_col] <= shift_cutoff_low)]
            elif shift_cutoff_high is not None:
                sig_res = sig_res[(sig_res[shift_col] >= shift_cutoff_high)]

            texts = []
            for gene in genes:
                if colour_sig_only:
                    gene_res = sig_res[sig_res['gene'] == gene]
                else:
                    gene_res = self.results[self.results['gene'] == gene]
                plt.scatter(gene_res[shift_col], -np.log10(gene_res[sig_col]), c=colours[colour_idx], label=gene)
                zero_q = gene_res[gene_res[sig_col] == 0]
                plt.scatter(zero_q[shift_col], -np.log10(zero_q[sig_col] + zero_p_offset), marker='^',
                            c=colours[colour_idx], label=None)
                colour_idx += 1

                if show_labels:
                    non_zero_q = gene_res[gene_res[sig_col] > 0]
                    for i, row in non_zero_q.iterrows():
                        texts.append(plt.text(row[shift_col], -np.log10(row[sig_col]), row['gene']))
                    for i, row in zero_q.iterrows():
                        texts.append(plt.text(row[shift_col], -np.log10(row[sig_col] + zero_p_offset), row['gene']))

            if show_labels:
                adjust_text(texts, arrowprops=dict(arrowstyle='->', color='red', alpha=0.4))
        plt.legend(bbox_to_anchor=(1.5, 1))
        plt.tight_layout()

    def plot_result_distribution(self, column):
        ordered_values = sorted(self.results[column])
        plt.scatter(range(len(ordered_values)), ordered_values)
        plt.ylabel(column)

    def change_lookup(self, new_lookup, inplace=False):
        if not inplace:
            # Temporarily set the old lookup to None, as some lookups cannot be copied in this manner
            old_lookup = self.lookup
            self.lookup = None
            d2 = deepcopy(self)
            d2.lookup = new_lookup
            self.lookup = old_lookup
            if hasattr(new_lookup, "setup_project"):
                new_lookup.setup_project(d2)
            return d2
        else:
            self.lookup = new_lookup
            if hasattr(self.lookup, "setup_project"):
                self.lookup.setup_project(self)

