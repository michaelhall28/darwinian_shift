"""
Classes for storing the mutational spectra of datasets.
"""

import pandas as pd
from collections import Counter
import json
from .utils.util_functions import reverse_complement


class MutationalSpectrum:
    """
    Base class. Should not be used directly.
    Just a few functions that are used in all spectra
    And using a base class means we can check if an object is a spectrum of any type
    """
    precalculated = None

    def set_project(self, project):
        self.project = project

    def get_spectrum(self):
        pass

    def get_mut_rate(self, key):
        return self.spectrum.loc[key, self.rate_column]

    def set_name(self):
        if self.name is None:
            self.name = self.default_name.format(self.k)
        else:
            self.name = str(self.name)

    def fix_spectrum(self):
        """
        Prevent more mutations being added to the spectrum.
        Useful for calculating the spectrum from one set of data, then testing on another.
        :return:
        """
        self.precalculated = True

    def reset(self):
        pass

class EvenMutationalSpectrum(MutationalSpectrum):
    """
    The simplest mutational spectrum.
    All nucleotide substitutions have the same probability of occurring
    """
    default_name = 'EvenMutationalSpectrum'
    precalculated=True
    def __init__(self, name=None):
        self.name = name
        self.set_name_and_columns()
        
    def set_name_and_columns(self):
        self.set_name()
        self.set_columns()

    def set_name(self):
        if self.name is None:
            self.name = self.default_name
        else:
            self.name = str(self.name)

    def set_columns(self):
        self.rate_column = 'mutation_rate_' + self.name

    def add_transcript_muts(self, t):
        pass

    def apply_spectrum(self, seq_obj, df):
        """
        Apply to a dataframe of mutations.
        :param seq_obj: Section object. Not used here. Argument kept to be consistent with the other spectrum classes.
        :param df: Dataframe of mutations.
        :return:
        """
        # Set the mutation rate to be equal for all mutations
        # The relative mutation rate is used, so the absolute value doesn't matter. Use 1.0.
        df.loc[:, self.rate_column] = 1.0
        return df

###### Spectra calculated from the given data set
class KmerMutationalSpectrum(MutationalSpectrum):
    """
    Class for spectra based on the nucleotide change and the surrounding nucleotide context of the reference base.
    This can be used for trinucleotide spectra for example.

    Inherited by the classes GlobalKmerSpectrum (all genes share the same mutational spectrum) and
    TranscriptKmerSpectrum (each transcript has a mutational spectrum calculated from just the transcript mutations).
    """
    write_attrs = ('name', 'ignore_strand', 'centre_index', 'k', 'missing_value', 'kmer_column_raw')
    output_columns = ()
    collect_counts = True

    def __init__(self, deduplicate_spectrum=False, k=3, ignore_strand=False, missing_value=0, name=None, run_init=True):
        """

        :param deduplicate_spectrum: Remove duplicate mutations (same position, reference and alternate base)
        before calculating the mutation rate.
        :param k: Size of kmer nucleotide context. Use 3 for trinucleotides.
        :param ignore_strand: If True, the equivalent mutations in the opposite strand will be combined into a single
        category. E.g. A>C will be the same as T>G.
        :param missing_value: Used to replace missing values (where no mutations of that type are observed).
        Useful to make non-zero in some cases.
        :param name: Name to give the spectrum. Will be added to some columns in the statistical results to
        differentiate the output from the different spectra used.
        :param run_init: This is set to false when using a file to input spectrum
        """
        self.k = None
        self.centre_index = None
        self.deduplicate_spectrum = None
        self.ignore_strand = None
        self.missing_value = None
        self.name = None
        self.rate_column = None
        self.kmer_column_raw = None
        self.kmer_column = None
        self.strand_mut_column = None
        self.mut_count_column = None
        self.seq_count_column = None
        self.project = None
        self.spectrum = None
        self.precalculated=None

        self.kmer_counts = Counter()
        self.observed_mut_counts = Counter()

        if run_init:
            if k % 2 == 0 or int(k) != k or k < 0:
                raise  ValueError('k must be zero or a positive odd integer.')
            self.precalculated = False
            self.k = k
            self.kmer_column_raw = '{}mer_ref'.format(k)

            self.centre_index = int((k-1)/2)
            self.deduplicate_spectrum = deduplicate_spectrum
            self.ignore_strand = ignore_strand
            self.missing_value = missing_value
            self.name = name
            self.rate_column = None
            self.set_name_and_columns()

            self.project = None

            self.spectrum = None

    def reset(self):
        if not self.precalculated:
            self.spectrum = None
            self.kmer_counts = Counter()
            self.observed_mut_counts = Counter()

    def set_name_and_columns(self):
        self.set_name()
        self.set_columns()

    def set_columns(self):
        self.rate_column = 'mutation_rate_' + self.name
        self.kmer_column = self.kmer_column_raw + '_' + self.name
        self.strand_mut_column = 'strand_mut_' + self.name
        self.mut_count_column = 'mut_count_' + self.name
        self.seq_count_column = 'seq_count_' + self.name

    def combine_strand_kmer_counts(self, kmer_counts):
        # Combine the reverse complements
        # Keep kmers that mutated A or C, reverse complement kmers that mutate T or G
        kmer_count2 = Counter()
        for k, c in kmer_counts.items():
            if k[self.centre_index] in 'AC':
                kmer_count2[k] += c
            else:
                rev = reverse_complement(k)
                kmer_count2[rev] += c

        return kmer_count2

    def get_spectrum(self):
        if not self.precalculated:
            self.calculate_spectrum()

    def calculate_spectrum(self):
        pass

    def _strand_swap_kmer(self, row):
        kmer = row[self.kmer_column]
        if kmer[self.centre_index] in 'AC':
            return kmer
        else:
            return reverse_complement(kmer)

    def _strand_swap_mut(self, row):
        kmer = row[self.kmer_column]
        mut = row[self.strand_mut_column]
        if kmer[self.centre_index] in 'AC':
            return mut
        else:
            return reverse_complement(mut)

    def combine_strands(self, df):
        df=df.copy()  # Don't overwrite in case used by other spectra.
        stranded_kmers = df.apply(self._strand_swap_kmer, axis=1)
        stranded_muts = df.apply(self._strand_swap_mut, axis=1)
        df.loc[:, self.kmer_column] = stranded_kmers
        df.loc[:, self.strand_mut_column] = stranded_muts
        return df

    def _strand_swap_dict_key(self, k):
        ref_kmer, mut = k.split('>')
        if ref_kmer[self.centre_index] in 'AC':
            return ref_kmer + '>' +  mut
        else:
            return reverse_complement(ref_kmer) + '>' + reverse_complement(mut)

    def combine_strands_for_dict(self, observed_mut_counts):
        combined_counts = Counter()
        for k, v in observed_mut_counts.items():
            combined_counts[self._strand_swap_dict_key(k)] += v

        return combined_counts

    def get_spectrum_dataframe(self, kmer_counts, observed_mut_counts):
        """
        Get the spectrum dataframe from a dictionary of kmer counts in the sequence
        and a dictionary of observed mutations.
        Any kmer changes that are not observed will not be included.
        :param trinuc_counts:
        :param mutations:
        :return:
        """
        spectrum = []
        for k, mut_count in observed_mut_counts.items():
            kmer, mut = k.split('>')
            seq_count = kmer_counts[kmer]  # Count of the kmer in the genome/transcript sequence
            if seq_count > 0:
                relative_rate = mut_count / seq_count
            else:
                relative_rate = 0

            spectrum.append({self.kmer_column: kmer, self.strand_mut_column: mut, self.rate_column: relative_rate,
                             self.mut_count_column: mut_count, self.seq_count_column: seq_count})

        spectrum = pd.DataFrame(spectrum)

        return spectrum

    def write_to_file(self, output_file):
        """
        Output the spectrum to a text file.
        Includes a header with the spectrum information.
        :param output_file:
        :return:
        """
        output_dict = {'spectrum_type': self.__class__.__name__}
        for a in self.write_attrs:
            output_dict[a] = getattr(self, a)

        with open(output_file, 'w') as fh:
            fh.write(json.dumps(output_dict) + '\n')

        cols = [getattr(self, c) for c in self.output_columns]
        self.spectrum[cols].sort_values(cols).to_csv(output_file, mode='a', index=False,
                                                                              header=False)


class GlobalKmerSpectrum(KmerMutationalSpectrum):
    """
    Kmer spectrum calculated from all exonic mutations in a dataset.
    For example, a trinucleotide spectrum.
    """

    default_name = 'glob_k{}'
    output_columns = ('kmer_column', 'strand_mut_column', 'rate_column', 'mut_count_column', 'seq_count_column')

    def add_transcript_muts(self, t):
        self.kmer_counts.update(t.transcript_ref_kmer_counts[self.k])
        self.observed_mut_counts.update(t._get_kmer_counts_for_observed_mutations()[self.deduplicate_spectrum][self.k])

    def calculate_spectrum(self):
        if self.ignore_strand:
            self.kmer_counts = self.combine_strand_kmer_counts(self.kmer_counts)
            self.observed_mut_counts = self.combine_strands_for_dict(self.observed_mut_counts)

        self.spectrum = self.get_spectrum_dataframe(self.kmer_counts, self.observed_mut_counts)

    def apply_spectrum(self, seq_obj, df):
        """
        Apply to a dataframe of mutations.
        The dataframe must already have the kmer and strand_mut column

        :param seq_obj: Section object. Not used here. Argument kept to be consistent with the other spectrum classes.
        :param df: Dataframe of mutations.
        :return:
        """
        df[self.kmer_column] = df[self.kmer_column_raw]
        df[self.strand_mut_column] = df['strand_mut']
        if self.ignore_strand:
            df = self.combine_strands(df)
        df = pd.merge(df, self.spectrum, left_on=[self.kmer_column, self.strand_mut_column],
                      right_on=[self.kmer_column, self.strand_mut_column], how='left', suffixes=["_x", ""])
        df.fillna(value={self.rate_column: self.missing_value}, inplace=True)
        return df


class TranscriptKmerSpectrum(KmerMutationalSpectrum):
    """
    Kmer spectrum where each transcript has a spectrum calculated from only the exonic mutations in that transcript.
    For example, a trinucleotide spectrum.
    """
    default_name = 'tran_k{}'
    output_columns = ('transcript_id_col', 'kmer_column', 'strand_mut_column', 'rate_column', 'mut_count_column', 'seq_count_column')
    transcript_id_col = 'transcript_id'  # so can use the same function to read spectrum from file as the global spectrum class

    def add_transcript_muts(self, t):
        pass

    def calculate_spectrum(self):
        pass

    def fix_spectrum(self):
        """
        Prevent more mutations being added to the spectrum.
        Useful for calculating the spectrum from one set of data, then testing on another.
        :return:
        """
        if self.project.low_mem:
            print('Warning: TranscriptKmerSpectrum is not stored as low_mem=True')
        else:
            self.get_complete_spectrum()
        self.precalculated = True

    def get_transcript_spectrum(self, t):
        if not self.precalculated:
            ref_kmer_counts = t.transcript_ref_kmer_counts[self.k]
            observed_mut_counts = t._get_kmer_counts_for_observed_mutations()[self.deduplicate_spectrum][self.k]

            if self.ignore_strand:
                ref_kmer_counts = self.combine_strand_kmer_counts(ref_kmer_counts)
                observed_mut_counts = self.combine_strands_for_dict(observed_mut_counts)

            spectrum = self.get_spectrum_dataframe(ref_kmer_counts, observed_mut_counts)
            spectrum['transcript_id'] = t.transcript_id
            return spectrum
        else:
            return self.spectrum

    def get_complete_spectrum(self):
        # Get the concatenated spectrum for all transcripts in the project
        # Will not work with low_mem=True as no transcripts will be stored.
        if not self.project.low_mem and not self.precalculated:
            res = []
            for transcript in self.project.transcript_objs.values():
                res.append(self.get_transcript_spectrum(transcript))
            self.spectrum = pd.concat(res)
        elif self.project.low_mem:
            print('Cannot collect per transcript if DarwinianShift.low_mem=True')

        return self.spectrum

    def apply_spectrum(self, seq_obj, df):
        """
        Apply to a dataframe of mutations.
        The dataframe must already have the kmer and strand_mut column

        :param seq_obj: Section object.
        :param df: Dataframe of mutations.
        :return:
        """
        spectrum = self.get_transcript_spectrum(seq_obj.transcript)
        df[self.kmer_column] = df[self.kmer_column_raw]
        df[self.strand_mut_column] = df['strand_mut']
        if self.ignore_strand:
            df = self.combine_strands(df)
        df = pd.merge(df, spectrum, left_on=['transcript_id', self.kmer_column, self.strand_mut_column],
                      how='left', right_on=['transcript_id', self.kmer_column, self.strand_mut_column],
                      suffixes=["_x", ""])
        df.fillna(value={self.rate_column: self.missing_value}, inplace=True)
        return df


###### Spectra read from a file
def read_spectrum(input_file, name=None):
    """
    Load a spectrum from an input file.
    :param input_file:
    :param name: Name to give the new spectrum. If not given, the name in the input file will be used.
    :return:
    """
    # This can be expanded in future with more input formats if needed
    # For now, only set up for the Kmer spectrum classes above.
    return read_kmer_spectrum(input_file, name)


def read_kmer_spectrum(input_file, name=None):
    """
    Load a kmer mutational spectrum from a file.
    :param input_file:
    :param name: If not None, will overwrite the name in the file.
    :return: A GlobalKmerSpectrum or TranscriptKmerSpectrum object.
    """
    # Read the first line of the file to get the header information.
    with open(input_file) as fh:
        line = fh.readline()
        spectrum_dict = json.loads(line.strip())

    # Set up the spectrum class and attributes based on the header information.
    spectrum_type = spectrum_dict.pop('spectrum_type')
    if spectrum_type == 'GlobalKmerSpectrum':
        spectrum = GlobalKmerSpectrum(run_init=False)
    elif spectrum_type == 'TranscriptKmerSpectrum':
        spectrum = TranscriptKmerSpectrum(run_init=False)
    else:
        raise ValueError('Do not recognise input file format')

    for a, v in spectrum_dict.items():
        if a != 'name':
            setattr(spectrum, a, v)
        elif name is None:
            setattr(spectrum, a, v)

    spectrum.set_name_and_columns()

    # Read the spectrum itself and assign to the class.
    spectrum.spectrum = pd.read_csv(input_file, skiprows=1, header=None,
                                names=[getattr(spectrum, c) for c in spectrum.output_columns])
    spectrum.precalculated = True

    if name is not None:  # Overwrite the name
        spectrum.name = name
    return spectrum


