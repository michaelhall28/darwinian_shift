import numpy as np
from Bio.Seq import Seq
import wget
from urllib3.exceptions import HTTPError
import urllib.parse
import urllib.request
import gzip
import pandas as pd

COMPLEMENTS = {'A': 'T', 'C': 'G', 'G': 'C', 'T': 'A', 'N':'N'}

def reverse_complement(seq, keep_order=False):
    s = ''.join([COMPLEMENTS[b] for b in seq])
    if keep_order:
        return s
    else:
        return s[::-1]


# def get_mut_trinuc(row):
#     if not pd.isnull(row['ref_trinuc']):
#         try:
#             ref_trinuc = row['ref_trinuc']
#             mut = row['mut']
#             if row['strand'] == -1:
#                 mut = reverse_complement(mut)
#             mut_trinuc = ref_trinuc[0] + mut + ref_trinuc[2]
#         except Exception as e:
#             print(row['ref_trinuc'])
#             print(row['mut'])
#             raise
#         return mut_trinuc
#     else:
#         return np.nan


def output_transcript_sequences_to_fasta(ds_object, file_name, aa_sequence=True):
    """
    Output all the transcript sequences from a project to file.
    Can either output the nucleotides or the amino acid sequence.
    The name of each sequence will be the ensembl transcript id.
    Useful for running external tools.
    :param ds_object: The DarwinianShift object that has been set up with the project data.
    Will have found all transcripts during the init.
    :param file_name: Output file name
    :param aa_sequence: True to output the amino acid sequence, False to output the nucleotide sequence.
    :return:
    """
    with open(file_name, 'w') as fh:
        for t in ds_object.transcript_objs.values():
            seq = t.nuc_sequence
            if aa_sequence:
                # Convert to amino acid sequence
                seq = Seq(seq).translate()
            fh.write(">{}\n".format(t.transcript_id))
            fh.write("{}\n".format(seq[:-1]))  # Remove the */stop from the end


def sort_multiple_arrays_using_one(*arrays):
    """
    Sorts by the first array/list
    Returns as an array
    """
    assert all([len(arrays[i]) == len(arrays[0]) for i in range(1, len(arrays))]), 'Arrays must all be same length'
    a = np.array(arrays)
    return a[:, a[0].argsort()]


PDB_DOWNLOAD_BASE_URL="https://files.rcsb.org/download"

def download_pdb_file(pdb_id, output_dir='.', file_type='pdb'):
    url = PDB_DOWNLOAD_BASE_URL + "/{}.{}".format(pdb_id, file_type)
    try:
        wget.download(url, output_dir)
    except HTTPError as e:
        print(type(e).__name__, e, 'Failed to download file from', url)
        raise e


UNIPROT_UPLOADLISTS_URL = 'https://www.uniprot.org/uploadlists/'
def get_uniprot_acc_from_transcript_id(transcript_id):
    params = {
    'from': 'ENSEMBL_TRS_ID',
    'to': 'ACC',
    'format': 'list',
    'query': transcript_id
    }

    data = urllib.parse.urlencode(params)
    data = data.encode('utf-8')
    req = urllib.request.Request(UNIPROT_UPLOADLISTS_URL, data)
    with urllib.request.urlopen(req) as f:
        response = f.read()
    return response.decode('utf-8').strip()


def read_sbs_from_vcf(vcf_file):
    """
    Only reads the chrom, pos, ref and alts from the VCF file.
    Not strict about checking the headers, and will not filter any mutations.
    Lines with multiple alts will be split into one mutation for each alt.
    Non-single-base-substitutions (e.g. deletions, indels, MNPs) will be skipped.
    :param vcf_file:
    :return: pandas dataframe
    """
    if vcf_file.endswith('.gz'):
        vcf_opener = gzip.open
    else:
        vcf_opener = open

    variants = []
    with vcf_opener(vcf_file) as vf:
        bases = {'A', 'C', 'G', 'T'}

        for line in vf:
            if line.startswith("#"):
                continue
            variant = line.split("\t")
            ref = variant[3]
            if ref not in bases:
                continue  # Only interested in single nucleotide substitutions
            alts = variant[4].split(',')
            if alts is None:
                continue  # Only interested in single nucleotide substitutions
            for alt in alts:
                if alt not in bases:
                    continue  # Only interested in single nucleotide substitutions
                chrom, pos = variant[0:2]
                variants.append((chrom, pos, ref, alt))

    df = pd.DataFrame(variants, columns=['chr', 'pos', 'ref', 'mut'])
    df['pos'] = df['pos'].astype(int)
    return df


# Functions to add the required columns to COSMIC TSV data.
def _get_cosmic_sbs_muts(row):
    """
    We want the chromosome, position, reference, alternate
    """
    chrom, pos = row['Mutation genome position'].split(':')

    if chrom == '23':
        chrom = 'X'
    elif chrom == '24':
        chrom = 'Y'
    elif chrom == '25':
        chrom = 'MT'
    pos = pos.split('-')[0]
    ref_mut = row['Mutation CDS']  # Expect this to be like c.123T>A for SNVs.
    strand = row['Mutation strand']
    ref, mut = ref_mut[-3], ref_mut[-1]
    try:
        if ref not in ['A', 'C', 'G', 'T'] or mut not in ['A', 'C', 'G', 'T'] or ref_mut[-2] != '>' or not ref_mut[
            -4].isdigit():
            # Not a single base substitution.
            mut = np.nan
            ref = row['Mutation CDS']
        elif strand == '-':
            mut = COMPLEMENTS[mut]
            ref = COMPLEMENTS[ref]
    except Exception as e:
        print(row)
        raise e

    return {'chr': chrom, 'pos': int(pos), 'ref': ref, 'mut': mut}


def process_cosmic_mutations(cosmic_mutations, verbose=False):
    """
    Convert cosmic mutations to a dataframe that can be used as a DarwinianShift input.
    Can provide a file path to the cosmic tsv or a dataframe.

    Removes duplicate mutations.
    """

    if isinstance(cosmic_mutations, str):
        cosmic_mutations = pd.read_csv(cosmic_mutations, sep="\t")

    if verbose:
        print('Starting with {} mutations'.format(len(cosmic_mutations)))
    # Remove mutations that don't have a Mutation genome position
    cosmic_mutations = cosmic_mutations[~pd.isnull(cosmic_mutations['Mutation genome position'])]
    if verbose:
        print('After removing mutations without a position: {} mutations'.format(len(cosmic_mutations)))
    # Remove duplicate mutations (same sample or same tumour)
    cosmic_mutations = cosmic_mutations.drop_duplicates(subset=['Mutation genome position', 'Sample name'])
    cosmic_mutations = cosmic_mutations.drop_duplicates(subset=['Mutation genome position', 'ID_tumour'])
    if verbose:
        print('After removing duplicates: {} mutations'.format(len(cosmic_mutations)))

    # Get the chromosomal coordinates and base changes for the single base substitutions
    ds_input_columns = pd.DataFrame(list(cosmic_mutations.apply(_get_cosmic_sbs_muts, axis=1)))
    cosmic_mutations = cosmic_mutations.reset_index(drop=True)
    cosmic_mutations = pd.concat([cosmic_mutations, ds_input_columns], axis=1)

    return cosmic_mutations