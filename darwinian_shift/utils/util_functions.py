import numpy as np
from Bio.Seq import Seq
import wget
from urllib3.exceptions import HTTPError

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
