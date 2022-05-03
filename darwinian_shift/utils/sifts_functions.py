"""
Functions for downloading and reading SIFTS files, which contain the alignment of pdb residues with the protein sequence
"""
import os
from Bio.SeqUtils import seq1
import pandas as pd
import urllib.request
import xmlschema
from xmlschema import XMLSchemaException
import gzip


def download_sifts_xml(pdbid, sifts_dir):
    """
    Download the SIFTS xml file from the EBI FTP site.
    :param pdbid: String. ID of the PDB file. Case insensitive.
    :param sifts_dir: Directory in which to save the SIFTS xml file.
    :return:
    """
    address = 'ftp://ftp.ebi.ac.uk/pub/databases/msd/sifts/xml/{}.xml.gz'.format(pdbid.lower())
    with urllib.request.urlopen(address) as r:
        data = r.read()

    with open(os.path.join(sifts_dir, "{}.xml.gz".format(pdbid.lower())), 'wb') as fh:
        fh.write(data)


def read_xml(pdbid, sifts_dir, schema_location="http://www.ebi.ac.uk/pdbe/docs/sifts/eFamily.xsd"):
    """
    Read the SIFTS xml file and convert to a more human-readable tab-separated file containing only the PDB residue
    numbers and the UniProt residue numbers.
    :param pdbid: The PDB ID of the structure.
    :param sifts_dir: The directory in which the SIFTS xml and TSV files are saved
    :param schema_location: The address of the schema for the SIFTS xml file.
    :return:
    """
    schema = xmlschema.XMLSchema(schema_location)

    # Download the xml file if it doesn't already exist in the directory.
    f = os.path.join(sifts_dir, "{}.xml.gz".format(pdbid.lower()))
    if not os.path.isfile(f):
        print('Downloading SIFTS xml for', pdbid)
        download_sifts_xml(pdbid, sifts_dir)

    # Name of the output file.
    output = os.path.join(sifts_dir, "{}.txt".format(pdbid.lower()))

    results = []

    # Read the xml file into a dictionary
    try:
        sifts_result = schema.to_dict(gzip.open(f))
    except XMLSchemaException as e:
        # lax validation allows reading of xml if schema not updated with latest changes (e.g. SCOP2).
        sifts_result, errors = schema.to_dict(gzip.open(f), validation='lax')

    for entity in sifts_result['entity']:
        for segment in entity['segment']:
            for residue in segment['listResidue']['residue']:
                # Require a entry to have both a PDB residue and a UniProt residu defined.
                pdb = None
                uni = None
                for crossref in residue['crossRefDb']:
                    if crossref['@dbSource'] == 'PDB':
                        pdb = crossref
                    elif crossref['@dbSource'] == 'UniProt':
                        uni = crossref

                if pdb is not None and uni is not None:
                    results.append([sifts_result['@dbAccessionId'],
                                    pdb['@dbChainId'],
                                    seq1(pdb['@dbResName']),
                                    pdb['@dbResNum'],
                                    uni['@dbAccessionId'],
                                    uni['@dbResName'],
                                    uni['@dbResNum']])

    # Write the list of results as a TSV file.
    write_sifts_output(output, results)


def write_sifts_output(output_file, results):
    """
    Outputs the information from the SIFTS xml file as a TSV file.
    :param output_file: File path for the output.
    :param results: List of results. Each entry is a list of PDB ID, PDB chain, PDB residue, PDB position,
    UniProt ID, UniProt residue, and Uniprot position.
    :return: None.
    """
    with open(output_file, 'w') as fh:
        fh.write("pdb id\tpdb chain\tpdb residue\tpdb position\tuniprot id\tuniprot residue\tuniprot position\n")
        for r in results:
            fh.write("\t".join(r) + "\n")


def get_sifts_alignment(pdbid, sifts_dir, download=True, convert_numeric=True):
    """
    Returns a pandas DataFrame with columns for the position of residues in the PDB file and positions in the
    UniProt protein sequence.
    :param pdbid: PDB ID
    :param sifts_dir: The directory in which the SIFTS xml and TSV files are saved
    :param download: If True and the SIFTS files do not already exist, will download the SIFTS information. If False,
    will only use files that have already been downloaded.
    :param convert_numeric: If True, will force the "pdb position" column of the output dataframe to be numeric.
    :return: pandas DataFrame, or None if no SIFTS files exist and download=False.
    """
    aln_file = os.path.join(sifts_dir, "{}.txt".format(pdbid.lower()))
    if not os.path.isfile(aln_file):
        if download:
            read_xml(pdbid, sifts_dir)
        else:
            return None
    df = pd.read_csv(aln_file, sep="\t")
    if convert_numeric:
        df['pdb position'] = pd.to_numeric(df['pdb position'], errors='coerce')
    return df


def get_sifts_alignment_for_chain(pdb_id, pdb_chain, sifts_directory, download=True):
    """
    Returns a dataframe of the positions of residues in the PDB file and in the UniProt protein.
    Filters the SIFTS results to just return the information for the given chain.
    :param pdb_id: String. PDB ID
    :param pdb_chain: String. The chain in the PDB structure.
    :param sifts_directory: The directory in which the SIFTS xml and TSV files are saved
    :param download: If True and the SIFTS files do not already exist, will download the SIFTS information. If False,
    will only use files that have already been downloaded.
    :return: pandas dataframe, or None if the information was not found.
    """
    sifts = get_sifts_alignment(pdb_id, sifts_directory, download=download)
    if sifts is None and not download:
        print('SIFTS alignment for PDB structure {} not found'.format(pdb_id))
    elif sifts is None:
        print('PDB structure {} not found in SIFTS.'.format(pdb_id))
    elif len(sifts) == 0:
        print('No alignment information in the SIFTS file.')
        sifts = None
    else:
        sifts = sifts[sifts['pdb chain'] == pdb_chain]
        if len(sifts) == 0:
            print('No SIFTS alignment information for the chain.')
            sifts = None
    return sifts


def get_pdb_positions(residues, pdb_id, pdb_chain, sifts_directory, download=True):
    """
    Given a list of residue numbers in a protein sequence, a PDB ID and chain, will return a dataframe including the
    residue number in the PDB file.
    :param residues: List/array of int. The residue numbers in the protein sequence.
    :param pdb_id: String. PDB ID
    :param pdb_chain: String. The chain in the PDB structure.
    :param sifts_directory: The directory in which the SIFTS xml and TSV files are saved
    :param download:  If True and the SIFTS files do not already exist, will download the SIFTS information. If False,
    will only use files that have already been downloaded.
    :return: pandas dataframe, or None if the SIFTS alignment is not available.
    """
    sifts_alignment = get_sifts_alignment_for_chain(pdb_id, pdb_chain, sifts_directory, download=download)
    if sifts_alignment is None:
        return None
    r = pd.DataFrame({'residue': residues})
    sifts = pd.merge(r, sifts_alignment, left_on='residue', right_on='uniprot position', how='inner')
    sifts = sifts[~pd.isnull(sifts['pdb position'])]  #  Remove residues that are not in the pdb file
    return sifts