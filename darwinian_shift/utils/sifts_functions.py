# For reading SIFTS files, which contain the alignment pdb residues with the protein sequence
import os
from Bio.SeqUtils import seq1
import pandas as pd
import urllib.request
import xmlschema
from xmlschema import XMLSchemaException
import gzip


def download_sifts_xml(pdbid, sifts_dir):
    address = 'ftp://ftp.ebi.ac.uk/pub/databases/msd/sifts/xml/{}.xml.gz'.format(pdbid.lower())
    with urllib.request.urlopen(address) as r:
        data = r.read()

    with open(os.path.join(sifts_dir, "{}.xml.gz".format(pdbid.lower())), 'wb') as fh:
        fh.write(data)


def read_xml(pdbid, sifts_dir, schema_location="http://www.ebi.ac.uk/pdbe/docs/sifts/eFamily.xsd"):
    schema = xmlschema.XMLSchema(schema_location)
    f = os.path.join(sifts_dir, "{}.xml.gz".format(pdbid.lower()))
    if not os.path.isfile(f):
        print('Downloading SIFTS xml for', pdbid)
        download_sifts_xml(pdbid, sifts_dir)
    output = os.path.join(sifts_dir, "{}.txt".format(pdbid.lower()))
    results = []
    try:
        sifts_result = schema.to_dict(gzip.open(f))
    except XMLSchemaException as e:
        # lax validation allows reading of xml if schema not updated with latest changes (e.g. SCOP2).
        sifts_result, errors = schema.to_dict(gzip.open(f), validation='lax')

    for entity in sifts_result['entity']:
        for segment in entity['segment']:
            for residue in segment['listResidue']['residue']:
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

    write_sifts_output(output, results)


def write_sifts_output(output_file, results):
    with open(output_file, 'w') as fh:
        fh.write("pdb id\tpdb chain\tpdb residue\tpdb position\tuniprot id\tuniprot residue\tuniprot position\n")
        for r in results:
            fh.write("\t".join(r) + "\n")


def get_sifts_alignment(pdbid, sifts_dir, download=True, convert_numeric=True):
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
    sifts = get_sifts_alignment(pdb_id, sifts_directory, download=download)
    if sifts is None and not download:
        print('PDB structure not in SIFTS directory.')
    elif sifts is None:
        print('PDB structure not found in SIFTS.')
    elif len(sifts) == 0:
        print('No sifts alignment information, but file exists.')
        sifts = None
    else:
        sifts = sifts[sifts['pdb chain'] == pdb_chain]
        if len(sifts) == 0:
            print('No sifts alignment information for the chain.')
            sifts = None
    return sifts


def get_pdb_positions(residues, pdb_id, pdb_chain, sifts_directory, download=True):
    sifts_alignment = get_sifts_alignment_for_chain(pdb_id, pdb_chain, sifts_directory, download=download)
    if sifts_alignment is None:
        return None
    r = pd.DataFrame({'residue': residues})
    sifts = pd.merge(r, sifts_alignment, left_on='residue', right_on='uniprot position', how='inner')
    sifts = sifts[~pd.isnull(sifts['pdb position'])]  #  Remove residues that are not in the pdb file
    return sifts