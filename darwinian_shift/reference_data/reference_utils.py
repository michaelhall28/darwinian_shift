import requests, sys
import os
import wget
import glob
import subprocess

REF_DIR = os.path.dirname(os.path.realpath(__file__))


## Functions to download genomes from the ensembl ftp
ENSEMBL_REST_SERVER = "https://rest.ensembl.org"
# GRCH37_REST_SERVER = "http://grch37.rest.ensembl.org"

ENSEMBL_FTP ="ftp://ftp.ensembl.org/pub/"
GRCH37_FTP = "ftp://ftp.ensembl.org/pub/grch37/"


def _ensembl_get(ext):
    r = requests.get(ENSEMBL_REST_SERVER + ext, headers={"Content-Type": "application/json"})

    if not r.ok:
        r.raise_for_status()
        sys.exit()

    decoded = r.json()
    return decoded


def get_latest_ensembl_release_and_assembly(source_genome):
    release = _ensembl_get("/info/data/?")['releases'][0]
    genome_build = _ensembl_get("/info/assembly/{}?".format(source_genome.lower()))['assembly_name'].split('.')[0]
    return release, genome_build


def download_genome_fasta(source_genome, output_dir=None, grch37=False):
    ensembl_release, genome_build1 = get_latest_ensembl_release_and_assembly(source_genome)
    if grch37:
        genome_build = 'GRCh37'
        ftp = GRCH37_FTP
    else:
        genome_build = genome_build1
        ftp = ENSEMBL_FTP

    url_directory = os.path.join(ftp, "release-{}/fasta/{}/dna/".format(ensembl_release, source_genome.lower()))
    file_name = "{}.{}.dna.primary_assembly.fa.gz".format(source_genome.capitalize(), genome_build)
    url = os.path.join(url_directory, file_name)

    if output_dir is None:
        output_dir = os.path.join(REF_DIR, source_genome,
                                  "ensembl-{}".format(ensembl_release))
    if grch37:
        output_dir = os.path.join(output_dir, "GRCh37")

    os.makedirs(output_dir, exist_ok=True)

    print('Downloading {} to {}'.format(url, output_dir))
    wget.download(url, output_dir)

    print('bgzipping and indexing the fasta file')
    convert_to_bgzip_and_index(os.path.join(output_dir, file_name))

    return output_dir

def convert_to_bgzip_and_index(file_name):
    if file_name.endswith('.gz'):
        subprocess.run(["gunzip", "{}".format(file_name)])
        file_name = file_name[:-3]
    subprocess.run(["bgzip", "{}".format(file_name)])
    subprocess.run(["samtools", "faidx", "{}".format(file_name + '.gz')])


## Functions to download a table of exons from ensembl biomart
BIOMART_URL = 'http://www.ensembl.org/biomart/martservice?query='
GRCH37_BIOMART_URL = 'http://grch37.ensembl.org/biomart/martservice?query='

def get_biomart_xml(source_genome):
    source_genome = source_genome.split('_')
    sp = source_genome[0][0] + source_genome[1]

    x = '<?xml version="1.0" encoding="UTF-8"?>'
    x += '<!DOCTYPE Query> <Query  virtualSchemaName = "default" formatter = "TSV" header = "0" uniqueRows = "0" count = "" datasetConfigVersion = "0.6" completionStamp = "1">'
    x += '<Dataset name = "{}_gene_ensembl" interface = "default" >'.format(sp)
    x += '<Attribute name = "ensembl_gene_id" />'
    x += '<Attribute name = "ensembl_transcript_id" />'
    x += '<Attribute name = "chromosome_name" />'
    x += '<Attribute name = "strand" />'
    x += '<Attribute name = "cds_length" />'
    x += '<Attribute name = "rank" />'
    x += '<Attribute name = "genomic_coding_start" />'
    x += '<Attribute name = "genomic_coding_end" />'
    x += '<Attribute name = "external_gene_name" />'
    x += '</Dataset>'
    x += '</Query>'

    return x


def download_biomart_data(source_genome, output_dir, grch37=False):
    x = get_biomart_xml(source_genome)
    if grch37:
        url = GRCH37_BIOMART_URL
    else:
        url = BIOMART_URL
    url += x

    os.makedirs(output_dir, exist_ok=True)

    output_file = os.path.join(output_dir, 'biomart_exons_{}.txt'.format(source_genome))

    print('Downloading biomart data to {}'.format(output_file))
    success = False
    with open(output_file, 'w') as fh:
        r = requests.get(url, stream=True)
        header = "Gene stable ID\tTranscript stable ID\tChromosome/scaffold name\tStrand\tCDS Length\tExon rank in transcript\tGenomic coding start\tGenomic coding end\tGene name\n"
        fh.write(header)
        for line in r.iter_lines():
            line = line.decode()
            if line.strip() == "[success]":
                success = True
            else:
                fh.write(line + '\n')

    if not success:
        raise ValueError("Failed to successfully download biomart data")


## Functions to download the both the references files.
def download_reference_data_from_latest_ensembl(source_genome):
    print('Downloading and processing reference data. This will take a while.')
    output_dir = download_genome_fasta(source_genome)
    download_biomart_data(source_genome, output_dir)


def download_grch37_reference_data():
    # A special, common case to use the older human reference genome.
    print('Downloading and processing reference data. This will take a while.')
    output_dir = download_genome_fasta('homo_sapiens', grch37=True)
    download_biomart_data('homo_sapiens', output_dir, grch37=True)



## Function to get the reference data for a requested source_genome
def get_source_genome_reference_file_paths(source_genome, ensembl_release=None):
    if source_genome.lower() == 'grch37':
        source_genome_dir = os.path.join(REF_DIR, 'homo_sapiens')
    else:
        source_genome_dir = os.path.join(REF_DIR, source_genome)

    if ensembl_release is None:
        releases = [int(d.split('-')[-1]) for d in glob.glob(os.path.join(source_genome_dir, "ensembl-*"))]
        ensembl_release = max(releases)
    d = os.path.join(source_genome_dir, 'ensembl-{}'.format(ensembl_release))

    if source_genome.lower() == 'grch37':
        d = os.path.join(d, "GRCh37")
        source_genome = 'homo_sapiens'

    exon_file = os.path.join(d, 'biomart_exons_{}.txt'.format(source_genome))
    reference_file = glob.glob(os.path.join(d, "*.dna.primary_assembly.fa.gz"))[0]
    return exon_file, reference_file
