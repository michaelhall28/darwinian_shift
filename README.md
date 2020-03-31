# darwinian_shift
Statistical testing of selection of quantifiable features of somatic mutations.  

Some basic instructions are provided in this readme.  

## Installation
To install, clone the git repository  
`git clone https://github.com/michaelhall28/darwinian_shift`  

Then pip install the dependencies  
`cd darwinian_shift`  
`pip install -r requirements.txt`  

and install the code  
`pip install .`  
Or add the repository to your PYTHONPATH.  


### Reference data
Need a bgzip compressed genome fasta file (.fa.gz) with a faidx index and a table of exon locations.  
For human GRCh38 or GRCh37 or mouse GRCm38, the required reference files can be downloaded from release 0.0.1 of this repository.  
These need to be unzipped in the darwinian_shift/reference_data directory.  

You can download the reference files for a particular species from ensembl, e.g.
```
from darwinian_shift import download_reference_data_from_latest_ensembl
download_reference_data_from_latest_ensembl('homo_sapiens')
```

Individual reference fasta files and exon files can also be specified.  


## Running

#### Input mutation data
Data needs to be in a tab separated file with columns chr, pos, ref and mut (order does not matter, any additional columns will be ignored).  
Alternatively, a pandas dataframe of mutations with those column headings can be used.   
The chromosome names should not have the "chr" prefix, i.e. should be named like "1" not "chr1".  

#### Feature scores
The scores are added to mutations using a lookup function/class.   
There are several examples ready to use in the lookup_classes directory.   

A lookup class is initialised and then passed to the DarwinianShift class.  


#### Basic use
Here we use a BigWig file to score each mutation based on a conservation score.  
The reference genome and exons for GRCh37 must be downloaded first for this to run (see **Reference data** above).

```
from darwinian_shift import DarwinianShift, BigWigLookup

data_file = "tests/test_data/test_mutation_data.tsv"
bigwig_data_file = "tests/lookup_tests/phylop_bw_section.bw"


d = DarwinianShift(data=data_file,
  source_genome='grch37',
  lookup=BigwigLookup(bigwig_data_file)
  )

notch3 = d.run_gene('NOTCH3')
notch3_results = notch3.get_results_dictionary()
print(notch3_results['CDF_perm_glob_k3_pvalue'])
notch3.plot_boxplot()
```
