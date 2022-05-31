# darwinian_shift
Statistical testing of selection of quantifiable features of somatic mutations.  

Some basic instructions are provided in this readme. More detailed examples are in the Tutorial.ipynb notebook.

The Root code for weighted Anderson-Darling tests is from Trusina et. al. https://doi.org/10.1088/1742-6596/1525/1/012109

## Installation
To install, clone the git repository  
`git clone https://github.com/michaelhall28/darwinian_shift`  

Use conda to install the dependencies, activate the environment and run Root to compile the code for comparing datasets  
`cd darwinian_shift`  
`conda env create -f environment.yml`  
`conda activate dsenv`  
`root -b -l -q darwinian_shift/dataset_comparison/root/homtests.C+`

Then either install the code:  
`pip install .`  
Or add the repository to your PYTHONPATH.  


### Installing with fewer dependencies

To install only a minimal set of dependencies, run:  
`git clone https://github.com/michaelhall28/darwinian_shift`  
`cd darwinian_shift`  
`conda env create -f environment_minimal.yml`  
`conda activate dsenv_minimal`  
`pip install .`  

However, not all functions will work with this environment.  

Functionality which requires the additional packages included in environment.yml:
- BigWigLookup:
  - Can score mutations using a BigWig file, e.g. Phylop conservation scores.
  - requires installation of pyBigWig
- StructuralDistanceLookup
  - Scores mutations based on distances between atoms in PDB files
  - requires installation of MDAnalysis
- ProDyLookup
  - Various options to score mutations based on protein structural dynamics
  - requires installation of ProDy
- FreeSASALookup
  - Scores mutations based on solvent accessible surface area
  - requires installation of FreeSASA
- Comparison of mutation distributions between data sets
  - Requires the installation of ROOT and compiling of ROOT code (see above)

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
print(notch3_results['CDF_MC_glob_k3_pvalue'])
notch3.plot_boxplot()
```

## Tips

There are various tutorials available in the [GitHub wiki](https://github.com/michaelhall28/darwinian_shift/wiki). 

The UniProt update in May 2022 altered their ID mapping (which was used to convert Ensembl transcript IDs to UniProt accessions). If you want to use the `UniProtLookup` or `PDBeKBLookup` classes or the `uniprot_exploration` or `pdbe_kb_exploration` functions, make sure you have the lastest code from the repository (30/05/22 or later).    

