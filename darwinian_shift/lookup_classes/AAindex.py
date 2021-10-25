import pandas as pd
import numpy as np
from .errors import MetricLookupException

class TableLookupError(MetricLookupException): pass

class AAindexEntry():
    def __init__(self, accession, data_description, pmid, authors, title, journal_ref, similar, aadata):
        self.accession = accession
        self.data_description = data_description
        self.pmid = pmid
        self.authors = authors
        self.title = title
        self.journal_ref = journal_ref
        self.similar = similar
        self.aadata = aadata

    def __repr__(self):
        s = ""
        s += 'Accession: {}\n'.format(self.accession)
        s += 'data_description: {}\n'.format(self.data_description)
        s += 'pmid: {}\n'.format(self.pmid)
        s += 'title: {}\n'.format(self.title)
        s += 'journal_ref: {}\n'.format(self.journal_ref)
        s += 'similar:\n{}\n'.format(self.similar)
        s += 'aadata:\n{}\n'.format(self.aadata)
        return s


class AAindex():
    # Read and store the aaindex data
    # Make a big dataframe of results.
    def __init__(self, aaindex_file):
        self.data = {}
        self.index = None

        self._read_aaindex(aaindex_file)
        self._make_index()

    def _read_aaindex(self, aaindex_file):
        """
        From the documentation

        * Each entry has the following format.                                 *
        *                                                                      *
        * H Accession number                                                   *
        * D Data description                                                   *
        * R PMID                                                               *
        * A Author(s)                                                          *
        * T Title of the article                                               *
        * J Journal reference                                                  *
        * * Comment or missing                                                 *
        * C Accession numbers of similar entries with the correlation          *
        *   coefficients of 0.8 (-0.8) or more (less).                         *
        *   Notice: The correlation coefficient is calculated with zeros       *
        *   filled for missing values.                                         *
        * I Amino acid index data in the following order                       *
        *   Ala    Arg    Asn    Asp    Cys    Gln    Glu    Gly    His    Ile *
        *   Leu    Lys    Met    Phe    Pro    Ser    Thr    Trp    Tyr    Val *

        """
        res = []
        aaline = 0
        with open(aaindex_file) as fh:
            similar_data = []
            similar = False
            aa_res = []
            aa_line = 0
            for line in fh:
                if line.startswith('//'):
                    self.data[desc] = AAindexEntry(accession, desc, pmid, authors, title, journal_ref, similar_data,
                                                   aa_res)
                    aa_line = 0
                    similar_data = []
                    aa_res = []
                elif aa_line > 0:
                    aa_res.extend(self._read_aa(line, aa_line))
                    aa_line += 1
                    if aa_line == 3:
                        aa_res = pd.DataFrame(aa_res, columns=['AA', 'Value'])
                elif line.startswith('H'):
                    # Accession number
                    accession = self._read_line(line)
                elif line.startswith('D'):
                    desc = self._read_line(line)
                    D = True
                elif line.startswith('R'):
                    pmid = self._read_line(line)
                    D = False
                elif line.startswith('A'):
                    authors = self._read_line(line)
                    A = True
                elif line.startswith('T'):
                    title = self._read_line(line)
                    A = False
                    T = True
                elif line.startswith('J'):
                    journal_ref = self._read_line(line)
                    J = True
                    T = False
                elif line.startswith('C'):
                    J = False
                    similar = True
                    similar_data.extend(self._read_similar(line))
                elif line.startswith('I'):
                    aa_line = 1
                    similar = False
                    if similar_data:
                        similar_data = pd.DataFrame(similar_data, columns=['accession', 'corr_coef'])
                    else:
                        similar_data = None
                elif similar:
                    similar_data.extend(self._read_similar(line))
                elif D:
                    desc += " " + line.strip()
                elif A:
                    authors += " " + line.strip()
                elif T:
                    title += " " + line.strip()
                elif J:
                    journal_ref += " " + line.strip()

    def _read_line(self, line):
        try:
            res = line.strip().split()
            res = " ".join(res[1:])
        except IndexError as e:  # Empty
            res = None
        return res

    def _read_similar(self, line):
        res = []
        for i, j in enumerate(line.strip()[2:].split()):
            if i % 2 == 0:
                acc = j
            else:
                res.append({'accession': acc, 'corr_coef': self._convert_to_numeric(j)})
        return res

    def _read_aa(self, line, aa_line):
        if aa_line == 1:
            amino_acids = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I']
        elif aa_line == 2:
            amino_acids = ['L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']
        res = []
        for aa, value in zip(amino_acids, line.strip().split()):
            res.append({'AA': aa, 'Value': self._convert_to_numeric(value)})
        return res

    def _convert_to_numeric(self, string):
        # Integer values are written as 16.
        if string == 'NA':
            return np.nan
        elif string.endswith('.'):
            return int(string[:-1])
        else:
            return float(string)

    def _make_index(self):
        columns = ['accession', 'data_description', 'pmid', 'authors', 'title', 'journal_ref']
        res = []
        for i, aa in self.data.items():
            res.append({k: getattr(aa, k) for k in columns})
        self.index = pd.DataFrame(res, columns=columns)

    def data_from_description(self, desc):
        if desc in self.data:
            return self.data[desc]
        else:
            possible_matches = []
            for t in self.data.keys():
                if desc.lower() in t.lower():
                    possible_matches.append(t)
            print('Try one of these titles:')
            for p in possible_matches:
                print("\t", p)
            raise TableLookupError('No AAindex table with the requested description')

    def get_aadata(self, desc):
        data = self.data_from_description(desc)
        if data is not None:
            return data.aadata

    def get_data(self, desc):
        data = self.data_from_description(desc)
        if data is not None:
            return data

    def get_all_summary_info(self):
        return self.index

    def print_all_data_descriptions(self):
        for i in self.index['data_description']:
            print(i)


class AAindexLookup:
    """
    Assigns a score to each mutation based on one of the tables in the AAindex,
    Kawashima and Kanehisa, AAindex: Amino Acid index database, Nucleic Acids Research, 2000

    The data first needs dowloading from https://www.genome.jp/aaindex/
    The required file is aaindex1

    The score returned will be the change from the wild type amino acid value to the mutant amino acid value.

    The tables can be searched by entering a term as the table description. This will raise an error, but will print
    a list of the table titles that contain the term:
    E.g.
    > AAindexLookup("/path/to/aaindex1", table_description='polar', abs_value=False)
      Try one of these titles:
	      Polarizability parameter (Charton-Charton, 1982)
	      Polarity (Grantham, 1974)
	      Mean polarity (Radzicka-Wolfenden, 1988)
	      Polar requirement (Woese, 1973)
	      Polarity (Zimmerman et al., 1968)
    """
    def __init__(self, aaindex_file, table_description, abs_value, name=None):
        """

        :param aaindex_file: File path to aaindex1
        :param table_description: The exact name of the table to use.
        :param abs_value: If true, will return the absolute value of the change from wild type to mutant amino acid.
        :param name: Name of the lookup to appear on plot axes.
        """
        aaindex = AAindex(aaindex_file)
        self.aatable = aaindex.get_aadata(table_description)  # Will raise an error if not a matching description
        self.abs_value = abs_value
        if name is not None:
            self.name = name  # Will appear on some plot axes
        else:
            self.name = "Change in " + table_description

    def __call__(self, seq_object):
        return self._get_scores(seq_object.null_mutations)

    def _get_scores(self, df):
        merge_df = pd.merge(df, self.aatable, left_on=['aaref'], right_on=['AA'], how='left')
        merge_df['ref_score'] = merge_df['Value']
        merge_df.drop(['AA','Value'], axis=1, inplace=True)
        merge_df = pd.merge(merge_df, self.aatable, left_on=['aamut'], right_on=['AA'], how='left')
        merge_df['mut_score'] = merge_df['Value']
        scores = (merge_df['mut_score'] - merge_df['ref_score']).values
        if self.abs_value:
            return np.abs(scores)
        return scores