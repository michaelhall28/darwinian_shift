import pandas as pd
import os

class IUPRED2ALookup:
    """
    For reading the text results file of IUPred2A when running either long or short disorder.
    These files have to be downloaded prior to running the analysis.

    Requires the iupred_file attribute to be defined for the section being analysed. This should be the file
    path of the iupred text results file relative to the iupred_results_dir .

    """

    def __init__(self, iupred_results_dir=".", results_column="IUPRED SCORE", name='IUPRED2A score'):
        """

        :param iupred_results_dir: Directory where the iupred results are stored
        :param results_column: Name of column to use for the mutation scores.
        :param name: Name of the lookup to appear on plot axes.
        """
        self.results_column = results_column
        self.iupred_results_dir = iupred_results_dir
        self.name = name

    def __call__(self, seq_object):
        return self._get_scores(seq_object.null_mutations, seq_object.iupred_file)

    def read_iupreda2_results(self, file_name):
        res_lines = []
        columns = None
        with open(os.path.join(self.iupred_results_dir, file_name)) as fh:

            for line in fh:
                if line.startswith("#"):
                    columns = line[2:].strip().split("\t")
                    continue
                if line.startswith('<'):
                    continue

                line = line.strip().split("\t")
                if len(line) > 1:
                    res_lines.append(line)

        results = pd.DataFrame(res_lines, columns=columns)
        results['POS'] = results["POS"].astype(int)
        results[self.results_column] = results[self.results_column].astype(float)
        return results

    def _get_scores(self, df, iupred_file):
        iupred = self.read_iupreda2_results(iupred_file)
        merge_df = pd.merge(df, iupred, left_on='residue', right_on='POS', how='left')
        scores = merge_df[self.results_column].values
        return scores