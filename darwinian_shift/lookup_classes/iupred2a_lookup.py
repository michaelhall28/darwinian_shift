import pandas as pd

class IUPRED2ALookup:
    def __init__(self, iuored_results_file, results_column="IUPRED2 REDOX PLUS", name='IUPRED2A score'):
        self.results_column = results_column
        self.iupred_results = self.read_iupreda2_results(iuored_results_file)
        self.name = name  # Will appear on some plot axes

    def __call__(self, seq_object):
        return self._get_scores(seq_object.transcript_id, seq_object.null_mutations)

    def read_iupreda2_results(self, file_name):
        with open(file_name) as fh:
            transcript = None
            in_results = False
            results = {}
            transcript_res_lines = []
            for line in fh:
                if line.startswith(">"):
                    transcript = line.strip()[1:]
                    in_results = True
                elif line.startswith("####"):
                    in_results = False
                    a = pd.DataFrame(transcript_res_lines[1:], columns=transcript_res_lines[0])
                    a = a[~pd.isnull(a["AMINO ACID"])]
                    a['# POS'] = a["# POS"].astype(int)
                    a[self.results_column] = a[self.results_column].astype(float)
                    results[transcript] = a
                    transcript_res_lines = []
                elif in_results:
                    transcript_res_lines.append(line.strip().split("\t"))
        return results

    def _get_scores(self, transcript_id, df):
        iupred = self.iupred_results[transcript_id]
        merge_df = pd.merge(df, iupred, left_on='residue', right_on='# POS', how='left')
        scores = merge_df[self.results_column].values
        return scores