from collections import namedtuple
import pandas as pd
import numpy as np

Scores = namedtuple('Scores', ['Benign', 'Likely_benign', 'Uncertain_significance', 'not_provided',
                               'Conflicting_interpretations_of_pathogenicity',
                               'Likely_pathogenic', 'Pathogenic', 'missing'
                               ])


class ClinvarLookup:
    # Where there are multiple entries, will take the highest score

    default_scores = Scores(Benign=0, Likely_benign=0,
                            Uncertain_significance=0, not_provided=0,
                            Conflicting_interpretations_of_pathogenicity=0,
                            Likely_pathogenic=1, Pathogenic=1, missing=0)

    def __init__(self, clinvar_variant_summary_file, assembly, scores=None, clinsigsimple=False,
                 clinsigsimple_missing_score=0):
        self.clinvar_data = pd.read_csv(clinvar_variant_summary_file, sep="\t")
        self.clinvar_data = self.clinvar_data[self.clinvar_data['Assembly'] == assembly]
        self.clinsigsimple = clinsigsimple
        self.clinsigsimple_missing_score = clinsigsimple_missing_score

        if scores is None:
            self.scores = self.default_scores
        else:
            self.scores = scores

    def __call__(self, seq_object):
        return self._get_scores(seq_object.chrom, seq_object.null_mutations)

    def _get_score(self, clinsig):
        if pd.isnull(clinsig):
            return self.scores.missing

        split_clinsig = clinsig.replace("/", ",_").replace(" ", "_").split(',_')
        scores = [getattr(self.scores, c, np.nan) for c in split_clinsig]
        return np.nanmax(scores)

    def _merge_with_clinvar(self, chrom, df):
        chrom_clinvar = self.clinvar_data[self.clinvar_data['Chromosome'] == chrom]
        merge_df = pd.merge(df, chrom_clinvar, left_on=['pos', 'ref', 'mut'],
                            right_on=['Start', 'ReferenceAllele', 'AlternateAllele'], how='left')
        return merge_df

    def _get_scores(self, chrom, df):
        merge_df = self._merge_with_clinvar(chrom, df)

        if self.clinsigsimple:
            score = merge_df['ClinSigSimple']
            if not pd.isnull(self.clinsigsimple_missing_score):
                return score.fillna(self.clinsigsimple_missing_score)
            return score
        else:
            return merge_df['ClinicalSignificance'].apply(self._get_score).values

    def annotate_df(self, chrom, df):
        return self._merge_with_clinvar(chrom, df)