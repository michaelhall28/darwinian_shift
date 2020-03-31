from sqlalchemy import create_engine


class PSICLookup:
    """Uses the Polyphen2 sqlite database to get PSIC scores for missense mutations"""
    def __init__(self, sqlfile, name='PSIC score'):
        engine = create_engine('sqlite:///' + sqlfile)
        self.conn = engine.connect()
        self.name = name  # Will appear on some plot axes

    def __call__(self, seq_object):
        return self._get_scores(seq_object.chrom, seq_object.null_mutations)

    def get_score_for_mutation(self, chrom, row):
        res = self.conn.execute("SELECT dscore "
                                "FROM features JOIN scores USING(id) "
                                "WHERE chrom = 'chr{}' AND chrpos = {}"
                                " AND nt1 = '{}' AND nt2 = '{}'".format(chrom, row['pos'], row['strand_ref'],
                                                                        row['strand_mut'])
                                ).fetchone()
        if res is None:
            return None
        else:
            return res[0]

    def _get_scores(self, chrom, df):
        scores = df.apply(lambda x: self.get_score_for_mutation(chrom, x), axis=1).values
        return scores