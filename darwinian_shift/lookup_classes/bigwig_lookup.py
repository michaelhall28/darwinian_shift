import pyBigWig

class BigwigLookup:
    def __init__(self, bw_file, chr_prefix=True, name='Bigwig values'):
        self.bw = pyBigWig.open(bw_file)
        self.chr_prefix= chr_prefix
        self.name = name  # Will appear on some plot axes

    def __call__(self, seq_object):
        return self._get_scores(seq_object.chrom, seq_object.null_mutations)

    def _get_score_for_range(self, chrom, start, end):
        reverse = False
        if start > end:
            reverse = True
            start, end = end, start

        # Convert to Python int. pyBigWig doesn't like numpy.int64.
        values = self.bw.values(chrom, int(start) - 1, int(end))

        if reverse:
            values = values[::-1]
        return values

    def _get_scores(self, chrom, df):
        min_pos = df['pos'].min()
        max_pos = df['pos'].max()
        # For SNPs only.
        if not isinstance(chrom, str):
            chrom = str(chrom)
        if self.chr_prefix and not chrom.startswith('chr'):
            chrom = 'chr' + chrom
        elif not self.chr_prefix and chrom.startswith('chr'):
            chrom = chrom[3:]

        res = self._get_score_for_range(chrom, min_pos, max_pos)
        scores = df.apply(lambda x: res[x['pos'] - min_pos], axis=1).values
        return scores