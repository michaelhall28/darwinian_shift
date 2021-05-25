class VariantMatchLookup:
    """
    Match variants against a list provided.
    Or can match any column in the variant dataframe.
    """

    def __init__(self, match_column='aachange', target_key='match_list', name=None):
        self.match_column = match_column
        self.target_key = target_key
        if name is None:
            self.name = 'Variant matching ' + match_column
        else:
            self.name = name  # Will appear on some plot axes

    def __call__(self, seq_object):
        target_list = getattr(seq_object, self.target_key, None)
        if target_list is None:
            raise ValueError('Variant match list {} not defined for section {}'.format(self.target_key,
                                                                                       seq_object.section_id))
        return self._get_match(seq_object.null_mutations, target_list)

    def _get_match(self, df, target_list):
        return df[self.match_column].isin(target_list)