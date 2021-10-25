class VariantMatchLookup:
    """
    Match variants against a list provided.
    The match can be made using any column in the variant dataframe.

    Example:
        # Set up the lookup to match against the aachange column.
        v = VariantMatchLookup(
                match_column='aachange',
                target_key='match_list'
            )
        # Run with a match_list listing the aachanges to match against.
        d.run_section({
            'gene': GENESYMBOL,
            'match_list': ['S100R', 'G140Y', 'V155K']
            }
        )
        Mutations in the match list will be scored 1, all other mutations will be scored zero.

    """

    def __init__(self, match_column='aachange', target_key='match_list', name=None):
        """

        :param match_column: The column in the dataframe to use to match the mutations.
        :param target_key: The name of the section attribute which lists the target mutations.
        :param name: Name of the lookup to appear on plot axes.
        """
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