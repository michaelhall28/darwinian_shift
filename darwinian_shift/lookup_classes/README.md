Lookup classes
---------------

Classes which lookup the given metric for every possible mutation in the given region.
E.g. FoldXLookup assigns the ∆∆G value calculated from FoldX to each mutation in the pdb structure.

The classes are passed a Section object when called and can use any attributes of the object to lookup/calculate the metric.

A selection of lookup classes are provided here, but any function/class can be used as long as:
- it takes a Section object as an argument
- it returns an array of values for each of the mutations in the null_mutations dataframe of the Section.
