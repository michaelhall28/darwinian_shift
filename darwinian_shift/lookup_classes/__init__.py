from .bigwig_lookup import BigwigLookup
from .iupred2a_lookup import IUPRED2ALookup
from .foldx_lookup import FoldXLookup
from .AAindex import AAindexLookup
from .prody_lookup import ProDyLookup
from .structure_distance import StructureDistanceLookup
from .combination_lookup import ORLookup, ANDLookup, MutationExclusionLookup
from .dummy_lookup import DummyValuesPosition, DummyValuesRandom, DummyValuesFixed
from .uniprot_lookup import UniprotLookup
from .sequence_distance_lookup import SequenceDistanceLookup
from .psic_lookup import PSICLookup
from .clinvar_lookup import ClinvarLookup
from .phosphorylation_lookup import PhosphorylationLookup