import warnings

not_loaded = []

from .iupred2a_lookup import IUPRED2ALookup
from .foldx_lookup import FoldXLookup
from .AAindex import AAindexLookup
from .combination_lookup import ORLookup, ANDLookup, MutationExclusionLookup
from .dummy_lookup import DummyValuesPosition, DummyValuesRandom, DummyValuesFixed
from .uniprot_lookup import UniprotLookup
from .sequence_distance_lookup import SequenceDistanceLookup
from .clinvar_lookup import ClinvarLookup
from .phosphorylation_lookup import PhosphorylationLookup
from .pdbekb_lookup import PDBeKBLookup
from .variant_match_lookup import VariantMatchLookup

try:
    from .bigwig_lookup import BigwigLookup
except ImportError as e:
    not_loaded.append("BigwigLookup")

try:
    from .structure_distance import StructureDistanceLookup
except ImportError as e:
    not_loaded.append("StructureDistanceLookup")

try:
    from .prody_lookup import ProDyLookup
except ImportError as e:
    not_loaded.append("ProDyLookup")

try:
    from .freeSASA_lookup import FreeSASALookup
except ImportError as e:
    not_loaded.append("FreeSASALookup")

if not_loaded:
    msg = "Could not load the lookups: {}"
    msg += "\nIf these are needed, please check the required package dependencies are installed (see README)."
    msg = msg.format(", ".join(not_loaded))
    warnings.warn(msg)