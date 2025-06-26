"""
Vendored Descript Audio Codec (DAC) for zonos_local_lib
"""
# Attempt to import audiotools and handle if not found.
# The DAC code itself will raise ImportErrors if audiotools is truly needed and not present.
try:
    import audiotools
    # The following lines might modify a global state in audiotools.
    # These were in the original dac/__init__.py.
    # For our vendored version, we refer to `.model` etc.
    # The original `dac.**` might need to be `src.zonos_local_lib.dac_codec.**` if audiotools uses it for discovery,
    # but it's safer to assume these are not strictly necessary for basic model usage here or handle them if errors arise.
    # if hasattr(audiotools, 'ml') and hasattr(audiotools.ml, 'BaseModel'):
    #     audiotools.ml.BaseModel.INTERN += ["src.zonos_local_lib.dac_codec.**"]
    #     audiotools.ml.BaseModel.EXTERN += ["einops"]
except ImportError:
    print("WARNING: Vendored DAC - audiotools not found. Some functionalities might be limited or fail later.")
    pass

from .model import DAC
from .model import DACFile # DACFile might be used by DAC.compress/decompress methods

__all__ = ["DAC", "DACFile"]
