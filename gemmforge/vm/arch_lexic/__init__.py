from .abstract_arch_lexic import AbstractArchLexic
from .nvidia_arch_lexic import NvidiaArchLexic
from .amd_arch_lexic import AmdArchLexic
from .sycl_arch_lexic import SyclArchLexic


def lexic_factory(backend):
  if backend == "cuda":
    return NvidiaArchLexic()
  elif backend == "hip":
    return AmdArchLexic()
  elif backend == "hipsycl" or backend == "oneapi":
    return SyclArchLexic()
  else:
    raise ValueError(f'Unknown backend, given: {backend}')
