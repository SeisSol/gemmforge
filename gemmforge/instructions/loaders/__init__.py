from gemmforge.symbol_table import SymbolType
from gemmforge.exceptions import InternalError
from .shr_mem_loaders import ExtendedPatchLoader, ExactPatchLoader
from .shr_transpose_mem_loaders import ExtendedTransposePatchLoader, ExactTransposePatchLoader, ArbitraryLeadingDimensionExactTransposePatchLoader
from math import ceil


def shm_mem_loader_factory(vm, dest, src, shr_mem, num_threads, load_and_transpose=False):
  params = {'vm': vm,
            'dest': dest,
            'src': src,
            'shr_mem': shr_mem,
            'num_threads': num_threads,
            'load_and_transpose': load_and_transpose}

  num_loads_per_column = ceil(src.data_view.rows / num_threads) * num_threads

  if src.obj.leading_dimension_given:
    if src.data_view.lead_dim != src.data_view.rows:
      # We are dealing with a tensor slice that is not contigously stored,
      # In that case Extended doesn't really give any advantage
      if load_and_transpose:
        # TODO: Implement a better method of loading of a transposed tensor slice later
        # possibly using the same prime number approach as Ravil's
        return ArbitraryLeadingDimensionExactTransposePatchLoader(**params)
      return ExactPatchLoader(**params)
    #else:
    #  raise Exception(f"{src.data_view.lead_dim} == {src.data_view.rows}")

  if src.data_view.lead_dim > num_loads_per_column:
    if load_and_transpose:
      #return ExactTransposePatchLoader(**params)
      return ExactTransposePatchLoader(**params)
    else:
      return ExactPatchLoader(**params)
  else:
    if load_and_transpose:
      return ExtendedTransposePatchLoader(**params)
    else:
      return ExtendedPatchLoader(**params)
