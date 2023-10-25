from math import ceil

from gemmforge.tensor.dense import DenseTensor
from .shr_mem_loaders import ExactPatchLoader, ExactTensorLoader, ExtendedPatchLoader, ExtendedTensorLoader
from .shr_transpose_mem_loaders import ArbitraryLeadingDimensionExactTransposePatchLoader, ExactTransposePatchLoader, \
    ExtendedTransposePatchLoader


def shm_mem_loader_factory(vm, dest, src, shr_mem, num_threads, load_and_transpose=False, prefer_exact=False):
  params = {'vm': vm,
            'dest': dest,
            'src': src,
            'shr_mem': shr_mem,
            'num_threads': num_threads,
            'load_and_transpose': load_and_transpose,
            'prefer_exact': prefer_exact}

  if isinstance(src.obj, DenseTensor):
    # num_loads_per_column = ceil(src.data_view.dimensions[0] / num_threads) * num_threads
    # if src.data_view.dimensions[0] > num_loads_per_column:
    #  return ExactTensorLoader(**params)
    # else:
    return ExtendedTensorLoader(**params)
  else:
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

    #if load_and_transpose and prefer_exact:
    #  return ExactTransposePatchLoader(**params)
    #elif src.data_view.lead_dim > num_loads_per_column:
    if load_and_transpose and prefer_exact and (src.obj.num_rows % num_threads == 0 or num_threads % src.obj.num_rows == 0):
      return ArbitraryLeadingDimensionExactTransposePatchLoader(**params)

    if src.data_view.lead_dim != src.obj.num_rows:
      if load_and_transpose:
        return ExactTransposePatchLoader(**params)
      else:
        return ExactPatchLoader(**params)
    else:
      if load_and_transpose:
        return ExtendedTransposePatchLoader(**params)
      else:
        return ExtendedPatchLoader(**params)
