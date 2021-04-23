from ..arch import Architecture
from ..matrix import DenseMatrix
from .gemm.nvidia import NvidiaGemmThreadPolicy
from .csa.nvidia import NvidiaCsaThreadPolicy
from .initializer.nvidia import NvidiaInitializerThreadPolicy


class TheadPolicyFactory:
  def __init__(self):
    pass

  @classmethod
  def get_gemm_policy(cls,
                      arch: Architecture,
                      reals_per_op: int,
                      num_threads: int,
                      bytes_per_real: int,
                      op1: DenseMatrix,
                      op2: DenseMatrix,
                      res: DenseMatrix):
    default_policy = NvidiaGemmThreadPolicy(arch,
                                            reals_per_op,
                                            num_threads,
                                            bytes_per_real,
                                            op1,
                                            op2,
                                            res)
    if arch.manufacturer in ['nvidia', 'amd', 'dg1']:
      return default_policy
    else:
      raise RuntimeError('unknown manufacturer')

  @classmethod
  def get_csa_policy(cls,
                     arch: Architecture,
                     num_threads: int,
                     op1: DenseMatrix,
                     op2: DenseMatrix):
    default_policy = NvidiaCsaThreadPolicy(arch, num_threads, op1, op2)
    if arch.manufacturer in ['nvidia', 'amd', 'dg1']:
      return default_policy
    else:
      raise RuntimeError('unknown manufacturer')

  @classmethod
  def get_initializer_policy(cls,
                             arch: Architecture,
                             num_threads: int,
                             op: DenseMatrix):
    default_policy = NvidiaInitializerThreadPolicy(arch, num_threads, op)
    if arch.manufacturer in ['nvidia', 'amd', 'dg1']:
      return default_policy
    else:
      raise RuntimeError('unknown manufacturer')