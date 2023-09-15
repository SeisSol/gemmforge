from typing import List

from gemmforge.tensor.dense import DenseTensor
from gemmforge.thread_policies.product.generic import GenericProductThreadPolicy
from gemmforge.vm import VM
from .csa.generic import GenericCsaThreadPolicy
from .gemm.generic import GenericGemmThreadPolicy
from .gemm.only_register_based import OnlyRegisterBasedThreadPolicy
from ..matrix import DenseMatrix


class TheadPolicyFactory:
  ALLOWED_MANUFACTURES = ['nvidia', 'amd', 'intel']

  def __init__(self):
    pass

  @classmethod
  def get_gemm_policy(cls,
                      vm: VM,
                      shr_mem_per_op: int,
                      num_threads: int,
                      op1: DenseMatrix,
                      op2: DenseMatrix,
                      res: DenseMatrix):

    hw_descr = vm.get_hw_descr()
    if hw_descr.manufacturer in TheadPolicyFactory.ALLOWED_MANUFACTURES:
      if shr_mem_per_op == 0:
        return OnlyRegisterBasedThreadPolicy(vm,
                                             num_threads,
                                             op1,
                                             op2,
                                             res)
      else:
        return GenericGemmThreadPolicy(vm,
                                       shr_mem_per_op,
                                       num_threads,
                                       op1,
                                       op2,
                                       res)
    else:
      raise RuntimeError('unknown manufacturer')

  @classmethod
  def get_csa_policy(cls,
                     vm: VM,
                     num_threads: int,
                     op1: DenseMatrix,
                     op2: DenseMatrix):
    default_policy = GenericCsaThreadPolicy(vm, num_threads, op1, op2)
    hw_descr = vm.get_hw_descr()
    if hw_descr.manufacturer in TheadPolicyFactory.ALLOWED_MANUFACTURES:
      return default_policy
    else:
      raise RuntimeError('unknown manufacturer')

  @classmethod
  def get_product_policy(cls,
                         vm: VM,
                         shr_mem_per_op: int,
                         num_threads: int,
                         ops: List[DenseTensor],
                         res: DenseTensor):
    default_policy = GenericProductThreadPolicy(vm,
                                                shr_mem_per_op,
                                                num_threads,
                                                ops,
                                                res)
    hw_descr = vm.get_hw_descr()
    if hw_descr.manufacturer in TheadPolicyFactory.ALLOWED_MANUFACTURES:
      return default_policy
    else:
      raise RuntimeError('unknown manufacturer')
