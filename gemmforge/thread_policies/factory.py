from gemmforge.vm import VM
from ..matrix import DenseMatrix, SparseMatrix
from .gemm.generic import GenericGemmThreadPolicy
from .gemm.only_register_based import OnlyRegisterBasedThreadPolicy
from .csa.generic import GenericCsaThreadPolicy
from typing import Union
from .gemm.dense_sparse import GenericDenseSparseGemmThreadPolicy
from .gemm.dense_sparse import DenseSparseOnlyRegisterBasedThreadPolicy, GenericDenseSparseGemmThreadPolicy
from .gemm.sparse_dense import GenericSparseDenseGemmThreadPolicy, SparseDenseOnlyRegisterBasedThreadPolicy

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
                      op2: Union[DenseMatrix, SparseMatrix],
                      res: DenseMatrix):

    hw_descr = vm.get_hw_descr()
    if hw_descr.manufacturer in TheadPolicyFactory.ALLOWED_MANUFACTURES:
      if shr_mem_per_op == 0:
        if isinstance(op1, DenseMatrix) and isinstance(op2, DenseMatrix):
          return OnlyRegisterBasedThreadPolicy(vm,
                                               num_threads,
                                               op1,
                                               op2,
                                               res)
        elif isinstance(op1, SparseMatrix):
          return SparseDenseOnlyRegisterBasedThreadPolicy(vm,
                                                          num_threads,
                                                          op1,
                                                          op2,
                                                          res)
        elif isinstance(op2, SparseMatrix):
          return DenseSparseOnlyRegisterBasedThreadPolicy(vm,
                                                          num_threads,
                                                          op1,
                                                          op2,
                                                          res)
        else:
          raise RuntimeError('Unknown Matrix type')
      else:
        if isinstance(op1, DenseMatrix) and isinstance(op2, DenseMatrix):
          return GenericGemmThreadPolicy(vm,
                                         shr_mem_per_op,
                                         num_threads,
                                         op1,
                                         op2,
                                         res)
        elif isinstance(op1, SparseMatrix):
          return GenericSparseDenseGemmThreadPolicy(vm,
                                                    shr_mem_per_op,
                                                    num_threads,
                                                    op1,
                                                    op2,
                                                    res)
        elif isinstance(op2, SparseMatrix):
          return GenericDenseSparseGemmThreadPolicy(vm,
                                                    shr_mem_per_op,
                                                    num_threads,
                                                    op1,
                                                    op2,
                                                    res)
        else:
          raise RuntimeError('Unknown Matrix type')
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
