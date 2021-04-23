from ..arch import Architecture
from ..matrix import DenseMatrix
from abc import ABC, abstractmethod


class AbstractUniOpThreadPolicy(ABC):
  def __init__(self,
               arch: Architecture,
               num_threads: int,
               op1: DenseMatrix):
    self._arch: Architecture = arch
    self._num_threads: int = num_threads
    self._op1: DenseMatrix = op1

  @abstractmethod
  def get_num_ops_per_block(self):
    pass


class AbstractBinaryOpThreadPolicy(AbstractUniOpThreadPolicy):
  def __init__(self,
               arch: Architecture,
               num_threads: int,
               op1: DenseMatrix,
               op2: DenseMatrix):
    super().__init__(arch, num_threads, op1)
    self._op2: DenseMatrix = op2


class AbstractGemmLikeThreadPolicy(AbstractBinaryOpThreadPolicy):
  def __init__(self,
               arch: Architecture,
               reals_per_op: int,
               num_threads: int,
               bytes_per_real: int,
               op1: DenseMatrix,
               op2: DenseMatrix,
               res: DenseMatrix):
    super().__init__(arch, num_threads, op1, op2)
    self._reals_per_op: int = reals_per_op
    self._bytes_per_real = bytes_per_real
    self._res: DenseMatrix = res



