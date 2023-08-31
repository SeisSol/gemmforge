from .abstract_instruction import AbstractInstruction
from gemmforge.vm import VM
from gemmforge.symbol_table import SymbolType, Symbol, DataView, InverseSymbolTable
from gemmforge.basic_types import GeneralLexicon, DataFlowDirection, RegMemObject
from gemmforge.exceptions import InternalError
from abc import abstractmethod


class ShrMemBasedProduct(AbstractInstruction):
  """This is a gemm operation which is based on pre-loading data into
  the shared memory. This operation performs well on Nvidia
  and AMD GPUs"""

  def __init__(self, **kwargs):
    super(ShrMemBasedProduct, self).__init__(kwargs['vm'])
    self._ops = kwargs['ops']
    self._dest = kwargs['dest']
    self._num_threads = kwargs['num_threads']

    self._is_ready = True

  def gen_code(self, writer):
    writer("/*")
    writer(f"TODO")
    writer("*/")

  def __str__(self) -> str:
    return f'{self._dest.name} = product(TODO...)'

class RegisterOnlyProduct(AbstractInstruction):
  def __init__(self, **kwargs):
    super(RegisterOnlyProduct, self).__init__(kwargs['vm'])
    raise Exception("Register Only Product Kernel is not yet supported")