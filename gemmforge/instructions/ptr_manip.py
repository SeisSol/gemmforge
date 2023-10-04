from gemmforge.basic_types import DataFlowDirection, GeneralLexicon
from gemmforge.common import get_extra_offset_name
from gemmforge.exceptions import GenerationError
from gemmforge.symbol_table import SymbolType
from gemmforge.vm import VM
from .abstract_instruction import AbstractInstruction


class GetElementPtr(AbstractInstruction):
  def __init__(self,
               vm: VM,
               src,
               dest,
               include_extra_offset=True):
    super(GetElementPtr, self).__init__(vm)
    self._src = src
    self._dest = dest
    self._include_extra_offset = include_extra_offset
    self._is_ready = True
    self._loop_additional_offset = None
    self._num_mults_per_block = 0

  def is_ready(self):
    return True
    """
    if self._src.stype == SymbolType.Batch:
      return True
    else:
      if self._num_mults_per_block != 0:
        return True
      else:
        return False
    """

  def set_mults_per_block(self, num_mults_per_block):
    self._num_mults_per_block = num_mults_per_block

  def gen_code(self, writer):
    if self._src.stype == SymbolType.Batch:
      batch_obj = self._src.obj
      batch_addressing = batch_obj.addressing

      if self._include_extra_offset:
        extra_offset = f' + {get_extra_offset_name(self._src.name)}'
      else:
        extra_offset = ''

      address = ''
      if batch_addressing == "strided":
        main_offset = f'{GeneralLexicon.BATCH_ID} * {batch_obj.get_real_volume()}'
        sub_offset = f'{batch_obj.get_offset_to_first_element()}'
        address = f'{main_offset} + {sub_offset}{extra_offset}'
      elif batch_addressing == "pointer_based":
        main_offset = f'{GeneralLexicon.BATCH_ID}'
        sub_offset = f'{batch_obj.get_offset_to_first_element()}'
        address = f'{main_offset}][{sub_offset}{extra_offset}'
        if self._loop_additional_offset:
          address += self._loop_additional_offset
      elif batch_addressing == "none":
        address = f'{batch_obj.get_offset_to_first_element()}{extra_offset}'
      else:
        GenerationError(f'unknown addressing of {self._src.name}, given {batch_addressing}')

      rhs = f'&{self._src.name}[{address}]'

      lhs = 'const ' if self._src.obj.direction == DataFlowDirection.SOURCE else ''
      lhs += f'{self._vm.fp_as_str()} * const __restrict__ {self._dest.name}'
      writer(f'{lhs} = {rhs};')
    else:
      batch_obj = self._src.obj
      batch_addressing = batch_obj.addressing

      address = f'{batch_obj.get_offset_to_first_element()}'

      rhs = f'&{self._src.name}[{address}]'

      lhs = 'const ' if self._src.obj.direction == DataFlowDirection.SOURCE else ''
      lhs += f'{self._vm.fp_as_str()} * const __restrict__ {self._dest.name}'
      writer(f'{lhs} = {rhs};')

  def __str__(self) -> str:
    return f'{self._dest.name} = getelementptr_b2g {self._src.name};'
