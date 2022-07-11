from .abstract_instruction import AbstractInstruction
from gemmforge.vm import VM
from gemmforge.symbol_table import DataView, Symbol
from gemmforge.basic_types import GeneralLexicon
from gemmforge.exceptions import InternalError


class RegisterAlloc(AbstractInstruction):
  def __init__(self,
               vm: VM,
               dest: Symbol,
               init_value: float):
    super(RegisterAlloc, self).__init__(vm)
    self._dest = dest
    self._init_value = init_value
    self._is_ready = True
    

  def gen_code(self, writer):
    if self._dest.obj.size < 1:
      raise InternalError('size of reg. obj must be at least 1')

    if self._dest.obj.size == 1:
      init_value = ''
      if isinstance(self._init_value, float):
        init_value = f' = {self._init_value}'
      result = f'{self._vm.fp_as_str()} {self._dest.obj.name}{init_value};'
    else:
      init_values_list = ''
      if isinstance(self._init_value, float):
        real_literal = self._vm.get_real_literal()
        init_values = ', '.join([f'{str(self._init_value)}{real_literal}'] * self._dest.obj.size)
        init_values_list = f' = {{{init_values}}}'
      result = f'{self._vm.fp_as_str()} {self._dest.obj.name}[{self._dest.obj.size}]{init_values_list};' + "// test test comment"
    writer(result)

  def __str__(self) -> str:
    return f'{self._dest.obj.name} = alloc_regs {self._dest.obj.size}; '


class ShrMemAlloc(AbstractInstruction):
  def __init__(self,
               vm: VM,
               dest: Symbol,
               size,
               op2: Symbol):
    super(ShrMemAlloc, self).__init__(vm)
    self._size = size
    self._dest = dest
    self._is_ready = False
    self._op2 = op2

  def gen_code(self, writer):
    op2_data_view = self._op2.data_view
    shrmem_obj = self._dest.obj
    common_shrmem = f'{GeneralLexicon.TOTAL_SHR_MEM}'
    common_shrmem_size = shrmem_obj.get_total_size()

    lexic = self._vm.get_lexic()
    shr_mem_decl = lexic.declare_shared_memory_inline(name=common_shrmem,
                                                      precision=self._vm.fp_as_str(),
                                                      size=common_shrmem_size,
                                                      alignment=8)
    if op2_data_view.spp == None:
      if shr_mem_decl:
        writer(f'{shr_mem_decl}; riri7')

      address = f'{shrmem_obj.get_size_per_mult()} * {lexic.thread_idx_y} '
      writer(f'{self._vm.fp_as_str()} * {shrmem_obj.name} = &{common_shrmem}[{address}]; riri6' )

  def is_ready(self):
    shrmem_obj = self._dest.obj
    if shrmem_obj.get_total_size():
      return True
    else:
      return False

  def __str__(self):
    return f'{self._dest.name} = alloc_shr [{self._dest.obj.get_total_size_as_str()}];'
