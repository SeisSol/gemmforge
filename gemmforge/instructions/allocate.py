from gemmforge.basic_types import GeneralLexicon
from gemmforge.exceptions import InternalError
from gemmforge.symbol_table import Symbol
from gemmforge.vm import VM
from .abstract_instruction import AbstractInstruction


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

    if isinstance(self._dest.obj.size, float):
      assert self._dest.obj.size == int(self._dest.obj.size)
      self._dest.obj.size = int(self._dest.obj.size)

    if self._dest.obj.size == 1:
      init_value = ''
      if isinstance(self._init_value, float):
        init_value = f' = {self._init_value}'
      result = f'{self._vm.fp_as_str()} {self._dest.obj.name}{init_value};'
    else:
      init_values_list = ''
      if isinstance(self._init_value, float):
        real_literal = self._vm.get_real_literal()
        init_values = ', '.join([f'{str(self._init_value)}{real_literal}'] * int(self._dest.obj.size))
        if self._init_value != 0.0:
          init_values_list = f'{str(self._init_value)}{real_literal},'
        else:
          init_values_list = f' = {{{init_values}}}'
      assert self._dest.obj.size == int(self._dest.obj.size)
      result = f'{self._vm.fp_as_str()} {self._dest.obj.name}[{self._dest.obj.size}]{init_values_list};'
    writer(result)

  def __str__(self) -> str:
    return f'{self._dest.obj.name} = alloc_regs {self._dest.obj.size};'


class ShrMemAlloc(AbstractInstruction):
  def __init__(self,
               vm: VM,
               dest: Symbol,
               size):
    super(ShrMemAlloc, self).__init__(vm)
    self._size = size
    self._dest = dest
    self._is_ready = False

  def gen_code(self, writer):
    shrmem_obj = self._dest.obj
    common_shrmem = f'{GeneralLexicon.TOTAL_SHR_MEM}'
    common_shrmem_size = shrmem_obj.get_total_size()

    lexic = self._vm.get_lexic()
    shr_mem_decl = lexic.declare_shared_memory_inline(name=common_shrmem,
                                                      precision=self._vm.fp_as_str(),
                                                      size=common_shrmem_size,
                                                      alignment=8)

    if shr_mem_decl:
      writer(f'{shr_mem_decl};')

    address = f'{shrmem_obj.get_size_per_mult()} * {lexic.thread_idx_y}'
    writer(f'{self._vm.fp_as_str()} * {shrmem_obj.name} = &{common_shrmem}[{address}];')

  def is_ready(self):
    shrmem_obj = self._dest.obj
    if shrmem_obj.get_total_size():
      return True
    else:
      return False

  def __str__(self):
    return f'{self._dest.name} = new_alloc_shr [{self._dest.obj.get_total_size_as_str()}];'


class ShrMemNewAlloc(AbstractInstruction):
  def __init__(self,
               vm: VM,
               dest: Symbol,
               size):
    super(ShrMemNewAlloc, self).__init__(vm)
    self._size = size
    self._dest = dest
    self._is_ready = False
    self._mults_per_block = None

  def gen_code(self, writer):
    shrmem_obj = self._dest.obj
    lexic = self._vm.get_lexic()

    common_shrmem = f'{shrmem_obj.name}_alloc'
    common_shrmem_size = shrmem_obj.get_total_size() * self._mults_per_block

    shr_mem_decl = lexic.declare_shared_memory_inline(name=common_shrmem,
                                                      precision=self._vm.fp_as_str(),
                                                      size=common_shrmem_size,
                                                      alignment=8)

    if shr_mem_decl:
      if writer != None:
        writer(f'{shr_mem_decl};')

    address = f'{shrmem_obj.get_total_size()} * {lexic.thread_idx_y}'
    if writer != None:
      writer(f'{self._vm.fp_as_str()} * {shrmem_obj.name} = &{common_shrmem}[{address}];')

  def is_ready(self):
    if self._mults_per_block:
      return True
    else:
      return False

  def set_mults_per_block(self, mults_per_block):
    self._mults_per_block = mults_per_block

  def __str__(self):
    return f'{self._dest.name} = new_alloc_shr [{self._dest.obj.name}];'


class ShrMemNewAssign(AbstractInstruction):
  def __init__(self,
               vm: VM,
               dest: Symbol,
               src: str):
    super(ShrMemNewAssign, self).__init__(vm)
    self._src = src
    self._dest = dest
    self._is_ready = False
    self._mults_per_block = None

  def gen_code(self, writer):
    shrmem_obj = self._dest.obj
    lexic = self._vm.get_lexic()

    address = f'{shrmem_obj.get_total_size()} * {lexic.thread_idx_y}'
    writer(f'{self._vm.fp_as_str()} * {shrmem_obj.name} = &{self._src}[0];')

  def is_ready(self):
    if self._mults_per_block:
      return True
    else:
      return False

  def set_mults_per_block(self, mults_per_block):
    self._mults_per_block = mults_per_block

  def __str__(self):
    return f'{self._dest.name} = new_alloc_shr [{self._dest.obj.name}];'