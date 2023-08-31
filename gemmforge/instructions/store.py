from .abstract_instruction import AbstractInstruction
from gemmforge.vm import VM
from gemmforge.symbol_table import SymbolType, Symbol, DataView, InverseSymbolTable
from gemmforge.basic_types import GeneralLexicon, DataFlowDirection, RegMemObject
from gemmforge.exceptions import InternalError


class StoreRegToGlb(AbstractInstruction):
  def __init__(self,
               vm: VM,
               dest: Symbol,
               src: Symbol,
               alpha: float,
               beta: float,
               num_threads: int):
    super(StoreRegToGlb, self).__init__(vm)

    if dest.stype != SymbolType.Global:
      raise InternalError('store: operand `dest` is not in glb mem.')

    if src.stype != SymbolType.Register:
      raise InternalError('store: operand `src` is not a register obj')

    self._dest = dest
    self._src = src
    self._alpha = alpha
    self._beta = beta
    self._num_threads = num_threads
    self._is_ready = True

  def gen_code(self, writer):
    dest_matrix = self._dest.obj
    dest_name = self._dest.name
    precision = self._vm.fp_as_str()

    with writer.If(self.gen_mask_threads(self._num_threads)):
      writer.Pragma("unroll")
      with writer.For(f'int n = 0; n < {dest_matrix.get_actual_num_cols()}; ++n'):
        rhs = "{}[{} + {} * n]".format(dest_name,
                                       self._vm.get_lexic().thread_idx_x,
                                       dest_matrix.leading_dimension)

        real_suffix = 'f' if precision == "float" else ''

        src_access = '' if self._src.obj.size == 1 else '[n]'
        if not isinstance(self._alpha, float):
          lhs = f'{self._alpha} * {self._src.name}{src_access}'
        else:
          if self._alpha == 1.0:
            lhs = f'{self._src.name}{src_access}'
          else:
            lhs = f'{self._alpha}{real_suffix} * {self._src.name}{src_access}'

        if not isinstance(self._beta, float):
          lhs += f' + {self._beta} * {rhs}'
        else:
          if self._beta != 0.0:
            if self._beta == 1.0:
              lhs += f' + {rhs}'
            else:
              const = f'{self._beta}{real_suffix}'
              lhs += f' + {const} * {rhs}'

        writer(f'{rhs} = {lhs};')

  def __str__(self) -> str:
    return 'not implemented'

class StoreRegToGlbTensor(AbstractInstruction):
  def __init__(self,
               vm: VM,
               dest: Symbol,
               src: Symbol,
               alpha: float,
               beta: float,
               num_threads: int):
    super(StoreRegToGlbTensor, self).__init__(vm)

    if dest.stype != SymbolType.Global:
      raise InternalError(f'store: operand `dest` is not in glb mem: {dest}')

    if src.stype != SymbolType.Register:
      raise InternalError(f'store: operand `src` is not a register obj: {src}')

    self._dest = dest
    self._src = src
    self._alpha = alpha
    self._beta = beta
    self._num_threads = num_threads
    self._is_ready = True

  def gen_code(self, writer):
    dest_matrix = self._dest.obj
    dest_name = self._dest.name
    precision = self._vm.fp_as_str()

    with writer.If(self.gen_mask_threads(self._num_threads)):
      #writer.Pragma("unroll")
      print("WARNING: TODO: StoreRegToGlobal for Tensor")
      writer.Comment("TODO: StoreRegToGlobal")
      """
      with writer.For(f'int n = 0; n < {dest_matrix.get_actual_num_cols()}; ++n'):
        rhs = "{}[{} + {} * n]".format(dest_name,
                                       self._vm.get_lexic().thread_idx_x,
                                       dest_matrix.leading_dimension)

        real_suffix = 'f' if precision == "float" else ''

        src_access = '' if self._src.obj.size == 1 else '[n]'
        if not isinstance(self._alpha, float):
          lhs = f'{self._alpha} * {self._src.name}{src_access}'
        else:
          if self._alpha == 1.0:
            lhs = f'{self._src.name}{src_access}'
          else:
            lhs = f'{self._alpha}{real_suffix} * {self._src.name}{src_access}'

        if not isinstance(self._beta, float):
          lhs += f' + {self._beta} * {rhs}'
        else:
          if self._beta != 0.0:
            if self._beta == 1.0:
              lhs += f' + {rhs}'
            else:
              const = f'{self._beta}{real_suffix}'
              lhs += f' + {const} * {rhs}'

        writer(f'{rhs} = {lhs};')
      """
      real_suffix = 'f' if precision == "float" else ''
      thread_idx_x = self._vm.get_lexic().thread_idx_x,
      dest = self._dest
      begin_char = "i"
      current_char = begin_char
      for dim in dest.data_view.dimensions[:]:
        writer.Pragma("unroll")
        writer.For(f'int {current_char} = 0; {current_char} < {dim}; ++{current_char}').__enter__()
        current_char =  chr(ord(current_char) + 1)

      offsetStr1 = ""
      current_char = begin_char
      for i in range(len(dest.data_view.dimensions)):
        offsetStr1 += f" + {dest.data_view.dimensions[i]} * {current_char}"
        current_char =  chr(ord(current_char) + 1)

      rhs = "{}[{} {}]".format(dest_name,
                                  self._vm.get_lexic().thread_idx_x,
                                  offsetStr1)

      real_suffix = 'f' if precision == "float" else ''

      src_access = '' if self._src.obj.size == 1 else f'[0{offsetStr1}]'
      if not isinstance(self._alpha, float):
        lhs = f'{self._alpha} * {self._src.name}{src_access}'
      else:
        if self._alpha == 1.0:
          lhs = f'{self._src.name}{src_access}'
        else:
          lhs = f'{self._alpha}{real_suffix} * {self._src.name}{src_access}'

      if not isinstance(self._beta, float):
        lhs += f' + {self._beta} * {rhs}'
      else:
        if self._beta != 0.0:
          if self._beta == 1.0:
            lhs += f' + {rhs}'
          else:
            const = f'{self._beta}{real_suffix}'
            lhs += f' + {const} * {rhs}'

        writer(f'{rhs} = {lhs};')

      for dim in reversed(dest.data_view.dimensions):
        writer.For(f'int i = 0; i < {dim}; ++i').__exit__(type=None, value=None, traceback=None)


  def __str__(self) -> str:
    return 'not implemented'