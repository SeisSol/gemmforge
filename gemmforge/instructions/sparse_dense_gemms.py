from .abstract_instruction import AbstractInstruction
from gemmforge.vm import VM
from gemmforge.symbol_table import SymbolType, Symbol, DataView, InverseSymbolTable
from gemmforge.basic_types import GeneralLexicon, DataFlowDirection, RegMemObject
from gemmforge.exceptions import InternalError
from abc import abstractmethod


class ShrMemBasedSparseDenseGemm(AbstractInstruction):
  def __init__(self, **kwargs):
    super(ShrMemBasedSparseDenseGemm, self).__init__(kwargs['vm'])
    self._trans_a = kwargs['trans_a']
    self._trans_b = kwargs['trans_b']
    self._op1 = kwargs['op1']
    self._op2 = kwargs['op2']
    self._intermediate_dest = kwargs['intermediate_dest']
    self._register_dest = kwargs['register_dest']
    self._num_threads = kwargs['num_threads']
    self._mat_a = kwargs['mat_a']

    if self._op1.obj.get_values() == None and self._op1.stype == SymbolType.Batch:
      raise InternalError('gemm: `op1` is a batch type, must be either glb. or shr.')

    if self._op2.stype == SymbolType.Batch:
      raise InternalError('gemm: `op2` is a batch type, must be either glb. or shr.')

    if self._register_dest.stype != SymbolType.Register:
      raise InternalError('gemm: `dest` must be a register obj.')

    if self._trans_a:
      raise Exception('Sparse Matrix is not supported to be transposed (provide the storage order with the list of coordinates).')

    self._is_ready = True

  def gen_code(self, writer):
    value_var = 'value'
    op1_data_view = self._op1.data_view
    op2_data_view = self._op2.data_view
    thread_idx_x = self._vm.get_lexic().thread_idx_x
    with writer.If(self.gen_mask_threads(self._op2.obj.get_actual_num_cols())):
      writer(f'{self._vm.fp_as_str()} {value_var};')

      writer.Emptyline()

      rows_b = op2_data_view.columns
      cols_b = op2_data_view.rows
  
      for k in range(0, cols_b):
        non_zeros = self._mat_a.get_coo_per_col()[k]

        if len(non_zeros) == 0:
          continue

        #op2_addr = f'{thread_idx_x} + {op2_data_view.lead_dim} * {k}'
        op2_addr = f'{thread_idx_x} * {op2_data_view.lead_dim} + {k}'
        writer(f'{value_var} = {self._op2.name}[{op2_addr}];')

        writer.Emptyline()
        self._get_inner_loop_sparse_with_a_col(writer, value_var, k, self._mat_a.get_values())

  def _get_inner_loop_sparse_with_a_col(self, writer, op2_value, b_col_id, val_a=None):
    # Iterate the first column first then the second etc. (coo_b[0] if col major, otherwise coo_b[1] if row major)
    # As we iterate we need to find the element in the real ordering (coordiantes)
    # This function iterates a column until the end
    # Multiply an element of ith column of B with the ith column of A
    a_col_id = b_col_id
    non_zeros = self._mat_a.get_coo_per_col()[a_col_id]
    if len(non_zeros) > 0:
      value_known = val_a != None
      writer.Comment(f"Mul begin col {a_col_id}")

      if self._trans_a:
        row_id = a_col_id
        for col_id in non_zeros:
          it = self._mat_a.find_1d_offset(row_id, col_id)
          res_access = f"[{col_id}]"

          if not value_known:
            writer(f'{self._register_dest.name}{res_access} += {self._op1.name}[{it}] * {op2_value};')
          else:
            writer(
              f'{self._register_dest.name}{res_access} += {val_a[it]}{self._vm.get_real_literal()} * {op2_value};')

      else:
        for row_id in non_zeros:
          it = self._mat_a.find_1d_offset(row_id, a_col_id)
          res_access = f"[{row_id}]"

          if not value_known:
            writer(f'{self._register_dest.name}{res_access} += {self._op1.name}[{it}] * {op2_value};')
          else:
            writer(
              f'{self._register_dest.name}{res_access} += {val_a[it]}{self._vm.get_real_literal()} * {op2_value};')

      writer.Comment(f"Mul end col {a_col_id}")
      writer.Emptyline()

  def __str__(self) -> str:
    return f'{self._register_dest.name} = gemm({self._op1.name}, {self._op2.name})'


class RegisterOnlySparseDenseGemm(AbstractInstruction):
  def __init__(self, **kwargs):
    super(RegisterOnlySparseDenseGemm, self).__init__(kwargs['vm'])
    self._trans_a = kwargs['trans_a']
    self._trans_b = kwargs['trans_b']
    self._op1 = kwargs['op1']
    self._op2 = kwargs['op2']
    self._dest = kwargs['dest']
    self._num_threads = kwargs['num_threads']
    self._mat_a = kwargs['mat_a']

    if self._op1.stype == SymbolType.Batch:
      raise InternalError('gemm: `op1` is a batch type, must be either glb. or shr.')

    if self._op2.stype == SymbolType.Batch:
      raise InternalError('gemm: `op2` is a batch type, must be either glb. or shr.')

    if self._dest.stype != SymbolType.Register:
      raise InternalError('gemm: `dest` must be a register obj.')

    if self._trans_a:
      raise Exception('Sparse Matrix is not supported to be transposed (provide the storage order with the list of coordinates).')

    self._is_ready = True

  def gen_code(self, writer):
    raise Exception("Register Only Sparse-by-Dense Matrix Implementation is not supported")

  def __str__(self) -> str:
    return f'{self._dest.name} = rb_gemm({self._op1.name}, {self._op2.name})'
