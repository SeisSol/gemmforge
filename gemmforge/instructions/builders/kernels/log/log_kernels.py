from gemmforge.basic_types import DataFlowDirection
from gemmforge.instructions.builders import GetElementPtrBuilder, RegistersAllocBuilder, ShrMemAllocBuilder
from gemmforge.instructions.builders.allocator_builder import ShrMemNewAllocBuilder
from gemmforge.instructions.builders.gemms.gemm_builder import ShrMemBasedDenseGemmBuilder
from gemmforge.instructions.builders.kernels.gemms.base_kernel import BaseGemmKernelBuilder
from gemmforge.instructions.store import StoreRegToGlb
from gemmforge.matrix.dense import DenseMatrix
from gemmforge.symbol_table import SymbolType


class ShrMemBasedLoopOverGemmKernelBuilder(BaseGemmKernelBuilder):
  """ This is the base class for building complete gemm kernels."""

  def __init__(self, **kwargs):
    super(ShrMemBasedLoopOverGemmKernelBuilder, self).__init__(**kwargs)
    self._apply_log_loop_heuristics = kwargs['apply_log_loop_heuristics']
    self._load_both_matrices = kwargs['load_both_matrices']

  def build_kernel(self):
    builder = ShrMemAllocBuilder(self._vm, self._symbol_table)
    builder.build(size=None)
    self._instructions.extend(builder.get_instructions())
    self._shr_mem_obj = builder.get_resultant_obj()

    # generate the rest instructions i.e., load to shr. mem, compute, store
    builder = ShrMemBasedDenseGemmBuilder(self._vm,
                                          self._symbol_table,
                                          self._reg_array_obj,
                                          self._shr_mem_obj,
                                          self._num_active_threads,
                                          self._apply_log_loop_heuristics,
                                          self._load_both_matrices)

    builder.build(trans_a=self._trans_a,
                  trans_b=self._trans_b,
                  op1=self._symbol_table[self._mat_a],
                  op2=self._symbol_table[self._mat_b],
                  dest=self._symbol_table[self._reg_array_obj])

    self._shr_mem_loads = builder.get_srh_mem_loads()
    self._instructions.extend(builder.get_instructions())

  def get_reg_array_obj(self):
    return self._reg_array_obj

  def get_shr_mem_obj(self):
    return self._shr_mem_obj

  def get_shr_mem_loads(self):
    return self._shr_mem_loads

  def build_prologue(self):
    builder = GetElementPtrBuilder(self._vm, self._symbol_table)
    for symbol in self._symbol_table.from_global.values():
      if symbol.obj in [self._mat_a, self._mat_b, self._mat_c]:
        builder.build(symbol)
        self._instructions.extend(builder.get_instructions())

    # create an array of registers
    builder = RegistersAllocBuilder(self._vm, self._symbol_table)
    builder.build(self._mat_c.get_actual_num_cols(), 0.0)
    self._instructions.extend(builder.get_instructions())
    self._reg_array_obj = builder.get_resultant_obj()

  def build_epilogue(self):
    store = StoreRegToGlb(self._vm,
                          self._symbol_table[self._mat_c],
                          self._symbol_table[self._reg_array_obj],
                          self._alpha,
                          self._beta,
                          self._num_compute_threads)
    self._instructions.append(store)

  def build(self):
    self.build_prologue()
    self.build_kernel()
    self.build_epilogue()
